"""
CineScene v2 - scene-aware triplet generation.

Builds anchor/positive/negative samples from enriched movie data. Positives are
selected from shared genre, director, cast, mood, and keyword signals. Negatives
prefer hard semantic neighbours that do not share the main relevance signals.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


LOCAL_BGE_PATH = Path("models/bge-large-en-v1.5")


def clean_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none"}:
        return ""
    return text


def as_text_list(values: Iterable) -> List[str]:
    if not isinstance(values, list):
        return []
    return [clean_text(value) for value in values if clean_text(value)]


def movie_title(movie: Dict) -> str:
    return clean_text(movie.get("title")) or clean_text(movie.get("original_title")) or "Unknown"


def resolve_model_path(model_name: str) -> str:
    if model_name == "BAAI/bge-large-en-v1.5" and (LOCAL_BGE_PATH / "config.json").exists():
        return str(LOCAL_BGE_PATH)
    return model_name


class TripletGeneratorV2:
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        hard_negative_ratio: float = 0.7,
        semantic_hard_negatives: str = "auto",
        seed: int = 42,
    ):
        self.model_name = resolve_model_path(model_name)
        self.hard_negative_ratio = hard_negative_ratio
        self.semantic_hard_negatives = self._resolve_semantic_mode(semantic_hard_negatives)
        self.random = random.Random(seed)
        np.random.seed(seed)

        self.model = None
        if self.semantic_hard_negatives:
            print(f"Loading base model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        else:
            print("Semantic hard-negative mining disabled; using fast structural negatives")

    def _resolve_semantic_mode(self, mode: str) -> bool:
        if mode == "on":
            return True
        if mode == "off":
            return False
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def load_enriched_data(self, path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        usable = [movie for movie in data if self._has_training_signal(movie)]
        print(f"Loaded {len(data)} enriched movies; {len(usable)} usable movies")
        return usable

    def _has_training_signal(self, movie: Dict) -> bool:
        return movie_title(movie) != "Unknown" and bool(self._main_text(movie))

    def _main_text(self, movie: Dict) -> str:
        return (
            clean_text(movie.get("rich_text"))
            or clean_text(movie.get("overview"))
            or clean_text(movie.get("wiki_plot"))
            or clean_text(movie.get("cmu_plot"))
        )

    def _field_set(self, movie: Dict, key: str) -> set:
        return {item.lower() for item in as_text_list(movie.get(key, []))}

    def _precompute_features(self, movies: List[Dict]) -> List[Dict]:
        features = []
        for movie in movies:
            features.append(
                {
                    "genres": self._field_set(movie, "genres"),
                    "mood_tags": self._field_set(movie, "mood_tags"),
                    "keywords": self._field_set(movie, "keywords"),
                    "cast": self._field_set(movie, "cast"),
                    "director": clean_text(movie.get("director")).lower(),
                }
            )
        return features

    def _build_inverted_index(self, features: List[Dict]) -> Dict[str, Dict[str, set]]:
        inverted = {
            "genres": {},
            "mood_tags": {},
            "keywords": {},
            "cast": {},
            "director": {},
        }
        for idx, item in enumerate(features):
            for field in ["genres", "mood_tags", "keywords", "cast"]:
                for value in item[field]:
                    inverted[field].setdefault(value, set()).add(idx)
            if item["director"]:
                inverted["director"].setdefault(item["director"], set()).add(idx)
        return inverted

    def pre_embed_corpus(self, movies: List[Dict]) -> np.ndarray:
        if not self.model:
            raise RuntimeError("Semantic hard-negative mining needs an embedding model")
        print("Pre-computing movie embeddings for hard negative mining...")
        texts = []
        for movie in movies:
            scenes = " ".join(as_text_list(movie.get("scene_descriptions", []))[:3])
            mood = " ".join(as_text_list(movie.get("mood_tags", [])))
            keywords = " ".join(as_text_list(movie.get("keywords", []))[:12])
            combined = f"{self._main_text(movie)} {scenes} {mood} {keywords}".strip()
            texts.append(combined or movie_title(movie))

        return self.model.encode(
            texts,
            batch_size=16,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def _positive_score(self, anchor: Dict, candidate: Dict) -> int:
        genres = len(self._field_set(anchor, "genres") & self._field_set(candidate, "genres"))
        moods = len(self._field_set(anchor, "mood_tags") & self._field_set(candidate, "mood_tags"))
        keywords = len(self._field_set(anchor, "keywords") & self._field_set(candidate, "keywords"))
        cast = len(self._field_set(anchor, "cast") & self._field_set(candidate, "cast"))
        same_director = int(clean_text(anchor.get("director")) == clean_text(candidate.get("director")) and clean_text(anchor.get("director")) != "")

        return genres * 3 + moods * 2 + keywords + cast + same_director * 4

    def _positive_score_from_features(self, anchor: Dict, candidate: Dict) -> int:
        genres = len(anchor["genres"] & candidate["genres"])
        moods = len(anchor["mood_tags"] & candidate["mood_tags"])
        keywords = len(anchor["keywords"] & candidate["keywords"])
        cast = len(anchor["cast"] & candidate["cast"])
        same_director = int(anchor["director"] == candidate["director"] and anchor["director"] != "")
        return genres * 3 + moods * 2 + keywords + cast + same_director * 4

    def _is_positive_candidate(self, anchor: Dict, candidate: Dict) -> bool:
        return self._positive_score(anchor, candidate) >= 3

    def _is_too_similar_for_negative(self, anchor: Dict, candidate: Dict) -> bool:
        if clean_text(anchor.get("director")) == clean_text(candidate.get("director")) and clean_text(anchor.get("director")):
            return True
        if len(self._field_set(anchor, "genres") & self._field_set(candidate, "genres")) >= 2:
            return True
        if len(self._field_set(anchor, "mood_tags") & self._field_set(candidate, "mood_tags")) >= 3:
            return True
        return False

    def _find_positives(self, anchor_idx: int, movies: List[Dict]) -> List[int]:
        anchor = movies[anchor_idx]
        scored = []
        for idx, movie in enumerate(movies):
            if idx == anchor_idx:
                continue
            score = self._positive_score(anchor, movie)
            if score >= 3:
                scored.append((idx, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return [idx for idx, _ in scored[:75]]

    def _find_positives_fast(self, anchor_idx: int, features: List[Dict], inverted: Dict[str, Dict[str, set]]) -> List[int]:
        anchor = features[anchor_idx]
        candidate_ids = set()
        for field in ["genres", "mood_tags", "keywords", "cast"]:
            for value in anchor[field]:
                candidate_ids.update(inverted[field].get(value, set()))
        if anchor["director"]:
            candidate_ids.update(inverted["director"].get(anchor["director"], set()))
        candidate_ids.discard(anchor_idx)

        scored = []
        for idx in candidate_ids:
            score = self._positive_score_from_features(anchor, features[idx])
            if score >= 3:
                scored.append((idx, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [idx for idx, _ in scored[:75]]

    def find_hard_negatives(
        self,
        anchor_idx: int,
        positive_idx: int,
        all_embeddings: np.ndarray,
        movies: List[Dict],
        k: int = 5,
    ) -> List[int]:
        similarities = cosine_similarity(all_embeddings[anchor_idx].reshape(1, -1), all_embeddings)[0]
        sorted_indices = np.argsort(similarities)[::-1]
        hard_negatives = []

        for idx in sorted_indices:
            idx = int(idx)
            if idx in {anchor_idx, positive_idx}:
                continue
            if self._is_positive_candidate(movies[anchor_idx], movies[idx]):
                continue
            if self._is_too_similar_for_negative(movies[anchor_idx], movies[idx]):
                continue
            hard_negatives.append(idx)
            if len(hard_negatives) >= k:
                break

        return hard_negatives

    def _random_negative(self, anchor_idx: int, positive_idx: int, movies: List[Dict]) -> int:
        candidates = [
            idx
            for idx, movie in enumerate(movies)
            if idx not in {anchor_idx, positive_idx}
            and not self._is_positive_candidate(movies[anchor_idx], movie)
        ]
        if not candidates:
            candidates = [idx for idx in range(len(movies)) if idx not in {anchor_idx, positive_idx}]
        return self.random.choice(candidates)

    def _random_negative_fast(self, anchor_idx: int, positive_idx: int, features: List[Dict]) -> int:
        anchor = features[anchor_idx]
        total = len(features)
        for _ in range(80):
            idx = self.random.randrange(total)
            if idx in {anchor_idx, positive_idx}:
                continue
            if self._positive_score_from_features(anchor, features[idx]) < 3:
                return idx
        return self.random.choice([idx for idx in range(total) if idx not in {anchor_idx, positive_idx}])

    def _build_movie_text(self, movie: Dict) -> str:
        plot = (
            clean_text(movie.get("overview"))
            or clean_text(movie.get("wiki_plot"))
            or clean_text(movie.get("cmu_plot"))
            or self._main_text(movie)
        )
        scenes = " ".join(as_text_list(movie.get("scene_descriptions", []))[:3])
        mood = ", ".join(as_text_list(movie.get("mood_tags", []))[:8])
        genres = ", ".join(as_text_list(movie.get("genres", [])))
        keywords = ", ".join(as_text_list(movie.get("keywords", []))[:10])
        director = clean_text(movie.get("director"))

        parts = [f"Title: {movie_title(movie)}"]
        if genres:
            parts.append(f"Genres: {genres}")
        if director:
            parts.append(f"Director: {director}")
        if plot:
            parts.append(f"Plot: {plot[:900]}")
        if scenes:
            parts.append(f"Scenes: {scenes[:700]}")
        if mood:
            parts.append(f"Mood: {mood}")
        if keywords:
            parts.append(f"Keywords: {keywords}")

        return " | ".join(parts)

    def generate_triplets(self, movies: List[Dict], num_triplets_per_movie: int = 3) -> List[Tuple[str, str, str]]:
        all_embeddings = self.pre_embed_corpus(movies) if self.semantic_hard_negatives else None
        features = self._precompute_features(movies)
        inverted = self._build_inverted_index(features)
        triplets: List[Tuple[str, str, str]] = []

        print(f"Generating triplets ({num_triplets_per_movie} per movie)...")
        for anchor_idx, anchor_movie in enumerate(tqdm(movies, desc="Building triplets")):
            positives = self._find_positives_fast(anchor_idx, features, inverted)
            if not positives:
                continue

            sample_count = min(num_triplets_per_movie, len(positives))
            sampled_positives = self.random.sample(positives, sample_count)

            for positive_idx in sampled_positives:
                if all_embeddings is not None and self.random.random() < self.hard_negative_ratio:
                    hard_negatives = self.find_hard_negatives(anchor_idx, positive_idx, all_embeddings, movies, k=1)
                    negative_idx = hard_negatives[0] if hard_negatives else self._random_negative_fast(anchor_idx, positive_idx, features)
                else:
                    negative_idx = self._random_negative_fast(anchor_idx, positive_idx, features)

                triplets.append(
                    (
                        self._build_movie_text(anchor_movie),
                        self._build_movie_text(movies[positive_idx]),
                        self._build_movie_text(movies[negative_idx]),
                    )
                )

        print(f"Generated {len(triplets)} triplets")
        return triplets

    def save_triplets(self, triplets: List[Tuple[str, str, str]], output_path: str):
        data = [{"anchor": a, "positive": p, "negative": n} for a, p, n in triplets]
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved triplets to {output_path}")

    def save_quality_report(self, triplets: List[Tuple[str, str, str]], output_path: str):
        texts = [text for triplet in triplets for text in triplet]
        report = {
            "total_triplets": len(triplets),
            "total_texts": len(texts),
            "empty_title_texts": sum(1 for text in texts if text.startswith("Title:  ") or "Title: Unknown" in text),
            "average_text_length": round(float(np.mean([len(text) for text in texts])), 2) if texts else 0,
            "model": self.model_name,
            "hard_negative_ratio": self.hard_negative_ratio,
            "semantic_hard_negatives": self.semantic_hard_negatives,
        }
        report_path = Path(output_path).with_suffix(".quality.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Saved quality report to {report_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build CineScene v2 triplets")
    parser.add_argument("--input", default="data/processed/movies_enriched.json")
    parser.add_argument("--output", default="data/processed/triplets_v2.json")
    parser.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--triplets-per-movie", type=int, default=3)
    parser.add_argument("--hard-negative-ratio", type=float, default=0.7)
    parser.add_argument("--semantic-hard-negatives", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    generator = TripletGeneratorV2(
        model_name=args.model,
        hard_negative_ratio=args.hard_negative_ratio,
        semantic_hard_negatives=args.semantic_hard_negatives,
        seed=args.seed,
    )
    movies = generator.load_enriched_data(args.input)
    triplets = generator.generate_triplets(movies, num_triplets_per_movie=args.triplets_per_movie)
    generator.save_triplets(triplets, args.output)
    generator.save_quality_report(triplets, args.output)
    print("Triplet generation complete")


if __name__ == "__main__":
    main()
