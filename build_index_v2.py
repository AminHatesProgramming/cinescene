"""
CineScene v2 FAISS index builder.

Creates the semantic movie index used by the backend. It prefers the fine-tuned
model when available, falls back to the local BGE base model when requested, and
saves both v2 and legacy index filenames for compatibility.
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, List

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
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


def default_catalog_path() -> str:
    combined = Path("data/processed/cinescene_catalog.json")
    if combined.exists():
        return str(combined)
    return "data/processed/movies_enriched.json"


def resolve_model_path(model_path: str, use_base_model: bool = False) -> str:
    if use_base_model:
        if (LOCAL_BGE_PATH / "config.json").exists():
            return str(LOCAL_BGE_PATH)
        return "BAAI/bge-large-en-v1.5"
    if Path(model_path).exists():
        return model_path
    if (LOCAL_BGE_PATH / "config.json").exists():
        print(f"Fine-tuned model not found at {model_path}; using local base model")
        return str(LOCAL_BGE_PATH)
    return model_path


class IndexBuilderV2:
    def __init__(
        self,
        model_path: str = "models/cinescene-v2/final",
        use_base_model: bool = False,
        batch_size: int = 16,
        max_seq_length: int = 256,
        fp16: bool = False,
    ):
        self.model_path = resolve_model_path(model_path, use_base_model)
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.precision = "float16" if self.device == "cuda" and fp16 else "float32"
        self.max_seq_length = max(64, int(max_seq_length))
        print(f"Loading embedding model: {self.model_path}")
        self.model = SentenceTransformer(self.model_path, device=self.device)
        self.model.max_seq_length = min(int(self.model.max_seq_length), self.max_seq_length)
        if self.precision == "float16":
            self.model.half()
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(
            f"Embedding dimension: {self.dimension} "
            f"({self.device}, {self.precision}, max_seq={self.model.max_seq_length})"
        )

    def load_enriched_data(self, path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            movies = json.load(f)
        usable = [movie for movie in movies if movie_title(movie) != "Unknown" and self._movie_document(movie)]
        print(f"Loaded {len(movies)} movies; indexing {len(usable)} usable movies")
        return usable

    def _movie_document(self, movie: Dict) -> str:
        rich_text = clean_text(movie.get("rich_text"))
        if rich_text and "Title:" in rich_text:
            return rich_text
        timeline_text = []
        for scene in movie.get("scene_timeline", []) or []:
            if isinstance(scene, dict):
                timeline_text.extend(
                    [
                        clean_text(scene.get("visual_caption")),
                        clean_text(scene.get("transcript")),
                        ", ".join(as_text_list(scene.get("visual_tags", []))),
                        ", ".join(as_text_list(scene.get("mood_tags", []))),
                        ", ".join(as_text_list(scene.get("keywords", []))),
                    ]
                )

        parts = [
            f"Title: {movie_title(movie)}",
            f"Type: {clean_text(movie.get('media_type')) or 'movie'}",
            f"Year: {clean_text(movie.get('release_year'))}",
            f"Genres: {', '.join(as_text_list(movie.get('genres', [])))}",
            f"Director: {clean_text(movie.get('director'))}",
            f"Cast: {', '.join(as_text_list(movie.get('cast', []))[:8])}",
            f"Plot: {clean_text(movie.get('overview')) or clean_text(movie.get('wiki_plot')) or clean_text(movie.get('cmu_plot'))}",
            f"Scenes: {' '.join(as_text_list(movie.get('scene_descriptions', []))[:30])}",
            f"Timeline: {' '.join(timeline_text[:80])}",
            f"Mood: {', '.join(as_text_list(movie.get('mood_tags', [])))}",
            f"Keywords: {', '.join(as_text_list(movie.get('keywords', []))[:15])}",
            f"Visual tags: {', '.join(as_text_list(movie.get('visual_tags', []))[:15])}",
        ]
        return " | ".join(part for part in parts if not part.endswith(": "))

    def build_embeddings(self, movies: List[Dict]) -> np.ndarray:
        docs = [self._movie_document(movie) for movie in movies]
        print("Encoding movies for FAISS...")
        embeddings = self.model.encode(
            docs,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype("float32")

    def build_faiss_index(self, embeddings: np.ndarray, use_hnsw: bool = True) -> faiss.Index:
        if use_hnsw:
            print("Building FAISS HNSW index...")
            index = faiss.IndexHNSWFlat(self.dimension, 32)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 64
        else:
            print("Building FAISS FlatIP index...")
            index = faiss.IndexFlatIP(self.dimension)

        index.add(embeddings)
        print(f"Index contains {index.ntotal} vectors")
        return index

    def prepare_metadata(self, movies: List[Dict]) -> List[Dict]:
        metadata = []
        for movie in movies:
            metadata.append(
                {
                    "id": movie.get("id"),
                    "title": movie_title(movie),
                    "media_type": clean_text(movie.get("media_type")) or "movie",
                    "season": movie.get("season"),
                    "episode": movie.get("episode"),
                    "year": clean_text(movie.get("release_year")) or "N/A",
                    "genres": as_text_list(movie.get("genres", [])),
                    "director": clean_text(movie.get("director")) or "Unknown",
                    "cast": as_text_list(movie.get("cast", []))[:8],
                    "overview": clean_text(movie.get("overview"))[:600],
                    "rating": float(movie.get("vote_average") or 0.0),
                    "popularity": float(movie.get("popularity") or 0.0),
                    "scene_descriptions": as_text_list(movie.get("scene_descriptions", []))[:30],
                    "scene_timeline": movie.get("scene_timeline", [])[:30] if isinstance(movie.get("scene_timeline"), list) else [],
                    "scene_count": movie.get("scene_count"),
                    "duration_sec": movie.get("duration_sec"),
                    "first_keyframe": movie.get("first_keyframe"),
                    "mood_tags": as_text_list(movie.get("mood_tags", [])),
                    "keywords": as_text_list(movie.get("keywords", []))[:15],
                    "visual_tags": as_text_list(movie.get("visual_tags", []))[:15],
                    "rich_text": self._movie_document(movie),
                    "source": movie.get("source", "tmdb_enriched"),
                    "source_video": movie.get("source_video", ""),
                }
            )
        return metadata

    def save_index(
        self,
        index: faiss.Index,
        metadata: List[Dict],
        index_path: str = "data/index/faiss_index_v2.bin",
        metadata_path: str = "data/index/metadata_v2.pkl",
        save_legacy: bool = True,
    ):
        index_path_obj = Path(index_path)
        metadata_path_obj = Path(metadata_path)
        index_path_obj.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, str(index_path_obj))
        with open(metadata_path_obj, "wb") as f:
            pickle.dump(metadata, f)

        if save_legacy:
            faiss.write_index(index, str(index_path_obj.parent / "movies.index"))
            with open(metadata_path_obj.parent / "metadata.pkl", "wb") as f:
                pickle.dump(metadata, f)

        report = {
            "movies_indexed": len(metadata),
            "embedding_dimension": self.dimension,
            "model": self.model_path,
            "device": self.device,
            "precision": self.precision,
            "batch_size": self.batch_size,
            "max_seq_length": self.model.max_seq_length,
            "index_path": str(index_path_obj),
            "metadata_path": str(metadata_path_obj),
            "legacy_files_written": save_legacy,
        }
        with open(index_path_obj.parent / "index_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Saved index to {index_path_obj}")
        print(f"Saved metadata to {metadata_path_obj}")

    def build_and_save(
        self,
        enriched_path: str = "data/processed/movies_enriched.json",
        index_path: str = "data/index/faiss_index_v2.bin",
        metadata_path: str = "data/index/metadata_v2.pkl",
        use_hnsw: bool = True,
    ):
        movies = self.load_enriched_data(enriched_path)
        embeddings = self.build_embeddings(movies)
        index = self.build_faiss_index(embeddings, use_hnsw=use_hnsw)
        metadata = self.prepare_metadata(movies)
        self.save_index(index, metadata, index_path=index_path, metadata_path=metadata_path)
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Index building complete")


def parse_args():
    parser = argparse.ArgumentParser(description="Build CineScene v2 FAISS index")
    parser.add_argument("--input", default=default_catalog_path())
    parser.add_argument("--model", default="models/cinescene-v2/final")
    parser.add_argument("--index", default="data/index/faiss_index_v2.bin")
    parser.add_argument("--metadata", default="data/index/metadata_v2.pkl")
    parser.add_argument("--base-model", action="store_true")
    parser.add_argument("--flat", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--fp16", action="store_true", help="Use FP16 on CUDA (usually faster on RTX GPUs)")
    return parser.parse_args()


def main():
    args = parse_args()
    builder = IndexBuilderV2(
        model_path=args.model,
        use_base_model=args.base_model,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        fp16=args.fp16,
    )
    builder.build_and_save(
        enriched_path=args.input,
        index_path=args.index,
        metadata_path=args.metadata,
        use_hnsw=not args.flat,
    )


if __name__ == "__main__":
    main()
