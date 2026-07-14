"""
CineScene hybrid retrieval engine.

Combines semantic FAISS retrieval, lightweight lexical retrieval, reciprocal
rank fusion, optional cross-encoder reranking, and query-time filters.
"""

from __future__ import annotations

import os
import pickle
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
from query_processor import QueryProcessor
from sentence_transformers import SentenceTransformer

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover - dependency fallback
    BM25Okapi = None

try:
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover - dependency fallback
    CrossEncoder = None


LOCAL_BGE_PATH = Path("models/bge-large-en-v1.5")


def clean_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none"}:
        return ""
    return text


def resolve_existing(paths: List[str]) -> Optional[str]:
    for path in paths:
        if Path(path).exists():
            return path
    return None


def resolve_model_path(model_path: str, use_base_model: bool = False) -> str:
    if use_base_model:
        if (LOCAL_BGE_PATH / "config.json").exists():
            return str(LOCAL_BGE_PATH)
        return "BAAI/bge-large-en-v1.5"
    if Path(model_path).exists():
        return model_path
    if (LOCAL_BGE_PATH / "config.json").exists():
        return str(LOCAL_BGE_PATH)
    return model_path


def resolve_model_from_index_report(default_model_path: str, index_path: Optional[str]) -> str:
    if not index_path:
        return default_model_path
    report_path = Path(index_path).parent / "index_report.json"
    if not report_path.exists():
        return default_model_path
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        model = report.get("model")
        if model and Path(model).exists():
            return model
    except Exception:
        return default_model_path
    return default_model_path


class HybridSearchEngine:
    def __init__(
        self,
        model_path: str = "models/cinescene-v2/final",
        index_path: str = "data/index/faiss_index_v2.bin",
        metadata_path: str = "data/index/metadata_v2.pkl",
        use_base_model: bool = False,
        cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        enable_reranker: bool = True,
    ):
        self.index_path = resolve_existing([index_path, "data/index/movies.index"])
        self.metadata_path = resolve_existing([metadata_path, "data/index/metadata.pkl"])
        reported_model = resolve_model_from_index_report(model_path, self.index_path)
        self.model_path = resolve_model_path(reported_model, use_base_model=use_base_model)

        if not self.index_path or not self.metadata_path:
            raise FileNotFoundError(
                "FAISS index is missing. Run build_index_v2.py after generating or enriching movies."
            )

        print(f"Loading embedding model: {self.model_path}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        use_fp16 = self.device == "cuda" and os.getenv("CINESCENE_EMBEDDING_FP16", "0") == "1"
        self.precision = "float16" if use_fp16 else "float32"
        self.model = SentenceTransformer(self.model_path, device=self.device)
        self.model.max_seq_length = min(
            int(self.model.max_seq_length),
            max(64, int(os.getenv("CINESCENE_EMBEDDING_MAX_SEQ", "256"))),
        )
        if use_fp16:
            self.model.half()

        print(f"Loading FAISS index: {self.index_path}")
        self.index = faiss.read_index(self.index_path)

        print(f"Loading metadata: {self.metadata_path}")
        with open(self.metadata_path, "rb") as f:
            self.movie_metadata = pickle.load(f)

        from scene_index import load_scene_index

        self.scene_index, self.scene_metadata = load_scene_index(expected_dimension=int(self.index.d))
        self.scene_offset = len(self.movie_metadata)
        self.metadata = self.movie_metadata + self.scene_metadata
        self.parent_metadata = {
            str(item.get("id")): item
            for item in self.movie_metadata
            if item.get("id") is not None
        }

        self.query_processor = QueryProcessor()
        self._build_lexical_index()

        self.cross_encoder = None
        if enable_reranker and CrossEncoder is not None:
            try:
                print(f"Loading cross-encoder: {cross_encoder_name}")
                self.cross_encoder = CrossEncoder(cross_encoder_name)
            except Exception as exc:
                print(f"Cross-encoder unavailable, continuing without reranking: {exc}")

        print(
            f"Hybrid search ready with {len(self.movie_metadata)} titles and "
            f"{len(self.scene_metadata)} scene vectors"
        )

    def _doc_for_meta(self, meta: Dict) -> str:
        timeline_text = []
        for scene in meta.get("scene_timeline", []) or []:
            if isinstance(scene, dict):
                timeline_text.extend(
                    [
                        clean_text(scene.get("visual_caption")),
                        clean_text(scene.get("transcript")),
                        " ".join(scene.get("visual_tags", []) or []),
                        " ".join(scene.get("mood_tags", []) or []),
                        " ".join(scene.get("keywords", []) or []),
                    ]
                )
        return " ".join(
            [
                clean_text(meta.get("title")),
                clean_text(meta.get("overview")),
                " ".join(meta.get("genres", []) or []),
                " ".join(meta.get("mood_tags", []) or []),
                " ".join(meta.get("keywords", []) or []),
                " ".join(meta.get("visual_tags", []) or []),
                " ".join(meta.get("scene_descriptions", []) or []),
                " ".join(timeline_text),
                clean_text(meta.get("director")),
                clean_text(meta.get("rich_text")),
                clean_text(meta.get("source_video")),
            ]
        )

    def _build_lexical_index(self):
        self.lexical_docs = [self._lexical_tokens(self._doc_for_meta(meta)) for meta in self.metadata]
        self.lexical_sets = [set(tokens) for tokens in self.lexical_docs]
        self.title_token_sets = [set(self._lexical_tokens(meta.get("title", ""))) for meta in self.metadata]
        self.transcript_token_sets = []
        for meta in self.metadata:
            scene = meta.get("matched_scene") or {}
            transcript = scene.get("transcript", "") if isinstance(scene, dict) else ""
            self.transcript_token_sets.append(set(self._lexical_tokens(transcript)))
        if BM25Okapi is not None:
            self.bm25 = BM25Okapi(self.lexical_docs)
        else:
            self.bm25 = None

    @staticmethod
    def _lexical_tokens(text: str) -> List[str]:
        tokens = re.findall(r"[^\W_]+", clean_text(text).lower(), flags=re.UNICODE)
        normalized = []
        for token in tokens:
            if token.isascii():
                if len(token) > 4 and token.endswith("ies"):
                    token = token[:-3] + "y"
                elif len(token) > 4 and token.endswith("es") and token[-3] in {"s", "x", "z"}:
                    token = token[:-2]
                elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
                    token = token[:-1]
            if len(token) > 2:
                normalized.append(token)
        return normalized

    def vector_search(self, query: str, k: int = 50) -> List[Tuple[int, float]]:
        query_emb = self.model.encode(query, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        query_emb = query_emb.reshape(1, -1)
        movie_k = min(k, len(self.movie_metadata))
        distances, indices = self.index.search(query_emb, movie_k)
        results = [(int(idx), float(score)) for idx, score in zip(indices[0], distances[0]) if int(idx) >= 0]

        if self.scene_index is not None and self.scene_metadata:
            scene_k = min(max(k, 80), len(self.scene_metadata))
            scene_distances, scene_indices = self.scene_index.search(query_emb, scene_k)
            results.extend(
                (self.scene_offset + int(idx), float(score))
                for idx, score in zip(scene_indices[0], scene_distances[0])
                if int(idx) >= 0
            )
        return sorted(results, key=lambda item: item[1], reverse=True)

    def lexical_search(self, query: str, k: int = 50) -> List[Tuple[int, float]]:
        k = min(k, len(self.metadata))
        query_tokens = self._lexical_tokens(query)
        if self.bm25 is not None:
            scores = self.bm25.get_scores(query_tokens)
        else:
            token_set = set(query_tokens)
            scores = np.array([len(token_set & set(doc)) / max(1, len(token_set)) for doc in self.lexical_docs])
        top_indices = np.argsort(scores)[::-1][:k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def direct_overlap_search(self, query: str, k: int = 50) -> List[Tuple[int, float]]:
        terms = set(self._lexical_tokens(query))
        if not terms:
            return []
        scored = []
        for idx, meta in enumerate(self.metadata):
            doc_terms = self.lexical_sets[idx]
            overlap = len(terms & doc_terms)
            if overlap:
                transcript_terms = self.transcript_token_sets[idx]
                transcript_overlap = len(terms & transcript_terms)
                source_bonus = min(2, transcript_overlap) if meta.get("record_type") == "scene" else 0
                scored.append((idx, float(overlap + source_bonus)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]

    def reciprocal_rank_fusion(
        self,
        vector_results: List[Tuple[int, float]],
        lexical_results: List[Tuple[int, float]],
        k: int = 60,
    ) -> List[Tuple[int, float]]:
        scores: Dict[int, float] = {}
        for rank, (idx, _) in enumerate(vector_results):
            scores[idx] = scores.get(idx, 0.0) + 1 / (k + rank + 1)
        for rank, (idx, _) in enumerate(lexical_results):
            scores[idx] = scores.get(idx, 0.0) + 1 / (k + rank + 1)
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)

    def lexical_boost(self, query: str, results: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        query_terms = set(self._lexical_tokens(query))
        if not query_terms:
            return results

        boosted = []
        for idx, score in results:
            meta = self.metadata[idx]
            doc_terms = self.lexical_sets[idx]
            title_terms = self.title_token_sets[idx]
            overlap = len(query_terms & doc_terms)
            title_overlap = len(query_terms & title_terms)
            source_boost = 0.0
            if meta.get("record_type") == "scene" and overlap >= 2:
                transcript_terms = self.transcript_token_sets[idx]
                transcript_overlap = len(query_terms & transcript_terms)
                source_boost = min(0.12, 0.022 * transcript_overlap) if transcript_overlap >= 2 else min(0.012, 0.002 * overlap)
            boosted.append((idx, score + overlap * 0.003 + title_overlap * 0.008 + source_boost))
        boosted.sort(key=lambda item: item[1], reverse=True)
        return boosted

    def apply_filters(self, results: List[Tuple[int, float]], filters: Dict) -> List[Tuple[int, float]]:
        filtered = []
        for idx, score in results:
            meta = self.metadata[idx]

            if filters.get("genres"):
                meta_genres = set(meta.get("genres", []) or [])
                if not any(genre in meta_genres for genre in filters["genres"]):
                    continue

            if filters.get("year_range"):
                try:
                    year = int(meta.get("year"))
                    start, end = filters["year_range"]
                    if not (start <= year <= end):
                        continue
                except Exception:
                    continue

            if filters.get("director"):
                if filters["director"].lower() not in clean_text(meta.get("director")).lower():
                    continue

            filtered.append((idx, score))
        return filtered

    def cross_encoder_rerank(
        self,
        query: str,
        candidates: List[Tuple[int, float]],
        top_k: int = 20,
    ) -> List[Tuple[int, float]]:
        if not self.cross_encoder or not candidates:
            return candidates

        pairs = []
        selected = candidates[:top_k]
        for idx, _ in selected:
            meta = self.metadata[idx]
            doc_text = self._doc_for_meta(meta)[:1800]
            pairs.append([query, doc_text])

        ce_scores = self.cross_encoder.predict(pairs)
        reranked = []
        for (idx, fusion_score), ce_score in zip(selected, ce_scores):
            reranked.append((idx, float(0.7 * ce_score + 0.3 * fusion_score)))

        reranked.sort(key=lambda item: item[1], reverse=True)
        return reranked + candidates[top_k:]

    def _format_result(self, idx: int, score: float, rank: int, matched_scenes: Optional[List[Dict]] = None) -> Dict:
        candidate = self.metadata[idx]
        parent_id = candidate.get("parent_id") or candidate.get("id")
        meta = candidate if candidate.get("record_type") == "scene" else self.parent_metadata.get(str(parent_id), candidate)
        scene_matches = matched_scenes or []
        timeline = meta.get("scene_timeline", []) or []
        if not timeline and scene_matches:
            timeline = scene_matches
        return {
            "rank": rank,
            "id": parent_id,
            "title": meta.get("title", "Unknown"),
            "media_type": meta.get("media_type", "movie"),
            "season": meta.get("season"),
            "episode": meta.get("episode"),
            "year": meta.get("year", "N/A"),
            "genres": meta.get("genres", []) or [],
            "director": meta.get("director", "Unknown"),
            "cast": meta.get("cast", []) or [],
            "overview": meta.get("overview", ""),
            "rating": meta.get("rating", meta.get("vote_average", 0.0)),
            "popularity": meta.get("popularity", 0.0),
            "scenes": meta.get("scene_descriptions", []) or [],
            "scene_timeline": timeline,
            "matched_scene": scene_matches[0] if scene_matches else None,
            "matched_scenes": scene_matches,
            "scene_count": meta.get("scene_count"),
            "first_keyframe": meta.get("first_keyframe"),
            "visual_tags": meta.get("visual_tags", []) or [],
            "mood": meta.get("mood_tags", []) or [],
            "keywords": meta.get("keywords", []) or [],
            "source": meta.get("source", ""),
            "source_video": meta.get("source_video", ""),
            "score": round(float(score), 4),
        }

    def _collapse_titles(self, results: List[Tuple[int, float]]) -> List[Tuple[int, float, List[Dict]]]:
        grouped: Dict[str, Dict] = {}
        for idx, score in results:
            meta = self.metadata[idx]
            parent_id = str(meta.get("parent_id") or meta.get("id") or f"row:{idx}")
            group = grouped.setdefault(parent_id, {"idx": idx, "score": score, "scenes": []})
            if score > group["score"]:
                group["idx"] = idx
                group["score"] = score
            scene = meta.get("matched_scene")
            if meta.get("record_type") == "scene" and isinstance(scene, dict):
                scene_with_score = dict(scene)
                scene_with_score["match_score"] = round(float(score), 4)
                group["scenes"].append(scene_with_score)

        collapsed = []
        for group in grouped.values():
            scenes = sorted(group["scenes"], key=lambda item: item.get("match_score", 0), reverse=True)
            coverage_bonus = min(0.012, max(0, len(scenes) - 1) * 0.002)
            collapsed.append((group["idx"], float(group["score"]) + coverage_bonus, scenes[:5]))
        return sorted(collapsed, key=lambda item: item[1], reverse=True)

    def search(self, query: str, top_k: int = 10, use_reranking: bool = True) -> List[Dict]:
        processed = self.query_processor.process(query)
        search_query = processed["expanded"]
        filters = processed["filters"]

        vector_results = self.vector_search(search_query, k=60)
        lexical_results = self.lexical_search(search_query, k=60)
        overlap_results = self.direct_overlap_search(search_query, k=80)
        fused_results = self.reciprocal_rank_fusion(vector_results, lexical_results + overlap_results)
        fused_results = self.lexical_boost(search_query, fused_results)

        if filters.get("genres") or filters.get("year_range") or filters.get("director"):
            fused_results = self.apply_filters(fused_results, filters)

        if use_reranking:
            fused_results = self.cross_encoder_rerank(search_query, fused_results, top_k=min(20, len(fused_results)))

        collapsed = self._collapse_titles(fused_results)
        return [
            self._format_result(idx, score, rank + 1, matched_scenes=scenes)
            for rank, (idx, score, scenes) in enumerate(collapsed[:top_k])
        ]

    def status(self) -> Dict:
        return {
            "movies": len(self.movie_metadata),
            "scene_vectors": len(self.scene_metadata),
            "index_vectors": int(self.index.ntotal) + (int(self.scene_index.ntotal) if self.scene_index is not None else 0),
            "movie_vectors": int(self.index.ntotal),
            "model": self.model_path,
            "device": self.device,
            "precision": self.precision,
            "index_path": self.index_path,
            "metadata_path": self.metadata_path,
            "reranker": bool(self.cross_encoder),
            "lexical": "bm25" if self.bm25 is not None else "token_overlap",
        }


if __name__ == "__main__":
    engine = HybridSearchEngine(enable_reranker=False)
    for item in engine.search("dark lonely science fiction movie", top_k=5, use_reranking=False):
        print(f"{item['rank']}. {item['title']} ({item['year']}) - {item['score']}")
