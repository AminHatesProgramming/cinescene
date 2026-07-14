"""Scene-level FAISS index used alongside the movie catalog index.

The movie index answers broad plot, genre, and mood requests. This index stores
one vector per detected scene so dialogue and precise visual descriptions can
return the correct movie, episode, and timecode.
"""

from __future__ import annotations

import gc
import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from build_index_v2 import as_text_list, clean_text, resolve_model_path


SCENE_INDEX_PATH = Path("data/index/scenes.index")
SCENE_METADATA_PATH = Path("data/index/scene_metadata.pkl")
SCENE_REPORT_PATH = Path("data/index/scene_index_report.json")
OFFLINE_CATALOG_PATH = Path("data/processed/offline_media_enriched.json")


def _scene_text(parent: Dict, scene: Dict, previous_scene: Dict | None = None, next_scene: Dict | None = None) -> str:
    """Build a compact retrieval document with a little timeline context."""

    context = []
    if previous_scene:
        context.append(clean_text(previous_scene.get("transcript"))[:220])
    if next_scene:
        context.append(clean_text(next_scene.get("transcript"))[:220])

    parts = [
        f"Title: {clean_text(parent.get('title'))}",
        f"Type: {clean_text(parent.get('media_type')) or 'movie'}",
        f"Scene: {scene.get('scene_number') or ''}",
        f"Time: {scene.get('start_sec', 0)} to {scene.get('end_sec', 0)} seconds",
        f"Visual description: {clean_text(scene.get('visual_caption'))}",
        f"Dialogue: {clean_text(scene.get('transcript'))}",
        f"Visual tags: {', '.join(as_text_list(scene.get('visual_tags', [])))}",
        f"Mood: {', '.join(as_text_list(scene.get('mood_tags', [])))}",
        f"Keywords: {', '.join(as_text_list(scene.get('keywords', [])))}",
        f"Adjacent dialogue: {' '.join(item for item in context if item)}",
    ]
    return " | ".join(part for part in parts if not part.endswith(": "))


def prepare_scene_records(catalog: Iterable[Dict]) -> Tuple[List[str], List[Dict]]:
    documents: List[str] = []
    metadata: List[Dict] = []

    for parent in catalog:
        timeline = parent.get("scene_timeline") or []
        if not isinstance(timeline, list):
            continue
        for index, scene in enumerate(timeline):
            if not isinstance(scene, dict):
                continue
            previous_scene = timeline[index - 1] if index > 0 and isinstance(timeline[index - 1], dict) else None
            next_scene = timeline[index + 1] if index + 1 < len(timeline) and isinstance(timeline[index + 1], dict) else None
            document = _scene_text(parent, scene, previous_scene, next_scene)
            if not document.strip():
                continue

            scene_payload = dict(scene)
            metadata.append(
                {
                    "id": scene.get("scene_id") or f"{parent.get('id')}:scene:{index + 1}",
                    "parent_id": parent.get("id"),
                    "record_type": "scene",
                    "title": parent.get("title", "Unknown"),
                    "media_type": parent.get("media_type", "movie"),
                    "season": parent.get("season"),
                    "episode": parent.get("episode"),
                    "year": parent.get("release_year") or "N/A",
                    "genres": as_text_list(parent.get("genres", [])),
                    "director": parent.get("director") or "Unknown",
                    "cast": as_text_list(parent.get("cast", []))[:8],
                    "overview": clean_text(parent.get("overview"))[:600],
                    "rating": float(parent.get("vote_average") or 0.0),
                    "popularity": float(parent.get("popularity") or 0.0),
                    "mood_tags": as_text_list(scene.get("mood_tags", [])),
                    "keywords": as_text_list(scene.get("keywords", []))[:20],
                    "visual_tags": as_text_list(scene.get("visual_tags", []))[:20],
                    "source": parent.get("source", "offline_video_ingestion"),
                    "source_video": parent.get("source_video", ""),
                    "scene_count": parent.get("scene_count", len(timeline)),
                    "duration_sec": parent.get("duration_sec"),
                    "first_keyframe": parent.get("first_keyframe"),
                    "matched_scene": scene_payload,
                    "rich_text": document,
                }
            )
            documents.append(document)

    return documents, metadata


def build_scene_index(
    model_path: str = "",
    catalog_path: Path | str = OFFLINE_CATALOG_PATH,
    batch_size: int = 32,
    recover_catalog: bool = True,
    max_seq_length: int = 256,
    fp16: bool = False,
) -> Dict:
    catalog_file = Path(catalog_path)
    recovery = None
    if recover_catalog and catalog_file == OFFLINE_CATALOG_PATH:
        from ingestion.offline_video import rebuild_offline_catalog_from_scene_files

        recovery = rebuild_offline_catalog_from_scene_files()
    catalog = json.loads(catalog_file.read_text(encoding="utf-8")) if catalog_file.exists() else []
    documents, metadata = prepare_scene_records(catalog if isinstance(catalog, list) else [])
    if not model_path:
        movie_report_path = SCENE_INDEX_PATH.parent / "index_report.json"
        movie_report = json.loads(movie_report_path.read_text(encoding="utf-8")) if movie_report_path.exists() else {}
        model_path = str(movie_report.get("model") or "models/bge-large-en-v1.5")
    resolved_model = resolve_model_path(model_path, use_base_model=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = "float16" if device == "cuda" and fp16 else "float32"
    model = SentenceTransformer(resolved_model, device=device)
    model.max_seq_length = min(int(model.max_seq_length), max(64, int(max_seq_length)))
    if precision == "float16":
        model.half()
    dimension = int(model.get_sentence_embedding_dimension())

    if documents:
        embeddings = model.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
    else:
        embeddings = np.empty((0, dimension), dtype="float32")

    index = faiss.IndexFlatIP(dimension)
    if len(embeddings):
        index.add(embeddings)

    SCENE_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(SCENE_INDEX_PATH))
    with open(SCENE_METADATA_PATH, "wb") as handle:
        pickle.dump(metadata, handle)

    report = {
        "scene_vectors": len(metadata),
        "offline_titles": len({str(item.get('parent_id')) for item in metadata}),
        "embedding_dimension": dimension,
        "model": resolved_model,
        "device": device,
        "precision": precision,
        "batch_size": batch_size,
        "max_seq_length": model.max_seq_length,
        "index_path": str(SCENE_INDEX_PATH),
        "metadata_path": str(SCENE_METADATA_PATH),
        "catalog_recovery": recovery,
    }
    SCENE_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def load_scene_index(expected_dimension: int | None = None):
    if not SCENE_INDEX_PATH.exists() or not SCENE_METADATA_PATH.exists():
        return None, []
    index = faiss.read_index(str(SCENE_INDEX_PATH))
    with open(SCENE_METADATA_PATH, "rb") as handle:
        metadata = pickle.load(handle)
    if expected_dimension and int(index.d) != int(expected_dimension):
        return None, []
    if int(index.ntotal) != len(metadata):
        return None, []
    return index, metadata


if __name__ == "__main__":
    print(json.dumps(build_scene_index(), indent=2, ensure_ascii=False))
