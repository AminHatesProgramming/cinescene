from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data" / "processed" / "cinescene_catalog.json"
OUTPUT_PATH = ROOT / "docs" / "data" / "catalog.sample.json"
KEYFRAME_OUTPUT = ROOT / "docs" / "assets" / "keyframes"


def clean_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none"} else text


def as_list(values: Optional[Iterable]) -> List:
    return values if isinstance(values, list) else []


def year_for(movie: Dict) -> str:
    return clean_text(movie.get("release_year")) or clean_text(movie.get("year")) or "N/A"


def rating_for(movie: Dict) -> float:
    try:
        return float(movie.get("vote_average") or movie.get("rating") or 0)
    except Exception:
        return 0.0


def copy_keyframe(path_value: str) -> str:
    if not path_value:
        return ""
    source = Path(path_value)
    if not source.is_absolute():
        source = ROOT / source
    if not source.exists():
        return ""
    KEYFRAME_OUTPUT.mkdir(parents=True, exist_ok=True)
    target = KEYFRAME_OUTPUT / source.name
    shutil.copy2(source, target)
    return f"assets/keyframes/{target.name}"


def normalize_timeline(movie: Dict) -> List[Dict]:
    timeline = []
    for scene in as_list(movie.get("scene_timeline"))[:12]:
        if not isinstance(scene, dict):
            continue
        copied_keyframe = copy_keyframe(clean_text(scene.get("keyframe_path")))
        timeline.append(
            {
                "scene_number": scene.get("scene_number"),
                "start_sec": scene.get("start_sec"),
                "end_sec": scene.get("end_sec"),
                "duration_sec": scene.get("duration_sec"),
                "visual_caption": clean_text(scene.get("visual_caption")),
                "transcript": clean_text(scene.get("transcript")),
                "mood_tags": as_list(scene.get("mood_tags"))[:6],
                "keywords": as_list(scene.get("keywords"))[:8],
                "visual_tags": as_list(scene.get("visual_tags"))[:6],
                "keyframe": copied_keyframe,
            }
        )
    return timeline


def search_text_for(item: Dict) -> str:
    pieces = [
        item.get("title", ""),
        item.get("overview", ""),
        " ".join(item.get("genres", [])),
        " ".join(item.get("mood", [])),
        " ".join(item.get("keywords", [])),
        " ".join(item.get("visual_tags", [])),
        " ".join(item.get("scenes", [])),
    ]
    for scene in item.get("scene_timeline", []):
        pieces.extend(
            [
                scene.get("visual_caption", ""),
                scene.get("transcript", ""),
                " ".join(scene.get("mood_tags", [])),
                " ".join(scene.get("keywords", [])),
                " ".join(scene.get("visual_tags", [])),
            ]
        )
    return re.sub(r"\s+", " ", " ".join(pieces).lower()).strip()


def to_pwa_item(movie: Dict) -> Dict:
    timeline = normalize_timeline(movie)
    poster = copy_keyframe(clean_text(movie.get("first_keyframe")))
    if not poster and timeline:
        poster = clean_text(timeline[0].get("keyframe"))
    scenes = as_list(movie.get("scene_descriptions"))[:20]
    item = {
        "id": movie.get("id"),
        "title": clean_text(movie.get("title")) or clean_text(movie.get("original_title")) or "Unknown",
        "media_type": clean_text(movie.get("media_type")) or "movie",
        "season": movie.get("season"),
        "episode": movie.get("episode"),
        "year": year_for(movie),
        "genres": as_list(movie.get("genres"))[:8],
        "director": clean_text(movie.get("director")) or "Unknown",
        "cast": as_list(movie.get("cast"))[:8],
        "overview": clean_text(movie.get("overview"))[:900],
        "rating": rating_for(movie),
        "popularity": float(movie.get("popularity") or 0),
        "scenes": scenes,
        "scene_timeline": timeline,
        "scene_count": movie.get("scene_count") or len(timeline) or len(scenes),
        "mood": as_list(movie.get("mood_tags"))[:12],
        "keywords": as_list(movie.get("keywords"))[:18],
        "visual_tags": as_list(movie.get("visual_tags"))[:12],
        "source": movie.get("source", "tmdb_enriched"),
        "source_video": clean_text(movie.get("source_video")),
        "poster": poster,
    }
    item["score_hint"] = 3000 if item["source"] == "offline_video_ingestion" else 0
    item["search_text"] = search_text_for(item)
    return item


def main():
    movies = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    offline = [movie for movie in movies if movie.get("source") == "offline_video_ingestion"]
    tmdb = [movie for movie in movies if movie.get("source") != "offline_video_ingestion"]
    tmdb.sort(key=lambda movie: float(movie.get("popularity") or 0), reverse=True)
    selected = offline + tmdb[: max(0, 420 - len(offline))]
    items = [to_pwa_item(movie) for movie in selected]
    payload = {
        "generated_from": str(CATALOG_PATH.relative_to(ROOT)),
        "count": len(items),
        "offline_count": sum(1 for item in items if item["source"] == "offline_video_ingestion"),
        "items": items,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT_PATH), "count": len(items), "offline": payload["offline_count"]}, indent=2))


if __name__ == "__main__":
    main()
