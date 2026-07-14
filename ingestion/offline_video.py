"""
Offline video ingestion for CineScene.

This module turns an available local video into scene-level text records. It is
designed to work in two modes:

1. Lightweight mode with OpenCV only: detects visual scene boundaries from frame
   histogram changes and creates keyframe-backed scene records.
2. Enriched mode when optional captioning/transcription tools are installed:
   captions and transcripts can be attached to each scene before indexing.

The module never crawls protected services. It processes local files that the
project owner has permission to analyze.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


VIDEO_DIR = Path("data/offline_videos")
INGESTION_DIR = Path("data/processed/video_ingestion")
KEYFRAME_DIR = INGESTION_DIR / "keyframes"
OFFLINE_CATALOG = Path("data/processed/offline_media_enriched.json")
COMBINED_CATALOG = Path("data/processed/cinescene_catalog.json")
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".wmv"}
SUBTITLE_EXTENSIONS = {".srt", ".vtt"}


@dataclass
class SceneRecord:
    scene_id: str
    movie_title: str
    source_video: str
    start_sec: float
    end_sec: float
    duration_sec: float
    keyframe_path: Optional[str]
    visual_caption: str
    transcript: str = ""
    mood_tags: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    media_type: str = "movie"
    season: Optional[int] = None
    episode: Optional[int] = None

    @property
    def rich_text(self) -> str:
        parts = [
            f"Title: {self.movie_title}",
            f"Type: {self.media_type}",
            f"Scene time: {self.start_sec:.1f}s to {self.end_sec:.1f}s",
        ]
        if self.season is not None and self.episode is not None:
            parts.append(f"Episode: S{self.season:02d}E{self.episode:02d}")
        if self.visual_caption:
            parts.append(f"Visual scene: {self.visual_caption}")
        if self.transcript:
            parts.append(f"Dialogue/transcript: {self.transcript}")
        if self.mood_tags:
            parts.append(f"Mood: {', '.join(self.mood_tags)}")
        if self.keywords:
            parts.append(f"Keywords: {', '.join(self.keywords)}")
        return " | ".join(parts)


def _import_cv2():
    try:
        import cv2

        return cv2
    except Exception as exc:
        raise RuntimeError("OpenCV is required for offline video scene detection. Install opencv-python.") from exc


def _video_hash(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _copy_video(video_path: Path) -> Path:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    target = VIDEO_DIR / video_path.name
    if video_path.resolve() != target.resolve():
        shutil.copy2(video_path, target)
    return target


def _safe_title(title: str, fallback: str) -> str:
    title = (title or "").strip()
    return title if title else Path(fallback).stem


def parse_media_identity(path: Path, title_override: str = "") -> Dict:
    name = path.stem
    match = re.search(r"[Ss](\d{1,2})[ ._-]*[Ee](\d{1,2})", name)
    season = int(match.group(1)) if match else None
    episode = int(match.group(2)) if match else None
    media_type = "series" if match else "movie"

    cleaned = re.sub(r"[Ss]\d{1,2}[ ._-]*[Ee]\d{1,2}.*", "", name)
    cleaned = re.sub(r"[._-]+", " ", cleaned).strip()
    title = _safe_title(title_override, cleaned or name)

    return {
        "title": title,
        "media_type": media_type,
        "season": season,
        "episode": episode,
    }


def find_sidecar_subtitle(video_path: Path) -> Optional[Path]:
    for ext in SUBTITLE_EXTENSIONS:
        candidate = video_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def parse_timestamp(value: str) -> float:
    value = value.strip().replace(",", ".")
    parts = value.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    return float(value)


def load_subtitle_segments(path: Optional[Path]) -> List[Dict]:
    if not path or not path.exists():
        return []

    text = path.read_text(encoding="utf-8", errors="ignore")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", text)
    segments = []
    for block in blocks:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if not lines:
            continue
        time_line = next((line for line in lines if "-->" in line), "")
        if not time_line:
            continue
        start_raw, end_raw = [part.strip().split(" ")[0] for part in time_line.split("-->", 1)]
        subtitle_text = " ".join(line for line in lines if line != time_line and not line.isdigit())
        subtitle_text = re.sub(r"<[^>]+>", "", subtitle_text).strip()
        if subtitle_text:
            segments.append(
                {
                    "start": parse_timestamp(start_raw),
                    "end": parse_timestamp(end_raw),
                    "text": subtitle_text,
                }
            )
    return segments


def subtitle_text_for_range(segments: List[Dict], start_sec: float, end_sec: float, max_chars: int = 700) -> str:
    parts = [
        segment["text"]
        for segment in segments
        if segment["end"] >= start_sec and segment["start"] <= end_sec
    ]
    text = " ".join(parts)
    return text[:max_chars]


def _infer_mood_and_keywords(text: str) -> Dict[str, List[str]]:
    text_lower = text.lower()
    mood_rules = {
        "dark": ["night", "shadow", "blood", "crime", "alone", "fear", "murder"],
        "romantic": ["love", "kiss", "wedding", "relationship"],
        "suspenseful": ["chase", "detective", "mystery", "escape", "threat"],
        "action": ["fight", "explosion", "gun", "battle", "run"],
        "science fiction": ["space", "alien", "robot", "future", "planet"],
        "dramatic": ["cry", "family", "conflict", "loss"],
    }
    moods = [mood for mood, terms in mood_rules.items() if any(term in text_lower for term in terms)]

    keyword_candidates = [
        "night",
        "city",
        "car",
        "fight",
        "chase",
        "space",
        "forest",
        "room",
        "crowd",
        "dialogue",
        "music",
        "explosion",
        "detective",
        "family",
        "love",
    ]
    keywords = [keyword for keyword in keyword_candidates if keyword in text_lower]
    return {"mood_tags": moods, "keywords": keywords}


def _caption_from_visual_features(brightness: float, motion_hint: float, start_sec: float, end_sec: float) -> str:
    light = "dark low-key" if brightness < 70 else "bright" if brightness > 155 else "balanced"
    energy = "high-motion" if motion_hint > 0.35 else "quiet"
    return (
        f"A {light}, {energy} visual segment from {start_sec:.1f}s to {end_sec:.1f}s. "
        "Attach a vision-language captioner for object-level scene text."
    )


def detect_scenes(
    video_path: str,
    movie_title: str = "",
    min_scene_sec: float = 8.0,
    threshold: float = 0.45,
    sample_fps: float = 1.0,
) -> List[SceneRecord]:
    cv2 = _import_cv2()

    source = Path(video_path)
    if not source.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    stored_video = _copy_video(source)
    identity = parse_media_identity(stored_video, movie_title)
    title = identity["title"]
    video_id = _video_hash(stored_video)
    subtitle_segments = load_subtitle_segments(find_sidecar_subtitle(source) or find_sidecar_subtitle(stored_video))

    cap = cv2.VideoCapture(str(stored_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {stored_video}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / native_fps if frame_count else 0.0
    step = max(1, int(native_fps / max(sample_fps, 0.1)))
    min_scene_frames = int(min_scene_sec * native_fps)

    boundaries = [0]
    frame_no = 0
    previous_hist = None
    previous_frame = None
    brightness_samples: List[float] = []
    motion_samples: List[float] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_no % step != 0:
            frame_no += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist)
        brightness_samples.append(float(gray.mean()))

        if previous_frame is not None:
            diff = cv2.absdiff(gray, previous_frame)
            motion_samples.append(float(diff.mean() / 255.0))

        if previous_hist is not None:
            distance = cv2.compareHist(previous_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if distance >= threshold and frame_no - boundaries[-1] >= min_scene_frames:
                boundaries.append(frame_no)

        previous_hist = hist
        previous_frame = gray
        frame_no += 1

    cap.release()
    if frame_count and boundaries[-1] != frame_count - 1:
        boundaries.append(frame_count - 1)
    elif not frame_count:
        boundaries.append(frame_no)

    records: List[SceneRecord] = []
    KEYFRAME_DIR.mkdir(parents=True, exist_ok=True)

    for scene_index in range(max(0, len(boundaries) - 1)):
        start_frame = boundaries[scene_index]
        end_frame = boundaries[scene_index + 1]
        start_sec = start_frame / native_fps
        end_sec = end_frame / native_fps
        if end_sec - start_sec < 1:
            continue

        keyframe_path = _extract_keyframe(cv2, stored_video, video_id, scene_index, start_frame, end_frame, native_fps)
        brightness = brightness_samples[min(scene_index, len(brightness_samples) - 1)] if brightness_samples else 110.0
        motion = motion_samples[min(scene_index, len(motion_samples) - 1)] if motion_samples else 0.0
        caption = _caption_from_visual_features(brightness, motion, start_sec, end_sec)
        transcript = subtitle_text_for_range(subtitle_segments, start_sec, end_sec)
        inferred = _infer_mood_and_keywords(f"{caption} {transcript}")

        records.append(
            SceneRecord(
                scene_id=f"{video_id}-{scene_index + 1:03d}",
                movie_title=title,
                source_video=str(stored_video),
                start_sec=round(start_sec, 2),
                end_sec=round(end_sec, 2),
                duration_sec=round(end_sec - start_sec, 2),
                keyframe_path=str(keyframe_path) if keyframe_path else None,
                visual_caption=caption,
                transcript=transcript,
                mood_tags=inferred["mood_tags"],
                keywords=inferred["keywords"],
                media_type=identity["media_type"],
                season=identity["season"],
                episode=identity["episode"],
            )
        )

    save_scene_records(records, video_id)
    return records


def _extract_keyframe(cv2, video_path: Path, video_id: str, scene_index: int, start_frame: int, end_frame: int, fps: float):
    cap = cv2.VideoCapture(str(video_path))
    middle = start_frame + max(0, math.floor((end_frame - start_frame) / 2))
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    target = KEYFRAME_DIR / f"{video_id}_{scene_index + 1:03d}.jpg"
    cv2.imwrite(str(target), frame)
    return target


def save_scene_records(records: List[SceneRecord], video_id: str) -> Path:
    INGESTION_DIR.mkdir(parents=True, exist_ok=True)
    payload = [asdict(record) | {"rich_text": record.rich_text} for record in records]
    output = INGESTION_DIR / f"{video_id}_scenes.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return output


def scene_records_to_movie_document(records: List[SceneRecord]) -> Dict:
    if not records:
        raise ValueError("No scene records to convert")

    title = records[0].movie_title
    all_moods = sorted({mood for record in records for mood in record.mood_tags})
    all_keywords = sorted({keyword for record in records for keyword in record.keywords})
    scene_descriptions = [record.rich_text for record in records]

    return {
        "id": f"offline:{records[0].scene_id.split('-')[0]}",
        "title": title,
        "original_title": title,
        "media_type": records[0].media_type,
        "season": records[0].season,
        "episode": records[0].episode,
        "release_year": "",
        "genres": [],
        "keywords": all_keywords,
        "cast": [],
        "director": "",
        "overview": f"Offline-ingested film document built from {len(records)} detected scenes.",
        "scene_descriptions": scene_descriptions,
        "mood_tags": all_moods,
        "rich_text": " | ".join(scene_descriptions),
        "source": "offline_video_ingestion",
    }


def find_video_files(root: str) -> List[Path]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Offline media folder not found: {root}")
    if root_path.is_file():
        return [root_path] if root_path.suffix.lower() in VIDEO_EXTENSIONS else []
    return sorted(path for path in root_path.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)


def load_json_list(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def save_json_list(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def crawl_offline_videos(
    root: str,
    title_prefix: str = "",
    min_scene_sec: float = 8.0,
    threshold: float = 0.45,
    sample_fps: float = 1.0,
    update_catalog: bool = True,
) -> Dict:
    videos = find_video_files(root)
    documents = []
    jobs = []

    for video in videos:
        identity = parse_media_identity(video, title_prefix)
        scenes = detect_scenes(
            str(video),
            movie_title=identity["title"],
            min_scene_sec=min_scene_sec,
            threshold=threshold,
            sample_fps=sample_fps,
        )
        document = scene_records_to_movie_document(scenes) if scenes else None
        if document:
            document["source_video"] = str(video)
            documents.append(document)
        jobs.append(
            {
                "video": str(video),
                "title": identity["title"],
                "media_type": identity["media_type"],
                "season": identity["season"],
                "episode": identity["episode"],
                "scene_count": len(scenes),
                "document_id": document["id"] if document else None,
            }
        )

    if update_catalog and documents:
        existing = load_json_list(OFFLINE_CATALOG)
        by_id = {str(item.get("id")): item for item in existing}
        for document in documents:
            by_id[str(document.get("id"))] = document
        offline_catalog = list(by_id.values())
        save_json_list(OFFLINE_CATALOG, offline_catalog)

        tmdb_catalog = load_json_list(Path("data/processed/movies_enriched.json"))
        combined = tmdb_catalog + offline_catalog
        save_json_list(COMBINED_CATALOG, combined)
    else:
        offline_catalog = load_json_list(OFFLINE_CATALOG)

    report = {
        "root": str(root),
        "videos_found": len(videos),
        "videos_processed": len(jobs),
        "documents_created": len(documents),
        "offline_catalog": str(OFFLINE_CATALOG),
        "combined_catalog": str(COMBINED_CATALOG),
        "jobs": jobs,
    }
    report_path = INGESTION_DIR / "offline_crawl_report.json"
    save_json_list(report_path, [report])
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect scenes from a local video")
    parser.add_argument("video")
    parser.add_argument("--title", default="")
    parser.add_argument("--crawl", action="store_true", help="Treat input as a folder and crawl videos recursively")
    args = parser.parse_args()

    if args.crawl:
        print(json.dumps(crawl_offline_videos(args.video, title_prefix=args.title), indent=2, ensure_ascii=False))
    else:
        scenes = detect_scenes(args.video, movie_title=args.title)
        print(f"Detected {len(scenes)} scenes")
