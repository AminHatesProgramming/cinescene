from __future__ import annotations

import gc
import os
import json
import re
import threading
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.storage import AppMemory


ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "frontend"
INDEX_REPORT_PATH = ROOT / "data" / "index" / "index_report.json"
OFFLINE_CATALOG_PATH = ROOT / "data" / "processed" / "offline_media_enriched.json"
COMBINED_CATALOG_PATH = ROOT / "data" / "processed" / "cinescene_catalog.json"
UPLOAD_DIR = ROOT / "data" / "offline_videos" / "uploads"
VIDEO_DIR = ROOT / "data" / "offline_videos"
MAX_UPLOAD_BYTES = int(os.getenv("CINESCENE_MAX_UPLOAD_GB", "12")) * 1024 * 1024 * 1024

app = FastAPI(
    title="CineScene API",
    description="Semantic movie retrieval with offline video scene ingestion.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

memory = AppMemory()
JOBS: Dict[str, Dict] = {}
JOBS_LOCK = threading.Lock()
ENGINE_LOCK = threading.RLock()
SEARCH_ENGINE = None
SEARCH_ENGINE_ERROR: Optional[str] = None
SEARCH_ENGINE_LOADED = False


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(8, ge=1, le=20)
    use_reranking: bool = False


class FavoriteRequest(BaseModel):
    movie: Dict


class FeedbackRequest(BaseModel):
    query: str
    movie_title: str
    signal: str = Field(..., pattern="^(relevant|not_relevant|love|hide)$")
    note: Optional[str] = None


class OfflineCrawlRequest(BaseModel):
    root: str = Field(..., min_length=1)
    title_prefix: str = ""
    min_scene_sec: float = Field(8.0, ge=1.0, le=120.0)
    max_scene_sec: float = Field(90.0, ge=5.0, le=600.0)
    threshold: float = Field(0.45, ge=0.05, le=1.0)
    sample_fps: float = Field(1.0, ge=0.1, le=5.0)
    update_catalog: bool = True
    extract_embedded_subtitles: bool = True
    transcribe_audio: bool = False
    enable_vision: bool = False
    whisper_model: str = Field("small", pattern="^(tiny|base|small|medium|large-v2|large-v3)$")
    vision_model: str = "Salesforce/blip-image-captioning-base"


class CrawlProbeRequest(BaseModel):
    root: str = Field(..., min_length=1)


class IndexBuildRequest(BaseModel):
    input_path: str = ""
    model_path: str = "models/bge-large-en-v1.5"
    batch_size: int = Field(8, ge=1, le=256)
    use_hnsw: bool = False
    use_base_model: bool = False


def _now() -> float:
    return round(time.time(), 3)


def create_job(kind: str, label: str) -> Dict:
    job_id = str(uuid4())
    job = {
        "id": job_id,
        "kind": kind,
        "label": label,
        "status": "queued",
        "created_at": _now(),
        "updated_at": _now(),
        "elapsed_sec": 0.0,
        "result": None,
        "error": None,
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
    return job


def update_job(job_id: str, **fields):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job.update(fields)
        job["updated_at"] = _now()
        job["elapsed_sec"] = round(job["updated_at"] - job["created_at"], 3)


def get_job(job_id: str) -> Dict:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        payload = dict(job)
    if payload.get("status") in {"queued", "running"}:
        payload["elapsed_sec"] = round(_now() - float(payload.get("created_at", _now())), 3)
    return payload


def session_id(value: Optional[str] = Header(default=None, alias="X-CineScene-Session")) -> str:
    return value or "anonymous"


def get_search_engine():
    global SEARCH_ENGINE, SEARCH_ENGINE_ERROR, SEARCH_ENGINE_LOADED

    with ENGINE_LOCK:
        if SEARCH_ENGINE_LOADED:
            return SEARCH_ENGINE, SEARCH_ENGINE_ERROR
        try:
            from hybrid_search import HybridSearchEngine

            enable_reranker = os.getenv("CINESCENE_ENABLE_RERANKER", "0") == "1"
            SEARCH_ENGINE = HybridSearchEngine(enable_reranker=enable_reranker)
            SEARCH_ENGINE_ERROR = None
        except Exception as exc:
            SEARCH_ENGINE = None
            SEARCH_ENGINE_ERROR = str(exc)
        SEARCH_ENGINE_LOADED = True
        return SEARCH_ENGINE, SEARCH_ENGINE_ERROR


def clear_search_engine():
    global SEARCH_ENGINE, SEARCH_ENGINE_ERROR, SEARCH_ENGINE_LOADED

    with ENGINE_LOCK:
        SEARCH_ENGINE = None
        SEARCH_ENGINE_ERROR = None
        SEARCH_ENGINE_LOADED = False


def release_runtime_models(include_media: bool = False):
    """Free cached GPU models before an index build to avoid 4 GB VRAM spikes."""

    clear_search_engine()
    if include_media:
        try:
            from ingestion.media_intelligence import release_media_models

            release_media_models()
        except Exception:
            pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def rebuild_scene_index_safely(model_path: str, batch_size: int = 16) -> Dict:
    from scene_index import build_scene_index

    with ENGINE_LOCK:
        release_runtime_models(include_media=True)
        report = build_scene_index(model_path=model_path, batch_size=batch_size)
        clear_search_engine()
        return report


def rebuild_all_indexes_safely(payload: IndexBuildRequest, input_path: str, before_scene=None):
    from build_index_v2 import IndexBuilderV2
    from scene_index import build_scene_index

    with ENGINE_LOCK:
        release_runtime_models(include_media=True)
        builder = IndexBuilderV2(
            model_path=payload.model_path,
            use_base_model=payload.use_base_model,
            batch_size=payload.batch_size,
        )
        builder.build_and_save(enriched_path=input_path, use_hnsw=payload.use_hnsw)
        if before_scene:
            before_scene()
        scene_report = build_scene_index(
            model_path=payload.model_path,
            batch_size=min(payload.batch_size, 64),
        )
        clear_search_engine()
        engine, error = get_search_engine()
        return scene_report, engine, error


def read_json(path: Path, default):
    try:
        if not path.exists():
            return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def active_embedding_model() -> str:
    report = read_json(INDEX_REPORT_PATH, {})
    return str(report.get("model") or "models/bge-large-en-v1.5")


def keyframe_url(path_value: Optional[str]) -> Optional[str]:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        candidate = ROOT / path_value
        if candidate.exists():
            path = candidate
    return f"/media/keyframes/{path.name}" if path.name else None


def video_url(path_value: Optional[str]) -> Optional[str]:
    if not path_value:
        return None
    try:
        path = Path(path_value)
        if not path.is_absolute():
            path = ROOT / path
        relative = path.resolve().relative_to(VIDEO_DIR.resolve())
        return "/media/videos/" + "/".join(relative.parts)
    except Exception:
        return None


def decorate_search_media(results: List[Dict]) -> List[Dict]:
    decorated = []
    for result in results:
        item = dict(result)
        item["first_keyframe_url"] = keyframe_url(item.get("first_keyframe"))
        item["video_url"] = video_url(item.get("source_video"))
        for field in ("matched_scene",):
            scene = item.get(field)
            if isinstance(scene, dict):
                scene = dict(scene)
                scene["keyframe_url"] = keyframe_url(scene.get("keyframe_path"))
                item[field] = scene
        for field in ("matched_scenes", "scene_timeline"):
            scenes = []
            for scene in item.get(field, []) or []:
                if isinstance(scene, dict):
                    scene = dict(scene)
                    scene["keyframe_url"] = keyframe_url(scene.get("keyframe_path"))
                scenes.append(scene)
            item[field] = scenes
        decorated.append(item)
    return decorated


def safe_upload_name(filename: str, fallback: str) -> str:
    original = Path(filename or fallback)
    stem = re.sub(r"[^A-Za-z0-9._ -]+", "_", original.stem).strip(" ._") or Path(fallback).stem
    suffix = re.sub(r"[^A-Za-z0-9.]", "", original.suffix.lower())
    return f"{stem[:120]}{suffix}"


async def save_upload(upload: UploadFile, target: Path, max_bytes: int = MAX_UPLOAD_BYTES) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(target, "wb") as handle:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                handle.close()
                target.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Uploaded file exceeds the configured size limit.")
            handle.write(chunk)
    await upload.close()
    return written


def runtime_status():
    cuda = {"available": False, "device": None}
    try:
        import torch

        cuda["available"] = bool(torch.cuda.is_available())
        if cuda["available"]:
            cuda["device"] = torch.cuda.get_device_name(0)
    except Exception:
        pass

    offline_catalog = read_json(OFFLINE_CATALOG_PATH, [])
    combined_catalog = read_json(COMBINED_CATALOG_PATH, [])
    index_report = read_json(INDEX_REPORT_PATH, {})
    scene_index_report = read_json(ROOT / "data" / "index" / "scene_index_report.json", {})
    return {
        "cuda": cuda,
        "index_report": index_report,
        "scene_index_report": scene_index_report,
        "offline_documents": len(offline_catalog) if isinstance(offline_catalog, list) else 0,
        "combined_documents": len(combined_catalog) if isinstance(combined_catalog, list) else 0,
        "paths": {
            "index_report": str(INDEX_REPORT_PATH),
            "offline_catalog": str(OFFLINE_CATALOG_PATH),
            "combined_catalog": str(COMBINED_CATALOG_PATH),
        },
    }


def crawler_status_payload():
    from ingestion.offline_video import INGESTION_DIR, KEYFRAME_DIR, clean_media_title, crawler_capabilities

    offline_catalog = read_json(OFFLINE_CATALOG_PATH, [])
    combined_catalog = read_json(COMBINED_CATALOG_PATH, [])
    report_path = INGESTION_DIR / "offline_crawl_report.json"
    report_rows = read_json(report_path, [])
    latest_report = report_rows[-1] if isinstance(report_rows, list) and report_rows else {}

    scene_files = sorted(INGESTION_DIR.glob("*_scenes.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    scene_previews = []
    scene_total = 0
    for index, scene_file in enumerate(scene_files):
        scenes = read_json(scene_file, [])
        if not isinstance(scenes, list):
            continue
        scene_total += len(scenes)
        if index >= 12:
            continue
        first = scenes[0] if scenes else {}
        timeline = []
        for scene in scenes[:8]:
            timeline.append(
                {
                    "scene_id": scene.get("scene_id"),
                    "scene_number": scene.get("scene_number"),
                    "start_sec": scene.get("start_sec"),
                    "end_sec": scene.get("end_sec"),
                    "duration_sec": scene.get("duration_sec"),
                    "visual_caption": scene.get("visual_caption", ""),
                    "transcript": scene.get("transcript", ""),
                    "mood_tags": scene.get("mood_tags", []) or [],
                    "keywords": scene.get("keywords", []) or [],
                    "visual_tags": scene.get("visual_tags", []) or [],
                    "keyframe_url": keyframe_url(scene.get("keyframe_path")),
                    "rich_text": scene.get("rich_text", "")[:900],
                }
            )
        scene_previews.append(
            {
                "file": str(scene_file),
                "scene_count": len(scenes),
                "title": clean_media_title(first.get("movie_title") or first.get("title") or scene_file.stem),
                "media_type": first.get("media_type"),
                "season": first.get("season"),
                "episode": first.get("episode"),
                "source_video": first.get("source_video", ""),
                "first_scene": first.get("rich_text", "")[:600],
                "keyframe_path": first.get("keyframe_path"),
                "keyframe_url": keyframe_url(first.get("keyframe_path")),
                "timeline": timeline,
            }
        )

    keyframe_count = len(list(KEYFRAME_DIR.glob("*.jpg"))) if KEYFRAME_DIR.exists() else 0
    offline_count = len(offline_catalog) if isinstance(offline_catalog, list) else 0
    combined_count = len(combined_catalog) if isinstance(combined_catalog, list) else 0
    series_count = 0
    movie_count = 0
    if isinstance(offline_catalog, list):
        series_count = sum(1 for item in offline_catalog if item.get("media_type") == "series")
        movie_count = sum(1 for item in offline_catalog if item.get("media_type") != "series")
    active_jobs = []
    with JOBS_LOCK:
        for job in JOBS.values():
            if job.get("kind") in {"video_upload", "offline_crawl", "index_rebuild"} and job.get("status") in {"queued", "running"}:
                active_jobs.append(dict(job))

    return {
        "ok": True,
        "crawler_enabled": crawler_capabilities().get("enabled", False),
        "capabilities": crawler_capabilities(),
        "offline_documents": offline_count,
        "offline_series_documents": series_count,
        "offline_movie_documents": movie_count,
        "combined_documents": combined_count,
        "scene_files": len(scene_files),
        "scene_total": scene_total,
        "keyframes": keyframe_count,
        "latest_report": latest_report,
        "report_history": report_rows[-8:] if isinstance(report_rows, list) else [],
        "active_jobs": active_jobs,
        "scene_previews": scene_previews,
        "paths": {
            "offline_catalog": str(OFFLINE_CATALOG_PATH),
            "combined_catalog": str(COMBINED_CATALOG_PATH),
            "report": str(report_path),
            "keyframes": str(KEYFRAME_DIR),
        },
    }


@app.get("/api/health")
def health():
    engine, error = get_search_engine()
    return {
        "ok": error is None,
        "engine": engine.status() if engine else None,
        "error": error,
        "memory_db": str(memory.db_path),
        "frontend": FRONTEND_DIR.exists(),
        "runtime": runtime_status(),
    }


@app.get("/api/crawl/status")
@app.get("/api/crawler/status", include_in_schema=False)
def crawl_status():
    return crawler_status_payload()


@app.post("/api/crawl/probe")
def crawl_probe(payload: CrawlProbeRequest):
    try:
        from ingestion.offline_video import find_sidecar_subtitles, find_video_files, parse_media_identity

        videos = find_video_files(payload.root)
        preview = []
        for video in videos[:20]:
            identity = parse_media_identity(video)
            preview.append(
                {
                    "path": str(video),
                    "title": identity["title"],
                    "media_type": identity["media_type"],
                    "season": identity["season"],
                    "episode": identity["episode"],
                    "subtitle_files": [str(path) for path in find_sidecar_subtitles(video)],
                }
            )
        return {
            "ok": True,
            "root": payload.root,
            "videos_found": len(videos),
            "preview": preview,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    return get_job(job_id)


@app.post("/api/reload")
def reload_engine():
    clear_search_engine()
    engine, error = get_search_engine()
    return {
        "ok": error is None,
        "engine": engine.status() if engine else None,
        "error": error,
        "runtime": runtime_status(),
    }


@app.post("/api/search")
def search(payload: SearchRequest, sid: str = Header(default="anonymous", alias="X-CineScene-Session")):
    with ENGINE_LOCK:
        engine, error = get_search_engine()
        if error or engine is None:
            raise HTTPException(status_code=503, detail=f"Search engine is not ready: {error}")
        results = decorate_search_media(
            engine.search(payload.query, top_k=payload.top_k, use_reranking=payload.use_reranking)
        )
    memory.add_search(sid, payload.query, payload.top_k, payload.use_reranking, results)
    return {"query": payload.query, "count": len(results), "results": results}


@app.get("/api/history")
def history(sid: str = Header(default="anonymous", alias="X-CineScene-Session")):
    return {"items": memory.recent_searches(sid)}


@app.post("/api/favorites")
def add_favorite(payload: FavoriteRequest, sid: str = Header(default="anonymous", alias="X-CineScene-Session")):
    memory.add_favorite(sid, payload.movie)
    return {"ok": True}


@app.get("/api/favorites")
def favorites(sid: str = Header(default="anonymous", alias="X-CineScene-Session")):
    return {"items": memory.favorites(sid)}


@app.post("/api/feedback")
def feedback(payload: FeedbackRequest, sid: str = Header(default="anonymous", alias="X-CineScene-Session")):
    memory.add_feedback(sid, payload.query, payload.movie_title, payload.signal, payload.note)
    return {"ok": True}


def run_uploaded_video_job(job_id: str, upload_root: Path, options: Dict, sid: str):
    start = time.perf_counter()
    update_job(job_id, status="running", stage="scene_detection", progress=4)
    try:
        from ingestion.offline_video import crawl_offline_videos

        def progress(event: Dict):
            fraction = event.get("overall_fraction")
            if fraction is None:
                total = max(1, int(event.get("total") or 1))
                fraction = int(event.get("processed") or 0) / total
            update_job(
                job_id,
                status="running",
                progress=round(4 + min(1.0, float(fraction)) * 74, 2),
                current_video=event.get("current_video"),
                stage=event.get("stage", "scene_detection"),
            )

        report = crawl_offline_videos(
            str(upload_root),
            title_prefix=options["movie_title"],
            min_scene_sec=options["min_scene_sec"],
            max_scene_sec=options["max_scene_sec"],
            threshold=options["threshold"],
            sample_fps=options["sample_fps"],
            update_catalog=True,
            extract_embedded_subtitles=options["extract_embedded_subtitles"],
            transcribe_audio=options["transcribe_audio"],
            enable_vision=options["enable_vision"],
            whisper_model=options["whisper_model"],
            vision_model=options["vision_model"],
            progress_callback=progress,
        )
        update_job(job_id, stage="scene_index", progress=82)
        scene_index_report = rebuild_scene_index_safely(active_embedding_model(), batch_size=16)

        for item in report.get("jobs", []):
            memory.add_ingestion(
                sid,
                item.get("title") or options["movie_title"],
                item.get("video", ""),
                [{"scene_count": item.get("scene_count"), "document_id": item.get("document_id")}],
                scene_count=int(item.get("scene_count") or 0),
            )
        update_job(
            job_id,
            status="completed",
            stage="ready",
            progress=100,
            elapsed_sec=round(time.perf_counter() - start, 3),
            result={
                "ok": True,
                "report": report,
                "scene_index": scene_index_report,
                "crawler": crawler_status_payload(),
            },
        )
    except Exception as exc:
        update_job(job_id, status="failed", error=str(exc), traceback=traceback.format_exc(limit=8))


@app.post("/api/ingest/video")
async def ingest_video(
    video: UploadFile = File(...),
    subtitles: Optional[List[UploadFile]] = File(default=None),
    movie_title: str = Form(default=""),
    min_scene_sec: float = Form(default=4.0),
    max_scene_sec: float = Form(default=45.0),
    threshold: float = Form(default=0.35),
    sample_fps: float = Form(default=2.0),
    extract_embedded_subtitles: bool = Form(default=True),
    transcribe_audio: bool = Form(default=False),
    enable_vision: bool = Form(default=False),
    whisper_model: str = Form(default="small"),
    vision_model: str = Form(default="Salesforce/blip-image-captioning-base"),
    sid: str = Header(default="anonymous", alias="X-CineScene-Session"),
):
    from ingestion.offline_video import SUBTITLE_EXTENSIONS, VIDEO_EXTENSIONS

    video_name = safe_upload_name(video.filename or "uploaded-video.mp4", "uploaded-video.mp4")
    if Path(video_name).suffix.lower() not in VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported video format.")
    if not 1.0 <= min_scene_sec <= 120.0 or not 5.0 <= max_scene_sec <= 600.0:
        raise HTTPException(status_code=400, detail="Scene duration settings are outside the supported range.")
    if min_scene_sec >= max_scene_sec:
        raise HTTPException(status_code=400, detail="Maximum scene duration must be greater than minimum duration.")

    title = movie_title.strip() or Path(video_name).stem
    job = create_job("video_upload", f"Analyze {title}")
    upload_root = UPLOAD_DIR / job["id"]
    upload_root.mkdir(parents=True, exist_ok=True)
    video_target = upload_root / video_name
    try:
        video_bytes = await save_upload(video, video_target)
        subtitle_files = []
        for index, subtitle in enumerate(subtitles or [], start=1):
            suffix = Path(subtitle.filename or "subtitle.srt").suffix.lower()
            if suffix not in SUBTITLE_EXTENSIONS:
                raise HTTPException(status_code=400, detail=f"Unsupported subtitle format: {suffix}")
            target = upload_root / f"{video_target.stem}.upload-{index}{suffix}"
            subtitle_bytes = await save_upload(subtitle, target, max_bytes=64 * 1024 * 1024)
            subtitle_files.append({"name": target.name, "bytes": subtitle_bytes})
    except Exception as exc:
        update_job(job["id"], status="failed", error=str(exc))
        raise

    options = {
        "movie_title": title,
        "min_scene_sec": min_scene_sec,
        "max_scene_sec": max_scene_sec,
        "threshold": threshold,
        "sample_fps": sample_fps,
        "extract_embedded_subtitles": extract_embedded_subtitles,
        "transcribe_audio": transcribe_audio,
        "enable_vision": enable_vision,
        "whisper_model": whisper_model,
        "vision_model": vision_model,
    }
    update_job(
        job["id"],
        status="queued",
        stage="uploaded",
        progress=2,
        upload={"video": video_name, "bytes": video_bytes, "subtitles": subtitle_files},
    )
    thread = threading.Thread(
        target=run_uploaded_video_job,
        args=(job["id"], upload_root, options, sid),
        daemon=True,
    )
    thread.start()
    return {"ok": True, "job": get_job(job["id"]), "upload": {"video": video_name, "bytes": video_bytes, "subtitles": subtitle_files}}


@app.get("/api/ingestions")
def ingestions(sid: str = Header(default="anonymous", alias="X-CineScene-Session")):
    return {"items": memory.ingestions(sid)}


@app.post("/api/crawl/offline")
def crawl_offline(payload: OfflineCrawlRequest, sid: str = Header(default="anonymous", alias="X-CineScene-Session")):
    try:
        from ingestion.offline_video import crawl_offline_videos

        report = crawl_offline_videos(
            payload.root,
            title_prefix=payload.title_prefix,
            min_scene_sec=payload.min_scene_sec,
            max_scene_sec=payload.max_scene_sec,
            threshold=payload.threshold,
            sample_fps=payload.sample_fps,
            update_catalog=payload.update_catalog,
            extract_embedded_subtitles=payload.extract_embedded_subtitles,
            transcribe_audio=payload.transcribe_audio,
            enable_vision=payload.enable_vision,
            whisper_model=payload.whisper_model,
            vision_model=payload.vision_model,
        )
        scene_index_report = None
        if payload.update_catalog:
            scene_index_report = rebuild_scene_index_safely(active_embedding_model(), batch_size=16)
        for job in report.get("jobs", []):
            memory.add_ingestion(
                sid,
                job.get("title") or Path(job.get("video", "")).stem,
                job.get("video", ""),
                [{"scene_count": job.get("scene_count"), "document_id": job.get("document_id")}],
                scene_count=int(job.get("scene_count") or 0),
        )
        return {
            "ok": True,
            "report": report,
            "scene_index": scene_index_report,
            "next_steps": ["The scene index is updated automatically; search the dialogue or visual scene now."],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def run_crawl_job(job_id: str, payload: OfflineCrawlRequest, sid: str):
    start = time.perf_counter()
    update_job(job_id, status="running")
    try:
        from ingestion.offline_video import crawl_offline_videos

        def progress(event: Dict):
            total = max(1, int(event.get("total") or 1))
            processed = int(event.get("processed") or 0)
            fraction = event.get("overall_fraction")
            if fraction is None:
                fraction = processed / total
            update_job(
                job_id,
                status="running",
                progress=round(min(82, float(fraction) * 82), 2),
                current_video=event.get("current_video"),
                stage=event.get("stage"),
                processed=processed,
                total=total,
            )

        report = crawl_offline_videos(
            payload.root,
            title_prefix=payload.title_prefix,
            min_scene_sec=payload.min_scene_sec,
            max_scene_sec=payload.max_scene_sec,
            threshold=payload.threshold,
            sample_fps=payload.sample_fps,
            update_catalog=payload.update_catalog,
            extract_embedded_subtitles=payload.extract_embedded_subtitles,
            transcribe_audio=payload.transcribe_audio,
            enable_vision=payload.enable_vision,
            whisper_model=payload.whisper_model,
            vision_model=payload.vision_model,
            progress_callback=progress,
        )
        scene_index_report = None
        if payload.update_catalog:
            update_job(job_id, stage="scene_index", progress=86)
            scene_index_report = rebuild_scene_index_safely(active_embedding_model(), batch_size=16)
        for job in report.get("jobs", []):
            memory.add_ingestion(
                sid,
                job.get("title") or Path(job.get("video", "")).stem,
                job.get("video", ""),
                [{"scene_count": job.get("scene_count"), "document_id": job.get("document_id")}],
                scene_count=int(job.get("scene_count") or 0),
            )
        update_job(
            job_id,
            status="completed",
            progress=100,
            elapsed_sec=round(time.perf_counter() - start, 3),
            result={
                "ok": True,
                "report": report,
                "scene_index": scene_index_report,
                "crawler": crawler_status_payload(),
            },
        )
    except Exception as exc:
        update_job(
            job_id,
            status="failed",
            error=str(exc),
            traceback=traceback.format_exc(limit=6),
        )


@app.post("/api/crawl/offline-async")
def crawl_offline_async(payload: OfflineCrawlRequest, sid: str = Header(default="anonymous", alias="X-CineScene-Session")):
    job = create_job("offline_crawl", f"Crawl {payload.root}")
    update_job(job["id"], status="running")
    thread = threading.Thread(target=run_crawl_job, args=(job["id"], payload, sid), daemon=True)
    thread.start()
    return {"ok": True, "job": get_job(job["id"])}


@app.post("/api/index/rebuild")
def rebuild_index(payload: IndexBuildRequest):
    try:
        from build_index_v2 import default_catalog_path

        start = time.perf_counter()
        input_path = payload.input_path.strip() or default_catalog_path()
        scene_index_report, engine, error = rebuild_all_indexes_safely(payload, input_path)
        return {
            "ok": error is None,
            "elapsed_sec": round(time.perf_counter() - start, 3),
            "input_path": input_path,
            "engine": engine.status() if engine else None,
            "error": error,
            "runtime": runtime_status(),
            "crawler": crawler_status_payload(),
            "scene_index": scene_index_report,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def run_index_rebuild_job(job_id: str, payload: IndexBuildRequest):
    start = time.perf_counter()
    update_job(job_id, status="running")
    try:
        from build_index_v2 import default_catalog_path

        input_path = payload.input_path.strip() or default_catalog_path()
        update_job(job_id, stage="movie_index", progress=4)
        scene_index_report, engine, error = rebuild_all_indexes_safely(
            payload,
            input_path,
            before_scene=lambda: update_job(job_id, stage="scene_index", progress=92),
        )
        result = {
            "ok": error is None,
            "elapsed_sec": round(time.perf_counter() - start, 3),
            "input_path": input_path,
            "engine": engine.status() if engine else None,
            "error": error,
            "runtime": runtime_status(),
            "crawler": crawler_status_payload(),
            "scene_index": scene_index_report,
        }
        update_job(job_id, status="completed" if error is None else "failed", result=result, error=error)
    except Exception as exc:
        update_job(
            job_id,
            status="failed",
            error=str(exc),
            traceback=traceback.format_exc(limit=6),
        )


@app.post("/api/index/rebuild-async")
def rebuild_index_async(payload: IndexBuildRequest):
    job = create_job("index_rebuild", "Rebuild FAISS search index")
    update_job(job["id"], status="running")
    thread = threading.Thread(target=run_index_rebuild_job, args=(job["id"], payload), daemon=True)
    thread.start()
    return {"ok": True, "job": get_job(job["id"])}


if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")

try:
    from ingestion.offline_video import KEYFRAME_DIR

    KEYFRAME_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/media/keyframes", StaticFiles(directory=KEYFRAME_DIR), name="keyframes")
except Exception:
    pass

VIDEO_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/media/videos", StaticFiles(directory=VIDEO_DIR), name="videos")


@app.get("/")
def index():
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend not built")
    return FileResponse(index_file)
