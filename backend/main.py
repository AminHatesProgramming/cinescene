from __future__ import annotations

import os
import json
import tempfile
import threading
import time
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
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

app = FastAPI(
    title="CineScene API",
    description="Semantic movie retrieval with offline video scene ingestion.",
    version="2.0.0",
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


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(8, ge=1, le=20)
    use_reranking: bool = True


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
    threshold: float = Field(0.45, ge=0.05, le=1.0)
    sample_fps: float = Field(1.0, ge=0.1, le=5.0)
    update_catalog: bool = True


class IndexBuildRequest(BaseModel):
    input_path: str = ""
    model_path: str = "model"
    batch_size: int = Field(64, ge=1, le=256)
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


@lru_cache(maxsize=1)
def get_search_engine():
    try:
        from hybrid_search import HybridSearchEngine

        enable_reranker = os.getenv("CINESCENE_ENABLE_RERANKER", "0") == "1"
        return HybridSearchEngine(enable_reranker=enable_reranker), None
    except Exception as exc:
        return None, str(exc)


def read_json(path: Path, default):
    try:
        if not path.exists():
            return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


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
    return {
        "cuda": cuda,
        "index_report": index_report,
        "offline_documents": len(offline_catalog) if isinstance(offline_catalog, list) else 0,
        "combined_documents": len(combined_catalog) if isinstance(combined_catalog, list) else 0,
        "paths": {
            "index_report": str(INDEX_REPORT_PATH),
            "offline_catalog": str(OFFLINE_CATALOG_PATH),
            "combined_catalog": str(COMBINED_CATALOG_PATH),
        },
    }


def crawler_status_payload():
    from ingestion.offline_video import INGESTION_DIR, KEYFRAME_DIR, VIDEO_EXTENSIONS

    offline_catalog = read_json(OFFLINE_CATALOG_PATH, [])
    combined_catalog = read_json(COMBINED_CATALOG_PATH, [])
    report_path = INGESTION_DIR / "offline_crawl_report.json"
    report_rows = read_json(report_path, [])
    latest_report = report_rows[-1] if isinstance(report_rows, list) and report_rows else {}

    scene_files = sorted(INGESTION_DIR.glob("*_scenes.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    scene_previews = []
    scene_total = 0
    for scene_file in scene_files[:12]:
        scenes = read_json(scene_file, [])
        if not isinstance(scenes, list):
            continue
        scene_total += len(scenes)
        first = scenes[0] if scenes else {}
        scene_previews.append(
            {
                "file": str(scene_file),
                "scene_count": len(scenes),
                "title": first.get("movie_title") or first.get("title") or scene_file.stem,
                "media_type": first.get("media_type"),
                "season": first.get("season"),
                "episode": first.get("episode"),
                "source_video": first.get("source_video", ""),
                "first_scene": first.get("rich_text", "")[:600],
                "keyframe_path": first.get("keyframe_path"),
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

    return {
        "ok": True,
        "video_extensions": sorted(VIDEO_EXTENSIONS),
        "offline_documents": offline_count,
        "offline_series_documents": series_count,
        "offline_movie_documents": movie_count,
        "combined_documents": combined_count,
        "scene_files": len(scene_files),
        "scene_total_in_preview_files": scene_total,
        "keyframes": keyframe_count,
        "latest_report": latest_report,
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
def crawl_status():
    return crawler_status_payload()


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    return get_job(job_id)


@app.post("/api/reload")
def reload_engine():
    get_search_engine.cache_clear()
    engine, error = get_search_engine()
    return {
        "ok": error is None,
        "engine": engine.status() if engine else None,
        "error": error,
        "runtime": runtime_status(),
    }


@app.post("/api/search")
def search(payload: SearchRequest, sid: str = Header(default="anonymous", alias="X-CineScene-Session")):
    engine, error = get_search_engine()
    if error or engine is None:
        raise HTTPException(status_code=503, detail=f"Search engine is not ready: {error}")

    results = engine.search(payload.query, top_k=payload.top_k, use_reranking=payload.use_reranking)
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


@app.post("/api/ingest/video")
async def ingest_video(
    file: UploadFile = File(...),
    movie_title: str = "",
    sid: str = Header(default="anonymous", alias="X-CineScene-Session"),
):
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        from ingestion.offline_video import detect_scenes

        scenes = detect_scenes(str(tmp_path), movie_title=movie_title or Path(file.filename or "offline_video").stem)
        scene_payload = [scene.__dict__ | {"rich_text": scene.rich_text} for scene in scenes]
        memory.add_ingestion(sid, movie_title or Path(file.filename or "offline_video").stem, scenes[0].source_video if scenes else "", scene_payload)
        return {"ok": True, "scene_count": len(scene_payload), "scenes": scene_payload}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


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
            threshold=payload.threshold,
            sample_fps=payload.sample_fps,
            update_catalog=payload.update_catalog,
        )
        for job in report.get("jobs", []):
            memory.add_ingestion(
                sid,
                job.get("title") or Path(job.get("video", "")).stem,
                job.get("video", ""),
                [{"scene_count": job.get("scene_count"), "document_id": job.get("document_id")}],
        )
        return {
            "ok": True,
            "report": report,
            "next_steps": [
                "Rebuild the FAISS index with build_index_v2.py so crawled videos become searchable.",
                "Call /api/reload or restart the app after rebuilding the index.",
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def run_crawl_job(job_id: str, payload: OfflineCrawlRequest, sid: str):
    start = time.perf_counter()
    update_job(job_id, status="running")
    try:
        from ingestion.offline_video import crawl_offline_videos

        report = crawl_offline_videos(
            payload.root,
            title_prefix=payload.title_prefix,
            min_scene_sec=payload.min_scene_sec,
            threshold=payload.threshold,
            sample_fps=payload.sample_fps,
            update_catalog=payload.update_catalog,
        )
        for job in report.get("jobs", []):
            memory.add_ingestion(
                sid,
                job.get("title") or Path(job.get("video", "")).stem,
                job.get("video", ""),
                [{"scene_count": job.get("scene_count"), "document_id": job.get("document_id")}],
            )
        update_job(
            job_id,
            status="completed",
            elapsed_sec=round(time.perf_counter() - start, 3),
            result={
                "ok": True,
                "report": report,
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
        from build_index_v2 import IndexBuilderV2, default_catalog_path

        start = time.perf_counter()
        input_path = payload.input_path.strip() or default_catalog_path()
        builder = IndexBuilderV2(
            model_path=payload.model_path,
            use_base_model=payload.use_base_model,
            batch_size=payload.batch_size,
        )
        builder.build_and_save(
            enriched_path=input_path,
            use_hnsw=payload.use_hnsw,
        )
        get_search_engine.cache_clear()
        engine, error = get_search_engine()
        return {
            "ok": error is None,
            "elapsed_sec": round(time.perf_counter() - start, 3),
            "input_path": input_path,
            "engine": engine.status() if engine else None,
            "error": error,
            "runtime": runtime_status(),
            "crawler": crawler_status_payload(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def run_index_rebuild_job(job_id: str, payload: IndexBuildRequest):
    start = time.perf_counter()
    update_job(job_id, status="running")
    try:
        from build_index_v2 import IndexBuilderV2, default_catalog_path

        input_path = payload.input_path.strip() or default_catalog_path()
        builder = IndexBuilderV2(
            model_path=payload.model_path,
            use_base_model=payload.use_base_model,
            batch_size=payload.batch_size,
        )
        builder.build_and_save(
            enriched_path=input_path,
            use_hnsw=payload.use_hnsw,
        )
        get_search_engine.cache_clear()
        engine, error = get_search_engine()
        result = {
            "ok": error is None,
            "elapsed_sec": round(time.perf_counter() - start, 3),
            "input_path": input_path,
            "engine": engine.status() if engine else None,
            "error": error,
            "runtime": runtime_status(),
            "crawler": crawler_status_payload(),
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


@app.get("/")
def index():
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend not built")
    return FileResponse(index_file)
