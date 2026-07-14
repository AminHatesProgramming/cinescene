from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.main import app


def time_call(label, fn):
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    print(f"{label}: {elapsed:.3f}s")
    return result, elapsed


def create_synthetic_video(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (192, 128))
    colors = [(18, 18, 18), (230, 230, 230), (40, 90, 190)]
    for color in colors:
        for _ in range(35):
            frame = np.zeros((128, 192, 3), dtype=np.uint8)
            frame[:] = color
            writer.write(frame)
    writer.release()


def create_subtitle(path: Path):
    path.write_text(
        "1\n00:00:00,000 --> 00:00:03,000\nA lonely detective enters a dark room.\n\n"
        "2\n00:00:03,000 --> 00:00:06,000\nThe city lights flicker while he investigates the mystery.\n",
        encoding="utf-8",
    )


def main():
    client = TestClient(app)
    session_id = "benchmark-session"

    print("CineScene benchmark")
    print("=" * 64)

    health_response, health_time = time_call("health", lambda: client.get("/api/health"))
    health = health_response.json()
    print(json.dumps(health, indent=2, ensure_ascii=False))

    queries = [
        "dark lonely science fiction movie",
        "romantic comedy in New York with witty dialogue",
        "detective thriller in a rainy city",
        "space adventure with alien civilization",
        "sad drama about family and loss",
    ]

    print("\nSearch")
    print("-" * 64)
    search_times = []
    latest_results = []
    for query in queries:
        response, elapsed = time_call(
            f"query | {query}",
            lambda q=query: client.post(
                "/api/search",
                json={"query": q, "top_k": 5, "use_reranking": False},
                headers={"X-CineScene-Session": session_id},
            ),
        )
        search_times.append(elapsed)
        payload = response.json()
        latest_results = payload.get("results", [])
        titles = [item["title"] for item in latest_results[:3]]
        print(f"status={response.status_code} count={payload.get('count')} top3={titles}")

    print("\nMemory")
    print("-" * 64)
    history = client.get("/api/history", headers={"X-CineScene-Session": session_id}).json()
    print(f"history_items={len(history['items'])}")

    favorite_count = 0
    if latest_results:
        fav_response = client.post(
            "/api/favorites",
            json={"movie": latest_results[0]},
            headers={"X-CineScene-Session": session_id},
        )
        favorites = client.get("/api/favorites", headers={"X-CineScene-Session": session_id}).json()
        favorite_count = len(favorites["items"])
        print(f"favorite_status={fav_response.status_code} favorites={favorite_count}")

    print("\nOffline crawler / video ingestion")
    print("-" * 64)
    crawl_root = ROOT / "data" / "offline_videos" / "benchmark_folder"
    source_video = crawl_root / "Benchmark.Show.S01E01.mp4"
    create_synthetic_video(source_video)
    create_subtitle(source_video.with_suffix(".srt"))

    crawl_response, ingestion_time = time_call(
        "crawl_offline_folder",
        lambda: client.post(
            "/api/crawl/offline",
            json={
                "root": str(crawl_root),
                "title_prefix": "Benchmark Show",
                "min_scene_sec": 2,
                "threshold": 0.2,
                "sample_fps": 2,
                "update_catalog": True,
            },
            headers={"X-CineScene-Session": session_id},
        ),
    )
    ingestion_payload = crawl_response.json()
    report = ingestion_payload.get("report", {})
    print(f"status={crawl_response.status_code} videos={report.get('videos_processed')} docs={report.get('documents_created')}")

    scene_files = sorted((ROOT / "data" / "processed" / "video_ingestion").glob("*_scenes.json"))
    latest_scene_file = None
    scenes = []
    for scene_file in scene_files:
        payload = json.loads(scene_file.read_text(encoding="utf-8"))
        if any("Benchmark Show" in scene.get("rich_text", "") or "Benchmark" in scene.get("movie_title", "") for scene in payload):
            latest_scene_file = scene_file
            scenes = payload
    keyframes = [scene.get("keyframe_path") for scene in scenes if scene.get("keyframe_path") and Path(scene["keyframe_path"]).exists()]
    transcript_found = any("detective" in scene.get("rich_text", "").lower() for scene in scenes)
    print(f"scene_file={latest_scene_file}")
    print(f"scene_count={len(scenes)} keyframes_found={len(keyframes)} transcript_found={transcript_found}")
    if scenes:
        print(f"first_scene={scenes[0].get('rich_text', '')[:240]}")

    ingestion_history = client.get("/api/ingestions", headers={"X-CineScene-Session": session_id}).json()
    print(f"ingestion_history_items={len(ingestion_history['items'])}")

    print("\nIndex rebuild after crawler")
    print("-" * 64)
    rebuild_response, rebuild_time = time_call(
        "rebuild_index",
        lambda: client.post(
            "/api/index/rebuild",
            json={
                "input_path": "",
                "model_path": "model",
                "batch_size": 64,
                "use_hnsw": False,
                "use_base_model": False,
            },
            headers={"X-CineScene-Session": session_id},
        ),
    )
    rebuild_payload = rebuild_response.json()
    print(
        f"status={rebuild_response.status_code} ok={rebuild_payload.get('ok')} "
        f"elapsed={rebuild_payload.get('elapsed_sec')}s"
    )

    print("\nOffline retrieval check")
    print("-" * 64)
    offline_queries = [
        "lonely detective enters a dark room",
        "city lights flicker while he investigates the mystery",
        "suspenseful detective mystery in a dark room",
    ]
    offline_hits = 0
    for query in offline_queries:
        response, elapsed = time_call(
            f"offline query | {query}",
            lambda q=query: client.post(
                "/api/search",
                json={"query": q, "top_k": 5, "use_reranking": False},
                headers={"X-CineScene-Session": session_id},
            ),
        )
        payload = response.json()
        results = payload.get("results", [])
        top = results[0] if results else {}
        is_hit = top.get("source") == "offline_video_ingestion" and top.get("title") == "Benchmark Show"
        offline_hits += int(is_hit)
        print(
            f"status={response.status_code} top={top.get('title')} "
            f"source={top.get('source')} hit={is_hit} time={elapsed:.3f}s"
        )

    print("\nSummary")
    print("-" * 64)
    print(f"health_ok={health.get('ok')}")
    print(f"movies_indexed={health.get('engine', {}).get('movies')}")
    print(f"index_vectors={health.get('engine', {}).get('index_vectors')}")
    print(f"search_avg={statistics.mean(search_times):.3f}s")
    print(f"search_min={min(search_times):.3f}s")
    print(f"search_max={max(search_times):.3f}s")
    print(f"history_items={len(history['items'])}")
    print(f"favorites={favorite_count}")
    print(f"crawler_videos={report.get('videos_processed')}")
    print(f"crawler_documents={report.get('documents_created')}")
    print(f"crawler_scene_count={len(scenes)}")
    print(f"crawler_keyframes={len(keyframes)}")
    print(f"crawler_transcript_found={transcript_found}")
    print(f"crawler_time={ingestion_time:.3f}s")
    print(f"rebuild_ok={rebuild_payload.get('ok')}")
    print(f"rebuild_time={rebuild_time:.3f}s")
    print(f"offline_recall_at_1={offline_hits}/{len(offline_queries)}")


if __name__ == "__main__":
    main()
