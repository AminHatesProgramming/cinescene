from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from backend.main import app


def assert_true(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def main():
    client = TestClient(app)

    start = time.perf_counter()
    health = client.get("/api/health").json()
    assert_true(health.get("ok") is True, f"health failed: {health.get('error')}")
    engine = health.get("engine", {})
    assert_true(engine.get("movie_vectors", 0) >= 1, "movie index is empty")
    assert_true(engine.get("scene_vectors", 0) >= 1, "scene index is empty")

    crawler = client.get("/api/crawl/status").json()
    assert_true(crawler.get("ok") is True, "crawler status failed")
    assert_true(crawler.get("crawler_enabled") is True, "crawler is not enabled")
    assert_true(crawler.get("offline_documents", 0) >= 1, "no offline documents indexed")
    assert_true(crawler.get("keyframes", 0) >= 1, "no crawler keyframes found")
    assert_true(crawler.get("scene_total", 0) >= 1, "no crawler scenes found")
    capabilities = crawler.get("capabilities", {})
    assert_true(capabilities.get("ffmpeg") is True, "FFmpeg capability is unavailable")
    assert_true(capabilities.get("embedded_subtitles") is True, "embedded subtitle extraction is unavailable")
    assert_true("Detector" in str(capabilities.get("scene_detector")), "adaptive scene detector is unavailable")

    search = client.post(
        "/api/search",
        json={
            "query": "suspenseful detective mystery in a dark room",
            "top_k": 5,
            "use_reranking": False,
        },
        headers={"X-CineScene-Session": "final-smoke-test"},
    ).json()
    results = search.get("results", [])
    assert_true(results, "search returned no results")
    top = results[0]
    assert_true(top.get("source") == "offline_video_ingestion", f"offline result was not rank 1: {top}")
    assert_true(top.get("title") == "Benchmark Show", f"unexpected English scene match: {top.get('title')}")
    matched_scene = top.get("matched_scene") or {}
    assert_true(matched_scene.get("transcript"), "top result has no matched scene transcript")
    assert_true(matched_scene.get("end_sec") is not None, "top result has no matched scene timestamp")

    persian = client.post(
        "/api/search",
        json={
            "query": "کارآگاهی تنها وارد اتاقی تاریک می شود و دنبال سرنخ می گردد",
            "top_k": 5,
            "use_reranking": False,
        },
        headers={"X-CineScene-Session": "final-smoke-test"},
    ).json()
    persian_results = persian.get("results", [])
    assert_true(persian_results, "Persian search returned no results")
    assert_true(persian_results[0].get("title") == "Benchmark Show", "Persian scene retrieval missed rank 1")

    payload = {
        "ok": True,
        "elapsed_sec": round(time.perf_counter() - start, 3),
        "movies": engine.get("movies"),
        "movie_vectors": engine.get("movie_vectors"),
        "scene_vectors": engine.get("scene_vectors"),
        "index_vectors": engine.get("index_vectors"),
        "offline_documents": crawler.get("offline_documents"),
        "keyframes": crawler.get("keyframes"),
        "scene_total": crawler.get("scene_total"),
        "crawler_enabled": crawler.get("crawler_enabled"),
        "top_result": {
            "title": top.get("title"),
            "source": top.get("source"),
            "score": top.get("score"),
            "scene_count": top.get("scene_count"),
            "matched_scene": matched_scene.get("scene_number"),
        },
        "persian_top_result": persian_results[0].get("title"),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
