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

    crawler = client.get("/api/crawl/status").json()
    assert_true(crawler.get("ok") is True, "crawler status failed")
    assert_true(crawler.get("offline_documents", 0) >= 1, "no offline documents indexed")
    assert_true(crawler.get("keyframes", 0) >= 1, "no crawler keyframes found")

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

    payload = {
        "ok": True,
        "elapsed_sec": round(time.perf_counter() - start, 3),
        "movies": health.get("engine", {}).get("movies"),
        "index_vectors": health.get("engine", {}).get("index_vectors"),
        "offline_documents": crawler.get("offline_documents"),
        "keyframes": crawler.get("keyframes"),
        "top_result": {
            "title": top.get("title"),
            "source": top.get("source"),
            "score": top.get("score"),
        },
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
