from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hybrid_search import HybridSearchEngine


CASES = [
    {"query": "a thief enters layered dreams to plant an idea", "expected": ["Inception"]},
    {"query": "a marine meets blue aliens on the moon Pandora", "expected": ["Avatar"]},
    {"query": "an amnesiac man uses tattoos and notes to find a killer", "expected": ["Memento"]},
    {"query": "toys come alive when their owner leaves the room", "expected": ["Toy Story"]},
    {"query": "a giant shark terrorizes a seaside town", "expected": ["Jaws"]},
    {"query": "a computer hacker discovers reality is a simulation", "expected": ["The Matrix"]},
    {"query": "a romance aboard a ship that hits an iceberg", "expected": ["Titanic"]},
    {"query": "a hobbit carries a powerful ring toward a volcano", "expected": ["The Lord of the Rings", "The Fellowship of the Ring"]},
    {"query": "lonely detective enters a dark room and investigates a mystery", "expected": ["Benchmark Show"]},
    {"query": "کارآگاهی تنها وارد اتاق تاریک می‌شود و دنبال سرنخ می‌گردد", "expected": ["Benchmark Show"]},
]


def matches(title: str, expected: list[str]) -> bool:
    normalized = title.casefold()
    return any(value.casefold() in normalized or normalized in value.casefold() for value in expected)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark CineScene scene-to-title retrieval")
    parser.add_argument("--output", default="benchmarks/latest.json")
    parser.add_argument("--reranker", action="store_true", help="Enable the cross-encoder reranker")
    return parser.parse_args()


def main():
    args = parse_args()
    engine = HybridSearchEngine(enable_reranker=args.reranker)
    rows = []
    latencies = []
    for case in CASES:
        started = time.perf_counter()
        results = engine.search(case["query"], top_k=10, use_reranking=args.reranker)
        latency = time.perf_counter() - started
        latencies.append(latency)
        rank = next((index for index, item in enumerate(results, start=1) if matches(item["title"], case["expected"])), None)
        rows.append(
            {
                "query": case["query"],
                "expected": case["expected"],
                "rank": rank,
                "top_3": [item["title"] for item in results[:3]],
                "latency_sec": round(latency, 3),
            }
        )

    ranked = [row["rank"] for row in rows if row["rank"]]
    status = engine.status()
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": engine.model_path,
        "device": status.get("device"),
        "precision": status.get("precision"),
        "embedding_dimension": int(engine.index.d),
        "reranker": status.get("reranker", False),
        "cases": len(rows),
        "hit_at_1": round(sum(1 for rank in ranked if rank == 1) / len(rows), 4),
        "hit_at_5": round(sum(1 for rank in ranked if rank <= 5) / len(rows), 4),
        "mrr_at_10": round(sum(1 / rank for rank in ranked if rank <= 10) / len(rows), 4),
        "mean_latency_sec": round(statistics.mean(latencies), 4),
        "movie_vectors": status.get("movie_vectors"),
        "scene_vectors": status.get("scene_vectors"),
        "index_vectors": status.get("index_vectors"),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved benchmark report to {output}")


if __name__ == "__main__":
    main()
