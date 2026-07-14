from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import faiss
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_index_v2 import IndexBuilderV2, movie_title


def write_index_atomic(index: faiss.Index, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    faiss.write_index(index, str(temporary))
    temporary.replace(path)


def write_pickle_atomic(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        pickle.dump(value, handle)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description="Append new catalog documents to the active CineScene movie index")
    parser.add_argument("--input", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    report_path = ROOT / "data" / "index" / "index_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    index_path = ROOT / str(report.get("index_path") or "data/index/faiss_index_v2.bin")
    metadata_path = ROOT / str(report.get("metadata_path") or "data/index/metadata_v2.pkl")
    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Build the main movie index before appending documents")

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    documents = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(documents, list):
        raise ValueError("Input catalog must be a JSON list")

    with open(metadata_path, "rb") as handle:
        metadata = pickle.load(handle)
    known_ids = {str(item.get("id")) for item in metadata}
    additions = [item for item in documents if str(item.get("id")) not in known_ids and movie_title(item) != "Unknown"]
    if not additions:
        print(json.dumps({"appended": 0, "index_vectors": len(metadata)}, indent=2))
        return

    builder = IndexBuilderV2(
        model_path=str(report.get("model") or "models/bge-large-en-v1.5"),
        batch_size=args.batch_size,
        max_seq_length=int(report.get("max_seq_length") or 256),
        fp16=str(report.get("precision")) == "float16",
    )
    started = time.perf_counter()
    embeddings = builder.build_embeddings(additions)
    index = faiss.read_index(str(index_path))
    if int(index.d) != int(embeddings.shape[1]):
        raise ValueError(f"Index dimension {index.d} does not match embedding dimension {embeddings.shape[1]}")
    index.add(embeddings)
    metadata.extend(builder.prepare_metadata(additions))

    write_index_atomic(index, index_path)
    write_index_atomic(index, index_path.parent / "movies.index")
    write_pickle_atomic(metadata, metadata_path)
    write_pickle_atomic(metadata, metadata_path.parent / "metadata.pkl")

    report["movies_indexed"] = len(metadata)
    report["incremental_updates"] = int(report.get("incremental_updates") or 0) + 1
    report["last_incremental_source"] = str(input_path.relative_to(ROOT))
    report["last_incremental_count"] = len(additions)
    report["updated_at"] = datetime.now(timezone.utc).isoformat()
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if hasattr(builder, "model"):
        del builder.model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "appended": len(additions),
                "index_vectors": int(index.ntotal),
                "elapsed_sec": round(time.perf_counter() - started, 3),
                "model": report.get("model"),
                "dimension": int(index.d),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
