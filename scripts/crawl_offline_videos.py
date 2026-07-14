from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion.offline_video import crawl_offline_videos


def main():
    parser = argparse.ArgumentParser(description="Crawl local movie/series videos and build CineScene offline records")
    parser.add_argument("root", help="Video file or folder to crawl recursively")
    parser.add_argument("--title-prefix", default="", help="Optional title override/prefix")
    parser.add_argument("--min-scene-sec", type=float, default=8.0)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--no-catalog", action="store_true", help="Do not update offline/combined catalogs")
    args = parser.parse_args()

    report = crawl_offline_videos(
        args.root,
        title_prefix=args.title_prefix,
        min_scene_sec=args.min_scene_sec,
        threshold=args.threshold,
        sample_fps=args.sample_fps,
        update_catalog=not args.no_catalog,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
