# CineScene Architecture

## System Overview

```text
TMDB / enriched plot data
        |
        v
movies_enriched.json
        |
        +------------------------------+
        |                              |
        v                              v
triplet generation              offline video crawler
        |                              |
        v                              v
embedding fine-tuning        scene JSON + keyframes + transcript + visual signals
        |                              |
        +--------------+---------------+
                       |
                       v
              cinescene_catalog.json
                       |
                       v
                FAISS vector index
                       |
                       v
       FastAPI backend + memory + feedback
                       |
                       v
              frontend search app
```

## Runtime Components

- `backend/main.py`
  - API health and runtime status
  - semantic search
  - history, favorites, and feedback
  - async offline crawl jobs
  - async FAISS rebuild jobs

- `hybrid_search.py`
  - sentence-transformer embeddings
  - FAISS vector retrieval
  - BM25 lexical retrieval
  - direct overlap search
  - source-aware boost for offline scene matches

- `ingestion/offline_video.py`
  - recursive local folder crawling
  - scene boundary detection with OpenCV
  - keyframe extraction
  - sidecar subtitle parsing
  - per-scene brightness/motion/contrast/cut signals
  - lightweight mood/keyword/visual tag inference
  - movie/series metadata parsing

- `docs/`
  - static GitHub Pages PWA
  - installable app shell
  - service worker offline cache
  - compact browser-side sample catalog

## Deployment Strategy

GitHub Pages can only host static files, so the public site is a polished PWA showcase generated from `docs/` and served from the `gh-pages` branch.

The full production-style local system is still available through FastAPI:

```powershell
.\.venv\Scripts\python.exe app.py
```

This split keeps the public demo fast and easy to share while preserving the complete machine-learning pipeline locally.
