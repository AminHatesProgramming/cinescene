# CineScene

Semantic movie, series, and offline scene retrieval.

**Live PWA:** https://aminhatesprogramming.github.io/cinescene

CineScene lets a user describe a plot, scene, mood, or cinematic feeling and get relevant titles back. The full local system uses FastAPI, FAISS, enriched TMDB data, offline video scene ingestion, search memory, favorites, feedback, and benchmark tooling. The GitHub Pages build is a polished static PWA showcase with an offline-ready sample catalog.

## Highlights

- Semantic movie and scene search
- Offline local video crawler for movies and series
- Scene boundary detection, keyframe extraction, subtitle ingestion, visual signal tags, mood/keyword tagging
- FastAPI backend with SQLite-backed memory
- FAISS index rebuild from inside the app
- Background jobs for long crawler/index operations
- Installable GitHub Pages PWA
- Final smoke test and full benchmark scripts

## Demo Modes

### GitHub Pages PWA

The public PWA source lives in `docs/`. The live site is served from the `gh-pages` branch.

```text
https://aminhatesprogramming.github.io/cinescene
```

It runs fully in the browser using `docs/data/catalog.sample.json`, supports install/offline caching, and demonstrates the offline crawler result with real keyframes and a scene timeline.

### Full Local System

```powershell
.\.venv\Scripts\python.exe app.py
```

Open:

```text
http://127.0.0.1:8000
```

Use the `Crawler` tab to crawl local videos, inspect scene/keyframe output, rebuild the FAISS index, and search the extracted scene text.
Use `Probe` before crawling to verify that the folder contains supported media and sidecar subtitles.

## Offline Video Pipeline

```text
local video folder
  -> recursive crawler
  -> scene boundary detection
  -> keyframes
  -> sidecar SRT/VTT transcript extraction
  -> visual signals, mood tags, and keywords
  -> offline_media_enriched.json
  -> cinescene_catalog.json
  -> FAISS embeddings
  -> semantic search
```

Series filenames such as `Show.Name.S01E01.mp4` are parsed into season and episode metadata.

The crawler processes local/offline media files that the user has permission to analyze. It does not bypass protected streaming services.

## Run Order

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Prepare data:

```powershell
python preprocess_tmdb.py
python enrich_dataset.py
python build_triplets_v2.py
```

Fine-tune on a CUDA machine:

```powershell
python finetune_v2.py --batch-size 4 --epochs 3
```

Build the index:

```powershell
python build_index_v2.py --model model --flat --batch-size 64
```

Run the app:

```powershell
python app.py
```

## Crawler CLI

```powershell
.\.venv\Scripts\python.exe .\scripts\crawl_offline_videos.py "D:\Series" --title-prefix "My Library" --min-scene-sec 6 --max-scene-sec 90 --threshold 0.4 --sample-fps 1
```

Generated outputs:

```text
data/processed/video_ingestion/*_scenes.json
data/processed/video_ingestion/keyframes/*.jpg
data/processed/offline_media_enriched.json
data/processed/cinescene_catalog.json
```

## Validation

Fast final smoke test:

```powershell
.\.venv\Scripts\python.exe .\scripts\smoke_test_final.py
```

Full benchmark:

```powershell
.\.venv\Scripts\python.exe .\scripts\benchmark_cinescene.py
```

Latest verified benchmark snapshot:

```text
health_ok=True
movies_indexed=4803
index_vectors=4803
search_avg=0.607s
crawler_videos=1
crawler_documents=1
crawler_scene_count=3
crawler_keyframes=3
crawler_transcript_found=True
rebuild_ok=True
rebuild_time=218.748s
offline_recall_at_1=3/3
```

## Important Files

- `backend/main.py`: FastAPI API, search, memory, crawler jobs, index rebuild jobs
- `frontend/`: full local backend-connected UI
- `docs/`: GitHub Pages PWA
- `hybrid_search.py`: FAISS + lexical hybrid search
- `ingestion/offline_video.py`: local video scene ingestion
- `build_index_v2.py`: FAISS index builder
- `scripts/smoke_test_final.py`: quick final readiness test
- `scripts/benchmark_cinescene.py`: full benchmark
- `scripts/build_pwa_catalog.py`: refresh the GitHub Pages sample catalog from the local catalog
- `scripts/publish_github_pages.ps1`: create/push the GitHub repository when `GH_TOKEN` is available
- `.github/workflows/pages.yml`: static PWA validation

## Publish To GitHub Pages

Create a GitHub Personal Access Token with `repo` permission, then run:

```powershell
$env:GH_TOKEN = "YOUR_TOKEN_HERE"
.\scripts\publish_github_pages.ps1
```

The script creates/pushes the main repository and publishes `docs/` to the `gh-pages` branch:

```text
https://github.com/aminhatesprogramming/cinescene
```

GitHub Pages then serves:

```text
https://aminhatesprogramming.github.io/cinescene
```

## Future Upgrades

- Whisper for speech-to-text when subtitles are missing
- BLIP/vision-language captioning for object-level scene descriptions
- Larger series catalog ingestion
- GPU fine-tuning and indexing for faster experiments
