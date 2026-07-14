# CineScene Final Demo Guide

## 0. Public PWA

Open the GitHub Pages showcase:

```text
https://aminhatesprogramming.github.io/cinescene
```

Show that it is installable, cached as a PWA, and can search the sample catalog
without the Python backend.

## 1. Start The App

```powershell
.\.venv\Scripts\python.exe app.py
```

Open:

```text
http://127.0.0.1:8000
```

Use `Ctrl + F5` in the browser after code changes.

## 2. What To Show First

1. The top metrics bar:
   - indexed catalog size
   - offline documents
   - active embedding model
   - CUDA/CPU status

2. Search examples:
   - `dark lonely science fiction movie`
   - `detective thriller in a rainy city`
   - `space adventure with alien civilization`

3. Memory:
   - search history
   - favorites
   - relevance feedback

## 3. Offline Video Crawler Demo

Open the `Crawler` tab in the app. It now contains the full workflow:

- crawler health counters: offline docs, scene count, keyframes, OpenCV capability
- path probing before a crawl
- single video scene detection
- recursive folder crawling for movies and series
- tunable scene parameters including min/max scene duration
- crawler output preview with scene timeline, transcript, visual tags, and keyframes
- background FAISS index rebuild with visible job status

Put local videos in a folder. Series filenames can use this pattern:

```text
Show.Name.S01E01.mp4
Show.Name.S01E01.srt
```

Use the in-app `Offline folder path` field. For the included sample, click
`Benchmark`, then `Probe`, then `Crawl Folder`.

CLI equivalent:

```powershell
.\.venv\Scripts\python.exe .\scripts\crawl_offline_videos.py "D:\Series" --min-scene-sec 6 --max-scene-sec 90 --threshold 0.4
```

Crawler output should create:

```text
data/processed/video_ingestion/*_scenes.json
data/processed/video_ingestion/keyframes/*.jpg
data/processed/offline_media_enriched.json
data/processed/cinescene_catalog.json
```

Then click `Rebuild Search Index` in the app. The job panel should show progress
while the backend rebuilds FAISS. CLI equivalent:

```powershell
.\.venv\Scripts\python.exe .\build_index_v2.py --model model --flat --batch-size 64
```

After rebuild, search for text from the subtitle or scene. A healthy offline
demo query is:

```text
suspenseful detective mystery in a dark room
```

The expected top result for the included benchmark media is `Benchmark Show`
from `offline_video_ingestion`.

## 4. Benchmark

Fast smoke test:

```powershell
.\.venv\Scripts\python.exe .\scripts\smoke_test_final.py
```

Full benchmark:

```powershell
.\.venv\Scripts\python.exe .\scripts\benchmark_cinescene.py
```

Healthy output should include:

```text
health_ok=True
search_avg=0.607s
crawler_videos=1
crawler_documents=1
crawler_scene_count=3
crawler_keyframes=3
crawler_transcript_found=True
rebuild_time=218.748s
offline_recall_at_1=3/3
```

## 5. Current Limits To State Honestly

- The crawler processes local/offline files the user has permission to analyze.
- Scene descriptions include subtitle text plus brightness, motion, contrast, cut, mood, keyword, and visual-tag signals.
- For stronger scene understanding, add a vision-language captioner such as BLIP.
- For dialogue without subtitles, add Whisper speech-to-text.
- For best retrieval quality, run the BGE fine-tuning pipeline on a working CUDA setup.
