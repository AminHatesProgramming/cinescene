# CineScene Final Demo Guide

This walkthrough demonstrates the complete user story in a short, defensible order: **upload a video, turn it into scene memory, then retrieve its title from a remembered scene.**

## 1. Public First Impression

Open:

```text
https://aminhatesprogramming.github.io/cinescene
```

Show the responsive Retrieval Studio, installable PWA, scene cards, browser memory, and processed scene timeline. Mention that this public build is static by design; the same interface connects to Python, FAISS, and GPU models in the full runtime.

## 2. Start The Full Runtime

```powershell
.\.venv\Scripts\python.exe app.py
```

Open `http://127.0.0.1:8000` and hard-refresh once with `Ctrl+F5` if an older cached stylesheet is visible.

The sidebar status should become **Engine online**. The Discover metrics should report:

- 7,615 catalog/movie vectors;
- 158 scene vectors and 7,773 total vectors;
- 3 currently ingested offline media documents;
- NVIDIA GeForce GTX 1650 Ti through CUDA.

Open **System** and briefly show that scene detection, FFmpeg subtitles, faster-whisper, BLIP vision, the movie index, and the scene index are independently observable.

## 3. Prove Scene Retrieval

Run these searches in **Discover**:

```text
a lonely detective enters a dark room and investigates a mystery
```

```text
کارآگاهی تنها وارد اتاقی تاریک می شود و دنبال سرنخ می گردد
```

For the included synthetic benchmark, `Benchmark Show` should rank first. Point out the matched keyframe, transcript, timestamp, source label, and retrieval score. This proves that the result is coming from a scene vector rather than only from the whole-film plot.

Then try catalog-oriented queries:

```text
a dream collapses while people run through a rotating hallway
```

```text
a shark terrorizes a seaside town
```

## 4. Upload A Video In The App

Open **Scene Lab**.

1. Drag an MP4/MKV/MOV/AVI/WEBM file into **Media source**.
2. Add SRT/VTT tracks when available.
3. Enter a clean movie or episode title.
4. Keep **Embedded subtitles** enabled.
5. Enable **Speech transcription** only when subtitles are unavailable.
6. Enable **Visual captioning** for the strongest scene descriptions.
7. Choose **Analyze and index video**.

The live pipeline should advance through:

```text
Upload
  -> Scene boundaries
  -> Scene intelligence
  -> Vector index
  -> Search ready
```

When complete, Scene Lab displays the generated keyframes and timeline. Return to Discover and search for a sentence from the subtitle or a visual action from one scene. No manual full-index rebuild is needed after upload.

## 5. Demonstrate Subtitle Fallbacks

Explain the timed-text priority:

1. attached SRT/VTT files;
2. embedded subtitle stream extracted by FFmpeg;
3. faster-whisper transcription when no subtitle is available.

The scene record stores `subtitle_sources` and `caption_source`, so the origin of each generated description is auditable.

## 6. Demonstrate The Folder Crawler

The lower part of Scene Lab supports batch media folders. Use **Probe** before starting to confirm supported videos and sidecar subtitles.

Recommended series layout:

```text
D:\Series\Show.Name.S01E01.mkv
D:\Series\Show.Name.S01E01.fa.srt
D:\Series\Show.Name.S01E02.mkv
D:\Series\Show.Name.S01E02.fa.srt
```

CLI equivalent:

```powershell
.\.venv\Scripts\python.exe .\scripts\crawl_offline_videos.py "D:\Series" --min-scene-sec 4 --max-scene-sec 45 --threshold 0.35 --sample-fps 2
```

The crawler recursively detects episodes, parses season/episode metadata, cleans release tags from filenames, and persists one JSON record per scene.

## 7. Run Verification

Fast API and ingestion smoke test:

```powershell
.\.venv\Scripts\python.exe .\scripts\smoke_test_final.py
```

Retrieval quality benchmark:

```powershell
.\.venv\Scripts\python.exe .\scripts\retrieval_quality_benchmark.py
```

Responsive visual regression test:

```powershell
node .\scripts\visual_smoke_test.js http://127.0.0.1:8000
```

The quality report includes Hit@1, Hit@5, MRR@10, latency, and the rank of every expected title. The visual test covers 1440px desktop and 390px mobile, browser errors, and horizontal overflow.

Verified final benchmark:

```text
Hit@1       0.90
Hit@5       1.00
MRR@10      0.95
Mean query  0.237 s
```

The real upload regression completed in about 36 seconds including a CUDA scene-index rebuild, created three timestamped scenes and three keyframes, attached the supplied SRT, and returned `Benchmark Show` at rank 1 with a playable source URL.

## 8. Explain The Engineering Decisions

- **Why two indexes?** Whole-title vectors provide broad plot recall; scene vectors preserve local dialogue, objects, actions, and timecodes.
- **Why collapse by title?** Multiple matching scenes from one movie should improve that movie's evidence, not flood the result list.
- **Why subtitles plus vision?** Dialogue alone misses silent visual moments; vision alone misses names and exact lines.
- **Why a static PWA and local backend?** GitHub Pages is ideal for a shareable installable showcase but cannot execute Python, FAISS, FFmpeg, or CUDA.
- **Why no Netflix crawler?** Protected streaming automation and DRM bypass are neither legal nor technically appropriate. CineScene accepts authorized local files or a future licensed provider feed.

## 9. Current Scope

CineScene is a complete research prototype and presentation-ready local system. Retrieval quality still depends on catalog coverage and the quality of scene text. Adding licensed full-film/episode media and reliable subtitles increases recall without changing the architecture.
