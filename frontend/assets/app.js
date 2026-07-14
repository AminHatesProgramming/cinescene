const sessionKey = "cinescene-session";
const sessionId = localStorage.getItem(sessionKey) ||
  (window.crypto?.randomUUID ? window.crypto.randomUUID() : `session-${Date.now()}-${Math.random().toString(16).slice(2)}`);
localStorage.setItem(sessionKey, sessionId);

const jsonHeaders = {
  "Content-Type": "application/json",
  "X-CineScene-Session": sessionId,
};

const state = {
  view: "discover",
  query: "",
  results: [],
  selectedVideo: null,
  subtitles: [],
  health: null,
  crawler: null,
  currentJob: null,
  lastProgress: 0,
};

const els = Object.fromEntries(
  [
    "viewTitle", "mobileMenuButton", "reloadButton", "statusDot", "sidebarStatus", "sidebarModel",
    "activeJobBadge", "engineVectorCount", "queryInput", "topKInput", "rerankInput", "searchButton",
    "clearButton", "metricIndexed", "metricScenes", "metricOffline", "metricCuda", "resultCount", "results",
    "crawlerHealth", "crawlerDocs", "crawlerScenes", "crawlerFrames", "crawlerNotice", "crawlerPreview",
    "videoInput", "videoDropZone", "videoDropTitle", "videoDropMeta", "chooseVideoButton", "selectedVideo",
    "removeVideoButton", "movieTitleInput", "subtitleInput", "chooseSubtitleButton", "subtitleSummary", "subtitleList",
    "ingestForm", "embeddedSubtitleInput", "transcribeInput", "visionInput", "startAnalysisButton",
    "uploadMinSceneInput", "uploadMaxSceneInput", "uploadThresholdInput", "uploadFpsInput",
    "uploadMinSceneValue", "uploadMaxSceneValue", "uploadThresholdValue", "uploadFpsValue",
    "jobPanel", "jobTitle", "jobPercent", "jobBar", "jobStatus", "pipelineSummary", "pipelineSteps",
    "refreshCrawlerButton", "crawlForm", "crawlPathInput", "crawlTitleInput", "crawlMinSceneInput",
    "crawlMaxSceneInput", "crawlThresholdInput", "crawlFpsInput", "crawlCatalogInput", "probeCrawlerPathButton",
    "probeList", "historyList", "favoriteList", "ingestionList", "rebuildIndexButton", "systemSearchState",
    "systemSearchDetail", "systemDetectorState", "systemDetectorDetail", "systemSubtitleState", "systemSubtitleDetail",
    "systemVisionState", "systemVisionDetail", "runtimeReport", "playerDialog", "scenePlayer", "playerTitle",
    "playerTimecode", "playerCaption", "closePlayerButton", "toastRegion",
  ].map((id) => [id, document.getElementById(id)])
);

function refreshIcons(root = document) {
  if (window.lucide?.createIcons) window.lucide.createIcons({ root });
}

async function api(path, options = {}) {
  const response = await fetch(path, { ...options, headers: options.headers || jsonHeaders });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || payload.error || response.statusText || "Request failed");
  return payload;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatBytes(bytes) {
  const value = Number(bytes || 0);
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`;
  if (value < 1024 ** 3) return `${(value / 1024 ** 2).toFixed(1)} MB`;
  return `${(value / 1024 ** 3).toFixed(2)} GB`;
}

function timecode(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds || 0)));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  return hours
    ? `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`
    : `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

function compactNumber(value) {
  const number = Number(value || 0);
  return new Intl.NumberFormat("en", { notation: number >= 10000 ? "compact" : "standard" }).format(number);
}

function toast(message, type = "info") {
  const item = document.createElement("div");
  item.className = `toast ${type}`;
  item.textContent = message;
  els.toastRegion.appendChild(item);
  window.setTimeout(() => item.remove(), 4800);
}

function setBusy(value) {
  document.body.classList.toggle("busy", value);
  els.searchButton.disabled = value;
  els.reloadButton.disabled = value;
}

function setView(view, updateHash = true) {
  const valid = ["discover", "ingest", "library", "system"];
  state.view = valid.includes(view) ? view : "discover";
  document.querySelectorAll(".view").forEach((item) => item.classList.toggle("active", item.dataset.view === state.view));
  document.querySelectorAll(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.viewLink === state.view));
  const titles = { discover: "Discover", ingest: "Scene Lab", library: "Library", system: "System" };
  els.viewTitle.textContent = titles[state.view];
  document.body.classList.remove("nav-open");
  if (updateHash) history.replaceState(null, "", `#${state.view}`);
  if (state.view === "library") loadMemory();
  if (state.view === "ingest") loadCrawlerStatus();
}

function chips(items, limit = 6) {
  return (items || []).slice(0, limit).map((item) => `<span class="chip">${escapeHtml(item)}</span>`).join("");
}

function sourceLabel(movie) {
  if (movie.source === "offline_video_ingestion") return "Scene index";
  if (movie.source === "public_series_metadata" || movie.source === "tvmaze_series") return "Series catalog";
  if (movie.source === "tmdb_enriched") return "Catalog";
  return movie.source || "Catalog";
}

function mediaLabel(movie) {
  if (movie.media_type === "series" && movie.season && movie.episode) {
    return `Series S${String(movie.season).padStart(2, "0")}E${String(movie.episode).padStart(2, "0")}`;
  }
  return movie.media_type === "series" ? "Series" : "Movie";
}

function matchedScene(movie) {
  return movie.matched_scene || movie.matched_scenes?.[0] || null;
}

function renderResults(results) {
  state.results = results || [];
  els.resultCount.textContent = state.results.length ? `${state.results.length} ranked titles` : "No matches";
  if (!state.results.length) {
    els.results.className = "results-grid empty-results";
    els.results.innerHTML = `<div class="empty-visual"><i data-lucide="circle-slash-2"></i></div><strong>No close matches found.</strong><span>Try dialogue fragments, a visible action, a location, or the mood of the scene.</span>`;
    refreshIcons(els.results);
    return;
  }

  els.results.className = "results-grid";
  els.results.innerHTML = state.results.map((movie, index) => {
    const scene = matchedScene(movie);
    const image = scene?.keyframe_url || movie.first_keyframe_url;
    const text = scene?.transcript || scene?.visual_caption || "No scene-level transcript is available for this catalog title.";
    const tags = [...(scene?.mood_tags || []), ...(scene?.visual_tags || []), ...(movie.genres || [])];
    const score = Number(movie.score || 0).toFixed(3);
    const media = image
      ? `<div class="result-media"><img src="${escapeHtml(image)}" alt="Keyframe from ${escapeHtml(movie.title)}" loading="lazy" />`
      : `<div class="result-media no-image"><span class="placeholder-number">${String(movie.rank || index + 1).padStart(2, "0")}</span>`;
    return `<article class="result-card">
      ${media}<span class="result-rank">Rank ${escapeHtml(movie.rank || index + 1)}</span><span class="result-source">${escapeHtml(sourceLabel(movie))}</span></div>
      <div class="result-body">
        <div class="result-title-row"><div><h4>${escapeHtml(movie.title)}</h4><small>${escapeHtml(mediaLabel(movie))} · ${escapeHtml(movie.year || "N/A")}${movie.director && movie.director !== "Unknown" ? ` · ${escapeHtml(movie.director)}` : ""}</small></div><span class="score-ring" title="Retrieval score">${score}</span></div>
        <p class="result-overview">${escapeHtml(movie.overview || "Scene-level offline media document.")}</p>
        ${scene ? `<div class="matched-scene"><div class="scene-label"><span>Best matching scene</span><time>${timecode(scene.start_sec)} - ${timecode(scene.end_sec)}</time></div><p dir="auto">${escapeHtml(text)}</p></div>` : ""}
        <div class="chip-row">${chips(tags, 7)}</div>
        <div class="result-actions">
          ${movie.video_url && scene ? `<button class="button secondary" type="button" data-action="play" data-index="${index}"><i data-lucide="play"></i>Play scene</button>` : ""}
          <button class="icon-button ghost" type="button" data-action="favorite" data-index="${index}" title="Save" aria-label="Save result"><i data-lucide="bookmark"></i></button>
          <button class="icon-button ghost" type="button" data-action="relevant" data-index="${index}" title="Relevant" aria-label="Mark relevant"><i data-lucide="thumbs-up"></i></button>
        </div>
      </div>
    </article>`;
  }).join("");
  refreshIcons(els.results);
}

function renderSearchLoading() {
  els.resultCount.textContent = "Searching scene vectors";
  els.results.className = "results-grid empty-results";
  els.results.innerHTML = `<div class="empty-visual"><i data-lucide="loader-circle"></i></div><strong>Matching scene memory...</strong><span>Combining semantic, dialogue, and lexical signals.</span>`;
  refreshIcons(els.results);
}

async function runSearch(queryOverride) {
  const query = (queryOverride || els.queryInput.value).trim();
  if (!query) {
    toast("Describe a scene first.", "error");
    els.queryInput.focus();
    return;
  }
  state.query = query;
  els.queryInput.value = query;
  setView("discover");
  setBusy(true);
  renderSearchLoading();
  try {
    const payload = await api("/api/search", {
      method: "POST",
      body: JSON.stringify({ query, top_k: Number(els.topKInput.value || 8), use_reranking: els.rerankInput.checked }),
    });
    renderResults(payload.results || []);
    loadMemory();
  } catch (error) {
    els.resultCount.textContent = "Search error";
    els.results.className = "results-grid empty-results";
    els.results.innerHTML = `<div class="empty-visual"><i data-lucide="triangle-alert"></i></div><strong>Search engine is unavailable.</strong><span>${escapeHtml(error.message)}</span>`;
    refreshIcons(els.results);
  } finally {
    setBusy(false);
  }
}

function updateHealth(payload) {
  state.health = payload;
  const engine = payload.engine || {};
  const runtime = payload.runtime || {};
  const cuda = runtime.cuda || {};
  const ok = Boolean(payload.ok);
  els.statusDot.className = `status-dot ${ok ? "ok" : "error"}`;
  els.sidebarStatus.textContent = ok ? "Engine online" : "Engine error";
  els.sidebarModel.textContent = engine.model ? String(engine.model).split(/[\\/]/).pop() : payload.error || "Unavailable";
  els.engineVectorCount.textContent = compactNumber(engine.index_vectors || 0);
  els.metricIndexed.textContent = compactNumber(engine.movies || 0);
  els.metricScenes.textContent = compactNumber(engine.scene_vectors || runtime.scene_index_report?.scene_vectors || 0);
  els.metricOffline.textContent = compactNumber(runtime.offline_documents || 0);
  els.metricCuda.textContent = cuda.available ? (cuda.device || "CUDA") : "CPU";
  els.systemSearchState.textContent = ok ? "Online" : "Unavailable";
  els.systemSearchDetail.textContent = `${engine.movie_vectors || 0} title vectors + ${engine.scene_vectors || 0} scene vectors`;
  renderRuntimeReport();
}

async function loadHealth() {
  try {
    updateHealth(await api("/api/health", { method: "GET", headers: { "X-CineScene-Session": sessionId } }));
  } catch (error) {
    updateHealth({ ok: false, error: error.message, engine: {}, runtime: {} });
  }
}

function setCrawlerNotice(message, mode = "info") {
  els.crawlerNotice.textContent = message;
  els.crawlerNotice.dataset.mode = mode;
}

function renderRuntimeReport() {
  if (!els.runtimeReport) return;
  const report = {
    search: state.health?.engine || {},
    runtime: state.health?.runtime || {},
    crawler: state.crawler ? {
      enabled: state.crawler.crawler_enabled,
      capabilities: state.crawler.capabilities,
      offline_documents: state.crawler.offline_documents,
      scene_total: state.crawler.scene_total,
      keyframes: state.crawler.keyframes,
      latest_report: state.crawler.latest_report,
    } : {},
  };
  els.runtimeReport.textContent = JSON.stringify(report, null, 2);
}

function updateCrawlerDashboard(payload) {
  state.crawler = payload;
  const caps = payload.capabilities || {};
  const enabled = Boolean(payload.crawler_enabled);
  els.crawlerHealth.className = `health-badge ${enabled ? "ok" : "error"}`;
  els.crawlerHealth.innerHTML = `<span></span>${enabled ? "Pipeline ready" : "Pipeline unavailable"}`;
  els.crawlerDocs.textContent = compactNumber(payload.offline_documents || 0);
  els.crawlerScenes.textContent = compactNumber(payload.scene_total || 0);
  els.crawlerFrames.textContent = compactNumber(payload.keyframes || 0);
  els.systemDetectorState.textContent = enabled ? "Ready" : "Missing";
  els.systemDetectorDetail.textContent = caps.scene_detector || "OpenCV is unavailable";
  els.systemSubtitleState.textContent = caps.ffmpeg ? "FFmpeg ready" : caps.whisper_optional ? "Whisper ready" : "Sidecar only";
  const speechState = caps.whisper_model_cached ? "local model" : caps.whisper_optional ? "on demand" : "no";
  els.systemSubtitleDetail.textContent = `Sidecar: yes · embedded: ${caps.embedded_subtitles ? "yes" : "no"} · speech: ${speechState}`;
  els.systemVisionState.textContent = caps.vision_captioner_optional ? "Available" : "Unavailable";
  els.systemVisionDetail.textContent = caps.vision_model || "Install the vision dependencies";

  const previews = payload.scene_previews || [];
  if (!previews.length) {
    els.crawlerPreview.className = "media-timeline empty-inline";
    els.crawlerPreview.textContent = "No processed scenes yet.";
  } else {
    els.crawlerPreview.className = "media-timeline";
    els.crawlerPreview.innerHTML = previews.slice(0, 8).map((item) => {
      const episode = item.media_type === "series" && item.season && item.episode
        ? `S${String(item.season).padStart(2, "0")}E${String(item.episode).padStart(2, "0")}` : (item.media_type || "movie");
      const scenes = (item.timeline || []).slice(0, 8).map((scene) => {
        const image = scene.keyframe_url ? `<img src="${escapeHtml(scene.keyframe_url)}" alt="Scene ${escapeHtml(scene.scene_number)}" loading="lazy" />` : `<span class="frame-placeholder"></span>`;
        const text = scene.transcript || scene.visual_caption || "Visual scene";
        return `<article class="scene-tile">${image}<div><strong>Scene ${escapeHtml(scene.scene_number)} · ${timecode(scene.start_sec)}</strong><p dir="auto">${escapeHtml(text)}</p></div></article>`;
      }).join("");
      return `<article class="timeline-media"><div class="timeline-media-header"><h4>${escapeHtml(item.title)}</h4><span>${escapeHtml(episode)} · ${escapeHtml(item.scene_count)} scenes</span></div><div class="scene-strip">${scenes}</div></article>`;
    }).join("");
  }
  renderRuntimeReport();
  refreshIcons();
}

async function loadCrawlerStatus() {
  try {
    updateCrawlerDashboard(await api("/api/crawl/status", { method: "GET", headers: { "X-CineScene-Session": sessionId } }));
  } catch (error) {
    setCrawlerNotice(error.message, "error");
  }
}

function selectVideo(file) {
  if (!file) return;
  if (!file.type.startsWith("video/") && !/\.(mp4|mkv|avi|mov|webm|m4v|wmv)$/i.test(file.name)) {
    toast("Choose a supported video file.", "error");
    return;
  }
  state.selectedVideo = file;
  els.selectedVideo.hidden = false;
  els.selectedVideo.querySelector("strong").textContent = file.name;
  els.selectedVideo.querySelector("span").textContent = `${formatBytes(file.size)} · local media`;
  els.videoDropTitle.textContent = "Video selected";
  els.videoDropMeta.textContent = file.name;
  if (!els.movieTitleInput.value.trim()) els.movieTitleInput.value = file.name.replace(/\.[^.]+$/, "").replace(/[._]+/g, " ");
  els.startAnalysisButton.disabled = false;
}

function clearVideo() {
  state.selectedVideo = null;
  els.videoInput.value = "";
  els.selectedVideo.hidden = true;
  els.videoDropTitle.textContent = "Drop a video here";
  els.videoDropMeta.textContent = "MP4, MKV, MOV, AVI, WEBM";
  els.startAnalysisButton.disabled = true;
}

function setSubtitles(files) {
  const incoming = Array.from(files || []).filter((file) => /\.(srt|vtt)$/i.test(file.name));
  state.subtitles = [...state.subtitles, ...incoming].filter((file, index, array) => array.findIndex((item) => item.name === file.name && item.size === file.size) === index);
  els.subtitleSummary.textContent = state.subtitles.length ? `${state.subtitles.length} track${state.subtitles.length > 1 ? "s" : ""} selected` : "No sidecar selected";
  els.subtitleList.innerHTML = state.subtitles.map((file) => `<span>${escapeHtml(file.name)} · ${formatBytes(file.size)}</span>`).join("");
}

function updateRangeOutputs() {
  els.uploadMinSceneValue.textContent = `${els.uploadMinSceneInput.value}s`;
  els.uploadMaxSceneValue.textContent = `${els.uploadMaxSceneInput.value}s`;
  els.uploadThresholdValue.textContent = Number(els.uploadThresholdInput.value).toFixed(2);
  els.uploadFpsValue.textContent = `${els.uploadFpsInput.value} fps`;
}

const stageOrder = ["uploaded", "scene_detection", "scene_records_ready", "scene_index", "ready"];

function updatePipeline(stage, status = "running") {
  const normalized = stageOrder.includes(stage) ? stage : stage === "processing_video" ? "scene_detection" : stage === "processed_video" ? "scene_records_ready" : "uploaded";
  const currentIndex = status === "completed" ? stageOrder.length : stageOrder.indexOf(normalized);
  els.pipelineSteps.querySelectorAll("li").forEach((item, index) => {
    item.classList.toggle("complete", index < currentIndex || status === "completed");
    item.classList.toggle("active", status !== "completed" && index === currentIndex);
  });
}

function showJob(job, overrideMessage = "") {
  state.currentJob = job;
  els.activeJobBadge.hidden = ["completed", "failed"].includes(job.status);
  els.jobPanel.hidden = false;
  const progress = Math.max(state.lastProgress, Number(job.progress || 0));
  state.lastProgress = job.status === "completed" ? 100 : progress;
  els.jobBar.style.width = `${Math.min(100, state.lastProgress)}%`;
  els.jobPercent.textContent = `${Math.round(state.lastProgress)}%`;
  els.jobTitle.textContent = job.label || "Scene analysis";
  const current = job.current_video ? ` · ${String(job.current_video).split(/[\\/]/).pop()}` : "";
  els.jobStatus.textContent = overrideMessage || `${job.stage || job.status}${current}`;
  els.pipelineSummary.textContent = job.status === "completed" ? "Search index updated" : job.status === "failed" ? "Pipeline failed" : "Analysis in progress";
  updatePipeline(job.stage || "uploaded", job.status);
}

function sleep(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

async function pollJob(job, onComplete) {
  showJob(job);
  for (let attempt = 0; attempt < 18000; attempt += 1) {
    await sleep(1200);
    const current = await api(`/api/jobs/${encodeURIComponent(job.id)}`, { method: "GET", headers: { "X-CineScene-Session": sessionId } });
    showJob(current);
    if (current.status === "completed") {
      showJob(current, `Completed in ${current.elapsed_sec || 0}s`);
      await onComplete(current.result || {});
      return current;
    }
    if (current.status === "failed") {
      showJob(current, current.error || "Pipeline failed");
      throw new Error(current.error || "Pipeline failed");
    }
  }
  throw new Error("The job is still running. Check System for its status.");
}

function uploadFormData(formData, onProgress) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", "/api/ingest/video");
    request.setRequestHeader("X-CineScene-Session", sessionId);
    request.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total);
    });
    request.addEventListener("load", () => {
      let payload = {};
      try { payload = JSON.parse(request.responseText || "{}"); } catch (_) { payload = {}; }
      if (request.status >= 200 && request.status < 300) resolve(payload);
      else reject(new Error(payload.detail || request.statusText || "Upload failed"));
    });
    request.addEventListener("error", () => reject(new Error("Upload connection failed")));
    request.send(formData);
  });
}

async function ingestVideo(event) {
  event.preventDefault();
  if (!state.selectedVideo) return;
  state.lastProgress = 0;
  els.startAnalysisButton.disabled = true;
  const form = new FormData();
  form.append("video", state.selectedVideo, state.selectedVideo.name);
  state.subtitles.forEach((file) => form.append("subtitles", file, file.name));
  form.append("movie_title", els.movieTitleInput.value.trim());
  form.append("min_scene_sec", els.uploadMinSceneInput.value);
  form.append("max_scene_sec", els.uploadMaxSceneInput.value);
  form.append("threshold", els.uploadThresholdInput.value);
  form.append("sample_fps", els.uploadFpsInput.value);
  form.append("extract_embedded_subtitles", String(els.embeddedSubtitleInput.checked));
  form.append("transcribe_audio", String(els.transcribeInput.checked));
  form.append("enable_vision", String(els.visionInput.checked));
  form.append("whisper_model", "small");
  form.append("vision_model", "Salesforce/blip-image-captioning-base");

  try {
    els.jobPanel.hidden = false;
    updatePipeline("uploaded");
    const payload = await uploadFormData(form, (fraction) => {
      const percent = Math.round(fraction * 100);
      showJob({ status: "running", stage: "uploaded", progress: fraction * 10, label: "Uploading media" }, `Uploading · ${percent}%`);
    });
    await pollJob(payload.job, async (result) => {
      const report = result.report || {};
      setCrawlerNotice(`${report.scenes_created || 0} scenes indexed from ${report.videos_processed || 0} video.`, "ok");
      toast("Video analyzed. Scene search is ready.");
      updateCrawlerDashboard(result.crawler || {});
      await Promise.all([loadHealth(), loadMemory(), loadCrawlerStatus()]);
    });
  } catch (error) {
    setCrawlerNotice(error.message, "error");
    toast(error.message, "error");
  } finally {
    els.startAnalysisButton.disabled = !state.selectedVideo;
  }
}

async function crawlOfflineFolder(event) {
  event.preventDefault();
  const root = els.crawlPathInput.value.trim();
  if (!root) {
    toast("Enter a folder path.", "error");
    return;
  }
  state.lastProgress = 0;
  try {
    const payload = await api("/api/crawl/offline-async", {
      method: "POST",
      body: JSON.stringify({
        root,
        title_prefix: els.crawlTitleInput.value.trim(),
        min_scene_sec: Number(els.uploadMinSceneInput.value || 4),
        max_scene_sec: Number(els.uploadMaxSceneInput.value || 45),
        threshold: Number(els.uploadThresholdInput.value || 0.35),
        sample_fps: Number(els.uploadFpsInput.value || 2),
        update_catalog: true,
        extract_embedded_subtitles: els.embeddedSubtitleInput.checked,
        transcribe_audio: els.transcribeInput.checked,
        enable_vision: els.visionInput.checked,
        whisper_model: "small",
        vision_model: "Salesforce/blip-image-captioning-base",
      }),
    });
    await pollJob(payload.job, async (result) => {
      updateCrawlerDashboard(result.crawler || {});
      setCrawlerNotice(`${result.report?.scenes_created || 0} scenes indexed.`, "ok");
      toast("Folder crawl completed.");
      await loadHealth();
    });
  } catch (error) {
    setCrawlerNotice(error.message, "error");
    toast(error.message, "error");
  }
}

async function probeOfflineFolder() {
  const root = els.crawlPathInput.value.trim();
  if (!root) {
    toast("Enter a folder path.", "error");
    return;
  }
  try {
    const payload = await api("/api/crawl/probe", { method: "POST", body: JSON.stringify({ root }) });
    els.probeList.innerHTML = (payload.preview || []).map((item) => `<article class="probe-item"><strong>${escapeHtml(item.title)}</strong><span>${escapeHtml(item.media_type)} · ${(item.subtitle_files || []).length} subtitle track(s)</span><span>${escapeHtml(item.path)}</span></article>`).join("") || `<article class="probe-item"><strong>No media found</strong><span>${escapeHtml(root)}</span></article>`;
    toast(`${payload.videos_found || 0} video files found.`);
  } catch (error) {
    els.probeList.innerHTML = `<article class="probe-item"><strong>Probe failed</strong><span>${escapeHtml(error.message)}</span></article>`;
    toast(error.message, "error");
  }
}

function renderLibraryList(container, items, options = {}) {
  if (!items?.length) {
    container.innerHTML = `<div class="library-item"><strong>${escapeHtml(options.empty || "Nothing here yet")}</strong><span>—</span></div>`;
    return;
  }
  container.innerHTML = items.map((item) => {
    const title = options.title(item);
    const body = options.body(item);
    const query = options.query ? ` data-query="${escapeHtml(options.query(item))}"` : "";
    return `<${options.query ? "button" : "div"} class="library-item"${query}><strong>${escapeHtml(title)}</strong><span>${escapeHtml(body)}</span></${options.query ? "button" : "div"}>`;
  }).join("");
}

async function loadMemory() {
  try {
    const [history, favorites, ingestions] = await Promise.all([
      api("/api/history", { method: "GET", headers: { "X-CineScene-Session": sessionId } }),
      api("/api/favorites", { method: "GET", headers: { "X-CineScene-Session": sessionId } }),
      api("/api/ingestions", { method: "GET", headers: { "X-CineScene-Session": sessionId } }),
    ]);
    renderLibraryList(els.historyList, history.items, { empty: "No searches yet", title: (item) => item.query, body: (item) => `${item.results?.length || 0} results · ${item.created_at}`, query: (item) => item.query });
    renderLibraryList(els.favoriteList, favorites.items, { empty: "No saved titles", title: (item) => item.title, body: (item) => `${mediaLabel(item)} · ${item.year || "N/A"}` });
    renderLibraryList(els.ingestionList, ingestions.items, { empty: "No local media", title: (item) => item.movie_title, body: (item) => `${item.scene_count || 0} records · ${item.created_at}` });
  } catch (error) {
    console.warn("Memory load failed", error);
  }
}

async function saveFavorite(movie) {
  await api("/api/favorites", { method: "POST", body: JSON.stringify({ movie }) });
  toast(`${movie.title} saved.`);
  loadMemory();
}

async function sendFeedback(movie) {
  await api("/api/feedback", { method: "POST", body: JSON.stringify({ query: state.query || els.queryInput.value, movie_title: movie.title, signal: "relevant" }) });
  toast("Relevance signal recorded.");
}

function openPlayer(movie) {
  const scene = matchedScene(movie);
  if (!movie.video_url || !scene) return;
  els.playerTitle.textContent = movie.title;
  els.playerTimecode.textContent = `${timecode(scene.start_sec)} - ${timecode(scene.end_sec)}`;
  els.playerCaption.textContent = scene.transcript || scene.visual_caption || "";
  els.scenePlayer.src = movie.video_url;
  els.scenePlayer.addEventListener("loadedmetadata", () => {
    els.scenePlayer.currentTime = Number(scene.start_sec || 0);
  }, { once: true });
  els.playerDialog.showModal();
}

async function reloadEngine() {
  setBusy(true);
  try {
    const payload = await api("/api/reload", { method: "POST", body: "{}" });
    updateHealth(payload);
    toast("Search engine reloaded.");
  } catch (error) {
    toast(error.message, "error");
  } finally {
    setBusy(false);
  }
}

async function rebuildIndexes() {
  state.lastProgress = 0;
  setView("system");
  try {
    const payload = await api("/api/index/rebuild-async", {
      method: "POST",
      body: JSON.stringify({ input_path: "", model_path: "models/bge-large-en-v1.5", batch_size: 8, use_hnsw: false, use_base_model: false }),
    });
    await pollJob(payload.job, async (result) => {
      updateHealth(result);
      updateCrawlerDashboard(result.crawler || {});
      toast("Movie and scene indexes rebuilt.");
    });
  } catch (error) {
    toast(error.message, "error");
  }
}

document.querySelectorAll("[data-view-link]").forEach((item) => item.addEventListener("click", (event) => {
  event.preventDefault();
  setView(item.dataset.viewLink);
}));

document.querySelectorAll("[data-query]").forEach((item) => item.addEventListener("click", () => runSearch(item.dataset.query)));
document.querySelectorAll("[data-analysis-mode]").forEach((item) => item.addEventListener("click", () => {
  document.querySelectorAll("[data-analysis-mode]").forEach((button) => button.classList.toggle("active", button === item));
  const deep = item.dataset.analysisMode === "deep";
  els.transcribeInput.checked = deep;
  els.visionInput.checked = deep;
}));

els.mobileMenuButton.addEventListener("click", () => document.body.classList.toggle("nav-open"));
els.searchButton.addEventListener("click", () => runSearch());
els.reloadButton.addEventListener("click", reloadEngine);
els.clearButton.addEventListener("click", () => {
  els.queryInput.value = "";
  els.queryInput.focus();
});
els.queryInput.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") runSearch();
});

els.chooseVideoButton.addEventListener("click", (event) => { event.stopPropagation(); els.videoInput.click(); });
els.videoDropZone.addEventListener("click", () => els.videoInput.click());
els.videoDropZone.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") els.videoInput.click(); });
els.videoInput.addEventListener("change", () => selectVideo(els.videoInput.files[0]));
els.removeVideoButton.addEventListener("click", clearVideo);
["dragenter", "dragover"].forEach((name) => els.videoDropZone.addEventListener(name, (event) => { event.preventDefault(); els.videoDropZone.classList.add("dragging"); }));
["dragleave", "drop"].forEach((name) => els.videoDropZone.addEventListener(name, (event) => { event.preventDefault(); els.videoDropZone.classList.remove("dragging"); }));
els.videoDropZone.addEventListener("drop", (event) => selectVideo(event.dataTransfer.files[0]));
els.chooseSubtitleButton.addEventListener("click", () => els.subtitleInput.click());
els.subtitleInput.addEventListener("change", () => setSubtitles(els.subtitleInput.files));
els.ingestForm.addEventListener("submit", ingestVideo);

[els.uploadMinSceneInput, els.uploadMaxSceneInput, els.uploadThresholdInput, els.uploadFpsInput].forEach((input) => input.addEventListener("input", updateRangeOutputs));
els.crawlForm.addEventListener("submit", crawlOfflineFolder);
els.probeCrawlerPathButton.addEventListener("click", probeOfflineFolder);
els.refreshCrawlerButton.addEventListener("click", loadCrawlerStatus);
els.rebuildIndexButton.addEventListener("click", rebuildIndexes);

els.results.addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-action]");
  if (!button) return;
  const movie = state.results[Number(button.dataset.index)];
  if (!movie) return;
  try {
    if (button.dataset.action === "play") openPlayer(movie);
    if (button.dataset.action === "favorite") await saveFavorite(movie);
    if (button.dataset.action === "relevant") await sendFeedback(movie);
  } catch (error) {
    toast(error.message, "error");
  }
});

els.historyList.addEventListener("click", (event) => {
  const item = event.target.closest("[data-query]");
  if (item) runSearch(item.dataset.query);
});
els.closePlayerButton.addEventListener("click", () => els.playerDialog.close());
els.playerDialog.addEventListener("close", () => { els.scenePlayer.pause(); els.scenePlayer.removeAttribute("src"); els.scenePlayer.load(); });

window.addEventListener("hashchange", () => setView(location.hash.slice(1), false));

updateRangeOutputs();
refreshIcons();
setView(location.hash.slice(1) || "discover", false);
Promise.all([loadHealth(), loadCrawlerStatus(), loadMemory()]);
