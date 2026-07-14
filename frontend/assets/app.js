const sessionKey = "cinescene-session";

function createSessionId() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return window.crypto.randomUUID();
  }
  const randomPart = Math.random().toString(16).slice(2);
  return `session-${Date.now()}-${randomPart}`;
}

const sessionId = localStorage.getItem(sessionKey) || createSessionId();
localStorage.setItem(sessionKey, sessionId);

const headers = {
  "Content-Type": "application/json",
  "X-CineScene-Session": sessionId,
};

const state = {
  query: "",
  results: [],
  loading: false,
};

const els = {
  status: document.getElementById("systemStatus"),
  reload: document.getElementById("reloadButton"),
  metricIndexed: document.getElementById("metricIndexed"),
  metricOffline: document.getElementById("metricOffline"),
  metricModel: document.getElementById("metricModel"),
  metricCuda: document.getElementById("metricCuda"),
  query: document.getElementById("queryInput"),
  topK: document.getElementById("topKInput"),
  rerank: document.getElementById("rerankInput"),
  search: document.getElementById("searchButton"),
  clear: document.getElementById("clearButton"),
  results: document.getElementById("results"),
  resultCount: document.getElementById("resultCount"),
  history: document.getElementById("historyList"),
  favorites: document.getElementById("favoriteList"),
  ingestion: document.getElementById("ingestionList"),
  crawlerPreview: document.getElementById("crawlerPreview"),
  crawlerDocs: document.getElementById("crawlerDocs"),
  crawlerScenes: document.getElementById("crawlerScenes"),
  crawlerFrames: document.getElementById("crawlerFrames"),
  crawlerNotice: document.getElementById("crawlerNotice"),
  ingestForm: document.getElementById("ingestForm"),
  crawlForm: document.getElementById("crawlForm"),
  rebuildIndex: document.getElementById("rebuildIndexButton"),
  refreshCrawler: document.getElementById("refreshCrawlerButton"),
  sampleCrawlerPath: document.getElementById("sampleCrawlerPathButton"),
  jobPanel: document.getElementById("jobPanel"),
  jobBar: document.getElementById("jobBar"),
  jobTitle: document.getElementById("jobTitle"),
  jobStatus: document.getElementById("jobStatus"),
  video: document.getElementById("videoInput"),
  movieTitle: document.getElementById("movieTitleInput"),
  crawlPath: document.getElementById("crawlPathInput"),
  crawlTitle: document.getElementById("crawlTitleInput"),
  crawlMinScene: document.getElementById("crawlMinSceneInput"),
  crawlThreshold: document.getElementById("crawlThresholdInput"),
  crawlFps: document.getElementById("crawlFpsInput"),
  crawlCatalog: document.getElementById("crawlCatalogInput"),
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: options.headers || headers,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || payload.error || response.statusText);
  }
  return payload;
}

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function showError(message) {
  els.results.className = "results-list empty-state";
  els.results.textContent = message || "Something went wrong.";
  els.resultCount.textContent = "Error";
}

function setBusy(value) {
  state.loading = value;
  document.body.classList.toggle("busy", value);
  [els.search, els.reload, els.rebuildIndex, els.refreshCrawler, els.sampleCrawlerPath]
    .filter(Boolean)
    .forEach((button) => {
      button.disabled = value;
    });
}

function showJob(job, message) {
  if (!els.jobPanel) return;
  els.jobPanel.hidden = false;
  els.jobTitle.textContent = job.label || job.kind || "Working";
  els.jobStatus.textContent = message || `${job.status || "running"} · ${job.elapsed_sec || 0}s`;
  const elapsed = Number(job.elapsed_sec || 0);
  const progress = job.status === "completed" ? 100 : job.status === "failed" ? 100 : Math.min(92, 8 + elapsed * 0.24);
  els.jobBar.style.width = `${progress}%`;
  els.jobPanel.dataset.status = job.status || "running";
}

function hideJobSoon() {
  window.setTimeout(() => {
    if (els.jobPanel && els.jobPanel.dataset.status === "completed") {
      els.jobPanel.hidden = true;
    }
  }, 4500);
}

async function pollJob(job, onComplete) {
  let current = job;
  showJob(current);
  for (let attempt = 0; attempt < 360; attempt += 1) {
    await sleep(1500);
    current = await api(`/api/jobs/${encodeURIComponent(current.id)}`, {
      method: "GET",
      headers: { "X-CineScene-Session": sessionId },
    });
    showJob(current);
    if (current.status === "completed") {
      showJob(current, `Completed in ${current.elapsed_sec}s`);
      await onComplete(current.result || {});
      hideJobSoon();
      return current;
    }
    if (current.status === "failed") {
      showJob(current, current.error || "Job failed");
      throw new Error(current.error || "Job failed");
    }
  }
  throw new Error("Job timed out. Check the terminal for progress.");
}

function benchmarkFolderPath() {
  return "C:\\Users\\Webhouse\\Desktop\\quera\\cinescene\\data\\offline_videos\\benchmark_folder";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function chips(items, limit = 6) {
  return (items || [])
    .slice(0, limit)
    .map((item) => `<span class="chip">${escapeHtml(item)}</span>`)
    .join("");
}

function mediaLabel(movie) {
  if (movie.media_type === "series" && movie.season && movie.episode) {
    return `Series S${String(movie.season).padStart(2, "0")}E${String(movie.episode).padStart(2, "0")}`;
  }
  if (movie.media_type === "series") return "Series";
  return movie.source === "offline_video_ingestion" ? "Offline video" : "Movie";
}

function sourceLabel(movie) {
  if (movie.source === "offline_video_ingestion") return "Offline crawler";
  if (movie.source === "tmdb_enriched") return "TMDB enriched";
  return movie.source || "Catalog";
}

function updateMetrics(payload) {
  const engine = payload.engine || {};
  const runtime = payload.runtime || {};
  const cuda = runtime.cuda || {};
  els.metricIndexed.textContent = engine.movies ?? "-";
  els.metricOffline.textContent = runtime.offline_documents ?? "0";
  els.metricModel.textContent = engine.model || "-";
  els.metricCuda.textContent = cuda.available ? cuda.device || "Available" : "CPU";
}

function setCrawlerNotice(message, mode = "info") {
  if (!els.crawlerNotice) return;
  els.crawlerNotice.textContent = message;
  els.crawlerNotice.dataset.mode = mode;
}

function updateCrawlerDashboard(payload) {
  if (!payload || !els.crawlerPreview) return;
  els.crawlerDocs.textContent = payload.offline_documents ?? "-";
  els.crawlerScenes.textContent = payload.scene_files ?? "-";
  els.crawlerFrames.textContent = payload.keyframes ?? "-";

  const previews = payload.scene_previews || [];
  if (!previews.length) {
    els.crawlerPreview.innerHTML = `<div class="side-item">No crawled scene files yet.</div>`;
    return;
  }

  els.crawlerPreview.innerHTML = previews
    .map((item) => {
      const episode =
        item.media_type === "series" && item.season && item.episode
          ? `S${String(item.season).padStart(2, "0")}E${String(item.episode).padStart(2, "0")}`
          : item.media_type || "movie";
      return `<div class="side-item">
        <strong>${escapeHtml(item.title)} (${escapeHtml(episode)})</strong>
        <span>${escapeHtml(item.scene_count)} scenes</span>
        <span>${escapeHtml(item.source_video || item.file)}</span>
        <span>${escapeHtml(item.first_scene || "Scene text is empty.")}</span>
      </div>`;
    })
    .join("");
}

function renderResults(results) {
  state.results = results || [];
  els.resultCount.textContent = state.results.length ? `${state.results.length} movies` : "No matches";

  if (!state.results.length) {
    els.results.className = "results-list empty-state";
    els.results.textContent = "No matching movie was returned.";
    return;
  }

  els.results.className = "results-list";
  els.results.innerHTML = state.results
    .map((movie, index) => {
      const scenes = (movie.scenes || [])
        .slice(0, 2)
        .map((scene) => `<li>${escapeHtml(scene)}</li>`)
        .join("");

      return `
        <article class="movie-card">
          <div class="movie-header">
            <div>
              <h2 class="movie-title">${movie.rank}. ${escapeHtml(movie.title)} <span>${escapeHtml(movie.year)}</span></h2>
              <div class="meta-row">
                <span class="chip media-chip">${escapeHtml(mediaLabel(movie))}</span>
                <span class="chip">${escapeHtml(sourceLabel(movie))}</span>
                <span class="chip">${escapeHtml(movie.director || "Unknown director")}</span>
                <span class="chip">Rating ${escapeHtml(movie.rating ?? 0)}</span>
                <span class="chip">Score <span class="score">${escapeHtml(movie.score)}</span></span>
              </div>
            </div>
          </div>
          <div class="tag-row">${chips(movie.genres, 5)}${chips(movie.mood, 5)}</div>
          <p class="overview">${escapeHtml(movie.overview || "No overview available.")}</p>
          ${scenes ? `<ol class="scene-list">${scenes}</ol>` : ""}
          <div class="movie-actions">
            <button data-action="favorite" data-index="${index}">Favorite</button>
            <button data-action="love" data-index="${index}">Relevant</button>
            <button data-action="hide" data-index="${index}">Not relevant</button>
          </div>
        </article>
      `;
    })
    .join("");
}

async function runSearch(queryOverride) {
  const query = (queryOverride || els.query.value).trim();
  if (!query) return;

  setBusy(true);
  els.resultCount.textContent = "Searching";
  try {
    const payload = await api("/api/search", {
      method: "POST",
      body: JSON.stringify({
        query,
        top_k: Number(els.topK.value || 8),
        use_reranking: els.rerank.checked,
      }),
    });
    state.query = query;
    renderResults(payload.results);
    loadMemory().catch((error) => console.warn("Memory refresh failed", error));
  } catch (error) {
    console.error("Search failed", error);
    showError(error.message);
  } finally {
    setBusy(false);
  }
}

async function loadHealth() {
  try {
    const payload = await api("/api/health", { method: "GET", headers: { "X-CineScene-Session": sessionId } });
    if (payload.ok) {
      els.status.className = "system-pill ok";
      els.status.textContent = `${payload.engine.movies} movies indexed`;
      updateMetrics(payload);
    } else {
      els.status.className = "system-pill error";
      els.status.textContent = payload.error || "Engine not ready";
      updateMetrics(payload);
    }
    loadCrawlerStatus().catch((error) => console.warn("Crawler status failed", error));
  } catch (error) {
    els.status.className = "system-pill error";
    els.status.textContent = error.message;
  }
}

async function loadCrawlerStatus() {
  const payload = await api("/api/crawl/status", { method: "GET", headers: { "X-CineScene-Session": sessionId } });
  updateCrawlerDashboard(payload);
  const report = payload.latest_report || {};
  const processed = report.videos_processed ?? 0;
  const docs = report.documents_created ?? payload.offline_documents ?? 0;
  setCrawlerNotice(`Crawler ready. Latest run: ${processed} videos processed, ${docs} documents created.`, "info");
  return payload;
}

async function reloadEngine() {
  setBusy(true);
  try {
    const payload = await api("/api/reload", { method: "POST" });
    if (!payload.ok) throw new Error(payload.error || "Engine reload failed");
    els.status.className = "system-pill ok";
    els.status.textContent = `${payload.engine.movies} movies indexed`;
    updateMetrics(payload);
    await loadCrawlerStatus();
  } catch (error) {
    els.status.className = "system-pill error";
    els.status.textContent = error.message;
  } finally {
    setBusy(false);
  }
}

async function rebuildSearchIndex() {
  setBusy(true);
  setCrawlerNotice("Rebuilding FAISS index. This can take a few minutes on CPU.", "warn");
  try {
    const payload = await api("/api/index/rebuild-async", {
      method: "POST",
      body: JSON.stringify({
        input_path: "",
        model_path: "model",
        batch_size: 64,
        use_hnsw: false,
        use_base_model: false,
      }),
    });
    await pollJob(payload.job, async (result) => {
      if (!result.ok) throw new Error(result.error || "Index rebuild failed");
      updateMetrics(result);
      updateCrawlerDashboard(result.crawler || {});
      els.status.className = "system-pill ok";
      els.status.textContent = `${result.engine.movies} movies indexed`;
      setCrawlerNotice(`Index rebuilt in ${result.elapsed_sec}s. Crawled media is searchable now.`, "ok");
    });
  } catch (error) {
    setCrawlerNotice(error.message, "error");
  } finally {
    setBusy(false);
  }
}

function renderSideList(container, items, options = {}) {
  if (!items || !items.length) {
    container.innerHTML = `<div class="side-item">${options.empty || "Nothing stored yet."}</div>`;
    return;
  }

  container.innerHTML = items
    .map((item) => {
      const title = options.title ? options.title(item) : item.title || item.query || item.movie_title;
      const body = options.body ? options.body(item) : item.created_at || "";
      return `<button class="side-item" ${options.query ? `data-query="${escapeHtml(options.query(item))}"` : ""}>
        <strong>${escapeHtml(title)}</strong>
        <span>${escapeHtml(body)}</span>
      </button>`;
    })
    .join("");
}

async function loadMemory() {
  let history = { items: [] };
  let favorites = { items: [] };
  let ingestions = { items: [] };

  try {
    [history, favorites, ingestions] = await Promise.all([
      api("/api/history", { method: "GET", headers: { "X-CineScene-Session": sessionId } }),
      api("/api/favorites", { method: "GET", headers: { "X-CineScene-Session": sessionId } }),
      api("/api/ingestions", { method: "GET", headers: { "X-CineScene-Session": sessionId } }),
    ]);
  } catch (error) {
    console.warn("Could not load memory panes", error);
  }

  renderSideList(els.history, history.items, {
    empty: "Searches will appear here.",
    title: (item) => item.query,
    body: (item) => `${item.results.length} results`,
    query: (item) => item.query,
  });

  renderSideList(els.favorites, favorites.items, {
    empty: "Favorites will appear here.",
    title: (item) => item.title,
    body: (item) => `${item.year || "N/A"} · ${(item.genres || []).slice(0, 2).join(", ")}`,
  });

  renderSideList(els.ingestion, ingestions.items, {
    empty: "Detected scene jobs will appear here.",
    title: (item) => item.movie_title,
    body: (item) => `${item.scene_count} scenes`,
  });
}

async function saveFavorite(movie) {
  await api("/api/favorites", {
    method: "POST",
    body: JSON.stringify({ movie }),
  });
  await loadMemory();
}

async function sendFeedback(movie, signal) {
  await api("/api/feedback", {
    method: "POST",
    body: JSON.stringify({
      query: state.query || els.query.value,
      movie_title: movie.title,
      signal,
    }),
  });
}

async function ingestVideo(event) {
  event.preventDefault();
  const file = els.video.files[0];
  if (!file) return;

  setBusy(true);
  const form = new FormData();
  form.append("file", file);
  const title = els.movieTitle.value.trim();
  const url = title ? `/api/ingest/video?movie_title=${encodeURIComponent(title)}` : "/api/ingest/video";

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "X-CineScene-Session": sessionId },
      body: form,
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || response.statusText);
    await loadMemory();
  } catch (error) {
    els.ingestion.innerHTML = `<div class="side-item">${escapeHtml(error.message)}</div>`;
  } finally {
    setBusy(false);
  }
}

async function crawlOfflineFolder(event) {
  event.preventDefault();
  const root = els.crawlPath.value.trim();
  if (!root) return;

  setBusy(true);
  try {
    const payload = await api("/api/crawl/offline-async", {
      method: "POST",
      body: JSON.stringify({
        root,
        title_prefix: els.crawlTitle.value.trim(),
        min_scene_sec: Number(els.crawlMinScene.value || 6),
        threshold: Number(els.crawlThreshold.value || 0.4),
        sample_fps: Number(els.crawlFps.value || 1),
        update_catalog: els.crawlCatalog.checked,
      }),
    });
    await pollJob(payload.job, async (result) => {
      const report = result.report || {};
      els.ingestion.innerHTML = `<div class="side-item">
        <strong>Crawl complete</strong>
        <span>${escapeHtml(report.videos_found)} videos found</span>
        <span>${escapeHtml(report.videos_processed)} videos processed</span>
        <span>${escapeHtml(report.documents_created)} searchable documents created</span>
        <span>${escapeHtml(report.combined_catalog)}</span>
      </div>`;
      updateCrawlerDashboard(result.crawler || {});
      setCrawlerNotice("Crawl complete. Rebuild the search index to make these scenes searchable.", "ok");
      await loadMemory();
      await loadCrawlerStatus();
    });
  } catch (error) {
    els.ingestion.innerHTML = `<div class="side-item">${escapeHtml(error.message)}</div>`;
    setCrawlerNotice(error.message, "error");
  } finally {
    setBusy(false);
  }
}

document.querySelectorAll(".tabs button").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".tabs button").forEach((item) => item.classList.remove("active"));
    document.querySelectorAll(".tab-view").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    document.getElementById(`${button.dataset.tab}Tab`).classList.add("active");
  });
});

document.querySelectorAll("[data-query]").forEach((button) => {
  button.addEventListener("click", () => {
    els.query.value = button.dataset.query;
    runSearch(button.dataset.query);
  });
});

els.results.addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-action]");
  if (!button) return;
  const movie = state.results[Number(button.dataset.index)];
  if (!movie) return;

  if (button.dataset.action === "favorite") {
    await saveFavorite(movie);
    button.textContent = "Saved";
  } else {
    const signal = button.dataset.action === "love" ? "relevant" : "not_relevant";
    await sendFeedback(movie, signal);
    button.textContent = "Sent";
  }
});

els.history.addEventListener("click", (event) => {
  const item = event.target.closest("[data-query]");
  if (!item) return;
  els.query.value = item.dataset.query;
  runSearch(item.dataset.query);
});

els.search.addEventListener("click", () => runSearch());
els.reload.addEventListener("click", reloadEngine);
els.rebuildIndex.addEventListener("click", rebuildSearchIndex);
els.refreshCrawler.addEventListener("click", () => loadCrawlerStatus().catch((error) => setCrawlerNotice(error.message, "error")));
els.sampleCrawlerPath.addEventListener("click", () => {
  els.crawlPath.value = benchmarkFolderPath();
  els.crawlTitle.value = "Benchmark Show";
  els.crawlMinScene.value = "2";
  els.crawlThreshold.value = "0.2";
  els.crawlFps.value = "2";
  setCrawlerNotice("Benchmark folder loaded. Run Crawl Folder, then Rebuild Search Index.", "info");
});
els.clear.addEventListener("click", () => {
  els.query.value = "";
  els.results.className = "results-list empty-state";
  els.results.textContent = "Describe a scene, mood, plot, or cinematic feeling.";
  els.resultCount.textContent = "No query yet";
});
els.ingestForm.addEventListener("submit", ingestVideo);
els.crawlForm.addEventListener("submit", crawlOfflineFolder);

els.query.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    runSearch();
  }
});

loadHealth();
loadMemory();
loadCrawlerStatus().catch((error) => console.warn("Crawler status failed", error));
