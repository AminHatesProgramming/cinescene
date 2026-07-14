const state = {
  catalog: [],
  query: "",
  installPrompt: null,
  history: JSON.parse(localStorage.getItem("cinescene-pwa-history") || "[]"),
  favorites: JSON.parse(localStorage.getItem("cinescene-pwa-favorites") || "[]"),
};

const els = {
  status: document.getElementById("statusPill"),
  install: document.getElementById("installButton"),
  catalog: document.getElementById("metricCatalog"),
  offline: document.getElementById("metricOffline"),
  mode: document.getElementById("metricMode"),
  latency: document.getElementById("metricLatency"),
  query: document.getElementById("queryInput"),
  topK: document.getElementById("topKInput"),
  source: document.getElementById("sourceFilter"),
  search: document.getElementById("searchButton"),
  clear: document.getElementById("clearButton"),
  results: document.getElementById("results"),
  resultMeta: document.getElementById("resultMeta"),
  history: document.getElementById("historyList"),
  favorites: document.getElementById("favoriteList"),
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function tokenize(text) {
  return String(text || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, " ")
    .split(/\s+/)
    .filter((token) => token.length > 2);
}

function unique(values) {
  return [...new Set(values.filter(Boolean))];
}

function saveMemory() {
  localStorage.setItem("cinescene-pwa-history", JSON.stringify(state.history.slice(0, 20)));
  localStorage.setItem("cinescene-pwa-favorites", JSON.stringify(state.favorites.slice(0, 30)));
}

function setStatus(message, mode = "ok") {
  els.status.textContent = message;
  els.status.className = `status-pill ${mode}`;
}

function mediaLabel(item) {
  if (item.media_type === "series" && item.season && item.episode) {
    return `Series S${String(item.season).padStart(2, "0")}E${String(item.episode).padStart(2, "0")}`;
  }
  return item.media_type === "series" ? "Series" : "Movie";
}

function chips(items, limit = 6) {
  return (items || [])
    .slice(0, limit)
    .map((item) => `<span class="chip">${escapeHtml(item)}</span>`)
    .join("");
}

function timecode(seconds) {
  const value = Number(seconds || 0);
  const minutes = Math.floor(value / 60);
  const secs = Math.floor(value % 60);
  return `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

function sourceLabel(source) {
  if (source === "offline_video_ingestion") return "Offline crawler";
  if (source === "tmdb_enriched") return "TMDB enriched";
  return source || "Catalog";
}

function scoreItem(item, query) {
  const terms = unique(tokenize(query));
  if (!terms.length) return 0;

  const text = item.search_text || "";
  const title = String(item.title || "").toLowerCase();
  const overview = String(item.overview || "").toLowerCase();
  const scenes = (item.scenes || []).join(" ").toLowerCase();
  const mood = (item.mood || []).join(" ").toLowerCase();
  const genres = (item.genres || []).join(" ").toLowerCase();
  const keywords = (item.keywords || []).join(" ").toLowerCase();
  const visualTags = (item.visual_tags || []).join(" ").toLowerCase();
  const timeline = (item.scene_timeline || [])
    .map((scene) => `${scene.visual_caption || ""} ${scene.transcript || ""} ${(scene.visual_tags || []).join(" ")} ${(scene.keywords || []).join(" ")}`)
    .join(" ")
    .toLowerCase();
  let score = 0;

  for (const term of terms) {
    if (text.includes(term)) score += 1.3;
    if (title.includes(term)) score += 3.2;
    if (overview.includes(term)) score += 1.2;
    if (scenes.includes(term)) score += 2.3;
    if (mood.includes(term)) score += 1.9;
    if (genres.includes(term)) score += 1.5;
    if (keywords.includes(term)) score += 2.0;
    if (visualTags.includes(term)) score += 1.7;
    if (timeline.includes(term)) score += 2.6;
  }

  const queryLower = query.toLowerCase();
  if (queryLower.length > 12 && text.includes(queryLower.slice(0, 80))) score += 8;
  if (item.source === "offline_video_ingestion" && terms.filter((term) => scenes.includes(term)).length >= 2) {
    score += 7.5;
  }
  score += Math.min(2.5, Number(item.rating || 0) / 4);
  score += Math.min(2.0, Math.log10(Number(item.popularity || 0) + 1));
  return score;
}

function searchCatalog(query) {
  const source = els.source.value;
  const topK = Number(els.topK.value || 8);
  return state.catalog
    .filter((item) => source === "all" || item.source === source)
    .map((item) => ({ item, score: scoreItem(item, query) }))
    .filter((entry) => entry.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

function renderResults(entries, query, elapsedMs) {
  els.latency.textContent = `${elapsedMs.toFixed(0)}ms`;
  els.resultMeta.textContent = entries.length ? `${entries.length} matches` : "No matches";

  if (!entries.length) {
    els.results.className = "results-list empty-state";
    els.results.textContent = "No matching title was found in the PWA sample catalog.";
    return;
  }

  els.results.className = "results-list";
  els.results.innerHTML = entries
    .map(({ item, score }, index) => {
      const poster = item.poster ? `<img class="poster" src="./${escapeHtml(item.poster)}" alt="${escapeHtml(item.title)} keyframe" />` : "";
      const timeline = (item.scene_timeline || [])
        .slice(0, 4)
        .map((scene) => {
          const keyframe = scene.keyframe ? `<img class="mini-frame" src="./${escapeHtml(scene.keyframe)}" alt="">` : "";
          return `<li>${keyframe}<span><strong>${escapeHtml(timecode(scene.start_sec))}</strong> ${escapeHtml(scene.transcript || scene.visual_caption || "")}</span></li>`;
        })
        .join("");
      const scenes = timeline || (item.scenes || [])
        .slice(0, 4)
        .map((scene) => `<li><span>${escapeHtml(scene)}</span></li>`)
        .join("");
      return `<article class="movie-card ${item.poster ? "has-poster" : ""}">
        <div class="movie-card-inner">
          ${poster}
          <div>
            <h2 class="movie-title">${index + 1}. ${escapeHtml(item.title)} <span>${escapeHtml(item.year)}</span></h2>
            <div class="meta-row">
              <span class="chip source-chip">${escapeHtml(sourceLabel(item.source))}</span>
              <span class="chip">${escapeHtml(mediaLabel(item))}</span>
              <span class="chip">${escapeHtml(item.director || "Unknown director")}</span>
              <span class="chip">Rating ${escapeHtml(item.rating || 0)}</span>
              <span class="chip score-chip">Score ${score.toFixed(2)}</span>
            </div>
            <div class="tag-row">${chips(item.genres, 4)}${chips(item.mood, 5)}${chips(item.keywords, 4)}${chips(item.visual_tags, 4)}</div>
            <p class="overview">${escapeHtml(item.overview || "No overview available in the sample catalog.")}</p>
            ${scenes ? `<ol class="scene-list">${scenes}</ol>` : ""}
            <div class="movie-actions">
              <button data-fav="${escapeHtml(item.id)}" type="button">Favorite</button>
              <button data-query="${escapeHtml(item.title)} ${escapeHtml((item.mood || []).join(" "))}" type="button">Similar</button>
            </div>
          </div>
        </div>
      </article>`;
    })
    .join("");

  state.query = query;
  state.history = [{ query, at: new Date().toLocaleString(), count: entries.length }, ...state.history.filter((item) => item.query !== query)].slice(0, 20);
  saveMemory();
  renderMemory();
}

function runSearch(queryOverride) {
  const query = (queryOverride || els.query.value).trim();
  if (!query) return;
  els.query.value = query;
  const start = performance.now();
  const entries = searchCatalog(query);
  renderResults(entries, query, performance.now() - start);
}

function renderMemory() {
  els.history.innerHTML = state.history.length
    ? state.history
        .map((item) => `<button class="side-item" data-query="${escapeHtml(item.query)}" type="button">
          <strong>${escapeHtml(item.query)}</strong>
          <span>${escapeHtml(item.count)} results - ${escapeHtml(item.at)}</span>
        </button>`)
        .join("")
    : `<div class="side-item"><span>Searches will appear here.</span></div>`;

  els.favorites.innerHTML = state.favorites.length
    ? state.favorites
        .map((item) => `<button class="side-item" data-query="${escapeHtml(item.title)}" type="button">
          <strong>${escapeHtml(item.title)}</strong>
          <span>${escapeHtml(sourceLabel(item.source))} - ${escapeHtml(item.year)}</span>
        </button>`)
        .join("")
    : `<div class="side-item"><span>Favorites will appear here.</span></div>`;
}

async function loadCatalog() {
  try {
    const response = await fetch("./data/catalog.sample.json", { cache: "no-cache" });
    if (!response.ok) throw new Error(`Catalog HTTP ${response.status}`);
    const payload = await response.json();
    state.catalog = payload.items || [];
    els.catalog.textContent = state.catalog.length;
    els.offline.textContent = state.catalog.filter((item) => item.source === "offline_video_ingestion").length;
    setStatus("Ready offline", "ok");
    renderMemory();
    runSearch("suspenseful detective mystery in a dark room");
  } catch (error) {
    console.error(error);
    setStatus(error.message, "warn");
    els.results.className = "results-list empty-state";
    els.results.textContent = "Could not load the PWA sample catalog.";
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
  button.addEventListener("click", () => runSearch(button.dataset.query));
});

els.search.addEventListener("click", () => runSearch());
els.clear.addEventListener("click", () => {
  els.query.value = "";
  els.resultMeta.textContent = "Ready";
  els.results.className = "results-list empty-state";
  els.results.textContent = "Search by plot, mood, scene, character situation, or cinematic feeling.";
});
els.query.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") runSearch();
});
els.results.addEventListener("click", (event) => {
  const favButton = event.target.closest("[data-fav]");
  const queryButton = event.target.closest("[data-query]");
  if (favButton) {
    const item = state.catalog.find((entry) => String(entry.id) === favButton.dataset.fav);
    if (item && !state.favorites.some((entry) => entry.id === item.id)) {
      state.favorites = [item, ...state.favorites].slice(0, 30);
      favButton.textContent = "Saved";
      saveMemory();
      renderMemory();
    }
  }
  if (queryButton) runSearch(queryButton.dataset.query);
});
els.history.addEventListener("click", (event) => {
  const item = event.target.closest("[data-query]");
  if (item) runSearch(item.dataset.query);
});
els.favorites.addEventListener("click", (event) => {
  const item = event.target.closest("[data-query]");
  if (item) runSearch(item.dataset.query);
});

window.addEventListener("beforeinstallprompt", (event) => {
  event.preventDefault();
  state.installPrompt = event;
  els.install.hidden = false;
});
els.install.addEventListener("click", async () => {
  if (!state.installPrompt) return;
  state.installPrompt.prompt();
  await state.installPrompt.userChoice;
  state.installPrompt = null;
  els.install.hidden = true;
});

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("./sw.js").catch((error) => console.warn("Service worker registration failed", error));
  });
}

loadCatalog();
