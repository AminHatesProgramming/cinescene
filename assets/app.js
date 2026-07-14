const state = {
  catalog: [],
  history: JSON.parse(localStorage.getItem("cinescene-pwa-history") || "[]"),
  favorites: JSON.parse(localStorage.getItem("cinescene-pwa-favorites") || "[]"),
  installPrompt: null,
  selectedVideo: null,
  selectedVideoUrl: "",
};

const ids = [
  "viewTitle", "mobileMenuButton", "installButton", "statusDot", "sidebarStatus", "sidebarModel",
  "metricCatalog", "metricIndexed", "metricOffline", "metricScenes", "metricLatency", "queryInput",
  "topKInput", "sourceFilter", "searchButton", "clearButton", "resultMeta", "results", "statusPill",
  "sceneTimeline", "historyList", "favoriteList", "videoDropZone", "videoInput", "videoDropTitle",
  "videoDropMeta", "chooseVideoButton", "selectedVideo", "removeVideoButton", "subtitleInput",
  "previewVideoButton", "playerDialog", "scenePlayer", "playerTitle", "playerTimecode", "playerCaption",
  "closePlayerButton", "toastRegion",
];
const els = Object.fromEntries(ids.map((id) => [id, document.getElementById(id)]));

const persianExpansions = {
  "کارآگاه": "detective investigation mystery",
  "کارآگاهی": "detective investigation mystery",
  "تنها": "lonely alone isolated",
  "تاریک": "dark shadow night",
  "اتاق": "room interior",
  "سرنخ": "clue mystery investigation",
  "فضا": "space science fiction",
  "فضایی": "space science fiction alien",
  "رویا": "dream surreal mind bending",
  "خواب": "dream surreal",
  "عاشقانه": "romance romantic love",
  "ترسناک": "horror frightening supernatural",
  "تعقیب": "chase pursuit action",
  "باران": "rain rainy",
  "جنگ": "war battle soldiers",
  "دریا": "ocean sea water",
  "کودک": "child children family",
};

function refreshIcons(root = document) {
  if (window.lucide?.createIcons) window.lucide.createIcons({ root });
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function normalizeText(value) {
  return String(value || "")
    .toLowerCase()
    .replaceAll("ي", "ی")
    .replaceAll("ك", "ک")
    .replaceAll("\u200c", " ")
    .replace(/[^\p{L}\p{N}\s-]/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function expandedQuery(query) {
  const normalized = normalizeText(query);
  const expansions = [];
  for (const [term, replacement] of Object.entries(persianExpansions)) {
    if (normalized.includes(term)) expansions.push(replacement);
  }
  return `${normalized} ${expansions.join(" ")}`.trim();
}

function tokenize(value) {
  return [...new Set(expandedQuery(value).split(/\s+/).filter((token) => token.length > 2))];
}

function formatBytes(bytes) {
  const value = Number(bytes || 0);
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

function saveMemory() {
  localStorage.setItem("cinescene-pwa-history", JSON.stringify(state.history.slice(0, 24)));
  localStorage.setItem("cinescene-pwa-favorites", JSON.stringify(state.favorites.slice(0, 36)));
}

function toast(message, type = "info") {
  const item = document.createElement("div");
  item.className = `toast ${type}`;
  item.textContent = message;
  els.toastRegion.appendChild(item);
  window.setTimeout(() => item.remove(), 4200);
}

function setView(view, updateHash = true) {
  const next = ["discover", "scenes", "library", "system"].includes(view) ? view : "discover";
  document.querySelectorAll(".view").forEach((item) => item.classList.toggle("active", item.dataset.view === next));
  document.querySelectorAll(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.viewLink === next));
  const titles = { discover: "Discover", scenes: "Scene Lab", library: "Library", system: "System" };
  els.viewTitle.textContent = titles[next];
  document.body.classList.remove("nav-open");
  if (updateHash) history.replaceState(null, "", `#${next}`);
  if (next === "library") renderMemory();
}

function setStatus(message, ok = true) {
  els.statusPill.textContent = message;
  els.statusPill.className = `status-pill ${ok ? "ok" : "warn"}`;
  els.statusDot.className = `status-dot ${ok ? "ok" : "error"}`;
  els.sidebarStatus.textContent = message;
}

function sourceLabel(item) {
  if (item.source === "offline_video_ingestion") return "Scene index";
  if (item.media_type === "series") return "Series catalog";
  return "Movie catalog";
}

function mediaLabel(item) {
  if (item.media_type === "series" && item.season && item.episode) {
    return `Series S${String(item.season).padStart(2, "0")}E${String(item.episode).padStart(2, "0")}`;
  }
  return item.media_type === "series" ? "Series" : "Movie";
}

function sceneText(scene) {
  return normalizeText([
    scene.visual_caption,
    scene.transcript,
    ...(scene.mood_tags || []),
    ...(scene.keywords || []),
    ...(scene.visual_tags || []),
  ].join(" "));
}

function scoreText(text, terms, weight = 1) {
  let score = 0;
  for (const term of terms) {
    if (text.includes(term)) score += weight;
  }
  return score;
}

function bestScene(item, terms) {
  let best = null;
  let bestScore = 0;
  for (const scene of item.scene_timeline || []) {
    const transcript = normalizeText(scene.transcript);
    const visual = normalizeText(`${scene.visual_caption || ""} ${(scene.visual_tags || []).join(" ")}`);
    const tags = normalizeText(`${(scene.keywords || []).join(" ")} ${(scene.mood_tags || []).join(" ")}`);
    const score = scoreText(transcript, terms, 3.1) + scoreText(visual, terms, 2.6) + scoreText(tags, terms, 1.8);
    if (score > bestScore) {
      best = scene;
      bestScore = score;
    }
  }
  return { scene: best, score: bestScore };
}

function scoreItem(item, query) {
  const terms = tokenize(query);
  if (!terms.length) return { score: 0, scene: null };
  const title = normalizeText(item.title);
  const overview = normalizeText(item.overview);
  const genres = normalizeText((item.genres || []).join(" "));
  const moods = normalizeText((item.mood || []).join(" "));
  const keywords = normalizeText((item.keywords || []).join(" "));
  const scenes = normalizeText((item.scenes || []).join(" "));
  const match = bestScene(item, terms);
  let score = match.score;
  score += scoreText(title, terms, 4.2);
  score += scoreText(scenes, terms, 2.4);
  score += scoreText(keywords, terms, 2.0);
  score += scoreText(moods, terms, 1.8);
  score += scoreText(genres, terms, 1.5);
  score += scoreText(overview, terms, 1.2);
  score += Math.min(1.8, Number(item.rating || 0) / 5);
  if (item.source === "offline_video_ingestion" && match.score >= 4) score += 5;
  return { score, scene: match.scene };
}

function searchCatalog(query) {
  const source = els.sourceFilter.value;
  const topK = Number(els.topKInput.value || 8);
  return state.catalog
    .filter((item) => {
      if (source === "all") return true;
      if (source === "offline") return item.source === "offline_video_ingestion";
      if (source === "series") return item.media_type === "series";
      if (source === "movies") return item.media_type !== "series";
      return true;
    })
    .map((item) => ({ item, ...scoreItem(item, query) }))
    .filter((entry) => entry.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

function chips(items, limit = 7) {
  return (items || []).slice(0, limit).map((item) => `<span class="chip">${escapeHtml(item)}</span>`).join("");
}

function renderResults(entries, query, elapsedMs) {
  els.metricLatency.textContent = `${elapsedMs.toFixed(0)}ms`;
  els.resultMeta.textContent = entries.length ? `${entries.length} ranked titles` : "No matches";
  if (!entries.length) {
    els.results.className = "results-grid empty-results";
    els.results.innerHTML = `<div class="empty-visual"><i data-lucide="circle-slash-2"></i></div><strong>No close match in the public sample.</strong><span>Try a dialogue fragment, location, action, or mood.</span>`;
    refreshIcons(els.results);
    return;
  }

  els.results.className = "results-grid";
  els.results.innerHTML = entries.map(({ item, score, scene }, index) => {
    const image = scene?.keyframe || item.poster;
    const media = image
      ? `<div class="result-media"><img src="./${escapeHtml(image)}" alt="Keyframe from ${escapeHtml(item.title)}" loading="lazy" />`
      : `<div class="result-media no-image"><span class="placeholder-number">${String(index + 1).padStart(2, "0")}</span>`;
    const sceneCopy = scene?.transcript || scene?.visual_caption || (item.scenes || [])[0] || "No scene transcript is available in the public sample.";
    const tags = [...(scene?.mood_tags || []), ...(scene?.visual_tags || []), ...(item.genres || []), ...(item.mood || [])];
    return `<article class="result-card">
      ${media}<span class="result-rank">Rank ${index + 1}</span><span class="result-source">${escapeHtml(sourceLabel(item))}</span></div>
      <div class="result-body">
        <div class="result-title-row"><div><h4>${escapeHtml(item.title)}</h4><small>${escapeHtml(mediaLabel(item))} &middot; ${escapeHtml(item.year || "N/A")}${item.director && item.director !== "Unknown" ? ` &middot; ${escapeHtml(item.director)}` : ""}</small></div><span class="score-ring" title="Showcase retrieval score">${score.toFixed(1)}</span></div>
        <p class="result-overview">${escapeHtml(item.overview || "Scene-level offline media document.")}</p>
        ${scene ? `<div class="matched-scene"><div class="scene-label"><span>Best matching scene</span><time>${timecode(scene.start_sec)} - ${timecode(scene.end_sec)}</time></div><p dir="auto">${escapeHtml(sceneCopy)}</p></div>` : ""}
        <div class="chip-row">${chips(tags)}</div>
        <div class="result-actions">
          <button class="button secondary" type="button" data-similar="${escapeHtml(item.title)} ${(item.mood || []).map(escapeHtml).join(" ")}"><i data-lucide="scan-search"></i>Find similar</button>
          <button class="icon-button ghost" type="button" data-favorite="${escapeHtml(item.id)}" title="Save" aria-label="Save result"><i data-lucide="bookmark"></i></button>
        </div>
      </div>
    </article>`;
  }).join("");
  refreshIcons(els.results);

  state.history = [{ query, at: new Date().toLocaleString(), count: entries.length }, ...state.history.filter((entry) => entry.query !== query)].slice(0, 24);
  saveMemory();
}

function runSearch(queryOverride) {
  const query = String(queryOverride || els.queryInput.value).trim();
  if (!query) {
    toast("Describe a scene first.", "error");
    els.queryInput.focus();
    return;
  }
  setView("discover");
  els.queryInput.value = query;
  els.resultMeta.textContent = "Searching scene memory";
  const start = performance.now();
  const entries = searchCatalog(query);
  renderResults(entries, query, performance.now() - start);
}

function renderMemory() {
  els.historyList.innerHTML = state.history.length
    ? state.history.map((item) => `<button class="library-item" type="button" data-query="${escapeHtml(item.query)}"><strong>${escapeHtml(item.query)}</strong><span>${escapeHtml(item.at)} &middot; ${item.count} results</span></button>`).join("")
    : `<div class="empty-inline">Searches made on this device will appear here.</div>`;
  els.favoriteList.innerHTML = state.favorites.length
    ? state.favorites.map((item) => `<button class="library-item" type="button" data-query="${escapeHtml(item.title)}"><strong>${escapeHtml(item.title)}</strong><span>${escapeHtml(sourceLabel(item))} &middot; ${escapeHtml(item.year)}</span></button>`).join("")
    : `<div class="empty-inline">Saved titles will appear here.</div>`;
  refreshIcons();
}

function renderTimeline() {
  const offline = state.catalog.filter((item) => item.source === "offline_video_ingestion");
  if (!offline.length) {
    els.sceneTimeline.className = "media-timeline empty-inline";
    els.sceneTimeline.textContent = "No processed scene records are bundled in this public build.";
    return;
  }
  els.sceneTimeline.className = "media-timeline";
  els.sceneTimeline.innerHTML = offline.map((item) => {
    const scenes = (item.scene_timeline || []).slice(0, 12).map((scene) => `<article class="scene-tile">
      ${scene.keyframe ? `<img src="./${escapeHtml(scene.keyframe)}" alt="${escapeHtml(item.title)} scene" loading="lazy" />` : `<div class="frame-placeholder"><i data-lucide="image-off"></i></div>`}
      <div><strong>${timecode(scene.start_sec)} - ${timecode(scene.end_sec)}</strong><p dir="auto">${escapeHtml(scene.transcript || scene.visual_caption || "Visual scene record")}</p></div>
    </article>`).join("");
    return `<section class="timeline-media"><div class="timeline-media-header"><h4>${escapeHtml(item.title)}</h4><span>${Number(item.scene_count || 0).toLocaleString()} detected scenes</span></div><div class="scene-strip">${scenes}</div></section>`;
  }).join("");
  refreshIcons(els.sceneTimeline);
}

function selectVideo(file) {
  if (!file || !file.type.startsWith("video/")) {
    toast("Choose a supported video file.", "error");
    return;
  }
  state.selectedVideo = file;
  els.selectedVideo.hidden = false;
  els.selectedVideo.querySelector("strong").textContent = file.name;
  els.selectedVideo.querySelector("span").textContent = `${formatBytes(file.size)} - remains on this device`;
  els.videoDropTitle.textContent = "Video ready to preview";
  els.videoDropMeta.textContent = file.name;
  els.previewVideoButton.disabled = false;
}

function clearVideo() {
  state.selectedVideo = null;
  els.videoInput.value = "";
  els.selectedVideo.hidden = true;
  els.videoDropTitle.textContent = "Drop a video here";
  els.videoDropMeta.textContent = "MP4, MKV, MOV, AVI, WEBM";
  els.previewVideoButton.disabled = true;
  if (state.selectedVideoUrl) URL.revokeObjectURL(state.selectedVideoUrl);
  state.selectedVideoUrl = "";
}

function previewVideo() {
  if (!state.selectedVideo) return;
  if (state.selectedVideoUrl) URL.revokeObjectURL(state.selectedVideoUrl);
  state.selectedVideoUrl = URL.createObjectURL(state.selectedVideo);
  els.scenePlayer.src = state.selectedVideoUrl;
  els.playerTitle.textContent = state.selectedVideo.name;
  els.playerDialog.showModal();
  els.scenePlayer.play().catch(() => {});
}

async function loadCatalog() {
  try {
    const response = await fetch("./data/catalog.sample.json", { cache: "no-cache" });
    if (!response.ok) throw new Error(`Catalog HTTP ${response.status}`);
    const payload = await response.json();
    state.catalog = payload.items || [];
    const offline = state.catalog.filter((item) => item.source === "offline_video_ingestion");
    const seriesCount = state.catalog.filter((item) => item.media_type === "series").length;
    const sceneCount = offline.reduce((sum, item) => sum + Number(item.scene_count || (item.scene_timeline || []).length), 0);
    els.metricCatalog.textContent = state.catalog.length.toLocaleString();
    els.metricIndexed.textContent = state.catalog.length.toLocaleString();
    els.metricOffline.textContent = seriesCount.toLocaleString();
    els.metricScenes.textContent = sceneCount.toLocaleString();
    els.sidebarModel.textContent = `${sceneCount.toLocaleString()} scene records cached`;
    setStatus("Showcase ready", true);
    renderTimeline();
    renderMemory();
    runSearch("a lonely detective enters a dark room and investigates a mystery");
  } catch (error) {
    setStatus("Catalog unavailable", false);
    els.resultMeta.textContent = "Load error";
    els.results.className = "results-grid empty-results";
    els.results.innerHTML = `<div class="empty-visual"><i data-lucide="triangle-alert"></i></div><strong>Could not load the public catalog.</strong><span>${escapeHtml(error.message)}</span>`;
    refreshIcons(els.results);
  }
}

document.addEventListener("click", (event) => {
  const viewLink = event.target.closest("[data-view-link]");
  const queryLink = event.target.closest("[data-query]");
  const similar = event.target.closest("[data-similar]");
  const favorite = event.target.closest("[data-favorite]");
  if (viewLink) {
    event.preventDefault();
    setView(viewLink.dataset.viewLink);
  }
  if (queryLink) runSearch(queryLink.dataset.query);
  if (similar) runSearch(similar.dataset.similar);
  if (favorite) {
    const item = state.catalog.find((entry) => String(entry.id) === favorite.dataset.favorite);
    if (item && !state.favorites.some((entry) => String(entry.id) === String(item.id))) {
      state.favorites = [item, ...state.favorites].slice(0, 36);
      saveMemory();
      favorite.classList.add("selected");
      toast(`${item.title} saved.`, "success");
    }
  }
});

els.mobileMenuButton.addEventListener("click", () => document.body.classList.toggle("nav-open"));
els.searchButton.addEventListener("click", () => runSearch());
els.clearButton.addEventListener("click", () => {
  els.queryInput.value = "";
  els.queryInput.focus();
});
els.queryInput.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") runSearch();
});

els.chooseVideoButton.addEventListener("click", (event) => {
  event.stopPropagation();
  els.videoInput.click();
});
els.videoDropZone.addEventListener("click", () => els.videoInput.click());
els.videoDropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") els.videoInput.click();
});
els.videoInput.addEventListener("change", () => selectVideo(els.videoInput.files?.[0]));
els.videoDropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  els.videoDropZone.classList.add("dragging");
});
els.videoDropZone.addEventListener("dragleave", () => els.videoDropZone.classList.remove("dragging"));
els.videoDropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  els.videoDropZone.classList.remove("dragging");
  selectVideo(event.dataTransfer.files?.[0]);
});
els.removeVideoButton.addEventListener("click", clearVideo);
els.previewVideoButton.addEventListener("click", previewVideo);
els.closePlayerButton.addEventListener("click", () => {
  els.scenePlayer.pause();
  els.playerDialog.close();
});

window.addEventListener("hashchange", () => setView(location.hash.slice(1), false));
window.addEventListener("beforeinstallprompt", (event) => {
  event.preventDefault();
  state.installPrompt = event;
  els.installButton.hidden = false;
});
els.installButton.addEventListener("click", async () => {
  if (!state.installPrompt) return;
  state.installPrompt.prompt();
  await state.installPrompt.userChoice;
  state.installPrompt = null;
  els.installButton.hidden = true;
});

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => navigator.serviceWorker.register("./sw.js").catch(console.warn));
}

refreshIcons();
setView(location.hash.slice(1) || "discover", false);
loadCatalog();
