const { chromium } = require("playwright-core");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const CHROME = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const BASE_URL = process.argv[2] || process.env.CINESCENE_VISUAL_URL || "http://127.0.0.1:8000";
const STATIC_MODE = process.argv[3] === "static" || process.env.CINESCENE_VISUAL_MODE === "static";
const ARTIFACT_PREFIX = STATIC_MODE ? "visual-pwa" : "visual";

async function verifyPage(page, viewportName) {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });

  await page.goto(`${BASE_URL}/#discover`, { waitUntil: "networkidle" });
  await page.waitForFunction(
    (expected) => document.querySelector("#sidebarStatus")?.textContent === expected,
    STATIC_MODE ? "Showcase ready" : "Engine online",
    { timeout: 60_000 },
  );
  await page.screenshot({ path: path.join(ROOT, "artifacts", `${ARTIFACT_PREFIX}-${viewportName}-discover.png`), fullPage: true });

  if (viewportName === "desktop") {
    await page.fill("#queryInput", "lonely detective enters a dark room");
    await page.click("#searchButton");
    await page.waitForSelector(".result-card", { timeout: 60_000 });
    await page.screenshot({ path: path.join(ROOT, "artifacts", `${ARTIFACT_PREFIX}-desktop-results.png`), fullPage: true });

    await page.click(`[data-view-link="${STATIC_MODE ? "scenes" : "ingest"}"]`);
    await page.waitForSelector(STATIC_MODE ? "#scenesView.active" : "#ingestView.active");
    await page.screenshot({ path: path.join(ROOT, "artifacts", `${ARTIFACT_PREFIX}-desktop-scene-lab.png`), fullPage: true });
  } else {
    await page.click("#mobileMenuButton");
    await page.click(`.nav-item[data-view-link="${STATIC_MODE ? "scenes" : "ingest"}"]`);
    await page.waitForSelector(STATIC_MODE ? "#scenesView.active" : "#ingestView.active");
    await page.screenshot({ path: path.join(ROOT, "artifacts", `${ARTIFACT_PREFIX}-mobile-scene-lab.png`), fullPage: true });
  }

  const overflow = await page.evaluate(() => ({
    body: document.body.scrollWidth > window.innerWidth + 1,
    viewport: window.innerWidth,
    scrollWidth: document.body.scrollWidth,
  }));
  return { viewport: viewportName, errors, overflow };
}

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true });
  try {
    const desktop = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });
    const mobile = await browser.newPage({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 1 });
    const results = [await verifyPage(desktop, "desktop"), await verifyPage(mobile, "mobile")];
    console.log(JSON.stringify(results, null, 2));
    if (results.some((item) => item.errors.length || item.overflow.body)) process.exitCode = 1;
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
