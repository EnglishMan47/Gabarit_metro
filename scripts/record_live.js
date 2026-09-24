// Records the live view (ROS 2 node in Docker, `gabarit demo <bag>`) for the demo video.
//   node scripts/record_live.js [outDir] [seconds]
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright-core');

const OUT = path.resolve(process.argv[2] || 'artifacts/release');
const SECONDS = Number(process.argv[3] || 30);
const URL = process.env.GABARIT_URL || 'http://127.0.0.1:8080';
const CHROME = process.env.CHROME || 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch({ headless: true, executablePath: CHROME,
    args: ['--use-gl=angle', '--enable-webgl', '--ignore-gpu-blocklist'] });
  const ctx = await browser.newContext({ viewport: { width: 1600, height: 900 },
    recordVideo: { dir: OUT, size: { width: 1600, height: 900 } } });
  const page = await ctx.newPage();
  await page.goto(URL);
  await page.waitForFunction(() => document.querySelector('#hudState').textContent !== 'ОЖИДАНИЕ', null, { timeout: 120000 });
  await page.waitForTimeout(4000);
  await page.screenshot({ path: path.join(OUT, 'live-start.png') });
  await page.click('[data-cam="target"]');
  await page.waitForTimeout(SECONDS * 400);
  await page.screenshot({ path: path.join(OUT, 'live-obstacle.png') });
  await page.click('[data-cam="chase"]');
  await page.waitForTimeout(SECONDS * 600);
  await ctx.close();
  await browser.close();
  const vids = fs.readdirSync(OUT).filter((f) => f.endsWith('.webm') && f !== 'demo.webm').map((f) => path.join(OUT, f));
  vids.sort((a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs);
  if (vids.length) { fs.renameSync(vids[0], path.join(OUT, 'demo-live-ros2.webm')); console.log('video demo-live-ros2.webm'); }
})();
