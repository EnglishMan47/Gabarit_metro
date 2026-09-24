// Records the demo video and screenshots of the web interface (needs the
// server running on :8000 and `npm i playwright-core`).
//   node scripts/record_demo.js [outDir]
//   GABARIT_URL=http://127.0.0.1:8010 NO_VIDEO=1 node scripts/record_demo.js artifacts/release_v3
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright-core');

const OUT = path.resolve(process.argv[2] || 'artifacts/release');
const URL = process.env.GABARIT_URL || 'http://127.0.0.1:8000';
const CHROME = process.env.CHROME || 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch({ headless: true, executablePath: CHROME,
    args: ['--use-gl=angle', '--enable-webgl', '--ignore-gpu-blocklist'] });
  // NO_VIDEO=1: screenshots only (video recording needs Playwright's ffmpeg)
  const video = process.env.NO_VIDEO ? {} : { recordVideo: { dir: OUT, size: { width: 1600, height: 900 } } };
  const ctx = await browser.newContext({ viewport: { width: 1600, height: 900 }, deviceScaleFactor: 1, ...video });
  const page = await ctx.newPage();
  page.on('pageerror', (e) => console.log('pageerror', e.message));
  const wait = (ms) => page.waitForTimeout(ms);
  const shot = (name) => page.screenshot({ path: path.join(OUT, name) });
  const status = () => page.locator('#statusLabel').textContent();
  const frameNo = async () => { const t = await page.locator('#frameLabel').textContent(); return Number((t.match(/кадр (\d+)/) || [0, 0])[1]); };

  await page.goto(URL);
  await page.waitForFunction(() => document.querySelector('#bagSelect').options.length > 0);
  await page.waitForFunction(() => document.querySelector('#frameLabel').textContent.includes('кадр'));
  await wait(2500);
  await shot('ui-start.png');

  // 1. positive ride: a person crosses the track, an object lies on the rail
  await page.click('#btnPlay');
  await page.waitForFunction(() => document.querySelector('#statusLabel').textContent.includes('ПРЕПЯТСТВИЕ'), null, { timeout: 60000 });
  await wait(1500);
  await shot('ui-obstacle.png');
  await wait(2500);
  if (await page.locator('#toast').isVisible()) await page.click('#toastConfirm');
  await page.click('[data-cam="target"]');
  await wait(4000);
  await shot('ui-target.png');
  await page.click('[data-cam="top"]');
  await wait(3000);
  await page.click('[data-cam="chase"]');
  await page.waitForFunction(() => [...document.querySelectorAll('.ob-info')].some((e) => e.textContent.includes('на рельсе')), null, { timeout: 90000 }).catch(() => {});
  await wait(2000);
  await shot('ui-rail-object.png');
  if (await page.locator('#toast').isVisible()) await page.click('#toastConfirm');
  await wait(2000);
  await page.click('#btnPlay');

  // 2. moving train, synthetic person placed 190 m ahead
  await page.selectOption('#bagSelect', { index: 4 });
  await page.waitForFunction(() => document.querySelector('#frameLabel').textContent.includes('кадр 1 '), null, { timeout: 60000 });
  await wait(1500);
  await page.click('[data-cam="cab"]');
  await page.fill('#synDist', '190');
  await page.dispatchEvent('#synDist', 'input');
  await page.click('#synAdd');
  await wait(800);
  await page.click('#btnPlay');
  await page.waitForFunction(() => document.querySelector('#statusLabel').textContent.includes('ПРЕПЯТСТВИЕ'), null, { timeout: 90000 });
  await wait(1200);
  await shot('ui-synthetic-far.png');
  if (await page.locator('#toast').isVisible()) await page.click('#toastConfirm');
  await wait(5000);
  await page.click('[data-cam="chase"]');
  await wait(5000);
  await shot('ui-synthetic-near.png');
  await page.click('#btnPlay');
  await page.click('button[data-tab="perf"]');
  await wait(1500);
  await shot('ui-perf.png');
  await page.click('button[data-tab="log"]');
  await wait(1500);
  await shot('ui-journal.png');
  console.log('status at end:', await status(), 'frame', await frameNo());
  await ctx.close();
  await browser.close();
  const vids = fs.readdirSync(OUT).filter((f) => f.endsWith('.webm')).map((f) => path.join(OUT, f));
  vids.sort((a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs);
  if (vids.length) { fs.renameSync(vids[0], path.join(OUT, 'demo.webm')); console.log('video', path.join(OUT, 'demo.webm')); }
})();
