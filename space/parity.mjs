// Run the browser pipeline (static/app.js -> pipeline.js, onnxruntime-web wasm) on images,
// then smoke-test the UI: upload, camera permission denied, and a fake live camera.
// usage: node parity.mjs <site-url> <site-dir> <names.json> <out.json>
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { chromium } from '@playwright/test';

const [url, siteDir, namesFile, outFile] = process.argv.slice(2);
const names = JSON.parse(readFileSync(namesFile, 'utf8'));
const ui = [];
const check = (name, ok, detail) => ui.push({ name, ok: Boolean(ok), detail: String(detail) });
const text = (page, sel) => page.locator(sel).textContent();

async function open(context) {
  const page = await context.newPage();
  page.on('pageerror', (e) => console.error('pageerror:', e.message));
  await page.goto(url);
  await page.waitForFunction(() => window.aslReady === true, null, { timeout: 120_000 });
  return page;
}

// 1) pipeline parity, 2) upload tab - default context
// Headless Edge (ships with Windows): Playwright's headless shell cannot open cameras at all.
const browser = await chromium.launch({ channel: 'msedge', args: ['--use-fake-device-for-media-stream'] });
const plain = await browser.newContext();
const page = await open(plain);
check('model loads, status says ready', (await text(page, '#status')).startsWith('Ready'), await text(page, '#status'));
const out = {};
for (const name of names) out[name] = await page.evaluate((n) => window.aslParity(`parity/${n}`), name);

await page.getByRole('tab', { name: 'Upload a photo' }).click();
await page.setInputFiles('#file', join(siteDir, 'parity', names[0]));
await page.waitForFunction(() => document.querySelector('#verdict').textContent !== '-', null, { timeout: 30_000 });
const upload = await text(page, '#verdict');
check(`upload ${names[0]} -> ${names[0][0]}`, upload.startsWith(`${names[0][0]} (`), upload);

// 3) camera permission denied (no grant in this context)
await page.getByRole('tab', { name: 'Webcam (live)' }).click();
await page.click('#start');
await page.waitForFunction(() => document.querySelector('#status').dataset.kind === 'error', null, { timeout: 15_000 });
const denied = await text(page, '#status');
check('denied camera explains itself', /denied/i.test(denied), denied);

// 4) fake camera granted: the live loop runs and, with no hand, never guesses
const granted = await browser.newContext({ permissions: ['camera'] });
const live = await open(granted);
await live.click('#start');
await live.waitForFunction(() => document.querySelector('#status').textContent.startsWith('Camera on'), null, { timeout: 15_000 });
const counted = await live.evaluate(async () => {
  // show() resets the canvas width every processed frame; count those over 3 s
  const canvas = document.querySelector('#camview');
  let frames = 0;
  const obs = new MutationObserver(() => frames++);
  obs.observe(canvas, { attributes: true, attributeFilter: ['width'] });
  await new Promise((r) => setTimeout(r, 3000));
  obs.disconnect();
  return { verdict: document.querySelector('#verdict').textContent, fps: frames / 3 };
});
check('live loop throttled to about 5 fps', counted.fps >= 3 && counted.fps <= 6, `${counted.fps.toFixed(1)} fps`);
check('no hand -> no guess', counted.verdict.startsWith('No hand found'), counted.verdict);

await browser.close();
writeFileSync(outFile, JSON.stringify({ probs: out, ui }));
