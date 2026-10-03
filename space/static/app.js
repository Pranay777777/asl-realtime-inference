// ASL alphabet recognizer - runs entirely in the browser: MediaPipe finds the hand,
// onnxruntime-web (wasm) runs the INT8 model, nothing is uploaded anywhere.
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs';
import { FilesetResolver, HandLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs';
import { boxFromPoints, calibrated, centreBox, modelInput, SIZE, top, verdict } from './pipeline.js';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
ort.env.wasm.numThreads = 1; // static hosting has no cross-origin isolation for threads

const FRAME_MS = 200; // about 5 inferences per second
const HAND_MODEL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

const $ = (id) => document.getElementById(id);
const state = { session: null, labels: null, temperature: 1, threshold: 1, hands: null, stream: null };

function setStatus(text, kind = 'info') {
  const el = $('status');
  el.textContent = text;
  el.dataset.kind = kind;
}

async function loadModel() {
  const [labels, cal] = await Promise.all([
    fetch('labels.json').then((r) => r.json()),
    fetch('calibration.json').then((r) => r.json()),
  ]);
  state.labels = Object.keys(labels).sort((a, b) => a - b).map((k) => labels[k]);
  state.temperature = Number(cal.temperature);
  state.threshold = Number(cal.threshold);
  state.session = await ort.InferenceSession.create('asl_int8.onnx', { executionProviders: ['wasm'] });
}

async function loadHands() {
  const files = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm'
  );
  state.hands = await HandLandmarker.createFromOptions(files, {
    baseOptions: { modelAssetPath: HAND_MODEL },
    runningMode: 'VIDEO',
    numHands: 1,
    minHandDetectionConfidence: 0.5,
  });
}

// RGBA pixels of an image, video frame or bitmap at its own resolution.
function pixelsOf(source, w, h) {
  const c = new OffscreenCanvas(w, h);
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(source, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h).data;
}

async function probsFor(rgba, w, box) {
  const input = new ort.Tensor('float32', modelInput(rgba, w, box), [1, SIZE, SIZE, 3]);
  const name = state.session.inputNames[0];
  const out = await state.session.run({ [name]: input });
  return calibrated(out[state.session.outputNames[0]].data, state.temperature);
}

function handBox(result, w, h) {
  const lm = result?.landmarks?.[0];
  if (!lm) return null;
  return boxFromPoints(lm.map((p) => p.x * w), lm.map((p) => p.y * h), w, h);
}

function show(canvas, source, w, h, box) {
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(source, 0, 0, w, h);
  if (box) {
    ctx.strokeStyle = '#00c878';
    ctx.lineWidth = Math.max(2, Math.round(w / 200));
    ctx.strokeRect(box[0], box[1], box[2] - box[0], box[3] - box[1]);
  }
}

function report(probs, note = '') {
  const v = verdict(probs, state.labels, state.threshold);
  $('verdict').textContent = v.text + note;
  $('verdict').dataset.kind = v.letter ? 'letter' : 'abstain';
  $('top').replaceChildren(
    ...top(probs, state.labels).map(({ label, p }) => {
      const li = document.createElement('li');
      li.innerHTML = `<span>${label}</span><meter min="0" max="1" value="${p}"></meter><span>${(p * 100).toFixed(1)}%</span>`;
      return li;
    })
  );
}

// ---- webcam ----
async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus('This browser cannot open a camera here (it needs HTTPS). Try the upload tab.', 'error');
    return;
  }
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
  } catch (err) {
    const messages = {
      NotAllowedError: 'Camera permission was denied. Allow it in the address bar and press Start again, or use the upload tab.',
      SecurityError: 'Camera permission was denied. Allow it in the address bar and press Start again, or use the upload tab.',
      NotFoundError: 'No camera was found on this device. Use the upload tab instead.',
      OverconstrainedError: 'No suitable camera was found. Use the upload tab instead.',
      NotReadableError: 'The camera is busy in another app. Close it and press Start again.',
      NotSupportedError: 'This browser cannot use a camera on this page. Use the upload tab instead.',
    };
    setStatus(messages[err.name] ?? `Could not open the camera (${err.name}).`, 'error');
    return;
  }
  const video = $('video');
  video.srcObject = state.stream;
  await video.play();
  $('start').hidden = true;
  $('stop').hidden = false;
  setStatus('Camera on. Hold one hand in view and sign a letter.', 'ok');
  loop();
}

function stopCamera() {
  state.stream?.getTracks().forEach((t) => t.stop());
  state.stream = null;
  $('start').hidden = false;
  $('stop').hidden = true;
  setStatus('Camera off.', 'info');
}

let busy = false;
let last = 0;
function loop(now = 0) {
  if (!state.stream) return;
  requestAnimationFrame(loop);
  if (busy || now - last < FRAME_MS) return;
  last = now;
  busy = true;
  frame(now).finally(() => (busy = false));
}

async function frame(now) {
  const video = $('video');
  const w = video.videoWidth, h = video.videoHeight;
  if (!w || !h) return;
  const box = handBox(state.hands.detectForVideo(video, now), w, h);
  show($('camview'), video, w, h, box);
  if (!box) {
    // live frames never guess: an empty scene can still score high on some letter
    $('verdict').textContent = 'No hand found - hold one hand in view';
    $('verdict').dataset.kind = 'abstain';
    $('top').replaceChildren();
    return;
  }
  report(await probsFor(pixelsOf(video, w, h), w, box));
}

// ---- upload ----
async function onUpload(file) {
  if (!file) return;
  const bitmap = await createImageBitmap(file, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
  const w = bitmap.width, h = bitmap.height;
  const rgba = pixelsOf(bitmap, w, h);
  await state.hands.setOptions({ runningMode: 'IMAGE' });
  let box = handBox(state.hands.detect(bitmap), w, h);
  await state.hands.setOptions({ runningMode: 'VIDEO' });
  let note = '';
  if (!box) {
    // MediaPipe misses some close, dark crops (like the training photos): read the centre square
    box = centreBox(w, h);
    note = ' - no hand detected, read the centre of the image';
  }
  show($('upview'), bitmap, w, h, box);
  report(await probsFor(rgba, w, box), note);
}

// ---- tabs ----
function selectTab(name) {
  for (const tab of document.querySelectorAll('[role=tab]')) {
    const on = tab.dataset.tab === name;
    tab.setAttribute('aria-selected', String(on));
    $(tab.getAttribute('aria-controls')).hidden = !on;
  }
  if (name !== 'webcam' && state.stream) stopCamera();
}

// Parity hook (space/parity.py): the exact pipeline on an image URL with the centre box.
async function parity(url) {
  const bitmap = await createImageBitmap(await (await fetch(url)).blob(), {
    colorSpaceConversion: 'none',
    premultiplyAlpha: 'none',
  });
  const w = bitmap.width, h = bitmap.height;
  const rgba = pixelsOf(bitmap, w, h), box = centreBox(w, h);
  return { probs: await probsFor(rgba, w, box), input: Array.from(modelInput(rgba, w, box)) };
}

async function main() {
  for (const tab of document.querySelectorAll('[role=tab]')) tab.onclick = () => selectTab(tab.dataset.tab);
  $('start').onclick = startCamera;
  $('stop').onclick = stopCamera;
  $('file').onchange = (e) => onUpload(e.target.files[0]).catch((err) => setStatus(`Could not read that image (${err.message}).`, 'error'));
  setStatus('Loading the model (about 3 MB) and the hand detector…', 'busy');
  try {
    await Promise.all([loadModel(), loadHands()]);
  } catch (err) {
    setStatus(`Could not load the model: ${err.message}. Reload the page to try again.`, 'error');
    throw err;
  }
  $('start').disabled = false;
  $('file').disabled = false;
  setStatus('Ready. Press Start camera, or upload a photo of one hand.', 'ok');
  window.aslParity = parity;
  window.aslReady = true;
}

main();
