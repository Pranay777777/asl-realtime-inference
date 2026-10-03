// Browser port of space/pipeline.py - keep the two in step (space/parity.py checks them).
// Pure functions only: no DOM, no model loading.

export const SIZE = 160; // model input: 160x160 RGB, NHWC, float32 0-255 (it rescales internally)
export const PAD = 1.4; // crop margin around the hand, as in predict.py
export const MOTION_LETTERS = new Set(['J', 'Z']); // signed with movement

// Square box around hand landmarks (pixels), padded and clipped. Python's int() truncates.
export function boxFromPoints(xs, ys, w, h) {
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  const half = (Math.max(maxX - minX, maxY - minY) * PAD) / 2;
  const x0 = Math.trunc(Math.max(0, cx - half)), y0 = Math.trunc(Math.max(0, cy - half));
  const x1 = Math.trunc(Math.min(w, cx + half)), y1 = Math.trunc(Math.min(h, cy + half));
  return x1 - x0 > 10 && y1 - y0 > 10 ? [x0, y0, x1, y1] : null;
}

// The centre square, as predict.py reads single images.
export function centreBox(w, h) {
  const side = Math.min(h, w);
  const x0 = Math.floor((w - side) / 2), y0 = Math.floor((h - side) / 2);
  return [x0, y0, x0 + side, y0 + side];
}

// OpenCV's saturate_cast<uchar>(float): round half to even, then clamp.
function toUint8(v) {
  let r = Math.round(v);
  if (Math.abs(v % 1) === 0.5) r = 2 * Math.round(v / 2);
  return Math.min(255, Math.max(0, r));
}

// One axis of cv2.INTER_AREA when shrinking: [{src, dst, weight}] (resizeArea_ in OpenCV).
function areaTab(ssize, dsize) {
  const scale = ssize / dsize;
  const tab = [];
  for (let dx = 0; dx < dsize; dx++) {
    const fsx1 = dx * scale, fsx2 = fsx1 + scale;
    const cell = Math.min(scale, ssize - fsx1);
    let sx1 = Math.ceil(fsx1);
    const sx2 = Math.floor(fsx2);
    if (sx1 - fsx1 > 1e-3) tab.push({ src: sx1 - 1, dst: dx, weight: (sx1 - fsx1) / cell });
    for (let sx = sx1; sx < sx2; sx++) tab.push({ src: sx, dst: dx, weight: 1 / cell });
    if (fsx2 - sx2 > 1e-3 && sx2 < ssize) tab.push({ src: sx2, dst: dx, weight: Math.min(Math.min(fsx2 - sx2, 1), cell) / cell });
  }
  return tab;
}

// One axis of cv2.INTER_AREA when enlarging: OpenCV falls back to these linear weights.
function linearTab(ssize, dsize) {
  const scale = ssize / dsize, inv = dsize / ssize;
  const tab = [];
  for (let dx = 0; dx < dsize; dx++) {
    let sx = Math.floor(dx * scale);
    let fx = (dx + 1) - (sx + 1) * inv;
    fx = fx <= 0 ? 0 : fx - Math.floor(fx);
    if (sx < 0) { fx = 0; sx = 0; }
    if (sx >= ssize - 1) { fx = 0; sx = ssize - 1; }
    tab.push({ src: sx, dst: dx, weight: 1 - fx });
    if (fx > 0) tab.push({ src: sx + 1, dst: dx, weight: fx });
  }
  return tab;
}

// Crop rgba (w x h, 4 bytes per pixel) to box and resize to SIZE x SIZE like
// cv2.resize(..., INTER_AREA) on uint8 RGB, then float32 NHWC 0-255.
export function modelInput(rgba, w, box) {
  const [x0, y0, x1, y1] = box;
  const cw = x1 - x0, ch = y1 - y0;
  const xtab = cw >= SIZE ? areaTab(cw, SIZE) : linearTab(cw, SIZE);
  const ytab = ch >= SIZE ? areaTab(ch, SIZE) : linearTab(ch, SIZE);
  // horizontal pass into float rows, then vertical pass
  const rows = new Float64Array(ch * SIZE * 3);
  for (let y = 0; y < ch; y++) {
    const srcRow = ((y0 + y) * w + x0) * 4, dstRow = y * SIZE * 3;
    for (const { src, dst, weight } of xtab) {
      const s = srcRow + src * 4, d = dstRow + dst * 3;
      rows[d] += rgba[s] * weight;
      rows[d + 1] += rgba[s + 1] * weight;
      rows[d + 2] += rgba[s + 2] * weight;
    }
  }
  const acc = new Float64Array(SIZE * SIZE * 3);
  for (const { src, dst, weight } of ytab) {
    const s = src * SIZE * 3, d = dst * SIZE * 3;
    for (let i = 0; i < SIZE * 3; i++) acc[d + i] += rows[s + i] * weight;
  }
  const out = new Float32Array(SIZE * SIZE * 3);
  for (let i = 0; i < out.length; i++) out[i] = toUint8(acc[i]);
  return out;
}

// Softmax as if the logits were divided by the fitted temperature (calibrate.py).
export function calibrated(probs, temperature) {
  const logits = Array.from(probs, (p) => Math.log(Math.min(1, Math.max(1e-12, p))) / temperature);
  const m = Math.max(...logits);
  const e = logits.map((l) => Math.exp(l - m));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / sum);
}

export function verdict(probs, labels, threshold) {
  let best = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
  const letter = labels[best], pct = `${Math.round(probs[best] * 100)}%`;
  if (probs[best] < threshold) return { letter: null, text: `Not confident (best guess ${letter}, ${pct})` };
  if (MOTION_LETTERS.has(letter)) {
    return { letter, text: `${letter} (${pct}) - J and Z involve motion; this demo reads still frames` };
  }
  return { letter, text: `${letter} (${pct})` };
}

export function top(probs, labels, k = 3) {
  return Array.from(probs, (p, i) => ({ label: labels[i], p }))
    .sort((a, b) => b.p - a.p)
    .slice(0, k);
}
