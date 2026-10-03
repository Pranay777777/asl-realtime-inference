"""The recognition pipeline shared by the Gradio app (app.py) and the parity check.

static/pipeline.js is a line-by-line port of these functions for the browser; keep the two
in step (space/parity.py compares them on the 26 test images).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

SIZE = 160  # model input: 160x160 RGB, NHWC, float32 0-255 (it rescales internally)
PAD = 1.4  # crop margin around the hand, as in predict.py
MOTION_LETTERS = {"J", "Z"}  # signed with movement; a single frame cannot show them

Box = tuple[int, int, int, int]  # x0, y0, x1, y1 (exclusive)


def box_from_points(xs: Sequence[float], ys: Sequence[float], w: int, h: int) -> Box | None:
    """Square box around hand landmarks (pixels), padded and clipped to the frame."""
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half = max(max(xs) - min(xs), max(ys) - min(ys)) * PAD / 2
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(w, cx + half)), int(min(h, cy + half))
    return (x0, y0, x1, y1) if x1 - x0 > 10 and y1 - y0 > 10 else None


def centre_box(w: int, h: int) -> Box:
    """The centre square, as predict.py reads single images."""
    side = min(h, w)
    x0, y0 = (w - side) // 2, (h - side) // 2
    return (x0, y0, x0 + side, y0 + side)


def model_input(rgb: np.ndarray, box: Box) -> np.ndarray:
    """Crop, area-resize to 160x160, float32 0-255, NHWC with a batch of one."""
    x0, y0, x1, y1 = box
    crop = rgb[y0:y1, x0:x1]
    return cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)[None]


def calibrated(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Softmax as if the logits were divided by the fitted temperature (calibrate.py)."""
    logits = np.log(np.clip(probs, 1e-12, 1.0)) / temperature
    e = np.exp(logits - logits.max())
    return e / e.sum()


class Recognizer:
    def __init__(self, models: Path) -> None:
        self.session = ort.InferenceSession(
            str(models / "asl_int8.onnx"), providers=["CPUExecutionProvider"]
        )
        self.input = self.session.get_inputs()[0].name
        labels = json.loads((models / "labels.json").read_text())
        self.labels = {int(k): v for k, v in labels.items()}
        cal = json.loads((models / "calibration.json").read_text())
        self.temperature, self.threshold = float(cal["temperature"]), float(cal["threshold"])

    def probs(self, rgb: np.ndarray, box: Box) -> np.ndarray:
        raw = self.session.run(None, {self.input: model_input(rgb, box)})[0][0]
        return calibrated(raw, self.temperature)

    def verdict(self, probs: np.ndarray) -> str:
        best = int(np.argmax(probs))
        letter, conf = self.labels[best], float(probs[best])
        if conf < self.threshold:
            return f"Not confident (best guess {letter}, {conf:.0%})"
        if letter in MOTION_LETTERS:
            return f"{letter} ({conf:.0%}) - J and Z involve motion; this demo reads still frames"
        return f"{letter} ({conf:.0%})"
