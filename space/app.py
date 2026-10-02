"""ASL alphabet recognizer - live webcam demo (Hugging Face Space, free CPU tier).

MediaPipe finds the hand, the INT8 ONNX model classifies the crop, and the calibrated
confidence decides between a letter and "Not confident". Models are uploaded to the
Space next to this file (models/ is not in the GitHub repo).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import gradio as gr
import mediapipe as mp
import numpy as np
import onnxruntime as ort

HERE = Path(__file__).parent
MODELS = HERE / "models"
SIZE = 160  # the model's input is 160x160 RGB, 0-255; it rescales internally
PAD = 1.4  # crop margin around the hand, as in predict.py
MOTION_LETTERS = {"J", "Z"}  # signed with movement; a single frame cannot show them

session = ort.InferenceSession(str(MODELS / "asl_int8.onnx"), providers=["CPUExecutionProvider"])
INPUT = session.get_inputs()[0].name
LABELS = {int(k): v for k, v in json.loads((MODELS / "labels.json").read_text()).items()}
_cal = json.loads((MODELS / "calibration.json").read_text())
TEMPERATURE, THRESHOLD = float(_cal["temperature"]), float(_cal["threshold"])

hands = mp.solutions.hands.Hands(
    static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
)


def calibrated(probs: np.ndarray) -> np.ndarray:
    """Softmax as if the logits were divided by the fitted temperature (calibrate.py)."""
    logits = np.log(np.clip(probs, 1e-12, 1.0)) / TEMPERATURE
    e = np.exp(logits - logits.max())
    return e / e.sum()


def hand_box(rgb: np.ndarray) -> tuple[int, int, int, int] | None:
    """Square box around the first detected hand, padded and clipped to the frame."""
    found = hands.process(rgb)
    if not found.multi_hand_landmarks:
        return None
    h, w = rgb.shape[:2]
    pts = found.multi_hand_landmarks[0].landmark
    xs, ys = [p.x * w for p in pts], [p.y * h for p in pts]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half = max(max(xs) - min(xs), max(ys) - min(ys)) * PAD / 2
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(w, cx + half)), int(min(h, cy + half))
    return (x0, y0, x1, y1) if x1 - x0 > 10 and y1 - y0 > 10 else None


def classify(rgb_crop: np.ndarray) -> np.ndarray:
    x = cv2.resize(rgb_crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
    return calibrated(session.run(None, {INPUT: x[None]})[0][0])


def recognise(frame: np.ndarray | None, fallback: bool = False):
    if frame is None:
        return None, "Show one hand to the camera", {}
    rgb = np.ascontiguousarray(frame[..., :3])
    box = hand_box(rgb)
    note = ""
    if box is None and not fallback:
        return rgb, "No hand found - hold one hand in view", {}
    if box is None:
        # Uploads only: MediaPipe misses some close, dark crops (like the training photos), so
        # read the centre square, as predict.py does for single images, and say so. Live
        # frames never guess - an empty scene can still score high on some letter.
        h, w = rgb.shape[:2]
        side = min(h, w)
        x0, y0 = (w - side) // 2, (h - side) // 2
        box = (x0, y0, x0 + side, y0 + side)
        note = " - no hand detected, read the centre of the frame"
    x0, y0, x1, y1 = box
    probs = classify(rgb[y0:y1, x0:x1])
    top = np.argsort(probs)[::-1][:3]
    scores = {LABELS[int(i)]: float(probs[i]) for i in top}
    best, conf = LABELS[int(top[0])], float(probs[top[0]])
    if conf < THRESHOLD:
        verdict = f"Not confident (best guess {best}, {conf:.0%})"
    elif best in MOTION_LETTERS:
        verdict = f"{best} ({conf:.0%}) - J and Z involve motion; this demo reads still frames"
    else:
        verdict = f"{best} ({conf:.0%})"
    verdict += note
    shown = rgb.copy()
    cv2.rectangle(shown, (x0, y0), (x1, y1), (0, 200, 120), 3)
    return shown, verdict, scores


ABOUT = f"""
Fingerspell a **static** ASL letter to your webcam. MediaPipe finds the hand, an **INT8 ONNX**
MobileNetV2 (2.69 MB) classifies the crop on CPU, and a temperature-calibrated confidence
(T = {TEMPERATURE}) below **{THRESHOLD:.2f}** answers *Not confident* instead of guessing.

Limits: trained on the Kaggle *ASL Alphabet* photos (few signers, plain backgrounds), so
expect lower accuracy on real webcams, other lighting and cluttered backgrounds. Single
letters only - not a sign-language interpreter. Frames are processed in memory and not stored.
[Code, model card and benchmark](https://github.com/Pranay777777/asl-realtime-inference)
"""

with gr.Blocks(title="ASL alphabet recognizer") as demo:
    gr.Markdown("# ASL alphabet recognizer")
    gr.Markdown(ABOUT)
    with gr.Row():
        with gr.Column():
            with gr.Tab("Webcam (live)"):
                cam = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Camera")
            with gr.Tab("Upload a photo"):
                photo = gr.Image(sources=["upload"], type="numpy", label="Photo of one hand")
        with gr.Column():
            seen = gr.Image(label="What the model sees (hand box)", interactive=False)
            verdict = gr.Textbox(label="Prediction", interactive=False)
            scores = gr.Label(num_top_classes=3, label="Top 3 (calibrated)")
    outputs = [seen, verdict, scores]
    cam.stream(recognise, inputs=cam, outputs=outputs, stream_every=0.25)
    photo.change(lambda f: recognise(f, fallback=True), inputs=photo, outputs=outputs)

if __name__ == "__main__":
    demo.launch()
