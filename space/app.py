"""ASL alphabet recognizer - live webcam demo (Hugging Face Space, free CPU tier).

MediaPipe finds the hand, the INT8 ONNX model classifies the crop, and the calibrated
confidence decides between a letter and "Not confident". Models are uploaded to the
Space next to this file (models/ is not in the GitHub repo).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import gradio as gr
import mediapipe as mp
import numpy as np

from pipeline import Recognizer, box_from_points, centre_box

HERE = Path(__file__).parent
rec = Recognizer(HERE / "models")
LABELS, TEMPERATURE, THRESHOLD = rec.labels, rec.temperature, rec.threshold

hands = mp.solutions.hands.Hands(
    static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
)


def hand_box(rgb: np.ndarray):
    """Square box around the first detected hand, padded and clipped to the frame."""
    found = hands.process(rgb)
    if not found.multi_hand_landmarks:
        return None
    h, w = rgb.shape[:2]
    pts = found.multi_hand_landmarks[0].landmark
    return box_from_points([p.x * w for p in pts], [p.y * h for p in pts], w, h)


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
        box = centre_box(rgb.shape[1], rgb.shape[0])
        note = " - no hand detected, read the centre of the frame"
    x0, y0, x1, y1 = box
    probs = rec.probs(rgb, box)
    top = np.argsort(probs)[::-1][:3]
    scores = {LABELS[int(i)]: float(probs[i]) for i in top}
    verdict = rec.verdict(probs) + note
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
