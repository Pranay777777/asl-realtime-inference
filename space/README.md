---
title: ASL Alphabet Recognizer
emoji: 🤟
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.29.0
python_version: "3.11"
app_file: app.py
pinned: false
short_description: Live webcam ASL letters - INT8 ONNX on CPU, abstains when unsure
---

# ASL alphabet recognizer

Live webcam demo of [asl-realtime-inference](https://github.com/Pranay777777/asl-realtime-inference):
MediaPipe finds the hand, an INT8 ONNX MobileNetV2 (2.69 MB) classifies it on CPU, and a
temperature-calibrated confidence below the fitted threshold answers *Not confident*
instead of guessing. See the [model card](https://github.com/Pranay777777/asl-realtime-inference/blob/main/MODEL_CARD.md)
for how it was measured and its limits.

Files: `app.py` and `requirements.txt` come from the repository's `space/` folder;
`models/asl_int8.onnx`, `models/labels.json` and `models/calibration.json` are uploaded
here only (model files stay out of the GitHub repository).
