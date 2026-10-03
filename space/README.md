---
title: ASL Alphabet Recognizer
emoji: 🤟
colorFrom: green
colorTo: blue
sdk: static
app_file: index.html
pinned: false
short_description: ASL letters live in your browser - INT8 ONNX, abstains
---

# ASL alphabet recognizer

Live webcam demo of [asl-realtime-inference](https://github.com/Pranay777777/asl-realtime-inference),
running entirely in the browser: MediaPipe finds the hand, the INT8 ONNX MobileNetV2 (2.69 MB)
classifies it with onnxruntime-web (wasm), and a temperature-calibrated confidence below the
fitted threshold answers *Not confident* instead of guessing. No frame leaves the page.

The browser pipeline is a port of the Python one and is checked against it before every
deploy (`space/parity.py`: same top-1 on the 26 test letters, probabilities within 0.02,
identical model inputs). See the [model card](https://github.com/Pranay777777/asl-realtime-inference/blob/main/MODEL_CARD.md)
for how the model was measured and its limits.

Files: the web app comes from the repository's `space/static/`; `asl_int8.onnx` and
`calibration.json` are uploaded here only (model files stay out of the GitHub repository).
