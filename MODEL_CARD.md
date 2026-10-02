# Model card - ASL alphabet recognizer

## Model

A MobileNetV2 transfer-learning image classifier (`asl_mobilenetv2`, see `train.py`) that
maps a 160x160 RGB crop of one hand to one of the 26 letters A-Z.
J and Z involve motion; `predict.py` detects them from MediaPipe fingertip
trajectories, not from this model. Shipped as Keras, ONNX FP32 and ONNX INT8.

## Intended use

Practising and demonstrating the static ASL fingerspelling alphabet with a
webcam. **Not** a sign-language interpreter: it recognises single letters, not
words, grammar, facial expressions or two-handed signs.

## Data

The Kaggle *ASL Alphabet* dataset (static poses, A-Z used). Scored on 2600 images from 16 letters of a local copy of the training data (letters not available: A, B, C, D, E, F, G, H, I, J). `train.py`'s held-out split cannot be rebuilt from this copy, so **some scored images may have been seen in training and the accuracy figures are optimistic**; the latency, size and INT8-vs-FP32 comparisons are unaffected.
The dataset was captured with few signers, similar backgrounds and lighting,
so these numbers are an upper bound for real webcams.

Word-level video datasets such as WLASL and MS-ASL were considered for
cross-dataset testing, but they contain signed *words* in video, not static
alphabet photos, so they cannot score a letter classifier; a fair external test
needs another fingerspelling image set (not yet done).

## Performance

Scored on 2600 images (16 of 26 letters from a local copy of the training data; may overlap training images, so accuracy is optimistic), input 160x160; batch size 1 on Intel64 Family 6 Model 140 Stepping 1, GenuineIntel (8 threads).

| Variant | Size (MB) | Accuracy | Δ accuracy | p50 (ms) | p95 (ms) | FPS | Speed-up |
|---|---:|---:|---:|---:|---:|---:|---:|
| Keras FP32 | 24.96 | 99.92% | +0.00 pp | 120.1 | 169.22 | 7.8 | 1.0x |
| ONNX FP32 | 9.04 | 99.92% | +0.00 pp | 2.06 | 2.69 | 471.1 | 60.4x |
| ONNX INT8 | 2.69 | 99.96% | +0.04 pp | 1.33 | 1.56 | 757.3 | 97.0x |

INT8 changes accuracy by +0.04 percentage points against
ONNX FP32 and is 1.6x faster per image, at
2.69 MB instead of 9.04 MB.

## Confidence and abstaining

Raw softmax scores are over- or under-confident, so they are calibrated with
temperature scaling (T = 0.397): expected calibration error
0.072 → 0.001 on images not used to fit it.

Below a calibrated confidence of **0.72** the app shows *not
confident* instead of a letter. On held-out images that answers
100% of the time with 99.9% accuracy, against
99.9% if it always answered. In accessibility tooling a
"not confident" is better than a confident wrong letter. Values live in
`models/calibration.json`; `predict.py` reads them.

## Limitations and risks

- Few signers and uniform backgrounds: expect lower accuracy for other skin
  tones, hand sizes, lighting and cluttered backgrounds - not yet measured.
- Letters that look alike are the main confusions; `predict.py` already
  demands a bigger top-1/top-2 margin for X, U, S, V and E.
- Webcam frames differ from dataset photos; the abstain threshold was fitted
  on dataset images.
- Do not use it where a misread letter has consequences (medical, legal,
  safety); it is a learning and demo tool.

## Reproduce

```
python -m inference.export_onnx
python -m inference.quantize --data_dir <asl_alphabet_train>
python -m inference.benchmark --data_dir <asl_alphabet_train>
python -m inference.calibrate --data_dir <asl_alphabet_train>
python -m inference.model_card
```
