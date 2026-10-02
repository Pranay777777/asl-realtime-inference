### Sign Language Recognizer (CNN)

Real-time American Sign Language (ASL) alphabet recognizer (A–Z) using TensorFlow/Keras and OpenCV.
Supports webcam prediction, single-image testing, dynamic letters (J, Z) via MediaPipe fingertip tracking, and easy evaluation with confusion matrix.

## Project structure
sign-language-recognizer/
├── train.py                 # training script
├── eval.py                  # offline evaluation / confusion matrix
├── predict.py               # realtime webcam + single-image test
├── requirements.txt
├── README.md
├── tests/                   # minimal CI-style checks
│   └── ...
├── data/
│   └── asl_alphabet/        # dataset root (see below)
└── models/
    ├── cnn_model.h5         # trained model (created by train.py)
    ├── labels.json          # index→label mapping
    ├── confusion_matrix.png # saved by eval.py
    └── val_report.txt       # saved by eval.py

1) Setup

OS: Windows 10/11 • Python: 3.12 • Hardware: CPU (GPU optional)

# 1. create & activate venv
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. install dependencies (pins match this project)
pip install -r requirements.txt

# 3. sanity check
python -c "import tensorflow as tf, cv2; print('OK', tf.__version__, cv2.__version__)"

2) Dataset

Use the ASL Alphabet dataset (A–Z, static poses). Download from Kaggle and unzip to:

data/asl_alphabet/asl_alphabet_train/<LETTER>/*.jpg

Tip: Make sure there are 26 folders (A…Z). No extra classes.

3) Train

Typical starter config (64×64, CPU-friendly):

python train.py `
  --data_dir data\asl_alphabet\asl_alphabet_train `
  --out_dir models `
  --img_size 64 64 `
  --batch_size 64 `
  --epochs 12 `
  --val_split 0.1 `
  --patience 4 `
  --letters_only true

Artifacts saved to models/:
cnn_model.h5
labels.json
history.csv (if enabled by your script)

best weights via EarlyStopping/ModelCheckpoint

Resume training
If resuming, img_size must match the saved model’s input (use eval.py/predict.py printout to confirm).
python train.py ... --resume_path models\cnn_model.h5


If you see a “Shape mismatch … dense/kernel” error, you changed --img_size. Re-train from scratch or keep sizes consistent.

4) Evaluate

Quick check (few batches):

python eval.py `
  --data_dir data\asl_alphabet\asl_alphabet_train `
  --model_path models\cnn_model.h5 `
  --labels_path models\labels.json `
  --img_size 64 64 `
  --batch_size 64 `
  --val_split 0.1 `
  --letters_only true `
  --max_batches 2


Full split:
python eval.py `
  --data_dir data\asl_alphabet\asl_alphabet_train `
  --model_path models\cnn_model.h5 `
  --labels_path models\labels.json `
  --img_size 64 64 `
  --batch_size 64 `
  --val_split 0.1 `
  --letters_only true


Outputs:
models/val_report.txt (per-class precision/recall/F1, support)
models/confusion_matrix.png (+ confusion_matrix.npy)

Console prints overall accuracy

5) Predict (webcam)
python predict.py `
  --model_path models\cnn_model.h5 `
  --labels_path models\labels.json `
  --camera 0 `
  --conf_thresh 0.40 `
  --smooth_k 11 `
  --tta_angles -15 0 15


# Keys (during webcam):

a — append current predicted letter to text

space — append a space

d — delete last character

c — clear text

+ / - — grow/shrink manual ROI

s — save current ROI to debug_roi.jpg

q — quit

# Useful flags:

--no_tta to disable angle averaging

--clahe to normalize lighting

--skin_mask to suppress background

--mirror to mirror the webcam view

--use_mediapipe to auto-align ROI by detected hand (optional)

--dynamic to enable dynamic letters (J, Z) via fingertip tracking

Dynamic letters (optional):

python predict.py ... --dynamic
# tuning:
# --dynamic_len 24 --dynamic_min_move 0.35 --dynamic_cooldown 18

6) Predict (single image)
python predict.py --model_path models\cnn_model.h5 --labels_path models\labels.json --test_image path\to\image.jpg


Pro-tip: Save an ROI from webcam with s, then test it:

python predict.py --model_path models\cnn_model.h5 --labels_path models\labels.json --test_image debug_roi.jpg

7) Run tests
pytest -q


All tests should pass. They check:

imports, CLI parsing, dataset folders,

that eval.py runs and produces artifacts on a tiny slice,

label mapping integrity, etc.

## Troubleshooting

Nothing predicts vs. dataset works: save debug_roi.jpg (s) and run single-image test. If that works, adjust ROI or use --use_mediapipe.

Wrong size / resume error: your new --img_size doesn’t match the saved model. Keep sizes consistent or re-train.

PowerShell multiline errors: prefer backticks (`) for line continuation or run commands on one line.

MediaPipe slow/missing GPU: MediaPipe runs on CPU here; if frame-rate drops, disable --use_mediapipe and keep --tta_angles short (e.g., -15 0 15).

Confusable letters (X, U, S, V, E): we apply margin gating and EMA in predict.py. For more robustness, we’ll add improved training augmentations later.

<!-- inference:start -->
## Fast inference: ONNX + INT8

Scored on 2600 images (16 of 26 letters from a local copy of the training data; may overlap training images, so accuracy is optimistic), input 160x160; batch size 1 on Intel64 Family 6 Model 140 Stepping 1, GenuineIntel (8 threads).

| Variant | Size (MB) | Accuracy | Δ accuracy | p50 (ms) | p95 (ms) | FPS | Speed-up |
|---|---:|---:|---:|---:|---:|---:|---:|
| Keras FP32 | 24.96 | 99.92% | +0.00 pp | 120.1 | 169.22 | 7.8 | 1.0x |
| ONNX FP32 | 9.04 | 99.92% | +0.00 pp | 2.06 | 2.69 | 471.1 | 60.4x |
| ONNX INT8 | 2.69 | 99.96% | +0.04 pp | 1.33 | 1.56 | 757.3 | 97.0x |

Calibrated abstain threshold 0.72: answers 100% of held-out images at 99.9% accuracy, says *not confident* otherwise. Details, limitations and how to reproduce: [MODEL_CARD.md](MODEL_CARD.md).
<!-- inference:end -->
