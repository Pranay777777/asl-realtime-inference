"""Write MODEL_CARD.md and the README results section from the measured numbers (step 86).

    python -m inference.model_card

Reads results/benchmark.json and results/calibration.json; nothing is typed by hand.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from inference.benchmark import table

START, END = "<!-- inference:start -->", "<!-- inference:end -->"


def data_text(bench: dict[str, Any]) -> str:
    data = bench.get("data") or {"held_out": True}
    if data.get("held_out"):
        return (
            "Scored on the validation split `train.py` held out (validation_split 0.1, "
            f"seed 1337): {bench['images']} images the model never trained on."
        )
    missing = ", ".join(data.get("missing", [])) or "none"
    return (
        f"Scored on {bench['images']} images from {len(data.get('letters', []))} letters of a local "
        f"copy of the training data (letters not available: {missing}). `train.py`'s held-out "
        "split cannot be rebuilt from this copy, so **some scored images may have been seen in "
        "training and the accuracy figures are optimistic**; the latency, size and INT8-vs-FP32 "
        "comparisons are unaffected."
    )


def card(bench: dict[str, Any], cal: dict[str, Any]) -> str:
    rows = {r["variant"]: r for r in bench["rows"]}
    fp32, int8 = rows["ONNX FP32"], rows["ONNX INT8"]
    held = cal["held_out"]
    return f"""# Model card - ASL alphabet recognizer

## Model

{"A MobileNetV2 transfer-learning" if "mobilenet" in bench.get("model_name", "") else "A small convolutional"} image classifier (`{bench.get("model_name", "model")}`, see `train.py`) that
maps a {bench['input']} RGB crop of one hand to one of the 26 letters A-Z.
J and Z involve motion; `predict.py` detects them from MediaPipe fingertip
trajectories, not from this model. Shipped as Keras, ONNX FP32 and ONNX INT8.

## Intended use

Practising and demonstrating the static ASL fingerspelling alphabet with a
webcam. **Not** a sign-language interpreter: it recognises single letters, not
words, grammar, facial expressions or two-handed signs.

## Data

The Kaggle *ASL Alphabet* dataset (static poses, A-Z used). {data_text(bench)}
The dataset was captured with few signers, similar backgrounds and lighting,
so these numbers are an upper bound for real webcams.

Word-level video datasets such as WLASL and MS-ASL were considered for
cross-dataset testing, but they contain signed *words* in video, not static
alphabet photos, so they cannot score a letter classifier; a fair external test
needs another fingerspelling image set (not yet done).

## Performance

{table(bench)}
INT8 changes accuracy by {(int8['accuracy'] - fp32['accuracy']) * 100:+.2f} percentage points against
ONNX FP32 and is {fp32['mean_ms'] / int8['mean_ms']:.1f}x faster per image, at
{int8['size_mb']} MB instead of {fp32['size_mb']} MB.

## Confidence and abstaining

Raw softmax scores are over- or under-confident, so they are calibrated with
temperature scaling (T = {cal['temperature']}): expected calibration error
{cal['ece_before']:.3f} → {cal['ece_after']:.3f} on images not used to fit it.

Below a calibrated confidence of **{cal['threshold']:.2f}** the app shows *not
confident* instead of a letter. On held-out images that answers
{held['coverage']:.0%} of the time with {held['accepted_accuracy']:.1%} accuracy, against
{held['overall_accuracy']:.1%} if it always answered. In accessibility tooling a
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
"""


def readme_section(bench: dict[str, Any], cal: dict[str, Any]) -> str:
    held = cal["held_out"]
    return (
        f"{START}\n## Fast inference: ONNX + INT8\n\n{table(bench)}\n"
        f"Calibrated abstain threshold {cal['threshold']:.2f}: answers {held['coverage']:.0%} "
        f"of held-out images at {held['accepted_accuracy']:.1%} accuracy, says *not confident* "
        f"otherwise. Details, limitations and how to reproduce: [MODEL_CARD.md](MODEL_CARD.md).\n"
        f"{END}"
    )


def main() -> None:
    bench = json.loads(Path("results/benchmark.json").read_text(encoding="utf-8"))
    cal = json.loads(Path("results/calibration.json").read_text(encoding="utf-8"))
    Path("MODEL_CARD.md").write_text(card(bench, cal), encoding="utf-8")
    readme = Path("README.md")
    text = readme.read_text(encoding="utf-8")
    section = readme_section(bench, cal)
    if START in text:
        text = re.sub(re.escape(START) + ".*?" + re.escape(END), lambda _: section, text, flags=re.S)
    else:
        text = text.rstrip() + "\n\n" + section + "\n"
    readme.write_text(text, encoding="utf-8")
    print("MODEL_CARD.md and README.md updated")


if __name__ == "__main__":
    main()
