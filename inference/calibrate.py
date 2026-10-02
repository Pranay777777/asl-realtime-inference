"""Confidence calibration and the abstain threshold (plan step 86).

    python -m inference.calibrate --data_dir <asl_alphabet_train> --target 0.99

Half of the held-out validation images fit a temperature (so "90% sure" means
right about 90% of the time) and pick the lowest confidence at which answers
reach the target accuracy; the other half reports how well that holds. Writes
models/calibration.json, which predict.py reads: below the threshold it says
"not confident" instead of guessing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from inference.benchmark import onnx_predictor
from inference.data import describe, load_split
from inference.metrics import (
    abstain_report,
    choose_threshold,
    ece,
    fit_temperature,
    with_temperature,
)


def calibrate(probs: np.ndarray, labels: np.ndarray, target: float) -> dict[str, Any]:
    order = np.random.default_rng(1337).permutation(len(labels))
    fit, check = order[: len(order) // 2], order[len(order) // 2 :]
    temperature = fit_temperature(probs[fit], labels[fit])
    calibrated = with_temperature(probs, temperature)
    threshold = choose_threshold(calibrated[fit], labels[fit], target)
    return {
        "temperature": temperature,
        "threshold": threshold,
        "target_accuracy": target,
        "ece_before": ece(probs[check], labels[check]),
        "ece_after": ece(calibrated[check], labels[check]),
        "held_out": abstain_report(calibrated[check], labels[check], threshold),
        "images": int(len(labels)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, default=Path("models/asl_fp32.onnx"))
    parser.add_argument("--target", type=float, default=0.99)
    parser.add_argument("--limit", type=int, default=2600)
    parser.add_argument("--out", type=Path, default=Path("models/calibration.json"))
    args = parser.parse_args()
    predict = onnx_predictor(args.onnx)
    import onnxruntime as ort

    shape = ort.InferenceSession(str(args.onnx)).get_inputs()[0].shape
    images, labels = load_split(args.data_dir, (int(shape[1]), int(shape[2])), "validation", args.limit)
    probs = np.concatenate([predict(images[i : i + 64]) for i in range(0, len(images), 64)])
    report = calibrate(probs, labels, args.target)
    report["data"] = describe(args.data_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    Path("results").mkdir(exist_ok=True)
    Path("results/calibration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
