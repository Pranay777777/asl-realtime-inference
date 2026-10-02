"""Accuracy vs latency for Keras FP32, ONNX FP32 and ONNX INT8 (plan steps 83 and 84).

    python -m inference.benchmark --data_dir <asl_alphabet_train>

Accuracy is measured on the validation split train.py held out; latency is one
image at a time (what the webcam loop does), after a warm-up, reported as
p50 / p95 and frames per second. Writes results/benchmark.json and
results/tradeoff.md.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from inference.data import describe, load_split
from inference.metrics import latency_summary

Predict = Callable[[np.ndarray], np.ndarray]


def timed(predict: Predict, images: np.ndarray, runs: int, warmup: int = 20) -> list[float]:
    for i in range(warmup):
        predict(images[i % len(images)][None])
    out = []
    for i in range(runs):
        start = time.perf_counter()
        predict(images[i % len(images)][None])
        out.append((time.perf_counter() - start) * 1000)
    return out


def accuracy(predict: Predict, images: np.ndarray, labels: np.ndarray, batch: int = 64) -> float:
    hits = 0
    for i in range(0, len(images), batch):
        hits += int((predict(images[i : i + batch]).argmax(axis=1) == labels[i : i + batch]).sum())
    return round(hits / len(images), 4)


def onnx_predictor(path: Path, provider: str = "CPUExecutionProvider") -> Predict:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=[provider])
    name = session.get_inputs()[0].name
    return lambda x: session.run(None, {name: x.astype(np.float32)})[0]


def run(args: argparse.Namespace) -> dict[str, Any]:
    import onnxruntime as ort
    import tensorflow as tf
    from tensorflow import keras

    model = keras.models.load_model(args.model)
    height, width = model.input_shape[1:3]
    images, labels = load_split(args.data_dir, (height, width), "validation", limit=args.limit)
    variants: list[tuple[str, Path, Predict]] = [
        ("Keras FP32", args.model, lambda x: model(x.astype(np.float32), training=False).numpy()),
        ("ONNX FP32", args.fp32, onnx_predictor(args.fp32)),
        ("ONNX INT8", args.int8, onnx_predictor(args.int8)),
    ]
    if "CUDAExecutionProvider" in ort.get_available_providers():  # pragma: no cover
        variants.append(("ONNX FP32 (GPU)", args.fp32, onnx_predictor(args.fp32, "CUDAExecutionProvider")))
    rows = []
    for name, path, predict in variants:
        rows.append(
            {
                "variant": name,
                "size_mb": round(path.stat().st_size / 1e6, 2),
                "accuracy": accuracy(predict, images, labels),
                **latency_summary(timed(predict, images, args.runs)),
            }
        )
        print(f"  {name}: {rows[-1]}")
    return {
        "images": int(len(images)),
        "data": describe(args.data_dir),
        "input": f"{height}x{width}",
        "model_name": model.name,
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "cpus": os.cpu_count(),
            "tensorflow": tf.__version__,
            "onnxruntime": ort.__version__,
        },
        "rows": rows,
    }


def table(report: dict[str, Any]) -> str:
    base = report["rows"][0]
    lines = [
        f"Scored on {report['images']} images ({report.get('data', {}).get('note', 'validation split held out by train.py')}), input {report['input']}; "
        f"batch size 1 on {report['machine']['processor']} ({report['machine']['cpus']} threads).",
        "",
        "| Variant | Size (MB) | Accuracy | Δ accuracy | p50 (ms) | p95 (ms) | FPS | Speed-up |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in report["rows"]:
        delta = (r["accuracy"] - base["accuracy"]) * 100
        speed = base["mean_ms"] / r["mean_ms"] if r["mean_ms"] else 0
        lines.append(
            f"| {r['variant']} | {r['size_mb']} | {r['accuracy']:.2%} | {delta:+.2f} pp | "
            f"{r['p50_ms']} | {r['p95_ms']} | {r['fps']} | {speed:.1f}x |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/model.keras"))
    parser.add_argument("--fp32", type=Path, default=Path("models/asl_fp32.onnx"))
    parser.add_argument("--int8", type=Path, default=Path("models/asl_int8.onnx"))
    parser.add_argument("--limit", type=int, default=2600, help="validation images to score")
    parser.add_argument("--runs", type=int, default=300, help="timed single-image runs")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    report = run(args)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out / "tradeoff.md").write_text(table(report), encoding="utf-8")
    print(table(report))


if __name__ == "__main__":
    main()
