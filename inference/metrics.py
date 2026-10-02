"""Pure-numpy metrics: latency percentiles, calibration (ECE, temperature) and the abstain threshold.

No TensorFlow or ONNX here, so these are unit-tested fast and run anywhere.
"""

from __future__ import annotations

import numpy as np


def latency_summary(ms: list[float]) -> dict[str, float]:
    """p50 / p95 / mean in milliseconds and the throughput they imply."""
    arr = np.asarray(ms, dtype=np.float64)
    mean = float(arr.mean())
    return {
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "mean_ms": round(mean, 2),
        "fps": round(1000.0 / mean, 1) if mean > 0 else 0.0,
    }


def with_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Rescale softmax outputs as if their logits were divided by `temperature`."""
    logits = np.log(np.clip(probs, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)).mean())


def fit_temperature(probs: np.ndarray, labels: np.ndarray) -> float:
    """The temperature (0.25-10) that minimises negative log-likelihood - a 1-D grid search."""
    grid = np.exp(np.linspace(np.log(0.25), np.log(10.0), 400))
    losses = [nll(with_temperature(probs, t), labels) for t in grid]
    return round(float(grid[int(np.argmin(losses))]), 3)


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error: how far confidence is from accuracy, bin by bin."""
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        inside = (confidence > low) & (confidence <= high)
        if inside.any():
            total += inside.mean() * abs(correct[inside].mean() - confidence[inside].mean())
    return round(float(total), 4)


def choose_threshold(probs: np.ndarray, labels: np.ndarray, target: float) -> float:
    """The lowest confidence threshold whose accepted predictions reach `target` accuracy.

    Below the threshold the app says "not confident" instead of guessing. Returns 1.0
    (abstain on everything) if no threshold reaches the target.
    """
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    for threshold in np.unique(np.round(confidence, 3)):
        accepted = confidence >= threshold
        if accepted.any() and correct[accepted].mean() >= target:
            return round(float(threshold), 3)
    return 1.0


def abstain_report(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    """Coverage (share answered) and accuracy of the answered, at a threshold."""
    accepted = probs.max(axis=1) >= threshold
    correct = probs.argmax(axis=1) == labels
    return {
        "coverage": round(float(accepted.mean()), 4),
        "accepted_accuracy": round(float(correct[accepted].mean()), 4) if accepted.any() else 0.0,
        "overall_accuracy": round(float(correct.mean()), 4),
    }
