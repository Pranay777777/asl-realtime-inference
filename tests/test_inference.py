# tests/test_inference.py - the maths behind steps 83, 84 and 86 (numpy only, no model needed)
import json

import numpy as np
import pytest

from inference.benchmark import table
from inference.calibrate import calibrate
from inference.metrics import (
    abstain_report,
    choose_threshold,
    ece,
    fit_temperature,
    latency_summary,
    with_temperature,
)
from inference.model_card import END, START, card, readme_section


def overconfident(n=2000, seed=0):
    """Labels drawn from moderate true probabilities, reported sharpened (T = 1/3)."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 1.2, (n, 5))
    true = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    labels = np.array([rng.choice(5, p=p) for p in true])
    return with_temperature(true, 1 / 3), labels


def test_latency_summary_reports_percentiles_and_fps():
    out = latency_summary([10.0] * 95 + [30.0] * 5)
    assert out["p50_ms"] == 10.0 and out["p95_ms"] > 10.0 and out["fps"] == pytest.approx(90.9, 0.1)


def test_temperature_one_changes_nothing_and_sharpening_is_undone():
    probs, labels = overconfident()
    assert np.allclose(with_temperature(probs, 1.0), probs)
    temperature = fit_temperature(probs, labels)
    assert 2.4 < temperature < 3.6  # recovers the 3x sharpening
    assert ece(with_temperature(probs, temperature), labels) < ece(probs, labels)


def test_threshold_reaches_the_target_accuracy_on_what_it_answers():
    probs, labels = overconfident()
    threshold = choose_threshold(probs, labels, 0.75)
    report = abstain_report(probs, labels, threshold)
    assert report["accepted_accuracy"] >= 0.75 > report["overall_accuracy"]
    assert 0 < report["coverage"] < 1
    assert choose_threshold(np.full((4, 2), 0.5), np.array([0, 1, 0, 1]), 0.99) == 1.0


def test_calibrate_fits_on_one_half_and_reports_on_the_other():
    probs, labels = overconfident()
    report = calibrate(probs, labels, 0.75)
    assert report["ece_after"] < report["ece_before"] and report["images"] == 2000
    assert report["held_out"]["accepted_accuracy"] > report["held_out"]["overall_accuracy"]


BENCH = {
    "images": 2600,
    "input": "64x64",
    "model_name": "asl_mobilenetv2",
    "machine": {"processor": "Intel64", "cpus": 8},
    "rows": [
        {"variant": "Keras FP32", "size_mb": 10.0, "accuracy": 0.97, "p50_ms": 40, "p95_ms": 55, "mean_ms": 42, "fps": 23.8},
        {"variant": "ONNX FP32", "size_mb": 9.0, "accuracy": 0.97, "p50_ms": 4, "p95_ms": 5, "mean_ms": 4.2, "fps": 238},
        {"variant": "ONNX INT8", "size_mb": 2.5, "accuracy": 0.965, "p50_ms": 2, "p95_ms": 3, "mean_ms": 2.1, "fps": 476},
    ],
}
CAL = {
    "temperature": 1.4, "threshold": 0.62, "target_accuracy": 0.99, "ece_before": 0.04,
    "ece_after": 0.01, "images": 2600,
    "held_out": {"coverage": 0.93, "accepted_accuracy": 0.991, "overall_accuracy": 0.97},
}


def test_table_and_model_card_come_from_the_measurements():
    md = table(BENCH)
    assert "| ONNX INT8 | 2.5 | 96.50% | -0.50 pp | 2 | 3 | 476 | 20.0x |" in md
    text = card(BENCH, CAL)
    assert "MobileNetV2 transfer-learning" in text and "**0.62**" in text
    assert "2.0x faster per image" in text and "0.040 → 0.010" in text
    section = readme_section(BENCH, CAL)
    assert section.startswith(START) and section.endswith(END) and "93%" in section
    json.dumps(BENCH)  # the fixtures stay JSON-shaped like the real reports


def test_partial_copy_maps_letters_to_model_indices_and_is_flagged(tmp_path):
    from PIL import Image

    from inference.data import describe, load_split

    for folder in ("B", "D", "nothing"):
        (tmp_path / folder).mkdir()
        for i in range(10):
            Image.new("RGB", (32, 24), (i * 20, 0, 0)).save(tmp_path / folder / f"{i}.jpg")
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({str(i): c for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}))
    info = describe(tmp_path, labels)
    assert info["held_out"] is False and info["letters"] == ["B", "D"] and "A" in info["missing"]
    val_x, val_y = load_split(tmp_path, (16, 16), "validation", val_split=0.5, labels_path=labels)
    tr_x, tr_y = load_split(tmp_path, (16, 16), "training", val_split=0.5, labels_path=labels)
    assert val_x.shape == (10, 16, 16, 3) and val_x.dtype == np.uint8 and len(tr_y) == 10
    assert set(val_y) | set(tr_y) == {1, 3}  # B and D in model order; "nothing" ignored
    again_x, _ = load_split(tmp_path, (16, 16), "validation", val_split=0.5, labels_path=labels)
    assert np.array_equal(val_x, again_x)
    partial = {**BENCH, "data": info}
    assert "optimistic" in card(partial, CAL) and "held out" not in table(partial).split("\n")[0]
