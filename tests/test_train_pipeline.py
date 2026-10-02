"""train.py: importable without side effects, builds the model, loads data, and a real
(tiny) run writes model.keras. The last one guards a bug where fine-tuning and the final
save sat under the 'skip Phase 1' branch and never ran with the default settings."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import train as T

ROOT = Path(__file__).resolve().parents[1]


def _tiny_dataset(root: Path, per_class: int = 4, size: int = 32) -> Path:
    """A..Z folders of random images - enough to exercise the pipeline, not to learn."""
    rng = np.random.default_rng(0)
    for letter in T.ASCII_A_TO_Z:
        d = root / letter
        d.mkdir(parents=True)
        for i in range(per_class):
            pixels = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(d / f"{letter}{i}.png")
    return root


def test_build_model_output_shape():
    model, base = T.build_model((64, 64, 3), num_classes=26, backbone="cnn")
    assert model.output_shape[-1] == 26
    assert base is None  # only the MobileNetV2 backbone is returned for fine-tuning


def test_make_datasets_one_batch(tmp_path):
    data = _tiny_dataset(tmp_path / "data")
    train_ds, val_ds, class_names = T.make_datasets(
        str(data), img_size=(32, 32), batch_size=8, val_split=0.25, letters_only=True
    )
    assert class_names == T.ASCII_A_TO_Z
    x, y = next(iter(train_ds))
    assert x.shape[1:] == (32, 32, 3) and y.shape[1:] == (26,)
    # pixels stay 0..255: the model rescales inside (see build_model)
    assert float(x.numpy().max()) <= 255.0 and float(x.numpy().min()) >= 0.0


def test_a_real_run_saves_the_model_and_labels(tmp_path):
    data = _tiny_dataset(tmp_path / "data")
    out = tmp_path / "out"
    cmd = [
        sys.executable, str(ROOT / "train.py"),
        "--data_dir", str(data), "--out_dir", str(out),
        "--img_size", "32", "32", "--batch_size", "16",
        "--backbone", "cnn", "--freeze_epochs", "1", "--fine_tune_epochs", "0",
        "--class_weights", "off",
    ]  # fmt: skip
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=ROOT)
    assert done.returncode == 0, done.stderr[-2000:]
    assert (out / "model.keras").is_file()  # was never written with freeze_epochs > 0
    labels = json.loads((out / "labels.json").read_text())
    assert [labels[str(i)] for i in range(26)] == T.ASCII_A_TO_Z
