"""Images + labels for scoring, mapped onto the model's own output indices.

Two cases, picked automatically from what is in --data_dir:

* exactly the 26 folders A-Z  -> the same validation split train.py used
  (image_dataset_from_directory, validation_split 0.1, seed 1337), so nothing is
  scored on images the model trained on (held_out = True);
* anything else (some letters missing, extra folders such as del/nothing/space)
  -> the A-Z folders that exist, split deterministically here. train.py's split
  cannot be reproduced from a partial copy, so these images may overlap the
  training set (held_out = False) and the model card says so.

Labels always follow models/labels.json (the order predict.py uses), falling
back to A-Z. Images are returned as uint8 RGB (callers cast to float32), which
keeps 2,600 images at 160x160 near 200 MB instead of 800 MB.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ASCII_A_TO_Z = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def model_labels(path: Path = Path("models/labels.json")) -> list[str]:
    """Class names in model output order: list, {"0": "A"} or {"A": 0}; else A-Z."""
    if not path.exists():
        return list(ASCII_A_TO_Z)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "labels" in raw:
        raw = raw["labels"]
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if all(str(k).lstrip("-").isdigit() for k in raw):
        return [str(raw[k]) for k in sorted(raw, key=lambda k: int(k))]
    return [str(k) for k, _ in sorted(raw.items(), key=lambda kv: int(kv[1]))]


def letter_dirs(data_dir: Path) -> list[str]:
    return sorted(p.name for p in Path(data_dir).iterdir() if p.is_dir() and p.name in ASCII_A_TO_Z)


def describe(data_dir: Path, labels_path: Path = Path("models/labels.json")) -> dict:
    """What load_split will score on; written into the reports and the model card."""
    subdirs = sorted(p.name for p in Path(data_dir).iterdir() if p.is_dir())
    present = letter_dirs(data_dir)
    names = model_labels(labels_path)
    held_out = subdirs == ASCII_A_TO_Z
    return {
        "letters": present,
        "missing": [c for c in names if c in ASCII_A_TO_Z and c not in present],
        "held_out": held_out,
        "note": (
            "validation split held out by train.py (validation_split 0.1, seed 1337)"
            if held_out
            else f"{len(present)} of 26 letters from a local copy of the training data; "
            "may overlap training images, so accuracy is optimistic"
        ),
    }


def _keras_split(data_dir, img_size, subset, limit, val_split, seed):
    from tensorflow import keras

    ds = keras.utils.image_dataset_from_directory(
        str(data_dir), labels="inferred", label_mode="int", class_names=ASCII_A_TO_Z,
        image_size=img_size, batch_size=64, shuffle=True, seed=seed,
        validation_split=val_split, subset=subset,
    )
    images, labels, taken = [], [], 0
    for x, y in ds:
        images.append(np.clip(np.rint(x.numpy()), 0, 255).astype(np.uint8))
        labels.append(y.numpy())
        taken += len(y)
        if limit is not None and taken >= limit:
            break
    xs, ys = np.concatenate(images), np.concatenate(labels)
    return (xs[:limit], ys[:limit]) if limit is not None else (xs, ys)


def _local_split(data_dir, img_size, subset, limit, val_split, seed):
    from PIL import Image

    files = [
        (f, letter)
        for letter in letter_dirs(data_dir)
        for f in sorted((Path(data_dir) / letter).iterdir())
        if f.suffix.lower() in IMAGE_EXT
    ]
    if not files:
        raise SystemExit(f"no A-Z image folders found in {data_dir}")
    order = np.random.default_rng(seed).permutation(len(files))
    n_val = int(len(files) * val_split)
    chosen = order[len(files) - n_val :] if subset == "validation" else order[: len(files) - n_val]
    chosen = chosen[:limit] if limit is not None else chosen
    height, width = img_size
    xs = np.empty((len(chosen), height, width, 3), dtype=np.uint8)
    names = [files[i][1] for i in chosen]
    for row, i in enumerate(chosen):
        with Image.open(files[i][0]) as im:
            xs[row] = np.asarray(im.convert("RGB").resize((width, height), Image.BILINEAR))
    return xs, names


def load_split(
    data_dir: Path,
    img_size: tuple[int, int],
    subset: str,
    limit: int | None = None,
    val_split: float = 0.1,
    seed: int = 1337,
    labels_path: Path = Path("models/labels.json"),
) -> tuple[np.ndarray, np.ndarray]:
    """(uint8 RGB images, label indices in model output order) for "training" or "validation"."""
    names = model_labels(labels_path)
    index = {name: i for i, name in enumerate(names)}
    if describe(data_dir, labels_path)["held_out"]:
        xs, ys = _keras_split(data_dir, img_size, subset, limit, val_split, seed)
        remap = np.array([index[c] for c in ASCII_A_TO_Z])  # keras A-Z index -> model index
        return xs, remap[ys]
    xs, letters = _local_split(data_dir, img_size, subset, limit, val_split, seed)
    unknown = sorted(set(letters) - set(index))
    if unknown:
        raise SystemExit(f"letters {unknown} are not in the model's labels {names}")
    return xs, np.array([index[c] for c in letters], dtype=np.int64)
