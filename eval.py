# eval.py
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow import keras


def load_class_names(labels_path: Path):
    # labels.json is {"0":"A","1":"B",...}; force index order
    labels = json.loads(labels_path.read_text())
    idx_to_label = {int(k): v for k, v in labels.items()}
    return [idx_to_label[i] for i in range(len(idx_to_label))]

def plot_confusion(cm, class_names, out_png):
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (Validation)")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=90)
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--labels_path", type=str, required=True)
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="models")
    ap.add_argument("--img_size", nargs=2, type=int, default=[160, 160])
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--val_split", type=float, default=0.10)
    ap.add_argument("--letters_only", type=lambda s: str(s).lower() != "false", default=True)
    ap.add_argument("--max_batches", type=int, default=0,
                    help="stop after this many batches (0 = all); for quick checks and tests")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model & labels
    model = keras.models.load_model(args.model_path)
    class_names = load_class_names(Path(args.labels_path))

    # IMPORTANT: no rescale here (model has Rescaling to [-1,1] inside)
    datagen = keras.preprocessing.image.ImageDataGenerator(validation_split=args.val_split)

    flow = datagen.flow_from_directory(
        directory=str(data_dir),
        target_size=tuple(args.img_size),
        batch_size=args.batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        classes=class_names,      # exact same order as labels.json
    )

    # Run once over validation
    y_true_idx, y_pred_idx = [], []
    steps_total = int(np.ceil(flow.samples / flow.batch_size))
    if args.max_batches > 0:
        steps_total = min(steps_total, args.max_batches)
    for _ in range(steps_total):
        x, y = next(flow)
        preds = model.predict(x, verbose=0)
        y_true_idx.extend(np.argmax(y, axis=1))
        y_pred_idx.extend(np.argmax(preds, axis=1))

    y_true_idx = np.array(y_true_idx)
    y_pred_idx = np.array(y_pred_idx)

    acc = accuracy_score(y_true_idx, y_pred_idx)
    cm  = confusion_matrix(y_true_idx, y_pred_idx, labels=list(range(len(class_names))))
    report = classification_report(
        y_true_idx, y_pred_idx, labels=list(range(len(class_names))),
        target_names=class_names, digits=3, zero_division=0
    )

    np.save(out_dir / "confusion_matrix.npy", cm)
    with open(out_dir / "val_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Overall accuracy: {acc:.4f}\n\n")
        f.write(report)
    plot_confusion(cm, class_names, out_dir / "confusion_matrix.png")

    print(f"[EVAL] Overall accuracy: {acc:.4f}")
    print(f"[EVAL] Saved: {out_dir/'val_report.txt'}")
    print(f"[EVAL] Saved: {out_dir/'confusion_matrix.png'}")
