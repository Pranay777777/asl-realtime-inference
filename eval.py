# eval.py
import argparse, json, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow import keras

ASCII_A_TO_Z = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

def make_valid_flow(data_dir: Path, img_size, batch_size, val_split, letters_only):
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=val_split,
    )
    classes = ASCII_A_TO_Z if letters_only else None
    flow = datagen.flow_from_directory(
        directory=str(data_dir),
        target_size=tuple(img_size),
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        classes=classes,
    )
    return flow

def plot_confusion(cm, class_names, out_png):
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (Validation)")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default="models/cnn_model.h5")
    ap.add_argument("--labels_path", type=str, default="models/labels.json")
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="models")
    ap.add_argument("--img_size", nargs=2, type=int, default=[64, 64])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--val_split", type=float, default=0.10)
    ap.add_argument("--letters_only", type=lambda s: str(s).lower() != "false", default=True)
    ap.add_argument("--max_batches", type=int, default=0, help="0 = full validation; else limit for quick runs")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- after loading the model ---
    model = keras.models.load_model(args.model_path)
    mh, mw = model.input_shape[1:3]
    args.img_size = [mh, mw]   # (H, W)  <-- FIXED

    # --- load labels & force class order ---
    labels = json.loads(Path(args.labels_path).read_text())
    idx_to_label = {int(k): v for k, v in labels.items()}
    class_names = [idx_to_label[i] for i in range(len(idx_to_label))]

    datagen = keras.preprocessing.image.ImageDataGenerator(
    validation_split=args.val_split
    )

    # Use ONE flow, locked to labels.json order
    flow = datagen.flow_from_directory(
        directory=str(data_dir),
        target_size=tuple(args.img_size),  # (H,W)
        batch_size=args.batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        classes=class_names,
    )

    # --- remove make_valid_flow() usage below; iterate over `flow` ---
    steps_total = math.ceil(flow.samples / flow.batch_size)
    steps = steps_total if args.max_batches <= 0 else min(args.max_batches, steps_total)

    y_true_idx, y_pred_idx = [], []
    for _ in range(steps):
        x, y = next(flow)
        preds = model.predict(x, verbose=0)
        y_true_idx.extend(np.argmax(y, axis=1))
        y_pred_idx.extend(np.argmax(preds, axis=1))

    y_true_idx = np.array(y_true_idx)
    y_pred_idx = np.array(y_pred_idx)

    # Metrics
    acc = accuracy_score(y_true_idx, y_pred_idx)
    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=list(range(len(class_names))))
    # report = classification_report(y_true_idx, y_pred_idx, target_names=class_names, digits=3)

    # AFTER:
    label_indices = list(range(len(class_names)))
    report = classification_report(
        y_true_idx, y_pred_idx,
        labels=label_indices,
        target_names=class_names,
        digits=3,
        zero_division=0,
    )
    # Save artifacts
    # AFTER:
    
    np.save(out_dir / "confusion_matrix.npy", cm)

    with open(out_dir / "val_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Overall accuracy: {acc:.4f}\n\n")
        f.write(report)

    plot_confusion(cm, class_names, out_dir / "confusion_matrix.png")

    print(f"[EVAL] Overall accuracy: {acc:.4f}")
    print(f"[EVAL] Saved: {out_dir/'val_report.txt'}")
    print(f"[EVAL] Saved: {out_dir/'confusion_matrix.png'}")
