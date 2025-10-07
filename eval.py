import argparse
import json
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

def make_valid_flow(data_dir: Path, img_size, batch_size, val_split, labels_path):
    # Load class names from labels.json
    with open(labels_path, "r") as f:
        class_names = json.load(f)

    datagen = keras.preprocessing.image.ImageDataGenerator(validation_split=val_split)
    val_flow = datagen.flow_from_directory(
        str(data_dir),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
        subset="validation",
        classes=class_names,  # Force class order from labels.json
    )
    return val_flow, class_names

def plot_confusion(cm, class_names, out_png):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(out_png)
    plt.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True, help="Path to the trained model (.keras)")
    p.add_argument("--labels_path", type=str, required=True, help="Path to labels.json")
    p.add_argument("--data_dir", type=str, required=True, help="Path to validation dataset")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--img_size", nargs=2, type=int, default=[160, 160])
    p.add_argument("--val_split", type=float, default=0.10)
    args = p.parse_args()

    # Paths
    model_path = Path(args.model_path)
    labels_path = Path(args.labels_path)
    data_dir = Path(args.data_dir)

    # Load model
    model = keras.models.load_model(model_path)
    print(f"[INFO] Loaded model from {model_path}")

    # Validation dataset
    img_size = tuple(args.img_size)
    val_flow, class_names = make_valid_flow(data_dir, img_size, args.batch_size, args.val_split, labels_path)

    # Predictions
    y_true = val_flow.classes
    y_pred = model.predict(val_flow, verbose=1)
    y_pred = np.argmax(y_pred, axis=1)

    # Metrics
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names)
    accuracy = accuracy_score(y_true, y_pred)

    # Save results
    out_dir = model_path.parent
    with open(out_dir / "val_report.txt", "w") as f:
        f.write(report)
    np.save(out_dir / "confusion_matrix.npy", cm)
    plot_confusion(cm, class_names, out_dir / "confusion_matrix.png")

    # Print overall accuracy
    print(f"[INFO] Overall accuracy: {accuracy:.4f}")
    print(f"[DONE] Results saved to {out_dir}")