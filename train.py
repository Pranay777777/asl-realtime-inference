# train.py  (tf.data version; captures class_names before prefetch)
import argparse, json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ASCII_A_TO_Z = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

def build_model(input_shape, num_classes, backbone="mobilenetv2"):
    inputs = keras.Input(shape=input_shape)

    # Train-time augmentation
    aug = keras.Sequential([
        layers.RandomRotation(0.10),
        layers.RandomZoom(0.10, 0.10),
        layers.RandomTranslation(0.10, 0.10),
        layers.RandomContrast(0.12),
    ], name="augmentation")
    x = aug(inputs)

    if backbone == "cnn":
        # your original small CNN (kept for reference)
        x = layers.Rescaling(1./255)(x)
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.30)(x)

        x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.35)(x)

        x = layers.Flatten()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.50)(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        model = keras.Model(inputs, outputs, name="asl_cnn")
        return model, None

    # ---- Transfer learning: MobileNetV2 (best speed/accuracy tradeoff) ----
    # Scale to [-1,1] which is what MobileNetV2 expects
    x = layers.Rescaling(scale=1./127.5, offset=-1)(x)

    base = keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    base.trainable = False
    base._name = "backbone"   # make it easy to find later

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.20)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="probs")(x)
    model = keras.Model(inputs, outputs, name="asl_mobilenetv2")
    return model, base


def make_datasets(data_dir, img_size, batch_size, val_split, letters_only):
    classes = ASCII_A_TO_Z if letters_only else None

    # Create raw datasets first
    raw_train = keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=classes,      # restrict to A..Z when provided
        color_mode="rgb",
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True,
        seed=1337,
        validation_split=val_split,
        subset="training",
    )
    raw_val = keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=classes,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=img_size,
        shuffle=False,
        seed=1337,
        validation_split=val_split,
        subset="validation",
    )

    # Capture class_names BEFORE wrapping with cache/prefetch
    class_names = raw_train.class_names

    # Performance: cache & prefetch
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = raw_train.cache().prefetch(AUTOTUNE)
    val_ds   = raw_val.cache().prefetch(AUTOTUNE)
    return train_ds, val_ds, class_names

def compute_class_weights_ds(train_ds, num_classes):
    """
    Fast class weights from dataset batches:
    weight_i = total / (num_classes * count_i)
    """
    counts = np.zeros((num_classes,), dtype=np.int64)
    for _x, y in train_ds:
        counts += y.numpy().sum(axis=0).astype(np.int64)
    total = counts.sum()
    counts = np.maximum(counts, 1)
    weights = total / (num_classes * counts.astype(np.float64))
    return {i: float(w) for i, w in enumerate(weights)}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True,
                   help=r"e.g. data\asl_alphabet\asl_alphabet_train")
    p.add_argument("--out_dir", type=str, default="models")
    p.add_argument("--img_size", nargs=2, type=int, default=[64, 64])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--val_split", type=float, default=0.10)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--letters_only", type=lambda s: str(s).lower() != "false",
                   default=True, help="True = restrict to A–Z")
    p.add_argument("--class_weights", choices=["off", "auto"], default="auto",
                   help="off = none, auto = compute from training ds")
    p.add_argument("--backbone", choices=["cnn", "mobilenetv2"], default="mobilenetv2")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--label_smoothing", type=float, default=0.10)
    p.add_argument("--freeze_epochs", type=int, default=8)
    p.add_argument("--fine_tune_epochs", type=int, default=12)
    p.add_argument("--fine_tune_at", type=int, default=100, help="unfreeze from this layer index; -1 to keep frozen")
    p.add_argument("--resume_from", type=str, default="", help="Path to a saved model to resume (.h5/.keras)")


    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_w, img_h = args.img_size
    img_size = (img_w, img_h)
    input_shape = (img_h, img_w, 3)

    # Datasets + class_names
    train_ds, val_ds, class_names = make_datasets(
        str(data_dir),
        img_size=img_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        letters_only=args.letters_only
    )

    num_classes = len(class_names)
    print(f"[INFO] classes ({num_classes}):", {name: i for i, name in enumerate(class_names)})

    # Save label map (idx -> label)
    labels = {i: name for i, name in enumerate(class_names)}
    with open(out_dir / "labels.json", "w") as f:
        json.dump(labels, f, indent=2)


    # -------- Model (build OR resume) --------
    resume_path = Path(args.resume_from) if args.resume_from else None
    model, base = None, None

    # Replace your try_get_backbone(...) helper with this:
    def find_backbone(m):
        # 1) exact name used by our current code
        try:
            return m.get_layer("backbone")
        except Exception:
            pass
        # 2) common MobileNetV2 names in older checkpoints
        for l in m.layers:
            n = getattr(l, "name", "")
            if "mobilenetv2" in n.lower() or "mobilenet" in n.lower():
                return l
        return None


    if resume_path and resume_path.exists():
        print(f"[RESUME] Found: {resume_path}")
        suf = resume_path.suffix.lower()
        try:
            if suf == ".keras":
                model = keras.models.load_model(resume_path)
                base = find_backbone(model)
                print("[RESUME] Loaded full model (.keras).")
            elif suf in (".h5", ".hdf5"):
                model = keras.models.load_model(resume_path)   # may fail on bad H5
                base = find_backbone(model)
                print("[RESUME] Loaded full model (.h5).")
            else:
                raise ValueError(f"Unsupported resume file: {resume_path}")
        except Exception as e:
            print(f"[RESUME] Full-model load failed ({e}). Falling back to weights load...")
            model, base = build_model(input_shape, num_classes, backbone=args.backbone)
            try:
                model.load_weights(str(resume_path))           # may fail if file is empty/truncated
                base = find_backbone(model)
                print("[RESUME] Loaded weights into freshly built model.")
            except Exception as e2:
                print(f"[RESUME] Weights load failed ({e2}). Proceeding with fresh model.")
                # keep the freshly built model
    else:
        model, base = build_model(input_shape, num_classes, backbone=args.backbone)

        # Compile for Phase 1 (frozen backbone)
    optimizer_p1 = keras.optimizers.Adam(learning_rate=float(args.lr))
    loss_p1 = keras.losses.CategoricalCrossentropy(label_smoothing=float(args.label_smoothing))
    model.compile(
        optimizer=optimizer_p1,
        loss=loss_p1,
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")]
    )

    # Callbacks (unchanged)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "model_best.keras"),  # native Keras format
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=args.patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(2, args.patience // 2), min_lr=1e-6
        ),
    ]


    # Class weights (unchanged)
    class_weight = None
    if args.class_weights == "auto":
        print("[INFO] Computing class weights (one pass over training ds)...")
        class_weight = compute_class_weights_ds(train_ds, num_classes)
        print("[INFO] Class weights:", class_weight)

    # -------- Phase 1: train head with backbone frozen --------
if args.freeze_epochs > 0:
    print("[TRAIN] Phase 1: frozen backbone")
    hist1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=args.freeze_epochs, callbacks=callbacks,
        class_weight=class_weight, verbose=1,
    )
else:
    print("[TRAIN] Skipping Phase 1 (freeze_epochs=0)")


    # -------- Phase 2: fine-tune part of the backbone --------
    if base is not None and args.fine_tune_at >= 0:
        print(f"[TRAIN] Phase 2: unfreezing backbone from layer {args.fine_tune_at}")
        base.trainable = True
        for l in base.layers[:args.fine_tune_at]:
            l.trainable = False

        # Lower LR for fine-tuning
        # Lower LR for fine-tuning (explicit kw args + metric object)
        optimizer_ft = keras.optimizers.Adam(learning_rate=float(args.lr) * 0.1)
        loss_ft = keras.losses.CategoricalCrossentropy(label_smoothing=float(args.label_smoothing))

        model.compile(
            optimizer=optimizer_ft,
            loss=loss_ft,
            metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")]
        )

        hist2 = model.fit(
            train_ds, validation_data=val_ds,
            epochs=args.freeze_epochs + args.fine_tune_epochs,
            initial_epoch=args.freeze_epochs,
            callbacks=callbacks, class_weight=class_weight, verbose=1,
        )

    model.save(out_dir / "model.keras")

    print("[DONE] Model saved to", out_dir / "model.keras")
    print("[DONE] Labels saved to", out_dir / "labels.json")
