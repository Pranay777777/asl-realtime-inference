"""Export the Keras model to ONNX (plan step 83).

    python -m inference.export_onnx --model models/model.keras --out models/asl_fp32.onnx

The model's own Rescaling layer stays inside, so the ONNX input is the same as
the Keras one: RGB pixels 0-255, float32, shape (batch, H, W, 3). Augmentation
layers are traced with training=False and vanish.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def export(model_path: Path, out: Path, opset: int = 17) -> tuple[int, int]:
    import tensorflow as tf
    import tf2onnx
    from tensorflow import keras

    model = keras.models.load_model(model_path)
    height, width = model.input_shape[1:3]
    spec = tf.TensorSpec((None, height, width, 3), tf.float32, name="image")

    @tf.function(input_signature=[spec])
    def serve(image: tf.Tensor) -> tf.Tensor:
        return model(image, training=False)

    out.parent.mkdir(parents=True, exist_ok=True)
    tf2onnx.convert.from_function(serve, input_signature=[spec], opset=opset, output_path=str(out))

    import onnxruntime as ort

    probe = np.random.default_rng(0).uniform(0, 255, (4, height, width, 3)).astype(np.float32)
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    onnx_probs = session.run(None, {session.get_inputs()[0].name: probe})[0]
    keras_probs = model(probe, training=False).numpy()
    gap = float(np.abs(onnx_probs - keras_probs).max())
    if gap > 1e-3:
        raise SystemExit(f"ONNX and Keras disagree by {gap:.2e} on the same input")
    print(f"{out} ({out.stat().st_size / 1e6:.1f} MB), input {height}x{width}, max diff {gap:.1e}")
    return height, width


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/model.keras"))
    parser.add_argument("--out", type=Path, default=Path("models/asl_fp32.onnx"))
    args = parser.parse_args()
    export(args.model, args.out)


if __name__ == "__main__":
    main()
