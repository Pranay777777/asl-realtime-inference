"""Quantize the ONNX model to INT8 with real calibration images (plan step 83).

    python -m inference.quantize --data_dir <asl_alphabet_train> --fp32 models/asl_fp32.onnx

Static QDQ quantization: weights per-channel int8, activations uint8, ranges
calibrated on images from the *training* split (see inference/data.py), so the
images used for scoring are not used for calibration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from inference.data import load_split


def quantize(fp32: Path, int8: Path, data_dir: Path, samples: int = 300) -> None:
    import onnxruntime as ort
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    session = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
    feed = session.get_inputs()[0]
    height, width = int(feed.shape[1]), int(feed.shape[2])
    images, _ = load_split(data_dir, (height, width), "training", limit=samples)

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self.batches = iter(images[i : i + 1].astype(np.float32) for i in range(len(images)))

        def get_next(self) -> dict[str, np.ndarray] | None:
            batch = next(self.batches, None)
            return None if batch is None else {feed.name: batch}

    prepared = int8.with_name(int8.stem + "_prep.onnx")
    quant_pre_process(str(fp32), str(prepared))
    quantize_static(
        str(prepared),
        str(int8),
        Reader(),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
    )
    prepared.unlink(missing_ok=True)
    print(f"{int8} ({int8.stat().st_size / 1e6:.1f} MB) from {len(images)} calibration images")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--fp32", type=Path, default=Path("models/asl_fp32.onnx"))
    parser.add_argument("--out", type=Path, default=Path("models/asl_int8.onnx"))
    parser.add_argument("--samples", type=int, default=300)
    args = parser.parse_args()
    quantize(args.fp32, args.out, args.data_dir, args.samples)


if __name__ == "__main__":
    main()
