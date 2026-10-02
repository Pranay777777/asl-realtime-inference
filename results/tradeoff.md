Scored on 2600 images (16 of 26 letters from a local copy of the training data; may overlap training images, so accuracy is optimistic), input 160x160; batch size 1 on Intel64 Family 6 Model 140 Stepping 1, GenuineIntel (8 threads).

| Variant | Size (MB) | Accuracy | Δ accuracy | p50 (ms) | p95 (ms) | FPS | Speed-up |
|---|---:|---:|---:|---:|---:|---:|---:|
| Keras FP32 | 24.96 | 99.92% | +0.00 pp | 120.1 | 169.22 | 7.8 | 1.0x |
| ONNX FP32 | 9.04 | 99.92% | +0.00 pp | 2.06 | 2.69 | 471.1 | 60.4x |
| ONNX INT8 | 2.69 | 99.96% | +0.04 pp | 1.33 | 1.56 | 757.3 | 97.0x |
