import numpy as np
import predict  # safe: main guarded by if __name__ == "__main__"

def test_preprocess_frame_shapes():
    # fake 480x640 frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    roi = (50, 60, 224, 224)
    out = predict.preprocess_frame(frame, roi, (64, 64))
    assert out is not None
    assert out.shape == (1, 64, 64, 3)
    assert out.dtype == np.float32
    assert 0.0 <= out.min() <= out.max() <= 1.0
