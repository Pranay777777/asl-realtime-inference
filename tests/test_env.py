def test_imports():
    import tensorflow as tf, cv2, PIL, numpy as np, sklearn  # noqa
    assert tf.__version__ >= "2.10"
