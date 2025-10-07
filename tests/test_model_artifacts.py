import json
from pathlib import Path
import numpy as np
import cv2
from tensorflow import keras
import pytest

MODEL = Path("models/cnn_model.h5")
LABELS = Path("models/labels.json")
A_DIR = Path("data/asl_alphabet/asl_alphabet_train/A")

@pytest.mark.skipif(not (MODEL.exists() and LABELS.exists()), reason="Model not trained yet")
def test_single_image_inference():
    model = keras.models.load_model(MODEL)
    idx_to_label = {int(k): v for k, v in json.loads(LABELS.read_text()).items()}
    img_path = next(A_DIR.glob("*.jpg"))
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (64,64)).astype("float32")/255.0
    preds = model.predict(img[None,...], verbose=0)[0]
    top = int(np.argmax(preds))
    assert idx_to_label[top] in [chr(i) for i in range(ord("A"), ord("Z")+1)]
