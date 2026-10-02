import subprocess
import sys
from pathlib import Path

import pytest

PREDICT = Path("predict.py")
MODEL = Path("models/model.keras")
LABELS = Path("models/labels.json")
A_DIR = Path("data/asl_alphabet/asl_alphabet_train/A")

@pytest.mark.skipif(not (MODEL.exists() and LABELS.exists() and PREDICT.exists()), reason="Artifacts or script missing")
def test_predict_cli_on_single_image():
    img = next(A_DIR.glob("*.jpg"))
    cmd = [sys.executable, str(PREDICT),
           "--model_path", str(MODEL),
           "--labels_path", str(LABELS),
           "--img_size", "160", "160",
           "--test_image", str(img)]
    cp = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert cp.returncode == 0, cp.stderr
    # should contain "Top-1: X (YY.Y%)"
    assert "Top-1:" in cp.stdout
