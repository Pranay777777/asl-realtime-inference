import subprocess
import sys
from pathlib import Path

import pytest

MODEL = Path("models/model.keras")
LABELS = Path("models/labels.json")
EVAL = Path("eval.py")
DATA = Path("data/asl_alphabet/asl_alphabet_train")
OUT = Path("models")

@pytest.mark.skipif(not (MODEL.exists() and LABELS.exists() and EVAL.exists()), reason="Artifacts or eval.py missing")
def test_eval_generates_artifacts(tmp_path):
    # run quick eval (1 batch) into normal models/ to keep it simple
    cmd = [sys.executable, str(EVAL),
           "--data_dir", str(DATA),
           "--model_path", str(MODEL),
           "--labels_path", str(LABELS),
           "--img_size", "160", "160",
           "--batch_size", "64",
           "--val_split", "0.1",
           "--letters_only", "true",
           "--max_batches", "1",
           "--out_dir", str(tmp_path)]  # never overwrite the real report in models/
    cp = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert cp.returncode == 0, cp.stderr
    assert (tmp_path / "val_report.txt").exists()
    assert (tmp_path / "confusion_matrix.png").exists()
