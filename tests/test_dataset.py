from pathlib import Path

import pytest

DATA_DIR = Path("data/asl_alphabet/asl_alphabet_train")
# The Kaggle dataset is downloaded locally and never committed; CI has no copy.
pytestmark = pytest.mark.skipif(not DATA_DIR.exists(), reason="dataset not downloaded")

def test_dataset_dirs_present():
    assert DATA_DIR.exists(), f"Missing: {DATA_DIR}"
    # We train on A–Z, but the archive also includes del/nothing/space.
    letters = [chr(i) for i in range(ord("A"), ord("Z")+1)]
    missing = [c for c in letters if not (DATA_DIR / c).exists()]
    assert not missing, f"Missing class folders: {missing}"

def test_sample_images_exist():
    # At least one image in 'A'
    a_imgs = list((DATA_DIR / "A").glob("*.jpg"))
    assert len(a_imgs) > 0, "No images found in class A"
