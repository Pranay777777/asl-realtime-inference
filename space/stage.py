"""Assemble the static Space: the committed web files plus the model files from models/.

The model and calibration never enter git; they are copied here for a local run, the parity
check and the upload.
"""

from __future__ import annotations

import shutil
from pathlib import Path

SPACE = Path(__file__).resolve().parent
ROOT = SPACE.parent
WEB_FILES = ["index.html", "app.js", "pipeline.js", "style.css", "labels.json"]
MODEL_FILES = ["asl_int8.onnx", "calibration.json"]


def stage(dest: Path) -> list[Path]:
    """Copy every file the static Space serves into dest; return them."""
    dest.mkdir(parents=True, exist_ok=True)
    sources = [SPACE / "static" / f for f in WEB_FILES] + [ROOT / "models" / f for f in MODEL_FILES]
    missing = [str(s) for s in sources if not s.is_file()]
    if missing:
        raise SystemExit(f"missing: {', '.join(missing)}")
    return [Path(shutil.copy2(s, dest / s.name)) for s in sources]
