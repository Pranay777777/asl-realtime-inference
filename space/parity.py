"""Parity gate: the browser pipeline must match the Python one before a deploy.

Runs the 26 letter images of the Kaggle test folder through pipeline.py (cv2 + onnxruntime)
and through static/pipeline.js in headless Chromium (onnxruntime-web, wasm), both on the
centre-square path, and compares: top-1 must agree on all and the largest difference in any
calibrated probability must stay under 0.02. Exit 1 otherwise.

    python space/parity.py            # needs `npm install` in space/ once
"""

from __future__ import annotations

import functools
import http.server
import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

from pipeline import Recognizer, centre_box, model_input
from stage import ROOT, SPACE, stage

TEST_DIR = ROOT / "data" / "asl_alphabet" / "asl_alphabet_test"
MAX_DIFF = 0.02


def letters() -> list[Path]:
    images = sorted(p for p in TEST_DIR.glob("*_test.jpg") if len(p.stem.split("_")[0]) == 1)
    if len(images) != 26:
        raise SystemExit(f"expected 26 letter images in {TEST_DIR}, found {len(images)}")
    return images


def python_run(rec: Recognizer, images: list[Path]) -> dict[str, dict]:
    out = {}
    for path in images:
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        box = centre_box(rgb.shape[1], rgb.shape[0])
        out[path.name] = {"probs": rec.probs(rgb, box), "input": model_input(rgb, box)[0]}
    return out


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args: object) -> None:
        pass


def browser_run(site: Path, images: list[Path]) -> tuple[dict[str, dict], list[dict]]:
    (site / "parity").mkdir(exist_ok=True)
    for path in images:
        shutil.copy2(path, site / "parity" / path.name)
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(_QuietHandler, directory=str(site))
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    names, out = site / "names.json", site / "browser.json"
    names.write_text(json.dumps([p.name for p in images]))
    try:
        url = f"http://127.0.0.1:{server.server_port}/index.html"
        node = shutil.which("node") or "node"
        cmd = [node, str(SPACE / "parity.mjs"), url, str(site), str(names), str(out)]
        subprocess.run(cmd, check=True, cwd=SPACE, timeout=600)  # noqa: S603
    finally:
        server.shutdown()
    result = json.loads(out.read_text())
    probs = {
        k: {"probs": np.asarray(v["probs"]), "input": np.asarray(v["input"], dtype=np.float32)}
        for k, v in result["probs"].items()
    }
    return probs, result["ui"]


def main() -> int:
    images = letters()
    site = SPACE / ".stage" / "parity"
    shutil.rmtree(site, ignore_errors=True)
    stage(site)
    rec = Recognizer(ROOT / "models")
    py = python_run(rec, images)
    js, ui = browser_run(site, images)
    agree, worst, pixels = 0, 0.0, 0.0
    print(f"{'image':<12} {'python':>12} {'browser':>12} {'max |dp|':>9} {'max |dpx|':>9}")
    for path in images:
        a, b = py[path.name]["probs"], js[path.name]["probs"]
        dpx = float(np.max(np.abs(py[path.name]["input"].ravel() - js[path.name]["input"])))
        pixels = max(pixels, dpx)
        ia, ib = int(np.argmax(a)), int(np.argmax(b))
        diff = float(np.max(np.abs(a - b)))
        agree += ia == ib
        worst = max(worst, diff)
        la, lb = rec.labels[ia], rec.labels[ib]
        print(f"{path.name:<12} {la:>3} {a[ia]:.4f}   {lb:>3} {b[ib]:.4f}   {diff:9.6f} {dpx:9.1f}")
    print("\nUI checks (headless Edge):")
    for c in ui:
        print(f"  {'ok  ' if c['ok'] else 'FAIL'} {c['name']}: {c['detail']}")
    ok = agree == len(images) and worst < MAX_DIFF and all(c["ok"] for c in ui)
    print(f"top-1 agreement {agree}/{len(images)}; max probability difference {worst:.6f} "
          f"(limit {MAX_DIFF}); largest model-input pixel difference {pixels:.1f} of 255 "
          f"-> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
