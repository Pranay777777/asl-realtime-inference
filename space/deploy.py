"""Publish the static Space (free tier): the browser app plus the model files.

    hf auth login                                   # once, in your own terminal
    cd space && npm install                         # once: headless browser for the gate
    python space/deploy.py [--repo 7Pranay77/asl-alphabet-recognizer]

The parity gate (space/parity.py) runs first and must pass. Only the files the page serves
are uploaded - the web files from space/static/ and asl_int8.onnx + calibration.json from
models/ - plus the Space README. Re-running updates the Space in place.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

import parity
from stage import SPACE, stage


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="7Pranay77/asl-alphabet-recognizer")
    args = ap.parse_args()

    if parity.main() != 0:
        print("parity gate failed - not deploying")
        return 1

    site = SPACE / ".stage" / "deploy"
    shutil.rmtree(site, ignore_errors=True)
    files = stage(site) + [Path(shutil.copy2(SPACE / "README.md", site / "README.md"))]

    api = HfApi()
    print(f"signed in as {api.whoami()['name']}")
    api.create_repo(args.repo, repo_type="space", space_sdk="static", exist_ok=True)
    api.create_commit(
        repo_id=args.repo,
        repo_type="space",
        operations=[CommitOperationAdd(path_in_repo=f.name, path_or_fileobj=str(f)) for f in files],
        commit_message="Deploy static app and INT8 model",
    )
    for f in files:
        print(f"uploaded {f.name} ({f.stat().st_size:,} bytes)")
    print(f"https://huggingface.co/spaces/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
