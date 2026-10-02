"""Create or update the Hugging Face Space and upload the app plus the model files.

    hf auth login                                   # once, in your own terminal
    python space/deploy.py [--repo 7Pranay77/asl-alphabet-recognizer]

Only the three files the app reads are uploaded from models/; nothing else leaves the
machine. Re-running updates the Space in place.
"""

import argparse
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

ROOT = Path(__file__).resolve().parent.parent
APP_FILES = ["app.py", "requirements.txt", "README.md"]
MODEL_FILES = ["asl_int8.onnx", "labels.json", "calibration.json"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="7Pranay77/asl-alphabet-recognizer")
    args = ap.parse_args()
    api = HfApi()
    print(f"signed in as {api.whoami()['name']}")
    api.create_repo(args.repo, repo_type="space", space_sdk="gradio", exist_ok=True)
    ops = [(ROOT / "space" / f, f) for f in APP_FILES]
    ops += [(ROOT / "models" / f, f"models/{f}") for f in MODEL_FILES]
    for local, _ in ops:
        if not local.is_file():
            raise SystemExit(f"missing {local}")

    api.create_commit(
        repo_id=args.repo,
        repo_type="space",
        operations=[
            CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local))
            for local, remote in ops
        ],
        commit_message="Deploy app and INT8 model",
    )
    print(f"https://huggingface.co/spaces/{args.repo}")


if __name__ == "__main__":
    main()
