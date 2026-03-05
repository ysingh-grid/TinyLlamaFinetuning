#!/usr/bin/env python3
"""
download_models.py

Download all required MLX models to their expected local paths.
Skips any model whose directory already contains a config.json (idempotent).

Usage:
    python download_models.py              # all models
    python download_models.py --models tinyllama_4bit phi_2 qwen
    make download-models                   # all models via Makefile
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("ERROR: huggingface_hub not found. Run: pip install huggingface-hub")
    sys.exit(1)

# ── Model registry ────────────────────────────────────────────────────────────
# hub_id: exact mlx-community repo ID (verified from model READMEs)
# local_path: where the project expects it (referenced by evaluation/models.json)
MODELS: dict[str, dict] = {
    "tinyllama_4bit": {
        "name": "TinyLlama 1.1B Chat 4-bit (QLoRA base + eval)",
        "hub_id": "mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit",
        "local_path": "./models/tinyllama-4bit-base",
    },
    "phi_2": {
        "name": "Phi-2 2.7B 4-bit",
        "hub_id": "mlx-community/phi-2-hf-4bit-mlx",
        "local_path": "./models/phi-2-hf-4bit-mlx",
    },
    "qwen": {
        "name": "Qwen 1.5 1.8B Chat 4-bit (also used as local judge model)",
        "hub_id": "mlx-community/Qwen1.5-1.8B-Chat-4bit",
        "local_path": "./models/qwen1.5-1.8b-chat-4bit",
    },
}


def _is_downloaded(local_path: str) -> bool:
    """Return True if the model directory exists and contains config.json."""
    p = Path(local_path)
    return p.is_dir() and (p / "config.json").exists()


def download_model(key: str, spec: dict, force: bool = False) -> bool:
    local = spec["local_path"]
    if not force and _is_downloaded(local):
        print(f"  ✓ {spec['name']} — already present at {local}")
        return False

    print(f"  ↓ {spec['name']}")
    print(f"    Hub: {spec['hub_id']}  →  local: {local}")
    Path(local).mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=spec["hub_id"],
        local_dir=local,
        local_dir_use_symlinks=False,
    )
    print(f"    Done.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Download required MLX models.")
    parser.add_argument(
        "--models",
        nargs="*",
        choices=list(MODELS.keys()),
        default=list(MODELS.keys()),
        help="Which models to download (default: all). Choices: " + ", ".join(MODELS),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if already present.",
    )
    args = parser.parse_args()

    print("Downloading MLX models...")
    downloaded = 0
    for key in args.models:
        spec = MODELS[key]
        if download_model(key, spec, force=args.force):
            downloaded += 1

    skipped = len(args.models) - downloaded
    print(f"\nDone. Downloaded: {downloaded}  Skipped (already present): {skipped}")


if __name__ == "__main__":
    main()
