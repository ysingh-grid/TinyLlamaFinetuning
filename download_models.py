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
import time
from pathlib import Path


def _ts() -> str:
    return time.strftime("%H:%M:%S")

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
        "size_note": "~0.7 GB",
    },
    "phi_2": {
        "name": "Phi-2 2.7B 4-bit",
        "hub_id": "mlx-community/phi-2-hf-4bit-mlx",
        "local_path": "./models/phi-2-hf-4bit-mlx",
        "size_note": "~1.6 GB",
    },
    "qwen": {
        "name": "Qwen 1.5 1.8B Chat 4-bit (also used as local judge model)",
        "hub_id": "mlx-community/Qwen1.5-1.8B-Chat-4bit",
        "local_path": "./models/qwen1.5-1.8b-chat-4bit",
        "size_note": "~1.1 GB",
    },
}


def _is_downloaded(local_path: str) -> bool:
    """Return True if the model directory exists and contains config.json."""
    p = Path(local_path)
    return p.is_dir() and (p / "config.json").exists()


def download_model(key: str, spec: dict, force: bool = False) -> bool:
    local = spec["local_path"]
    size_note = spec.get("size_note", "")
    if not force and _is_downloaded(local):
        print(f"[{_ts()}]   ✓ {spec['name']} — already present at {local}")
        return False

    print(f"[{_ts()}]   ↓ Downloading {spec['name']}  ({size_note})")
    print(f"[{_ts()}]     Hub: {spec['hub_id']}  →  {local}")
    print(f"[{_ts()}]     (huggingface_hub will show per-file progress below)")
    Path(local).mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    snapshot_download(
        repo_id=spec["hub_id"],
        local_dir=local,
        local_dir_use_symlinks=False,
    )
    elapsed = time.monotonic() - t0
    print(f"[{_ts()}]     Download complete in {elapsed:.1f}s")
    _patch_tokenizer_config(local)
    return True


def _patch_tokenizer_config(local_path: str) -> None:
    """Replace 'TokenizersBackend' tokenizer_class with 'PreTrainedTokenizerFast'.

    Some mlx-community snapshots store this legacy class name which newer
    transformers versions no longer recognise.  Patching it once at download
    time fixes the issue for every subsequent tool (mlx_lm subprocesses,
    Streamlit, perplexity script, etc.) without needing Python-level shims.
    """
    import json as _json

    cfg_path = Path(local_path) / "tokenizer_config.json"
    if not cfg_path.exists():
        return
    try:
        cfg = _json.loads(cfg_path.read_text(encoding="utf-8"))
        if cfg.get("tokenizer_class") == "TokenizersBackend":
            cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
            cfg_path.write_text(_json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"    Patched tokenizer_class → PreTrainedTokenizerFast")
    except Exception:
        pass


# ── Public helper for import-time use inside other scripts ────────────────────

def ensure_local_models(keys: list[str] | None = None, quiet: bool = False) -> None:
    """Download any missing local models.

    Call this at the top of any script that loads a local model path.
    It is a no-op when all models are already present.

    Args:
        keys: subset of MODELS keys to check (default: all three).
        quiet: suppress "already present" lines.
    """
    target_keys = keys if keys is not None else list(MODELS.keys())
    missing = [k for k in target_keys if not _is_downloaded(MODELS[k]["local_path"])]
    if not missing:
        return

    total_size = " + ".join(MODELS[k].get("size_note", "?") for k in missing)
    print(f"[{_ts()}] [download_models] {len(missing)} model(s) missing — downloading ({total_size})…")
    for k in missing:
        download_model(k, MODELS[k])


def ensure_model_path(local_path: str) -> None:
    """Download whichever registered model lives at *local_path* if missing.

    Useful in scripts that read the model path from a YAML config.
    """
    for spec in MODELS.values():
        if Path(spec["local_path"]).resolve() == Path(local_path).resolve():
            if not _is_downloaded(local_path):
                print(f"[download_models] Model not found at {local_path} — downloading…")
                download_model(spec["local_path"], spec)
            return
    # Not a registered model (e.g. Hub ID or unknown path) — nothing to do.


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

    total_size = " + ".join(MODELS[k].get("size_note", "?") for k in args.models)
    print(f"[{_ts()}] Downloading {len(args.models)} MLX model(s)  ({total_size})…")
    wall_start = time.monotonic()
    downloaded = 0
    for key in args.models:
        spec = MODELS[key]
        if download_model(key, spec, force=args.force):
            downloaded += 1

    skipped = len(args.models) - downloaded
    elapsed = time.monotonic() - wall_start
    print(f"\n[{_ts()}] Done in {elapsed:.1f}s — downloaded: {downloaded}  skipped (already present): {skipped}")


if __name__ == "__main__":
    main()
