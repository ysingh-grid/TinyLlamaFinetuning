#!/usr/bin/env python3
"""Launch llama.cpp `llama-server` for a quantized GGUF (local / edge only)."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models" / "model-Q4_K_M.gguf"
DEFAULT_HF_REPO = "ysingh-aiml/tinyllama-alpaca-lora-gguf"
QUANT_FILES = {
    "q4_k_m": "model-Q4_K_M.gguf",
    "q5_k_m": "model-Q5_K_M.gguf",
    "q8_0": "model-Q8_0.gguf",
}


def download_gguf(
    repo_id: str,
    filename: str,
    dest_dir: Path,
    revision: str | None = None,
) -> Path:
    from huggingface_hub import hf_hub_download

    dest_dir.mkdir(parents=True, exist_ok=True)
    out = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(dest_dir),
        local_dir_use_symlinks=False,
        revision=revision,
    )
    return Path(out)


def resolve_llama_server() -> str:
    env = os.environ.get("LLAMA_SERVER", "").strip()
    if env:
        return env
    which = shutil.which("llama-server")
    return which or ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Start llama.cpp llama-server with a GGUF model.")
    parser.add_argument(
        "--quant",
        choices=sorted(QUANT_FILES.keys()),
        default=None,
        help="Pick a Hub filename under --hf-repo (sets model to models/<file>).",
    )
    parser.add_argument("--model", type=Path, default=None, help="Path to .gguf (default: models/model-Q4_K_M.gguf)")
    parser.add_argument(
        "--hf-repo",
        default=os.environ.get("TASK4_GGUF_REPO", DEFAULT_HF_REPO),
        help="Hugging Face model repo id for --fetch (default: env TASK4_GGUF_REPO or built-in)",
    )
    parser.add_argument("--revision", default=None, help="Hub git revision (branch / tag / commit) for download")
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Do not download from the Hub if the model file is missing",
    )
    parser.add_argument(
        "--fetch-only",
        action="store_true",
        help="Download the GGUF from the Hub then exit (no llama-server)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--ctx-size", type=int, default=2048)
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=0,
        help="GPU/Metal offload layer count; 0 = CPU only",
    )
    args = parser.parse_args()

    if args.quant:
        filename = QUANT_FILES[args.quant]
        model_path = (ROOT / "models" / filename).resolve()
    elif args.model is not None:
        model_path = args.model.resolve()
    else:
        model_path = DEFAULT_MODEL.resolve()

    if not model_path.is_file():
        if args.no_fetch:
            print(
                f"Model not found: {model_path}\n"
                "Remove --no-fetch to download from the Hub, or place a .gguf at this path.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        print(f"Downloading {model_path.name} from {args.hf_repo} …", file=sys.stderr)
        try:
            downloaded = download_gguf(
                args.hf_repo,
                model_path.name,
                model_path.parent,
                revision=args.revision,
            )
        except Exception as e:
            print(f"Download failed: {e}", file=sys.stderr)
            raise SystemExit(1) from e
        model_path = downloaded.resolve()
        if not model_path.is_file():
            print(f"Expected file after download: {model_path}", file=sys.stderr)
            raise SystemExit(1)
        print(f"Model ready: {model_path}", file=sys.stderr)

    if args.fetch_only:
        print(model_path)
        raise SystemExit(0)

    exe = resolve_llama_server()
    if not exe:
        print(
            "llama-server not found. Build llama.cpp and set LLAMA_SERVER=/path/to/llama-server "
            "or put `llama-server` on PATH.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    cmd = [
        exe,
        "-m",
        str(model_path),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--threads",
        str(args.threads),
        "--ctx-size",
        str(args.ctx_size),
        "--n-gpu-layers",
        str(args.n_gpu_layers),
        "--parallel",
        "1",
        "--no-warmup",
    ]
    print("Running:", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
