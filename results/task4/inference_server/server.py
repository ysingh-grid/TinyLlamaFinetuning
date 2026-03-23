#!/usr/bin/env python3
"""Launch a llama.cpp server for the smallest working GGUF model."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LLAMA_SERVER = ROOT / "llama.cpp" / "build" / "bin" / "llama-server"
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "model-Q4_K_M.gguf"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--ctx-size", type=int, default=2048)
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    args = parser.parse_args()
    cmd = [
        str(LLAMA_SERVER),
        "-m",
        str(args.model),
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
