#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.pipeline import load_models_config, load_prompts, run_generation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate model responses for evaluation prompts.")
    parser.add_argument("--models-config", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print per-model progress every N prompts (0 disables).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = load_models_config(args.models_config)
    prompts = load_prompts(args.prompts)
    output_map = run_generation(
        models=models,
        prompts=prompts,
        out_dir=args.out_dir,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        overwrite=args.overwrite,
        progress_every=args.progress_every,
    )
    print("Generated response files:")
    for name, path in sorted(output_map.items()):
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
