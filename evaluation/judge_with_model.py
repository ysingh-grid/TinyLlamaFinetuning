#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.pipeline import run_model_judging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge pairwise tasks using a local MLX judge model.")
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = run_model_judging(
        tasks_path=args.tasks,
        out_path=args.out,
        judge_model=args.judge_model,
        seed=args.seed,
        max_tokens=args.max_tokens,
    )
    print(f"Wrote judgments: {out_path}")


if __name__ == "__main__":
    main()
