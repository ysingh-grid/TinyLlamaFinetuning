#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.pipeline import run_pairing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create blind pairwise judging tasks from model responses.")
    parser.add_argument("--responses-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--pairs",
        default=None,
        help="Comma-separated list: modelA:modelB,modelC:modelD. Omit for all pair combinations.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-prompts-per-pair", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks_path, key_path = run_pairing(
        responses_dir=args.responses_dir,
        out_dir=args.out_dir,
        raw_pairs=args.pairs,
        seed=args.seed,
        max_prompts_per_pair=args.max_prompts_per_pair,
    )
    print(f"Wrote judging tasks: {tasks_path}")
    print(f"Wrote judging key: {key_path}")


if __name__ == "__main__":
    main()
