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
    parser.add_argument(
        "--counterbalance-sides",
        dest="counterbalance_sides",
        action="store_true",
        default=True,
        help="Balance model_1 left/right assignment per pair (default: enabled).",
    )
    parser.add_argument(
        "--no-counterbalance-sides",
        dest="counterbalance_sides",
        action="store_false",
        help="Disable balanced side assignment and use independent random left/right choices.",
    )
    parser.add_argument(
        "--shuffle-tasks",
        dest="shuffle_tasks",
        action="store_true",
        default=True,
        help="Shuffle final judging task order across pairs/prompts (default: enabled).",
    )
    parser.add_argument(
        "--no-shuffle-tasks",
        dest="shuffle_tasks",
        action="store_false",
        help="Keep judging tasks grouped by pair then prompt_id.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks_path, key_path = run_pairing(
        responses_dir=args.responses_dir,
        out_dir=args.out_dir,
        raw_pairs=args.pairs,
        seed=args.seed,
        max_prompts_per_pair=args.max_prompts_per_pair,
        counterbalance_sides=args.counterbalance_sides,
        shuffle_tasks=args.shuffle_tasks,
    )
    print(f"Wrote judging tasks: {tasks_path}")
    print(f"Wrote judging key: {key_path}")
    print(f"Wrote pairing summary: {args.out_dir / 'pairing_summary.json'}")


if __name__ == "__main__":
    main()
