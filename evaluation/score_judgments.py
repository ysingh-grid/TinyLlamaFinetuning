#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.pipeline import run_scoring


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score pairwise judgments and generate reports.")
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pair_json, rollup_json, report_md = run_scoring(
        key_path=args.key,
        judgments_path=args.judgments,
        out_dir=args.out_dir,
        seed=args.seed,
    )
    print(f"Wrote pair metrics: {pair_json}")
    print(f"Wrote model rollup: {rollup_json}")
    print(f"Wrote markdown report: {report_md}")


if __name__ == "__main__":
    main()
