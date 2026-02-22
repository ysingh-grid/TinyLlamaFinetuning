#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def score_row(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    n = len(stripped)
    alpha = sum(ch.isalpha() for ch in stripped)
    digits = sum(ch.isdigit() for ch in stripped)
    weird = sum(ch in "{}[]<>|\\`~" for ch in stripped)

    # Heuristic: collapsed generations tend to be digit/symbol-heavy with low alphabetic content.
    alpha_ratio = alpha / n
    digit_ratio = digits / n
    weird_ratio = weird / n

    if alpha_ratio < 0.35:
        return False
    if digit_ratio > 0.20:
        return False
    if weird_ratio > 0.05:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Heuristic smoke check for generated response quality.")
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument(
        "--max-bad-rate",
        type=float,
        default=0.20,
        help="Fail if bad_rows / total_rows exceeds this threshold.",
    )
    args = parser.parse_args()

    rows = []
    with args.responses.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))

    if not rows:
        raise SystemExit("No rows found in responses file.")

    bad = 0
    for row in rows:
        if not score_row(str(row.get("response", ""))):
            bad += 1

    bad_rate = bad / len(rows)
    print(f"rows={len(rows)} bad={bad} bad_rate={bad_rate:.3f}")
    if bad_rate > args.max_bad_rate:
        raise SystemExit(
            f"Smoke quality check failed: bad_rate={bad_rate:.3f} > max_bad_rate={args.max_bad_rate:.3f}"
        )


if __name__ == "__main__":
    main()
