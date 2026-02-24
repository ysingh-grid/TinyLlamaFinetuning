#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.pipeline import (
    load_models_config,
    load_prompts,
    run_generation,
    run_model_judging,
    run_pairing,
    run_scoring,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full evaluation pipeline: generation -> pairing -> judging -> scoring.")
    parser.add_argument("--models-config", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=0,
        help="Limit total prompts used for generation (0 = use all).",
    )
    parser.add_argument("--out-root", type=Path, default=Path("evaluation/runs"))
    parser.add_argument(
        "--pairs",
        default=None,
        help="Comma-separated list: modelA:modelB,modelC:modelD. Omit for all combinations.",
    )
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
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print per-model progress every N prompts during generation (0 disables).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument(
        "--judge-mode",
        choices=["manual", "model"],
        default="manual",
        help="manual: stop after pairing and wait for judgments file. model: auto-judge with --judge-model.",
    )
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judgments", type=Path, default=None)
    parser.add_argument("--judge-max-tokens", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_root / run_id
    responses_dir = run_dir / "responses"
    pairing_dir = run_dir / "pairing"
    scoring_dir = run_dir / "scoring"

    models = load_models_config(args.models_config)
    prompts = load_prompts(args.prompts)
    if args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]

    print(f"Run dir: {run_dir}")
    run_generation(
        models=models,
        prompts=prompts,
        out_dir=responses_dir,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        overwrite=args.overwrite,
        progress_every=args.progress_every,
    )

    tasks_path, key_path = run_pairing(
        responses_dir=responses_dir,
        out_dir=pairing_dir,
        raw_pairs=args.pairs,
        seed=args.seed,
        max_prompts_per_pair=args.max_prompts_per_pair,
        counterbalance_sides=args.counterbalance_sides,
        shuffle_tasks=args.shuffle_tasks,
    )

    if args.judge_mode == "manual":
        if args.judgments is None:
            print("Manual judging mode selected.")
            print(f"Use tasks file: {tasks_path}")
            print(f"Check pairing summary: {pairing_dir / 'pairing_summary.json'}")
            print("Create judgments JSONL with fields: item_id, winner (left|right|tie|invalid)")
            print("Then run evaluation/score_judgments.py with the key + your judgments file.")
            return
        judgments_path = args.judgments
    else:
        if not args.judge_model:
            raise ValueError("--judge-model is required when --judge-mode model")
        judgments_path = pairing_dir / "judgments_model.jsonl"
        run_model_judging(
            tasks_path=tasks_path,
            out_path=judgments_path,
            judge_model=args.judge_model,
            seed=args.seed,
            max_tokens=args.judge_max_tokens,
        )

    pair_json, rollup_json, report_md = run_scoring(
        key_path=key_path,
        judgments_path=judgments_path,
        out_dir=scoring_dir,
        seed=args.seed,
    )

    print(f"Pair metrics JSON: {pair_json}")
    print(f"Model rollup JSON: {rollup_json}")
    print(f"Report markdown: {report_md}")


if __name__ == "__main__":
    main()
