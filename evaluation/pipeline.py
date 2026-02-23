#!/usr/bin/env python3
from __future__ import annotations

import csv
import itertools
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model: str
    adapter_path: Optional[str] = None
    system_prompt: Optional[str] = None


@dataclass(frozen=True)
class PairSpec:
    model_1: str
    model_2: str


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_num}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_num}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def parse_pairs(raw_pairs: Optional[str], available_models: Sequence[str]) -> List[PairSpec]:
    if not raw_pairs:
        return [PairSpec(a, b) for a, b in itertools.combinations(available_models, 2)]

    available = set(available_models)
    parsed: List[PairSpec] = []
    seen: set[Tuple[str, str]] = set()

    for chunk in [part.strip() for part in raw_pairs.split(",") if part.strip()]:
        if ":" not in chunk:
            raise ValueError(f"Invalid pair '{chunk}'. Expected format modelA:modelB")
        left, right = [part.strip() for part in chunk.split(":", 1)]
        if left not in available:
            raise ValueError(f"Unknown model in pair '{chunk}': {left}")
        if right not in available:
            raise ValueError(f"Unknown model in pair '{chunk}': {right}")
        if left == right:
            raise ValueError(f"Pair uses same model twice: '{chunk}'")
        key = (left, right)
        if key in seen:
            continue
        seen.add(key)
        parsed.append(PairSpec(left, right))

    if not parsed:
        raise ValueError("No valid pairs provided")
    return parsed


def load_models_config(path: Path) -> List[ModelSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Models config must be a JSON array")

    models: List[ModelSpec] = []
    seen = set()
    for idx, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"models[{idx}] must be an object")
        name = str(item.get("name", "")).strip()
        model = str(item.get("model", "")).strip()
        adapter_path = item.get("adapter_path")
        system_prompt = item.get("system_prompt")
        if not name:
            raise ValueError(f"models[{idx}] missing non-empty 'name'")
        if not model:
            raise ValueError(f"models[{idx}] missing non-empty 'model'")
        if name in seen:
            raise ValueError(f"Duplicate model name in config: {name}")
        seen.add(name)
        models.append(
            ModelSpec(
                name=name,
                model=_resolve_local_path(model) or model,
                adapter_path=_resolve_local_path(str(adapter_path)) if adapter_path else None,
                system_prompt=str(system_prompt) if system_prompt else None,
            )
        )
    return models


def _resolve_local_path(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    # Resolve explicit filesystem paths relative to repo root so commands
    # work from either project root or subdirectories (e.g. evaluation/).
    if value.startswith(".") or value.startswith("/"):
        return str((ROOT / value).resolve()) if value.startswith(".") else str(Path(value).resolve())
    return value


def load_prompts(path: Path) -> List[Dict]:
    prompts = read_jsonl(path)
    seen = set()
    normalized: List[Dict] = []

    for idx, row in enumerate(prompts, start=1):
        prompt_text = str(row.get("prompt", "")).strip()
        prompt_id = str(row.get("prompt_id") or f"p{idx:04d}")
        if not prompt_text:
            raise ValueError(f"Prompt row {idx} missing non-empty 'prompt'")
        if prompt_id in seen:
            raise ValueError(f"Duplicate prompt_id: {prompt_id}")
        seen.add(prompt_id)
        normalized.append(
            {
                "prompt_id": prompt_id,
                "prompt": prompt_text,
                "reference": row.get("reference"),
                "category": row.get("category"),
            }
        )
    return normalized


def _format_prompt_for_model(tokenizer, prompt: str, system_prompt: Optional[str]) -> str:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            # Some tokenizers expose apply_chat_template but do not define
            # tokenizer.chat_template (e.g. certain Phi conversions).
            pass
    if system_prompt:
        return f"System: {system_prompt}\nUser: {prompt}\nAssistant:"
    return f"User: {prompt}\nAssistant:"


def run_generation(
    models: Sequence[ModelSpec],
    prompts: Sequence[Dict],
    out_dir: Path,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
    overwrite: bool,
    progress_every: int = 25,
) -> Dict[str, Path]:
    try:
        import mlx.core as mx
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Generation requires mlx + mlx_lm. Install dependencies in your venv first."
        ) from exc

    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")

    out_dir.mkdir(parents=True, exist_ok=True)
    mapping: Dict[str, Path] = {}

    for model_spec in models:
        output_path = out_dir / f"{model_spec.name}.jsonl"
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite.")

        kwargs = {}
        if model_spec.adapter_path:
            kwargs["adapter_path"] = model_spec.adapter_path

        print(f"Loading model: {model_spec.name} ({model_spec.model})", flush=True)
        model, tokenizer = load(model_spec.model, **kwargs)

        rows: List[Dict] = []
        total_prompts = len(prompts)
        for index, prompt_row in enumerate(prompts):
            mx.random.seed(seed)
            formatted = _format_prompt_for_model(tokenizer, prompt_row["prompt"], model_spec.system_prompt)
            sampler = make_sampler(temp=temperature, top_p=top_p)
            response = generate(
                model,
                tokenizer,
                prompt=formatted,
                sampler=sampler,
                max_tokens=max_tokens,
                verbose=False,
            )
            rows.append(
                {
                    "prompt_id": prompt_row["prompt_id"],
                    "prompt": prompt_row["prompt"],
                    "model": model_spec.name,
                    "response": response,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "seed": seed,
                }
            )
            if progress_every > 0:
                completed = index + 1
                if completed % progress_every == 0 or completed == total_prompts:
                    print(
                        f"[{model_spec.name}] generated {completed}/{total_prompts} prompts",
                        flush=True,
                    )

        write_jsonl(output_path, rows)
        mapping[model_spec.name] = output_path

        del model
        del tokenizer
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal"):
            mx.metal.clear_cache()

    return mapping


def load_responses_from_dir(responses_dir: Path) -> Dict[str, Dict[str, Dict]]:
    if not responses_dir.exists():
        raise FileNotFoundError(f"Responses dir not found: {responses_dir}")

    by_model: Dict[str, Dict[str, Dict]] = {}
    for path in sorted(responses_dir.glob("*.jsonl")):
        rows = read_jsonl(path)
        if not rows:
            continue
        first_model = str(rows[0].get("model", "")).strip()
        if not first_model:
            first_model = path.stem
        model_rows: Dict[str, Dict] = {}

        for row in rows:
            prompt_id = str(row.get("prompt_id", "")).strip()
            if not prompt_id:
                raise ValueError(f"Missing prompt_id in {path}")
            model_rows[prompt_id] = row

        by_model[first_model] = model_rows
    if not by_model:
        raise ValueError(f"No response files found in {responses_dir}")
    return by_model


def _pair_slug(model_1: str, model_2: str) -> str:
    return f"{model_1}__vs__{model_2}"


def build_judging_tasks(
    responses_by_model: Dict[str, Dict[str, Dict]],
    pairs: Sequence[PairSpec],
    seed: int,
    max_prompts_per_pair: int,
) -> Tuple[List[Dict], List[Dict]]:
    rng = random.Random(seed)

    tasks: List[Dict] = []
    key_rows: List[Dict] = []

    for pair in pairs:
        if pair.model_1 not in responses_by_model:
            raise ValueError(f"Missing responses for model: {pair.model_1}")
        if pair.model_2 not in responses_by_model:
            raise ValueError(f"Missing responses for model: {pair.model_2}")

        rows_1 = responses_by_model[pair.model_1]
        rows_2 = responses_by_model[pair.model_2]
        prompt_ids = sorted(set(rows_1.keys()) & set(rows_2.keys()))
        if not prompt_ids:
            raise ValueError(f"No overlapping prompt_ids for pair {pair.model_1}:{pair.model_2}")

        if max_prompts_per_pair > 0 and len(prompt_ids) > max_prompts_per_pair:
            prompt_ids = sorted(rng.sample(prompt_ids, max_prompts_per_pair))

        pair_slug = _pair_slug(pair.model_1, pair.model_2)

        for prompt_id in prompt_ids:
            row_1 = rows_1[prompt_id]
            row_2 = rows_2[prompt_id]
            left_is_model_1 = bool(rng.getrandbits(1))

            left_model = pair.model_1 if left_is_model_1 else pair.model_2
            right_model = pair.model_2 if left_is_model_1 else pair.model_1
            left_text = row_1["response"] if left_is_model_1 else row_2["response"]
            right_text = row_2["response"] if left_is_model_1 else row_1["response"]

            item_id = f"{pair_slug}__{prompt_id}"
            tasks.append(
                {
                    "item_id": item_id,
                    "pair": pair_slug,
                    "prompt_id": prompt_id,
                    "prompt": row_1.get("prompt", row_2.get("prompt", "")),
                    "left_response": left_text,
                    "right_response": right_text,
                }
            )
            key_rows.append(
                {
                    "item_id": item_id,
                    "pair": pair_slug,
                    "prompt_id": prompt_id,
                    "model_1": pair.model_1,
                    "model_2": pair.model_2,
                    "left_model": left_model,
                    "right_model": right_model,
                }
            )

    return tasks, key_rows


def run_pairing(
    responses_dir: Path,
    out_dir: Path,
    raw_pairs: Optional[str],
    seed: int,
    max_prompts_per_pair: int,
) -> Tuple[Path, Path]:
    responses = load_responses_from_dir(responses_dir)
    models = sorted(responses.keys())
    pairs = parse_pairs(raw_pairs, models)

    tasks, key_rows = build_judging_tasks(
        responses_by_model=responses,
        pairs=pairs,
        seed=seed,
        max_prompts_per_pair=max_prompts_per_pair,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = out_dir / "judging_tasks.jsonl"
    key_path = out_dir / "judging_key.jsonl"
    write_jsonl(tasks_path, tasks)
    write_jsonl(key_path, key_rows)
    return tasks_path, key_path


def _parse_judge_winner(text: str) -> str:
    normalized = text.strip().lower()
    aliases = {
        "left": "left",
        "a": "left",
        "right": "right",
        "b": "right",
        "tie": "tie",
        "draw": "tie",
        "equal": "tie",
        "invalid": "invalid",
        "skip": "invalid",
    }
    if normalized not in aliases:
        raise ValueError(f"Invalid winner label: {text}")
    return aliases[normalized]


def run_model_judging(
    tasks_path: Path,
    out_path: Path,
    judge_model: str,
    seed: int,
    max_tokens: int,
) -> Path:
    try:
        import mlx.core as mx
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Model judging requires mlx + mlx_lm. Install dependencies in your venv first."
        ) from exc

    # Keep tokenizer loading behavior consistent with run_generation.
    import transformers
    if not hasattr(transformers, "TokenizersBackend"):
        from transformers import PreTrainedTokenizerFast

        class TokenizersBackend(PreTrainedTokenizerFast):
            pass

        transformers.TokenizersBackend = TokenizersBackend

    tasks = read_jsonl(tasks_path)
    if not tasks:
        raise ValueError(f"No judging tasks found in {tasks_path}")

    judge_prompt_template = (
        "You are an impartial evaluator. Compare two responses for the prompt. "
        "Judge correctness, instruction-following, helpfulness, and safety. "
        "Return exactly one token: LEFT, RIGHT, or TIE.\n\n"
        "Prompt:\n{prompt}\n\n"
        "LEFT RESPONSE:\n{left}\n\n"
        "RIGHT RESPONSE:\n{right}\n\n"
        "Verdict (LEFT/RIGHT/TIE):"
    )

    model, tokenizer = load(judge_model)
    rows: List[Dict] = []

    for idx, task in enumerate(tasks):
        mx.random.seed(seed + idx)
        prompt = judge_prompt_template.format(
            prompt=task["prompt"],
            left=task["left_response"],
            right=task["right_response"],
        )
        sampler = make_sampler(temp=0.0, top_p=1.0)
        raw = generate(
            model,
            tokenizer,
            prompt=prompt,
            sampler=sampler,
            max_tokens=max_tokens,
            verbose=False,
        )
        text = raw.strip().split()[0].lower() if raw.strip() else "invalid"
        if text.startswith("left"):
            winner = "left"
        elif text.startswith("right"):
            winner = "right"
        elif text.startswith("tie"):
            winner = "tie"
        else:
            winner = "invalid"

        rows.append(
            {
                "item_id": task["item_id"],
                "winner": winner,
                "raw_judge_output": raw,
                "judge_model": judge_model,
            }
        )

    write_jsonl(out_path, rows)

    del model
    del tokenizer
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    elif hasattr(mx, "metal"):
        mx.metal.clear_cache()

    return out_path


def _bootstrap_ci(values: Sequence[float], rng: random.Random, n: int = 1000) -> Tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    samples: List[float] = []
    length = len(values)
    for _ in range(n):
        sample = [values[rng.randrange(length)] for _ in range(length)]
        samples.append(statistics.fmean(sample))
    samples.sort()
    low_index = max(0, math.floor(0.025 * n) - 1)
    high_index = min(n - 1, math.ceil(0.975 * n) - 1)
    return samples[low_index], samples[high_index]


def _score_pair(records: List[Dict], seed: int) -> Dict:
    if not records:
        raise ValueError("Cannot score empty pair")

    model_1 = records[0]["model_1"]
    model_2 = records[0]["model_2"]

    wins_1 = 0
    wins_2 = 0
    ties = 0
    invalid = 0

    # Per-item score for model_1: 1.0 win, 0.5 tie, 0.0 loss; invalid excluded.
    bootstrap_values: List[float] = []

    for row in records:
        winner = row["winner"]
        if winner == "invalid":
            invalid += 1
            continue

        if winner == "tie":
            ties += 1
            bootstrap_values.append(0.5)
            continue

        winning_model = row["left_model"] if winner == "left" else row["right_model"]
        if winning_model == model_1:
            wins_1 += 1
            bootstrap_values.append(1.0)
        elif winning_model == model_2:
            wins_2 += 1
            bootstrap_values.append(0.0)
        else:
            invalid += 1

    total_scored = wins_1 + wins_2 + ties
    if total_scored == 0:
        win_rate = float("nan")
        tie_rate = float("nan")
        effective = float("nan")
        ci_low = float("nan")
        ci_high = float("nan")
    else:
        win_rate = wins_1 / total_scored
        tie_rate = ties / total_scored
        decisive = wins_1 + wins_2
        effective = wins_1 / decisive if decisive else float("nan")
        ci_low, ci_high = _bootstrap_ci(bootstrap_values, rng=random.Random(seed), n=1000)

    return {
        "pair": records[0]["pair"],
        "model_1": model_1,
        "model_2": model_2,
        "wins_model_1": wins_1,
        "wins_model_2": wins_2,
        "ties": ties,
        "invalid": invalid,
        "total_scored": total_scored,
        "win_rate_model_1": win_rate,
        "effective_win_rate_model_1": effective,
        "tie_rate": tie_rate,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def run_scoring(
    key_path: Path,
    judgments_path: Path,
    out_dir: Path,
    seed: int,
) -> Tuple[Path, Path, Path]:
    key_rows = read_jsonl(key_path)
    judgments = read_jsonl(judgments_path)

    key_by_item = {row["item_id"]: row for row in key_rows}
    merged: List[Dict] = []

    for row in judgments:
        item_id = str(row.get("item_id", "")).strip()
        if not item_id or item_id not in key_by_item:
            continue
        key = key_by_item[item_id]
        try:
            winner = _parse_judge_winner(str(row.get("winner", "invalid")))
        except ValueError:
            winner = "invalid"

        merged.append(
            {
                "item_id": item_id,
                "pair": key["pair"],
                "model_1": key["model_1"],
                "model_2": key["model_2"],
                "left_model": key["left_model"],
                "right_model": key["right_model"],
                "winner": winner,
            }
        )

    grouped: Dict[str, List[Dict]] = {}
    for row in merged:
        grouped.setdefault(row["pair"], []).append(row)

    pair_scores = [_score_pair(records, seed=seed) for _, records in sorted(grouped.items())]

    model_rollup: Dict[str, Dict[str, float]] = {}
    for score in pair_scores:
        for key, wins, losses, ties in [
            (score["model_1"], score["wins_model_1"], score["wins_model_2"], score["ties"]),
            (score["model_2"], score["wins_model_2"], score["wins_model_1"], score["ties"]),
        ]:
            current = model_rollup.setdefault(
                key,
                {
                    "model": key,
                    "wins": 0.0,
                    "losses": 0.0,
                    "ties": 0.0,
                    "matches": 0.0,
                },
            )
            current["wins"] += float(wins)
            current["losses"] += float(losses)
            current["ties"] += float(ties)
            current["matches"] += float(wins + losses + ties)

    rollup_rows = sorted(
        (
            {
                **row,
                "win_rate": (row["wins"] / row["matches"]) if row["matches"] else float("nan"),
                "effective_win_rate": (
                    row["wins"] / (row["wins"] + row["losses"]) if (row["wins"] + row["losses"]) else float("nan")
                ),
            }
            for row in model_rollup.values()
        ),
        key=lambda item: (item["effective_win_rate"] if not math.isnan(item["effective_win_rate"]) else -1.0),
        reverse=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    pair_json = out_dir / "pair_metrics.json"
    rollup_json = out_dir / "model_rollup.json"
    report_md = out_dir / "report.md"
    pair_csv = out_dir / "pair_metrics.csv"
    rollup_csv = out_dir / "model_rollup.csv"

    pair_json.write_text(json.dumps(pair_scores, indent=2), encoding="utf-8")
    rollup_json.write_text(json.dumps(rollup_rows, indent=2), encoding="utf-8")

    with pair_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "pair",
            "model_1",
            "model_2",
            "wins_model_1",
            "wins_model_2",
            "ties",
            "invalid",
            "total_scored",
            "win_rate_model_1",
            "effective_win_rate_model_1",
            "tie_rate",
            "ci95_low",
            "ci95_high",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in pair_scores:
            writer.writerow(row)

    with rollup_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "model",
            "wins",
            "losses",
            "ties",
            "matches",
            "win_rate",
            "effective_win_rate",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rollup_rows:
            writer.writerow(row)

    lines = [
        "# Evaluation Report",
        "",
        "## Pairwise Metrics",
        "",
        "| Pair | Model 1 | Model 2 | Win Rate (M1) | Effective Win Rate (M1) | Tie Rate | 95% CI |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in pair_scores:
        lines.append(
            "| {pair} | {model_1} | {model_2} | {win:.3f} | {eff:.3f} | {tie:.3f} | [{low:.3f}, {high:.3f}] |".format(
                pair=row["pair"],
                model_1=row["model_1"],
                model_2=row["model_2"],
                win=row["win_rate_model_1"],
                eff=row["effective_win_rate_model_1"],
                tie=row["tie_rate"],
                low=row["ci95_low"],
                high=row["ci95_high"],
            )
        )

    lines.extend(
        [
            "",
            "## Model Rollup",
            "",
            "| Model | Wins | Losses | Ties | Matches | Win Rate | Effective Win Rate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rollup_rows:
        lines.append(
            "| {model} | {wins:.0f} | {losses:.0f} | {ties:.0f} | {matches:.0f} | {win:.3f} | {eff:.3f} |".format(
                model=row["model"],
                wins=row["wins"],
                losses=row["losses"],
                ties=row["ties"],
                matches=row["matches"],
                win=row["win_rate"],
                eff=row["effective_win_rate"],
            )
        )

    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return pair_json, rollup_json, report_md
