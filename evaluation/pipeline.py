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
    min_tokens: int = 0  # 0 = no minimum; >0 suppresses EOS until this many tokens generated


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
        min_tokens = int(item.get("min_tokens", 0))
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
                min_tokens=min_tokens,
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


def _truncate_at_repeated_ngram(text: str, n: int = 4) -> str:
    """Truncate text at the first repeated n-gram to prevent looping responses.

    Walks forward through word-level n-grams. The moment an n-gram is seen for
    the second time, returns everything up to (not including) that repeat.
    Returns the original text unchanged if no repetition is found.
    """
    words = text.split()
    if len(words) < n * 2:
        return text
    seen: set = set()
    for i in range(len(words) - n + 1):
        ng = tuple(words[i : i + n])
        if ng in seen:
            truncated = " ".join(words[:i]).rstrip(" ,.;:")
            return truncated if truncated else text
        seen.add(ng)
    return text


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

    # Some quantized local models (e.g. tinyllama-4bit-base) have
    # tokenizer_class=TokenizersBackend in their tokenizer_config.json.
    # Transformers refuses to load them unless the class is registered.
    # This shim mirrors the same patch used in sweep_finetune.py.
    try:
        import transformers
        if not hasattr(transformers, "TokenizersBackend"):
            from transformers import PreTrainedTokenizerFast

            class TokenizersBackend(PreTrainedTokenizerFast):  # type: ignore
                pass

            transformers.TokenizersBackend = TokenizersBackend  # type: ignore
    except Exception:
        pass

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

        # EOS suppression via logits_processors for min_tokens enforcement.
        # generate_step (via stream_generate) accepts logits_processors, not logit_bias.
        # A stateful closure counts tokens generated and suppresses EOS ids until
        # min_tokens threshold is reached, then becomes a no-op.
        eos_ids: List[int] = []
        if model_spec.min_tokens > 0:
            try:
                raw_eos = getattr(tokenizer, "eos_token_ids", None)
                if raw_eos is None:
                    eid = getattr(tokenizer, "eos_token_id", None)
                    raw_eos = [eid] if eid is not None else []
                eos_ids = [int(e) for e in raw_eos if e is not None]
            except Exception:
                eos_ids = []

        rows: List[Dict] = []
        total_prompts = len(prompts)
        for index, prompt_row in enumerate(prompts):
            # Seed once per model, not per prompt, to allow realistic sampling
            # variation across prompts (audit #8).
            if index == 0:
                mx.random.seed(seed)
            formatted = _format_prompt_for_model(tokenizer, prompt_row["prompt"], model_spec.system_prompt)
            sampler = make_sampler(temp=temperature, top_p=top_p)

            if model_spec.min_tokens > 0 and eos_ids:
                from mlx_lm import stream_generate  # type: ignore

                def _make_suppress_processor(min_n: int, eos_set: List[int]):
                    """Returns a logits_processor that suppresses EOS for first min_n tokens."""
                    _n = [0]
                    def _proc(tokens: mx.array, logits: mx.array) -> mx.array:
                        if _n[0] < min_n:
                            vals = logits.tolist()
                            flat = vals[0] if isinstance(vals[0], list) else vals
                            for eid in eos_set:
                                if eid < len(flat):
                                    flat[eid] = -1e9
                            logits = mx.array([flat]) if isinstance(vals[0], list) else mx.array(flat)
                        _n[0] += 1
                        return logits
                    return _proc

                def _make_repetition_penalty_processor(penalty: float = 1.5):
                    """Penalizes tokens that have already been generated.
                    For logits > 0: divide by penalty. For logits < 0: multiply by penalty.
                    This discourages repetition without completely blocking tokens."""
                    _seen: List[int] = []
                    def _proc(tokens: mx.array, logits: mx.array) -> mx.array:
                        if tokens is not None and tokens.size > 0:
                            _seen.extend(tokens.tolist() if hasattr(tokens, 'tolist') else [int(tokens)])
                        if not _seen:
                            return logits
                        vals = logits.tolist()
                        flat = vals[0] if isinstance(vals[0], list) else vals
                        seen_set = set(_seen)
                        for tid in seen_set:
                            if tid < len(flat):
                                if flat[tid] > 0:
                                    flat[tid] = flat[tid] / penalty
                                else:
                                    flat[tid] = flat[tid] * penalty
                        logits = mx.array([flat]) if isinstance(vals[0], list) else mx.array(flat)
                        return logits
                    return _proc

                processors = [
                    _make_suppress_processor(model_spec.min_tokens, eos_ids),
                    _make_repetition_penalty_processor(1.5),
                ]
                response_parts: List[str] = []
                for chunk in stream_generate(
                    model,
                    tokenizer,
                    prompt=formatted,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    logits_processors=processors,
                ):
                    response_parts.append(chunk.text)
                response = "".join(response_parts)
                # Post-process: truncate at first repeated 4-gram to eliminate
                # any looping that the token-level penalty didn't catch.
                response = _truncate_at_repeated_ngram(response, n=4)
            else:
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
                    "min_tokens": model_spec.min_tokens,
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


def _assign_left_side_by_prompt(
    prompt_ids: Sequence[str],
    rng: random.Random,
    counterbalance_sides: bool,
) -> Dict[str, bool]:
    if not prompt_ids:
        return {}

    if not counterbalance_sides:
        return {prompt_id: bool(rng.getrandbits(1)) for prompt_id in prompt_ids}

    shuffled_ids = list(prompt_ids)
    rng.shuffle(shuffled_ids)

    # Guarantee near-perfect side balance for each pair to reduce position bias.
    model_1_left_count = len(shuffled_ids) // 2
    if len(shuffled_ids) % 2 == 1:
        model_1_left_count += int(rng.getrandbits(1))

    model_1_left_ids = set(shuffled_ids[:model_1_left_count])
    return {prompt_id: (prompt_id in model_1_left_ids) for prompt_id in prompt_ids}


def _build_pairing_summary(
    key_rows: Sequence[Dict],
    seed: int,
    counterbalance_sides: bool,
    shuffle_tasks: bool,
) -> Dict:
    by_pair: Dict[str, Dict] = {}
    for row in key_rows:
        pair = str(row["pair"])
        stats = by_pair.setdefault(
            pair,
            {
                "pair": pair,
                "model_1": row["model_1"],
                "model_2": row["model_2"],
                "total_items": 0,
                "model_1_left": 0,
                "model_1_right": 0,
            },
        )
        stats["total_items"] += 1
        if row["left_model"] == row["model_1"]:
            stats["model_1_left"] += 1
        else:
            stats["model_1_right"] += 1

    pair_rows: List[Dict] = []
    total_items = 0
    max_abs_imbalance = 0
    for pair in sorted(by_pair.keys()):
        stats = by_pair[pair]
        total = int(stats["total_items"])
        imbalance = abs(int(stats["model_1_left"]) - int(stats["model_1_right"]))
        stats["abs_left_right_imbalance_model_1"] = imbalance
        stats["left_right_imbalance_ratio_model_1"] = (imbalance / total) if total else 0.0
        pair_rows.append(stats)
        total_items += total
        max_abs_imbalance = max(max_abs_imbalance, imbalance)

    return {
        "seed": seed,
        "counterbalance_sides": counterbalance_sides,
        "shuffle_tasks": shuffle_tasks,
        "pair_count": len(pair_rows),
        "total_items": total_items,
        "max_abs_left_right_imbalance_model_1": max_abs_imbalance,
        "pairs": pair_rows,
    }


def build_judging_tasks(
    responses_by_model: Dict[str, Dict[str, Dict]],
    pairs: Sequence[PairSpec],
    seed: int,
    max_prompts_per_pair: int,
    counterbalance_sides: bool = True,
    shuffle_tasks: bool = True,
) -> Tuple[List[Dict], List[Dict]]:
    rng = random.Random(seed)

    task_and_key_rows: List[Tuple[Dict, Dict]] = []

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
        model_1_left_by_prompt = _assign_left_side_by_prompt(
            prompt_ids=prompt_ids,
            rng=rng,
            counterbalance_sides=counterbalance_sides,
        )

        for prompt_id in prompt_ids:
            row_1 = rows_1[prompt_id]
            row_2 = rows_2[prompt_id]
            left_is_model_1 = model_1_left_by_prompt[prompt_id]

            left_model = pair.model_1 if left_is_model_1 else pair.model_2
            right_model = pair.model_2 if left_is_model_1 else pair.model_1
            left_text = row_1["response"] if left_is_model_1 else row_2["response"]
            right_text = row_2["response"] if left_is_model_1 else row_1["response"]

            item_id = f"{pair_slug}__{prompt_id}"
            task = {
                "item_id": item_id,
                "pair": pair_slug,
                "prompt_id": prompt_id,
                "prompt": row_1.get("prompt", row_2.get("prompt", "")),
                "left_response": left_text,
                "right_response": right_text,
            }
            key = {
                "item_id": item_id,
                "pair": pair_slug,
                "prompt_id": prompt_id,
                "model_1": pair.model_1,
                "model_2": pair.model_2,
                "left_model": left_model,
                "right_model": right_model,
            }
            task_and_key_rows.append((task, key))

    if shuffle_tasks:
        rng.shuffle(task_and_key_rows)

    tasks = [task for task, _ in task_and_key_rows]
    key_rows = [key for _, key in task_and_key_rows]
    return tasks, key_rows


def run_pairing(
    responses_dir: Path,
    out_dir: Path,
    raw_pairs: Optional[str],
    seed: int,
    max_prompts_per_pair: int,
    counterbalance_sides: bool = True,
    shuffle_tasks: bool = True,
) -> Tuple[Path, Path]:
    responses = load_responses_from_dir(responses_dir)
    models = sorted(responses.keys())
    pairs = parse_pairs(raw_pairs, models)

    tasks, key_rows = build_judging_tasks(
        responses_by_model=responses,
        pairs=pairs,
        seed=seed,
        max_prompts_per_pair=max_prompts_per_pair,
        counterbalance_sides=counterbalance_sides,
        shuffle_tasks=shuffle_tasks,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = out_dir / "judging_tasks.jsonl"
    key_path = out_dir / "judging_key.jsonl"
    summary_path = out_dir / "pairing_summary.json"
    write_jsonl(tasks_path, tasks)
    write_jsonl(key_path, key_rows)
    summary = _build_pairing_summary(
        key_rows=key_rows,
        seed=seed,
        counterbalance_sides=counterbalance_sides,
        shuffle_tasks=shuffle_tasks,
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
    max_response_chars: int = 600,
    progress_every: int = 25,
) -> Path:
    """Judge pairwise tasks with a local MLX model.

    Speed notes vs LM-Studio reasoning judge:
    - No HTTP overhead (direct Python call)
    - max_tokens=3 → only 3 output tokens instead of 500+ reasoning tokens
    - max_response_chars=600 keeps input short (fast prefill)
    - Crash-safe: each result written immediately; restart auto-resumes
    Combined these give ~30-100x faster throughput.
    """
    try:
        import mlx.core as mx
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Model judging requires mlx + mlx_lm. Install dependencies in your venv first."
        ) from exc

    import transformers
    if not hasattr(transformers, "TokenizersBackend"):
        from transformers import PreTrainedTokenizerFast

        class TokenizersBackend(PreTrainedTokenizerFast):
            pass

        transformers.TokenizersBackend = TokenizersBackend

    all_tasks = read_jsonl(tasks_path)
    if not all_tasks:
        raise ValueError(f"No judging tasks found in {tasks_path}")

    # ── Resume: skip already-written item_ids ──────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed_ids: set = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line:
                    try:
                        _row = json.loads(_line)
                        if _row.get("item_id"):
                            completed_ids.add(str(_row["item_id"]))
                    except json.JSONDecodeError:
                        pass

    tasks = [t for t in all_tasks if str(t["item_id"]) not in completed_ids]
    total_all = len(all_tasks)
    already_done = len(completed_ids)

    if already_done:
        print(f"Resuming local judge: {already_done}/{total_all} done, {len(tasks)} remaining")
    else:
        print(
            f"Local MLX judge: {len(tasks)} tasks | model={judge_model} | "
            f"max_tokens={max_tokens} | max_response_chars={max_response_chars}"
        )

    if not tasks:
        print("All tasks already judged.")
        return out_path

    # ── Judge prompt (concise — short input = fast prefill) ────────────────
    def _trunc(text: str) -> str:
        if len(text) <= max_response_chars:
            return text
        return text[:max_response_chars].rstrip() + "…"

    judge_prompt_template = (
        "Judge which response better answers the prompt.\n"
        "Criteria: correctness, instruction-following, helpfulness.\n"
        "Reply with exactly one word: LEFT, RIGHT, or TIE.\n\n"
        "Prompt: {prompt}\n\n"
        "LEFT: {left}\n\n"
        "RIGHT: {right}\n\n"
        "Verdict:"
    )

    model, tokenizer = load(judge_model)
    sampler = make_sampler(temp=0.0, top_p=1.0)

    import time as _time
    t0 = _time.time()

    with out_path.open("a", encoding="utf-8") as out_fh:
        for idx, task in enumerate(tasks, start=1):
            mx.random.seed(seed + idx)
            prompt = judge_prompt_template.format(
                prompt=str(task["prompt"]),
                left=_trunc(str(task["left_response"])),
                right=_trunc(str(task["right_response"])),
            )
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

            row = {
                "item_id": task["item_id"],
                "winner": winner,
                "raw_judge_output": raw,
                "judge_model": judge_model,
            }
            out_fh.write(json.dumps(row, ensure_ascii=True) + "\n")
            out_fh.flush()

            done_total = already_done + idx
            if progress_every > 0 and (idx % progress_every == 0 or idx == len(tasks)):
                elapsed = _time.time() - t0
                rate = idx / elapsed if elapsed > 0 else 0
                eta_s = (len(tasks) - idx) / rate if rate > 0 else 0
                pct = done_total / total_all * 100
                print(
                    f"Judged {done_total}/{total_all} ({pct:.0f}%)  "
                    f"{rate:.1f} tasks/s  ETA {eta_s/60:.0f}m",
                    flush=True,
                )

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
        net_margin = float("nan")
        ci_low = float("nan")
        ci_high = float("nan")
    else:
        win_rate = wins_1 / total_scored
        tie_rate = ties / total_scored
        decisive = wins_1 + wins_2
        effective = wins_1 / decisive if decisive else float("nan")
        net_margin = (wins_1 - wins_2) / total_scored
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
        "net_margin_model_1": net_margin,
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

    # Coverage guard: warn and assert if many judgments had no matching key.
    matched_count = len(merged)
    total_judgments = len(judgments)
    if total_judgments > 0:
        coverage = matched_count / total_judgments
        if coverage < 1.0:
            import sys
            print(
                f"WARNING: judgment coverage is {coverage:.1%} ({matched_count}/{total_judgments}). "
                f"{total_judgments - matched_count} judgments had no matching item_id in the key.",
                file=sys.stderr,
            )
        if coverage < 0.9:
            raise ValueError(
                f"Judgment coverage too low: {coverage:.1%}. "
                f"Only {matched_count}/{total_judgments} judgments matched an item_id in the key. "
                f"This likely indicates a pairing/judging mismatch."
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
                "net_margin": (
                    (row["wins"] - row["losses"]) / row["matches"] if row["matches"] else float("nan")
                ),
            }
            for row in model_rollup.values()
        ),
        key=lambda item: (item["net_margin"] if not math.isnan(item["net_margin"]) else -999.0),
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
            "net_margin_model_1",
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
            "net_margin",
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
        "| Pair | Model 1 | Model 2 | Win Rate (M1) | Effective Win Rate (M1) | Net Margin (M1) | Tie Rate | 95% CI |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in pair_scores:
        lines.append(
            "| {pair} | {model_1} | {model_2} | {win:.3f} | {eff:.3f} | {margin:+.1%} | {tie:.3f} | [{low:.3f}, {high:.3f}] |".format(
                pair=row["pair"],
                model_1=row["model_1"],
                model_2=row["model_2"],
                win=row["win_rate_model_1"],
                eff=row["effective_win_rate_model_1"],
                margin=row["net_margin_model_1"],
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
            "| Model | Wins | Losses | Ties | Matches | Win Rate | Effective Win Rate | Net Margin |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rollup_rows:
        lines.append(
            "| {model} | {wins:.0f} | {losses:.0f} | {ties:.0f} | {matches:.0f} | {win:.3f} | {eff:.3f} | {margin:+.1%} |".format(
                model=row["model"],
                wins=row["wins"],
                losses=row["losses"],
                ties=row["ties"],
                matches=row["matches"],
                win=row["win_rate"],
                eff=row["effective_win_rate"],
                margin=row["net_margin"],
            )
        )

    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return pair_json, rollup_json, report_md
