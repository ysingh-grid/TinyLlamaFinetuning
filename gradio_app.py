#!/usr/bin/env python3
"""
TinyLlama Gradio App

Three-tab Gradio UI:
- Tab 1: Inference / Compare (two models side by side)
- Tab 2: Data Collection (prompt → editable completion → JSONL)
- Tab 3: Dataset Stats / EDA (length stats + histogram + top/bottom examples)

Usage:
    .venv/bin/python gradio_app.py
    # → http://127.0.0.1:7860 by default
"""
from __future__ import annotations

import json
import os
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None  # type: ignore

try:
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    MLX_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    MLX_AVAILABLE = False
    mx = None  # type: ignore


ROOT = Path(__file__).parent
MODELS_CONFIG_PATH = ROOT / "evaluation" / "models.json"
EVAL_PROMPTS_PATH = ROOT / "evaluation" / "eval_prompts.jsonl"
COLLECTED_PATH = ROOT / "data" / "collected.jsonl"
TRAIN_PATH = ROOT / "data" / "train.jsonl"
EDA_REPORT_PATH = ROOT / "data" / "eda_report.md"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model: str
    adapter_path: Optional[str] = None
    system_prompt: str = ""
    min_tokens: int = 0


def _load_model_specs() -> Dict[str, ModelSpec]:
    if not MODELS_CONFIG_PATH.exists():
        raise FileNotFoundError(f"models.json not found at {MODELS_CONFIG_PATH}")
    raw = json.loads(MODELS_CONFIG_PATH.read_text())
    specs: Dict[str, ModelSpec] = {}
    for item in raw:
        specs[item["name"]] = ModelSpec(
            name=item["name"],
            model=item["model"],
            adapter_path=item.get("adapter_path"),
            system_prompt=item.get("system_prompt", "") or "",
            min_tokens=int(item.get("min_tokens", 0) or 0),
        )
    return specs


MODEL_SPECS: Dict[str, ModelSpec] = _load_model_specs()
MODEL_IDS: List[str] = list(MODEL_SPECS.keys())

_mlx_lock = threading.Lock()
_cache_lock = threading.Lock()
_io_lock = threading.Lock()

# model_id -> (model, tokenizer, last_used_ts)
_model_cache: Dict[str, Tuple[Any, Any, float]] = {}
_MAX_CACHE = 2


def _clear_mlx_cache() -> None:
    if not MLX_AVAILABLE:
        return
    try:
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal"):
            mx.metal.clear_cache()
    except Exception:
        # Best-effort only
        pass


def _get_model(model_id: str) -> Tuple[Any, Any]:
    if not MLX_AVAILABLE:
        raise RuntimeError(
            "MLX / mlx_lm not available. "
            "Install them in your virtualenv to use this app."
        )
    spec = MODEL_SPECS[model_id]
    now = time.time()
    with _cache_lock:
        if model_id in _model_cache:
            model, tokenizer, _ = _model_cache[model_id]
            _model_cache[model_id] = (model, tokenizer, now)
            return model, tokenizer

        # Evict LRU if at capacity
        if len(_model_cache) >= _MAX_CACHE:
            lru_id = min(_model_cache.items(), key=lambda kv: kv[1][2])[0]
            _model_cache.pop(lru_id, None)
            _clear_mlx_cache()

        load_kwargs: Dict[str, Any] = {}
        if spec.adapter_path:
            load_kwargs["adapter_path"] = spec.adapter_path

        model, tokenizer = load(spec.model, **load_kwargs)
        _model_cache[model_id] = (model, tokenizer, now)
        return model, tokenizer


def _truncate_ngram(text: str, n: int = 4) -> str:
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


def _make_suppress_eos(min_tokens: int, eos_ids: List[int]):
    if min_tokens <= 0 or not eos_ids or not MLX_AVAILABLE:
        return None
    count = [0]

    def _proc(tokens, logits):
        count[0] += 1
        if count[0] < min_tokens:
            vals = logits.tolist()
            flat = vals[0] if isinstance(vals[0], list) else vals
            for eid in eos_ids:
                if 0 <= eid < len(flat):
                    flat[eid] = -1e9
            logits_arr = mx.array([flat]) if isinstance(vals[0], list) else mx.array(flat)
            return logits_arr
        return logits

    return _proc


def _make_rep_penalty(penalty: float):
    if penalty <= 1.0 or not MLX_AVAILABLE:
        return None
    seen: List[int] = []

    def _proc(tokens, logits):
        if tokens is not None and hasattr(tokens, "size") and tokens.size > 0:
            seen.extend(tokens.tolist() if hasattr(tokens, "tolist") else [int(tokens)])
        if not seen:
            return logits
        vals = logits.tolist()
        flat = vals[0] if isinstance(vals[0], list) else vals
        for tid in set(seen):
            if 0 <= tid < len(flat):
                if flat[tid] > 0:
                    flat[tid] = flat[tid] / penalty
                else:
                    flat[tid] = flat[tid] * penalty
        logits_arr = mx.array([flat]) if isinstance(vals[0], list) else mx.array(flat)
        return logits_arr

    return _proc


def _format_chat_prompt(
    tokenizer: Any, user_prompt: str, system_prompt: str | None
) -> str:
    messages: List[Dict[str, str]] = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": user_prompt})
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prefix = f"System: {system_prompt.strip()}\n\n" if system_prompt else ""
        return f"{prefix}User: {user_prompt}\nAssistant:"


def _get_eos_ids(tokenizer: Any) -> List[int]:
    try:
        raw_eos = getattr(tokenizer, "eos_token_ids", None)
        if raw_eos is None:
            eid = getattr(tokenizer, "eos_token_id", None)
            raw_eos = [eid] if eid is not None else []
        return [int(e) for e in raw_eos if e is not None]
    except Exception:
        return []


def generate_once(
    model_id: str,
    prompt: str,
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    repetition_penalty: float,
) -> Tuple[str, int, float]:
    if not prompt.strip():
        return "", 0, 0.0

    spec = MODEL_SPECS[model_id]
    model, tokenizer = _get_model(model_id)

    effective_system = system_prompt.strip() or spec.system_prompt
    formatted = _format_chat_prompt(tokenizer, prompt, effective_system)
    sampler = make_sampler(temp=temperature, top_p=top_p)

    processors: List[Any] = []
    eos_ids = _get_eos_ids(tokenizer)
    sup = _make_suppress_eos(spec.min_tokens, eos_ids)
    if sup:
        processors.append(sup)
    rep = _make_rep_penalty(repetition_penalty)
    if rep:
        processors.append(rep)

    t0 = time.time()
    token_count = 0
    parts: List[str] = []

    with _mlx_lock:
        gen_kwargs: Dict[str, Any] = dict(
            model=model,
            tokenizer=tokenizer,
            prompt=formatted,
            max_tokens=max_tokens,
            sampler=sampler,
        )
        if processors:
            gen_kwargs["logits_processors"] = processors

        try:
            for chunk in stream_generate(**gen_kwargs):
                parts.append(chunk.text)
                token_count += 1
        except Exception as exc:
            return f"[Generation error: {exc}]", 0, 0.0

    text = "".join(parts)
    text = _truncate_ngram(text, n=4)
    elapsed = time.time() - t0
    return text, token_count, elapsed


# ---------------------------------------------------------------------------
# Tab 1 — Inference / Compare
# ---------------------------------------------------------------------------


def random_eval_prompt() -> str:
    if not EVAL_PROMPTS_PATH.exists():
        return ""
    lines: List[str] = []
    with EVAL_PROMPTS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                lines.append(obj.get("prompt", ""))
            except Exception:
                continue
    if not lines:
        return ""
    return random.choice(lines)


def compare_models(
    model_a: str,
    model_b: str,
    prompt: str,
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    repetition_penalty: float,
) -> Tuple[str, str, str, str]:
    if not MLX_AVAILABLE:
        msg = (
            "MLX / mlx_lm not available. "
            "Install them in your virtualenv to run generation."
        )
        return msg, msg, "", ""

    if not prompt.strip():
        return "Enter a prompt.", "Enter a prompt.", "", ""

    text_a, toks_a, el_a = generate_once(
        model_a, prompt, system_prompt, temperature, top_p, max_tokens, repetition_penalty
    )
    text_b, toks_b, el_b = generate_once(
        model_b, prompt, system_prompt, temperature, top_p, max_tokens, repetition_penalty
    )

    stats_a = (
        f"**Model:** `{model_a}`  \n"
        f"**Tokens:** {toks_a}  \n"
        f"**Elapsed:** {el_a:.2f}s"
    )
    stats_b = (
        f"**Model:** `{model_b}`  \n"
        f"**Tokens:** {toks_b}  \n"
        f"**Elapsed:** {el_b:.2f}s"
    )
    return text_a, text_b, stats_a, stats_b


# ---------------------------------------------------------------------------
# Tab 2 — Data Collection
# ---------------------------------------------------------------------------


def _ensure_data_dir() -> None:
    COLLECTED_PATH.parent.mkdir(parents=True, exist_ok=True)


def collected_count() -> str:
    if not COLLECTED_PATH.exists():
        return "0 examples collected"
    count = 0
    with COLLECTED_PATH.open() as f:
        for line in f:
            if line.strip():
                count += 1
    return f"{count} examples collected"


def generate_for_collection(
    model_id: str,
    prompt: str,
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    repetition_penalty: float,
) -> str:
    if not MLX_AVAILABLE:
        return (
            "MLX / mlx_lm not available. "
            "Install them in your virtualenv to run generation."
        )
    if not prompt.strip():
        return "Enter a prompt before generating."

    text, _, _ = generate_once(
        model_id, prompt, system_prompt, temperature, top_p, max_tokens, repetition_penalty
    )
    return text


def save_collected_example(prompt: str, response: str) -> Tuple[str, str]:
    if not prompt.strip() or not response.strip():
        return "Prompt and response must be non-empty.", collected_count()

    _ensure_data_dir()
    record = {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    }
    with _io_lock:
        with COLLECTED_PATH.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return "Saved.", collected_count()


def export_collected_file() -> Optional[str]:
    if not COLLECTED_PATH.exists():
        return None
    return str(COLLECTED_PATH)


# ---------------------------------------------------------------------------
# Tab 3 — Dataset Stats / EDA
# ---------------------------------------------------------------------------


def _iter_dataset_outputs() -> List[Tuple[int, int, int, str, str]]:
    """Return list of (idx, input_words, output_words, user_preview, assistant_preview)."""
    if not TRAIN_PATH.exists():
        return []
    rows: List[Tuple[int, int, int, str, str]] = []
    with TRAIN_PATH.open() as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            msgs = obj.get("messages") or []
            user = ""
            assistant = ""
            for m in msgs:
                if m.get("role") == "user":
                    user = str(m.get("content", ""))
                elif m.get("role") == "assistant":
                    assistant = str(m.get("content", ""))
            in_words = len(user.split())
            out_words = len(assistant.split())
            rows.append(
                (
                    idx,
                    in_words,
                    out_words,
                    user[:120].replace("\n", " "),
                    assistant[:120].replace("\n", " "),
                )
            )
    return rows


def load_dataset_stats():
    rows = _iter_dataset_outputs()
    if not rows:
        summary_md = (
            "Dataset not found or empty at `data/train.jsonl`. "
            "Run `prepare_dataset.py` first."
        )
        return summary_md, None, [], ""

    out_lengths = np.array([r[2] for r in rows], dtype=np.int32)
    total = int(len(out_lengths))
    mean = float(out_lengths.mean())
    median = float(np.median(out_lengths))
    p90 = float(np.percentile(out_lengths, 90))
    p99 = float(np.percentile(out_lengths, 99))
    min_v = int(out_lengths.min())
    max_v = int(out_lengths.max())

    summary_md = (
        f"**Total examples:** {total}  \n"
        f"**Output length (words)** — "
        f"min {min_v}, mean {mean:.1f}, median {median:.1f}, "
        f"p90 {p90:.1f}, p99 {p99:.1f}, max {max_v}"
    )

    # Histogram plot (10 buckets)
    fig = None
    if plt is not None:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(out_lengths, bins=10, color="#60a5fa", edgecolor="black", alpha=0.85)
        ax.set_xlabel("Answer length (words)")
        ax.set_ylabel("Count")
        ax.set_title("Answer length distribution (train set)")
        fig.tight_layout()

    # Top-10 / bottom-10 examples by answer length
    rows_sorted = sorted(rows, key=lambda r: r[2])
    bottom10 = rows_sorted[:10]
    top10 = rows_sorted[-10:]

    table_rows: List[Dict[str, Any]] = []
    for rank_type, subset in (("shortest", bottom10), ("longest", top10)):
        for idx, in_w, out_w, user_prev, asst_prev in subset:
            table_rows.append(
                {
                    "which": rank_type,
                    "index": idx,
                    "input_words": in_w,
                    "output_words": out_w,
                    "user_preview": user_prev,
                    "assistant_preview": asst_prev,
                }
            )

    # EDA report markdown (if present)
    eda_md = ""
    if EDA_REPORT_PATH.exists():
        eda_md = EDA_REPORT_PATH.read_text()

    return summary_md, fig, table_rows, eda_md


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="TinyLlama Gradio App") as demo:
        gr.Markdown(
            "### TinyLlama Instruction Tuning — Gradio App\n"
            "Use the tabs below to compare models, collect new data, and inspect "
            "dataset statistics."
        )

        with gr.Tab("Inference / Compare"):
            with gr.Row():
                model_a = gr.Dropdown(
                    MODEL_IDS,
                    value="full_ft" if "full_ft" in MODEL_IDS else MODEL_IDS[0],
                    label="Model A",
                )
                model_b = gr.Dropdown(
                    MODEL_IDS,
                    value="base" if "base" in MODEL_IDS else MODEL_IDS[-1],
                    label="Model B",
                )
            prompt = gr.Textbox(
                lines=5,
                label="Prompt",
                placeholder="Enter an instruction or question to compare models...",
            )
            sys_prompt = gr.Textbox(
                lines=3,
                label="System prompt (optional)",
                value="",
                placeholder="e.g. You are a helpful teaching assistant.",
            )
            with gr.Row():
                temperature = gr.Slider(
                    0.0,
                    2.0,
                    value=0.2,
                    step=0.05,
                    label="Temperature",
                )
                top_p = gr.Slider(
                    0.0,
                    1.0,
                    value=0.9,
                    step=0.01,
                    label="Top-p",
                )
                max_tokens = gr.Slider(
                    64,
                    1024,
                    value=512,
                    step=16,
                    label="Max tokens",
                )
                repetition_penalty = gr.Slider(
                    1.0,
                    2.0,
                    value=1.5,
                    step=0.05,
                    label="Repetition penalty",
                )

            with gr.Row():
                random_btn = gr.Button("🎲 Random Prompt")
                generate_btn = gr.Button("Generate Both", variant="primary")

            with gr.Row():
                with gr.Column():
                    out_a = gr.Textbox(
                        lines=15,
                        label="Model A response",
                        show_copy_button=True,
                    )
                    stats_a = gr.Markdown("")
                with gr.Column():
                    out_b = gr.Textbox(
                        lines=15,
                        label="Model B response",
                        show_copy_button=True,
                    )
                    stats_b = gr.Markdown("")

            random_btn.click(fn=random_eval_prompt, inputs=None, outputs=prompt)
            generate_btn.click(
                fn=compare_models,
                inputs=[
                    model_a,
                    model_b,
                    prompt,
                    sys_prompt,
                    temperature,
                    top_p,
                    max_tokens,
                    repetition_penalty,
                ],
                outputs=[out_a, out_b, stats_a, stats_b],
            )

        with gr.Tab("Data Collection"):
            gr.Markdown(
                "Use this tab to generate and curate new training examples. "
                "Edited responses are saved in `data/collected.jsonl` as "
                "`[user, assistant]` message pairs."
            )
            with gr.Row():
                dc_model = gr.Dropdown(
                    MODEL_IDS,
                    value="full_ft" if "full_ft" in MODEL_IDS else MODEL_IDS[0],
                    label="Model",
                )
                dc_temp = gr.Slider(
                    0.0,
                    2.0,
                    value=0.2,
                    step=0.05,
                    label="Temperature",
                )
                dc_top_p = gr.Slider(
                    0.0,
                    1.0,
                    value=0.9,
                    step=0.01,
                    label="Top-p",
                )
                dc_max_tokens = gr.Slider(
                    64,
                    1024,
                    value=512,
                    step=16,
                    label="Max tokens",
                )
                dc_rep = gr.Slider(
                    1.0,
                    2.0,
                    value=1.5,
                    step=0.05,
                    label="Repetition penalty",
                )
            dc_sys = gr.Textbox(
                lines=2,
                label="System prompt (optional)",
                placeholder="Optional system message to prepend.",
            )
            dc_prompt = gr.Textbox(
                lines=5,
                label="Prompt",
                placeholder="Enter a prompt to generate a training example...",
            )
            dc_response = gr.Textbox(
                lines=10,
                label="Model response (editable before saving)",
            )
            with gr.Row():
                dc_generate = gr.Button("Generate", variant="primary")
                dc_save = gr.Button("Save Example")
                dc_export = gr.Button("Export as Training JSONL")
            with gr.Row():
                dc_status = gr.Markdown("")
                dc_counter = gr.Markdown(collected_count())
            dc_file = gr.File(label="Download collected.jsonl", interactive=False)

            dc_generate.click(
                fn=generate_for_collection,
                inputs=[
                    dc_model,
                    dc_prompt,
                    dc_sys,
                    dc_temp,
                    dc_top_p,
                    dc_max_tokens,
                    dc_rep,
                ],
                outputs=dc_response,
            )
            dc_save.click(
                fn=save_collected_example,
                inputs=[dc_prompt, dc_response],
                outputs=[dc_status, dc_counter],
            )
            dc_export.click(fn=export_collected_file, inputs=None, outputs=dc_file)

        with gr.Tab("Dataset Stats / EDA"):
            gr.Markdown(
                "Click **Load Stats** to compute basic dataset statistics from "
                "`data/train.jsonl`. If `data/eda_report.md` exists, it will be "
                "rendered below."
            )
            load_btn = gr.Button("Load Stats", variant="primary")
            summary_md = gr.Markdown("")
            hist_plot = gr.Plot()
            table = gr.Dataframe(
                headers=[
                    "which",
                    "index",
                    "input_words",
                    "output_words",
                    "user_preview",
                    "assistant_preview",
                ],
                datatype=[
                    "str",
                    "number",
                    "number",
                    "number",
                    "str",
                    "str",
                ],
                label="Top-10 shortest / longest answers",
            )
            eda_md = gr.Markdown("")

            load_btn.click(
                fn=load_dataset_stats,
                inputs=None,
                outputs=[summary_md, hist_plot, table, eda_md],
            )

    return demo


def main() -> None:
    demo = build_interface()
    port = int(os.environ.get("GRADIO_PORT", "7860"))
    demo.queue().launch(server_port=port)


if __name__ == "__main__":
    main()

