#!/usr/bin/env python3
"""
TinyLlama LoRA Rank Ablation — Inference & Gradio Space
========================================================

Supports two modes:
  CLI   — python inference.py --rank 8 --prompt "Explain recursion."
  Space — python inference.py          (launches Gradio UI)

Requires Apple Silicon + mlx-lm (see requirements.txt).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterator

REPO_ROOT    = Path(__file__).resolve().parent
ADAPTER_ROOT = REPO_ROOT / "adapters"
RESULTS_FILE = REPO_ROOT / "results" / "latency.json"
BASE_MODEL   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
VALID_RANKS  = (4, 8, 16, 32, 64)
PARETO_RANK  = 8


# ── helpers ──────────────────────────────────────────────────────────────────

def _adapter_path(rank: int) -> str:
    path = ADAPTER_ROOT / f"r{rank}"
    if not path.exists():
        raise FileNotFoundError(
            f"Adapter for rank {rank} not found at {path}.\n"
            "Clone the full repo or run the training pipeline first."
        )
    return str(path)


def _load_metrics() -> dict:
    if RESULTS_FILE.exists():
        with RESULTS_FILE.open() as f:
            return json.load(f)
    return {}


def _build_prompt(tokenizer, text: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"<|user|>\n{text}\n<|assistant|>\n"


# ── core inference ────────────────────────────────────────────────────────────

def generate_response(
    prompt: str,
    rank: int = PARETO_RANK,
    max_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
    use_base_only: bool = False,
) -> tuple[str, float, int]:
    """
    Generate a response with the requested LoRA rank adapter.

    Returns
    -------
    (response_text, tokens_per_second, token_count)
    """
    try:
        from mlx_lm import generate, load  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "mlx-lm is required for inference. "
            "Install with: pip install mlx-lm"
        ) from exc

    adapter = None if use_base_only else _adapter_path(rank)
    model, tokenizer = load(BASE_MODEL, adapter_path=adapter)
    prompt_text = _build_prompt(tokenizer, prompt)

    t0 = time.perf_counter()
    response = generate(
        model,
        tokenizer,
        prompt_text,
        verbose=False,
        max_tokens=max_tokens,
        temp=temperature,
        top_p=top_p,
    )
    elapsed = time.perf_counter() - t0

    token_count = len(tokenizer.encode(response))
    tps = token_count / elapsed if elapsed > 0 else 0.0
    return response.strip(), round(tps, 2), token_count


def stream_response(
    prompt: str,
    rank: int = PARETO_RANK,
    max_tokens: int = 200,
    temperature: float = 0.7,
) -> Iterator[str]:
    """
    Yield response tokens one chunk at a time (for Gradio streaming).
    Falls back to non-streaming generate if mlx-lm stream_generate unavailable.
    """
    try:
        from mlx_lm import load, stream_generate  # type: ignore[import]
    except ImportError:
        text, _, _ = generate_response(prompt, rank, max_tokens, temperature)
        yield text
        return

    adapter = _adapter_path(rank)
    model, tokenizer = load(BASE_MODEL, adapter_path=adapter)
    prompt_text = _build_prompt(tokenizer, prompt)
    for chunk in stream_generate(
        model,
        tokenizer,
        prompt_text,
        max_tokens=max_tokens,
        temp=temperature,
    ):
        yield chunk


# ── CLI ───────────────────────────────────────────────────────────────────────

def cli() -> None:
    parser = argparse.ArgumentParser(
        description="TinyLlama LoRA rank-ablation inference"
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=PARETO_RANK,
        choices=VALID_RANKS,
        help=f"LoRA rank to use (default: {PARETO_RANK} — Pareto-optimal)",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Instruction prompt")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Run base TinyLlama without any LoRA adapter",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Model : {BASE_MODEL}")
    print(f"  Rank  : {'base (no adapter)' if args.base_only else args.rank}")
    print(f"  Prompt: {args.prompt[:80]}")
    print(f"{'='*60}\n")

    response, tps, n_tok = generate_response(
        args.prompt,
        rank=args.rank,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        use_base_only=args.base_only,
    )
    print(response)
    print(f"\n[{n_tok} tokens @ {tps} tok/s]")


# ── Gradio Space ──────────────────────────────────────────────────────────────

def build_gradio_app():
    try:
        import gradio as gr  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("pip install gradio") from exc

    metrics = _load_metrics()

    def _metrics_table() -> str:
        per_rank = metrics.get("per_rank", {})
        if not per_rank:
            return "Metrics not available."
        header = "| Rank | Tokens/sec | ms/tok | Mem (MB) | Test Loss | MCQ % | Pareto |\n"
        header += "|---:|---:|---:|---:|---:|---:|:---:|\n"
        rows = ""
        for key in sorted(per_rank.keys(), key=lambda k: int(k[1:])):
            m = per_rank[key]
            star = "★" if m["rank"] == metrics.get("pareto_optimal_rank") else ""
            rows += (
                f"| {m['rank']} | {m['tokens_per_sec']} | {m['ms_per_token']} | "
                f"{m['peak_mem_mb']:.0f} | {m['test_loss']} | {m['mcq_accuracy']} | {star} |\n"
            )
        return header + rows

    def _infer(prompt, rank, max_tokens, temperature):
        if not prompt.strip():
            return "Please enter a prompt.", ""
        try:
            response, tps, n_tok = generate_response(
                prompt, rank=int(rank), max_tokens=int(max_tokens), temperature=float(temperature)
            )
            stats = f"**{n_tok} tokens** generated at **{tps} tok/s** (rank {rank})"
            return response, stats
        except Exception as exc:  # noqa: BLE001
            return f"Error: {exc}", ""

    with gr.Blocks(title="TinyLlama LoRA Rank Ablation", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
# TinyLlama LoRA Rank Ablation
**Base model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0` fine-tuned with LoRA on [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) instructions (2,000 steps, α = 2r).

Select a LoRA rank to compare quality, speed, and memory usage. **Rank 8** is the Pareto-optimal choice (best accuracy/latency trade-off on Apple Silicon).

> ⚠️ Requires Apple Silicon (M1/M2/M3/M4). Uses [mlx-lm](https://github.com/ml-explore/mlx-lm) for inference.
"""
        )

        with gr.Row():
            with gr.Column(scale=3):
                prompt_box = gr.Textbox(
                    label="Instruction",
                    placeholder="e.g. Explain the difference between a process and a thread.",
                    lines=3,
                )
                with gr.Row():
                    rank_dropdown = gr.Dropdown(
                        choices=[4, 8, 16, 32, 64],
                        value=PARETO_RANK,
                        label="LoRA Rank (★ = Pareto-optimal)",
                    )
                    max_tok_slider = gr.Slider(32, 512, value=200, step=32, label="Max tokens")
                    temp_slider    = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
                run_btn  = gr.Button("Generate", variant="primary")
                output   = gr.Textbox(label="Response", lines=8, interactive=False)
                stats_md = gr.Markdown()

            with gr.Column(scale=2):
                gr.Markdown("### Rank ablation results")
                gr.Markdown(_metrics_table())
                gr.Image(str(REPO_ROOT / "plots" / "pareto_curve.png"),
                         label="Pareto Curve (accuracy vs. latency)")
                gr.Image(str(REPO_ROOT / "plots" / "memory_vs_rank.png"),
                         label="Memory & Throughput vs. Rank")

        run_btn.click(
            fn=_infer,
            inputs=[prompt_box, rank_dropdown, max_tok_slider, temp_slider],
            outputs=[output, stats_md],
        )
        gr.Examples(
            examples=[
                ["Explain the difference between a process and a thread.", 8, 200, 0.7],
                ["Write a Python function to reverse a linked list.", 8, 300, 0.7],
                ["What are the main causes of the French Revolution?", 16, 200, 0.5],
                ["Give me a recipe for chocolate chip cookies.", 4, 256, 0.8],
                ["Summarize the water cycle in one paragraph.", 32, 150, 0.6],
            ],
            inputs=[prompt_box, rank_dropdown, max_tok_slider, temp_slider],
        )

    return demo


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # If --prompt is in sys.argv → CLI mode; otherwise → Gradio Space
    if "--prompt" in sys.argv:
        cli()
    else:
        demo = build_gradio_app()
        demo.launch()
