"""
TinyLlama LoRA Rank Ablation — Gradio Space entry point.

Live inference:
  • Apple Silicon: mlx-lm (fast path) when `mlx` / `mlx-lm` are installed.
  • Linux / HF Spaces / other: Transformers + PEFT via `torch_inference.py`
    (MLX-format adapter weights are converted to PEFT on first use under /tmp).

Benchmark plots and metrics are always available.
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

ROOT          = Path(__file__).resolve().parent
RESULTS_FILE  = ROOT / "results" / "latency.json"
PARETO_RANK   = 8
VALID_RANKS   = (4, 8, 16, 32, 64)

# ── detect runtime environment ────────────────────────────────────────────────
try:
    import mlx.core  # type: ignore[import]
    from inference import generate_response
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

try:
    from torch_inference import _torch_deps_available

    TORCH_DEPS_OK = _torch_deps_available()
except ImportError:
    TORCH_DEPS_OK = False


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_metrics() -> dict:
    if RESULTS_FILE.exists():
        with RESULTS_FILE.open() as f:
            return json.load(f)
    return {}


def _metrics_md(metrics: dict) -> str:
    per_rank = metrics.get("per_rank", {})
    if not per_rank:
        return "_Metrics not available._"
    header = "| Rank | α | Tokens/sec | ms/tok | Mem (MB) | Test Loss | MCQ % | Pareto |\n"
    header += "|---:|---:|---:|---:|---:|---:|---:|:---:|\n"
    rows = ""
    for key in sorted(per_rank.keys(), key=lambda k: int(k[1:])):
        m  = per_rank[key]
        r  = m["rank"]
        alpha = r * 2
        star = "★" if r == metrics.get("pareto_optimal_rank") else ""
        rows += (
            f"| **{r}** | {alpha} | {m['tokens_per_sec']} | {m['ms_per_token']} | "
            f"{m['peak_mem_mb']:.0f} | {m['test_loss']} | {m['mcq_accuracy']} | {star} |\n"
        )
    return header + rows


# ── inference wrapper ─────────────────────────────────────────────────────────

def _run_inference(prompt: str, rank: int, max_tokens: int, temperature: float):
    if not prompt.strip():
        return "Please enter a prompt.", ""

    if MLX_AVAILABLE:
        try:
            response, tps, n_tok = generate_response(
                prompt, rank=rank, max_tokens=max_tokens, temperature=temperature
            )
            stats = f"**{n_tok} tokens** @ **{tps} tok/s** (mlx-lm, LoRA rank **{rank}**)"
            return response, stats
        except Exception as exc:  # noqa: BLE001
            return f"**MLX error:** {exc}", ""

    if TORCH_DEPS_OK:
        try:
            from torch_inference import generate_response_torch

            response, tps, n_tok = generate_response_torch(
                prompt,
                rank=rank,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            stats = (
                f"**{n_tok} tokens** @ **{tps} tok/s** "
                f"(Transformers + PEFT, LoRA rank **{rank}**)"
            )
            return response, stats
        except Exception as exc:  # noqa: BLE001
            return f"**Inference error:** {exc}", ""

    snippet = prompt[:60] + ("..." if len(prompt) > 60 else "")
    return (
        "⚠️ **No inference backend available in this environment.**\n\n"
        "On **Hugging Face Spaces**, install `torch`, `transformers`, `peft`, and `safetensors` "
        "(see `requirements.txt`).\n\n"
        "On **Apple Silicon** you can instead use MLX:\n"
        "```bash\npip install mlx mlx-lm\n"
        f"python inference.py --rank {rank} --prompt {repr(snippet)}\n```"
    ), ""


# ── build app ─────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    metrics = _load_metrics()

    if MLX_AVAILABLE:
        platform_note = "✅ **MLX available** — using mlx-lm for fast on-device inference."
    elif TORCH_DEPS_OK:
        platform_note = (
            "✅ **Transformers + PEFT** — LoRA adapters load on the Space (first generation "
            "per rank may take a minute while the base model downloads / merges)."
        )
    else:
        platform_note = (
            "⚠️ **No inference stack installed** — add PyTorch + PEFT dependencies "
            "(see `requirements.txt`) or run locally with MLX on Apple Silicon."
        )

    with gr.Blocks(
        title="TinyLlama LoRA Rank Ablation",
        theme=gr.themes.Soft(),
        css=(
            ".plot-img img { border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.15); } "
            ".response-md { min-height: 10rem; }"
        ),
    ) as demo:
        gr.Markdown(
            f"""
# 🦙 TinyLlama LoRA Rank Ablation

**Base model:** [`TinyLlama/TinyLlama-1.1B-Chat-v1.0`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
fine-tuned with LoRA on [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) instructions
(2 000 steps, α = 2r, ranks **r ∈ {{4, 8, 16, 32, 64}}**).

{platform_note}
"""
        )

        with gr.Tabs():
            # ── Tab 1: Live inference ──────────────────────────────────────
            with gr.Tab("💬 Inference"):
                with gr.Row():
                    with gr.Column(scale=3):
                        prompt_box = gr.Textbox(
                            label="Instruction",
                            placeholder="e.g. Explain the difference between a process and a thread.",
                            lines=3,
                        )
                        with gr.Row():
                            rank_dd     = gr.Dropdown(
                                choices=list(VALID_RANKS),
                                value=PARETO_RANK,
                                label=f"LoRA Rank  (★ = Pareto-optimal, default {PARETO_RANK})",
                            )
                            max_tok_sl  = gr.Slider(32, 512, value=200, step=32, label="Max tokens")
                            temp_sl     = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
                        run_btn   = gr.Button("Generate ▶", variant="primary")
                        # Markdown so fallback instructions (bold, ```bash```) render; plain model text is fine too
                        output_md = gr.Markdown(
                            label="Response",
                            value="",
                            elem_classes=["response-md"],
                        )
                        stats_md  = gr.Markdown()

                    with gr.Column(scale=2):
                        gr.Markdown("### Benchmark results")
                        gr.Markdown(_metrics_md(metrics))

                run_btn.click(
                    fn=_run_inference,
                    inputs=[prompt_box, rank_dd, max_tok_sl, temp_sl],
                    outputs=[output_md, stats_md],
                )
                gr.Examples(
                    examples=[
                        ["Explain the difference between a process and a thread.", 8, 200, 0.7],
                        ["Write a Python function to reverse a linked list.", 8, 300, 0.7],
                        ["What are the main causes of the French Revolution?", 16, 200, 0.5],
                        ["Give me a recipe for chocolate chip cookies.", 4, 256, 0.8],
                        ["Summarize the water cycle in one paragraph.", 32, 150, 0.6],
                    ],
                    inputs=[prompt_box, rank_dd, max_tok_sl, temp_sl],
                )

            # ── Tab 2: Plots ───────────────────────────────────────────────
            with gr.Tab("📊 Plots"):
                with gr.Row():
                    gr.Image(
                        str(ROOT / "plots" / "pareto_curve.png"),
                        label="Pareto Curve — Accuracy vs. Latency",
                        elem_classes=["plot-img"],
                    )
                    gr.Image(
                        str(ROOT / "plots" / "memory_vs_rank.png"),
                        label="Memory & Throughput vs. Rank",
                        elem_classes=["plot-img"],
                    )
                gr.Markdown(
                    """
**Pareto-optimal rank: 8** — best accuracy/latency balance on Apple Silicon.

| Finding | Detail |
|---|---|
| Test loss | Flat across ranks (1.262–1.270); rank has minimal effect after 2 000 steps |
| Memory | Grows only +94 MB from r4→r64 (3%) |
| Throughput | Peaks at r8 (99.1 tok/s); higher ranks add FLOPs without improving loss |
| MCQ accuracy | Non-monotone; instruction-following ≠ MCQ reasoning |
"""
                )

            # ── Tab 3: CLI / Local usage ───────────────────────────────────
            with gr.Tab("🖥️ Run Locally"):
                gr.Markdown(
                    f"""
## Local / server setup

**HF Spaces / Linux / CUDA** — dependencies in `requirements.txt` (`torch`, `transformers`, `peft`, …) enable **Generate** in the browser.

**Apple Silicon (faster)** — optional MLX stack:

```bash
git clone https://huggingface.co/spaces/ysingh-aiml/tinyllama-lora-rank-ablation
cd tinyllama-lora-rank-ablation
pip install mlx mlx-lm gradio
```

### CLI
```bash
# Pareto-optimal rank (r=8)
python inference.py --prompt "Explain recursion."

# Choose rank
python inference.py --rank 16 --prompt "Write a sorting algorithm in Python."

# Base model (no adapter)
python inference.py --base-only --prompt "What is gradient descent?"

# Full options
python inference.py --rank 4 --prompt "..." --max-tokens 300 --temperature 0.5
```

### Launch this Gradio Space locally
```bash
python app.py
# → http://localhost:7860
```

### Python API
```python
from inference import generate_response, stream_response

text, tps, n_tokens = generate_response(
    "Explain recursion to a 10-year-old.",
    rank={PARETO_RANK},
    max_tokens=200,
    temperature=0.7,
)
print(text)

for chunk in stream_response("Describe photosynthesis.", rank=8):
    print(chunk, end="", flush=True)
```

### Adapter files
| Path | Rank | Size |
|---|---|---|
| `adapters/r4/adapters.safetensors` | 4 | 1.6 MB |
| `adapters/r8/adapters.safetensors` | 8 | 3.1 MB |
| `adapters/r16/adapters.safetensors` | 16 | 6.3 MB |
| `adapters/r32/adapters.safetensors` | 32 | 13 MB |
| `adapters/r64/adapters.safetensors` | 64 | 25 MB |
"""
                )

    return demo


if __name__ == "__main__":
    build_app().launch()
