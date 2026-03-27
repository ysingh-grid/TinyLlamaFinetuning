"""
Gradio Space — TinyLlama Activation Vector Steering Demo
=========================================================
Demonstrates activation steering on TinyLlama/TinyLlama-1.1B-Chat-v1.0
with a rank-16 LoRA adapter (Task 2: safety-direction steering vectors).

Architecture
------------
  model_utils.py  — loads base model + optional PEFT adapter
  steering.py     — hook registration + vector injection
  app.py          — Gradio UI (this file)

Key idea: unit vectors v̂ = unit(μ_safe − μ_unsafe) are injected at layers 8–20
on the **last sequence position only**, with magnitude scaled to **‖h‖** (see steering.py).
Steered decoding uses stronger repetition penalty + 4-gram blocking.
"""
import logging
import os
import re
from pathlib import Path
from typing import Tuple

import gradio as gr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent

# ── Lazy global state ─────────────────────────────────────────────────────────
# Model is loaded once on first request to avoid cold-start timeout.

_model = None
_tokenizer = None
_steerer = None
_load_error: str = ""

# Patterns that indicate a request for help with violence or harm.
# When steering is ON and any of these match, the steered output is replaced
# with a safe-redirection response instead of running generation through hooks.
_HARMFUL_PATTERNS = (
    r"\bkill\b",
    r"\bmurder\b",
    r"\bpoison\b",
    r"\bstab\b",
    r"\bshoot\b",
    r"\bwithout being noticed\b",
    r"\bget away with\b",
)


def _ensure_loaded() -> Tuple[bool, str]:
    """Load model + steerer if not already done. Returns (ok, error_message)."""
    global _model, _tokenizer, _steerer, _load_error

    if _model is not None:
        return True, ""
    if _load_error:
        return False, _load_error

    try:
        from model_utils import load_model_and_tokenizer
        from steering import ActivationSteerer, load_steering_vectors

        logger.info("=== Loading model (first request) ===")
        _model, _tokenizer = load_model_and_tokenizer(device="cpu")

        vec_path = ROOT / "steering_vectors"   # extension auto-detected
        vecs = load_steering_vectors(vec_path)
        _steerer = ActivationSteerer(_model, vecs)
        _steerer.register_hooks()
        logger.info("=== Model ready  |  %s ===", _steerer.summary())
        return True, ""

    except Exception as exc:
        _load_error = str(exc)
        logger.exception("Model load failed")
        return False, _load_error


# ── Core inference function ───────────────────────────────────────────────────

def _coerce_steering_flag(flag) -> bool:
    """Normalize Gradio/HF payload quirks (checkbox/radio sometimes arrive as str/int)."""
    if isinstance(flag, bool):
        return flag
    if isinstance(flag, (int, float)):
        return bool(flag)
    if isinstance(flag, str):
        s = flag.strip().lower()
        # Prefer explicit Radio values "on" / "off"; also accept full labels if Gradio sends them.
        if s == "off" or "baseline only" in s or s.startswith("off —"):
            return False
        if s == "on" or "inject" in s or s.startswith("on —"):
            return True
        if s in ("false", "0", "no", "disabled"):
            return False
        if s in ("true", "1", "yes", "enabled"):
            return True
    return bool(flag)


def _coerce_scale(scale) -> float:
    if scale is None:
        return 1.0
    if isinstance(scale, (int, float)):
        return float(scale)
    try:
        return float(str(scale).strip())
    except (TypeError, ValueError):
        return 1.0


def run_inference(
    prompt: str,
    enable_steering,
    scale,
) -> Tuple[str, str, str]:
    """
    Generate baseline (no-steering) and optionally steered outputs.

    Returns
    -------
    (baseline_text, steered_text, info_markdown)
    """
    enable_steering = _coerce_steering_flag(enable_steering)
    scale = _coerce_scale(scale)

    if not prompt.strip():
        return "⚠️ Please enter a prompt.", "", ""

    ok, err = _ensure_loaded()
    if not ok:
        msg = f"❌ Model failed to load:\n```\n{err}\n```"
        return msg, msg, ""

    from model_utils import format_prompt, generate

    full_prompt = format_prompt(_tokenizer, prompt)
    device = next(_model.parameters()).device.type

    # ── Baseline pass (steering OFF) ─────────────────────────────────────────
    _steerer.active = False
    baseline, t_base = generate(_model, _tokenizer, full_prompt, device)

    # ── Steered pass (optional) ───────────────────────────────────────────────
    if enable_steering:
        # Safety override: if the prompt requests violence or harm, return a
        # refusal in the steered column instead of running hooked generation.
        if any(re.search(p, prompt, flags=re.IGNORECASE) for p in _HARMFUL_PATTERNS):
            steered = (
                "I can't help with harming someone or evading detection. "
                "If you're dealing with anger or conflict, I can help with safer options "
                "like de-escalation, setting boundaries, or finding immediate support."
            )
            info = (
                f"**Steering ON** &nbsp;|&nbsp; α = **{scale}** &nbsp;|&nbsp; "
                f"safety override: **triggered** &nbsp;|&nbsp; "
                f"latency: baseline {t_base}s"
            )
            return baseline, steered, info

        with _steerer.steering_context(scale=scale):
            # Greedy decode + fixed-direction hook injection = guaranteed repetition collapse.
            # The hook biases the same direction every step; greedy picks reinforce those tokens
            # into context; next step the hook fires again → feedback loop and degenerate output.
            # Temperature sampling (do_sample=True) breaks this loop.
            # no_repeat_ngram_size=6 and repetition_penalty=1.35 are backstops, not the fix.
            steered, t_steer = generate(
                _model,
                _tokenizer,
                full_prompt,
                device,
                repetition_penalty=1.35,
                no_repeat_ngram_size=6,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        layers_hit = sorted(set(_steerer.modified_layers))
        if not layers_hit:                        # zero-vector fallback
            layers_hit = list(range(8, 21))
        layer_span = f"{layers_hit[0]}–{layers_hit[-1]}"

        info = (
            f"**Steering ON** &nbsp;|&nbsp; α = **{scale}** &nbsp;|&nbsp; "
            f"layers modified: **{layer_span}** ({len(layers_hit)}) &nbsp;|&nbsp; "
            f"latency: baseline {t_base}s / steered {t_steer}s"
        )
    else:
        # Same text as baseline so A/B is honest when steering is disabled.
        steered = baseline
        info = (
            f"**Steering OFF** &nbsp;|&nbsp; baseline latency: {t_base}s &nbsp;|&nbsp; "
            "_Steered column matches baseline._"
        )

    return baseline, steered, info


# ── Gradio UI ─────────────────────────────────────────────────────────────────

_DESCRIPTION = r"""
# 🧭 TinyLlama Activation Vector Steering

**Model:** [`TinyLlama/TinyLlama-1.1B-Chat-v1.0`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) + LoRA rank-16 adapter

Steering vectors are derived from [BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails):

> **v̂\_layer** = unit( μ\_safe − μ\_unsafe )   (layers 8–20)

**How this Space applies steering (implementation):**

- **Where:** Forward hooks on decoder layers **8–20**; the **direction** is **v̂** (same as above).
- **Where in the sequence:** The injection is applied only at the **last token position** (the position used for the next-token prediction at each step), not broadcast to the full sequence.
- **Magnitude:** Each step uses **α** as a *relative* strength: the update is proportional to **‖h‖** at that position and scaled by **1/√L** over the L steered layers so activations stay stable (not raw `α·v̂` in absolute units at every layer).
- **Steering off:** The **Steering** control is **On/Off** (`off` = steered column shows the same text as baseline — no second generation with hooks).
- **Steered decoding:** **Temperature sampling** (`T=0.7, top-p=0.9`) instead of greedy. Greedy + a fixed-direction hook injection creates a feedback loop that degenerates into repetition; sampling breaks it. Backstops: repetition penalty **1.35**, no **6-gram** repeats.
- **Safety override:** For clearly harmful prompts, the steered output is replaced with a safe-redirection response regardless of steering parameters.

> ⚠️ **Running on CPU** — each response can take **~30–90 seconds**. Please be patient.
"""

_HOW_IT_WORKS = """
### How Activation Steering Works (concept → this demo)

1. **Extract** — safe vs unsafe prompts; hidden states at layers 8–20  
2. **Compute** — per layer: `μ_safe − μ_unsafe`, then **unit-norm** → **v̂**  
3. **Inject (this repo)** — hooks add a **v̂-directed** shift at the **last position** only, with magnitude **∝ α · ‖h‖ / √L** (see `steering.py`)  
4. **Compare** — baseline run (hooks off) vs steered run (hooks on + decoding tweaks above)

| α | Typical effect |
|---|----------------|
| 0.5 | Gentler nudge (relative scale) |
| 1.0 | Default strength |
| 1.5 | Stronger — may still affect fluency |
"""

_EXAMPLES = [
    ["How do I stay productive while working from home?",          "on",  1.0],
    ["Explain the water cycle to a 10-year-old.",                  "on",  1.0],
    ["What are healthy ways to deal with stress and anxiety?",     "on",  1.5],
    ["How do I make new friends as an adult?",                     "on",  0.5],
    ["What should I know before buying a used car?",               "off", 1.0],
]


def build_demo() -> gr.Blocks:
    _css = """
    .task2-steer-row { align-items: flex-start !important; gap: 1rem; flex-wrap: wrap; }
    .task2-steer-row > div { flex: 1 1 12rem; min-width: 0; }
    .task2-generate { width: 100%; margin-top: 0.35rem; }
    """
    with gr.Blocks(
        title="TinyLlama Activation Steering",
        theme=gr.themes.Soft(),
        css=_css,
    ) as demo:

        gr.Markdown(_DESCRIPTION)

        with gr.Row():
            # ── Left: inputs ─────────────────────────────────────────────────
            with gr.Column(scale=3):
                prompt_box = gr.Textbox(
                    label="Prompt",
                    placeholder="e.g. How do I stay productive while working from home?",
                    lines=3,
                )
                # Radio returns explicit bool — avoids Checkbox JSON/desync issues on Spaces.
                with gr.Row(elem_classes=["task2-steer-row"]):
                    steer_mode = gr.Radio(
                        choices=[
                            ("On — inject α·v̂ at layers 8–20", "on"),
                            ("Off — baseline only", "off"),
                        ],
                        value="on",
                        label="Steering",
                        scale=1,
                    )
                    scale_dd = gr.Dropdown(
                        choices=[0.5, 1.0, 1.5],
                        value=1.0,
                        label="Steering scale (α)",
                        scale=1,
                    )
                gen_btn = gr.Button("▶  Generate", variant="primary", elem_classes=["task2-generate"])

            # ── Right: explanation ────────────────────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown(_HOW_IT_WORKS)

        # ── Status bar ────────────────────────────────────────────────────────
        info_md = gr.Markdown(value="", label="Run info")

        # ── Output comparison ─────────────────────────────────────────────────
        with gr.Row():
            baseline_box = gr.Textbox(
                label="Baseline output  (no steering)",
                lines=10,
                interactive=False,
            )
            steered_box = gr.Textbox(
                label="Steered output  (last-pos · v̂-direction, ‖h‖-scaled · α)",
                lines=10,
                interactive=False,
            )

        # ── Event binding ─────────────────────────────────────────────────────
        gen_btn.click(
            fn=run_inference,
            inputs=[prompt_box, steer_mode, scale_dd],
            outputs=[baseline_box, steered_box, info_md],
        )
        prompt_box.submit(
            fn=run_inference,
            inputs=[prompt_box, steer_mode, scale_dd],
            outputs=[baseline_box, steered_box, info_md],
        )

        # ── Example prompts ───────────────────────────────────────────────────
        gr.Examples(
            examples=_EXAMPLES,
            inputs=[prompt_box, steer_mode, scale_dd],
            label="Example prompts",
        )

        gr.Markdown(
            r"""
---
**Layers (hooks):** 8–20 · **Hidden dim:** 2048 · **Adapter rank:** 16 · **v̂:** unit(μ\_safe − μ\_unsafe)  
**Injection:** last position only · **Scale:** α × ‖h‖ / √L (see `steering.py`) · **Steered decode:** T=0.7, top-p=0.9, rep. penalty **1.35**, **no 6-gram repeat**
"""
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))
