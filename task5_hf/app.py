"""
Gradio Space: Task 5 — Logit lens & LoRA layer importance (precomputed `results/`).
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def _load_json(name: str) -> dict:
    p = RESULTS / name
    if not p.exists():
        return {}
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def _img(name: str) -> str | None:
    p = RESULTS / name
    return str(p) if p.exists() else None


def build_ui() -> gr.Blocks:
    corr = _load_json("correlation_metrics.json")
    abl = _load_json("ablation_results.json")
    acc = _load_json("accuracy_by_layer.json")
    meta = abl.get("metadata", {})

    pearson = corr.get("lora_accuracy", {}).get("pearson_r", float("nan"))
    spearman = corr.get("lora_accuracy", {}).get("spearman_rho", float("nan"))
    pearson_g = corr.get("accuracy_gain_lora_minus_base", {}).get("pearson_r", float("nan"))

    mlx_base = meta.get("baseline_top_layer_accuracy_mlx_first_token", meta.get("baseline_top_layer_accuracy", 0))
    pt_ref = meta.get("pytorch_top_layer_mean_assistant_span", float("nan"))

    md = f"""
### Task 5 — Logit lens & layer-wise LoRA importance

**Model:** `{meta.get("base_model", "—")}` · **Adapter:** `{meta.get("adapter_path", "—")}` · **Prompts:** {meta.get("num_prompts", "—")}

**Plots (PyTorch):** mean top-1 over teacher-forced assistant tokens (deepest layer reference **{pt_ref:.4f}**).

**Ablation / 95% rule (MLX):** first assistant token after the prompt via full `model(prompt_ids)` forward.

| Metric | Value |
|--------|------:|
| Baseline (MLX, first-token) | **{float(mlx_base):.4f}** |
| 95% of baseline threshold | {meta.get("threshold_95pct_of_baseline", 0):.4f} |
| Minimum layers (ablation, keep top-importance) | **{meta.get("minimum_layers_for_threshold", "—")}** |
| Pearson r (importance vs LoRA accuracy) | {pearson:.4f} |
| Pearson r (importance vs Δ accuracy) | {pearson_g:.4f} |
| Spearman ρ (importance vs LoRA accuracy) | {spearman:.4f} |

See `minimum_layers_analysis.md` in the repo for the full ablation table and retrain notes.
"""

    with gr.Blocks(title="Task 5 — Logit lens & LoRA") as demo:
        gr.Markdown(md)
        with gr.Row():
            p1 = _img("prediction_accuracy_by_layer.png")
            p2 = _img("prediction_gain_by_layer.png")
            if p1:
                gr.Image(value=p1, label="LoRA accuracy vs depth")
            if p2:
                gr.Image(value=p2, label="Δ accuracy (LoRA − base) vs depth")
        with gr.Row():
            p3 = _img("lens_and_importance.png")
            p4 = _img("importance_accuracy_correlation.png")
            if p3:
                gr.Image(value=p3, label="Accuracy + LoRA L₂ importance")
            if p4:
                gr.Image(value=p4, label="Importance vs accuracy gain")
        with gr.Accordion("Raw JSON (accuracy)", open=False):
            gr.Code(json.dumps(acc, indent=2) if acc else "{}", language="json")
        with gr.Accordion("Correlation metrics", open=False):
            gr.Code(json.dumps(corr, indent=2) if corr else "{}", language="json")
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
