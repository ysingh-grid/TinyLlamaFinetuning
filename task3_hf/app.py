"""
Task 3 — Attention Visualization & Token Attribution (Hugging Face Space).

Precomputed artifacts in `results/` (run `generate_artifacts.py` locally to refresh).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def _load_json(name: str) -> dict | list:
    p = RESULTS / name
    if not p.exists():
        return {}
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def _load_md(name: str) -> str:
    p = RESULTS / name
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


def _img(name: str) -> str | None:
    p = RESULTS / name
    return str(p) if p.exists() else None


def _html_report_for_gradio(html_path: Path) -> str:
    """
    `attention_report.html` uses paths relative to `results/` (`heatmaps/*.png`).
    In `gr.HTML`, the browser resolves those against the Space root and images 404.
    Rewrite to Gradio 5 static file URLs (requires `allowed_paths` on launch).
    """
    txt = html_path.read_text(encoding="utf-8")
    # Route: GET /gradio_api/file={path} — path must be under allowed_paths (repo root).
    return txt.replace(
        'src="heatmaps/',
        'src="/gradio_api/file=results/heatmaps/',
    )


def _md_summary() -> str:
    s = _load_json("summary.json")
    if not s:
        return "_Run `python generate_artifacts.py` to populate `results/`._"
    return f"""
### Aggregate metrics (20 sample instructions)

| Metric | Value |
|--------|------:|
| Mean cosine similarity (base vs LoRA), averaged over heads | **{s.get('mean_cosine_base_vs_lora_per_sample', 0):.4f}** |
| Std across samples | {s.get('std_cosine_base_vs_lora_per_sample', 0):.4f} |
| Heads classified **instruction-following** (median split on attn to prompt) | **{s.get('instruction_following_heads', '?')}** |
| Heads classified **completion-local** | **{s.get('completion_local_heads', '?')}** |

LoRA adapter: **rank 16** Alpaca-style fine-tune. Base: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.
"""


def _cluster_md() -> str:
    h = _load_json("head_clusters.json")
    if not h:
        return "_No clustering data._"
    return f"""
**Agglomerative clustering** on **cosine distance** of per-head attention fingerprints (32×32 pooled maps, averaged over 20 generations).

- **Layers:** {h.get('n_layers', '?')} · **Heads per layer:** {h.get('n_heads', '?')}
- **Clusters:** {h.get('n_clusters', '?')}
- **Instruction-ratio threshold (median):** {h.get('threshold_instruction_ratio', 0):.4f}

See `head_roles.json` in the repo for per-head labels (**instruction_following** vs **completion_local**).
"""


def build_app() -> gr.Blocks:
    summary = _load_json("summary.json")
    heatmap_dir = RESULTS / "heatmaps"
    html_path = RESULTS / "attention_report.html"

    intro = r"""
# 🔎 TinyLlama LoRA — Attention & Token Attribution (Task 3)

**Goal:** interpret how **LoRA fine-tuning** changes **attention patterns** vs the **base** model on the **same** generated token sequences.

- **20 instructions** → generate with LoRA, then forward **base** and **LoRA** on identical token ids.
- **Head clustering:** cosine similarity of pooled attention maps → agglomerative clusters.
- **Head roles (empirical):** heads with high mean attention from **completion** positions to **instruction** tokens vs **completion** tokens → *instruction-following* vs *completion-local*.
- **Attention rollout** + **last-row attribution** for **5** example generations (PNGs + combined HTML).

Precomputed files live under **`results/`** in this Space repo.
"""

    with gr.Blocks(
        title="TinyLlama Attention Visualization (Task 3)",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(intro)

        with gr.Tabs():
            # ── Tab 1: Summary ──────────────────────────────────────────────
            with gr.Tab("📌 Summary"):
                gr.Markdown(_md_summary())
                gr.Markdown(_cluster_md())

            # ── Tab 2: Base vs LoRA comparison charts ───────────────────────
            with gr.Tab("📊 Base vs LoRA Comparison"):
                gr.Markdown(
                    "### Attention score Δ (LoRA − Base) across all layers and heads\n\n"
                    "These charts show how LoRA fine-tuning shifts per-head attention scores "
                    "on **instruction-following** and **completion** segments, computed over "
                    "20 sample prompts on identical token sequences."
                )

                cmp = _img("base_vs_lora_comparison.png")
                if cmp:
                    gr.Image(
                        cmp,
                        label="Base vs LoRA — per-sample head cosine similarity & net Δ",
                        show_label=True,
                    )

                diff = _img("base_vs_lora_attention_diff.png")
                if diff:
                    gr.Image(
                        diff,
                        label="Attention difference heatmaps (Δ instruction / Δ completion / layer-wise mean)",
                        show_label=True,
                    )

                if not cmp and not diff:
                    gr.Markdown("_Comparison PNGs not found — run `generate_artifacts.py` locally._")

                gr.Markdown("### Per-instruction cosine similarity table")
                df = _load_json("base_vs_lora_per_sample.json")
                if isinstance(df, list) and df:
                    hdr = "| # | Instruction (truncated) | Mean head cos sim |\n|---|-------------------------|------------------:|\n"
                    rows = []
                    for i, row in enumerate(df[:25]):
                        ins = str(row.get("instruction", ""))[:70].replace("|", "\\|")
                        rows.append(f"| {i+1} | {ins} | {row.get('mean_head_cosine_sim', 0):.4f} |")
                    gr.Markdown(hdr + "\n".join(rows))
                else:
                    gr.Markdown("_No per-sample table._")

            # ── Tab 3: Layer-wise analysis ───────────────────────────────────
            with gr.Tab("📈 Layer Analysis"):
                gr.Markdown(
                    "### Layer-wise instruction-following vs completion attention\n\n"
                    "Mean attention scores per layer for both models across instruction and completion segments. "
                    "Negative **Δ net** means LoRA shifts attention toward completion context."
                )
                layer_md = _load_md("layer_analysis.md")
                if layer_md:
                    gr.Markdown(layer_md)
                else:
                    gr.Markdown("_`results/layer_analysis.md` not found._")

            # ── Tab 4: Head clustering ───────────────────────────────────────
            with gr.Tab("🔵 Head Clustering"):
                gr.Markdown(
                    "### Agglomerative head clustering (cosine distance)\n\n"
                    "Attention fingerprints (32×32 pooled maps averaged over 20 prompts) are "
                    "clustered with average-linkage cosine distance for both **Base** and **LoRA** models. "
                    "The comparison panel shows how cluster membership shifts after fine-tuning."
                )

                base_cl = _img("head_clustering_Base.png")
                if base_cl:
                    gr.Image(base_cl, label="Base model — dendrogram & similarity matrix", show_label=True)

                lora_cl = _img("head_clustering_LoRA.png")
                if lora_cl:
                    gr.Image(lora_cl, label="LoRA model — dendrogram & similarity matrix", show_label=True)

                cmp_cl = _img("head_clustering_Comparison.png")
                if cmp_cl:
                    gr.Image(
                        cmp_cl,
                        label="Cluster comparison — centroid similarity & membership counts (Base vs LoRA)",
                        show_label=True,
                    )

                if not base_cl and not lora_cl and not cmp_cl:
                    gr.Markdown("_Clustering PNGs not found — run `generate_artifacts.py` locally._")

            # ── Tab 5: Multi-head visualization ─────────────────────────────
            with gr.Tab("🔬 Multi-Head View"):
                gr.Markdown(
                    "### Multi-head attention visualization (LoRA, prompt 0)\n\n"
                    "Top-4 **instruction-following** heads and top-4 **completion-local** heads for the first "
                    "sample prompt, showing response-token × prompt-token attention maps. "
                    "Instruction-following heads concentrate attention on the prompt column; "
                    "completion heads attend to recent response context."
                )
                mhv = _img("multi_head_visualization.png")
                if mhv:
                    gr.Image(mhv, label="Multi-head attention (LoRA, prompt 0)", show_label=True)
                else:
                    gr.Markdown("_`results/multi_head_visualization.png` not found._")

            # ── Tab 6: Rollout heatmaps ──────────────────────────────────────
            with gr.Tab("🖼️ Rollout Heatmaps"):
                gr.Markdown(
                    "Attention-rollout heatmaps and last-row token-attribution bars "
                    "(**LoRA** model, same token ids as base), for 5 example generations."
                )
                if heatmap_dir.exists():
                    imgs = sorted(heatmap_dir.glob("rollout_*.png"))
                    if imgs:
                        for r in imgs:
                            i = r.stem.replace("rollout_", "")
                            gr.Image(str(r), label=f"Example {int(i)+1} — attention rollout")
                            a = heatmap_dir / f"attr_{i}.png"
                            if a.exists():
                                gr.Image(str(a), label=f"Example {int(i)+1} — attribution (last row)")
                    else:
                        gr.Markdown("_No rollout PNGs found in `results/heatmaps/`._")
                else:
                    gr.Markdown("_No PNGs — run artifact generation._")

            # ── Tab 7: HTML report ───────────────────────────────────────────
            with gr.Tab("📄 HTML Report"):
                gr.Markdown(
                    "Static report: **rollout heatmaps** and **last-row attribution** bars for five examples. "
                    "Images are linked by **relative path** (`heatmaps/*.png`) so the HTML stays small and "
                    "preview works in this tab. Download the HTML + `heatmaps/` together if you open it offline."
                )
                if html_path.exists():
                    gr.File(str(html_path), label="Download attention_report.html")
                    try:
                        gr.HTML(_html_report_for_gradio(html_path))
                    except OSError:
                        pass
                else:
                    gr.Markdown("_Generate `results/attention_report.html` locally (`python generate_artifacts.py`)._")

    return demo


if __name__ == "__main__":
    # Allow Gradio to resolve assets under repo root if needed for previews
    build_app().launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        allowed_paths=[str(ROOT)],
    )
