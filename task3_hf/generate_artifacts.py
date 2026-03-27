#!/usr/bin/env python3
"""
Precompute attention analysis for Task 3: 20 instructions, clustering, HTML heatmaps (5 examples).

Usage:
  python generate_artifacts.py --device cpu --adapter ./adapters/r16

Requires: torch, transformers, peft, scikit-learn, matplotlib, numpy
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SAMPLES = ROOT / "samples" / "instructions.json"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def rebuild_attention_report(
    results_dir: Path = RESULTS,
    embed_images: bool = False,
) -> None:
    """
    Regenerate `heatmaps/*.png` and `attention_report.html` from existing
    `heatmap_examples.json` + `summary.json` (no model load). Use after editing viz code
    or to fix HTML without re-running the full pipeline.
    """
    sys.path.insert(0, str(ROOT))
    from html_viz import build_combined_html, plot_attribution_bar, plot_rollout_heatmap

    hp = results_dir / "heatmap_examples.json"
    sp = results_dir / "summary.json"
    if not hp.exists():
        raise SystemExit(f"Missing {hp}; run a full generate first.")
    heatmap_examples: list = json.loads(hp.read_text(encoding="utf-8"))
    summary = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}

    hm_dir = results_dir / "heatmaps"
    hm_dir.mkdir(parents=True, exist_ok=True)
    sections: list[tuple[str, str, str]] = []
    for ex_i, ex in enumerate(heatmap_examples):
        roll = np.asarray(ex["rollout"], dtype=np.float32)
        toks = ex["tokens"]
        pl = int(ex["prompt_len"])
        title = f"Example {ex_i + 1}: attention rollout (LoRA)"
        png_roll = hm_dir / f"rollout_{ex_i}.png"
        plot_rollout_heatmap(roll, toks, pl, title, png_roll)
        png_attr = hm_dir / f"attr_{ex_i}.png"
        attr = np.asarray(ex["last_row_attribution"], dtype=np.float32)
        plot_attribution_bar(
            attr,
            toks,
            pl,
            f"Token attribution (rollout last row) — ex {ex_i + 1}",
            png_attr,
        )
        desc = str(ex["instruction"])[:300]
        sections.append((f"Example {ex_i + 1}", desc, str(png_roll.resolve())))
        sections.append(
            (
                f"Example {ex_i + 1} — attribution",
                "Rollout mass from last position to all positions (instruction vs completion shaded).",
                str(png_attr.resolve()),
            )
        )

    mean_cos = float(summary.get("mean_cosine_base_vs_lora_per_sample", 0.0))
    extra = (
        f"<p>Mean cosine similarity (base vs LoRA) averaged over all heads: "
        f"<b>{mean_cos:.4f}</b></p>"
    )
    build_combined_html(
        sections,
        results_dir / "attention_report.html",
        extra_md=extra,
        embed_images=embed_images,
    )
    logger.info("Rebuilt %s", results_dir / "attention_report.html")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, default=ROOT / "adapters" / "r16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--quick", action="store_true", help="2 instructions only (debug)")
    parser.add_argument(
        "--embed-html-images",
        action="store_true",
        help="Inline PNGs as base64 in HTML (large file; avoid for HF Space preview).",
    )
    parser.add_argument(
        "--html-only",
        action="store_true",
        help="Only rebuild heatmap PNGs + attention_report.html from saved JSON (no model).",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))

    if args.html_only:
        rebuild_attention_report(RESULTS, embed_images=args.embed_html_images)
        return

    from attention_core import (
        attention_rollout,
        attentions_to_numpy_layers,
        base_vs_lora_head_similarity,
        cosine_similarity_matrix,
        head_feature_vector,
        instruction_completion_scores,
        load_instructions,
        load_base_model,
        load_lora_merged,
        load_tokenizer,
        run_pair_base_lora,
    )
    from html_viz import build_combined_html, plot_attribution_bar, plot_rollout_heatmap, tokens_for_plot

    try:
        from sklearn.cluster import AgglomerativeClustering
    except ImportError as e:
        raise SystemExit("pip install scikit-learn") from e

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    instructions = load_instructions(SAMPLES)
    if args.quick:
        instructions = instructions[:2]

    logger.info("Loading tokenizer + models…")
    tok = load_tokenizer()
    model_base = load_base_model(device)
    model_lora = load_lora_merged(Path(args.adapter), device)

    n_layers = len(model_base.model.layers)
    n_heads = model_base.config.num_attention_heads

    # Accumulate head features (LoRA) + role scores
    head_dim = 32 * 32
    sum_feat = np.zeros((n_layers, n_heads, head_dim), dtype=np.float64)
    sum_instr = np.zeros((n_layers, n_heads), dtype=np.float64)
    sum_comp = np.zeros((n_layers, n_heads), dtype=np.float64)
    per_sample_pair_sims: list[float] = []

    heatmap_examples: list[dict] = []

    for i, inst in enumerate(instructions):
        logger.info("[%s/%s] %s", i + 1, len(instructions), inst[:60])
        rb, rl = run_pair_base_lora(
            model_base,
            model_lora,
            tok,
            inst,
            device,
            max_new_tokens=args.max_new_tokens,
        )

        nb = attentions_to_numpy_layers(rb.attentions)
        nl = attentions_to_numpy_layers(rl.attentions)
        per_sample_pair_sims.append(base_vs_lora_head_similarity(nb, nl))

        pl, tl = rl.prompt_len, rl.total_len
        for li in range(n_layers):
            for hi in range(n_heads):
                ah = torch.from_numpy(nl[li][hi])
                ti, tc = instruction_completion_scores(ah, pl, tl)
                sum_instr[li, hi] += ti
                sum_comp[li, hi] += tc
                v = head_feature_vector(ah)
                sum_feat[li, hi] += v

        roll = attention_rollout(rl.attentions)
        last_row = roll[-1, :].copy()
        ids = rl.input_ids[0].tolist()
        token_strs = tokens_for_plot(tok, ids)

        # Full rollout JSON only for 5 demo examples (keep repo small)
        if len(heatmap_examples) < 5:
            heatmap_examples.append(
                {
                    "instruction": inst,
                    "prompt_len": pl,
                    "total_len": tl,
                    "rollout": roll.tolist(),
                    "last_row_attribution": last_row.tolist(),
                    "tokens": token_strs,
                    "pair_base_lora_mean_sim": per_sample_pair_sims[-1],
                }
            )

    n_inst = len(instructions)
    mean_feat = sum_feat / max(n_inst, 1)
    mean_instr = sum_instr / max(n_inst, 1)
    mean_comp = sum_comp / max(n_inst, 1)

    # Feature matrix [N_heads, D]
    feats = mean_feat.reshape(-1, head_dim).astype(np.float32)
    sim = cosine_similarity_matrix(feats)
    dist = 1.0 - np.clip(sim, -1.0, 1.0)
    np.fill_diagonal(dist, 0.0)

    n_clusters = min(8, max(2, feats.shape[0] // 32))
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average",
    )
    labels = clustering.fit_predict(dist)

    # Head roles: ratio instruction vs completion
    ratio = mean_instr / (mean_instr + mean_comp + 1e-8)
    flat_ratio = ratio.reshape(-1)
    thr = np.median(flat_ratio)
    roles = []
    for li in range(n_layers):
        for hi in range(n_heads):
            idx = li * n_heads + hi
            r = float(ratio[li, hi])
            roles.append(
                {
                    "layer": li,
                    "head": hi,
                    "cluster": int(labels[idx]),
                    "mean_attn_to_instruction": float(mean_instr[li, hi]),
                    "mean_attn_to_completion": float(mean_comp[li, hi]),
                    "instruction_ratio": r,
                    "role": "instruction_following" if r >= thr else "completion_local",
                }
            )

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "head_clusters.json").write_text(
        json.dumps(
            {
                "n_clusters": n_clusters,
                "n_layers": n_layers,
                "n_heads": n_heads,
                "labels": labels.tolist(),
                "threshold_instruction_ratio": float(thr),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (RESULTS / "head_roles.json").write_text(json.dumps(roles, indent=2), encoding="utf-8")

    summary = {
        "n_instructions": n_inst,
        "mean_cosine_base_vs_lora_per_sample": float(np.mean(per_sample_pair_sims)),
        "std_cosine_base_vs_lora_per_sample": float(np.std(per_sample_pair_sims)),
        "instruction_following_heads": sum(1 for r in roles if r["role"] == "instruction_following"),
        "completion_local_heads": sum(1 for r in roles if r["role"] == "completion_local"),
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Per-sample base vs LoRA (for bar chart / table)
    per_sample = [
        {"instruction": instructions[j][:200], "mean_head_cosine_sim": per_sample_pair_sims[j]}
        for j in range(len(instructions))
    ]
    (RESULTS / "base_vs_lora_per_sample.json").write_text(json.dumps(per_sample, indent=2), encoding="utf-8")

    (RESULTS / "heatmap_examples.json").write_text(json.dumps(heatmap_examples, indent=2), encoding="utf-8")

    # PNG heatmaps + combined HTML (5 examples)
    hm_dir = RESULTS / "heatmaps"
    hm_dir.mkdir(parents=True, exist_ok=True)
    sections: list[tuple[str, str, str]] = []
    for ex_i, ex in enumerate(heatmap_examples):
        roll = np.asarray(ex["rollout"], dtype=np.float32)
        toks = ex["tokens"]
        pl = int(ex["prompt_len"])
        title = f"Example {ex_i + 1}: attention rollout (LoRA)"
        png_roll = hm_dir / f"rollout_{ex_i}.png"
        plot_rollout_heatmap(roll, toks, pl, title, png_roll)
        png_attr = hm_dir / f"attr_{ex_i}.png"
        attr = np.asarray(ex["last_row_attribution"], dtype=np.float32)
        plot_attribution_bar(
            attr,
            toks,
            pl,
            f"Token attribution (rollout last row) — ex {ex_i + 1}",
            png_attr,
        )
        desc = ex["instruction"][:300]
        sections.append(
            (
                f"Example {ex_i + 1}",
                desc,
                str(png_roll),
            )
        )
        sections.append(
            (
                f"Example {ex_i + 1} — attribution",
                "Rollout mass from last position to all positions (instruction vs completion shaded).",
                str(png_attr),
            )
        )

    extra = (
        f"<p>Mean cosine similarity (base vs LoRA) averaged over all heads: "
        f"<b>{summary['mean_cosine_base_vs_lora_per_sample']:.4f}</b></p>"
    )
    build_combined_html(
        sections,
        RESULTS / "attention_report.html",
        extra_md=extra,
        embed_images=args.embed_html_images,
    )

    logger.info("Wrote results to %s", RESULTS)


if __name__ == "__main__":
    main()
