#!/usr/bin/env python3
"""
Task 5: Logit lens (per-layer), LoRA L₂ importance, correlation, ablation, selective retrain.

Outputs under ``results/``:
  - layer_predictions.json (LoRA)
  - base_layer_predictions.json (base model)
  - accuracy_by_layer.json (base, LoRA, gain)
  - lora_importance_by_layer.json
  - correlation_metrics.json
  - prediction_accuracy_by_layer.png (2-panel: base vs merged LoRA + zoom), prediction_gain_by_layer.png
  - lens_and_importance.png (normalized L₂ + accuracy; raw L₂ bars)
  - importance_accuracy_correlation.png (importance vs LoRA accuracy + vs gain)
  - ablation_results.json, minimum_layers_analysis.md
  - retrained_adapter/ (optional)

Usage::
  python run_analysis.py --adapter ../adapters/tinyllama-lora-alpaca --num-prompts 100
  python run_analysis.py --smoke-test --skip-retrain
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import torch
from mlx_lm import load
from mlx_lm.tuner.datasets import CacheDataset, load_dataset
from mlx_lm.tuner.trainer import TrainingArgs, train
import mlx.optimizers as optim

from logit_lens_core import (
    BASE_MODEL,
    accuracy_by_layer,
    convert_selected_layers_to_lora,
    correlation_for_layers,
    discover_lora_keys,
    dump_json,
    evaluate_ablation_accuracy,
    evaluate_first_token_accuracy_mlx,
    get_test_samples,
    load_adapter_importance,
    load_selected_adapter_weights,
)
from logit_lens_torch import (
    evaluate_logit_lens_pytorch,
    layer_accuracy_gain,
    load_base_causal,
    load_lora_merged,
    load_tokenizer,
)

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
DATA_DEFAULT = ROOT / "data" / "test.jsonl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_PLOT = True
except Exception:
    HAS_PLOT = False


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def plot_logit_lens_base_vs_lora(
    layer_ids: List[int],
    base_accs: List[float],
    lora_accs: List[float],
    adapted_layers: List[int],
    out_path: Path,
) -> None:
    """
    Two panels: full depth (shows why early layers look like “flat zeros”) and zoom on adapted layers.
    """
    if not HAS_PLOT:
        return
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [1.1, 1]})
    ax_top.plot(layer_ids, base_accs, "o-", label="Base model", color="#1f77b4", markersize=5)
    ax_top.plot(layer_ids, lora_accs, "o-", label="Merged LoRA", color="#ff7f0e", markersize=5)
    if adapted_layers:
        lo, hi = min(adapted_layers), max(adapted_layers)
        ax_top.axvspan(lo - 0.45, hi + 0.45, alpha=0.12, color="green", label="Layers with LoRA weights")
    ymax = max(max(base_accs) if base_accs else 0.0, max(lora_accs) if lora_accs else 0.0, 0.02)
    ax_top.set_ylim(0, min(1.0, ymax * 1.08))
    ax_top.set_ylabel("Mean top-1 accuracy")
    ax_top.set_title(
        "Logit lens: each layer’s hidden state → final RMSNorm → lm_head (same unembedding as the full model)"
    )
    ax_top.legend(loc="lower right", fontsize=9)
    ax_top.grid(alpha=0.3)
    ax_top.set_xticks(layer_ids[:: max(1, len(layer_ids) // 12)])

    ax_bot.plot(layer_ids, base_accs, "o-", label="Base", color="#1f77b4", markersize=6)
    ax_bot.plot(layer_ids, lora_accs, "o-", label="Merged LoRA", color="#ff7f0e", markersize=6)
    if adapted_layers:
        lo, hi = min(adapted_layers), max(adapted_layers)
        ax_bot.set_xlim(lo - 0.5, hi + 0.5)
        seg_b = [b for lid, b in zip(layer_ids, base_accs) if lo <= lid <= hi]
        seg_l = [a for lid, a in zip(layer_ids, lora_accs) if lo <= lid <= hi]
        yhi = max(max(seg_b) if seg_b else 0, max(seg_l) if seg_l else 0, 1e-6)
        ax_bot.set_ylim(0, min(1.0, yhi * 1.12))
    ax_bot.set_xlabel("Transformer layer index")
    ax_bot.set_ylabel("Mean top-1 accuracy")
    ax_bot.set_title("Zoom: layers where LoRA adapters exist (effect is visible here)")
    ax_bot.legend(loc="best", fontsize=9)
    ax_bot.grid(alpha=0.3)
    fig.text(
        0.5,
        0.01,
        "Early layers: representations are not yet aligned with vocabulary through lm_head (near-zero accuracy is expected). "
        "Compare Base vs LoRA on the lower panel.",
        ha="center",
        fontsize=9,
        style="italic",
    )
    plt.tight_layout(rect=[0, 0.035, 1, 1])
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_gain_on_adapted_layers(
    layer_ids: List[int],
    gain_by_layer: Dict[int, float],
    adapted_layers: List[int],
    out_path: Path,
) -> None:
    """Bar chart only on layers that carry LoRA (avoids a flat chart of zeros)."""
    if not HAS_PLOT:
        return
    adapted = sorted(set(adapted_layers) & set(layer_ids))
    if not adapted:
        return
    gains = [gain_by_layer.get(i, 0.0) for i in adapted]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["tab:green" if g >= 0 else "tab:red" for g in gains]
    ax.bar(adapted, gains, color=colors, alpha=0.88, edgecolor="black", linewidth=0.3)
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xlabel("Layer index (LoRA-adapted only)")
    ax.set_ylabel("Δ accuracy (LoRA − base)")
    ax.set_title("LoRA effect on logit-lens accuracy (mean over teacher-forced assistant tokens)")
    ax.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_importance_and_accuracy_stacked(
    layer_ids: List[int],
    lora_accs: List[float],
    imps: List[float],
    adapted_layers: List[int],
    out_path: Path,
) -> None:
    """
    Top: normalized L₂ mass and LoRA accuracy on 0–1 scale (comparable).
    Bottom: raw L₂ mass per layer (interpretable magnitude).
    """
    if not HAS_PLOT:
        return
    imp_max = max(imps) if imps and max(imps) > 0 else 1.0
    imp_norm = [v / imp_max for v in imps]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    w = 0.38
    x = np.array(layer_ids, dtype=float)
    ax1.bar(x - w / 2, imp_norm, width=w, alpha=0.55, label="L₂ mass (normalized to max=1)", color="#c44e52")
    ax1.plot(layer_ids, lora_accs, "o-", color="#4c72b0", linewidth=2, markersize=5, label="LoRA logit-lens accuracy")
    ax1.set_ylabel("0–1 scale")
    ax1.set_title("Normalized adapter mass vs logit-lens accuracy (same vertical scale)")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.25, axis="y")

    ax2.bar(layer_ids, imps, color="#8172b3", alpha=0.85, edgecolor="black", linewidth=0.3)
    if adapted_layers:
        lo, hi = min(adapted_layers), max(adapted_layers)
        ax2.axvspan(lo - 0.45, hi + 0.45, alpha=0.08, color="green")
    ax2.set_xlabel("Layer index")
    ax2.set_ylabel("Sum of ‖LoRA‖₂ (raw)")
    ax2.set_title("Raw LoRA parameter mass per layer (adapter checkpoint)")
    ax2.grid(alpha=0.25, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_scatter(
    xs: List[float],
    ys: List[float],
    xlabel: str,
    ylabel: str,
    title: str,
    stats: Dict[str, float],
    out_path: Path,
    layer_labels: List[int] | None = None,
) -> None:
    if not HAS_PLOT:
        return
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter(xs, ys, s=85, color="tab:purple", alpha=0.85, edgecolors="white", linewidths=0.5, zorder=3)
    if layer_labels is not None and len(layer_labels) == len(xs):
        for x, y, lab in zip(xs, ys, layer_labels):
            ax.annotate(str(lab), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8, alpha=0.9)
    if len(xs) >= 2 and max(xs) > min(xs):
        slope, intercept = np.polyfit(xs, ys, 1)
        xgrid = np.linspace(min(xs), max(xs), 100)
        ax.plot(xgrid, slope * xgrid + intercept, color="tab:gray", linewidth=2, zorder=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    pr = stats.get("pearson_r", float("nan"))
    sp = stats.get("spearman_rho", float("nan"))
    ax.text(
        0.02,
        0.98,
        f"Pearson r={pr:.3f}\nSpearman ρ={sp:.3f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.88),
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_minimum_layers_markdown(
    path: Path,
    baseline_acc: float,
    threshold: float,
    ablation_rows: List[Dict[str, Any]],
    min_layers: int,
    retrain_summary: Dict[str, Any],
    pytorch_top_layer_acc: float,
) -> None:
    lines = [
        "# Minimum layers for 95% instruction-following (proxy metrics)",
        "",
        "**Ablation (MLX)** uses a **full forward** on the prompt and the **first assistant token**",
        "top-1 at the last prompt position (same as `model(prompt_ids)`). The **95% rule** compares",
        "ablation accuracy to the **full-adapter** baseline under that same definition.",
        "",
        "**PyTorch logit-lens curves** in `results/*.png` use **mean top-1** over up to 32",
        "teacher-forced assistant positions (more stable than a single token). Reference: deepest layer **"
        f"{pytorch_top_layer_acc:.3f}**.",
        "",
        f"- Baseline (full LoRA, MLX full forward, first assistant token): **{baseline_acc:.3f}**",
        f"- 95% of that baseline: **{threshold:.3f}**",
        f"- Minimum number of kept (highest-importance) LoRA layers to reach threshold: **{min_layers}**",
        "",
        "## Layer ablation (keep top-importance LoRA layers, zero others)",
        "",
        "| Kept layers | Accuracy |",
        "|---:|---:|",
    ]
    for row in ablation_rows:
        lines.append(f"| {row['kept_layers']} | {row['accuracy']:.3f} |")
    lines.extend(
        [
            "",
            "## Selective retrain",
            "",
            f"- Strategy: **{retrain_summary.get('strategy', 'n/a')}**",
            f"- Layers trained: `{retrain_summary.get('selected_layers', [])}`",
            f"- Iterations: **{retrain_summary.get('iters', 0)}**",
            f"- Output: `{retrain_summary.get('adapter_path', '')}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def selective_retrain(
    adapter_path: Path,
    train_layer_indices: List[int],
    retrain_iters: int,
    output_dir: Path,
    data_dir: Path,
    max_seq_length: int = 512,
) -> Dict[str, Any]:
    """Attach LoRA only on ``train_layer_indices``, load weights, run MLX-LM short SFT."""
    base_model, tokenizer = load(BASE_MODEL)

    config_path = adapter_path / "adapter_config.json"
    with config_path.open("r", encoding="utf-8") as fh:
        adapter_cfg = json.load(fh)
    lora_config = adapter_cfg.get("lora_parameters", {})
    lora_config["keys"] = discover_lora_keys(adapter_path)

    convert_selected_layers_to_lora(base_model, train_layer_indices, lora_config)
    load_selected_adapter_weights(base_model, adapter_path)

    dataset_args = SimpleNamespace(
        data=str(data_dir),
        hf_dataset=False,
        train=True,
        test=False,
        mask_prompt=True,
        max_seq_length=max_seq_length,
    )
    train_dataset, valid_dataset, _ = load_dataset(dataset_args, tokenizer)
    train_dataset = CacheDataset(train_dataset)
    valid_dataset = CacheDataset(valid_dataset)

    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_file = output_dir / "adapters.safetensors"
    train_args = TrainingArgs(
        batch_size=1,
        iters=retrain_iters,
        val_batches=5,
        steps_per_report=10,
        steps_per_eval=25,
        steps_per_save=retrain_iters,
        max_seq_length=max_seq_length,
        adapter_file=str(adapter_file),
        grad_checkpoint=True,
        grad_accumulation_steps=1,
    )
    optimizer = optim.Adam(learning_rate=1e-5)

    train(base_model, optimizer, train_dataset, valid_dataset, args=train_args)

    adapter_config_out = {
        "base_model": BASE_MODEL,
        "source_adapter": str(adapter_path),
        "selected_layers": train_layer_indices,
        "fine_tune_type": "lora",
        "num_layers": len(train_layer_indices),
        "lora_parameters": lora_config,
    }
    dump_json(output_dir / "adapter_config.json", adapter_config_out)

    return {
        "base_model": BASE_MODEL,
        "source_adapter": str(adapter_path),
        "selected_layers": train_layer_indices,
        "iters": retrain_iters,
        "max_seq_length": max_seq_length,
        "adapter_path": str(output_dir),
        "train_dataset_size": len(train_dataset),
        "valid_dataset_size": len(valid_dataset),
        "status": "completed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 5: Logit lens & LoRA layer analysis")
    parser.add_argument("--adapter", type=Path, default=ROOT.parent / "adapters" / "tinyllama-lora-alpaca")
    parser.add_argument("--data", type=Path, default=DATA_DEFAULT, help="JSONL with messages (user/assistant)")
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--retrain-k", type=int, default=5, help="How many layers to include in selective retrain")
    parser.add_argument(
        "--retrain-strategy",
        choices=("depth", "least_importance"),
        default="depth",
        help="depth: train LoRA on layers 0..k-1; least_importance: train k adapted layers with smallest L₂ mass",
    )
    parser.add_argument("--retrain-iters", type=int, default=100)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="2 prompts and capped retrain iters; MLX first-token / ablation ratios can be noisy vs 100 prompts",
    )
    parser.add_argument("--skip-retrain", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        args.num_prompts = 2
        args.retrain_iters = min(args.retrain_iters, 10)

    adapter_path = args.adapter.resolve()
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")

    data_path = args.data.resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    ensure_clean_dir(RESULTS_DIR)
    retrain_dir = RESULTS_DIR / "retrained_adapter"

    samples = get_test_samples(data_path, args.num_prompts)
    logger.info("Loaded %d test prompts from %s", len(samples), data_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer()

    # --- PyTorch logit lens (MLX merged adapter logits are unreliable for this probe) ---
    logger.info("Loading base model (PyTorch) for logit lens…")
    model_base = load_base_causal(device)
    logger.info("Evaluating base model…")
    base_rows, base_correct, base_total = evaluate_logit_lens_pytorch(model_base, tok, samples, device)
    base_acc = accuracy_by_layer(base_correct, base_total)
    dump_json(
        RESULTS_DIR / "base_layer_predictions.json",
        {
            "metadata": {
                "base_model": BASE_MODEL,
                "num_prompts": len(base_rows),
                "target_definition": "first assistant token after chat prompt",
                "backend": "pytorch",
            },
            "samples": base_rows,
        },
    )
    del model_base
    torch.cuda.empty_cache() if device == "cuda" else None

    logger.info("Loading merged LoRA (PyTorch)…")
    model_lora_pt = load_lora_merged(adapter_path, device)
    num_layers = len(model_lora_pt.model.layers)

    logger.info("Evaluating LoRA model (%d layers)…", num_layers)
    lora_rows, lora_correct, lora_total = evaluate_logit_lens_pytorch(model_lora_pt, tok, samples, device)
    lora_acc = accuracy_by_layer(lora_correct, lora_total)
    gain = layer_accuracy_gain(base_acc, lora_acc)
    del model_lora_pt
    torch.cuda.empty_cache() if device == "cuda" else None

    # --- MLX model only for LoRA layer ablation (zeros LoRA scales per layer) ---
    logger.info("Loading MLX model for layer ablation…")
    model_lora, tokenizer_mlx = load(BASE_MODEL, adapter_path=str(adapter_path))

    mlx_baseline_first_token = evaluate_first_token_accuracy_mlx(model_lora, tokenizer_mlx, samples)

    dump_json(
        RESULTS_DIR / "layer_predictions.json",
        {
            "metadata": {
                "base_model": BASE_MODEL,
                "adapter_path": str(adapter_path),
                "num_prompts": len(lora_rows),
                "num_layers": num_layers,
                "target_definition": "first assistant token after chat prompt",
                "backend": "pytorch_merged_peft",
            },
            "samples": lora_rows,
        },
    )

    importance = load_adapter_importance(adapter_path, num_layers)
    dump_json(
        RESULTS_DIR / "lora_importance_by_layer.json",
        {
            "metadata": {
                "base_model": BASE_MODEL,
                "adapter_path": str(adapter_path),
                "num_layers": num_layers,
                "metric": "sum of L2 norms of LoRA A/B tensors per layer",
                "adapted_layers": [i for i, v in importance.items() if v > 0],
            },
            "importance_by_layer": importance,
        },
    )

    dump_json(
        RESULTS_DIR / "accuracy_by_layer.json",
        {
            "base": {str(k): v for k, v in sorted(base_acc.items())},
            "lora": {str(k): v for k, v in sorted(lora_acc.items())},
            "gain_lora_minus_base": {str(k): v for k, v in sorted(gain.items())},
        },
    )

    layer_ids = sorted(lora_acc.keys())
    acc_values = [lora_acc[i] for i in layer_ids]
    base_values = [base_acc[i] for i in layer_ids]
    imp_values = [importance.get(i, 0.0) for i in layer_ids]
    adapted_layers = [i for i in layer_ids if importance.get(i, 0.0) > 0]

    plot_logit_lens_base_vs_lora(
        layer_ids,
        base_values,
        acc_values,
        adapted_layers,
        RESULTS_DIR / "prediction_accuracy_by_layer.png",
    )
    plot_gain_on_adapted_layers(layer_ids, gain, adapted_layers, RESULTS_DIR / "prediction_gain_by_layer.png")
    plot_importance_and_accuracy_stacked(
        layer_ids,
        acc_values,
        imp_values,
        adapted_layers,
        RESULTS_DIR / "lens_and_importance.png",
    )

    corr_lora = correlation_for_layers(adapted_layers, importance, lora_acc) if adapted_layers else {}
    corr_gain = correlation_for_layers(adapted_layers, importance, gain) if adapted_layers else {}
    dump_json(
        RESULTS_DIR / "correlation_metrics.json",
        {
            "description": "Pearson/Spearman between per-layer LoRA L2 mass and logit-lens metrics",
            "lora_accuracy": corr_lora,
            "accuracy_gain_lora_minus_base": corr_gain,
        },
    )

    if adapted_layers:
        plot_correlation_scatter(
            [importance[i] for i in adapted_layers],
            [lora_acc[i] for i in adapted_layers],
            "LoRA L₂ importance",
            "LoRA logit-lens accuracy",
            "Importance vs LoRA accuracy (labels = layer index)",
            corr_lora,
            RESULTS_DIR / "importance_vs_lora_accuracy.png",
            layer_labels=adapted_layers,
        )
        plot_correlation_scatter(
            [importance[i] for i in adapted_layers],
            [gain[i] for i in adapted_layers],
            "LoRA L₂ importance",
            "Δ accuracy (LoRA − base)",
            "Importance vs accuracy gain (labels = layer index)",
            corr_gain,
            RESULTS_DIR / "importance_vs_accuracy_gain.png",
            layer_labels=adapted_layers,
        )
        # Back-compat filename
        plot_correlation_scatter(
            [importance[i] for i in adapted_layers],
            [gain[i] for i in adapted_layers],
            "LoRA L₂ importance",
            "Δ accuracy (LoRA − base)",
            "Importance vs accuracy gain (labels = layer index)",
            corr_gain,
            RESULTS_DIR / "importance_accuracy_correlation.png",
            layer_labels=adapted_layers,
        )

    pytorch_top_layer_acc = float(lora_acc[layer_ids[-1]]) if layer_ids else 0.0
    baseline_accuracy = mlx_baseline_first_token
    threshold = baseline_accuracy * 0.95

    sorted_by_imp = sorted(adapted_layers, key=lambda i: importance[i], reverse=True)
    ablation_rows: List[Dict[str, Any]] = []
    keep_candidates = sorted({1, 2, 4, 6, 8, 10, 12, len(sorted_by_imp)})
    for keep_count in keep_candidates:
        keep_count = min(keep_count, len(sorted_by_imp))
        kept = sorted_by_imp[:keep_count]
        active_accuracy = evaluate_ablation_accuracy(model_lora, tokenizer_mlx, samples, kept)
        ablation_rows.append(
            {
                "kept_layers": keep_count,
                "kept_layer_indices": kept,
                "accuracy": float(active_accuracy),
                "baseline_accuracy": float(baseline_accuracy),
                "accuracy_ratio": float(active_accuracy / baseline_accuracy) if baseline_accuracy else 0.0,
            }
        )

    min_layers = next(
        (row["kept_layers"] for row in ablation_rows if row["accuracy"] >= threshold),
        len(sorted_by_imp),
    )

    retrain_summary: Dict[str, Any] = {
        "status": "skipped",
        "strategy": args.retrain_strategy,
        "iters": args.retrain_iters,
        "selected_layers": [],
        "adapter_path": str(retrain_dir),
    }

    if not args.skip_retrain:
        k = max(0, args.retrain_k)
        if args.retrain_strategy == "depth":
            cand = sorted(adapted_layers)[:min(k, len(adapted_layers))] if adapted_layers else list(range(min(k, num_layers)))
            train_layers = cand
        else:
            train_layers = sorted(adapted_layers, key=lambda i: importance[i])[: min(k, len(adapted_layers))]
        logger.info("Selective retrain strategy=%s layers=%s", args.retrain_strategy, train_layers)
        if train_layers:
            retrain_meta = selective_retrain(
                adapter_path=adapter_path,
                train_layer_indices=train_layers,
                retrain_iters=args.retrain_iters,
                output_dir=retrain_dir,
                data_dir=ROOT / "data",
            )
            retrain_summary = {**retrain_meta, "strategy": args.retrain_strategy}

    ablation_payload = {
        "metadata": {
            "base_model": BASE_MODEL,
            "adapter_path": str(adapter_path),
            "num_prompts": len(samples),
            "baseline_top_layer_accuracy_mlx_first_token": float(baseline_accuracy),
            "pytorch_top_layer_mean_assistant_span": float(pytorch_top_layer_acc),
            "threshold_95pct_of_baseline": float(threshold),
            "minimum_layers_for_threshold": int(min_layers),
            "adapted_layers": adapted_layers,
        },
        "ablation_keep_top_importance_layers": ablation_rows,
        "retraining": retrain_summary,
    }
    dump_json(RESULTS_DIR / "ablation_results.json", ablation_payload)

    write_minimum_layers_markdown(
        RESULTS_DIR / "minimum_layers_analysis.md",
        baseline_accuracy,
        threshold,
        ablation_rows,
        min_layers,
        retrain_summary,
        pytorch_top_layer_acc,
    )

    dump_json(
        RESULTS_DIR / "layer_analysis.json",
        {
            "logit_lens_accuracy_base": base_acc,
            "logit_lens_accuracy_lora": lora_acc,
            "gain_lora_minus_base": gain,
            "lora_importance_l2": importance,
            "ablation": ablation_rows,
            "min_layers_for_95pct_baseline": min_layers,
            "correlation": {"lora_accuracy": corr_lora, "gain": corr_gain},
            "retraining": retrain_summary,
        },
    )

    logger.info("Done. Minimum layers (ablation) for 95%% of baseline: %s", min_layers)
    logger.info("Results: %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
