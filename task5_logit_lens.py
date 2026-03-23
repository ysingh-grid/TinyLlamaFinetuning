#!/usr/bin/env python3
"""
Task 5: Logit Lens & Layer-wise LoRA Importance Analysis

Outputs:
- results/task5/layer_predictions.json
- results/task5/prediction_accuracy_by_layer.png
- results/task5/lora_importance_by_layer.json
- results/task5/importance_accuracy_correlation.png
- results/task5/ablation_results.json
- results/task5/minimum_layers_analysis.md
- results/task5/lens_and_importance.png

Also performs a selective bottom-k LoRA retraining run and stores it under:
- results/task5/retrained_adapter_bottom5/
"""
import argparse
import json
import logging
import math
import shutil
import statistics
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_unflatten
from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.datasets import CacheDataset, load_dataset
from mlx_lm.tuner.trainer import TrainingArgs, train
from safetensors import safe_open
from scipy import stats as scipy_stats

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_PLOT = True
except Exception:
    HAS_PLOT = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_ADAPTER = "adapters/tinyllama-lora-alpaca"
RESULTS_DIR = Path("./results/task5")
DATA_DIR = Path("./data")
DEFAULT_BOTTOM_K = 5
DEFAULT_RETRAIN_ITERS = 100
DEFAULT_NUM_PROMPTS = 100


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def get_test_samples(num_prompts: int) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with (DATA_DIR / "test.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            messages = row.get("messages", [])
            if not messages:
                continue
            samples.append(row)
            if len(samples) >= num_prompts:
                break
    return samples


def first_user_message(messages: List[Dict[str, str]]) -> str:
    for msg in messages:
        if msg.get("role") == "user":
            return msg.get("content", "")
    return messages[0].get("content", "") if messages else ""


def assistant_text(messages: List[Dict[str, str]]) -> str:
    for msg in messages:
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def tokenize_prompt_and_target(tokenizer, messages: List[Dict[str, str]], tools=None) -> Tuple[List[int], List[int], int]:
    prompt_messages = messages[:-1] if messages and messages[-1].get("role") == "assistant" else messages
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    full_ids = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
    )
    if not isinstance(prompt_ids, list):
        prompt_ids = list(prompt_ids)
    if not isinstance(full_ids, list):
        full_ids = list(full_ids)
    if len(full_ids) <= len(prompt_ids):
        raise ValueError("Could not derive assistant target token from chat template.")
    target_ids = full_ids[len(prompt_ids):]
    return prompt_ids, full_ids, int(target_ids[0])


def token_text(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([token_id])
    except Exception:
        return str(token_id)


def safe_layer_output(layer, hidden_states):
    out = layer(hidden_states, mask=None, cache=None)
    if isinstance(out, tuple):
        return out[0]
    return out


def evaluate_logit_lens(model, tokenizer, samples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[int, int], Dict[int, int]]:
    num_layers = len(model.model.layers)
    correct_counts = {i: 0 for i in range(num_layers)}
    total_counts = {i: 0 for i in range(num_layers)}
    sample_rows: List[Dict[str, Any]] = []

    for sample_idx, row in enumerate(samples):
        messages = row["messages"]
        tools = row.get("tools")
        prompt_ids, full_ids, target_token_id = tokenize_prompt_and_target(tokenizer, messages, tools=tools)
        if len(prompt_ids) < 1:
            continue

        x = mx.array([prompt_ids], dtype=mx.int32)
        hidden = model.model.embed_tokens(x)
        prompt_text = first_user_message(messages)
        target_token_text = token_text(tokenizer, target_token_id)

        layer_predictions: List[Dict[str, Any]] = []
        for layer_idx, layer in enumerate(model.model.layers):
            hidden = safe_layer_output(layer, hidden)
            logits = model.lm_head(model.model.norm(hidden))
            pred_token_id = int(mx.argmax(logits[0, -1], axis=-1).item())
            is_correct = pred_token_id == target_token_id
            total_counts[layer_idx] += 1
            correct_counts[layer_idx] += int(is_correct)

            layer_predictions.append(
                {
                    "layer": layer_idx,
                    "pred_token_id": pred_token_id,
                    "pred_token_text": token_text(tokenizer, pred_token_id),
                    "correct": bool(is_correct),
                }
            )

        sample_rows.append(
            {
                "prompt_index": sample_idx,
                "prompt_text": prompt_text,
                "assistant_reference": assistant_text(messages),
                "prompt_token_count": len(prompt_ids),
                "target_token_id": target_token_id,
                "target_token_text": target_token_text,
                "layer_predictions": layer_predictions,
            }
        )

    return sample_rows, correct_counts, total_counts


def accuracy_by_layer(correct_counts: Dict[int, int], total_counts: Dict[int, int]) -> Dict[int, float]:
    return {
        layer_idx: (correct_counts[layer_idx] / total_counts[layer_idx]) if total_counts[layer_idx] else 0.0
        for layer_idx in sorted(total_counts)
    }


def load_adapter_importance(adapter_path: Path, num_layers: int) -> Dict[int, float]:
    adapter_file = adapter_path / "adapters.safetensors"
    importance = {i: 0.0 for i in range(num_layers)}
    if not adapter_file.exists():
        raise FileNotFoundError(f"Missing adapter weights: {adapter_file}")

    layer_weights: Dict[int, List[np.ndarray]] = {}
    with safe_open(str(adapter_file), framework="numpy") as f:
        for key in f.keys():
            if "model.layers." not in key or "lora_" not in key:
                continue
            parts = key.split(".")
            try:
                layer_idx = int(parts[2])
            except Exception:
                continue
            tensor = np.asarray(f.get_tensor(key))
            layer_weights.setdefault(layer_idx, []).append(tensor)

    for layer_idx, tensors in layer_weights.items():
        importance[layer_idx] = float(sum(np.linalg.norm(t) for t in tensors))
    return importance


def discover_lora_keys(adapter_path: Path) -> List[str]:
    config_path = adapter_path / "adapter_config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as fh:
            config = json.load(fh)
        keys = config.get("lora_parameters", {}).get("keys")
        if keys:
            return list(keys)
    return ["self_attn.q_proj", "self_attn.v_proj"]


def to_lora(module, config: Dict[str, Any]):
    if isinstance(module, (nn.Linear, nn.QuantizedLinear)):
        return LoRALinear.from_base(
            module,
            r=config["rank"],
            scale=config["scale"],
            dropout=config["dropout"],
        )
    if hasattr(module, "to_lora"):
        return module.to_lora(
            r=config["rank"],
            scale=config["scale"],
            dropout=config["dropout"],
        )
    raise ValueError(f"Unsupported module type for LoRA conversion: {type(module).__name__}")


def convert_selected_layers_to_lora(model, selected_layer_indices: Iterable[int], lora_config: Dict[str, Any]) -> None:
    keys = set(lora_config.get("keys") or ["self_attn.q_proj", "self_attn.v_proj"])
    for layer_idx in selected_layer_indices:
        block = model.model.layers[layer_idx]
        lora_modules = [(name, to_lora(module, lora_config)) for name, module in block.named_modules() if name in keys]
        if lora_modules:
            block.update_modules(tree_unflatten(lora_modules))


def load_selected_adapter_weights(model, adapter_path: Path) -> None:
    model.load_weights(str(adapter_path / "adapters.safetensors"), strict=False)


@contextmanager
def temporarily_zero_lora_layers(model, layer_indices: Iterable[int]):
    saved: List[Tuple[Any, float]] = []
    for layer_idx in layer_indices:
        for _, module in model.model.layers[layer_idx].named_modules():
            if hasattr(module, "lora_a") and hasattr(module, "lora_b"):
                saved.append((module, float(getattr(module, "scale", 1.0))))
                module.scale = 0.0
    try:
        yield
    finally:
        for module, scale in saved:
            module.scale = scale


def evaluate_ablation_accuracy(model, tokenizer, samples: List[Dict[str, Any]], active_layers: Iterable[int]) -> float:
    active = set(active_layers)
    all_adapted_layers = set()
    for idx, layer in enumerate(model.model.layers):
        if any(hasattr(module, "lora_a") and hasattr(module, "lora_b") for _, module in layer.named_modules()):
            all_adapted_layers.add(idx)

    inactive = sorted(all_adapted_layers - active)
    if not inactive:
        _, correct_counts, total_counts = evaluate_logit_lens(model, tokenizer, samples)
    else:
        with temporarily_zero_lora_layers(model, inactive):
            _, correct_counts, total_counts = evaluate_logit_lens(model, tokenizer, samples)

    accs = accuracy_by_layer(correct_counts, total_counts)
    return float(accs[max(accs.keys())]) if accs else 0.0


def compute_correlation(acc_by_layer: Dict[int, float], importance_by_layer: Dict[int, float], adapted_layers: List[int]) -> Dict[str, float]:
    accs = np.array([acc_by_layer[i] for i in adapted_layers], dtype=np.float64)
    imps = np.array([importance_by_layer[i] for i in adapted_layers], dtype=np.float64)
    if len(adapted_layers) < 3:
        return {}
    if np.std(accs) == 0.0 or np.std(imps) == 0.0:
        return {}
    pearson_r, pearson_p = scipy_stats.pearsonr(imps, accs)
    spearman_rho, spearman_p = scipy_stats.spearmanr(imps, accs)
    return {
        "pearson_r": float(pearson_r),
        "pearson_p_value": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p_value": float(spearman_p),
        "mean_accuracy": float(np.mean(accs)),
        "mean_importance": float(np.mean(imps)),
    }


def plot_accuracy_and_importance(layer_ids: List[int], accs: List[float], imps: List[float], out_path: Path) -> None:
    if not HAS_PLOT:
        return
    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.plot(layer_ids, accs, marker="o", linewidth=2, color="tab:blue", label="Logit lens accuracy")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Next-token accuracy", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.bar(layer_ids, imps, alpha=0.25, color="tab:red", label="LoRA importance")
    ax2.set_ylabel("L2 importance", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.tight_layout()
    plt.title("Logit Lens Accuracy and LoRA Importance by Layer")
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy(layer_ids: List[int], accs: List[float], out_path: Path) -> None:
    if not HAS_PLOT:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layer_ids, accs, marker="o", color="tab:blue", linewidth=2)
    ax.set_xlabel("Layer depth")
    ax.set_ylabel("Top-1 next-token accuracy")
    ax.set_title("Prediction Accuracy vs Layer Depth")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_correlation(layer_ids: List[int], accs: List[float], imps: List[float], stats: Dict[str, float], out_path: Path) -> None:
    if not HAS_PLOT:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(imps, accs, s=70, color="tab:purple")
    if len(imps) >= 2:
        slope, intercept = np.polyfit(imps, accs, 1)
        xs = np.linspace(min(imps), max(imps), 100)
        ax.plot(xs, slope * xs + intercept, color="tab:gray", linewidth=2)
    ax.set_xlabel("LoRA importance (L2 norm)")
    ax.set_ylabel("Logit lens accuracy")
    ax.set_title("Importance vs Accuracy Gain")
    ax.grid(alpha=0.25)
    ax.text(
        0.02,
        0.98,
        f"Pearson r={stats.get('pearson_r', float('nan')):.3f}\nSpearman ρ={stats.get('spearman_rho', float('nan')):.3f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_layer_importance_plot(layer_ids: List[int], accs: List[float], imps: List[float], out_path: Path) -> None:
    if not HAS_PLOT:
        return
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(layer_ids, accs, marker="o", color="tab:blue", label="Accuracy")
    ax1.set_xlabel("Layer depth")
    ax1.set_ylabel("Top-1 next-token accuracy", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.bar(layer_ids, imps, alpha=0.25, color="tab:red", label="Importance")
    ax2.set_ylabel("LoRA L2 importance", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.tight_layout()
    plt.title("Logit Lens Accuracy vs LoRA Importance")
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_minimum_layers_markdown(path: Path, baseline_acc: float, threshold: float, ablation_rows: List[Dict[str, Any]], min_layers: int, retrain_summary: Dict[str, Any]) -> None:
    lines = [
        "# Minimum Layers Analysis",
        "",
        f"- Baseline top-layer accuracy: **{baseline_acc:.3f}**",
        f"- 95% threshold: **{threshold:.3f}**",
        f"- Minimum layers required: **{min_layers}**",
        "",
        "## Ablation sweep",
        "",
        "| Kept layers | Accuracy |",
        "|---:|---:|",
    ]
    for row in ablation_rows:
        lines.append(f"| {row['kept_layers']} | {row['accuracy']:.3f} |")
    lines.extend(
        [
            "",
            "## Retraining summary",
            "",
            f"- Retrained only the **bottom {retrain_summary.get('bottom_k', DEFAULT_BOTTOM_K)}** least-important LoRA layers.",
            f"- Retraining iterations: **{retrain_summary.get('iters', DEFAULT_RETRAIN_ITERS)}**",
            f"- Selected layers: `{retrain_summary.get('selected_layers', [])}`",
            f"- Retrained adapter path: `{retrain_summary.get('adapter_path', '')}`",
            "",
            "The minimum-layer threshold above is defined using the 100-prompt logit-lens top-1 accuracy curve and the 95% of baseline criterion.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def selective_retrain_bottom_k(
    adapter_path: Path,
    bottom_k_layers: List[int],
    retrain_iters: int,
    output_dir: Path,
    max_seq_length: int = 512,
) -> Dict[str, Any]:
    base_model, tokenizer = load(BASE_MODEL)

    config_path = adapter_path / "adapter_config.json"
    with config_path.open("r", encoding="utf-8") as fh:
        adapter_cfg = json.load(fh)
    lora_config = adapter_cfg.get("lora_parameters", {})
    lora_config["keys"] = discover_lora_keys(adapter_path)

    convert_selected_layers_to_lora(base_model, bottom_k_layers, lora_config)
    load_selected_adapter_weights(base_model, adapter_path)

    dataset_args = SimpleNamespace(
        data=str(DATA_DIR),
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
        "selected_layers": bottom_k_layers,
        "fine_tune_type": "lora",
        "num_layers": len(bottom_k_layers),
        "lora_parameters": lora_config,
    }
    dump_json(output_dir / "adapter_config.json", adapter_config_out)

    retrain_metadata = {
        "base_model": BASE_MODEL,
        "source_adapter": str(adapter_path),
        "selected_layers": bottom_k_layers,
        "bottom_k": len(bottom_k_layers),
        "iters": retrain_iters,
        "max_seq_length": max_seq_length,
        "adapter_path": str(output_dir),
        "lora_parameters": lora_config,
        "train_dataset_size": len(train_dataset),
        "valid_dataset_size": len(valid_dataset),
        "status": "completed",
    }
    dump_json(output_dir / "retrain_procedure.json", retrain_metadata)
    dump_json(output_dir / "retrain_metrics.json", retrain_metadata)
    return retrain_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 5: Logit Lens & LoRA importance analysis")
    parser.add_argument("--adapter", type=str, default=DEFAULT_ADAPTER, help="Path to the LoRA adapter directory")
    parser.add_argument("--num-prompts", type=int, default=DEFAULT_NUM_PROMPTS, help="Number of test prompts")
    parser.add_argument("--bottom-k", type=int, default=DEFAULT_BOTTOM_K, help="How many least-important layers to retrain")
    parser.add_argument("--retrain-iters", type=int, default=DEFAULT_RETRAIN_ITERS, help="Iterations for selective retraining")
    parser.add_argument("--smoke-test", action="store_true", help="Use 2 prompts and 10 retrain iters for a quick check")
    parser.add_argument("--skip-retrain", action="store_true", help="Skip the selective retraining run")
    args = parser.parse_args()

    if args.smoke_test:
        args.num_prompts = 2
        args.retrain_iters = min(args.retrain_iters, 10)

    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")

    ensure_clean_dir(RESULTS_DIR)
    retrain_dir = RESULTS_DIR / "retrained_adapter_bottom5"

    logging.info("Loading adapted model for logit lens analysis...")
    model, tokenizer = load(BASE_MODEL, adapter_path=str(adapter_path))
    num_layers = len(model.model.layers)

    samples = get_test_samples(args.num_prompts)
    logging.info("Evaluating %d test prompts across %d layers...", len(samples), num_layers)
    sample_rows, correct_counts, total_counts = evaluate_logit_lens(model, tokenizer, samples)
    accs_by_layer = accuracy_by_layer(correct_counts, total_counts)

    layer_predictions = {
        "metadata": {
            "base_model": BASE_MODEL,
            "adapter_path": str(adapter_path),
            "num_prompts": len(sample_rows),
            "num_layers": num_layers,
            "target_definition": "first assistant token after the prompt",
        },
        "samples": sample_rows,
    }
    dump_json(RESULTS_DIR / "layer_predictions.json", layer_predictions)

    importance = load_adapter_importance(adapter_path, num_layers)
    importance_payload = {
        "metadata": {
            "base_model": BASE_MODEL,
            "adapter_path": str(adapter_path),
            "num_layers": num_layers,
            "adapted_layers": [idx for idx, value in importance.items() if value > 0],
        },
        "importance_by_layer": importance,
    }
    dump_json(RESULTS_DIR / "lora_importance_by_layer.json", importance_payload)

    layer_ids = sorted(accs_by_layer.keys())
    acc_values = [accs_by_layer[i] for i in layer_ids]
    imp_values = [importance.get(i, 0.0) for i in layer_ids]

    plot_accuracy(layer_ids, acc_values, RESULTS_DIR / "prediction_accuracy_by_layer.png")
    build_layer_importance_plot(layer_ids, acc_values, imp_values, RESULTS_DIR / "lens_and_importance.png")

    adapted_layers = [idx for idx in layer_ids if importance.get(idx, 0.0) > 0]
    corr_stats = compute_correlation(accs_by_layer, importance, adapted_layers) if adapted_layers else {}
    if corr_stats:
        plot_correlation(adapted_layers, [accs_by_layer[i] for i in adapted_layers], [importance[i] for i in adapted_layers], corr_stats, RESULTS_DIR / "importance_accuracy_correlation.png")
    else:
        plot_correlation(layer_ids, acc_values, imp_values, corr_stats, RESULTS_DIR / "importance_accuracy_correlation.png")

    if layer_ids:
        baseline_accuracy = accs_by_layer[layer_ids[-1]]
    else:
        baseline_accuracy = 0.0
    threshold = baseline_accuracy * 0.95

    # Ablation experiment: keep top-m most important adapted layers, zero the rest.
    sorted_adapted = sorted(adapted_layers, key=lambda idx: importance[idx], reverse=True)
    ablation_rows: List[Dict[str, Any]] = []
    keep_candidates = sorted(set([1, 2, 4, 6, 8, 10, 12, len(sorted_adapted)]))
    for keep_count in keep_candidates:
        keep_count = min(keep_count, len(sorted_adapted))
        kept = sorted_adapted[:keep_count]
        active_accuracy = evaluate_ablation_accuracy(model, tokenizer, samples, kept)
        ablation_rows.append(
            {
                "kept_layers": keep_count,
                "kept_layer_indices": kept,
                "accuracy": float(active_accuracy),
                "baseline_accuracy": float(baseline_accuracy),
                "accuracy_ratio": float(active_accuracy / baseline_accuracy) if baseline_accuracy else 0.0,
            }
        )

    min_layers = next((row["kept_layers"] for row in ablation_rows if row["accuracy"] >= threshold), len(sorted_adapted))

    retrain_summary: Dict[str, Any] = {
        "status": "skipped",
        "bottom_k": args.bottom_k,
        "iters": args.retrain_iters,
        "selected_layers": [],
        "adapter_path": str(retrain_dir),
    }

    if not args.skip_retrain:
        bottom_k = min(args.bottom_k, len(sorted_adapted))
        bottom_layers = sorted(sorted_adapted, key=lambda idx: importance[idx])[:bottom_k] if bottom_k else []
        logging.info("Selective retraining on bottom-%d layers: %s", bottom_k, bottom_layers)
        retrain_summary = selective_retrain_bottom_k(
            adapter_path=adapter_path,
            bottom_k_layers=bottom_layers,
            retrain_iters=args.retrain_iters,
            output_dir=retrain_dir,
            max_seq_length=512,
        )

    ablation_payload = {
        "metadata": {
            "base_model": BASE_MODEL,
            "adapter_path": str(adapter_path),
            "num_prompts": len(samples),
            "baseline_accuracy": float(baseline_accuracy),
            "threshold_95pct": float(threshold),
            "minimum_layers_needed": int(min_layers),
            "adapted_layers": adapted_layers,
        },
        "ablation_results": ablation_rows,
        "retraining": retrain_summary,
    }
    dump_json(RESULTS_DIR / "ablation_results.json", ablation_payload)

    minimum_layers_md = RESULTS_DIR / "minimum_layers_analysis.md"
    write_minimum_layers_markdown(minimum_layers_md, baseline_accuracy, threshold, ablation_rows, min_layers, retrain_summary)

    # Compatibility artifacts for earlier documentation.
    dump_json(
        RESULTS_DIR / "layer_analysis.json",
        {
            "logit_lens_accuracy": accs_by_layer,
            "lora_importance_l2": importance,
            "ablation_results": ablation_rows,
            "min_layers_for_95pct": min_layers,
            "baseline_accuracy": baseline_accuracy,
            "threshold_95pct": threshold,
            "retraining": retrain_summary,
        },
    )

    dump_json(
        RESULTS_DIR / "enhanced_analysis.json",
        {
            "lora_importance": importance,
            "bottom_k_layers": retrain_summary.get("selected_layers", []),
            "retrain_config": retrain_summary,
            "note": "Full logit-lens + importance analysis complete",
        },
    )

    logging.info("Task 5 complete.")
    logging.info("Minimum layers needed for 95%% performance: %d", min_layers)
    logging.info("Results written to %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
