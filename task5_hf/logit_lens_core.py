"""
Logit lens (per-layer hidden → norm → lm_head), LoRA L₂ importance, ablation helpers.

Uses MLX-LM (same stack as project LoRA training) for reproducibility with Alpaca adapters.
"""
from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Tuple

import mlx.core as mx
import numpy as np
from mlx.utils import tree_unflatten
from mlx_lm.tuner.lora import LoRALinear
from safetensors import safe_open
from scipy import stats as scipy_stats

import mlx.nn as nn

logger = logging.getLogger(__name__)

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def get_test_samples(data_path: Path, num_prompts: int) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with data_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
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


def tokenize_prompt_and_target(
    tokenizer,
    messages: List[Dict[str, str]],
    tools=None,
) -> Tuple[List[int], List[int], int]:
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
    target_ids = full_ids[len(prompt_ids) :]
    return prompt_ids, full_ids, int(target_ids[0])


def token_text(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([token_id])
    except Exception:
        return str(token_id)


def mlx_forward_logits(model: Any, input_ids: mx.array) -> mx.array:
    """Causal LM logits ``(batch, seq, vocab)`` from MLX-LM (matches ``model(x)``)."""
    out = model(input_ids)
    if isinstance(out, tuple):
        return out[0]
    return out


def evaluate_first_token_accuracy_mlx(
    model: Any,
    tokenizer,
    samples: List[Dict[str, Any]],
) -> float:
    """
    Full forward on **prompt** token ids; top-1 at last prompt position vs first assistant token.
    Uses the same path as generation (not a manual layer loop).
    """
    correct = 0
    total = 0
    for row in samples:
        messages = row["messages"]
        tools = row.get("tools")
        prompt_ids, _full_ids, target_token_id = tokenize_prompt_and_target(tokenizer, messages, tools=tools)
        if len(prompt_ids) < 1:
            continue
        x = mx.array([prompt_ids], dtype=mx.int32)
        logits = mlx_forward_logits(model, x)
        pred_token_id = int(mx.argmax(logits[0, -1], axis=-1).item())
        correct += int(pred_token_id == target_token_id)
        total += 1
    return float(correct / total) if total else 0.0


def accuracy_by_layer(correct_counts: Dict[int, int], total_counts: Dict[int, int]) -> Dict[int, float]:
    return {
        layer_idx: (correct_counts[layer_idx] / total_counts[layer_idx]) if total_counts[layer_idx] else 0.0
        for layer_idx in sorted(total_counts)
    }


def layer_accuracy_gain(
    base_acc: Dict[int, float],
    lora_acc: Dict[int, float],
) -> Dict[int, float]:
    keys = sorted(set(base_acc) | set(lora_acc))
    return {k: float(lora_acc.get(k, 0.0) - base_acc.get(k, 0.0)) for k in keys}


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


def evaluate_ablation_accuracy(
    model,
    tokenizer,
    samples: List[Dict[str, Any]],
    active_layers: Iterable[int],
) -> float:
    """Keep LoRA only on ``active_layers``; zero scales elsewhere. First-assistant-token top-1 (full MLX forward)."""
    active = set(active_layers)
    all_adapted_layers = set()
    for idx, _layer in enumerate(model.model.layers):
        if any(hasattr(module, "lora_a") and hasattr(module, "lora_b") for _, module in model.model.layers[idx].named_modules()):
            all_adapted_layers.add(idx)

    inactive = sorted(all_adapted_layers - active)
    if not inactive:
        return evaluate_first_token_accuracy_mlx(model, tokenizer, samples)
    with temporarily_zero_lora_layers(model, inactive):
        return evaluate_first_token_accuracy_mlx(model, tokenizer, samples)


def compute_correlation(
    y_metric: np.ndarray,
    importance: np.ndarray,
) -> Dict[str, float]:
    if len(y_metric) < 3:
        return {}
    if np.std(y_metric) == 0.0 or np.std(importance) == 0.0:
        return {}
    pearson_r, pearson_p = scipy_stats.pearsonr(importance, y_metric)
    spearman_rho, spearman_p = scipy_stats.spearmanr(importance, y_metric)
    return {
        "pearson_r": float(pearson_r),
        "pearson_p_value": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p_value": float(spearman_p),
        "mean_y": float(np.mean(y_metric)),
        "mean_importance": float(np.mean(importance)),
    }


def correlation_for_layers(
    adapted_layers: List[int],
    importance_by_layer: Dict[int, float],
    metric_by_layer: Dict[int, float],
) -> Dict[str, Any]:
    """Correlate LoRA L₂ importance with a per-layer scalar (accuracy or gain)."""
    imps = np.array([importance_by_layer[i] for i in adapted_layers], dtype=np.float64)
    ys = np.array([metric_by_layer[i] for i in adapted_layers], dtype=np.float64)
    stats_dict = compute_correlation(ys, imps)
    return {"layers": adapted_layers, "metric_by_layer": {i: metric_by_layer[i] for i in adapted_layers}, **stats_dict}
