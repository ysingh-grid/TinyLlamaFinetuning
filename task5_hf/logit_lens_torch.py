"""
PyTorch logit lens: single forward with ``output_hidden_states``, per-layer norm + lm_head.

Matches TinyLlama + merged PEFT adapters used elsewhere in this repo.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def load_tokenizer() -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_causal(device: str) -> AutoModelForCausalLM:
    m = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    return m.to(device).eval()


def load_lora_merged(adapter_dir: Path, device: str) -> AutoModelForCausalLM:
    """
    Load MLX-LM ``adapters.safetensors`` (lora_a/lora_b layout) into a merged HF causal LM.
    Standard PEFT ``adapter_config.json`` is not required; we read ``adapter_config.json`` for
    ``lora_parameters`` (rank, keys, layers) when present.
    """
    from peft import LoraConfig, TaskType, get_peft_model

    adapter_dir = adapter_dir.resolve()
    st_path = adapter_dir / "adapters.safetensors"
    if not st_path.exists():
        raise FileNotFoundError(f"Missing {st_path}")

    cfg_path = adapter_dir / "adapter_config.json"
    rank, lora_alpha, dropout = 16, 32, 0.05
    layers_to_transform = list(range(6, 22))
    if cfg_path.exists():
        import json

        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        lp = raw.get("lora_parameters") or {}
        rank = int(lp.get("rank", rank))
        # MLX ``scale`` 2.0 with rank 16 → alpha 32 (typical)
        sc = float(lp.get("scale", 2.0))
        lora_alpha = int(lp.get("lora_alpha", rank * sc))
        dropout = float(lp.get("dropout", dropout))
        # Infer layer range from checkpoint if possible
        from safetensors import safe_open

        with safe_open(str(st_path), framework="numpy") as f:
            li = set()
            for k in f.keys():
                if k.startswith("model.layers.") and "lora_" in k:
                    li.add(int(k.split(".")[2]))
            if li:
                layers_to_transform = sorted(li)

    base = load_base_causal(device)
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "v_proj"],
        layers_to_transform=layers_to_transform,
        bias="none",
    )
    model = get_peft_model(base, peft_cfg)
    sd = model.state_dict()
    mlx_to_peft = {}
    import numpy as np
    from safetensors import safe_open

    with safe_open(str(st_path), framework="numpy") as f:
        for key in f.keys():
            if "lora_a" not in key and "lora_b" not in key:
                continue
            # model.layers.6.self_attn.q_proj.lora_a
            parts = key.split(".")
            layer_i = parts[2]
            proj = parts[4]  # q_proj or v_proj
            comp = parts[5]  # lora_a or lora_b
            mlx_t = np.asarray(f.get_tensor(key))
            peft_prefix = f"base_model.model.model.layers.{layer_i}.self_attn.{proj}"
            if comp == "lora_a":
                # MLX: (in_features, r) → PEFT lora_A: (r, in_features)
                w = torch.from_numpy(mlx_t.T).float()
                mlx_to_peft[f"{peft_prefix}.lora_A.default.weight"] = w
            else:
                # MLX: (r, out_features) → PEFT lora_B: (out_features, r)
                w = torch.from_numpy(mlx_t.T).float()
                mlx_to_peft[f"{peft_prefix}.lora_B.default.weight"] = w

    missing = [k for k in mlx_to_peft if k not in sd]
    if missing:
        raise RuntimeError(f"PEFT state_dict keys mismatch (example missing): {missing[:3]}")
    for k, v in mlx_to_peft.items():
        sd[k].copy_(v.to(sd[k].device))
    model.load_state_dict(sd, strict=True)
    merged = model.merge_and_unload()
    return merged.to(device).eval()


def tokenize_prompt_and_target(
    tokenizer: AutoTokenizer,
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
    prompt_ids = list(prompt_ids)
    full_ids = list(full_ids)
    if len(full_ids) <= len(prompt_ids):
        raise ValueError("Could not derive assistant target token from chat template.")
    target_ids = full_ids[len(prompt_ids) :]
    return prompt_ids, full_ids, int(target_ids[0])


def token_text(tokenizer: AutoTokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([token_id])
    except Exception:
        return str(token_id)


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


@torch.inference_mode()
def evaluate_logit_lens_pytorch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    samples: List[Dict[str, Any]],
    device: str,
    max_assistant_positions: int = 32,
) -> Tuple[List[Dict[str, Any]], Dict[int, int], Dict[int, int]]:
    """
    Teacher-forced **assistant** positions: ``full_ids[:-1]`` forward, at each
    position ``pos >= len(prompt_ids)-1`` predict ``full_ids[pos+1]``. Averaged
    per layer (more stable than a single first-token probe).

    Still records the **first** assistant token row in ``layer_predictions`` for debugging.
    """
    n_layers = len(model.model.layers)
    correct_counts = {i: 0 for i in range(n_layers)}
    total_counts = {i: 0 for i in range(n_layers)}
    sample_rows: List[Dict[str, Any]] = []

    for sample_idx, row in enumerate(samples):
        messages = row["messages"]
        tools = row.get("tools")
        prompt_ids, full_ids, target_token_id = tokenize_prompt_and_target(tokenizer, messages, tools=tools)
        if len(prompt_ids) < 1 or len(full_ids) < 2:
            continue

        seq = full_ids[:-1]
        if len(seq) < 1:
            continue
        input_ids = torch.tensor([seq], device=device, dtype=torch.long)
        out = model(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        hs = out.hidden_states
        assert hs is not None
        full_logits = out.logits[0]  # teacher-forced next-token logits (last layer path)
        plen = len(prompt_ids)
        start = max(0, plen - 1)
        end = min(len(seq), start + max_assistant_positions)

        target_token_text = token_text(tokenizer, target_token_id)
        layer_predictions: List[Dict[str, Any]] = []

        for layer_idx in range(n_layers):
            correct_here = 0
            total_here = 0
            first_pred = None
            for pos in range(start, end):
                tgt = full_ids[pos + 1]
                if layer_idx == n_layers - 1:
                    logits = full_logits[pos].float()
                else:
                    h = hs[layer_idx + 1][0, pos, :].float()
                    h2 = model.model.norm(h.unsqueeze(0))
                    logits = model.lm_head(h2)[0, 0]
                pred_token_id = int(logits.argmax(dim=-1).item())
                if first_pred is None:
                    first_pred = (pred_token_id, token_text(tokenizer, pred_token_id))
                correct_here += int(pred_token_id == tgt)
                total_here += 1
            total_counts[layer_idx] += total_here
            correct_counts[layer_idx] += correct_here
            is_first = first_pred is not None and first_pred[0] == target_token_id
            layer_predictions.append(
                {
                    "layer": layer_idx,
                    "pred_token_id": first_pred[0] if first_pred else -1,
                    "pred_token_text": first_pred[1] if first_pred else "",
                    "correct_first_token": bool(is_first),
                    "assistant_token_accuracy": correct_here / total_here if total_here else 0.0,
                }
            )

        sample_rows.append(
            {
                "prompt_index": sample_idx,
                "prompt_text": first_user_message(messages),
                "assistant_reference": assistant_text(messages),
                "prompt_token_count": len(prompt_ids),
                "target_token_id": target_token_id,
                "target_token_text": target_token_text,
                "layer_predictions": layer_predictions,
                "metric": f"mean top-1 over first {end - start} assistant positions (teacher-forced)",
            }
        )

    return sample_rows, correct_counts, total_counts


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

