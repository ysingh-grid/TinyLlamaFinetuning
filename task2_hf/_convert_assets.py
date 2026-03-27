"""
One-time conversion script: run from project root to populate task2_hf/ assets.
  python task2_hf/_convert_assets.py

Produces:
  task2_hf/steering_vectors.pt          (unit-norm torch tensors, layers 8-20)
  task2_hf/adapters/r16/adapter_config.json
  task2_hf/adapters/r16/adapter_model.safetensors   (PEFT-format weights)
"""
import json
import pickle
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent  # project root
HF_DIR = ROOT / "task2_hf"

# ── 1. Steering vectors ───────────────────────────────────────────────────────

def convert_steering_vectors():
    pkl_path = ROOT / "results" / "task2" / "steering_vectors.pkl"
    if not pkl_path.exists():
        print(f"[SKIP] {pkl_path} not found — skipping steering vector conversion")
        return

    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    vecs_raw = raw.get("vectors", raw) if isinstance(raw, dict) and "vectors" in raw else raw

    result = {}
    import numpy as np
    for layer_idx, val in vecs_raw.items():
        if isinstance(val, dict):
            arr = np.asarray(val.get("unit_vector", val.get("raw_mean_difference")), dtype=np.float32)
        else:
            arr = np.asarray(val, dtype=np.float32)
        arr = arr.ravel()
        norm = np.linalg.norm(arr)
        if norm > 1e-9:
            arr = arr / norm
        result[int(layer_idx)] = torch.from_numpy(arr)

    out = HF_DIR / "steering_vectors.pt"
    torch.save(result, out)
    print(f"[OK] Saved {len(result)} steering vectors → {out.relative_to(ROOT)}")
    for idx, t in sorted(result.items()):
        print(f"     layer {idx}: shape={tuple(t.shape)}  norm={t.norm():.4f}")


# ── 2. MLX → PEFT adapter conversion ─────────────────────────────────────────
#
# MLX convention (from safetensors inspection):
#   lora_a: (in_features, rank)   e.g. (2048, 16)
#   lora_b: (rank, out_features)  e.g. (16, 2048)
#
# PEFT convention (state_dict for LlamaForCausalLM):
#   lora_A.weight: (rank, in_features)    ← lora_a.T
#   lora_B.weight: (out_features, rank)   ← lora_b.T
#
# PEFT key format  (PeftModel wraps base model under "base_model.model"):
#   base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.weight
#

def convert_adapter():
    src = ROOT / "results" / "task1" / "adapter_r16" / "adapters.safetensors"
    if not src.exists():
        print(f"[SKIP] {src} not found — skipping adapter conversion")
        return

    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError:
        print("[SKIP] safetensors not installed — skipping adapter conversion")
        return

    # Read MLX weights
    mlx_weights = {}
    with safe_open(src, framework="pt", device="cpu") as f:
        for key in f.keys():
            mlx_weights[key] = f.get_tensor(key)

    # Build PEFT state dict
    peft_state: dict[str, torch.Tensor] = {}
    for mlx_key, tensor in mlx_weights.items():
        # mlx_key:  model.layers.{i}.self_attn.{q_proj|v_proj}.{lora_a|lora_b}
        # peft_key: base_model.model.model.layers.{i}.self_attn.{q_proj|v_proj}.{lora_A|lora_B}.weight
        parts = mlx_key.split(".")
        # parts: ['model', 'layers', '<i>', 'self_attn', '<proj>', 'lora_a/b']
        layer_i = parts[2]
        proj    = parts[4]          # q_proj or v_proj
        ab      = parts[5]          # lora_a or lora_b

        peft_ab = "lora_A" if ab == "lora_a" else "lora_B"
        peft_key = f"base_model.model.model.layers.{layer_i}.self_attn.{proj}.{peft_ab}.weight"

        # Transpose: MLX stores (in, rank) and (rank, out); PEFT wants (rank, in) and (out, rank)
        peft_state[peft_key] = tensor.T.contiguous().to(torch.float32)

    out_dir = HF_DIR / "adapters" / "r16"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write adapter_model.safetensors
    save_file(peft_state, out_dir / "adapter_model.safetensors")
    print(f"[OK] Saved PEFT adapter ({len(peft_state)} tensors) → {(out_dir / 'adapter_model.safetensors').relative_to(ROOT)}")

    # Write adapter_config.json
    config = {
        "auto_mapping": None,
        "base_model_name_or_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": list(range(6, 22)),   # layers 6-21 (last 16 of 22)
        "lora_alpha": 32,                             # rank * scale = 16 * 2.0
        "lora_dropout": 0.05,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": 16,
        "revision": None,
        "target_modules": ["q_proj", "v_proj"],
        "task_type": "CAUSAL_LM",
    }
    with open(out_dir / "adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"[OK] Saved adapter_config.json → {(out_dir / 'adapter_config.json').relative_to(ROOT)}")


if __name__ == "__main__":
    import numpy as np  # noqa: F401 — imported early to surface missing dep
    convert_steering_vectors()
    convert_adapter()
    print("\nDone. task2_hf/ assets are ready.")
