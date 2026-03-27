"""
Transformers + PEFT inference for TinyLlama rank ablation (CPU / CUDA on HF Spaces).

MLX-exported adapters use `adapters.safetensors` with `lora_a`/`lora_b` naming and a
transposed layout vs Hugging Face PEFT. We convert once per rank into a temp dir
(typically under /tmp) and load with PeftModel, then merge for generation.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
ADAPTERS = ROOT / "adapters"
BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LAYERS_TO_TRANSFORM = list(range(6, 22))

_peft_dirs: Dict[int, Path] = {}
_model = None
_model_rank: Optional[int] = None
_tokenizer = None


def _torch_deps_available() -> bool:
    try:
        import peft  # noqa: F401
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM  # noqa: F401
    except ImportError:
        return False
    return True


def _device_and_dtype():
    import torch

    if torch.cuda.is_available():
        return "cuda", torch.float16
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def _mlx_safetensors_path(rank: int) -> Path:
    p = ADAPTERS / f"r{rank}" / "adapters.safetensors"
    if not p.is_file():
        raise FileNotFoundError(f"Missing adapter weights: {p}")
    return p


def _ensure_peft_adapter_dir(rank: int) -> Path:
    if rank in _peft_dirs:
        return _peft_dirs[rank]

    from safetensors.torch import load_file, save_file

    src = _mlx_safetensors_path(rank)
    mlx_state = load_file(str(src))
    peft_state = {}
    for k, v in mlx_state.items():
        if ".lora_a" in k:
            key = k.replace(".lora_a", ".lora_A")
            tensor = v.t().contiguous()
        elif ".lora_b" in k:
            key = k.replace(".lora_b", ".lora_B")
            tensor = v.t().contiguous()
        else:
            continue
        peft_state["base_model.model." + key + ".weight"] = tensor

    cfg = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": BASE_MODEL_ID,
        "bias": "none",
        "r": rank,
        "lora_alpha": rank * 2,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
        "layers_to_transform": LAYERS_TO_TRANSFORM,
    }

    tmp = Path(tempfile.mkdtemp(prefix=f"task1_peft_r{rank}_"))
    (tmp / "adapter_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    save_file(peft_state, str(tmp / "adapter_model.safetensors"))
    _peft_dirs[rank] = tmp
    logger.info("Prepared PEFT adapter for rank %s at %s", rank, tmp)
    return tmp


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
    return _tokenizer


def _load_merged_for_rank(rank: int):
    global _model, _model_rank
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    rank = int(rank)
    if _model_rank == rank and _model is not None:
        return _model

    if _model is not None:
        del _model
        _model = None
        _model_rank = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    device, dtype = _device_and_dtype()
    adapter_dir = _ensure_peft_adapter_dir(rank)

    logger.info("Loading TinyLlama + LoRA r=%s on %s (%s)", rank, device, dtype)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
    model = peft_model.merge_and_unload()
    model = model.to(device)
    model.eval()

    _model = model
    _model_rank = rank
    return model


def _format_prompt(tokenizer, user_text: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return (
        "<|system|>\nYou are a helpful assistant.</s>\n"
        f"<|user|>\n{user_text}</s>\n<|assistant|>\n"
    )


def generate_response_torch(
    prompt: str,
    rank: int,
    max_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Tuple[str, float, int]:
    """
    Returns (text, tokens_per_second, new_token_count).
    """
    import torch

    if not _torch_deps_available():
        raise ImportError("Install torch, transformers, peft, safetensors for non-MLX inference.")

    tok = _get_tokenizer()
    model = _load_merged_for_rank(rank)
    device = next(model.parameters()).device
    full_prompt = _format_prompt(tok, prompt)
    inputs = tok(full_prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[-1]

    gen_kw = dict(
        max_new_tokens=int(max_tokens),
        pad_token_id=tok.eos_token_id,
        repetition_penalty=1.1,
    )
    temp = float(temperature)
    if temp <= 0:
        gen_kw["do_sample"] = False
    else:
        gen_kw["do_sample"] = True
        gen_kw["temperature"] = temp
        gen_kw["top_p"] = float(top_p)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kw)
    elapsed = time.perf_counter() - t0

    new_tokens = out[0, input_len:]
    n_new = int(new_tokens.numel())
    text = tok.decode(new_tokens, skip_special_tokens=True).strip()
    tps = round(n_new / elapsed, 2) if elapsed > 0 else 0.0
    return text, tps, n_new
