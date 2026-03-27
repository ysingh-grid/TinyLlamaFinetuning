"""
TinyLlama + LoRA: attention extraction, rollout, head clustering features, base vs LoRA comparison.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def _peft_ok() -> bool:
    try:
        import peft  # noqa: F401
        return True
    except ImportError:
        return False


def load_tokenizer() -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(device: str = "cpu") -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    return model.to(device).eval()


def load_lora_merged(adapter_dir: Path, device: str = "cpu") -> AutoModelForCausalLM:
    """Load base + LoRA merged into a single causal LM."""
    model = load_base_model(device)
    if not adapter_dir.exists() or not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"LoRA adapter not found at {adapter_dir}")
    if not _peft_ok():
        raise RuntimeError("peft required for LoRA")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    model = model.merge_and_unload()
    return model.to(device).eval()


def format_chat_prompt(tokenizer: AutoTokenizer, user_text: str) -> str:
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


def generate_short(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_ids: torch.Tensor,
    device: str,
    max_new_tokens: int = 64,
) -> torch.Tensor:
    """Greedy decode; returns full token ids [1, total_len]."""
    prompt_ids = prompt_ids.to(device)
    with torch.no_grad():
        out = model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.15,
        )
    return out


def attention_rollout(
    attentions: Sequence[torch.Tensor],
    discard_ratio: float = 0.0,
) -> np.ndarray:
    """
    Classic attention rollout (Abnar & Zuidema). attentions: list of [B,H,S,S] per layer.
    Returns [S, S] numpy array (batch 0).
    """
    device = attentions[0].device
    batch, heads, seq, _ = attentions[0].shape
    # Average heads per layer, add identity for residual
    eye = torch.eye(seq, device=device, dtype=attentions[0].dtype)
    rollout = eye.clone()
    for attn in attentions:
        a = attn[0].mean(dim=0)  # [S, S]
        if discard_ratio > 0:
            flat = a.view(-1)
            k = int(flat.numel() * discard_ratio)
            if k > 0:
                _, idx = torch.topk(flat, k, largest=False)
                flat = flat.clone()
                flat[idx] = 0
                a = flat.view(seq, seq)
        a = a + eye
        a = a / a.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        rollout = torch.matmul(a, rollout)
    return rollout.cpu().numpy()


def head_feature_vector(
    attn_layer_head: torch.Tensor,
    target_size: int = 32,
) -> np.ndarray:
    """
    attn_layer_head: [S, S] single head. Resize to target_size x target_size and flatten.
    """
    a = attn_layer_head.float().unsqueeze(0).unsqueeze(0)  # [1,1,S,S]
    s = attn_layer_head.shape[0]
    if s < 2:
        return np.zeros(target_size * target_size, dtype=np.float32)
    a = F.interpolate(a, size=(target_size, target_size), mode="bilinear", align_corners=False)
    v = a.view(-1).cpu().numpy().astype(np.float32)
    n = np.linalg.norm(v)
    if n > 1e-8:
        v = v / n
    return v


def instruction_completion_scores(
    attn_head: torch.Tensor,
    prompt_len: int,
    total_len: int,
) -> Tuple[float, float]:
    """
    attn_head [S,S]. Completion region: rows prompt_len..total_len-1.
    Instruction columns: 0..prompt_len-1, completion columns: prompt_len..total_len-1.
    Returns (score_to_instruction, score_to_completion_self) averaged over completion rows.
    """
    if total_len <= prompt_len or prompt_len < 1:
        return 0.0, 0.0
    rows = slice(prompt_len, total_len)
    to_instr = attn_head[rows, :prompt_len].mean().item()
    to_comp = attn_head[rows, prompt_len:total_len].mean().item()
    return float(to_instr), float(to_comp)


@dataclass
class RunResult:
    attentions: Tuple[torch.Tensor, ...]
    prompt_len: int
    total_len: int
    input_ids: torch.Tensor  # CPU


def collect_one_example(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    instruction: str,
    device: str,
    max_new_tokens: int = 64,
) -> RunResult:
    prompt_text = format_chat_prompt(tokenizer, instruction)
    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    prompt_ids = enc["input_ids"].to(device)
    plen = prompt_ids.shape[1]

    full_ids = generate_short(model, tokenizer, prompt_ids, device, max_new_tokens=max_new_tokens)
    full_ids = full_ids.cpu()

    # Forward on full sequence with attentions
    batch = full_ids.to(device)
    mask = torch.ones_like(batch)
    with torch.no_grad():
        out = model(
            input_ids=batch,
            attention_mask=mask,
            output_attentions=True,
            return_dict=True,
        )
    att = out.attentions
    assert att is not None
    total_len = full_ids.shape[1]
    return RunResult(
        attentions=att,
        prompt_len=plen,
        total_len=total_len,
        input_ids=full_ids,
    )


def run_pair_base_lora(
    model_base: AutoModelForCausalLM,
    model_lora: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    instruction: str,
    device: str,
    max_new_tokens: int = 64,
) -> Tuple[RunResult, RunResult]:
    """
    Generate with LoRA, then forward **both** models on the same token sequence
    so base vs LoRA attention is comparable token-aligned.
    """
    prompt_text = format_chat_prompt(tokenizer, instruction)
    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    prompt_ids = enc["input_ids"].to(device)
    plen = prompt_ids.shape[1]

    full_ids = generate_short(model_lora, tokenizer, prompt_ids, device, max_new_tokens=max_new_tokens)
    full_ids = full_ids.cpu()
    batch = full_ids.to(device)
    mask = torch.ones_like(batch)

    def _forward(m: AutoModelForCausalLM) -> Tuple[torch.Tensor, ...]:
        with torch.no_grad():
            out = m(
                input_ids=batch,
                attention_mask=mask,
                output_attentions=True,
                return_dict=True,
            )
        assert out.attentions is not None
        return out.attentions

    att_b = _forward(model_base)
    att_l = _forward(model_lora)
    total_len = full_ids.shape[1]
    rb = RunResult(att_b, plen, total_len, full_ids.clone())
    rl = RunResult(att_l, plen, total_len, full_ids.clone())
    return rb, rl


def attentions_to_numpy_layers(
    attentions: Tuple[torch.Tensor, ...],
) -> List[np.ndarray]:
    """Each layer [B,H,S,S] -> [H,S,S] batch 0."""
    return [a[0].cpu().float().numpy() for a in attentions]


def cosine_similarity_matrix(
    features: np.ndarray,
) -> np.ndarray:
    """features [N, D]. Returns [N, N] cosine similarity."""
    f = np.nan_to_num(features.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    n = np.linalg.norm(f, axis=1, keepdims=True)
    n = np.maximum(n, 1e-8)
    f = f / n
    return np.clip(f @ f.T, -1.0, 1.0)


def base_vs_lora_head_similarity(
    attn_base: List[np.ndarray],
    attn_lora: List[np.ndarray],
) -> float:
    """
    Mean cosine similarity across all layers/heads between flattened averaged attention.
    attn_* : list of L arrays [H, S, S]
    """
    sims = []
    for b, l in zip(attn_base, attn_lora):
        h = b.shape[0]
        for hi in range(h):
            vb = b[hi].reshape(-1).astype(np.float64)
            vl = l[hi].reshape(-1)
            nb = np.linalg.norm(vb)
            nl = np.linalg.norm(vl)
            if nb < 1e-8 or nl < 1e-8:
                continue
            sims.append(float(np.dot(vb, vl) / (nb * nl)))
    return float(np.mean(sims)) if sims else 0.0


def load_instructions(path: Path) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [str(x) for x in data]
    raise ValueError("instructions file must be a JSON list")
