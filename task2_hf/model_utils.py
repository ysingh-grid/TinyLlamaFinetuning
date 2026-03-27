"""
Model and tokenizer loading for the TinyLlama activation steering demo.

Loads TinyLlama/TinyLlama-1.1B-Chat-v1.0 in float32 for CPU inference.
If a PEFT LoRA adapter is present at ./adapters/r16, it is merged into the
base model before returning.  The app degrades gracefully if peft is absent
or the adapter files are missing.
"""
import logging
import time
from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_DIR = Path(__file__).resolve().parent / "adapters" / "r16"
MAX_NEW_TOKENS = 100


# ── PEFT availability ─────────────────────────────────────────────────────────

def _peft_available() -> bool:
    try:
        import peft  # noqa: F401
        return True
    except ImportError:
        return False


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_and_tokenizer(device: str = "cpu") -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the base model and tokenizer.  Merges the LoRA adapter (r16) if
    available, falling back to the base model only.

    Parameters
    ----------
    device : str
        Target device ("cpu" for Hugging Face Spaces default).

    Returns
    -------
    (model, tokenizer)
    """
    logger.info("Loading tokenizer: %s", BASE_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading base model: %s  (device=%s, dtype=float32)", BASE_MODEL_ID, device)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=torch.float32,        # transformers ≥4.50 uses `dtype`; `torch_dtype` is deprecated
        low_cpu_mem_usage=True,
        attn_implementation="eager",  # required for TinyLlama RoPE compatibility on CPU
    )

    # Attempt to load the PEFT LoRA adapter (converted from MLX r16 checkpoint)
    if ADAPTER_DIR.exists() and (ADAPTER_DIR / "adapter_config.json").exists():
        if _peft_available():
            try:
                from peft import PeftModel
                logger.info("Applying PEFT LoRA adapter from %s", ADAPTER_DIR)
                model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
                model = model.merge_and_unload()
                logger.info("LoRA adapter merged — model is now fine-tuned TinyLlama r16")
            except Exception as exc:
                logger.warning(
                    "LoRA adapter load failed (%s) — falling back to base model", exc
                )
        else:
            logger.info("peft not installed — using base model only")
    else:
        logger.info("No adapter found at %s — using base model only", ADAPTER_DIR)

    model = model.to(device)
    model.eval()
    return model, tokenizer


# ── Prompt formatting ─────────────────────────────────────────────────────────

def format_prompt(tokenizer: AutoTokenizer, user_text: str) -> str:
    """
    Format user text using the TinyLlama chat template when available,
    otherwise fall back to a manually constructed template.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    # Manual TinyLlama chat template
    return (
        "<|system|>\n"
        "You are a helpful, respectful, and honest assistant.</s>\n"
        f"<|user|>\n{user_text}</s>\n"
        "<|assistant|>\n"
    )


# ── Generation ────────────────────────────────────────────────────────────────

def generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: str = "cpu",
    max_new_tokens: int = MAX_NEW_TOKENS,
    *,
    repetition_penalty: float = 1.1,
    no_repeat_ngram_size: int = 0,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> Tuple[str, float]:
    """
    Generate up to max_new_tokens new tokens.

    Parameters
    ----------
    model, tokenizer     : loaded objects
    prompt               : fully-formatted prompt string
    device               : target device
    max_new_tokens       : generation budget
    repetition_penalty   : >1 discourages repeating tokens.
    no_repeat_ngram_size : if >0, block exact n-gram repeats.
    do_sample            : True = temperature sampling; False = greedy.
    temperature          : sampling temperature (only used when do_sample=True).
    top_p                : nucleus sampling cutoff (only used when do_sample=True).

    Notes
    -----
    Steered decoding MUST use do_sample=True.  Greedy decode combined with a
    fixed-direction activation injection creates a positive-feedback loop:
    the hook biases next-token logits in one direction every step, greedy
    pick reinforces those tokens into context, the hook fires again in the
    same direction — causing the repetition collapse visible in output.
    Temperature sampling breaks this feedback loop.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[-1]

    gen_kw: dict = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kw["temperature"] = temperature
        gen_kw["top_p"] = top_p
    if no_repeat_ngram_size > 0:
        gen_kw["no_repeat_ngram_size"] = int(no_repeat_ngram_size)

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            **gen_kw,
        )
    elapsed = round(time.perf_counter() - t0, 2)

    new_tokens = output_ids[0, input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response, elapsed
