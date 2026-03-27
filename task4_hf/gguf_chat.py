"""
Load quantized GGUF from the Hub and run chat-style generation via llama-cpp-python.

Used by the Gradio Space so users can pick Q4_K_M / Q5_K_M / Q8_0 without llama-server.
"""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

DEFAULT_REPO = os.environ.get("TASK4_GGUF_REPO", "ysingh-aiml/tinyllama-alpaca-lora-gguf")

# Display label -> Hub filename
QUANT_CHOICES: Dict[str, str] = {
    "Q4_K_M (~637 MB)": "model-Q4_K_M.gguf",
    "Q5_K_M (~746 MB)": "model-Q5_K_M.gguf",
    "Q8_0 (~1.1 GB)": "model-Q8_0.gguf",
}

_llm = None
_llm_key: Optional[str] = None


def _deps_ok() -> Tuple[bool, str]:
    """Interactive Play tab needs llama-cpp-python (often omitted on HF to avoid build timeouts)."""
    try:
        import llama_cpp  # noqa: F401
    except ImportError as e:
        return False, str(e)
    return True, ""


def _download(filename: str, repo_id: str) -> Path:
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_files_only=False,
    )
    return Path(p)


def get_llm(quant_label: str, repo_id: str = DEFAULT_REPO):
    """
    Return (Llama instance, error_message). Only one model kept in memory at a time.
    """
    global _llm, _llm_key

    ok, err = _deps_ok()
    if not ok:
        return None, f"Missing dependency: {err}\nInstall `llama-cpp-python` and `huggingface_hub`."

    filename = QUANT_CHOICES.get(quant_label)
    if not filename:
        return None, f"Unknown quant: {quant_label}"

    cache_key = f"{repo_id}:{filename}"
    if _llm is not None and _llm_key == cache_key:
        return _llm, ""

    if _llm is not None:
        del _llm
        _llm = None
        _llm_key = None
        gc.collect()

    try:
        path = _download(filename, repo_id)
    except Exception as e:
        return None, f"Download failed: {e}"

    try:
        from llama_cpp import Llama

        threads = int(os.environ.get("TASK4_LLAMA_THREADS", "4"))
        n_ctx = int(os.environ.get("TASK4_LLAMA_CTX", "2048"))

        t0 = time.perf_counter()
        _llm = Llama(
            model_path=str(path),
            n_ctx=n_ctx,
            n_threads=threads,
            verbose=False,
        )
        _llm_key = cache_key
        load_s = round(time.perf_counter() - t0, 2)
        return _llm, f"loaded in {load_s}s"
    except Exception as e:
        _llm = None
        _llm_key = None
        return None, f"llama-cpp load error: {e}"


def generate_chat(
    quant_label: str,
    user_message: str,
    max_tokens: int = 128,
    repo_id: str = DEFAULT_REPO,
) -> Tuple[str, str]:
    """
    Returns (assistant_text, status_line).
    """
    user_message = (user_message or "").strip()
    if not user_message:
        return "", "Enter a message."

    llm, meta = get_llm(quant_label, repo_id=repo_id)
    if llm is None:
        return "", meta

    status_parts = [meta] if meta else []
    t0 = time.perf_counter()
    try:
        out = llm.create_chat_completion(
            messages=[{"role": "user", "content": user_message}],
            max_tokens=int(max_tokens),
            temperature=0.2,
        )
        text = out["choices"][0]["message"]["content"].strip()
    except Exception:
        # GGUF may lack embedded chat template — TinyLlama chat format fallback
        prompt = (
            "<|system|>\nYou are a helpful assistant.</s>\n"
            f"<|user|>\n{user_message}</s>\n<|assistant|>\n"
        )
        out2 = llm.create_completion(
            prompt=prompt,
            max_tokens=int(max_tokens),
            temperature=0.2,
            stop=["</s>", "<|user|>"],
        )
        text = (out2.get("choices") or [{}])[0].get("text", "").strip()

    elapsed = round(time.perf_counter() - t0, 2)

    status_parts.append(f"{elapsed}s · max_tokens={max_tokens}")
    return text, " · ".join(status_parts)
