#!/usr/bin/env python3
"""
streamlit_app.py

Streamlit UI for TinyLlama instruction tuning:
- Page 1: Inference / Compare (two models side by side)
- Page 2: Data Collection (prompt → editable completion → JSONL)
- Page 3: Dataset Stats / EDA (length stats + histogram + top/bottom examples)

Usage:
    .venv/bin/python -m streamlit run streamlit_app.py
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

# TokenizersBackend shim for quantised models whose configs reference this
# legacy tokenizer class. Mirrors the logic used in `evaluate_perplexity.py`.
try:
    import transformers  # type: ignore

    if not hasattr(transformers, "TokenizersBackend"):
        from transformers import PreTrainedTokenizerFast  # type: ignore

        class TokenizersBackend(PreTrainedTokenizerFast):  # type: ignore
            pass

        transformers.TokenizersBackend = TokenizersBackend  # type: ignore
except Exception:
    # If anything goes wrong here we fall back to the default transformers
    # behaviour; the worst case is the original error, but on supported
    # versions this block is a no-op.
    pass

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional
    plt = None  # type: ignore

try:
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    MLX_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    MLX_AVAILABLE = False
    mx = None  # type: ignore


ROOT = Path(__file__).parent
MODELS_CONFIG_PATH = ROOT / "evaluation" / "models.json"
EVAL_PROMPTS_PATH = ROOT / "evaluation" / "eval_prompts.jsonl"
COLLECTED_PATH = ROOT / "data" / "collected.jsonl"
TRAIN_PATH = ROOT / "data" / "train.jsonl"
EDA_REPORT_PATH = ROOT / "data" / "eda_report.md"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model: str
    adapter_path: Optional[str] = None
    system_prompt: str = ""
    min_tokens: int = 0


@st.cache_resource(show_spinner=False)
def load_model_specs() -> Dict[str, ModelSpec]:
    if not MODELS_CONFIG_PATH.exists():
        raise FileNotFoundError(f"models.json not found at {MODELS_CONFIG_PATH}")
    raw = json.loads(MODELS_CONFIG_PATH.read_text())
    specs: Dict[str, ModelSpec] = {}
    for item in raw:
        specs[item["name"]] = ModelSpec(
            name=item["name"],
            model=item["model"],
            adapter_path=item.get("adapter_path"),
            system_prompt=item.get("system_prompt", "") or "",
            min_tokens=int(item.get("min_tokens", 0) or 0),
        )
    return specs


MODEL_SPECS = load_model_specs()
MODEL_IDS = list(MODEL_SPECS.keys())

_mlx_lock = threading.Lock()
_cache_lock = threading.Lock()

# model_id -> (model, tokenizer, last_used_ts)
_model_cache: Dict[str, Tuple[Any, Any, float]] = {}
_MAX_CACHE = 2


def _clear_mlx_cache() -> None:
    if not MLX_AVAILABLE:
        return
    try:
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal"):
            mx.metal.clear_cache()
    except Exception:
        pass


def get_model(model_id: str) -> Tuple[Any, Any]:
    if not MLX_AVAILABLE:
        raise RuntimeError(
            "MLX / mlx_lm not available. Install them in your virtualenv to use this app."
        )
    spec = MODEL_SPECS[model_id]
    now = time.time()
    with _cache_lock:
        if model_id in _model_cache:
            model, tokenizer, _ = _model_cache[model_id]
            _model_cache[model_id] = (model, tokenizer, now)
            return model, tokenizer

        if len(_model_cache) >= _MAX_CACHE:
            lru_id = min(_model_cache.items(), key=lambda kv: kv[1][2])[0]
            _model_cache.pop(lru_id, None)
            _clear_mlx_cache()

        kwargs: Dict[str, Any] = {}
        if spec.adapter_path:
            kwargs["adapter_path"] = spec.adapter_path
        model, tokenizer = load(spec.model, **kwargs)
        _model_cache[model_id] = (model, tokenizer, now)
        return model, tokenizer


def _truncate_ngram(text: str, n: int = 4) -> str:
    words = text.split()
    if len(words) < n * 2:
        return text
    seen: set = set()
    for i in range(len(words) - n + 1):
        ng = tuple(words[i : i + n])
        if ng in seen:
            truncated = " ".join(words[:i]).rstrip(" ,.;:")
            return truncated if truncated else text
        seen.add(ng)
    return text


def _make_suppress_eos(min_tokens: int, eos_ids: List[int]):
    if min_tokens <= 0 or not eos_ids or not MLX_AVAILABLE:
        return None
    count = [0]

    def _proc(tokens, logits):
        count[0] += 1
        if count[0] < min_tokens:
            vals = logits.tolist()
            flat = vals[0] if isinstance(vals[0], list) else vals
            for eid in eos_ids:
                if 0 <= eid < len(flat):
                    flat[eid] = -1e9
            logits_arr = mx.array([flat]) if isinstance(vals[0], list) else mx.array(flat)
            return logits_arr
        return logits

    return _proc


def _make_rep_penalty(penalty: float):
    if penalty <= 1.0 or not MLX_AVAILABLE:
        return None
    seen: List[int] = []

    def _proc(tokens, logits):
        if tokens is not None and hasattr(tokens, "size") and tokens.size > 0:
            seen.extend(tokens.tolist() if hasattr(tokens, "tolist") else [int(tokens)])
        if not seen:
            return logits
        vals = logits.tolist()
        flat = vals[0] if isinstance(vals[0], list) else vals
        for tid in set(seen):
            if 0 <= tid < len(flat):
                if flat[tid] > 0:
                    flat[tid] = flat[tid] / penalty
                else:
                    flat[tid] = flat[tid] * penalty
        logits_arr = mx.array([flat]) if isinstance(vals[0], list) else mx.array(flat)
        return logits_arr

    return _proc


def _format_chat_prompt(tokenizer: Any, user_prompt: str, system_prompt: str | None) -> str:
    messages: List[Dict[str, str]] = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": user_prompt})
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prefix = f"System: {system_prompt.strip()}\n\n" if system_prompt else ""
        return f"{prefix}User: {user_prompt}\nAssistant:"


def _get_eos_ids(tokenizer: Any) -> List[int]:
    try:
        raw_eos = getattr(tokenizer, "eos_token_ids", None)
        if raw_eos is None:
            eid = getattr(tokenizer, "eos_token_id", None)
            raw_eos = [eid] if eid is not None else []
        return [int(e) for e in raw_eos if e is not None]
    except Exception:
        return []


def generate_once(
    model_id: str,
    prompt: str,
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    repetition_penalty: float,
) -> Tuple[str, int, float]:
    if not prompt.strip():
        return "", 0, 0.0
    spec = MODEL_SPECS[model_id]
    model, tokenizer = get_model(model_id)
    effective_system = system_prompt.strip() or spec.system_prompt
    formatted = _format_chat_prompt(tokenizer, prompt, effective_system)
    sampler = make_sampler(temp=temperature, top_p=top_p)

    processors: List[Any] = []
    eos_ids = _get_eos_ids(tokenizer)
    sup = _make_suppress_eos(spec.min_tokens, eos_ids)
    if sup:
        processors.append(sup)
    rep = _make_rep_penalty(repetition_penalty)
    if rep:
        processors.append(rep)

    t0 = time.time()
    token_count = 0
    parts: List[str] = []

    with _mlx_lock:
        gen_kwargs: Dict[str, Any] = dict(
            model=model,
            tokenizer=tokenizer,
            prompt=formatted,
            max_tokens=max_tokens,
            sampler=sampler,
        )
        if processors:
            gen_kwargs["logits_processors"] = processors
        try:
            for chunk in stream_generate(**gen_kwargs):
                parts.append(chunk.text)
                token_count += 1
        except Exception as exc:
            return f"[Generation error: {exc}]", 0, 0.0

    text = "".join(parts)
    text = _truncate_ngram(text, n=4)
    elapsed = time.time() - t0
    return text, token_count, elapsed


def random_eval_prompt() -> str:
    if not EVAL_PROMPTS_PATH.exists():
        return ""
    prompts: List[str] = []
    with EVAL_PROMPTS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                prompts.append(obj.get("prompt", ""))
            except Exception:
                continue
    if not prompts:
        return ""
    import random

    return random.choice(prompts)


def collected_count() -> int:
    if not COLLECTED_PATH.exists():
        return 0
    count = 0
    with COLLECTED_PATH.open() as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def save_collected_example(prompt: str, response: str) -> None:
    COLLECTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    }
    with COLLECTED_PATH.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@st.cache_data(show_spinner=False)
def load_dataset_rows() -> List[Tuple[int, int, int, str, str]]:
    """Return list of (idx, input_words, output_words, user_preview, assistant_preview)."""
    if not TRAIN_PATH.exists():
        return []
    rows: List[Tuple[int, int, int, str, str]] = []
    with TRAIN_PATH.open() as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            msgs = obj.get("messages") or []
            user = ""
            assistant = ""
            for m in msgs:
                if m.get("role") == "user":
                    user = str(m.get("content", ""))
                elif m.get("role") == "assistant":
                    assistant = str(m.get("content", ""))
            in_words = len(user.split())
            out_words = len(assistant.split())
            rows.append(
                (
                    idx,
                    in_words,
                    out_words,
                    user[:120].replace("\n", " "),
                    assistant[:120].replace("\n", " "),
                )
            )
    return rows


def render_inference_page():
    st.subheader("Inference / Compare")
    cols = st.columns(2)
    with cols[0]:
        model_a = st.selectbox("Model A", MODEL_IDS, index=MODEL_IDS.index("full_ft") if "full_ft" in MODEL_IDS else 0)
    with cols[1]:
        model_b = st.selectbox("Model B", MODEL_IDS, index=MODEL_IDS.index("base") if "base" in MODEL_IDS else 0)

    # Manage the prompt value manually via session_state to support the
    # "Random Prompt" button without conflicting with widget keys.
    current_prompt = st.session_state.get("inf_prompt", "")
    prompt = st.text_area(
        "Prompt",
        height=150,
        placeholder="Enter an instruction or question to compare models...",
        value=current_prompt,
    )
    system_prompt = st.text_area(
        "System prompt (optional)",
        height=80,
        placeholder="e.g. You are a helpful teaching assistant.",
        key="inf_system",
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        temperature = st.slider("Temperature", 0.0, 2.0, 0.2, 0.05)
    with c2:
        top_p = st.slider("Top-p", 0.0, 1.0, 0.9, 0.01)
    with c3:
        max_tokens = st.slider("Max tokens", 64, 1024, 512, 16)
    with c4:
        repetition_penalty = st.slider("Repetition penalty", 1.0, 2.0, 1.5, 0.05)

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        if st.button("🎲 Random Prompt"):
            st.session_state["inf_prompt"] = random_eval_prompt()
            st.rerun()
    with col_btn2:
        run = st.button("Generate Both", type="primary")

    if run:
        if not MLX_AVAILABLE:
            st.error(
                "MLX / mlx_lm not available in this environment. "
                "Install them in your virtualenv to run generation."
            )
            return
        if not prompt.strip():
            st.warning("Enter a prompt first.")
            return

        with st.spinner("Generating with Model A..."):
            text_a, toks_a, el_a = generate_once(
                model_a, prompt, system_prompt, temperature, top_p, max_tokens, repetition_penalty
            )
        with st.spinner("Generating with Model B..."):
            text_b, toks_b, el_b = generate_once(
                model_b, prompt, system_prompt, temperature, top_p, max_tokens, repetition_penalty
            )

        col_out1, col_out2 = st.columns(2)
        with col_out1:
            st.markdown(f"**Model A:** `{model_a}`  \nTokens: {toks_a} · Time: {el_a:.2f}s")
            st.text_area("Response A", value=text_a, height=260, key="resp_a")
        with col_out2:
            st.markdown(f"**Model B:** `{model_b}`  \nTokens: {toks_b} · Time: {el_b:.2f}s")
            st.text_area("Response B", value=text_b, height=260, key="resp_b")


def render_collection_page():
    st.subheader("Data Collection")
    st.markdown(
        "Use this page to generate and curate new training examples. "
        "Edited responses are saved in `data/collected.jsonl` as "
        "`[user, assistant]` message pairs."
    )

    top = st.columns(4)
    with top[0]:
        model_id = st.selectbox("Model", MODEL_IDS, index=MODEL_IDS.index("full_ft") if "full_ft" in MODEL_IDS else 0)
    with top[1]:
        temperature = st.slider("Temperature", 0.0, 2.0, 0.2, 0.05, key="dc_temp")
    with top[2]:
        top_p = st.slider("Top-p", 0.0, 1.0, 0.9, 0.01, key="dc_top_p")
    with top[3]:
        max_tokens = st.slider("Max tokens", 64, 1024, 512, 16, key="dc_max_tokens")

    rep = st.slider("Repetition penalty", 1.0, 2.0, 1.5, 0.05, key="dc_rep")
    sys = st.text_area(
        "System prompt (optional)",
        height=60,
        key="dc_sys",
    )
    prompt = st.text_area(
        "Prompt",
        height=120,
        key="dc_prompt",
    )
    current_response = st.session_state.get("dc_response", "")
    response = st.text_area(
        "Model response (editable before saving)",
        height=200,
        value=current_response,
    )

    col_btns = st.columns([1, 1, 1])
    with col_btns[0]:
        gen = st.button("Generate", type="primary")
    with col_btns[1]:
        save = st.button("Save Example")
    with col_btns[2]:
        download = st.button("Refresh Download Link")

    status_placeholder = st.empty()
    count_placeholder = st.empty()

    if gen:
        if not MLX_AVAILABLE:
            st.error(
                "MLX / mlx_lm not available in this environment. "
                "Install them in your virtualenv to run generation."
            )
        elif not prompt.strip():
            st.warning("Enter a prompt before generating.")
        else:
            with st.spinner("Generating response..."):
                text, _, _ = generate_once(
                    model_id, prompt, sys, temperature, top_p, max_tokens, rep
                )
            st.session_state["dc_response"] = text
            st.rerun()

    if save:
        if not prompt.strip() or not response.strip():
            status_placeholder.warning("Prompt and response must be non-empty.")
        else:
            save_collected_example(prompt, response)
            status_placeholder.success("Saved example to `data/collected.jsonl`.")
            count_placeholder.info(f"Total examples collected: {collected_count()}")

    if download:
        if not COLLECTED_PATH.exists():
            status_placeholder.warning("No collected examples yet.")
        else:
            with COLLECTED_PATH.open("rb") as f:
                data = f.read()
            st.download_button(
                "Download collected.jsonl",
                data,
                file_name="collected.jsonl",
                mime="application/jsonl",
            )


def render_eda_page():
    st.subheader("Dataset Stats / EDA")
    st.markdown(
        "Click **Load Stats** to compute basic dataset statistics from `data/train.jsonl`. "
        "If `data/eda_report.md` exists, it will be rendered below."
    )

    if st.button("Load Stats", type="primary"):
        rows = load_dataset_rows()
        if not rows:
            st.warning(
                "Dataset not found or empty at `data/train.jsonl`. "
                "Run `prepare_dataset.py` first."
            )
            return

        out_lengths = np.array([r[2] for r in rows], dtype=np.int32)
        total = int(len(out_lengths))
        mean = float(out_lengths.mean())
        median = float(np.median(out_lengths))
        p90 = float(np.percentile(out_lengths, 90))
        p99 = float(np.percentile(out_lengths, 99))
        min_v = int(out_lengths.min())
        max_v = int(out_lengths.max())

        st.markdown(
            f"**Total examples:** {total}  \n"
            f"**Output length (words)** — "
            f"min {min_v}, mean {mean:.1f}, median {median:.1f}, "
            f"p90 {p90:.1f}, p99 {p99:.1f}, max {max_v}"
        )

        if plt is not None:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.hist(out_lengths, bins=10, color="#60a5fa", edgecolor="black", alpha=0.85)
            ax.set_xlabel("Answer length (words)")
            ax.set_ylabel("Count")
            ax.set_title("Answer length distribution (train set)")
            fig.tight_layout()
            st.pyplot(fig)

        rows_sorted = sorted(rows, key=lambda r: r[2])
        bottom10 = rows_sorted[:10]
        top10 = rows_sorted[-10:]
        table_rows: List[Dict[str, Any]] = []
        for which, subset in (("shortest", bottom10), ("longest", top10)):
            for idx, in_w, out_w, user_prev, asst_prev in subset:
                table_rows.append(
                    {
                        "which": which,
                        "index": idx,
                        "input_words": in_w,
                        "output_words": out_w,
                        "user_preview": user_prev,
                        "assistant_preview": asst_prev,
                    }
                )
        if table_rows:
            st.dataframe(table_rows, use_container_width=True)

        if EDA_REPORT_PATH.exists():
            st.markdown("---")
            st.markdown("### EDA Report (`data/eda_report.md`)")
            st.markdown(EDA_REPORT_PATH.read_text())


def main() -> None:
    st.set_page_config(
        page_title="TinyLlama Streamlit UI",
        layout="wide",
    )
    st.title("TinyLlama Instruction Tuning — Streamlit UI")

    page = st.sidebar.radio(
        "Page",
        ["Inference / Compare", "Data Collection", "Dataset Stats / EDA"],
    )

    if page == "Inference / Compare":
        render_inference_page()
    elif page == "Data Collection":
        render_collection_page()
    else:
        render_eda_page()


if __name__ == "__main__":
    main()

