"""
TinyLlama Task 4 — Quantization & GGUF (Hugging Face Space).

Benchmark dashboard plus a **Play** tab: pick Q4_K_M / Q5_K_M / Q8_0, download from the
Hub model repo, and run chat via `llama-cpp-python` (CPU; first load is slow).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"
DEBUG_LOG_PATH = Path("/Users/ysingh/PyCharmMiscProject/.cursor/debug-365fe9.log")
DEBUG_SESSION_ID = "365fe9"


def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    import sys
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "id": f"log_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
        "timestamp": int(time.time() * 1000),
        "location": location,
        "message": message,
        "data": data,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
    }
    # region agent log
    try:
        with DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass
    # Also print to stderr so HF Space container logs capture it
    try:
        print(f"[DEBUG-{DEBUG_SESSION_ID}] {hypothesis_id} {location}: {message} | {data}", file=sys.stderr)
    except Exception:
        pass
    # endregion


def _load_json(name: str) -> dict:
    path = RESULTS / name
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _comparison_table_md(summary: dict) -> str:
    rows = summary.get("comparison_rows") or []
    if not rows:
        return "_No comparison rows in `task4_summary.json`._"
    hdr = (
        "| Quant | Size (MB) | tok/s | Latency (s) | Peak RSS (MB) | PPL | ΔPPL | B1 | B4 | B8 |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = []
    for r in rows:
        lines.append(
            f"| **{r['quantization']}** | {r['file_size_mb']:.2f} | {r['tokens_per_second']:.3f} | "
            f"{r['mean_latency_sec']:.4f} | {r['peak_rss_mb']:.2f} | {r['perplexity']:.4f} | "
            f"{r['perplexity_loss']:.4f} | {r['batch_1_tps']:.3f} | {r['batch_4_tps']:.3f} | "
            f"{r['batch_8_tps']:.3f} |"
        )
    return hdr + "\n".join(lines)


def _deployment_md(dep: dict) -> str:
    if not dep:
        return "_No `deployment_comparison.json`._"
    hw = dep.get("hardware", "")
    date = dep.get("comparison_date", "")
    models = dep.get("models") or {}
    parts = [f"**Hardware:** {hw}  \n**Recorded:** {date}\n"]
    for name, m in models.items():
        parts.append(f"\n### {name}\n")
        for k, v in m.items():
            parts.append(f"- **{k}:** `{v}`\n")
    rec = dep.get("recommendations") or {}
    if rec:
        parts.append("\n### Recommendations\n")
        for key, block in rec.items():
            if isinstance(block, dict):
                parts.append(
                    f"- **{key}:** {block.get('recommended', '')} — _{block.get('reason', '')}_\n"
                )
    return "".join(parts)


def _perplexity_md(ppl: dict) -> str:
    if not ppl:
        return "_No perplexity JSON._"
    lines = ["### Perplexity (held-out Alpaca validation)\n"]
    corp = ppl.get("corpus") or {}
    if corp:
        lines.append(
            f"- Corpus: `{corp.get('path', '')}` — rows ≈ **{corp.get('rows_used', '?')}**, "
            f"tokens ≈ **{corp.get('token_estimate', '?')}**\n"
        )
    for k, v in ppl.items():
        if k == "corpus" or not isinstance(v, dict):
            continue
        if "perplexity" in v:
            lines.append(
                f"- **{k}:** PPL **{v['perplexity']:.4f}** (σ ≈ {v.get('perplexity_std', 0):.4f})\n"
            )
    return "".join(lines)


def _play_intro_md() -> str:
    """Explain whether in-browser GGUF chat is available (depends on llama-cpp-python)."""
    try:
        from gguf_chat import _deps_ok

        ok, err = _deps_ok()
    except Exception as exc:
        ok = False
        err = f"{type(exc).__name__}: {exc}"
    # region agent log
    import platform
    _debug_log(
        "run-365fe9",
        "H2",
        "task4_hf/app.py:_play_intro_md",
        "Play intro dependency probe",
        {
            "deps_ok": bool(ok),
            "deps_error": err if not ok else "",
            "python_version": os.sys.version,
            "python_full": platform.python_version(),
            "libc": str(platform.libc_ver()),
            "machine": platform.machine(),
        },
    )
    # endregion
    base = (
        "Model repo: "
        "[ysingh-aiml/tinyllama-alpaca-lora-gguf](https://huggingface.co/ysingh-aiml/tinyllama-alpaca-lora-gguf)\n\n"
    )
    if ok:
        return (
            base
            + "Select a **GGUF** quantization, type a message, and hit **Generate**. "
            "The first run **downloads** the weights (~640 MB – 1.1 GB) and loads them with "
            "`llama-cpp-python` (CPU — first generation can take a minute)."
        )
    return (
        base
        + "### Interactive chat unavailable\n\n"
        f"`llama-cpp-python` failed to load: `{err}`\n\n"
        "**Run locally:**\n"
        "```bash\npip install -r requirements.txt\npython app.py\n```\n\n"
        "Or use the **Local server** tab with a system `llama-server` binary."
    )


def _play_deps_status() -> tuple[bool, str]:
    try:
        from gguf_chat import _deps_ok

        return _deps_ok()
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _play_generate(quant_label: str, message: str, max_tokens: float) -> tuple[str, str]:
    # region agent log
    _debug_log(
        "run-365fe9",
        "H4",
        "task4_hf/app.py:_play_generate",
        "Generate clicked",
        {
            "quant_label": quant_label,
            "message_len": len((message or "").strip()),
            "max_tokens": int(max_tokens) if max_tokens is not None else None,
        },
    )
    # endregion
    try:
        from gguf_chat import _deps_ok, generate_chat

        ok, err = _deps_ok()
        # region agent log
        _debug_log(
            "run-365fe9",
            "H2",
            "task4_hf/app.py:_play_generate",
            "Dependency check result in generate path",
            {
                "deps_ok": bool(ok),
                "deps_error": err if not ok else "",
                "python_version": os.sys.version,
            },
        )
        # endregion
        if not ok:
            # region agent log
            _debug_log(
                "run-365fe9",
                "H3",
                "task4_hf/app.py:_play_generate",
                "Returning missing dependency message",
                {"status": "missing_llama_cpp", "error": err},
            )
            # endregion
            return (
                "",
                f"**`llama-cpp-python` not installed** — {err}\n\n"
                "Install optional deps: `pip install -r requirements-play.txt` (use **Python 3.10/3.11** "
                "for binary wheels).",
            )
        text, status = generate_chat(
            quant_label,
            message,
            max_tokens=int(max_tokens),
        )
        # region agent log
        _debug_log(
            "run-365fe9",
            "H4",
            "task4_hf/app.py:_play_generate",
            "Generate returned",
            {"status": status, "output_len": len(text or "")},
        )
        # endregion
        return text, status
    except Exception as exc:  # noqa: BLE001
        # region agent log
        _debug_log(
            "run-365fe9",
            "H4",
            "task4_hf/app.py:_play_generate",
            "Generate exception",
            {"error_type": type(exc).__name__, "error": str(exc)},
        )
        # endregion
        return "", f"Error: {exc}"


def build_app() -> gr.Blocks:
    from gguf_chat import QUANT_CHOICES as _quants
    play_ok, play_err = _play_deps_status()
    # region agent log
    _debug_log(
        "run-365fe9",
        "H1",
        "task4_hf/app.py:build_app",
        "Play controls availability decided",
        {
            "play_ok": bool(play_ok),
            "play_err": play_err if not play_ok else "",
            "python_version": os.sys.version,
            "platform": os.sys.platform,
        },
    )
    # endregion

    summary = _load_json("task4_summary.json")
    dep = _load_json("deployment_comparison.json")
    ppl = _load_json("perplexity_results.json")

    intro = r"""
# TinyLlama — Quantization & GGUF (Task 4)

**Base:** [`TinyLlama/TinyLlama-1.1B-Chat-v1.0`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)  
**Pipeline:** fuse LoRA → convert to GGUF (llama.cpp) → quantize **Q4\_K\_M**, **Q5\_K\_M**, **Q8\_0** → benchmark latency, memory, perplexity, batched throughput.

**GGUF weights** live in **[ysingh-aiml/tinyllama-alpaca-lora-gguf](https://huggingface.co/ysingh-aiml/tinyllama-alpaca-lora-gguf)**.

The **Try a quant** tab lets you run GGUF generation in-browser via **`llama-cpp-python`** (CPU; first generation downloads weights and may take a minute). Use the **Local server** tab for `llama-server`.
"""

    with gr.Blocks(
        title="TinyLlama GGUF Quantization",
        theme=gr.themes.Soft(),
        css=".plot-img img { border-radius: 8px; max-width: 100%; }",
    ) as demo:
        gr.Markdown(intro)

        with gr.Tabs():
            with gr.Tab("🎮 Try a quant"):
                gr.Markdown(_play_intro_md())
                with gr.Row():
                    quant_dd = gr.Dropdown(
                        choices=list(_quants.keys()),
                        value=list(_quants.keys())[0],
                        label="Quantized model",
                        scale=1,
                    )
                    max_tok = gr.Slider(
                        32,
                        256,
                        value=128,
                        step=32,
                        label="Max new tokens",
                        scale=1,
                    )
                msg = gr.Textbox(
                    label="Message",
                    placeholder="Ask something…" if play_ok else "Install requirements-play.txt to enable chat.",
                    lines=4,
                    interactive=play_ok,
                )
                play_btn = gr.Button("Generate", variant="primary", interactive=play_ok)
                play_out = gr.Textbox(label="Assistant", lines=12, interactive=False)
                play_status_text = ""
                if not play_ok:
                    play_status_text = (
                        f"**Play disabled in this build** — missing dependency: `{play_err}`. "
                        "Use `requirements-play.txt` locally or Local server tab."
                    )
                play_status = gr.Markdown(value=play_status_text)
                # region agent log
                _debug_log(
                    "run-365fe9",
                    "H5",
                    "task4_hf/app.py:build_app",
                    "Play widget interactivity configured",
                    {
                        "button_interactive": bool(play_ok),
                        "textbox_interactive": bool(play_ok),
                        "status_text_len": len(play_status_text),
                    },
                )
                # endregion

                play_btn.click(
                    fn=_play_generate,
                    inputs=[quant_dd, msg, max_tok],
                    outputs=[play_out, play_status],
                )
                msg.submit(
                    fn=_play_generate,
                    inputs=[quant_dd, msg, max_tok],
                    outputs=[play_out, play_status],
                )

            with gr.Tab("📋 Summary"):
                gr.Markdown(_deployment_md(dep))
                gr.Markdown(_perplexity_md(ppl))

            with gr.Tab("📊 Comparison table"):
                gr.Markdown(_comparison_table_md(summary))
                gr.Markdown(
                    "Download raw CSV from the **Files** tab: `results/deployment_comparison.csv`."
                )

            with gr.Tab("📈 Plots"):
                with gr.Row():
                    gr.Image(str(PLOTS / "throughput_vs_quant.png"), label="Throughput", elem_classes=["plot-img"])
                    gr.Image(str(PLOTS / "size_vs_perplexity.png"), label="Size vs perplexity", elem_classes=["plot-img"])
                gr.Image(str(PLOTS / "batch_throughput.png"), label="Batched inference (llama-bench)", elem_classes=["plot-img"])

            with gr.Tab("🖥️ Local server"):
                gr.Markdown(
                    f"""
## Inference server (`llama-server`)

Weights: **[ysingh-aiml/tinyllama-alpaca-lora-gguf](https://huggingface.co/ysingh-aiml/tinyllama-alpaca-lora-gguf)** — the script downloads into `{ROOT}/models/` on first run if missing.

```bash
cd task4_hf
pip install -r inference_server/requirements.txt
export LLAMA_SERVER=/path/to/llama-server   # or use `llama-server` on PATH

# Default: fetch Q4_K_M if needed, then serve
python inference_server/server.py --port 8080 --n-gpu-layers 0

# Other quants: --quant q5_k_m | q8_0
# Download only: --fetch-only
# Offline: put a .gguf in models/ and pass --no-fetch
```

See `inference_server/README.md` for `--hf-repo`, `--model`, and API notes.
"""
                )

    return demo


if __name__ == "__main__":
    build_app().launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
