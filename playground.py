#!/usr/bin/env python3
"""
Model Comparison Playground
Usage:  .venv/bin/python playground.py
Browser: http://localhost:8765
"""
from __future__ import annotations

import json
import random
import threading
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

try:
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("WARNING: mlx_lm not found — model loading will fail.")

ROOT = Path(__file__).parent

# ── Model registry ────────────────────────────────────────────────────────────

MODEL_REGISTRY: List[Dict] = [
    {
        "id": "full_ft",
        "label": "TinyLlama Full FT",
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "adapter_path": "./mlx_best_models/full",
        "size": "1.1B",
        "badge": "Full FT",
        "hue": "#60a5fa",
    },
    {
        "id": "lora_ft",
        "label": "TinyLlama LoRA FT",
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "adapter_path": "./mlx_best_models/lora",
        "size": "1.1B",
        "badge": "LoRA",
        "hue": "#a78bfa",
    },
    {
        "id": "qlora_ft",
        "label": "TinyLlama QLoRA FT",
        "model": "./models/tinyllama-4bit-base",
        "adapter_path": "./mlx_best_models/qlora",
        "size": "1.1B 4-bit",
        "badge": "QLoRA",
        "hue": "#f472b6",
    },
    {
        "id": "base",
        "label": "TinyLlama Base",
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "adapter_path": None,
        "size": "1.1B",
        "badge": "Base",
        "hue": "#94a3b8",
    },
    {
        "id": "qwen_1.8b",
        "label": "Qwen 1.5 1.8B",
        "model": "./models/qwen1.5-1.8b-chat-4bit",
        "adapter_path": None,
        "size": "1.8B 4-bit",
        "badge": "Qwen",
        "hue": "#34d399",
    },
    {
        "id": "phi_2",
        "label": "Phi-2",
        "model": "./models/phi-2-hf-4bit-mlx",
        "adapter_path": None,
        "size": "2.7B 4-bit",
        "badge": "Phi-2",
        "hue": "#fb923c",
    },
]

# ── Global state ──────────────────────────────────────────────────────────────

_mlx_lock = threading.Lock()


class ModelSlot:
    def __init__(self) -> None:
        self.model_id: Optional[str] = None
        self.model = None
        self.tokenizer = None
        self.loading: bool = False
        self.error: Optional[str] = None

    def clear(self) -> None:
        self.model = None
        self.tokenizer = None
        self.model_id = None
        self.error = None
        if MLX_AVAILABLE:
            try:
                if hasattr(mx, "clear_cache"):
                    mx.clear_cache()
                elif hasattr(mx, "metal"):
                    mx.metal.clear_cache()
            except Exception:
                pass


slots: Dict[str, ModelSlot] = {"A": ModelSlot(), "B": ModelSlot()}

# ── Helpers ───────────────────────────────────────────────────────────────────


def _truncate_ngram(text: str, n: int = 4) -> str:
    words = text.split()
    if len(words) < n * 2:
        return text
    seen: set = set()
    for i in range(len(words) - n + 1):
        ng = tuple(words[i: i + n])
        if ng in seen:
            truncated = " ".join(words[:i]).rstrip(" ,.;:")
            return truncated if truncated else text
        seen.add(ng)
    return text


def _make_suppress_eos(min_tokens: int, eos_ids: list):
    """Suppress EOS tokens for the first min_tokens steps (pure Python list approach)."""
    if min_tokens <= 0 or not eos_ids:
        return None
    count = [0]

    def _proc(tokens, logits):
        count[0] += 1
        if count[0] < min_tokens:
            vals = logits.tolist()
            flat = vals[0] if isinstance(vals[0], list) else vals
            for eid in eos_ids:
                if eid < len(flat):
                    flat[eid] = -1e9
            logits = mx.array([flat]) if isinstance(vals[0], list) else mx.array(flat)
        return logits

    return _proc


def _make_rep_penalty(penalty: float):
    """Penalise already-generated tokens (pure Python list approach, no .at[].set())."""
    if penalty <= 1.0:
        return None
    seen: list = []

    def _proc(tokens, logits):
        if tokens is not None and hasattr(tokens, "size") and tokens.size > 0:
            seen.extend(tokens.tolist() if hasattr(tokens, "tolist") else [int(tokens)])
        if not seen:
            return logits
        vals = logits.tolist()
        flat = vals[0] if isinstance(vals[0], list) else vals
        for tid in set(seen):
            if tid < len(flat):
                flat[tid] = flat[tid] / penalty if flat[tid] > 0 else flat[tid] * penalty
        logits = mx.array([flat]) if isinstance(vals[0], list) else mx.array(flat)
        return logits

    return _proc


def _load_prompts() -> List[str]:
    path = ROOT / "evaluation" / "eval_prompts.jsonl"
    if not path.exists():
        return []
    prompts = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    prompts.append(json.loads(line)["prompt"])
                except Exception:
                    pass
    return prompts


_PROMPTS: List[str] = _load_prompts()

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Playground")


class LoadRequest(BaseModel):
    model_id: str


class GenerateRequest(BaseModel):
    prompt: str
    system: str = ""
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 512
    min_tokens: int = 15
    repetition_penalty: float = 1.5
    ngram_truncate: bool = True


@app.get("/api/models")
def api_models():
    return MODEL_REGISTRY


@app.get("/api/status")
def api_status():
    return {
        sid: {
            "model_id": s.model_id,
            "loaded": s.model is not None,
            "loading": s.loading,
            "error": s.error,
        }
        for sid, s in slots.items()
    }


@app.get("/api/random-prompt")
def api_random_prompt():
    if not _PROMPTS:
        return {"prompt": ""}
    return {"prompt": random.choice(_PROMPTS)}


@app.post("/api/load/{slot_id}")
def api_load(slot_id: str, req: LoadRequest):
    if slot_id not in slots:
        raise HTTPException(404, f"Unknown slot: {slot_id}")
    if not MLX_AVAILABLE:
        raise HTTPException(503, "mlx_lm not installed")
    cfg = next((m for m in MODEL_REGISTRY if m["id"] == req.model_id), None)
    if not cfg:
        raise HTTPException(404, f"Unknown model: {req.model_id}")

    slot = slots[slot_id]
    if slot.model_id == req.model_id and slot.model is not None:
        return {"ok": True, "model_id": req.model_id, "cached": True}

    slot.loading = True
    slot.error = None
    try:
        with _mlx_lock:
            slot.clear()
            kwargs: Dict = {}
            if cfg.get("adapter_path"):
                kwargs["adapter_path"] = cfg["adapter_path"]
            model, tokenizer = load(cfg["model"], **kwargs)
            slot.model = model
            slot.tokenizer = tokenizer
            slot.model_id = req.model_id
    except Exception as exc:
        slot.error = str(exc)
        slot.loading = False
        raise HTTPException(500, str(exc))
    slot.loading = False
    return {"ok": True, "model_id": req.model_id}


@app.post("/api/unload/{slot_id}")
def api_unload(slot_id: str):
    if slot_id not in slots:
        raise HTTPException(404)
    with _mlx_lock:
        slots[slot_id].clear()
    return {"ok": True}


def _gen_stream(slot_id: str, req: GenerateRequest) -> Iterator[str]:
    slot = slots[slot_id]
    if not slot.model or not slot.tokenizer:
        yield f"data: {json.dumps({'error': 'Model not loaded — click Load first.'})}\n\n"
        return

    messages = []
    if req.system.strip():
        messages.append({"role": "system", "content": req.system.strip()})
    messages.append({"role": "user", "content": req.prompt})

    try:
        formatted = slot.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prefix = f"System: {req.system}\n\n" if req.system.strip() else ""
        formatted = f"{prefix}User: {req.prompt}\nAssistant:"

    sampler = make_sampler(temp=req.temperature, top_p=req.top_p)
    processors = []
    raw_eos = getattr(slot.tokenizer, "eos_token_ids", None)
    if raw_eos is None:
        _eid = getattr(slot.tokenizer, "eos_token_id", None)
        raw_eos = [_eid] if _eid is not None else []
    eos_ids = [int(e) for e in raw_eos if e is not None]
    sup = _make_suppress_eos(req.min_tokens, eos_ids)
    if sup:
        processors.append(sup)
    rep = _make_rep_penalty(req.repetition_penalty)
    if rep:
        processors.append(rep)

    with _mlx_lock:
        t0 = time.time()
        token_count = 0
        full_parts: list = []
        try:
            gen_kw: Dict = dict(
                model=slot.model,
                tokenizer=slot.tokenizer,
                prompt=formatted,
                max_tokens=req.max_tokens,
                sampler=sampler,
            )
            if processors:
                gen_kw["logits_processors"] = processors
            for chunk in stream_generate(**gen_kw):
                tok = chunk.text
                full_parts.append(tok)
                token_count += 1
                yield f"data: {json.dumps({'tok': tok, 'n': token_count, 'el': round(time.time()-t0,2)})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            return

    full = "".join(full_parts)
    if req.ngram_truncate:
        clipped = _truncate_ngram(full, n=4)
        if clipped != full:
            yield f"data: {json.dumps({'clip': len(full)-len(clipped)})}\n\n"

    total = time.time() - t0
    tps = round(token_count / total, 1) if total > 0 else 0
    yield f"data: {json.dumps({'done': True, 'n': token_count, 'el': round(total,2), 'tps': tps})}\n\n"


@app.post("/api/generate/{slot_id}")
def api_generate(slot_id: str, req: GenerateRequest):
    if slot_id not in slots:
        raise HTTPException(404)
    return StreamingResponse(
        _gen_stream(slot_id, req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Model Playground</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}

:root{
  --bg:       #080810;
  --surface:  #0f0f1a;
  --card:     #13131f;
  --border:   rgba(255,255,255,.08);
  --border2:  rgba(255,255,255,.14);
  --text:     #e2e8f0;
  --muted:    #64748b;
  --accent-a: #60a5fa;
  --accent-b: #a78bfa;
  --danger:   #f87171;
  --success:  #4ade80;
  --radius:   12px;
  --radius-sm:8px;
}

html,body{height:100%;background:var(--bg);color:var(--text);font-family:'Inter',system-ui,sans-serif;font-size:14px;line-height:1.6;overflow:hidden}

/* ── Scrollbar ─────────────────────────────────────────────── */
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:rgba(255,255,255,.12);border-radius:99px}
::-webkit-scrollbar-thumb:hover{background:rgba(255,255,255,.22)}

/* ── Layout ────────────────────────────────────────────────── */
#root{display:flex;flex-direction:column;height:100vh;overflow:hidden}

/* ── Header ────────────────────────────────────────────────── */
header{
  display:flex;align-items:center;gap:12px;
  padding:0 20px;height:54px;
  background:rgba(15,15,26,.9);
  border-bottom:1px solid var(--border);
  backdrop-filter:blur(12px);
  flex-shrink:0;
}
.logo{display:flex;align-items:center;gap:10px;margin-right:auto}
.logo-icon{
  width:30px;height:30px;border-radius:8px;
  background:linear-gradient(135deg,#3b82f6,#8b5cf6);
  display:flex;align-items:center;justify-content:center;font-size:15px;
}
.logo-name{font-weight:700;font-size:15px;letter-spacing:-.3px}
.logo-sub{font-size:11px;color:var(--muted);margin-left:2px}

header .btn{display:flex;align-items:center;gap:7px;padding:7px 16px;border-radius:8px;border:none;cursor:pointer;font-size:13px;font-weight:500;font-family:inherit;transition:all .15s}
.btn-ghost{background:rgba(255,255,255,.06);color:var(--text)}
.btn-ghost:hover{background:rgba(255,255,255,.12)}
.btn-primary{background:linear-gradient(135deg,#3b82f6,#8b5cf6);color:#fff}
.btn-primary:hover{opacity:.9;transform:translateY(-1px)}
.btn-primary:active{transform:translateY(0)}
.btn-primary:disabled{opacity:.4;cursor:not-allowed;transform:none}
.btn-sm{padding:5px 10px;font-size:12px;border-radius:6px}
.btn-danger{background:rgba(248,113,113,.15);color:var(--danger);border:1px solid rgba(248,113,113,.25)}
.btn-danger:hover{background:rgba(248,113,113,.25)}

/* ── Prompt bar ─────────────────────────────────────────────── */
#prompt-bar{
  flex-shrink:0;
  padding:12px 16px;
  background:var(--surface);
  border-bottom:1px solid var(--border);
  display:flex;flex-direction:column;gap:8px;
}
.prompt-row{display:flex;gap:8px;align-items:flex-end}
#prompt-input{
  flex:1;resize:none;
  background:rgba(255,255,255,.05);
  border:1px solid var(--border);
  border-radius:var(--radius-sm);
  color:var(--text);
  font-family:inherit;font-size:13.5px;
  padding:9px 12px;
  min-height:38px;max-height:120px;
  transition:border-color .15s;
  field-sizing:content;
}
#prompt-input:focus{outline:none;border-color:rgba(99,102,241,.5);background:rgba(255,255,255,.07)}
#prompt-input::placeholder{color:var(--muted)}

.sys-toggle{
  font-size:11.5px;color:var(--muted);cursor:pointer;
  display:flex;align-items:center;gap:5px;user-select:none;
}
.sys-toggle:hover{color:var(--text)}
.sys-toggle .arrow{transition:transform .2s;display:inline-block}
.sys-toggle.open .arrow{transform:rotate(90deg)}

#sys-area{display:none;gap:8px}
#sys-area.open{display:flex}
#sys-input{
  flex:1;resize:none;height:54px;
  background:rgba(255,255,255,.04);
  border:1px solid var(--border);
  border-radius:var(--radius-sm);
  color:var(--text);font-family:inherit;font-size:12.5px;
  padding:7px 10px;
  transition:border-color .15s;
}
#sys-input:focus{outline:none;border-color:rgba(99,102,241,.4)}
#sys-input::placeholder{color:var(--muted)}

/* ── Panels ─────────────────────────────────────────────────── */
#panels{display:flex;flex:1;overflow:hidden;gap:0}

.panel{
  flex:1;display:flex;flex-direction:column;
  overflow:hidden;
  position:relative;
  transition:flex .3s;
}
.panel::before{
  content:'';
  position:absolute;inset:0;
  pointer-events:none;
  border-top:2px solid transparent;
  transition:border-color .3s;
  z-index:1;
}
.panel.A::before{border-color:var(--col-a, #60a5fa)}
.panel.B::before{border-color:var(--col-b, #a78bfa)}

.divider{width:1px;background:var(--border);flex-shrink:0}

/* ── Panel header ───────────────────────────────────────────── */
.ph{
  padding:10px 14px;
  background:var(--card);
  border-bottom:1px solid var(--border);
  display:flex;flex-direction:column;gap:8px;
  flex-shrink:0;
}
.ph-top{display:flex;align-items:center;gap:8px}
.slot-label{
  font-size:10px;font-weight:700;letter-spacing:.8px;
  padding:2px 8px;border-radius:4px;text-transform:uppercase;
}
.A .slot-label{background:rgba(96,165,250,.15);color:#60a5fa}
.B .slot-label{background:rgba(167,139,250,.15);color:#a78bfa}

.model-select{
  flex:1;
  background:rgba(255,255,255,.06);
  border:1px solid var(--border);
  border-radius:var(--radius-sm);
  color:var(--text);
  font-size:12.5px;font-family:inherit;
  padding:5px 10px;
  cursor:pointer;
  appearance:none;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%2364748b' stroke-width='2'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E");
  background-repeat:no-repeat;
  background-position:right 8px center;
  padding-right:28px;
}
.model-select:focus{outline:none;border-color:rgba(99,102,241,.5)}
.model-select option{background:#1a1a2e}

.model-badge{
  font-size:10.5px;font-weight:600;letter-spacing:.4px;
  padding:2px 8px;border-radius:20px;
  flex-shrink:0;
}

.load-btn{
  padding:5px 14px;border-radius:6px;border:none;cursor:pointer;
  font-size:12px;font-weight:600;font-family:inherit;
  transition:all .15s;white-space:nowrap;
  display:flex;align-items:center;gap:6px;
}
.load-btn.idle{background:rgba(255,255,255,.08);color:var(--text)}
.load-btn.idle:hover{background:rgba(255,255,255,.14)}
.load-btn.loading{background:rgba(99,102,241,.2);color:#818cf8;cursor:wait}
.load-btn.loaded{background:rgba(74,222,128,.12);color:var(--success)}
.load-btn.loaded:hover{background:rgba(74,222,128,.2)}
.load-btn.error{background:rgba(248,113,113,.12);color:var(--danger)}

/* ── Params ─────────────────────────────────────────────────── */
.params-toggle{
  display:flex;align-items:center;gap:5px;
  font-size:11.5px;color:var(--muted);cursor:pointer;
  user-select:none;transition:color .15s;
  width:fit-content;
}
.params-toggle:hover{color:var(--text)}
.params-toggle .arr{transition:transform .2s;font-size:9px}
.params-toggle.open .arr{transform:rotate(90deg)}

.params-grid{
  display:none;
  grid-template-columns:1fr 1fr;
  gap:8px 16px;
  padding-top:4px;
}
.params-grid.open{display:grid}

.param{display:flex;flex-direction:column;gap:3px}
.param-label{display:flex;justify-content:space-between;font-size:11px;color:var(--muted)}
.param-label span{color:var(--text);font-weight:500;font-family:'JetBrains Mono',monospace;font-size:11px}
.param input[type=range]{
  width:100%;height:4px;
  -webkit-appearance:none;
  border-radius:99px;
  outline:none;cursor:pointer;
}
.param input[type=range]::-webkit-slider-thumb{
  -webkit-appearance:none;
  width:14px;height:14px;border-radius:50%;
  border:2px solid var(--bg);
  cursor:pointer;
  transition:transform .1s;
}
.panel.A .param input[type=range]::-webkit-slider-thumb{background:#60a5fa}
.panel.B .param input[type=range]::-webkit-slider-thumb{background:#a78bfa}
.param input[type=range]::-webkit-slider-thumb:hover{transform:scale(1.2)}

.param-check{display:flex;align-items:center;gap:6px;cursor:pointer;font-size:11.5px;color:var(--muted);margin-top:2px}
.param-check input{accent-color:#818cf8;width:13px;height:13px}
.param-check:hover{color:var(--text)}

/* ── Response area ──────────────────────────────────────────── */
.resp-wrap{flex:1;overflow:hidden;display:flex;flex-direction:column}

.resp-area{
  flex:1;overflow-y:auto;
  padding:14px 16px;
  font-family:'JetBrains Mono','Fira Code',monospace;
  font-size:13px;line-height:1.75;
  white-space:pre-wrap;word-break:break-word;
  color:var(--text);
  position:relative;
}
.resp-area.empty::before{
  content:attr(data-placeholder);
  color:var(--muted);
  font-family:'Inter',sans-serif;
  font-style:italic;font-size:13px;
}
.cursor{
  display:inline-block;width:2px;height:1em;
  background:currentColor;margin-left:1px;
  vertical-align:text-bottom;
  animation:blink .8s step-end infinite;
}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}

/* generating ring */
.panel.generating::after{
  content:'';position:absolute;inset:0;pointer-events:none;
  border-radius:0;
  box-shadow:inset 0 0 0 1px rgba(99,102,241,.4);
  animation:pulse-ring 1.5s ease-in-out infinite;
  z-index:0;
}
@keyframes pulse-ring{0%,100%{opacity:.4}50%{opacity:1}}

/* ── Stats footer ───────────────────────────────────────────── */
.stats{
  padding:7px 14px;
  background:var(--card);
  border-top:1px solid var(--border);
  display:flex;gap:12px;align-items:center;
  flex-shrink:0;font-size:11.5px;
}
.stat{display:flex;align-items:center;gap:5px;color:var(--muted)}
.stat-val{font-weight:600;font-family:'JetBrains Mono',monospace;font-size:11px}
.A .stat-val{color:#60a5fa}
.B .stat-val{color:#a78bfa}
.stat-sep{color:var(--border2);margin:0 2px}

.gen-btn{
  margin-left:auto;padding:5px 14px;
  border-radius:6px;border:none;cursor:pointer;
  font-size:12px;font-weight:600;font-family:inherit;
  transition:all .15s;
}
.A .gen-btn{background:rgba(96,165,250,.15);color:#60a5fa;border:1px solid rgba(96,165,250,.25)}
.A .gen-btn:hover:not(:disabled){background:rgba(96,165,250,.25)}
.B .gen-btn{background:rgba(167,139,250,.15);color:#a78bfa;border:1px solid rgba(167,139,250,.25)}
.B .gen-btn:hover:not(:disabled){background:rgba(167,139,250,.25)}
.gen-btn:disabled{opacity:.35;cursor:not-allowed}

/* ── Spinner ────────────────────────────────────────────────── */
.spin{
  width:12px;height:12px;border-radius:50%;
  border:2px solid rgba(255,255,255,.15);
  border-top-color:#818cf8;
  animation:spin .7s linear infinite;
  display:inline-block;
}
@keyframes spin{to{transform:rotate(360deg)}}

/* ── Toast ──────────────────────────────────────────────────── */
#toast{
  position:fixed;bottom:24px;left:50%;transform:translateX(-50%) translateY(20px);
  background:#1e1e30;border:1px solid var(--border2);
  border-radius:8px;padding:9px 18px;font-size:13px;
  opacity:0;transition:all .25s;pointer-events:none;z-index:100;
  white-space:nowrap;
}
#toast.show{opacity:1;transform:translateX(-50%) translateY(0)}
#toast.error{border-color:rgba(248,113,113,.4);color:var(--danger)}

/* ── Generate Both progress ─────────────────────────────────── */
.both-progress{
  display:none;align-items:center;gap:8px;font-size:12px;color:var(--muted);
}
.both-progress.visible{display:flex}
.prog-dot{width:6px;height:6px;border-radius:50%;flex-shrink:0}
.prog-dot.a{background:#60a5fa}
.prog-dot.b{background:#a78bfa}
</style>
</head>
<body>
<div id="root">

<!-- ── Header ──────────────────────────────────────────────── -->
<header>
  <div class="logo">
    <div class="logo-icon">⚗</div>
    <div>
      <div class="logo-name">Model Playground <span class="logo-sub">– TinyLlama Fine-tune Lab</span></div>
    </div>
  </div>
  <div class="both-progress" id="both-prog">
    <div class="spin"></div>
    <span id="both-prog-text">Generating A…</span>
    <div class="prog-dot a" id="dot-a" style="opacity:.3"></div>
    <div class="prog-dot b" id="dot-b" style="opacity:.3"></div>
  </div>
  <button class="btn btn-ghost btn-sm" onclick="randomPrompt()" title="Load a random eval prompt">🎲 Random</button>
  <button class="btn btn-primary" id="gen-both-btn" onclick="generateBoth()" disabled>⚡ Generate Both</button>
</header>

<!-- ── Prompt bar ───────────────────────────────────────────── -->
<div id="prompt-bar">
  <div class="prompt-row">
    <textarea id="prompt-input" placeholder="Enter your prompt…" rows="2" onkeydown="promptKey(event)"></textarea>
    <button class="btn btn-ghost btn-sm" onclick="clearAll()" title="Clear both responses">✕ Clear</button>
  </div>
  <div style="display:flex;align-items:center;gap:12px">
    <div class="sys-toggle" id="sys-toggle" onclick="toggleSys()">
      <span class="arrow">▶</span> System prompt
    </div>
  </div>
  <div id="sys-area">
    <textarea id="sys-input" placeholder="Optional system prompt…"></textarea>
  </div>
</div>

<!-- ── Panels ───────────────────────────────────────────────── -->
<div id="panels">

  <!-- Panel A -->
  <div class="panel A" id="panel-A">
    <div class="ph">
      <div class="ph-top">
        <span class="slot-label">A</span>
        <select class="model-select" id="sel-A" onchange="onModelChange('A')"></select>
        <span class="model-badge" id="badge-A" style="display:none"></span>
        <button class="load-btn idle" id="load-A" onclick="loadModel('A')">Load</button>
      </div>
      <div>
        <div class="params-toggle" id="pt-A" onclick="toggleParams('A')">
          <span class="arr">▶</span> Parameters
        </div>
        <div class="params-grid" id="pg-A">
          <!-- Temperature -->
          <div class="param">
            <div class="param-label">Temperature <span id="v-temp-A">0.20</span></div>
            <input type="range" id="s-temp-A" min="0" max="2" step="0.05" value="0.2"
              oninput="syncSlider('temp','A')"
              style="background:linear-gradient(to right,#60a5fa 10%,rgba(255,255,255,.1) 10%)">
          </div>
          <!-- Top P -->
          <div class="param">
            <div class="param-label">Top P <span id="v-top_p-A">0.90</span></div>
            <input type="range" id="s-top_p-A" min="0" max="1" step="0.01" value="0.9"
              oninput="syncSlider('top_p','A')"
              style="background:linear-gradient(to right,#60a5fa 90%,rgba(255,255,255,.1) 90%)">
          </div>
          <!-- Max Tokens -->
          <div class="param">
            <div class="param-label">Max Tokens <span id="v-max_tokens-A">512</span></div>
            <input type="range" id="s-max_tokens-A" min="64" max="1024" step="16" value="512"
              oninput="syncSlider('max_tokens','A')"
              style="background:linear-gradient(to right,#60a5fa 44%,rgba(255,255,255,.1) 44%)">
          </div>
          <!-- Repetition Penalty -->
          <div class="param">
            <div class="param-label">Rep Penalty <span id="v-rep-A">1.50</span></div>
            <input type="range" id="s-rep-A" min="1" max="2" step="0.05" value="1.5"
              oninput="syncSlider('rep','A')"
              style="background:linear-gradient(to right,#60a5fa 50%,rgba(255,255,255,.1) 50%)">
          </div>
          <!-- Min Tokens -->
          <div class="param">
            <div class="param-label">Min Tokens <span id="v-min-A">15</span></div>
            <input type="range" id="s-min-A" min="0" max="100" step="1" value="15"
              oninput="syncSlider('min','A')"
              style="background:linear-gradient(to right,#60a5fa 15%,rgba(255,255,255,.1) 15%)">
          </div>
          <!-- N-gram truncate -->
          <div class="param" style="justify-content:flex-end;padding-top:6px">
            <label class="param-check">
              <input type="checkbox" id="ck-ngram-A" checked> 4-gram loop guard
            </label>
          </div>
        </div>
      </div>
    </div>

    <div class="resp-wrap">
      <div class="resp-area empty" id="resp-A" data-placeholder="Load a model, then generate…"></div>
    </div>

    <div class="stats" id="stats-A">
      <div class="stat">Tokens <span class="stat-val" id="st-tok-A">—</span></div>
      <span class="stat-sep">·</span>
      <div class="stat">t/s <span class="stat-val" id="st-tps-A">—</span></div>
      <span class="stat-sep">·</span>
      <div class="stat">Time <span class="stat-val" id="st-el-A">—</span></div>
      <button class="gen-btn" id="gb-A" onclick="generateSlot('A')" disabled>Generate</button>
    </div>
  </div>

  <div class="divider"></div>

  <!-- Panel B -->
  <div class="panel B" id="panel-B">
    <div class="ph">
      <div class="ph-top">
        <span class="slot-label">B</span>
        <select class="model-select" id="sel-B" onchange="onModelChange('B')"></select>
        <span class="model-badge" id="badge-B" style="display:none"></span>
        <button class="load-btn idle" id="load-B" onclick="loadModel('B')">Load</button>
      </div>
      <div>
        <div class="params-toggle" id="pt-B" onclick="toggleParams('B')">
          <span class="arr">▶</span> Parameters
        </div>
        <div class="params-grid" id="pg-B">
          <div class="param">
            <div class="param-label">Temperature <span id="v-temp-B">0.20</span></div>
            <input type="range" id="s-temp-B" min="0" max="2" step="0.05" value="0.2"
              oninput="syncSlider('temp','B')"
              style="background:linear-gradient(to right,#a78bfa 10%,rgba(255,255,255,.1) 10%)">
          </div>
          <div class="param">
            <div class="param-label">Top P <span id="v-top_p-B">0.90</span></div>
            <input type="range" id="s-top_p-B" min="0" max="1" step="0.01" value="0.9"
              oninput="syncSlider('top_p','B')"
              style="background:linear-gradient(to right,#a78bfa 90%,rgba(255,255,255,.1) 90%)">
          </div>
          <div class="param">
            <div class="param-label">Max Tokens <span id="v-max_tokens-B">512</span></div>
            <input type="range" id="s-max_tokens-B" min="64" max="1024" step="16" value="512"
              oninput="syncSlider('max_tokens','B')"
              style="background:linear-gradient(to right,#a78bfa 44%,rgba(255,255,255,.1) 44%)">
          </div>
          <div class="param">
            <div class="param-label">Rep Penalty <span id="v-rep-B">1.50</span></div>
            <input type="range" id="s-rep-B" min="1" max="2" step="0.05" value="1.5"
              oninput="syncSlider('rep','B')"
              style="background:linear-gradient(to right,#a78bfa 50%,rgba(255,255,255,.1) 50%)">
          </div>
          <div class="param">
            <div class="param-label">Min Tokens <span id="v-min-B">15</span></div>
            <input type="range" id="s-min-B" min="0" max="100" step="1" value="15"
              oninput="syncSlider('min','B')"
              style="background:linear-gradient(to right,#a78bfa 15%,rgba(255,255,255,.1) 15%)">
          </div>
          <div class="param" style="justify-content:flex-end;padding-top:6px">
            <label class="param-check">
              <input type="checkbox" id="ck-ngram-B" checked> 4-gram loop guard
            </label>
          </div>
        </div>
      </div>
    </div>

    <div class="resp-wrap">
      <div class="resp-area empty" id="resp-B" data-placeholder="Load a model, then generate…"></div>
    </div>

    <div class="stats" id="stats-B">
      <div class="stat">Tokens <span class="stat-val" id="st-tok-B">—</span></div>
      <span class="stat-sep">·</span>
      <div class="stat">t/s <span class="stat-val" id="st-tps-B">—</span></div>
      <span class="stat-sep">·</span>
      <div class="stat">Time <span class="stat-val" id="st-el-B">—</span></div>
      <button class="gen-btn" id="gb-B" onclick="generateSlot('B')" disabled>Generate</button>
    </div>
  </div>
</div><!-- #panels -->

<div id="toast"></div>

<script>
// ── State ─────────────────────────────────────────────────────────────────────
const loaded = {A: null, B: null};          // currently loaded model id
const generating = {A: false, B: false};
const modelMeta = {};                        // id → registry entry

// ── Boot ──────────────────────────────────────────────────────────────────────
async function boot() {
  const models = await fetch('/api/models').then(r => r.json());
  models.forEach(m => modelMeta[m.id] = m);
  ['A','B'].forEach(s => {
    const sel = document.getElementById(`sel-${s}`);
    models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.id;
      opt.textContent = `${m.label}  (${m.size})`;
      sel.appendChild(opt);
    });
    // Default: A → full_ft, B → base
    sel.value = s === 'A' ? 'full_ft' : 'base';
    refreshSliderTracks(s);
  });
}
boot();

// ── Model change ──────────────────────────────────────────────────────────────
function onModelChange(slot) {
  const id = document.getElementById(`sel-${slot}`).value;
  const isLoaded = loaded[slot] === id;
  const lb = document.getElementById(`load-${slot}`);

  if (isLoaded) {
    setLoadBtn(slot, 'loaded', '✓ Loaded');
  } else {
    setLoadBtn(slot, 'idle', 'Load');
  }

  // Reset response + stats on model change
  resetPanel(slot);
  updateGenerateButtons();
}

function resetPanel(slot) {
  const r = document.getElementById(`resp-${slot}`);
  r.textContent = '';
  r.classList.add('empty');
  setStats(slot, null);
}

// ── Load model ────────────────────────────────────────────────────────────────
async function loadModel(slot) {
  const id = document.getElementById(`sel-${slot}`).value;
  setLoadBtn(slot, 'loading', '');
  try {
    const res = await fetch(`/api/load/${slot}`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({model_id: id}),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Load failed');
    }
    loaded[slot] = id;
    const meta = modelMeta[id];
    setLoadBtn(slot, 'loaded', '✓ Loaded');
    setBadge(slot, meta);
    resetPanel(slot);
    updateGenerateButtons();
    toast(`Model ${meta.label} loaded into ${slot}`);
  } catch (e) {
    setLoadBtn(slot, 'error', '✗ Error');
    toast(e.message, true);
  }
}

function setLoadBtn(slot, state, label) {
  const btn = document.getElementById(`load-${slot}`);
  btn.className = `load-btn ${state}`;
  btn.innerHTML = state === 'loading'
    ? '<span class="spin"></span> Loading…'
    : label;
}

function setBadge(slot, meta) {
  const b = document.getElementById(`badge-${slot}`);
  b.textContent = meta.badge;
  b.style.display = 'inline-block';
  b.style.background = hexAlpha(meta.hue, 0.15);
  b.style.color = meta.hue;
  b.style.border = `1px solid ${hexAlpha(meta.hue, 0.3)}`;
}

// ── Generate ──────────────────────────────────────────────────────────────────
function getParams(slot) {
  return {
    prompt:              document.getElementById('prompt-input').value.trim(),
    system:              document.getElementById('sys-input').value,
    temperature:         +document.getElementById(`s-temp-${slot}`).value,
    top_p:               +document.getElementById(`s-top_p-${slot}`).value,
    max_tokens:          +document.getElementById(`s-max_tokens-${slot}`).value,
    repetition_penalty:  +document.getElementById(`s-rep-${slot}`).value,
    min_tokens:          +document.getElementById(`s-min-${slot}`).value,
    ngram_truncate:       document.getElementById(`ck-ngram-${slot}`).checked,
  };
}

async function generateSlot(slot) {
  const params = getParams(slot);
  if (!params.prompt) { toast('Enter a prompt first', true); return; }
  if (!loaded[slot])  { toast(`Load a model for panel ${slot} first`, true); return; }

  generating[slot] = true;
  updateGenerateButtons();
  document.getElementById(`panel-${slot}`).classList.add('generating');

  const resp = document.getElementById(`resp-${slot}`);
  resp.textContent = '';
  resp.classList.remove('empty');
  const cursor = document.createElement('span');
  cursor.className = 'cursor';
  resp.appendChild(cursor);

  setStats(slot, null, true);

  try {
    const res = await fetch(`/api/generate/${slot}`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(params),
    });

    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    let lastStats = null;
    let clipped = false;

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream:true});
      const lines = buf.split('\n');
      buf = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const d = JSON.parse(line.slice(6));

        if (d.error) {
          resp.insertBefore(document.createTextNode(`\n⚠ ${d.error}`), cursor);
          break;
        }
        if (d.tok !== undefined) {
          resp.insertBefore(document.createTextNode(d.tok), cursor);
          resp.scrollTop = resp.scrollHeight;
          lastStats = d;
        }
        if (d.clip) { clipped = true; }
        if (d.done) {
          lastStats = d;
        }
      }
    }

    // Remove cursor, show final response
    cursor.remove();
    if (lastStats) {
      setStats(slot, lastStats);
      if (clipped) toast(`Panel ${slot}: loop detected — response trimmed`);
    }

  } catch (e) {
    cursor.remove();
    toast(e.message, true);
  }

  generating[slot] = false;
  document.getElementById(`panel-${slot}`).classList.remove('generating');
  updateGenerateButtons();
}

async function generateBoth() {
  const prog = document.getElementById('both-prog');
  prog.classList.add('visible');

  document.getElementById('both-prog-text').textContent = 'Generating A…';
  document.getElementById('dot-a').style.opacity = '1';
  document.getElementById('dot-b').style.opacity = '.3';
  await generateSlot('A');

  document.getElementById('both-prog-text').textContent = 'Generating B…';
  document.getElementById('dot-a').style.opacity = '.3';
  document.getElementById('dot-b').style.opacity = '1';
  await generateSlot('B');

  prog.classList.remove('visible');
}

function updateGenerateButtons() {
  const anyGen = generating.A || generating.B;
  const aReady = !!loaded.A && !anyGen;
  const bReady = !!loaded.B && !anyGen;

  document.getElementById('gb-A').disabled = !aReady;
  document.getElementById('gb-B').disabled = !bReady;
  document.getElementById('gen-both-btn').disabled = !(aReady && bReady);
}

// ── Stats ─────────────────────────────────────────────────────────────────────
function setStats(slot, d, generating=false) {
  document.getElementById(`st-tok-${slot}`).textContent = d ? d.n   : generating ? '…' : '—';
  document.getElementById(`st-tps-${slot}`).textContent = d ? d.tps : generating ? '…' : '—';
  document.getElementById(`st-el-${slot}`).textContent  = d ? d.el+'s' : generating ? '…' : '—';
}

// ── Sliders ───────────────────────────────────────────────────────────────────
const sliderMeta = {
  temp:       {id:'s-temp',      val:'v-temp',      label:(v)=>parseFloat(v).toFixed(2), pct:(v)=>v/2*100},
  top_p:      {id:'s-top_p',     val:'v-top_p',     label:(v)=>parseFloat(v).toFixed(2), pct:(v)=>v*100},
  max_tokens: {id:'s-max_tokens',val:'v-max_tokens', label:(v)=>v,                        pct:(v)=>(v-64)/(1024-64)*100},
  rep:        {id:'s-rep',       val:'v-rep',        label:(v)=>parseFloat(v).toFixed(2), pct:(v)=>(v-1)*100},
  min:        {id:'s-min',       val:'v-min',        label:(v)=>v,                        pct:(v)=>v},
};

function syncSlider(key, slot) {
  const m = sliderMeta[key];
  const el = document.getElementById(`${m.id}-${slot}`);
  const pct = m.pct(+el.value);
  const col = slot === 'A' ? '#60a5fa' : '#a78bfa';
  el.style.background = `linear-gradient(to right,${col} ${pct}%,rgba(255,255,255,.1) ${pct}%)`;
  document.getElementById(`${m.val}-${slot}`).textContent = m.label(el.value);
}

function refreshSliderTracks(slot) {
  Object.keys(sliderMeta).forEach(k => syncSlider(k, slot));
}

// ── Misc ──────────────────────────────────────────────────────────────────────
function toggleSys() {
  const t = document.getElementById('sys-toggle');
  const a = document.getElementById('sys-area');
  t.classList.toggle('open');
  a.classList.toggle('open');
}

function toggleParams(slot) {
  document.getElementById(`pt-${slot}`).classList.toggle('open');
  document.getElementById(`pg-${slot}`).classList.toggle('open');
}

function clearAll() {
  ['A','B'].forEach(s => {
    resetPanel(s);
  });
}

function promptKey(e) {
  // Ctrl/Cmd+Enter triggers generate both
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    if (!document.getElementById('gen-both-btn').disabled) generateBoth();
  }
}

async function randomPrompt() {
  const d = await fetch('/api/random-prompt').then(r => r.json());
  if (d.prompt) document.getElementById('prompt-input').value = d.prompt;
}

function toast(msg, isError=false) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'show' + (isError ? ' error' : '');
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.className = '', 3200);
}

function hexAlpha(hex, alpha) {
  const r = parseInt(hex.slice(1,3),16);
  const g = parseInt(hex.slice(3,5),16);
  const b = parseInt(hex.slice(5,7),16);
  return `rgba(${r},${g},${b},${alpha})`;
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def root():
    return HTML


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    print(f"\n  Playground → http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
