---
title: TinyLlama LoRA Attention Visualization (Task 3)
emoji: 🔎
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.10.0
app_file: app.py
pinned: false
license: apache-2.0
language:
  - en
base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
tags:
  - interpretability
  - attention
  - lora
  - tinyllama
  - token-attribution
---

# Task 3 — Attention Visualization & Token Attribution

Interactive report for **TinyLlama-1.1B-Chat** with a **LoRA rank-16** Alpaca-style adapter.

## What this Space shows

1. **LoRA checkpoint** — PEFT adapter under `adapters/r16/` (merged for analysis offline).
2. **20 sample instructions** — generations with LoRA; **base** and **LoRA** models are run on the **same token ids** for fair comparison.
3. **Attention head clustering** — per-head fingerprints = bilinear-pooled **32×32** attention maps (cosine similarity → agglomerative clustering).
4. **Token attribution** — **attention rollout** across layers; bar charts = rollout **last row** (influence of each input position on the final prediction).
5. **HTML + PNG** — five example **rollout heatmaps** and **attribution** plots; combined `attention_report.html`.
6. **Instruction-following vs completion heads** — empirical split by mean attention from **completion** positions to **instruction** vs **completion** columns (median threshold on `instruction_ratio`).
7. **Base vs LoRA** — per-sample **mean cosine similarity** across all heads between base and LoRA attention tensors on identical sequences.

## Refresh artifacts locally

```bash
cd task3_hf
pip install torch transformers peft scikit-learn matplotlib numpy
python generate_artifacts.py --device cpu --adapter ./adapters/r16
# optional: --quick for 2 samples
# Rebuild only PNGs + attention_report.html from saved JSON (no model): --html-only
# Single-file HTML with inlined images (very large): add --embed-html-images
```

Then commit `results/` and push to this Space.

## Repo layout

| Path | Purpose |
|------|---------|
| `attention_core.py` | Model load, rollout, head features, base–LoRA comparison |
| `generate_artifacts.py` | Batch job: 20 instructions → JSON + PNG + HTML |
| `html_viz.py` | Heatmap / attribution PNGs + combined HTML |
| `samples/instructions.json` | 20 evaluation prompts |
| `results/` | Precomputed outputs (tables, heatmaps, report) |

## Hardware note

Artifact generation is **CPU/GPU** local; this **Space** only serves **static** precomputed files (no model download at runtime).
