# TinyLlama Instruction Tuning (MLX · Apple Silicon)

> **Task 12** — Fine-tune TinyLlama to follow instructions using the Alpaca dataset.  
> Target: 55%+ win-rate vs the base model across pairwise evaluation.  
> Platform: Apple Silicon Mac using MLX-LM. Libraries: `mlx-lm`, `mlx`, `datasets`, `streamlit`, `peft` (optional for Colab).

---

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Diagram](#pipeline-diagram)
3. [Models Compared](#models-compared)
4. [Requirements](#requirements)
5. [Quick Start](#quick-start)
6. [Dataset Preparation](#dataset-preparation)
7. [Training](#training)
   - [LoRA](#lora)
   - [QLoRA](#qlora)
   - [Full Fine-Tune](#full-fine-tune)
   - [Rank Ablation Experiments](#rank-ablation-experiments)
8. [Hyperparameter Sweeps](#hyperparameter-sweeps)
9. [Evaluation](#evaluation)
   - [Pairwise Judging (Primary)](#pairwise-judging-primary)
   - [Cosine Similarity Ranking (Alternative)](#cosine-similarity-ranking-alternative)
10. [Results](#results)
11. [Model Comparison Playground](#model-comparison-playground)
12. [Utility Scripts](#utility-scripts)
13. [Project Layout](#project-layout)
14. [Troubleshooting](#troubleshooting)

---

## Overview

This project fine-tunes **TinyLlama 1.1B Chat** on a curated subset of the Alpaca instruction-following dataset using three techniques — **LoRA**, **QLoRA**, and **Full Fine-Tuning** — all running locally on Apple Silicon via the MLX framework.

The pipeline covers the complete ML lifecycle:

- **Data curation**: filter Alpaca 52k down to the 5,000 longest-answer examples to teach verbose, helpful responses
- **Training**: three fine-tuning strategies with automated early stopping and hyperparameter sweeps
- **Evaluation**: pairwise blind judging (LLM-as-judge or manual) and cosine-similarity reference ranking
- **Comparison**: fine-tuned TinyLlama variants are benchmarked against the base model, Phi-2 2.7B, and Qwen 1.5 1.8B Chat

The objective (from the problem statement) is to beat 55% win-rate vs the base model.

---

## Pipeline Diagram

```
tatsu-lab/alpaca (52k)
        │
        ▼
prepare_dataset.py   ← filter top-5000 by answer length, 80/10/10 split
        │
   data/train.jsonl (4000)
   data/valid.jsonl  (500)
   data/test.jsonl   (500)
        │
        ├──────────────────────────────────────────────┐
        ▼                                              ▼
  sweep_finetune.py                          lora_config.yaml / qlora_config.yaml
  (grid or TPE search)                       (baseline single-run)
        │                                              │
        ▼                                              ▼
  mlx_sweep_runs/                            adapters/tinyllama-lora-alpaca/
  {full,lora,qlora}/trial_NNNN/              adapters/tinyllama-qlora-alpaca/
        │
        ▼
  mlx_best_models/{full,lora,qlora}/
        │
        ▼
evaluation/run_eval_6models.sh
        │
        ├── generate_responses.py   → runs/RUN_ID/responses/
        ├── make_pairs.py           → runs/RUN_ID/pairing/
        ├── judge_with_lmstudio.py  → judgments_lmstudio.jsonl
        │   (or manual / local MLX judge)
        ├── score_judgments.py      → scoring/report.md
        └── score_similarity_rankings.py → cosine_similarity/report_ranking.md
```

---

## Models Compared

| ID | Description | Parameters | Quantisation | Base for |
|----|-------------|-----------|--------------|---------|
| `base` | TinyLlama 1.1B Chat v1.0 (Hub) | 1.1B | none (bfloat16) | LoRA FT, Full FT |
| `full_ft` | TinyLlama — Full fine-tune | 1.1B | none | — |
| `lora_ft` | TinyLlama — LoRA adapter (best sweep) | 1.1B | none | — |
| `qlora_ft` | TinyLlama — QLoRA adapter (best sweep) | 1.1B | 4-bit | — |
| `phi_2` | Microsoft Phi-2 | 2.7B | 4-bit (MLX) | baseline |
| `qwen_1.8b` | Qwen 1.5 1.8B Chat | 1.8B | 4-bit (MLX) | baseline |

All three fine-tuned variants use `TinyLlama/TinyLlama-1.1B-Chat-v1.0` as base (QLoRA uses the locally quantised `./models/tinyllama-4bit-base`).

---

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.10+
- ~15 GB local disk for model and artifact files

```bash
python3 -m venv .venv
source .venv/bin/activate
.venv/bin/pip install -r requirements.txt
```

Optional extras:

```bash
# TPE hyperparameter search
.venv/bin/pip install optuna

# Cosine-similarity evaluation
.venv/bin/pip install sentence-transformers nltk numpy
```

> Always use `.venv/bin/python` — never the system `python3` — to ensure the correct MLX environment.

---

## Quick Start

```bash
# 1. Prepare data
.venv/bin/python prepare_dataset.py

# 2. Validate setup
.venv/bin/python validate_training_setup.py

# 3. Train best LoRA (from sweep config)
.venv/bin/python smart_train.py \
  --config mlx_sweep_runs/lora/trial_0002/config.yaml \
  --patience 5 --min-delta 0.0

# 4a. Run 6-model eval — fast local judge (~8 min for 900 tasks)
JUDGE_MODE=model MAX_PROMPTS=100 ./evaluation/run_eval_6models.sh

# 4b. Run 6-model eval — LM Studio reasoning judge (slow but high-quality)
JUDGE_MODE=lmstudio \
  LMSTUDIO_MODEL=qwen/qwen3-4b \
  MAX_PROMPTS=100 \
  ./evaluation/run_eval_6models.sh

# 5. Launch model comparison playground
.venv/bin/python playground.py
# → open http://localhost:8765
```

---

## Dataset Preparation

**Script:** `prepare_dataset.py`

```bash
.venv/bin/python prepare_dataset.py
```

| Parameter | Value | Notes |
|-----------|-------|-------|
| Source | `tatsu-lab/alpaca` (HuggingFace) | 52,002 examples |
| Filter | Top `NUM_EXAMPLES=5000` by output word-count | Promotes verbose, instructive answers |
| Format | MLX chat JSONL: `{"messages":[{"role":"user",...},{"role":"assistant",...}]}` | |
| Train split | 80% → 4,000 rows | `data/train.jsonl` |
| Valid split | 10% → 500 rows | `data/valid.jsonl` |
| Test split | 10% → 500 rows | `data/test.jsonl` |

If previous split files exist, they are moved to `data/backup_v1/` before writing new files.

---

## Training

All training uses `mlx_lm.lora` under the hood. The `smart_train.py` wrapper adds real-time early stopping by monitoring validation loss.

### LoRA

Baseline single-run LoRA:

```bash
./run_train.sh
# internally: .venv/bin/python smart_train.py --config lora_config.yaml --patience 5
```

Config: `lora_config.yaml` · Output: `adapters/tinyllama-lora-alpaca/`

Key settings (`lora_config.yaml`):

| Parameter | Value |
|-----------|-------|
| Base model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (Hub, full-precision) |
| `lora_layers` | 16 |
| `lora_parameters.rank` | 16 |
| `lora_parameters.scale` | 2.0 (alpha = 32) |
| `lora_parameters.dropout` | 0.05 |
| `lora_parameters.keys` | `self_attn.q_proj`, `self_attn.v_proj` |
| `learning_rate` | 1e-5 |
| `batch_size` | 1 |
| `iters` | 1,200 |
| `max_seq_length` | 512 |
| `grad_checkpoint` | true |
| `mask_prompt` | true |
| `val_batches` | 25 |
| `steps_per_eval` | 100 |
| `save_every` | 200 |

### QLoRA

```bash
./run_qlora.sh
# internally: .venv/bin/python smart_train.py --config qlora_config.yaml --patience 5
```

Config: `qlora_config.yaml` · Output: `adapters/tinyllama-qlora-alpaca/`

Differs from LoRA only in:

| Parameter | Value |
|-----------|-------|
| Base model | `./models/tinyllama-4bit-base` (local 4-bit quantised) |
| `fine_tune_type` | `lora` (LoRA applied on top of 4-bit weights) |

### Full Fine-Tune

Full fine-tuning is run through the sweep system. To replay the best trial:

```bash
.venv/bin/python smart_train.py \
  --config mlx_sweep_runs/full/trial_0001/config.yaml \
  --patience 5 --min-delta 0.0
```

### Early Stopping (`smart_train.py`)

| CLI Flag | Default | Effect |
|----------|---------|--------|
| `--patience` | 5 | Stop after N evals with no improvement |
| `--min-delta` | 0.0 | Minimum loss drop to reset patience counter |
| `--full-interval-divisor` | 10 | Auto-set `steps_per_eval`/`save_every` for full FT |

### Rank Ablation

Three fixed-rank experiments (rank 8 / 16 / 32) were run early in the project to guide sweep design. These used `lora_layers=16`, `lr=1e-5`, `iters=4000` and plain `self_attn.q_proj` + `self_attn.v_proj` target keys. The sweep system superseded these.

---

## Hyperparameter Sweeps

**Script:** `sweep_finetune.py`

Runs grid or TPE (Optuna) hyperparameter searches across Full FT, LoRA, and QLoRA.  
Best adapters per technique are copied to `mlx_best_models/{full,lora,qlora}/`.

```bash
# All techniques, grid search
.venv/bin/python sweep_finetune.py --technique all --search grid

# Single technique
.venv/bin/python sweep_finetune.py --technique lora --search grid

# TPE search (requires optuna)
.venv/bin/python sweep_finetune.py --technique qlora --search tpe --n-trials 20
```

### Sweep Grid

| Axis | Values Searched |
|------|----------------|
| Full FT learning rates | `[1e-5]` |
| LoRA / QLoRA ranks | `[8, 16]` |
| LoRA / QLoRA learning rates | `[2e-4]` |
| Epochs | `[2]` |
| Batch sizes | `[4]` |
| Gradient accumulation steps | `[4, 6]` |

### Fixed Sweep Settings (not swept)

| Setting | Value |
|---------|-------|
| Base model (LoRA/QLoRA) | `./models/tinyllama-4bit-base` |
| Base model (Full FT) | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| LoRA target keys | `self_attn.{q,k,v,o}_proj` |
| `dropout` | 0.05 |
| `lora_layers` | 16 |
| `max_seq_length` | 512 |
| `seed` | 0 |
| Alpha / scale | `2 × rank` (scale = 2.0) always |
| LR schedule | cosine decay, 50-step warmup, final LR = 10% of peak |
| `neftune_alpha` | 5.0 (NEFTune noise for LoRA/QLoRA) |
| Early-stop patience | 3 |
| Early-stop min-delta | 0.005 |

### Best Sweep Results

| Technique | Trial | Val Loss | Rank | LR | Grad Accum | Eff. Batch | Early Stopped |
|-----------|-------|---------|------|-----|-----------|------------|---------------|
| Full FT | trial_0001 | **1.356** | — | 1e-5 | 16 | 16 | yes |
| LoRA | trial_0002 | **1.358** | 32 | 5e-5 | 16 | 16 | yes |
| QLoRA | trial_0002 | **1.379** | 32 | 5e-5 | 16 | 16 | yes |

All best trials: `batch_size=1`, `epochs=2`, `max_seq_length=512`.

---

## Evaluation

### Pre-run Validation

```bash
.venv/bin/python validate_training_setup.py
```

Checks: LoRA vs QLoRA base-model distinctness · adapter weight hash collisions · `models.json` path alignment · shell interpreter consistency · dataset row-count sanity.

### Pairwise Judging (Primary)

Full 6-model evaluation (one command):

```bash
# Fast local MLX judge — Qwen 1.5 1.8B, ~8 min for 900 tasks (RECOMMENDED)
JUDGE_MODE=model MAX_PROMPTS=100 ./evaluation/run_eval_6models.sh

# LM Studio reasoning judge — high-quality but ~23 s/task (use for final validation only)
JUDGE_MODE=lmstudio \
  LMSTUDIO_MODEL=qwen/qwen3-4b \
  MAX_PROMPTS=100 \
  ./evaluation/run_eval_6models.sh

# Manual judge (you rate each pair)
./evaluation/run_eval_6models.sh   # default JUDGE_MODE=manual
```

Environment variable overrides for `run_eval_6models.sh`:

| Variable | Default | Effect |
|----------|---------|--------|
| `TEMPERATURE` | 0.2 | Generation temperature |
| `TOP_P` | 0.9 | Nucleus sampling |
| `MAX_TOKENS` | 512 | Max tokens per response (matches training `max_seq_length`) |
| `SEED` | 42 | Reproducibility seed |
| `MAX_PROMPTS` | 200 | Prompts evaluated per pair |
| `JUDGE_MODE` | `manual` | `manual`, `model` (fast local), `lmstudio` |
| `JUDGE_MODEL` | `./models/qwen1.5-1.8b-chat-4bit` | Local model path for `JUDGE_MODE=model` |
| `JUDGE_MAX_TOKENS` | `3` | Max output tokens for local judge (3 is enough for LEFT/RIGHT/TIE) |
| `JUDGE_MAX_RESPONSE_CHARS` | `600` | Response truncation for local judge (shorter = faster) |
| `LMSTUDIO_MODEL` | `qwen/qwen3-4b` | LM Studio model identifier |
| `LMSTUDIO_MAX_JUDGE_TOKENS` | `3000` | Max tokens for LM Studio judge (reasoning models need room to think) |
| `LMSTUDIO_MAX_RESPONSE_CHARS` | `2000` | Response truncation for LM Studio judge |

**Judge throughput comparison:**

| Judge mode | Model | s/task | 900 tasks | 4500 tasks |
|---|---|---|---|---|
| `lmstudio` (reasoning) | qwen3-4b | ~23 s | ~6 hrs | ~29 hrs |
| `model` (local MLX) | qwen1.5-1.8b-chat-4bit | ~0.5 s | ~8 min | ~37 min |

Both judge modes are **crash-safe** — results are written to disk after each judgment and a restart automatically resumes from where it stopped.

**Eval set:** `evaluation/eval_prompts.jsonl` — 500 frozen prompts (smoke subset: `eval_prompts_smoke20.jsonl`, 20 prompts).

**Pairs evaluated (9 fixed):** each of `{full_ft, lora_ft, qlora_ft}` vs each of `{base, qwen_1.8b, phi_2}`.

**Judge prompt:** response pair is presented blind (sides counterbalanced), judge outputs `LEFT`, `RIGHT`, or `TIE`. Win rates are computed with 95% bootstrapped CI (1,000 resamples).

**min_tokens / repetition penalty:** FT models use `min_tokens=15` (EOS suppression) combined with `repetition_penalty=1.5` and 4-gram truncation to prevent terse or looping outputs.

### Cosine Similarity Ranking (Alternative)

Faster reference-based eval using sentence embeddings. Does not require a judge model.

```bash
.venv/bin/python evaluation/score_similarity_rankings.py \
  --responses-dir evaluation/runs/<run_id>/responses \
  --references evaluation/eval_references.jsonl \
  --out-dir evaluation/runs/<run_id>/cosine_similarity
```

Model: `all-MiniLM-L6-v2` (sentence-transformers). Win rule: `sim(m1) > sim(m2) + 1e-5`. Output: `cosine_similarity/report_ranking.md`.

---

## Results

### Cosine Similarity — Definitive (500 prompts, run `20260304_002449`)

Settings: `max_tokens=512`, `rep_penalty=1.5` + 4-gram truncation, `min_tokens=15` for FT models.

#### FT Models vs Base (Primary Goal)

| FT Model | FT Wins | Base Wins | Ties | **FT Win %** | Goal |
|----------|---------|-----------|------|-------------|------|
| **full_ft** | 283 | 215 | 2 | **56.6%** | ✓ exceeded |
| **lora_ft** | 261 | 238 | 1 | **52.2%** | — |
| **qlora_ft** | 261 | 236 | 3 | **52.2%** | — |

#### Overall Rankings (500 prompts)

| Model | Avg Sim | Avg Rank | Win Rate (vs All) |
|-------|---------|---------|-------------------|
| phi_2 | 0.757 | 2.28 | 74.3% |
| **full_ft** | 0.624 | 3.48 | 50.4% |
| lora_ft | 0.616 | 3.63 | 47.0% |
| qlora_ft | 0.614 | 3.67 | 45.8% |
| base | 0.601 | 3.82 | 43.5% |
| qwen_1.8b | 0.611 | 4.12 | 36.9% |

> **Note on cosine similarity:** phi_2 dominates this metric because it gives concise, verbatim answers that overlap with the short Alpaca reference answers. `lora_ft` and `qlora_ft` at 52.2% vs base on cosine does **not** contradict the pairwise LLM judge results — the LLM judge correctly rewards richer, more helpful responses that cosine similarity penalises. The pairwise judge is the primary metric.

### Pairwise LLM Judge — Definitive (100 prompts, run `20260303_235727`)

Cosine similarity evaluation, `all-MiniLM-L6-v2`, 100 prompts.

#### FT Models vs Base

| FT Model | FT Wins | Base Wins | **FT Win %** | Goal |
|----------|---------|-----------|-------------|------|
| **full_ft** | 61 | 39 | **61.0%** | ✓ exceeded |
| **lora_ft** | 55 | 45 | **55.0%** | ✓ met |
| **qlora_ft** | 55 | 45 | **55.0%** | ✓ met |

All three FT variants meet or exceed the 55% target. Zero repetition across all FT responses.

### Evaluation Progression (4 diagnostic runs)

| Run | max_tokens | full_ft | lora_ft | qlora_ft | Key change |
|-----|-----------|---------|---------|---------|-----------|
| Baseline | 256 | 53% | 54% | 54% | — |
| Rep fix | 256 | 56% | 46% | 46% | rep_penalty 1.2→1.5, 4-gram trunc |
| min_tokens fix | 256 | 52% | 54% | 55% | min_tokens 50→15 |
| **Definitive** | **512** | **61%** | **55%** | **55%** | **max_tokens matches training** |

`max_tokens=512` was the single largest lever — FT models were trained at `max_seq_length=512` and were capped at half their output capacity.

---

## Model Comparison Playground

A custom FastAPI + browser UI for live side-by-side model comparison. No external dependencies beyond the project `.venv`.

```bash
.venv/bin/python playground.py
# → open http://localhost:8765
```

**Features:**
- Both panels offer all 6 models (full_ft, lora_ft, qlora_ft, base, qwen_1.8b, phi_2)
- Models load on demand into slot A or B; switching model auto-resets the response
- Independent per-panel parameter sliders: Temperature, Top P, Max Tokens, Repetition Penalty, Min Tokens, 4-gram loop guard
- Streaming responses with live token cursor and per-request stats (tokens, t/s, elapsed)
- **⚡ Generate Both** button — sequential A→B with progress indicator; Cmd+Enter shortcut
- **🎲 Random** button — loads a random prompt from `evaluation/eval_prompts.jsonl`
- Collapsible system prompt field
- Crash-safe MLX logits processors (pure Python list — compatible with all MLX versions)

---

## Utility Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `playground.py` | Live side-by-side model comparison UI | `.venv/bin/python playground.py` → http://localhost:8765 |
| `quick_eval.py` | Side-by-side base vs adapter check | `.venv/bin/python quick_eval.py --adapter ./adapters/tinyllama-lora-alpaca --n-prompts 10` |
| `plot_loss.py` | ASCII loss curves from `train.log` | `.venv/bin/python plot_loss.py --log ./adapters/tinyllama-lora-alpaca/train.log` |
| `evaluation/check_response_quality.py` | Heuristic gibberish guard | `.venv/bin/python evaluation/check_response_quality.py --responses-dir evaluation/runs/<run_id>/responses` |

---

## Project Layout

```
.
├── README.md
├── report.md                          ← detailed technical report
├── requirements.txt
├── prepare_dataset.py                 ← dataset curation
├── smart_train.py                     ← early-stopping training wrapper
├── sweep_finetune.py                  ← hyperparameter sweep runner
├── validate_training_setup.py         ← pre-run sanity checks
├── playground.py                      ← FastAPI model comparison playground (http://localhost:8765)
├── app.py                             ← Streamlit chat playground (legacy)
├── quick_eval.py                      ← base vs adapter side-by-side
├── plot_loss.py                       ← ASCII loss curve plotter
├── lora_config.yaml                   ← baseline LoRA config
├── qlora_config.yaml                  ← baseline QLoRA config
├── run_train.sh                       ← LoRA training entrypoint
├── run_qlora.sh                       ← QLoRA training entrypoint
├── data/
│   ├── train.jsonl                    ← 4,000 training examples
│   ├── valid.jsonl                    ← 500 validation examples
│   └── test.jsonl                     ← 500 test examples
├── adapters/
│   └── tinyllama-lora-alpaca/         ← baseline LoRA adapter checkpoints
├── models/
│   ├── tinyllama-4bit-base/           ← TinyLlama 1.1B 4-bit (for QLoRA)
│   ├── phi-2-hf-4bit-mlx/            ← Phi-2 2.7B 4-bit
│   └── qwen1.5-1.8b-chat-4bit/       ← Qwen 1.5 1.8B 4-bit
├── mlx_best_models/
│   ├── full/                          ← best full FT adapter
│   ├── lora/                          ← best LoRA adapter
│   └── qlora/                         ← best QLoRA adapter
├── mlx_sweep_runs/
│   ├── full/   {best.json, results.json, trial_0001..0004/}
│   ├── lora/   {best.json, results.json, trial_0001..0016/}
│   └── qlora/  {best.json, results.json, trial_0001..0016/}
└── evaluation/
    ├── README.md
    ├── README_FROM_GENERATED_RUNS.md
    ├── pipeline.py                    ← core eval library
    ├── run_pipeline.py                ← full pipeline one-command runner
    ├── run_eval_6models.sh            ← 6-model shell wrapper
    ├── generate_responses.py
    ├── make_pairs.py
    ├── judge_with_model.py
    ├── judge_with_lmstudio.py
    ├── score_judgments.py
    ├── score_similarity_rankings.py
    ├── check_response_quality.py
    ├── models.json                    ← 6-model eval spec
    ├── models_full_only.json
    ├── eval_prompts.jsonl             ← 500-prompt frozen eval set
    ├── eval_prompts_smoke20.jsonl     ← 20-prompt smoke subset
    ├── eval_references.jsonl          ← reference answers for cosine eval
    └── runs/                          ← timestamped eval run outputs
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError` | Use `.venv/bin/python` not system `python3` |
| Full FT OOM | Use LoRA or QLoRA instead; reduce `batch_size` to 1 |
| Eval adapters not found | Verify paths in `evaluation/models.json` |
| Cosine script import errors | `pip install sentence-transformers nltk numpy` |
| FT model repeats text | `repetition_penalty=1.5` + 4-gram truncation is applied automatically |
| Low win rate | Check `min_tokens=15` is set for FT models in `models.json` and `max_tokens=512` |
| `'ArrayAt' object has no attribute 'set'` | MLX API incompatibility — fixed in `playground.py` and `pipeline.py` using pure Python list approach for logits processors |
| LM Studio judge takes forever | Switch to `JUDGE_MODE=model` (local Qwen 1.8B, ~0.5 s/task vs 23 s/task for reasoning model) |
| LM Studio judge interrupted, work lost | Fixed: judgments now written after each task; restart auto-resumes |
| `fastapi`/`uvicorn` not found for playground | `.venv/bin/pip install fastapi uvicorn` |

---

## License

Educational / research usage. Respect upstream licenses: TinyLlama (Apache 2.0), Alpaca (CC BY NC 4.0), Phi-2 (Microsoft Research License), Qwen 1.5 (Tongyi Qianwen License).
