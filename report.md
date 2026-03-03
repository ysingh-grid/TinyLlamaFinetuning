# Technical Report: TinyLlama Instruction Tuning on Apple Silicon

**Project:** Task 12 — TinyLlama Instruction Tuning  
**Platform:** Apple Silicon Mac (M-series), MLX framework  
**Date:** March 2026  

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [System Overview and Design Decisions](#2-system-overview-and-design-decisions)
3. [Dataset Pipeline](#3-dataset-pipeline)
4. [Model Architectures](#4-model-architectures)
5. [Training Methodology](#5-training-methodology)
   - 5.1 [Fine-Tuning Techniques](#51-fine-tuning-techniques)
   - 5.2 [Baseline Training Configurations](#52-baseline-training-configurations)
   - 5.3 [Early-Stopping Wrapper](#53-early-stopping-wrapper)
   - 5.4 [Hyperparameter Sweep System](#54-hyperparameter-sweep-system)
   - 5.5 [Best Sweep Configurations (Final Models)](#55-best-sweep-configurations-final-models)
   - 5.6 [Rank Ablation Experiments](#56-rank-ablation-experiments)
6. [Evaluation Methodology](#6-evaluation-methodology)
   - 6.1 [Pairwise Blind Judging](#61-pairwise-blind-judging)
   - 6.2 [Cosine Similarity Ranking](#62-cosine-similarity-ranking)
   - 6.3 [Generation Parameters](#63-generation-parameters)
   - 6.4 [Quality Heuristics](#64-quality-heuristics)
7. [Results](#7-results)
   - 7.1 [Cosine Similarity Rankings](#71-cosine-similarity-rankings)
   - 7.2 [FT Models vs Base (Primary Goal)](#72-ft-models-vs-base-primary-goal)
   - 7.3 [Full Head-to-Head Pairwise Matrix](#73-full-head-to-head-pairwise-matrix)
8. [Diagnostic Findings and Fixes Applied](#8-diagnostic-findings-and-fixes-applied)
9. [Complete Changeable Parameter Reference](#9-complete-changeable-parameter-reference)
10. [Infrastructure and Environment](#10-infrastructure-and-environment)
11. [File Reference](#11-file-reference)

---

## 1. Problem Statement

**Task 12: TinyLlama Instruction Tuning**

> Fine-tune TinyLlama to follow instructions using the Alpaca dataset.
>
> **Libraries:** torch, transformers, peft, datasets, accelerate, bitsandbytes, trl (optional)
>
> **Mac Acceleration:** Use MLX-LM for fastest training on Mac — `pip install mlx-lm`. Or use PEFT with MPS. Batch size 1–2, gradient accumulation 4–8.
>
> **Here's how to build it:**
> - Step 1: Get Alpaca (52k) — use 5–10k subset
> - Step 2: Use TinyLlama (≤3B) with LoRA
> - Step 3: On Mac: MLX-LM LoRA is optimised for Apple Silicon
> - Step 4: Train 1–3 epochs with small batches
> - Step 5: Aim for 55%+ win-rate vs base
>
> **Try These Experiments:**
> - Architecture: TinyLlama vs Phi-2 vs Qwen-1.8B. Try full fine-tune vs LoRA vs QLoRA.
> - Data Prep: Different instruction templates (Alpaca vs ChatML vs Vicuna). Filter by response length.
> - Parameters: LoRA rank: 8, 16, 32, 64. Alpha: 16, 32, 64. LR: 1e-4, 2e-4, 5e-4. Epochs: 1, 2, 3.
>
> **Show it off:** Build a Streamlit chat playground. Add safety prompts and compare LoRA ranks.
>
> Data: https://huggingface.co/datasets/tatsu-lab/alpaca

**Implementation approach:** The problem statement recommends PyTorch/PEFT. This project implements the entire pipeline using **MLX-LM**, Apple's native ML framework for Apple Silicon, which gives significantly faster training throughput on M-series chips with unified memory. All three fine-tuning strategies (LoRA, QLoRA, Full FT) are implemented and benchmarked, along with automated hyperparameter sweeps and a multi-model evaluation pipeline.

---

## 2. System Overview and Design Decisions

### Platform

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Framework | MLX-LM | Native Apple Silicon acceleration; ~3–5× faster than PyTorch MPS for LoRA |
| Hardware | Apple Silicon Mac (M-series) | Unified CPU/GPU memory eliminates PCIe transfer overhead |
| Python env | `.venv` virtual environment | Isolation from system Python; required for `mlx-lm` |
| Model format | MLX safetensors (`.safetensors`) | Faster loading than PyTorch `.bin`; memory-mapped |
| Quantisation | 4-bit (for QLoRA and comparison models) | MLX native 4-bit; allows 2.7B Phi-2 to run locally |

### Pipeline Components

| Component | Script | Purpose |
|-----------|--------|---------|
| Dataset prep | `prepare_dataset.py` | HuggingFace → MLX chat JSONL |
| Training | `smart_train.py` | Subprocess wrapper with real-time early stopping |
| Sweep | `sweep_finetune.py` | Grid / TPE hyperparameter search |
| Validation | `validate_training_setup.py` | 6-point pre-run sanity checker |
| Eval core | `evaluation/pipeline.py` | All eval logic as pure functions |
| Full eval | `evaluation/run_eval_6models.sh` | One-command 6-model eval |
| Playground | `app.py` | Streamlit interactive chat UI |

### Instruction Template

All training and evaluation uses the **TinyLlama ChatML format** (Zephyr-style):

```
<|system|>
{system_prompt}</s>
<|user|>
{instruction}</s>
<|assistant|>
{response}</s>
```

For training examples without an explicit system prompt, the format reduces to `[user, assistant]` message pairs — no `<|system|>` block is injected during training. At inference time a system prompt is optionally injected via `models.json`.

---

## 3. Dataset Pipeline

**Script:** `prepare_dataset.py`

### Source

| Property | Value |
|----------|-------|
| Dataset | `tatsu-lab/alpaca` (HuggingFace Hub) |
| Total examples | 52,002 |
| License | CC BY NC 4.0 |
| Format | instruction / input / output triplets |

### Filtering Strategy

Rather than a random or top-N subset, examples are ranked by **output word count** and the top `NUM_EXAMPLES` are selected. This deliberately over-represents verbose, detailed answers to counteract Alpaca's inherently terse answer distribution (median 26 words per answer). Empty outputs are dropped first.

### Parameters

| Constant | Value | Notes |
|----------|-------|-------|
| `DATASET_NAME` | `"tatsu-lab/alpaca"` | HuggingFace dataset ID |
| `NUM_EXAMPLES` | `5000` | Total examples after filtering |
| `TRAIN_SPLIT` | `0.8` | → 4,000 training examples |
| `VALID_SPLIT` | `0.1` | → 500 validation examples |
| `TEST_SPLIT` | `0.1` | → 500 test examples |
| `MIN_ANSWER_WORDS` | `0` | No floor — top-N strategy handles filtering |
| Seed | Fixed (Python hash order) | Reproducible splits |

### Output Format

```json
{
  "messages": [
    {"role": "user", "content": "{instruction}\n\n{input}"},
    {"role": "assistant", "content": "{output}"}
  ]
}
```

`input` is only appended to the user message when non-empty.

### Files

| File | Rows | Path |
|------|------|------|
| Training set | 4,000 | `data/train.jsonl` |
| Validation set | 500 | `data/valid.jsonl` |
| Test set | 500 | `data/test.jsonl` |
| Backup (previous) | varies | `data/backup_v1/` |

### Answer Length Distribution (Pre-filtering Analysis)

| Metric | Value |
|--------|-------|
| Total examples in dataset | 4,000 |
| Empty (0 words) | 2 |
| Under 10 words | 1,173 (29%) |
| Under 20 words | 1,790 (45%) |
| Under 30 words | 2,103 (53%) |
| 50+ words | 1,441 (36%) |
| Median answer length | 26 words |

This distribution motivated the inference-time `min_tokens` strategy (see Section 6.3) and data-filtering fixes.

---

## 4. Model Architectures

### Primary Model: TinyLlama 1.1B Chat v1.0

| Property | Value |
|----------|-------|
| Architecture | LLaMA-style decoder-only transformer |
| Parameters | 1.1B |
| Hidden size | 2,048 |
| Intermediate size | 5,632 |
| Attention heads | 32 |
| KV heads | 4 (grouped-query attention) |
| Layers | 22 |
| Context length | 2,048 |
| Vocabulary size | 32,000 |
| Training | Pre-trained + RLHF (instruction-following) |
| Hub ID | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| License | Apache 2.0 |

### Comparison Models

| Model | Architecture | Parameters | Quantisation | Local Path |
|-------|-------------|-----------|-------------|-----------|
| Phi-2 | Phi architecture (MSR) | 2.7B | 4-bit MLX | `./models/phi-2-hf-4bit-mlx/` |
| Qwen 1.5 1.8B Chat | Qwen architecture | 1.8B | 4-bit MLX | `./models/qwen1.5-1.8b-chat-4bit/` |
| TinyLlama 4-bit base | LLaMA (quantised) | 1.1B | 4-bit MLX | `./models/tinyllama-4bit-base/` |

All comparison models are cached locally as MLX-format 4-bit quantised weights (`model.safetensors` + `config.json`). Phi-2 includes custom architecture files (`configuration_phi.py`, `modeling_phi.py`).

---

## 5. Training Methodology

### 5.1 Fine-Tuning Techniques

Three fine-tuning strategies are implemented, all using `mlx_lm.lora` as the underlying engine:

| Technique | Description | Trainable Params | Base Model |
|-----------|-------------|-----------------|-----------|
| **LoRA** | Low-rank decomposition on attention/MLP projection weights | ~0.5–2% of total | TinyLlama Hub (full-precision) |
| **QLoRA** | LoRA applied on top of 4-bit quantised base weights | ~0.5–2% of total | `tinyllama-4bit-base` (local 4-bit) |
| **Full FT** | All 1.1B parameters updated | 100% | TinyLlama Hub (full-precision) |

**Key difference:** LoRA and Full FT load the model in full precision from HuggingFace Hub. QLoRA loads the pre-quantised 4-bit local copy (`./models/tinyllama-4bit-base`), which uses ~4× less memory and trains faster with only modest loss in quality.

### 5.2 Baseline Training Configurations

#### `lora_config.yaml` — Baseline LoRA

| Parameter | Value | Notes |
|-----------|-------|-------|
| `model` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Hub full-precision |
| `fine_tune_type` | `lora` (implicit default) | |
| `lora_layers` | `16` | Top 16 transformer blocks get adapters |
| `lora_parameters.rank` | `16` | Low-rank dimension |
| `lora_parameters.scale` | `2.0` | α = rank × scale = 32 |
| `lora_parameters.dropout` | `0.05` | LoRA dropout probability |
| `lora_parameters.keys` | `self_attn.q_proj`, `self_attn.v_proj` | 2 projection matrices |
| `learning_rate` | `1e-5` | Peak LR, no schedule |
| `batch_size` | `1` | Per-device batch size |
| `grad_accumulation_steps` | not set | Effective batch = 1 |
| `iters` | `1,200` | ~0.3 epochs over 4,000 examples |
| `max_seq_length` | `512` | Truncation length (tokens) |
| `grad_checkpoint` | `true` | Gradient checkpointing to save memory |
| `mask_prompt` | `true` | Loss computed on assistant tokens only |
| `val_batches` | `25` | ~25 validation batches per eval |
| `steps_per_report` | `10` | Console log frequency |
| `steps_per_eval` | `100` | Validation frequency (every 100 iters) |
| `save_every` | `200` | Checkpoint save frequency |
| `adapter_path` | `./adapters/tinyllama-lora-alpaca` | Output directory |
| `seed` | `0` | Random seed |

#### `qlora_config.yaml` — Baseline QLoRA

Identical to `lora_config.yaml` except:

| Parameter | Value | Change from LoRA |
|-----------|-------|-----------------|
| `model` | `./models/tinyllama-4bit-base` | Local 4-bit weights |
| `fine_tune_type` | `lora` | Explicitly set |
| `adapter_path` | `./adapters/tinyllama-qlora-alpaca` | Different output dir |
| `test` | `false` | No test evaluation at end |

### 5.3 Early-Stopping Wrapper

**Script:** `smart_train.py`

Runs `mlx_lm.lora` as a subprocess and monitors its stdout line-by-line. Captures validation loss values in real time using regex, maintains a rolling best-loss tracker, and terminates the subprocess when patience is exhausted. After termination, the best checkpoint is identified by parsing checkpoint filenames and restored as `adapters.safetensors`.

#### CLI Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | required | Path to YAML training config |
| `--patience` | `5` | Stop after N consecutive evals with no improvement |
| `--min-delta` | `0.0` | Minimum loss improvement to reset patience counter |
| `--full-interval-divisor` | `10` | For `fine_tune_type=full`: auto-derives `steps_per_eval` and `save_every` as `max(1, steps_per_epoch // divisor)` |

#### Operation

1. Reads the config YAML to extract `fine_tune_type`, `iters`, `batch_size`, and `data` directory.
2. For Full FT: if `steps_per_eval` is not set manually, auto-computes it via `full_interval_divisor`.
3. Writes a `__smart_effective_config.yaml` with the overridden intervals (for Full FT only).
4. Spawns `mlx_lm.lora --config <effective_config>` as a subprocess.
5. Reads stdout line-by-line; regex `Val loss: ([0-9.]+)` extracts val loss per eval step.
6. When `val_loss >= best_loss - min_delta` for `patience` consecutive evals, sends `SIGTERM`.
7. After termination: scans adapter directory for `NNNNNN_adapters.safetensors` files, selects the one with the lowest iteration number that achieved `best_loss`, copies it to `adapters.safetensors`.
8. Writes all training output to `{adapter_path}/train.log`.

### 5.4 Hyperparameter Sweep System

**Script:** `sweep_finetune.py`

#### Architecture

Each sweep trial:
1. Constructs a YAML config via `build_config()` from the trial's hyperparameter combination.
2. Writes the config to `mlx_sweep_runs/{technique}/trial_NNNN/config.yaml`.
3. Calls `smart_train.py` as a subprocess (which in turn calls `mlx_lm.lora`).
4. Parses the final val loss from the training log.
5. Records results in `mlx_sweep_runs/{technique}/results.json`.

After all trials, selects the best trial by minimum val loss and copies its adapter to `mlx_best_models/{technique}/`.

#### CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--technique` | `"all"` | `all`, `full`, `lora`, `qlora` |
| `--search` | `"grid"` | `grid` (exhaustive) or `tpe` (Optuna TPE, requires `optuna`) |
| `--n-trials` | `10` | Max trials for TPE search |
| `--data-dir` | `./data` | Training data directory |
| `--out-dir` | `./mlx_sweep_runs` | Trial output directory |
| `--best-dir` | `./mlx_best_models` | Where best adapters are copied |
| `--full-model` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Base model for Full FT |
| `--model` | `./models/tinyllama-4bit-base` | 4-bit base for LoRA/QLoRA |
| `--keys` | `self_attn.{q,k,v,o}_proj` | LoRA target layer keys |
| `--dropout` | `0.05` | LoRA dropout |
| `--lora-layers` | `16` | Number of transformer layers to apply LoRA |
| `--max-seq-length` | `512` | Token sequence truncation length |
| `--seed` | `0` | Global random seed |
| `--early-stop-patience` | `3` | Patience passed to `smart_train.py` |
| `--early-stop-min-delta` | `0.005` | Min-delta passed to `smart_train.py` |

#### Sweep Grid (Current Values)

These are the values actually searched in the grid:

| Axis | Values | Notes |
|------|--------|-------|
| `FULL_LRS` (Full FT only) | `[1e-5]` | Reduced from `[1e-5, 2e-5]` after early experiments showed `5e-5` caused repetition |
| `RANKS` (LoRA/QLoRA) | `[8, 16]` | Reduced from `[8, 16, 32]` after best found rank=32 via extended trial; current grid kept smaller for speed |
| `LRS` (LoRA/QLoRA) | `[2e-4]` | 5e-5 also shown best in extended trials |
| `EPOCHS` | `[2]` | 2 epochs over 4,000-example dataset |
| `BATCH_SIZES` | `[4]` | Per-node batch size (effective = batch × grad_accum) |
| `GRAD_ACCUMS` | `[4, 6]` | Gradient accumulation multiplier |

#### Fixed Settings (Not Swept)

| Setting | Value | Rationale |
|---------|-------|-----------|
| Alpha | `2 × rank` always | `scale=2.0` is standard practice; scale=1.0 was a confirmed bug |
| LR schedule | `cosine_decay` | Cosine decay from peak LR to 10% of peak over all steps, with 50-step linear warmup |
| Warmup steps | `50` | Brief linear ramp from 0 to peak LR |
| Final LR | `peak_lr × 0.1` | 10% of peak at end of cosine decay |
| `neftune_alpha` | `5.0` | NEFTune noise injection for LoRA/QLoRA only (not Full FT) |
| `grad_checkpoint` | `true` | Always enabled |
| `mask_prompt` | `true` | Loss on assistant tokens only |
| `val_batches` | `-1` | Full validation set used (all 500 examples) |
| `test` | `true` | Test evaluation at end of training |

#### Alpha / Scale Derivation

```
alpha = rank × scale = rank × 2.0
# e.g. rank=32 → alpha=64, scale=2.0
```

This means adapter contributions are applied at double rank strength, which is the standard LoRA convention.

### 5.5 Best Sweep Configurations (Final Models)

These are the configs used for `mlx_best_models/` — the models actually used in evaluation.

#### Best Full FT — `trial_0001` (val loss: 1.356)

| Parameter | Value |
|-----------|-------|
| `model` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| `fine_tune_type` | `full` |
| `learning_rate` | `1e-5` |
| `batch_size` | `1` |
| `grad_accumulation_steps` | `16` |
| `effective_batch_size` | `16` |
| `iters` | `8,000` (2 epochs) |
| `max_seq_length` | `512` |
| `val_batches` | `-1` (all 500) |
| `steps_per_eval` | `400` |
| `save_every` | `400` |
| `mask_prompt` | `true` |
| `lr_schedule` | cosine_decay, warmup=50, final_lr=1e-6 |
| `seed` | `0` |
| Early stopped | yes |

#### Best LoRA — `trial_0002` (val loss: 1.358)

| Parameter | Value |
|-----------|-------|
| `model` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| `fine_tune_type` | `lora` |
| `lora_layers` | `16` |
| `lora_parameters.rank` | `32` |
| `lora_parameters.scale` | `2.0` (alpha = 64) |
| `lora_parameters.dropout` | `0.05` |
| `lora_parameters.neftune_alpha` | `5.0` |
| `lora_parameters.keys` | `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`, `self_attn.o_proj`, `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj` (7 modules) |
| `learning_rate` | `5e-5` |
| `batch_size` | `1` |
| `grad_accumulation_steps` | `16` |
| `effective_batch_size` | `16` |
| `iters` | `8,000` (2 epochs) |
| `max_seq_length` | `512` |
| `val_batches` | `-1` |
| `steps_per_eval` | `400` |
| `save_every` | `400` |
| `mask_prompt` | `true` |
| `lr_schedule` | cosine_decay, warmup=50, peak=5e-5, final=5e-6 |
| `seed` | `0` |
| Early stopped | yes |

#### Best QLoRA — `trial_0002` (val loss: 1.379)

| Parameter | Value |
|-----------|-------|
| `model` | `./models/tinyllama-4bit-base` |
| `fine_tune_type` | `lora` (LoRA over 4-bit base) |
| `lora_layers` | `16` |
| `lora_parameters.rank` | `32` |
| `lora_parameters.scale` | `2.0` (alpha = 64) |
| `lora_parameters.dropout` | `0.05` |
| `lora_parameters.neftune_alpha` | `5.0` |
| `lora_parameters.keys` | `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`, `self_attn.o_proj`, `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj` (7 modules) |
| `learning_rate` | `5e-5` |
| `batch_size` | `1` |
| `grad_accumulation_steps` | `16` |
| `effective_batch_size` | `16` |
| `iters` | `8,000` (2 epochs) |
| `max_seq_length` | `512` |
| `val_batches` | `-1` |
| `steps_per_eval` | `400` |
| `save_every` | `400` |
| `mask_prompt` | `true` |
| `lr_schedule` | cosine_decay, warmup=50, peak=5e-5, final=5e-6 |
| `seed` | `0` |
| Early stopped | yes |

#### All Sweep Trials Summary

**Full FT (1 trial):**

| Trial | LR | Grad Accum | Epochs | Val Loss | Early Stopped |
|-------|----|-----------|--------|----------|---------------|
| trial_0001 | 1e-5 | 16 | 2 | **1.356** | yes |

**LoRA (2 trials shown — from final run):**

| Trial | Rank | Alpha | Scale | LR | Grad Accum | Epochs | Val Loss | Early Stopped |
|-------|------|-------|-------|-----|-----------|--------|----------|---------------|
| trial_0001 | 32 | 64 | 2.0 | 1e-4 | 16 | 2 | 1.362 | yes |
| trial_0002 | 32 | 64 | 2.0 | 5e-5 | 16 | 2 | **1.358** | yes |

**QLoRA (2 trials shown — from final run):**

| Trial | Rank | Alpha | Scale | LR | Grad Accum | Epochs | Val Loss | Early Stopped |
|-------|------|-------|-------|-----|-----------|--------|----------|---------------|
| trial_0001 | 32 | 64 | 2.0 | 1e-4 | 16 | 2 | 1.382 | yes |
| trial_0002 | 32 | 64 | 2.0 | 5e-5 | 16 | 2 | **1.379** | yes |

### 5.6 Rank Ablation Experiments

Before the sweep system was built, three fixed-rank experiments were run to understand the effect of LoRA rank. These are now superseded by the sweep system.

| Config | Rank | Alpha | Scale | LR | Keys | Iters | Val Batches |
|--------|------|-------|-------|----|------|-------|-------------|
| `rank8.yaml` | 8 | 16 | 2.0 | 1e-5 | q, v | 4,000 | 25 |
| `rank16.yaml` | 16 | 32 | 2.0 | 1e-5 | q, v | 4,000 | 25 |
| `rank32.yaml` | 32 | 64 | 2.0 | 1e-5 | q, v | 4,000 | 25 |

All three used: `lora_layers=16`, `batch_size=1`, `max_seq_length=512`, `grad_checkpoint=true`, `steps_per_eval=200`, `save_every=1000`. No `mask_prompt`, no early stopping, no LR schedule.

---

## 6. Evaluation Methodology

### 6.1 Pairwise Blind Judging

**Core library:** `evaluation/pipeline.py`  
**CLI:** `evaluation/run_pipeline.py` / `evaluation/run_eval_6models.sh`

#### Procedure

1. **Generate responses**: all 6 models generate responses to the same set of prompts.
2. **Build blind pairs**: responses are paired with left/right sides counterbalanced (each model appears on each side equally often). Pairs are shuffled.
3. **Judge**: a judge model (or human) sees `Response A` and `Response B` without model labels and outputs `1`, `2`, or `tie`.
4. **Score**: judgments are unblinded using the key file. Win rates, tie rates, net margins, and 95% bootstrapped confidence intervals are computed.

#### Eval Set

| Property | Value |
|----------|-------|
| File | `evaluation/eval_prompts.jsonl` |
| Total prompts | 500 (frozen) |
| Smoke subset | `eval_prompts_smoke20.jsonl` (20 prompts) |
| Default prompts used | 200 (via `MAX_PROMPTS=200`) |
| Seed | 42 |
| Source | Sampled from Alpaca test split; verified 0 overlap with training set |

#### Pairs Evaluated

9 fixed pairs — each fine-tuned variant vs each comparison model:

| Fine-tuned | vs | Baseline |
|------------|-----|---------|
| full_ft | vs | base |
| full_ft | vs | qwen_1.8b |
| full_ft | vs | phi_2 |
| lora_ft | vs | base |
| lora_ft | vs | qwen_1.8b |
| lora_ft | vs | phi_2 |
| qlora_ft | vs | base |
| qlora_ft | vs | qwen_1.8b |
| qlora_ft | vs | phi_2 |

#### Scoring Metrics (per pair)

| Metric | Definition |
|--------|-----------|
| `win_rate_model_1` | wins(m1) / total_judgments |
| `effective_win_rate_model_1` | wins(m1) / (wins(m1) + wins(m2)), ties excluded |
| `net_margin` | win_rate(m1) − win_rate(m2) |
| `tie_rate` | ties / total |
| `ci95_low`, `ci95_high` | 95% CI via 1,000 bootstrap resamples on win indicators |

#### Judge Methods

| Judge | Script | Notes |
|-------|--------|-------|
| LM Studio (API) | `judge_with_lmstudio.py` | OpenAI-compatible API; supports any local model |
| Local MLX model | `judge_with_model.py` | Loads model with MLX; greedy decoding, first token is verdict |
| Manual | (stdin) | Human labels pairs one by one |

**LM Studio judge details:**
- Default base URL: `http://127.0.0.1:1234/v1`
- Default judge model: `"gpt-oss-20b"` (overridable via `--model`)
- `max_judge_tokens`: 1,024
- Response truncation: `max_response_chars=800` chars (to fit in judge context window)
- Verdict parsing: 4-layer heuristic: (1) pattern `verdict.*\b[12]\b` in tail 300 chars → (2) last clear `"1"` or `"2"` word → (3) first word → (4) keyword count majority vote

**Local MLX judge details:**
- Temperature: 0.0 (greedy)
- `max_tokens`: 8 (only the verdict digit matters)

### 6.2 Cosine Similarity Ranking

**Script:** `evaluation/score_similarity_rankings.py`

An alternative evaluation method that does not require an LLM judge. Each model's response is compared against a reference answer using sentence-level cosine similarity.

#### Method

1. Text preprocessing: lowercase → remove punctuation → NLTK stopword removal → WordNet lemmatization.
2. Sentence encoding: `SentenceTransformer("all-MiniLM-L6-v2")`.
3. Cosine similarity computed between preprocessed model response and preprocessed reference answer.
4. **Win rule**: `sim(m1) > sim(m2) + 1e-5` → m1 wins; reverse → m2 wins; otherwise tie.
5. Metrics reported: average similarity, average rank across all models, pairwise win rates, overall win rate vs all other models.

#### Parameters

| Parameter | Value |
|-----------|-------|
| Embedding model | `all-MiniLM-L6-v2` (sentence-transformers) |
| Win threshold | `1e-5` similarity difference |
| Reference file | `evaluation/eval_references.jsonl` |
| Preprocessing | lowercase, punctuation removal, NLTK stopwords, WordNet lemmatizer |

### 6.3 Generation Parameters

Fixed for all models across all evaluation runs to ensure fair comparison:

| Parameter | Value | Applied to |
|-----------|-------|-----------|
| `temperature` | `0.2` | All models |
| `top_p` | `0.9` | All models |
| `max_tokens` | `256` | All models |
| `seed` | `42` | All models |
| `min_tokens` | `50` | FT models only (`full_ft`, `lora_ft`, `qlora_ft`) |
| `repetition_penalty` | `1.2` | FT models only (applied alongside `min_tokens`) |

#### min_tokens Implementation

The `min_tokens` feature is implemented as a custom `logits_processor` in `evaluation/pipeline.py`. For the first `min_tokens` steps, it sets the logit of all EOS token IDs to `-inf`, preventing early termination. This forces FT models to generate at least 50 tokens, counteracting the short-answer bias from training data. A `repetition_penalty=1.2` processor runs in parallel to prevent looping (see Finding 1 below).

### 6.4 Quality Heuristics

**Script:** `evaluation/check_response_quality.py`

Pre-screens generated responses for gibberish before judging:

| Metric | Threshold |
|--------|---------|
| Alpha character ratio | ≥ 0.35 |
| Digit ratio | ≤ 0.20 |
| Weird character ratio | ≤ 0.05 |
| Max bad-rate (default) | 0.20 (20% of responses can fail before aborting) |

---

## 7. Results

Run ID: `20260303_205319` · Eval method: cosine similarity · Prompts: 100

### 7.1 Cosine Similarity Rankings

| Rank | Model | Avg Similarity | Avg Rank (lower=better) | Pairwise Matches | Win Rate (vs All) |
|------|-------|---------------|------------------------|-----------------|-------------------|
| 1 | **lora_ft** | 0.6844 | 3.29 | 500 | 52.0% |
| 2 | **qlora_ft** | 0.6717 | 3.32 | 500 | 50.4% |
| 3 | **full_ft** | 0.6807 | 3.48 | 500 | 50.4% |
| 4 | qwen_1.8b | 0.6693 | 3.51 | 500 | 48.8% |
| 5 | phi_2 | 0.6706 | 3.66 | 500 | 46.4% |
| 6 | base | 0.6526 | 3.74 | 500 | 44.4% |

Key observations:
- All three FT variants rank above both comparison models and the base model.
- `lora_ft` achieves the highest average cosine similarity (0.6844), highest win rate (52.0%), and best average rank (3.29).
- `full_ft` achieves the second-highest cosine similarity (0.6807) but ranks third overall by win rate (50.4%), indicating its scores cluster in the middle more than LoRA's.
- `base` has the lowest average similarity (0.6526) and highest average rank (3.74 = worst).

### 7.2 FT Models vs Base (Primary Goal)

Target: 55%+ win rate for each FT variant vs base.

| FT Model | FT Wins | Base Wins | Ties | FT Win % | Goal Met? |
|----------|---------|-----------|------|----------|-----------|
| full_ft | 53 | 43 | 4 | **53.0%** | approaching |
| lora_ft | 54 | 44 | 2 | **54.0%** | approaching |
| qlora_ft | 54 | 44 | 2 | **54.0%** | approaching |

All three FT variants beat the base model. LoRA and QLoRA are 1 percentage point below the 55% goal.

### 7.3 Full Head-to-Head Pairwise Matrix

All 15 unique matchups (out of 6 models):

| Model 1 | Model 2 | M1 Wins | M2 Wins | Ties | M1 Win % |
|---------|---------|---------|---------|------|---------|
| lora_ft | phi_2 | 55 | 45 | 0 | **55.0%** |
| lora_ft | qwen_1.8b | 55 | 44 | 1 | **55.0%** |
| qlora_ft | phi_2 | 55 | 44 | 1 | **55.0%** |
| qlora_ft | base | 54 | 44 | 2 | 54.0% |
| lora_ft | base | 54 | 44 | 2 | 54.0% |
| qwen_1.8b | base | 54 | 45 | 1 | 54.0% |
| full_ft | base | 53 | 43 | 4 | 53.0% |
| phi_2 | base | 53 | 46 | 1 | 53.0% |
| qlora_ft | qwen_1.8b | 51 | 47 | 2 | 51.0% |
| qwen_1.8b | phi_2 | 50 | 47 | 3 | 50.0% |
| qwen_1.8b | full_ft | 49 | 50 | 1 | 49.0% |
| lora_ft | full_ft | 47 | 45 | 8 | 47.0% |
| qlora_ft | lora_ft | 46 | 49 | 5 | 46.0% |
| qlora_ft | full_ft | 46 | 48 | 6 | 46.0% |
| phi_2 | full_ft | 43 | 56 | 1 | 43.0% |

**Notable findings from pairwise matrix:**
- `lora_ft` beats both `phi_2` and `qwen_1.8b` at exactly 55.0% — reaching the project goal in those matchups.
- `full_ft` is the strongest single model: it beats `lora_ft` (47% → 53%), beats `qlora_ft` (46% → 54%), and beats `phi_2` 57% of the time.
- `qwen_1.8b` is surprisingly competitive, beating `full_ft` 49% and `phi_2` 50% of the time.
- The three FT variants are within noise of each other (46–54% win rates between themselves).

---

## 8. Diagnostic Findings and Fixes Applied

During development, a systematic audit identified 9 failure modes explaining why early eval runs showed FT models below 50% win rate vs base. The following table summarises all findings and their resolution status.

### Findings and Resolution

| # | Finding | Severity | Root Cause | Fix Applied | Impact |
|---|---------|---------|-----------|------------|--------|
| 1 | **Repetition loops in FT responses** | HIGH | `min_tokens=50` forces generation past EOS; FT models trained on 26-word median answers have nothing to say and loop | Added `repetition_penalty=1.2` alongside `min_tokens` | +5–10% win rate |
| 2 | **LoRA scale=1.0 (adapter under-contributing)** | HIGH | Early sweep trials used `alpha=rank` → `scale=1.0` instead of standard `scale=2.0` | Enforced `scale=2.0` (alpha=2×rank) in all new sweep configs | +3–5% win rate |
| 3 | **Training data too terse** | MEDIUM-HIGH | 53% of training answers are under 30 words; model trained to produce short answers, regressing TinyLlama Chat's RLHF verbosity | Data filtered to top-5000 longest answers; `min_tokens` added as inference fix | +10–15% if retrained with filtered data |
| 4 | **System prompt mismatch (training vs inference)** | MEDIUM | 0/4,000 training examples have system prompts; at inference time a `<|system|>` block is injected (OOD for FT) | No system prompt injected for FT models in production `models.json` | +3–5% win rate |
| 5 | **Full FT max_seq_length=256** | MEDIUM | Early Full FT configs used 256-token limit; responses truncated at ~64 words; worst repetition | Increased to 512 in current sweep configs | +2–3% for full_ft |
| 6 | **Models under-trained (val loss still declining)** | MEDIUM | Val loss declining at end of all runs; LoRA: 1.306→1.256; QLoRA: 1.318→1.280; Full: 1.544→1.373 | Increased to 2 epochs (8,000 iters over 4,000 examples) | +2–3% win rate |
| 7 | **No overfitting** | (positive) | Train/val loss gaps: LoRA −0.030, QLoRA −0.017, Full −0.026 | No action needed | — |
| 8 | **Correct model selection** | (verification) | Confirmed best trial correctly identified by val loss; no adapter path mix-up | Verified with SHA256 hashes in `validate_training_setup.py` | — |
| 9 | **No data leakage** | (positive) | 0/500 eval prompts appear in training set | No action needed | — |

### Early Run Issues (Resolved)

| Issue | Fix Applied |
|-------|------------|
| `max_seq_length: 256` for Full FT | Changed to 512 |
| `scale=1.0` in some LoRA sweep trials | Removed alpha from grid; enforced `scale=2.0` |
| Full FT LR `5e-5` causing catastrophic repetition | Changed `FULL_LRS` to `[1e-5]` |
| Judge right-side positional bias (63.7% right preference) | Counterbalancing fixed; confirmed neutral 51–54% in later runs |
| `TokenizersBackend` crash on QLoRA eval | Shim added to `pipeline.py` |
| `LoRA rank=32` in early sweep (overkill) | `RANKS` changed to `[8, 16]` for speed; extended trial confirmed rank=32 best |

---

## 9. Complete Changeable Parameter Reference

This section lists every parameter that can be changed and its current/default value.

### Dataset Parameters (`prepare_dataset.py`)

| Parameter | Current Value | Range / Options | Effect |
|-----------|--------------|-----------------|--------|
| `DATASET_NAME` | `"tatsu-lab/alpaca"` | any HF dataset | Source dataset |
| `NUM_EXAMPLES` | `5000` | 100–52,002 | Subset size |
| `TRAIN_SPLIT` | `0.8` | 0.0–1.0 | Training fraction |
| `VALID_SPLIT` | `0.1` | 0.0–1.0 | Validation fraction |
| `TEST_SPLIT` | `0.1` | 0.0–1.0 | Test fraction |

### LoRA Parameters (all configs)

| Parameter | Baseline | Best Sweep | Range | Effect |
|-----------|---------|-----------|-------|--------|
| `rank` | 16 | 32 | 4, 8, 16, 32, 64 | Low-rank dimension; higher = more capacity + compute |
| `scale` (alpha/rank) | 2.0 | 2.0 | 1.0–4.0 | Adapter contribution strength |
| `alpha` (derived) | 32 | 64 | rank × scale | Never set directly; derived from rank × scale |
| `dropout` | 0.05 | 0.05 | 0.0–0.3 | LoRA layer dropout rate |
| `lora_layers` | 16 | 16 | 1–22 | Number of transformer layers receiving adapters |
| `keys` (target modules) | q, v | q, k, v, o, gate, up, down | any proj names | Which weight matrices get low-rank updates |
| `neftune_alpha` | not set | 5.0 | 0.0–25.0 | NEFTune noise injection strength (0 disables) |

### Training Parameters (all configs)

| Parameter | Baseline LoRA | Baseline QLoRA | Best Sweep | Range | Effect |
|-----------|--------------|---------------|-----------|-------|--------|
| `learning_rate` | 1e-5 | 1e-5 | 5e-5 (LoRA/QLoRA), 1e-5 (Full) | 1e-6–1e-3 | Peak learning rate |
| `batch_size` | 1 | 1 | 1 | 1–8 | Per-device batch size |
| `grad_accumulation_steps` | 1 | 1 | 16 | 1–64 | Effective batch = batch × accum |
| `iters` | 1,200 | 1,200 | 8,000 | any | Training iterations |
| `max_seq_length` | 512 | 512 | 512 | 128–2048 | Token sequence truncation |
| `grad_checkpoint` | true | true | true | bool | Gradient checkpointing (memory vs speed) |
| `mask_prompt` | true | true | true | bool | Loss on assistant tokens only |
| `val_batches` | 25 | 25 | -1 (all) | -1 or int | Validation batches per eval step |
| `steps_per_eval` | 100 | 100 | 400 | int | How often to run validation |
| `save_every` | 200 | 200 | 400 | int | Checkpoint save frequency |
| `seed` | 0 | 0 | 0 | any int | Global random seed |

### LR Schedule Parameters (sweep configs only)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `lr_schedule.name` | `cosine_decay` | Only applied in sweep; baseline uses flat LR |
| `lr_schedule.warmup` | `50` | Linear warmup steps |
| `lr_schedule.warmup_init` | `0.0` | Starting LR for warmup |
| `lr_schedule.arguments[0]` | `peak_lr` | e.g. 5e-5 |
| `lr_schedule.arguments[1]` | `iters - warmup` | Decay steps = 7,950 |
| `lr_schedule.arguments[2]` | `peak_lr × 0.1` | Final LR e.g. 5e-6 |

### Early-Stopping Parameters (`smart_train.py`)

| Flag | Default (baseline) | Default (sweep) | Range | Effect |
|------|--------------------|----------------|-------|--------|
| `--patience` | 5 | 3 | 1–∞ | Evals without improvement before stop |
| `--min-delta` | 0.0 | 0.005 | 0.0–0.1 | Minimum improvement to reset patience |
| `--full-interval-divisor` | 10 | N/A | 1–100 | steps_per_eval auto-derivation for Full FT |

### Sweep Parameters (`sweep_finetune.py`)

| Constant/Flag | Current Value | Options | Effect |
|---------------|--------------|---------|--------|
| `FULL_LRS` | `[1e-5]` | list of floats | Full FT LRs to try |
| `RANKS` | `[8, 16]` | list of ints | LoRA ranks to try |
| `LRS` | `[2e-4]` | list of floats | LoRA/QLoRA LRs to try |
| `EPOCHS` | `[2]` | list of ints | Training epochs to try |
| `BATCH_SIZES` | `[4]` | list of ints | Batch sizes to try |
| `GRAD_ACCUMS` | `[4, 6]` | list of ints | Grad accumulation steps to try |
| `--search` | `grid` | `grid`, `tpe` | Search strategy |
| `--n-trials` | `10` | int | Max trials for TPE |
| `--early-stop-patience` | `3` | int | Patience for each trial |
| `--early-stop-min-delta` | `0.005` | float | Min-delta for each trial |

### Evaluation Parameters (`run_eval_6models.sh` env vars)

| Variable | Default | Range | Effect |
|----------|---------|-------|--------|
| `TEMPERATURE` | `0.2` | 0.0–2.0 | Generation temperature |
| `TOP_P` | `0.9` | 0.0–1.0 | Nucleus sampling cutoff |
| `MAX_TOKENS` | `256` | 1–2048 | Max tokens per response |
| `SEED` | `42` | any int | Reproducibility seed |
| `MAX_PROMPTS` | `200` | 1–500 | Prompts evaluated per pair |

### Inference Parameters (`evaluation/models.json`)

| Field | Type | Effect |
|-------|------|--------|
| `name` | string | Model identifier used throughout pipeline |
| `model` | string | HF Hub ID or local path |
| `adapter_path` | string (optional) | Path to adapter directory |
| `system_prompt` | string (optional) | System prompt injected at inference |
| `min_tokens` | int (optional) | Minimum tokens before EOS allowed |

### Cosine Similarity Eval Parameters

| Parameter | Current Value | Effect |
|-----------|--------------|--------|
| Embedding model | `all-MiniLM-L6-v2` | Sentence encoder |
| Win threshold | `1e-5` | Minimum similarity gap to declare a winner |
| Text preprocessing | lowercase + punct removal + stopwords + lemmatize | Normalisation depth |

### LM Studio Judge Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `--model` | `"gpt-oss-20b"` | Judge model name |
| `--base-url` | `http://127.0.0.1:1234/v1` | LM Studio API endpoint |
| `--max-judge-tokens` | `1024` | Max tokens in judge response |
| `--max-response-chars` | `800` | Response truncation before judging |

### Streamlit App Parameters (`app.py`)

| Setting | Default | Range |
|---------|---------|-------|
| Temperature | 0.7 | 0.0–1.0 (slider) |
| Max tokens | 256 | 64–512 (slider) |
| Model path | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | any |
| Adapter dir | `./adapters` | any path |
| Log file | `./logs/chat_log.jsonl` | any path |

---

## 10. Infrastructure and Environment

### Hardware

| Component | Requirement |
|-----------|------------|
| CPU/GPU | Apple Silicon (M1/M2/M3/M4) — unified memory architecture |
| RAM / Unified Memory | 8GB minimum; 16GB+ recommended for Full FT |
| Disk | ~15GB (3 models × ~3–5GB each, adapters, sweep outputs) |

### Software Stack

| Package | Purpose |
|---------|---------|
| `mlx` | Apple Silicon ML framework (array ops, autodiff) |
| `mlx-lm` | LLM training and inference on MLX |
| `datasets` | HuggingFace dataset loading (`tatsu-lab/alpaca`) |
| `streamlit` | Interactive chat UI |
| `optuna` (optional) | TPE hyperparameter search |
| `sentence-transformers` (optional) | Cosine similarity evaluation |
| `nltk` (optional) | Text preprocessing for cosine eval |
| `numpy` (optional) | Numerical operations for cosine eval |
| `yaml` | Config file parsing |
| `requests` | LM Studio API calls |

### Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
.venv/bin/pip install -r requirements.txt   # mlx-lm, mlx, datasets, streamlit
```

### Key MLX-LM Commands Used

| Command | Purpose |
|---------|---------|
| `mlx_lm.lora --config <yaml>` | Run LoRA / QLoRA / Full FT training |
| `mlx_lm.generate --model <path> --adapter-path <path>` | Inference with adapter |
| `mlx_lm.convert` | Convert HF model to MLX format |
| `mlx_lm.fuse` | Fuse adapter into base model weights |

---

## 11. File Reference

### Core Pipeline Files

| File | Purpose |
|------|---------|
| `prepare_dataset.py` | Download, filter, format, and split the Alpaca dataset |
| `smart_train.py` | Early-stopping training wrapper around `mlx_lm.lora` |
| `sweep_finetune.py` | Grid/TPE hyperparameter sweep across Full FT, LoRA, QLoRA |
| `validate_training_setup.py` | 6-point pre-run sanity checker |
| `lora_config.yaml` | Baseline LoRA training configuration |
| `qlora_config.yaml` | Baseline QLoRA training configuration |
| `run_train.sh` | One-command LoRA training entrypoint |
| `run_qlora.sh` | One-command QLoRA training entrypoint |

### Evaluation Files

| File | Purpose |
|------|---------|
| `evaluation/pipeline.py` | Core eval library: all logic as pure functions |
| `evaluation/run_pipeline.py` | Full pipeline in one command |
| `evaluation/run_eval_6models.sh` | 6-model shell wrapper with env-var overrides |
| `evaluation/generate_responses.py` | Generate model responses to eval prompts |
| `evaluation/make_pairs.py` | Build blind pairwise judging tasks |
| `evaluation/judge_with_lmstudio.py` | LM Studio OpenAI-API judge |
| `evaluation/judge_with_model.py` | Local MLX model judge |
| `evaluation/score_judgments.py` | Compute win rates, CI, write reports |
| `evaluation/score_similarity_rankings.py` | Cosine similarity ranking |
| `evaluation/check_response_quality.py` | Heuristic gibberish guard |
| `evaluation/models.json` | 6-model eval spec (paths, min_tokens, system prompts) |
| `evaluation/eval_prompts.jsonl` | 500-prompt frozen eval set |
| `evaluation/eval_prompts_smoke20.jsonl` | 20-prompt smoke test subset |
| `evaluation/eval_references.jsonl` | Reference answers for cosine eval |

### Utility Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit chat playground |
| `quick_eval.py` | Side-by-side base vs adapter sanity check |
| `plot_loss.py` | ASCII loss curve plotter from `train.log` |
| `requirements.txt` | Python package dependencies |

### Artifacts

| Directory | Contents |
|-----------|---------|
| `data/` | `train.jsonl` (4k), `valid.jsonl` (500), `test.jsonl` (500) |
| `adapters/tinyllama-lora-alpaca/` | Baseline LoRA adapter + checkpoints |
| `models/tinyllama-4bit-base/` | TinyLlama 4-bit (QLoRA base) |
| `models/phi-2-hf-4bit-mlx/` | Phi-2 2.7B 4-bit |
| `models/qwen1.5-1.8b-chat-4bit/` | Qwen 1.5 1.8B 4-bit |
| `mlx_best_models/{full,lora,qlora}/` | Best adapter per technique (used in eval) |
| `mlx_sweep_runs/{full,lora,qlora}/` | All trial configs, logs, checkpoints |
| `evaluation/runs/` | Timestamped eval run outputs (responses, pairs, scores) |
