<![CDATA[# 🦙 TinyLlama-1.1B Instruction Tuning Pipeline (MLX on Apple Silicon)

> **End-to-end project:** raw dataset → three fine-tuning strategies → automated hyperparameter sweep → blind pairwise evaluation → validated Streamlit demo — all running **100% locally on your Mac** using Apple's MLX framework (Metal GPU).

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Hardware & Software Requirements](#2-hardware--software-requirements)
3. [Quick Start (Zero to Working in 5 Minutes)](#3-quick-start-zero-to-working-in-5-minutes)
4. [The Dataset: Source, Size & Format](#4-the-dataset-source-size--format)
5. [The Base Model: TinyLlama-1.1B-Chat](#5-the-base-model-tinyllama-11b-chat)
6. [Training Strategy 1 — LoRA (Low-Rank Adaptation)](#6-training-strategy-1--lora-low-rank-adaptation)
7. [Training Strategy 2 — QLoRA (Quantized LoRA)](#7-training-strategy-2--qlora-quantized-lora)
8. [Training Strategy 3 — Full Fine-Tuning](#8-training-strategy-3--full-fine-tuning)
9. [How Training Works Under the Hood](#9-how-training-works-under-the-hood)
10. [Hyperparameter Sweep System](#10-hyperparameter-sweep-system)
11. [Best Model Selection Logic](#11-best-model-selection-logic)
12. [LoRA Rank Experiments](#12-lora-rank-experiments)
13. [Full-FT Recovery Path (Safe Retrain)](#13-full-ft-recovery-path-safe-retrain)
14. [Evaluation Pipeline — Deep Dive](#14-evaluation-pipeline--deep-dive)
15. [Judging: Manual & LM-as-Judge](#15-judging-manual--lm-as-judge)
16. [Scoring, Metrics & Reports](#16-scoring-metrics--reports)
17. [Streamlit Chat UI](#17-streamlit-chat-ui)
18. [Validation & Regression Guard](#18-validation--regression-guard)
19. [Complete File Map](#19-complete-file-map)
20. [Every Command You'll Ever Need](#20-every-command-youll-ever-need)
21. [Troubleshooting](#21-troubleshooting)
22. [Design Decisions & Audit Trail](#22-design-decisions--audit-trail)

---

## 1. Architecture Overview

```
┌───────────────────────────────────────────────────────────────────┐
│                        PREPARE DATA                              │
│  tatsu-lab/alpaca (HuggingFace) ──► prepare_dataset.py           │
│  5,000 examples → 4,000 train / 500 valid / 500 test            │
│  Format: MLX chat JSONL  {messages: [{role, content}, ...]}      │
└─────────────────────────────┬─────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│                       TRAIN MODELS                               │
│                                                                   │
│  ┌──────────┐    ┌──────────┐    ┌──────────────┐                │
│  │  LoRA    │    │  QLoRA   │    │  Full FT     │                │
│  │ fp16 base│    │ 4-bit    │    │  fp16 base   │                │
│  │ +adapters│    │ base     │    │  all weights  │                │
│  │          │    │ +adapters│    │  updated      │                │
│  └────┬─────┘    └────┬─────┘    └──────┬───────┘                │
│       │               │                 │                         │
│       ▼               ▼                 ▼                         │
│  adapters/       adapters/        models/                         │
│  tinyllama-      tinyllama-       tinyllama-                      │
│  lora-alpaca     qlora-alpaca     full-alpaca                     │
└─────────────────────────────┬─────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│              HYPERPARAMETER SWEEP (optional)                      │
│  sweep_mlx_lora.py  →  grid search or TPE (Optuna)              │
│  Tests: ranks, alphas, LRs, batch sizes, grad accumulations     │
│  Selects: lowest validation loss checkpoint                      │
│  Copies best: mlx_best_models/{lora,qlora,full}                 │
└─────────────────────────────┬─────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│                     EVALUATE (6-model)                            │
│  500 frozen prompts × 6 models × fixed generation params         │
│  → Blind pairwise tasks (counterbalanced left/right)             │
│  → Judge (human manual OR LM-as-judge via LM Studio)            │
│  → Score: win rate, effective win rate, 95% bootstrap CI         │
│  → Markdown report                                               │
└─────────────────────────────┬─────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│                    STREAMLIT DEMO                                 │
│  app.py: single-model chat + side-by-side compare mode           │
│  Loads any adapter or full model from disk                       │
│  Safety mode toggle, generation parameter sliders                │
└───────────────────────────────────────────────────────────────────┘
```

---

## 2. Hardware & Software Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| **Machine** | Any Apple Silicon Mac | M1 Pro / M2 / M3 with 16 GB+ |
| **macOS** | 13.0 (Ventura) | 14.0+ (Sonoma) |
| **Python** | 3.10 | 3.11+ |
| **Unified Memory** | 8 GB (LoRA only) | 16 GB+ (Full FT) |
| **Disk Space** | ~15 GB | ~25 GB (with all model variants) |

> **Why Apple Silicon?** This project uses [MLX](https://github.com/ml-explore/mlx), Apple's machine learning framework that runs natively on Metal GPU. It does **not** use CUDA, PyTorch-GPU, or cloud APIs. Everything runs locally on your Mac.

---

## 3. Quick Start (Zero to Working in 5 Minutes)

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd PyCharmMiscProject

# 2. Create Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install all dependencies
.venv/bin/pip install -r requirements.txt

# 4. (Optional) Install Optuna for bayesian sweep search
.venv/bin/pip install optuna

# 5. Prepare the Alpaca dataset (downloads from HuggingFace, ~30 seconds)
.venv/bin/python prepare_dataset.py

# 6. Train your first LoRA adapter (~10-20 minutes on M1 Pro)
./run_train.sh

# 7. Test it with a quick prompt
.venv/bin/python -m mlx_lm.generate \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter-path ./adapters/tinyllama-lora-alpaca \
  --prompt "What is reinforcement learning?" \
  --max-tokens 256

# 8. Launch the Streamlit chat UI
.venv/bin/streamlit run app.py
```

That's it. You have a working fine-tuned model with a chat interface. Read on for the deep technical details.

---

## 4. The Dataset: Source, Size & Format

### Source

**[tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)** — Stanford's Alpaca dataset, a collection of 52,002 instruction-following examples generated from OpenAI's `text-davinci-003`. Each example has:
- `instruction`: the task description
- `input`: optional additional context
- `output`: the expected response

### How We Use It

The script `prepare_dataset.py` does the following:

1. **Downloads** the first 5,000 examples from the Alpaca `train` split via HuggingFace `datasets` library.
2. **Converts** each example into MLX-LM's chat format:
   ```json
   {
     "messages": [
       {"role": "user", "content": "<instruction>\n\nInput:\n<input>"},
       {"role": "assistant", "content": "<output>"}
     ]
   }
   ```
3. **Shuffles** with a fixed seed (`random.seed(42)`) for reproducibility.
4. **Splits** into three files with an 80/10/10 ratio:

| Split | File | Row Count | Purpose |
|---|---|---:|---|
| Train | `data/train.jsonl` | 4,000 | Model weight updates |
| Validation | `data/valid.jsonl` | 500 | Loss tracking during training, checkpoint selection |
| Test | `data/test.jsonl` | 500 | Final test loss after training completes |

### Separate Eval Prompt Set

The evaluation pipeline uses a **completely separate** set of 500 prompts (`evaluation/eval_prompts.jsonl`), drawn from Alpaca indices 10,001+. This prevents data leakage — the model is never trained on the prompts it's evaluated on.

Each eval prompt has this format:
```json
{
  "prompt_id": "alpaca_10001",
  "source": "ysingh-aiml/alpaca",
  "source_split": "train",
  "source_index": 10001,
  "prompt": "Name five Mediterranean countries.",
  "reference": "Greece, Italy, Spain, Turkey, France"
}
```

### Command

```bash
.venv/bin/python prepare_dataset.py
```

---

## 5. The Base Model: TinyLlama-1.1B-Chat

| Property | Value |
|---|---|
| **HuggingFace ID** | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| **Architecture** | LLaMA-2 (transformer decoder) |
| **Parameters** | 1.1 billion |
| **Context length** | 2,048 tokens |
| **Precision** | fp16 (bfloat16) |
| **Training data** | 3 trillion tokens (SlimPajama + StarCoder) |
| **Chat format** | Supports `apply_chat_template()` |
| **Disk size** | ~2.2 GB |

**Why TinyLlama?** It's the smallest model that still produces coherent instruction-following responses, making it ideal for local experimentation on Apple Silicon without requiring 32+ GB of RAM.

### Two Versions Used

| Name | Path | Precision | Used By |
|---|---|---|---|
| Full-precision | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (auto-downloaded from HF) | fp16 | LoRA, Full FT |
| 4-bit quantized | `./models/tinyllama-4bit-base` (local, pre-quantized) | 4-bit (group quant) | QLoRA only |

This distinction is critical: **LoRA trains adapter weights on top of the full-precision model**, while **QLoRA trains adapter weights on top of the 4-bit quantized model**. If both use the same base, they produce identical results (this was a bug we caught and fixed — see [Audit Trail](#22-design-decisions--audit-trail)).

---

## 6. Training Strategy 1 — LoRA (Low-Rank Adaptation)

### What It Does

Instead of updating all 1.1B parameters, LoRA freezes the base model and injects small trainable "adapter" matrices into specific attention layers. Only these adapters (~1-5M parameters depending on rank) are trained.

### Configuration: `lora_config.yaml`

| Parameter | Value | What It Means |
|---|---|---|
| `model` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | **Full-precision** base (hub model, auto-downloaded) |
| `fine_tune_type` | `lora` (implicit default) | Standard LoRA — no quantization |
| `lora_layers` | 16 | Apply LoRA to the last 16 transformer layers |
| `rank` | 16 | Adapter matrix rank (controls capacity) |
| `scale` | 2.0 | = alpha / rank = 32 / 16 (controls update magnitude) |
| `dropout` | 0.05 | Regularization on adapter weights |
| `keys` | `["self_attn.q_proj", "self_attn.v_proj"]` | Which attention matrices get adapters |
| `learning_rate` | 1e-5 | Adam optimizer learning rate |
| `batch_size` | 1 | Sequences per gradient step |
| `iters` | 1200 | Total training steps |
| `max_seq_length` | 512 | Maximum token sequence length |
| `mask_prompt` | `true` | Only compute loss on assistant tokens (not the prompt) |
| `grad_checkpoint` | `true` | Trade compute for memory savings |
| `steps_per_eval` | 100 | Validate every 100 steps |
| `save_every` | 200 | Save checkpoint every 200 steps |
| `adapter_path` | `./adapters/tinyllama-lora-alpaca` | Where adapter weights are saved |

### Output

```
./adapters/tinyllama-lora-alpaca/
├── adapter_config.json     # Hyperparameters + base model reference
├── adapters.safetensors    # Trained adapter weights (~4 MB)
└── adapter_model.safetensors (optional checkpoints)
```

### Command

```bash
./run_train.sh
```

### Estimated Time

~10-20 minutes on M1 Pro 16 GB.

---

## 7. Training Strategy 2 — QLoRA (Quantized LoRA)

### What It Does

Same concept as LoRA, but the **base model is loaded in 4-bit precision** (group quantization), reducing memory usage by ~75%. The adapter weights are still trained in higher precision. This lets you fine-tune on machines with less memory.

### Configuration: `qlora_config.yaml`

| Parameter | Value | Difference from LoRA |
|---|---|---|
| `model` | `./models/tinyllama-4bit-base` | **4-bit quantized** local model |
| `fine_tune_type` | `qlora` | Explicit QLoRA mode |
| `mask_prompt` | `true` | Same as LoRA |
| All other params | Same as LoRA | Identical rank, LR, layers, etc. |

### Key Difference

The **only** intended differences from LoRA are:
1. Base model is 4-bit quantized (smaller memory footprint)
2. `fine_tune_type` is `qlora` (tells MLX to handle quantized forward pass correctly)

### Output

```
./adapters/tinyllama-qlora-alpaca/
├── adapter_config.json
└── adapters.safetensors    # ~4 MB adapters (same size, different values)
```

### Command

```bash
./run_qlora.sh
```

### Estimated Time

~8-15 minutes on M1 Pro 16 GB (faster than LoRA due to quantized forward pass).

---

## 8. Training Strategy 3 — Full Fine-Tuning

### What It Does

Updates **every single parameter** in the 1.1B model. This gives the most capacity for learning but requires significantly more memory and can be unstable.

### Configuration: `full_config.yaml`

| Parameter | Value | Notes |
|---|---|---|
| `model` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Full-precision hub model |
| `fine_tune_type` | `full` | Update all weights |
| `learning_rate` | 1e-5 | Conservative — full FT is more sensitive to LR |
| `batch_size` | 1 | Memory-limited on most Macs |
| `iters` | 1200 | Total training steps |
| `adapter_path` | `./models/tinyllama-full-alpaca` | Output path (MLX still calls it "adapter_path") |

### ⚠️ Memory Warning

Full FT on TinyLlama-1.1B requires **~12-14 GB of unified memory**. If your Mac has only 8 GB, use LoRA or QLoRA instead.

### Output

```
./models/tinyllama-full-alpaca/
├── adapter_config.json      # Training metadata
└── adapters.safetensors     # ALL model weight deltas (~2.2 GB)
```

### Command

```bash
./run_full.sh
```

### Estimated Time

~30-60 minutes on M1 Pro 16 GB.

---

## 9. How Training Works Under the Hood

All three training scripts call the same underlying command:

```bash
.venv/bin/python -m mlx_lm.lora --config <config>.yaml
```

Here's what happens inside `mlx_lm.lora`:

1. **Load base model** from HuggingFace Hub (or local path for QLoRA's 4-bit model).
2. **Apply LoRA adapters** to specified layers (for `lora`/`qlora` modes) or mark all params trainable (for `full` mode).
3. **Load dataset** from `./data/{train,valid,test}.jsonl`.
4. **Training loop** for `iters` steps:
   - Sample a batch from `train.jsonl`
   - Tokenize with chat template
   - Forward pass (compute next-token prediction loss)
   - **If `mask_prompt: true`**: loss is computed **only** on assistant tokens, not the user prompt. This teaches the model to generate responses, not memorize prompts.
   - Backward pass → Adam optimizer update
   - Every `steps_per_report` steps: print training loss
   - Every `steps_per_eval` steps: compute validation loss on `valid.jsonl`
   - Every `save_every` steps: save checkpoint to `adapter_path`
5. **After training**: evaluate on `test.jsonl` and report test loss.

### What `mask_prompt: true` Does (Important!)

```
Tokens:  [USER] What is ML? [ASSISTANT] Machine learning is...
Loss:    ──── ignored ────  ────── computed on these ──────
```

Without prompt masking, the model wastes capacity learning to predict the prompt tokens — which it will never need to generate. With masking enabled, all learning signal focuses on response quality.

---

## 10. Hyperparameter Sweep System

The sweep system (`sweep_mlx_lora.py`) automates finding the best hyperparameters for each training strategy.

### How It Works

```
sweep_mlx_lora.py
    │
    ├── Generates hyperparameter combinations
    │   (grid search = all combos, TPE = smart sampling)
    │
    ├── For each combo:
    │   ├── Writes a YAML config to a temp directory
    │   ├── Runs: .venv/bin/python -m mlx_lm.lora --config <temp>.yaml
    │   ├── Parses training logs for validation/test loss
    │   └── Records results
    │
    ├── Selects best combo (lowest validation loss)
    │
    └── Copies best adapter to: mlx_best_models/<technique>/
```

### Search Spaces

**LoRA & QLoRA:**

| Parameter | Values Searched |
|---|---|
| Rank | 16, 32 |
| Alpha | 32, 64 |
| Learning Rate | 1e-4, 2e-4 |
| Epochs | 1 |
| Batch Size | 2 |
| Gradient Accumulation | 4, 6 |

This gives 2 × 2 × 2 × 1 × 1 × 2 = **16 combinations** per technique for grid search.

**Full FT:**

| Parameter | Values Searched |
|---|---|
| Learning Rate | 2e-5, 5e-5 |
| Epochs | 1 |
| Batch Size | 2 |
| Gradient Accumulation | 4, 6 |

This gives 2 × 1 × 1 × 2 = **4 combinations**.

### LoRA Attention Keys Targeted

For the sweep, adapters are applied to **four** attention projections (more aggressive than standalone configs):
```
self_attn.q_proj    (query)
self_attn.k_proj    (key)
self_attn.v_proj    (value)
self_attn.o_proj    (output)
```

### Base Model Routing

| Technique | Base Model Used | Why |
|---|---|---|
| `full` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (fp16 hub) | Full FT needs full precision |
| `lora` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (fp16 hub) | LoRA trains on full-precision base |
| `qlora` | `./models/tinyllama-4bit-base` (4-bit local) | QLoRA trains on quantized base |

### Commands

**Grid search — all three techniques:**
```bash
.venv/bin/python sweep_mlx_lora.py --technique all --search grid
```

**Grid search — LoRA only:**
```bash
.venv/bin/python sweep_mlx_lora.py --technique lora --search grid
```

**TPE (Bayesian) search — QLoRA, 20 trials:**
```bash
.venv/bin/python sweep_mlx_lora.py --technique qlora --search tpe --n-trials 20
```

**Custom data directory:**
```bash
.venv/bin/python sweep_mlx_lora.py --technique lora --data-dir ./data_custom
```

### Output

```
mlx_sweep_runs/
├── lora_r16_lr1e-4_ga4/          # Each run gets a descriptive directory
│   ├── config.yaml
│   ├── output/                    # Adapter checkpoints
│   │   ├── adapter_config.json
│   │   └── adapters.safetensors
│   └── train.log
├── lora_r32_lr2e-4_ga6/
│   └── ...
└── ...

mlx_best_models/                   # Best per technique (copied automatically)
├── lora/
│   ├── adapter_config.json
│   └── adapters.safetensors
├── qlora/
│   ├── adapter_config.json
│   └── adapters.safetensors
└── full/
    ├── adapter_config.json
    └── adapters.safetensors
```

---

## 11. Best Model Selection Logic

When the sweep finishes, it needs to pick the "best" run for each technique. Here's the logic:

### Loss Parsing Priority (`parse_loss()`)

1. **Test loss** (most reliable — single value at end of training)
2. **Minimum validation loss** across all checkpoints (best-checkpoint signal)
3. **Any loss mention** in logs (fallback)

The key insight: we use the **minimum** validation loss, not the last one. This selects the checkpoint where the model performed best, not just where training ended (which might have overfit).

### Evaluation Frequency

Checkpoints are evaluated and saved **at least twice per epoch**:
```python
eval_interval = max(1, steps_per_epoch // 2)
save_interval = max(1, steps_per_epoch // 2)
```

This ensures we don't miss a good checkpoint between infrequent evaluations.

---

## 12. LoRA Rank Experiments

For comparing different LoRA ranks in isolation, we provide standalone experiment configs:

| Config | Rank | Alpha (via scale) | Iterations | Output |
|---|---:|---:|---:|---|
| `experiments/rank8.yaml` | 8 | 16 (scale=2.0) | 4,000 | `./adapters/rank8_experiment` |
| `experiments/rank16.yaml` | 16 | 32 (scale=2.0) | 4,000 | `./adapters/rank16_experiment` |
| `experiments/rank32.yaml` | 32 | 64 (scale=2.0) | 4,000 | `./adapters/rank32_experiment` |

### Command

```bash
./run_experiments.sh
```

This runs all three sequentially (~30-60 minutes total). Results are visible in the Streamlit app.

---

## 13. Full-FT Recovery Path (Safe Retrain)

Full fine-tuning can sometimes produce gibberish or unstable outputs (especially with aggressive learning rates). The safe-retrain workflow provides a staged recovery:

### Stage 1: Sanity Check (`experiments/full_safe_sanity.yaml`)

| Setting | Value |
|---|---|
| Iterations | 200 (very short) |
| Learning Rate | 2e-5 |
| Batch Size | 1 |
| Steps per Eval | 50 |
| Output | `./mlx_best_models/full_retrain` |

```bash
./run_full_retrain_safe.sh sanity
```

### Stage 2: Smoke Evaluation (20 prompts)

```bash
./run_full_smoke_eval.sh
```

This generates responses from the retrained model on 20 prompts and checks quality. If >20% of responses are "bad" (empty, repetitive, or incoherent), it fails fast.

### Stage 3: Scale Up (`experiments/full_safe_scale.yaml`)

| Setting | Value |
|---|---|
| Iterations | 1,200 |
| Learning Rate | 3e-5 |

```bash
./run_full_retrain_safe.sh scale
```

### Stage 4: Full Epoch (`experiments/full_safe_epoch1.yaml`)

| Setting | Value |
|---|---|
| Iterations | 4,000 |
| Batch Size | 2 |
| Grad Accumulation | 2 |

```bash
./run_full_retrain_safe.sh epoch1
```

All three stages write to the same output directory: `./mlx_best_models/full_retrain`. The smoke eval script checks this exact path.

---

## 14. Evaluation Pipeline — Deep Dive

The evaluation system compares 6 models using blind pairwise comparisons. It is designed to prevent bias and ensure statistical rigor.

### The 6 Models

Defined in `evaluation/models.json`:

| Name | Model Path | Adapter Path | Notes |
|---|---|---|---|
| `full_ft` | Hub TinyLlama (fp16) | `./mlx_best_models/full` | Full fine-tuned |
| `lora_ft` | Hub TinyLlama (fp16) | `./mlx_best_models/lora` | LoRA fine-tuned |
| `qlora_ft` | `./models/tinyllama-4bit-base` | `./mlx_best_models/qlora` | QLoRA fine-tuned |
| `base` | Hub TinyLlama (fp16) | None | Unmodified baseline |
| `qwen_1.8b` | `./models/qwen1.5-1.8b-chat-4bit` | None | External baseline |
| `phi_2` | `./models/phi-2-hf-4bit-mlx` | None | External baseline |

### The 9 Comparisons

Each fine-tuned model is compared against each of the three baselines:
```
full_ft  vs base,   full_ft  vs qwen_1.8b,   full_ft  vs phi_2
lora_ft  vs base,   lora_ft  vs qwen_1.8b,   lora_ft  vs phi_2
qlora_ft vs base,   qlora_ft vs qwen_1.8b,   qlora_ft vs phi_2
```

### Pipeline Stages

```
Stage 1: GENERATION
  For each model × each prompt:
    → Format prompt with chat template
    → Generate response (temperature=0.2, top_p=0.9, max_tokens=256)
    → Save to: evaluation/runs/<run_id>/responses/<model_name>.jsonl
  
  RNG is seeded ONCE per model (not per prompt) for realistic variation.

Stage 2: PAIRING
  For each comparison pair × each prompt:
    → Create blind judging task:
      - Randomly assign which model is "left" vs "right"
      - Balance: 250 left / 250 right per pair (counterbalanced)
      - Shuffle all tasks across pairs (seeded, reproducible)
    → Save to: evaluation/runs/<run_id>/pairing/judging_tasks.jsonl

Stage 3: JUDGING
  Either:
    a) Manual: human reads each task and writes winner
    b) Model: LM-as-judge (via LM Studio or MLX) auto-judges

Stage 4: SCORING
  → Merge judgments with key (unmask left/right → model names)
  → Compute: win rate, tie rate, effective win rate
  → Bootstrap 95% CI on tie-adjusted score
  → Generate markdown report
```

### Generation Parameters (Fixed Across All Models)

| Parameter | Value | Why |
|---|---|---|
| `temperature` | 0.2 | Low randomness for reproducible comparison |
| `top_p` | 0.9 | Nucleus sampling |
| `max_tokens` | 256 | Sufficient for instruction-following tasks |
| `seed` | 42 | Reproducible RNG (seeded once per model) |
| `max_prompts` | 200 (default) | Statistically robust; override with env var |

### Command

```bash
# Full default run (200 prompts, manual judging)
SEED=42 ./evaluation/run_eval_6models.sh --judge-mode manual --progress-every 20

# Quick smoke test (10 prompts)
SEED=42 MAX_PROMPTS=10 ./evaluation/run_eval_6models.sh --judge-mode manual --progress-every 5

# Full 500-prompt run
SEED=42 MAX_PROMPTS=500 ./evaluation/run_eval_6models.sh --judge-mode manual --progress-every 20

# Auto-judge with a local model
SEED=42 ./evaluation/run_eval_6models.sh \
  --judge-mode model \
  --judge-model "Qwen/Qwen2.5-3B-Instruct" \
  --progress-every 20
```

### Output Directory Structure

```
evaluation/runs/<timestamp>/
├── responses/
│   ├── full_ft.jsonl        # 200 generated responses
│   ├── lora_ft.jsonl
│   ├── qlora_ft.jsonl
│   ├── base.jsonl
│   ├── qwen_1.8b.jsonl
│   └── phi_2.jsonl
├── pairing/
│   ├── judging_tasks.jsonl  # Blind tasks for the judge
│   ├── judging_key.jsonl    # Unblinded key (model ↔ left/right)
│   └── pairing_summary.json # Left/right balance verification
└── scoring/
    ├── pair_metrics.json    # Per-pair results
    ├── pair_metrics.csv
    ├── model_rollup.json    # Per-model aggregate
    ├── model_rollup.csv
    └── report.md            # Human-readable report
```

---

## 15. Judging: Manual & LM-as-Judge

### Option A: Manual Judging

After generation + pairing, the pipeline stops and prints:

```
Manual judging mode selected.
Use tasks file: evaluation/runs/<run_id>/pairing/judging_tasks.jsonl
```

Each line in the tasks file shows:
```json
{
  "item_id": "abc123",
  "prompt": "Name five Mediterranean countries.",
  "left": "Greece, Italy, Spain, Turkey, France.",
  "right": "The Mediterranean region includes many countries..."
}
```

Create a judgments file with one line per task:
```json
{"item_id": "abc123", "winner": "left"}
{"item_id": "def456", "winner": "right"}
{"item_id": "ghi789", "winner": "tie"}
```

Valid winner values: `left`, `right`, `tie`, `invalid`.

Then score:
```bash
.venv/bin/python evaluation/score_judgments.py \
  --key evaluation/runs/<run_id>/pairing/judging_key.jsonl \
  --judgments evaluation/runs/<run_id>/pairing/judgments_manual.jsonl \
  --out-dir evaluation/runs/<run_id>/scoring \
  --seed 42
```

### Option B: LM-as-Judge (via LM Studio)

Start a local LM Studio server with a judge model loaded, then:

```bash
# Verify LM Studio is running
curl -s http://127.0.0.1:1234/v1/models

# Judge with the loaded model
.venv/bin/python evaluation/judge_with_lmstudio.py \
  --tasks evaluation/runs/<run_id>/pairing/judging_tasks.jsonl \
  --out evaluation/runs/<run_id>/pairing/judgments_lmstudio.jsonl \
  --model "ministral-3-14b-reasoning" \
  --base-url "http://127.0.0.1:1234/v1" \
  --progress-every 100
```

### Judge Parser Logic

The judge's text response is parsed to extract a winner:

1. **First word check**: If the response starts with "Left", "Right", "Tie", "A", or "B" → use that.
2. **Keyword counting**: If ambiguous, count occurrences of "left", "right", "response A", "response B", "tie", "draw". Only assign if one side dominates exclusively.
3. **Fallback**: If still ambiguous → "invalid".

This avoids the bias of simple substring checks like `"left" in text` (which would always match "left" before "right" in verbose outputs).

---

## 16. Scoring, Metrics & Reports

### Metrics Computed

For each pair (e.g., `lora_ft vs base`):

| Metric | Formula | Meaning |
|---|---|---|
| **Win Rate** | wins_model_1 / total_scored | Raw proportion of wins |
| **Effective Win Rate** | wins_model_1 / (wins_model_1 + wins_model_2) | Ignoring ties |
| **Tie Rate** | ties / total_scored | Proportion of ties |
| **95% CI (Tie-Adj Score)** | Bootstrap CI on [win=1, tie=0.5, loss=0] | Confidence interval on tie-adjusted score |

### Bootstrap CI Methodology

1. For each judged pair, assign scores: `win=1.0`, `tie=0.5`, `loss=0.0`.
2. Resample these scores 1,000 times (with replacement).
3. Compute the mean of each resample.
4. Report the 2.5th and 97.5th percentile of the 1,000 means.

The CI column in the report is labeled `95% CI (Tie-Adj Score)` to clarify it's computed on the tie-adjusted metric, not the raw win rate.

### Coverage Assertion

The scoring step verifies that ≥90% of judgments match an `item_id` in the key. If coverage drops below 90%, it raises an error (preventing silent data drops). Between 90-100%, it prints a warning.

### Release Gates

| Gate | Threshold | Meaning |
|---|---|---|
| Effective win rate | > 0.55 | Model wins more than 55% of decisive comparisons |
| CI lower bound | > 0.50 | We're 97.5% confident the true score is above 0.50 |

### Command

```bash
.venv/bin/python evaluation/score_judgments.py \
  --key evaluation/runs/<run_id>/pairing/judging_key.jsonl \
  --judgments evaluation/runs/<run_id>/pairing/judgments_manual.jsonl \
  --out-dir evaluation/runs/<run_id>/scoring \
  --seed 42
```

---

## 17. Streamlit Chat UI

### What It Provides

- **Single model mode**: Chat with any loaded model/adapter
- **Compare mode**: Side-by-side generation from two different models
- **Safety mode**: Injects a safety system prompt
- **Model selector**: Auto-discovers all adapters in `./adapters/` and full models in `./models/`
- **Generation controls**: Temperature and max token sliders
- **Logging**: All interactions saved to `./logs/chat_log.jsonl`

### Command

```bash
.venv/bin/streamlit run app.py
```

Opens at `http://localhost:8501` in your browser.

### Key UI Features

| Feature | How to Use |
|---|---|
| Load a model | Sidebar → Select model → Click "Load Model" |
| Compare two models | Sidebar → Toggle "Compare Mode" → Select Model A and B |
| Safety mode | Sidebar → Toggle "Safety Mode" |
| Change temperature | Sidebar → Temperature slider (0.0 = deterministic, 1.0 = creative) |
| Change max tokens | Sidebar → Max Tokens slider (64-512) |

---

## 18. Validation & Regression Guard

The `validate_training_setup.py` script checks for common misconfigurations. Run it **before** any sweep or evaluation:

```bash
.venv/bin/python validate_training_setup.py
```

### What It Checks

| Check | What Goes Wrong If Broken |
|---|---|
| LoRA/QLoRA config distinctness | Both train the same experiment (wasted compute) |
| `fine_tune_type` difference | QLoRA doesn't get quantized forward pass |
| Base model difference | Both use same base → identical adapters |
| `mask_prompt: true` | Model learns to predict prompts → worse response quality |
| Adapter weight identity (SHA256) | Identical adapters = broken pipeline |
| Eval model alignment | Adapters loaded on wrong base → degraded quality |
| Retrain path contract | Smoke eval can't find retrained model |
| `.venv/bin/python` in shell scripts | Wrong Python interpreter → missing packages |
| Data count sanity | Script declares N but disk has M → confusion |

### Exit Codes

- `0`: All checks passed ✅
- `1`: Critical error(s) found ❌ (do not proceed)

---

## 19. Complete File Map

```
PyCharmMiscProject/
│
├── 📦 DATA PREPARATION
│   ├── prepare_dataset.py           # Downloads Alpaca, formats to chat JSONL
│   └── data/                        # Generated dataset
│       ├── train.jsonl              # 4,000 training examples
│       ├── valid.jsonl              # 500 validation examples
│       └── test.jsonl               # 500 test examples
│
├── 🏋️ TRAINING CONFIGS
│   ├── lora_config.yaml             # LoRA: fp16 base + adapter [rank=16, scale=2.0]
│   ├── qlora_config.yaml            # QLoRA: 4-bit base + adapter [rank=16, scale=2.0]
│   └── full_config.yaml             # Full FT: all weights updated [lr=1e-5]
│
├── 🚀 TRAINING SCRIPTS
│   ├── run_train.sh                 # LoRA training launcher
│   ├── run_qlora.sh                 # QLoRA training launcher
│   ├── run_full.sh                  # Full FT training launcher
│   └── test_model.sh               # Quick inference test
│
├── 🔬 EXPERIMENTS
│   ├── experiments/
│   │   ├── rank8.yaml               # LoRA rank 8 experiment
│   │   ├── rank16.yaml              # LoRA rank 16 experiment
│   │   ├── rank32.yaml              # LoRA rank 32 experiment
│   │   ├── full_safe_sanity.yaml    # Full FT recovery: sanity check
│   │   ├── full_safe_scale.yaml     # Full FT recovery: scale up
│   │   └── full_safe_epoch1.yaml    # Full FT recovery: full epoch
│   ├── run_experiments.sh           # Run all rank experiments
│   ├── run_full_retrain_safe.sh     # Staged full FT recovery
│   └── run_full_smoke_eval.sh       # 20-prompt smoke gate
│
├── 🔍 HYPERPARAMETER SWEEP
│   ├── sweep_mlx_lora.py            # Grid/TPE sweep runner
│   ├── sweep_train.sh               # Manual rank × scale sweep (legacy)
│   ├── mlx_sweep_runs/              # All sweep run outputs
│   └── mlx_best_models/             # Best model per technique
│       ├── lora/
│       ├── qlora/
│       ├── full/
│       └── full_retrain/
│
├── 📊 EVALUATION
│   ├── evaluation/
│   │   ├── models.json              # 6-model registry for eval
│   │   ├── eval_prompts.jsonl       # 500 frozen eval prompts
│   │   ├── eval_references.jsonl    # Reference answers
│   │   ├── run_eval_6models.sh      # One-command eval wrapper
│   │   ├── run_pipeline.py          # CLI entry point
│   │   ├── pipeline.py              # Core: generation → pairing → scoring
│   │   ├── generate_responses.py    # Standalone response generation
│   │   ├── make_pairs.py            # Standalone pairing builder
│   │   ├── judge_with_lmstudio.py   # LM Studio judge integration
│   │   ├── judge_with_model.py      # MLX-native judge
│   │   ├── score_judgments.py       # Standalone scoring CLI
│   │   ├── check_response_quality.py # Smoke test quality gate
│   │   ├── runs/                    # All evaluation run outputs
│   │   └── runs_smoke/              # Smoke evaluation outputs
│   │
│   └── deep_search_audit.md         # Codebase audit findings
│
├── 🖥️ DEMO UI
│   ├── app.py                        # Streamlit chat application
│   └── logs/                         # Chat interaction logs
│       └── chat_log.jsonl
│
├── 🛡️ VALIDATION
│   └── validate_training_setup.py    # Pre-flight regression guard
│
├── ⚙️ PROJECT CONFIG
│   ├── requirements.txt              # Python dependencies
│   ├── pyrightconfig.json            # Type checker config
│   └── .gitignore
│
└── models/                           # Pre-downloaded/quantized base models
    ├── tinyllama-4bit-base/          # 4-bit TinyLlama (for QLoRA)
    ├── qwen1.5-1.8b-chat-4bit/      # Qwen baseline (eval only)
    └── phi-2-hf-4bit-mlx/           # Phi-2 baseline (eval only)
```

---

## 20. Every Command You'll Ever Need

### Setup

```bash
# Create environment
python3 -m venv .venv
source .venv/bin/activate
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install optuna  # Optional, for TPE sweeps

# Prepare dataset
.venv/bin/python prepare_dataset.py

# Validate setup before training
.venv/bin/python validate_training_setup.py
```

### Training

```bash
# LoRA (~10-20 min)
./run_train.sh

# QLoRA (~8-15 min)
./run_qlora.sh

# Full Fine-Tuning (~30-60 min, needs 16 GB+)
./run_full.sh

# LoRA rank experiments (rank 8, 16, 32 — ~60 min total)
./run_experiments.sh
```

### Hyperparameter Sweeps

```bash
# Grid search — all techniques (~2-4 hours)
.venv/bin/python sweep_mlx_lora.py --technique all --search grid

# Grid search — single technique
.venv/bin/python sweep_mlx_lora.py --technique lora --search grid

# TPE search — 20 trials
.venv/bin/python sweep_mlx_lora.py --technique qlora --search tpe --n-trials 20
```

### Full-FT Recovery

```bash
./run_full_retrain_safe.sh sanity    # Quick sanity check
./run_full_smoke_eval.sh              # 20-prompt quality gate
./run_full_retrain_safe.sh scale     # Scale up training
./run_full_retrain_safe.sh epoch1    # Full epoch
```

### Evaluation

```bash
# Quick test (10 prompts)
SEED=42 MAX_PROMPTS=10 ./evaluation/run_eval_6models.sh --judge-mode manual --progress-every 5

# Default run (200 prompts)
SEED=42 ./evaluation/run_eval_6models.sh --judge-mode manual --progress-every 20

# Full run (500 prompts)
SEED=42 MAX_PROMPTS=500 ./evaluation/run_eval_6models.sh --judge-mode manual --progress-every 20

# Auto-judge with local model
SEED=42 ./evaluation/run_eval_6models.sh \
  --judge-mode model \
  --judge-model "Qwen/Qwen2.5-3B-Instruct" \
  --progress-every 20
```

### LM Studio Judging (existing runs)

```bash
RUN_ID=<timestamp>

# Build pairs (if not already done)
.venv/bin/python evaluation/make_pairs.py \
  --responses-dir "evaluation/runs/${RUN_ID}/responses" \
  --out-dir "evaluation/runs/${RUN_ID}/pairing" \
  --pairs "full_ft:base,lora_ft:base,qlora_ft:base" \
  --seed 42

# Judge via LM Studio
.venv/bin/python evaluation/judge_with_lmstudio.py \
  --tasks "evaluation/runs/${RUN_ID}/pairing/judging_tasks.jsonl" \
  --out "evaluation/runs/${RUN_ID}/pairing/judgments_lmstudio.jsonl" \
  --model "ministral-3-14b-reasoning" \
  --base-url "http://127.0.0.1:1234/v1" \
  --progress-every 100

# Score
.venv/bin/python evaluation/score_judgments.py \
  --key "evaluation/runs/${RUN_ID}/pairing/judging_key.jsonl" \
  --judgments "evaluation/runs/${RUN_ID}/pairing/judgments_lmstudio.jsonl" \
  --out-dir "evaluation/runs/${RUN_ID}/scoring" \
  --seed 42
```

### Quick Inference Test

```bash
.venv/bin/python -m mlx_lm.generate \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter-path ./adapters/tinyllama-lora-alpaca \
  --prompt "Explain gradient descent in simple terms." \
  --max-tokens 256
```

### Streamlit Demo

```bash
.venv/bin/streamlit run app.py
```

---

## 21. Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: mlx` | Wrong Python interpreter | Use `.venv/bin/python`, not system `python3` |
| `TokenizersBackend does not exist` | Library version mismatch | Already handled in pipeline — safe to ignore |
| `No safetensors found` for Phi-2 | Wrong path in models.json | Verify `./models/phi-2-hf-4bit-mlx` exists |
| Full FT outputs gibberish | Unstable training | Use safe retrain workflow (Section 13) |
| Eval looks stalled | No progress output | Add `--progress-every 20` |
| LoRA and QLoRA give identical results | Same base model used | Run `validate_training_setup.py` to check |
| `Judgment coverage too low` error | Pairing/judging ID mismatch | Regenerate pairs using same responses |
| Out of memory during Full FT | Not enough unified memory | Use LoRA or QLoRA instead |
| `adapter_path not found` during eval | Training output path mismatch | Check paths in `evaluation/models.json` |
| `prepare_dataset.py` gives different count | NUM_EXAMPLES changed | Delete `./data/` and rerun |

---

## 22. Design Decisions & Audit Trail

This codebase underwent a comprehensive audit (`deep_search_audit.md`) that identified and fixed 10 issues:

| # | Severity | Issue | Fix Applied |
|---|---|---|---|
| 1 | **Critical** | LoRA and QLoRA used same base model + same `fine_tune_type` | LoRA → hub fp16, QLoRA → 4-bit local; `fine_tune_type: qlora` |
| 2 | **Critical** | Eval loaded adapters on wrong base model | `models.json` now matches training base per technique |
| 3 | **High** | Full retrain wrote to `full` but smoke eval expected `full_retrain` | All retrain configs → `./mlx_best_models/full_retrain` |
| 4 | **High** | CI column labeled generically as `95% CI` | Now `95% CI (Tie-Adj Score)` |
| 5 | **High** | Best model selection used last loss (not best) | Now uses minimum validation loss across checkpoints |
| 6 | **Medium** | `mask_prompt: false` in trained adapters | Set `mask_prompt: true` everywhere |
| 7 | **Medium** | Judge parser had order-dependent substring bias | Rewritten with first-word + keyword-counting approach |
| 8 | **Medium** | RNG reseeded per prompt; only 50 eval prompts | Seed once per model; default 200 prompts |
| 9 | **Low** | Shell scripts used `python3` (system) | All use `.venv/bin/python` now |
| 10 | **Low** | `prepare_dataset.py` said 10K but disk had 5K | `NUM_EXAMPLES = 5000` |

The automated regression guard (`validate_training_setup.py`) prevents these issues from recurring.

---

## License

This project is for educational and research purposes. The Alpaca dataset is subject to its own license terms. TinyLlama is released under the Apache 2.0 license.
]]>
