# TinyLlama Finetuning + Evaluation Loop

This repository contains TinyLlama fine-tuning workflows (LoRA, QLoRA, full FT), a reproducible pairwise evaluation pipeline, and a Streamlit app for interactive model comparison.

## Goal
Run a disciplined loop:
1. Train models
2. Evaluate on fixed prompts with pairwise win-rate metrics
3. Iterate training until win-rate targets are met
4. Move to Streamlit UI only after targets are satisfied

## Current Model Set
- `full_ft`
- `lora_ft`
- `qlora_ft`
- `base`
- `qwen_1.8b`
- `phi_2`

Configured in: `evaluation/models.json`.

## Environment Setup
```bash
source .venv/bin/activate
```

If dependencies are missing:
```bash
.venv/bin/pip install -r requirements.txt
```

## Data
- Training split: `data/train.jsonl` (4000 rows)
- Validation split: `data/valid.jsonl` (500 rows)
- Test split: `data/test.jsonl` (500 rows)
- Frozen eval set: `evaluation/eval_prompts.jsonl` (500 prompts)
- Frozen references: `evaluation/eval_references.jsonl`

Eval set source is documented in `evaluation/README.md` and `evaluation/UPDATE_2026-02-23.md`.

## Checkpoints (Execution Plan)

### Checkpoint 1: Baseline Setup
- [x] Freeze evaluation prompts and references
- [x] Define 6-model evaluation config
- [x] Fix generation params for fair comparison (`temperature=0.2`, `top_p=0.9`, same `max_tokens`)
- [x] Set explicit seed in eval wrapper (`SEED`, default `42`)

### Checkpoint 2: Safe Full-FT Recovery Path
- [x] Add safe full-FT configs:
  - `experiments/full_safe_sanity.yaml`
  - `experiments/full_safe_scale.yaml`
  - `experiments/full_safe_epoch1.yaml`
- [x] Add safe retrain runner: `run_full_retrain_safe.sh`
- [x] Add 20-prompt smoke gate: `run_full_smoke_eval.sh`
- [x] Add response quality guard: `evaluation/check_response_quality.py`

### Checkpoint 3: Evaluation Pipeline
- [x] Generate responses for all configured models
- [x] Build blind pairwise tasks (9 required pairs)
- [x] Score win-rate metrics with bootstrap CI
- [x] Add per-model generation progress logging (`--progress-every`)

### Checkpoint 4: Iterative Train-Eval Loop
- [ ] Run full retrain sanity pass
- [ ] Pass smoke gate for retrained full model
- [ ] Run scaled/full epoch retrain
- [ ] Re-run full 500-prompt evaluation
- [ ] Hit win-rate target criteria

### Checkpoint 5: Streamlit UI
- [ ] Promote winning model set to UI comparison
- [ ] Validate response quality in manual interactive checks
- [ ] Finalize demo/release

## Train -> Eval Loop (Repeat Until Satisfied)

### Step A: Retrain (safe full-FT path)
```bash
./run_full_retrain_safe.sh sanity
./run_full_smoke_eval.sh
./run_full_retrain_safe.sh scale
./run_full_retrain_safe.sh epoch1
```

### Step B: Run 6-model evaluation
```bash
SEED=42 ./evaluation/run_eval_6models.sh --judge-mode manual --progress-every 20
```

### Step C: Manual judging
Use generated tasks:
- `evaluation/runs/<run_id>/pairing/judging_tasks.jsonl`

Create judgments file:
- `evaluation/runs/<run_id>/pairing/judgments_manual.jsonl`

Accepted labels:
- `A` / `B` / `Tie` (normalized by scorer)

Rubric (strict):
- instruction-following
- correctness
- relevance/helpfulness

### Step D: Score metrics
```bash
.venv/bin/python evaluation/score_judgments.py \
  --key evaluation/runs/<run_id>/pairing/judging_key.jsonl \
  --judgments evaluation/runs/<run_id>/pairing/judgments_manual.jsonl \
  --out-dir evaluation/runs/<run_id>/scoring \
  --seed 42
```

### Step E: Decide pass/fail
Per pair, track:
- win rate
- effective win rate (no ties)
- tie rate
- 95% bootstrap CI

Suggested gate:
- `effective_win_rate > 0.55`
- `ci95_low > 0.50`

If not met, tune/retrain and repeat from Step A.

## When to Move to Streamlit UI
Move only after Checkpoint 4 passes (target win-rates achieved).

Run app:
```bash
.venv/bin/streamlit run app.py
```

Use UI for:
- side-by-side qualitative checks
- safety prompt behavior checks
- regression sanity before finalizing

## Important Notes
- Keep seeds fixed for reproducibility.
- Do not evaluate with changing prompt set mid-loop.
- Exclude large artifacts from Git pushes (models/checkpoints/runs).
- If `full_ft` outputs collapse (gibberish), use the safe retrain sequence and smoke gate before full eval.

## Key Files
- Training/sweep: `sweep_mlx_lora.py`
- Safe full retrain: `run_full_retrain_safe.sh`
- Smoke eval: `run_full_smoke_eval.sh`
- Eval wrapper (fixed params): `evaluation/run_eval_6models.sh`
- Eval docs: `evaluation/README.md`
- Dated change log: `evaluation/UPDATE_2026-02-23.md`
