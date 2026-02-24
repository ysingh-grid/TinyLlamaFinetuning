# TinyLlama Finetuning Product (MLX)

End-to-end project to fine-tune TinyLlama on Alpaca-style data, compare training strategies (`full`, `lora`, `qlora`), evaluate with blind pairwise win-rate metrics, and ship a validated demo experience in Streamlit.

## Product Goal

Build a locally reproducible pipeline on Apple Silicon that moves from raw dataset to a demo-ready assistant.

Definition of done:
- At least one fine-tuned model beats the base model on the frozen eval set.
- Win-rate gates are met on the 500-prompt evaluation run.
- The promoted model passes qualitative checks in the Streamlit UI.

Suggested release gates:
- `effective_win_rate > 0.55`
- `ci95_low > 0.50`

## End-to-End Flow

1. Prepare dataset splits.
2. Train model variants (`lora`, `qlora`, `full`).
3. If full FT degrades, run safe-retrain recovery + smoke gate.
4. Run fixed-parameter 6-model pairwise evaluation.
5. Judge and score outputs.
6. Promote winning model(s) and validate in Streamlit.

## Repository Map

Core files:
- `prepare_dataset.py`: create `data/train.jsonl`, `data/valid.jsonl`, `data/test.jsonl`
- `run_train.sh`: LoRA training (`lora_config.yaml`)
- `run_qlora.sh`: QLoRA training (`qlora_config.yaml`)
- `run_full.sh`: full fine-tuning (`full_config.yaml`)
- `run_experiments.sh`: rank experiments (`experiments/rank8.yaml`, `rank16.yaml`, `rank32.yaml`)
- `sweep_mlx_lora.py`: scripted sweep search
- `app.py`: Streamlit product UI

Evaluation:
- `evaluation/run_eval_6models.sh`: one-command fixed pipeline wrapper
- `evaluation/run_pipeline.py`: generation -> pairing -> judging -> scoring
- `evaluation/models.json`: model registry for eval runs
- `evaluation/eval_prompts.jsonl`: frozen 500-prompt eval set
- `evaluation/score_judgments.py`: win rates + confidence intervals + report

Recovery tools (full FT):
- `run_full_retrain_safe.sh`
- `run_full_smoke_eval.sh`
- `experiments/full_safe_sanity.yaml`
- `experiments/full_safe_scale.yaml`
- `experiments/full_safe_epoch1.yaml`

## 1) Environment Setup

Prerequisites:
- macOS on Apple Silicon (MLX/Metal)
- Python 3.10+

Create and activate virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
.venv/bin/pip install -r requirements.txt
```

Optional for TPE sweeps:

```bash
.venv/bin/pip install optuna
```

## 2) Data Preparation

Build local train/valid/test from the first 10k Alpaca train examples:

```bash
.venv/bin/python prepare_dataset.py
```

Expected outputs:
- `data/train.jsonl` (~8000)
- `data/valid.jsonl` (~1000)
- `data/test.jsonl` (~1000)

## 3) Train Candidate Models

Run all three strategies so they can be compared fairly.

LoRA:

```bash
./run_train.sh
```

QLoRA:

```bash
./run_qlora.sh
```

Full FT:

```bash
./run_full.sh
```

Important:
- The shell scripts invoke `python3`, so run them from an activated `.venv`.
- Default output paths in configs:
  - `lora_config.yaml` -> `./adapters/tinyllama-lora-alpaca`
  - `qlora_config.yaml` -> `./adapters/tinyllama-qlora-alpaca`
  - `full_config.yaml` -> `./models/tinyllama-full-alpaca`

## 4) Optional: Rank Experiments and Sweeps

Rank experiments:

```bash
./run_experiments.sh
```

Programmatic sweep (grid):

```bash
.venv/bin/python sweep_mlx_lora.py --technique all --search grid
```

TPE example:

```bash
.venv/bin/python sweep_mlx_lora.py --technique lora --search tpe --n-trials 20
```

## 5) Full FT Recovery Path (If Quality Collapses)

If full FT starts producing gibberish or unstable outputs, use this path before full eval:

1. Sanity retrain:

```bash
./run_full_retrain_safe.sh sanity
```

2. 20-prompt smoke generation + quality gate:

```bash
./run_full_smoke_eval.sh
```

3. Scale retrain:

```bash
./run_full_retrain_safe.sh scale
```

4. One-epoch retrain:

```bash
./run_full_retrain_safe.sh epoch1
```

Smoke gate checks `./mlx_best_models/full_retrain` and fails fast if bad-response rate is too high.

## 6) Standardized Evaluation (500 Prompt Frozen Set)

Use the fixed 6-model setup in `evaluation/models.json`:
- `full_ft`
- `lora_ft`
- `qlora_ft`
- `base`
- `qwen_1.8b`
- `phi_2`

Run end-to-end evaluation wrapper:

```bash
SEED=42 ./evaluation/run_eval_6models.sh --judge-mode manual --progress-every 20
```

What this does:
- Generates responses for all models with fixed params (`temperature=0.2`, `top_p=0.9`, `max_tokens=256`).
- Builds blind pairs for 9 required comparisons.
- Stops in manual mode after creating judging tasks.
- Counterbalances left/right per pair and shuffles final task order (seeded, reproducible).

Run outputs are stored under:
- `evaluation/runs/<run_id>/responses`
- `evaluation/runs/<run_id>/pairing`
- `evaluation/runs/<run_id>/scoring` (after scoring)

### 6A) How to Run Commands Like This

Example command:

```bash
SEED=42 MAX_PROMPTS=10 ./evaluation/run_eval_6models.sh --judge-mode manual --progress-every 5
```

Breakdown:
- `SEED=42`: fixed random seed for reproducible generations.
- `MAX_PROMPTS=10`: temporary prompt cap for faster testing.
- `./evaluation/run_eval_6models.sh`: wrapper for fixed 6-model evaluation settings.
- `--judge-mode manual`: stop after building blind tasks and wait for human judgments.
- `--progress-every 5`: print generation progress every 5 prompts per model.

What is fixed by the wrapper:
- model list from `evaluation/models.json`
- pair list (9 required fine-tune vs baseline comparisons)
- generation params: `temperature=0.2`, `top_p=0.9`, `max_tokens=256`

Useful variants:

Quick smoke test:

```bash
SEED=42 MAX_PROMPTS=10 ./evaluation/run_eval_6models.sh --judge-mode manual --progress-every 5
```

Medium test:

```bash
SEED=42 MAX_PROMPTS=50 ./evaluation/run_eval_6models.sh --judge-mode manual --progress-every 10
```

Full run (frozen 500 prompts):

```bash
SEED=42 MAX_PROMPTS=500 ./evaluation/run_eval_6models.sh --judge-mode manual --progress-every 20
```

Model-judge run (automatic judging):

```bash
SEED=42 MAX_PROMPTS=50 ./evaluation/run_eval_6models.sh \
  --judge-mode model \
  --judge-model "Qwen/Qwen2.5-3B-Instruct" \
  --progress-every 10
```

Notes:
- A new output folder is created on each run under `evaluation/runs/<timestamp>/`.
- `--overwrite` is only needed when intentionally reusing an existing run directory.

## 7) Manual Judging and Scoring

Use generated tasks:
- `evaluation/runs/<run_id>/pairing/judging_tasks.jsonl`
- `evaluation/runs/<run_id>/pairing/pairing_summary.json` (verify left/right balance before judging)

Create manual judgments file:
- `evaluation/runs/<run_id>/pairing/judgments_manual.jsonl`

Each JSONL line:

```json
{"item_id":"...","winner":"left|right|tie|invalid"}
```

Score results:

```bash
.venv/bin/python evaluation/score_judgments.py \
  --key evaluation/runs/<run_id>/pairing/judging_key.jsonl \
  --judgments evaluation/runs/<run_id>/pairing/judgments_manual.jsonl \
  --out-dir evaluation/runs/<run_id>/scoring \
  --seed 42
```

Scoring outputs:
- `pair_metrics.json`
- `pair_metrics.csv`
- `model_rollup.json`
- `model_rollup.csv`
- `report.md`

## 8) Decision: Promote or Iterate

If gates are met:
- Promote the winning adapter/model into the active eval registry and demo flow.
- Run final qualitative checks in Streamlit.

If gates are not met:
1. Adjust configs (rank, LR, iterations, batch/grad accumulation).
2. Retrain candidate(s).
3. Re-run the same fixed eval pipeline.

## 9) Product Validation in Streamlit

Start UI:

```bash
.venv/bin/streamlit run app.py
```

Validate:
- Single-model quality on common user tasks
- Side-by-side comparison mode
- Safety mode behavior and refusals
- Latency/usability for demo readiness

## 10) Reproducibility Checklist

- Keep eval prompt set fixed (`evaluation/eval_prompts.jsonl`).
- Keep model list fixed for a given comparison run (`evaluation/models.json`).
- Keep generation parameters fixed across compared models.
- Record seed and `run_id`.
- Track config files used for each trained adapter/model.

## 11) Common Issues

`TokenizersBackend does not exist`
- Compatibility shim is already handled in the evaluation pipeline.

`No safetensors found` for Phi-2
- Confirm `evaluation/models.json` points to `./models/phi-2-hf-4bit-mlx`.

Full FT outputs are incoherent
- Run the safe retrain workflow and pass smoke gate before 500-prompt eval.

Eval looks stalled
- Use `--progress-every` (for example `--progress-every 20`).

## 12) Additional Documentation

- `evaluation/README.md`: deeper evaluation details
- `evaluation/UPDATE_2026-02-23.md`: latest evaluation changes

## 13) Eval Existing Generated Runs (LM Studio Minstral 3 14B)

Use this flow when `evaluation/runs/<run_id>/responses/*.jsonl` already exists and you want to finish judging + scoring.
For this machine, `gpt-oss-20b` did not fit memory in LM Studio, so the judge model is `ministral-3-14b-reasoning`.

```bash
# 0) From repo root
cd /Users/ysingh/PyCharmMiscProject
source .venv/bin/activate

# 1) Pick the run to evaluate
RUN_ID=20260224_103910
RUN_DIR="evaluation/runs/${RUN_ID}"

# 2) (If not already done) build blind pair tasks
PAIRS="full_ft:base,full_ft:qwen_1.8b,full_ft:phi_2,lora_ft:base,lora_ft:qwen_1.8b,lora_ft:phi_2,qlora_ft:base,qlora_ft:qwen_1.8b,qlora_ft:phi_2"
.venv/bin/python evaluation/make_pairs.py \
  --responses-dir "${RUN_DIR}/responses" \
  --out-dir "${RUN_DIR}/pairing" \
  --pairs "${PAIRS}" \
  --seed 42

# 3) Start LM Studio server (UI), load Minstral 3 14B Reasoning, then verify API
curl -s http://127.0.0.1:1234/v1/models

# 4) Judge with local Minstral 3 14B Reasoning through LM Studio
.venv/bin/python evaluation/judge_with_lmstudio.py \
  --tasks "${RUN_DIR}/pairing/judging_tasks.jsonl" \
  --out "${RUN_DIR}/pairing/judgments_oss20b.jsonl" \
  --model "ministral-3-14b-reasoning" \
  --base-url "http://127.0.0.1:1234/v1" \
  --progress-every 100

# 5) Score + report
.venv/bin/python evaluation/score_judgments.py \
  --key "${RUN_DIR}/pairing/judging_key.jsonl" \
  --judgments "${RUN_DIR}/pairing/judgments_oss20b.jsonl" \
  --out-dir "${RUN_DIR}/scoring" \
  --seed 42

# 6) View outputs
ls -lh "${RUN_DIR}/scoring"
cat "${RUN_DIR}/scoring/report.md"
```

## 14) Next Steps After Failed Eval Run

Use this sequence when FT variants underperform baselines (as in run `20260223_143739`):

1. Freeze this run as failed and do not promote any FT checkpoint.
2. Inspect scored artifacts:
   - `evaluation/runs/20260223_143739/scoring/report.md`
   - `evaluation/runs/20260223_143739/scoring/analysis.md`
3. Re-run safe full-FT recovery:
```bash
./run_full_retrain_safe.sh sanity
./run_full_smoke_eval.sh
./run_full_retrain_safe.sh scale
./run_full_retrain_safe.sh epoch1
```
4. Re-train LoRA/QLoRA with conservative settings (lower LR, fewer simultaneous changes).
5. Re-run the fixed evaluation pipeline with same prompts/pairs/seed.
6. Only promote if effective win rate clears `0.55` and CI lower bound clears `0.50`.

Quick rerun command set:

```bash
SEED=42 ./evaluation/run_eval_6models.sh --judge-mode manual --progress-every 20
```
