# Evaluation Pipeline

This folder provides an end-to-end pairwise evaluation workflow:

1. Generate responses for each model on a shared prompt set.
2. Create blind pairwise judging tasks.
3. Judge tasks (manually or with a local judge model).
4. Score win rates and produce a report.

## Current Frozen Eval Set

The current project eval set is already frozen and stored in:

- `evaluation/eval_prompts.jsonl`
- `evaluation/eval_references.jsonl`

Details:

- Source dataset: `ysingh-aiml/alpaca`
- Source split: `train`
- Source row range: `row_idx 10000..10499` (500 rows total)
- `evaluation/eval_prompts.jsonl` includes both `prompt` and `reference`.
- `evaluation/eval_references.jsonl` stores `prompt_id` + `reference` only.

## Fixed Generation Params

For apples-to-apples model comparison, use fixed generation settings for all models:

- `temperature=0.2`
- `top_p=0.9`
- `max_tokens=256`

Use the provided wrapper:

```bash
./evaluation/run_eval_6models.sh --judge-mode manual
```

You can still override via env vars if needed:

```bash
MAX_TOKENS=192 TEMPERATURE=0.2 TOP_P=0.9 ./evaluation/run_eval_6models.sh --judge-mode manual
```

## File Formats

### Models config (`models.json`)
JSON array of model specs:

```json
[
  {
    "name": "lora_ft",
    "model": "./models/tinyllama-4bit-base",
    "adapter_path": "./mlx_best_models/lora"
  }
]
```

Fields:
- `name` (required): unique model key used in outputs and pair specs.
- `model` (required): model ID/path passed to `mlx_lm.load`.
- `adapter_path` (optional): adapter directory for fine-tuned runs.
- `system_prompt` (optional): prepended to every prompt for this model.

### Prompts (`prompts.jsonl`)
Each line:

```json
{"prompt_id":"p001","category":"reasoning","prompt":"..."}
```

Required:
- `prompt`.

Optional:
- `prompt_id` (auto-generated if missing)
- `category`
- `reference`

For this repo's frozen set, use `evaluation/eval_prompts.jsonl`.

### Manual judgments (`judgments.jsonl`)
Each line:

```json
{"item_id":"lora_ft__vs__base__p001","winner":"left"}
```

`winner` accepted values: `left`, `right`, `tie`, `invalid`.
Aliases like `a`, `b`, `draw`, `skip` are normalized.

## Step-by-Step Usage

### 1) Generate responses

```bash
.venv/bin/python evaluation/generate_responses.py \
  --models-config evaluation/models.json \
  --prompts evaluation/eval_prompts.jsonl \
  --out-dir evaluation/runs/my_run/responses \
  --temperature 0.2 --top-p 0.9 --max-tokens 256 --seed 42
```

### 2) Build blind pairs

```bash
.venv/bin/python evaluation/make_pairs.py \
  --responses-dir evaluation/runs/my_run/responses \
  --out-dir evaluation/runs/my_run/pairing \
  --pairs "full_ft:base,full_ft:qwen_1.8b,full_ft:phi_2,lora_ft:base,lora_ft:qwen_1.8b,lora_ft:phi_2,qlora_ft:base,qlora_ft:qwen_1.8b,qlora_ft:phi_2" \
  --seed 42
```

Outputs:
- `judging_tasks.jsonl` (blind left/right responses)
- `judging_key.jsonl` (left/right -> model mapping for scoring)
- `pairing_summary.json` (audit of per-pair left/right balance + randomization settings)

Pairing defaults:
- Left/right placement is counterbalanced per pair (difference at most 1 item).
- Final judging task order is shuffled across pairs/prompts.

Optional flags:
- `--no-counterbalance-sides`
- `--no-shuffle-tasks`

### 3A) Manual judging

Create `evaluation/runs/my_run/pairing/judgments_manual.jsonl` with `item_id` + `winner`.

### 3B) Model-based judging

```bash
.venv/bin/python evaluation/judge_with_model.py \
  --tasks evaluation/runs/my_run/pairing/judging_tasks.jsonl \
  --out evaluation/runs/my_run/pairing/judgments_model.jsonl \
  --judge-model "Qwen/Qwen2.5-3B-Instruct" \
  --seed 42
```

### 4) Score and report

```bash
.venv/bin/python evaluation/score_judgments.py \
  --key evaluation/runs/my_run/pairing/judging_key.jsonl \
  --judgments evaluation/runs/my_run/pairing/judgments_manual.jsonl \
  --out-dir evaluation/runs/my_run/scoring \
  --seed 42
```

Outputs:
- `pair_metrics.json`
- `pair_metrics.csv`
- `model_rollup.json`
- `model_rollup.csv`
- `report.md`

## One-command Run

Manual mode (stops after pairing unless `--judgments` is provided):

```bash
.venv/bin/python evaluation/run_pipeline.py \
  --models-config evaluation/models.json \
  --prompts evaluation/eval_prompts.jsonl \
  --pairs "full_ft:base,full_ft:qwen_1.8b,full_ft:phi_2,lora_ft:base,lora_ft:qwen_1.8b,lora_ft:phi_2,qlora_ft:base,qlora_ft:qwen_1.8b,qlora_ft:phi_2" \
  --judge-mode manual
```

Model-judge mode:

```bash
.venv/bin/python evaluation/run_pipeline.py \
  --models-config evaluation/models.json \
  --prompts evaluation/eval_prompts.jsonl \
  --pairs "full_ft:base,full_ft:qwen_1.8b,full_ft:phi_2,lora_ft:base,lora_ft:qwen_1.8b,lora_ft:phi_2,qlora_ft:base,qlora_ft:qwen_1.8b,qlora_ft:phi_2" \
  --judge-mode model \
  --judge-model "Qwen/Qwen2.5-3B-Instruct"
```

## Notes

- The generation and model-judge stages require `mlx` and `mlx_lm`.
- External model IDs must be compatible with your local `mlx_lm` setup.
- Win-rate confidence intervals are bootstrap-based (1,000 resamples).

## Full FT Retrain Workflow

Manual full-FT retrain wrappers/config files were removed. Use sweep-generated full configs plus early stopping:

```bash
.venv/bin/python sweep_finetune.py --technique full --search grid --early-stop-patience 5 --early-stop-min-delta 0.0
```

To rerun a specific full trial with the smart wrapper:

```bash
.venv/bin/python smart_train.py --config mlx_sweep_runs/full/trial_0001/config.yaml --patience 5 --min-delta 0.0
```
