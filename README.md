# TinyLlama Instruction-Tuning Pipeline (MLX, Apple Silicon)

Local instruction-tuning and evaluation stack for TinyLlama using MLX.

Pipeline:

- Build dataset from Alpaca
- Train LoRA / QLoRA / Full FT variants
- Run pairwise evaluation (manual or model judge)
- Run cosine-similarity ranking as a fast alternative eval
- Test interactively in Streamlit

## Repository Overview

Primary scripts:

- `prepare_dataset.py`: builds train/valid/test JSONL files
- `run_train.sh`: LoRA training entrypoint
- `run_qlora.sh`: QLoRA training entrypoint
- `smart_train.py`: early-stopping wrapper around `mlx_lm.lora`
- `sweep_finetune.py`: grid/TPE hyperparameter sweeps
- `evaluation/run_eval_6models.sh`: one-command 6-model pairwise eval
- `evaluation/score_similarity_rankings.py`: cosine-similarity ranking report
- `app.py`: Streamlit chat playground

## Requirements

- Apple Silicon Mac
- Python 3.10+
- Local disk for model/artifact files (often 15GB+)

Base dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
.venv/bin/pip install -r requirements.txt
```

Optional dependencies:

- For TPE sweeps:

```bash
.venv/bin/pip install optuna
```

- For cosine similarity evaluation:

```bash
.venv/bin/pip install sentence-transformers nltk numpy
```

## Quick Start

1. Prepare dataset.

```bash
.venv/bin/python prepare_dataset.py
```

2. Train LoRA.

```bash
./run_train.sh
```

3. Quick generation test.

```bash
./test_model.sh "What is reinforcement learning?"
```

4. Launch app.

```bash
.venv/bin/streamlit run app.py
```

## Dataset Process

`prepare_dataset.py` does the following:

- Loads `tatsu-lab/alpaca`
- Drops empty outputs
- Sorts by output length
- Selects top `NUM_EXAMPLES=5000`
- Converts to MLX chat format
- Splits 80/10/10 into:
  - `data/train.jsonl`
  - `data/valid.jsonl`
  - `data/test.jsonl`

If previous split files exist, they are moved to `data/backup_v1/` once.

## Training Workflows

### LoRA

```bash
./run_train.sh
```

Config: `lora_config.yaml`  
Output: `adapters/tinyllama-lora-alpaca`

### QLoRA

```bash
./run_qlora.sh
```

Config: `qlora_config.yaml`  
Output: `adapters/tinyllama-qlora-alpaca`

### Full FT (from generated sweep config)

```bash
.venv/bin/python smart_train.py --config mlx_sweep_runs/full/trial_0001/config.yaml --patience 5 --min-delta 0.0
```

### Rank Experiments

```bash
./run_experiments.sh
```

Runs:

- `experiments/rank8.yaml`
- `experiments/rank16.yaml`
- `experiments/rank32.yaml`

## Hyperparameter Sweeps

All techniques (grid):

```bash
.venv/bin/python sweep_finetune.py --technique all --search grid
```

Single technique:

```bash
.venv/bin/python sweep_finetune.py --technique lora --search grid
```

TPE search:

```bash
.venv/bin/python sweep_finetune.py --technique qlora --search tpe --n-trials 20
```

Artifacts:

- Trial outputs: `mlx_sweep_runs/`
- Best adapters: `mlx_best_models/{lora,qlora,full}`

## Pairwise Evaluation (Primary)

Detailed eval docs: `evaluation/README.md`.

### One-command run

Manual judge mode:

```bash
SEED=42 ./evaluation/run_eval_6models.sh --judge-mode manual --progress-every 20
```

Model judge mode:

```bash
SEED=42 ./evaluation/run_eval_6models.sh \
  --judge-mode model \
  --judge-model "Qwen/Qwen2.5-3B-Instruct" \
  --progress-every 20
```

Limit prompts:

```bash
SEED=42 MAX_PROMPTS=25 ./evaluation/run_eval_6models.sh --judge-mode manual --progress-every 5
```

### Manual pair + score flow

```bash
.venv/bin/python evaluation/make_pairs.py \
  --responses-dir evaluation/runs/<run_id>/responses \
  --out-dir evaluation/runs/<run_id>/pairing \
  --pairs "full_ft:base,full_ft:qwen_1.8b,full_ft:phi_2,lora_ft:base,lora_ft:qwen_1.8b,lora_ft:phi_2,qlora_ft:base,qlora_ft:qwen_1.8b,qlora_ft:phi_2" \
  --seed 42

.venv/bin/python evaluation/score_judgments.py \
  --key evaluation/runs/<run_id>/pairing/judging_key.jsonl \
  --judgments evaluation/runs/<run_id>/pairing/judgments_manual.jsonl \
  --out-dir evaluation/runs/<run_id>/scoring \
  --seed 42
```

### LM Studio judge flow

```bash
.venv/bin/python evaluation/judge_with_lmstudio.py \
  --tasks evaluation/runs/<run_id>/pairing/judging_tasks.jsonl \
  --out evaluation/runs/<run_id>/pairing/judgments_lmstudio.jsonl \
  --model "ministral-3-14b-reasoning" \
  --base-url "http://127.0.0.1:1234/v1" \
  --progress-every 100
```

## Cosine Similarity Evaluation (Alternative)

Use this when you want a faster, reference-based ranking alternative to LLM judging.

1. Generate responses (if not already generated):

```bash
.venv/bin/python evaluation/generate_responses.py \
  --models-config evaluation/models.json \
  --prompts evaluation/eval_prompts.jsonl \
  --out-dir evaluation/runs/<run_id>/responses \
  --temperature 0.2 --top-p 0.9 --max-tokens 256 --seed 42
```

2. Run cosine ranking:

```bash
.venv/bin/python evaluation/score_similarity_rankings.py \
  --responses-dir evaluation/runs/<run_id>/responses \
  --references evaluation/eval_references.jsonl \
  --out-dir evaluation/runs/<run_id>/cosine_similarity
```

Output:

- `evaluation/runs/<run_id>/cosine_similarity/report_ranking.md`

Comparison rule used in script:

- `m1` wins if `sim(m1) > sim(m2) + 1e-5`
- `m2` wins if vice versa
- otherwise tie

## Validation

Before long sweeps/evals:

```bash
.venv/bin/python validate_training_setup.py
```

Checks include:

- LoRA vs QLoRA base-model distinctness
- adapter weight collisions
- eval model alignment
- shell interpreter consistency
- dataset row-count sanity

## Utility Scripts

Quick side-by-side sanity test:

```bash
.venv/bin/python quick_eval.py --adapter ./adapters/tinyllama-lora-alpaca --n-prompts 10
```

ASCII loss curves:

```bash
.venv/bin/python plot_loss.py --log ./adapters/tinyllama-lora-alpaca/train.log
```

## Streamlit App

```bash
.venv/bin/streamlit run app.py
```

## Project Layout

```text
.
├── app.py
├── prepare_dataset.py
├── run_train.sh
├── run_qlora.sh
├── run_experiments.sh
├── smart_train.py
├── sweep_finetune.py
├── validate_training_setup.py
├── test_model.sh
├── quick_eval.py
├── plot_loss.py
├── lora_config.yaml
├── qlora_config.yaml
├── experiments/
├── data/
├── adapters/
├── models/
├── mlx_sweep_runs/
├── mlx_best_models/
└── evaluation/
    ├── README.md
    ├── run_eval_6models.sh
    ├── run_pipeline.py
    ├── pipeline.py
    ├── generate_responses.py
    ├── make_pairs.py
    ├── judge_with_model.py
    ├── judge_with_lmstudio.py
    ├── score_judgments.py
    ├── score_similarity_rankings.py
    ├── models.json
    ├── eval_prompts.jsonl
    ├── eval_references.jsonl
    └── runs/
```

## Troubleshooting

- `ModuleNotFoundError`:
  - use `.venv/bin/python`, not system `python3`
- Full FT OOM:
  - use LoRA/QLoRA or lower memory load
- Eval adapters not found:
  - verify paths in `evaluation/models.json`
- Cosine script import errors:
  - install `sentence-transformers`, `nltk`, `numpy`

## License

Educational/research usage. Respect upstream licenses (TinyLlama, Alpaca, and model dependencies).
