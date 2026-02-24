# Eval From Generated Runs

Use this guide when you already have model responses in:

- `evaluation/runs/<run_id>/responses/*.jsonl`

and you want to complete the rest of the eval pipeline.

## 0) Prereq

```bash
source .venv/bin/activate
```

Set your run id (example shown):

```bash
RUN_ID=20260224_103910
RUN_DIR="evaluation/runs/${RUN_ID}"
```

## 1) Build blind pairwise tasks from existing responses

Run pairing using the fixed 9-pair comparison set:

```bash
PAIRS="full_ft:base,full_ft:qwen_1.8b,full_ft:phi_2,lora_ft:base,lora_ft:qwen_1.8b,lora_ft:phi_2,qlora_ft:base,qlora_ft:qwen_1.8b,qlora_ft:phi_2"

.venv/bin/python evaluation/make_pairs.py \
  --responses-dir "${RUN_DIR}/responses" \
  --out-dir "${RUN_DIR}/pairing" \
  --pairs "${PAIRS}" \
  --seed 42
```

Outputs:

- `${RUN_DIR}/pairing/judging_tasks.jsonl`
- `${RUN_DIR}/pairing/judging_key.jsonl`
- `${RUN_DIR}/pairing/pairing_summary.json`

## 2) Judge with local OSS20B on LM Studio (default)

In LM Studio:

- Start local server (OpenAI-compatible API).
- Load your OSS20B model (example model id: `gpt-oss-20b`).

Optional API check:

```bash
curl -s http://127.0.0.1:1234/v1/models
```

Run judging:

```bash
.venv/bin/python evaluation/judge_with_lmstudio.py \
  --tasks "${RUN_DIR}/pairing/judging_tasks.jsonl" \
  --out "${RUN_DIR}/pairing/judgments_oss20b.jsonl" \
  --model "gpt-oss-20b" \
  --base-url "http://127.0.0.1:1234/v1" \
  --progress-every 100
```

If your loaded model id in LM Studio is different, pass that exact id to `--model`.

## 3) Score judgments and generate reports

```bash
.venv/bin/python evaluation/score_judgments.py \
  --key "${RUN_DIR}/pairing/judging_key.jsonl" \
  --judgments "${RUN_DIR}/pairing/judgments_oss20b.jsonl" \
  --out-dir "${RUN_DIR}/scoring" \
  --seed 42
```

Scoring outputs:

- `${RUN_DIR}/scoring/pair_metrics.json`
- `${RUN_DIR}/scoring/pair_metrics.csv`
- `${RUN_DIR}/scoring/model_rollup.json`
- `${RUN_DIR}/scoring/model_rollup.csv`
- `${RUN_DIR}/scoring/report.md`

## 4) Manual judging fallback

If you need human labels instead, create:

- `${RUN_DIR}/pairing/judgments_manual.jsonl`

Each line:

```json
{"item_id":"...","winner":"left|right|tie|invalid"}
```

Accepted aliases:

- `a` -> `left`
- `b` -> `right`
- `draw`/`equal` -> `tie`
- `skip` -> `invalid`

Then score by replacing `judgments_oss20b.jsonl` with `judgments_manual.jsonl`.

## 5) Quick rerun pattern

If you want to regenerate pairing (without touching responses), write to a new pairing folder:

```bash
.venv/bin/python evaluation/make_pairs.py \
  --responses-dir "${RUN_DIR}/responses" \
  --out-dir "${RUN_DIR}/pairing_rebalanced" \
  --pairs "${PAIRS}" \
  --seed 42
```

Then judge + score against `pairing_rebalanced` files.
