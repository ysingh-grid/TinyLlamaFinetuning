#!/usr/bin/env bash
set -euo pipefail

SMOKE_PROMPTS="evaluation/eval_prompts_smoke20.jsonl"
SMOKE_MODELS="evaluation/models_full_smoke.json"
RUN_ROOT="evaluation/runs_smoke"

if [[ ! -f ".venv/bin/python" ]]; then
  echo "Missing .venv. Create it first."
  exit 2
fi

if [[ ! -f "evaluation/eval_prompts.jsonl" ]]; then
  echo "Missing evaluation/eval_prompts.jsonl"
  exit 2
fi

if [[ ! -d "./mlx_best_models/full_retrain" ]]; then
  echo "Missing retrained full adapter at ./mlx_best_models/full_retrain"
  exit 2
fi

# Build first-20 prompt subset from frozen eval set.
.venv/bin/python - <<'PY'
import json
from pathlib import Path

src = Path("evaluation/eval_prompts.jsonl")
dst = Path("evaluation/eval_prompts_smoke20.jsonl")
rows = []
with src.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 20:
            break
        rows.append(json.loads(line))
with dst.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")
print(f"Wrote {len(rows)} rows to {dst}")
PY

cat > "${SMOKE_MODELS}" <<'JSON'
[
  {
    "name": "full_ft_retrain",
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "adapter_path": "./mlx_best_models/full_retrain"
  },
  {
    "name": "base",
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  }
]
JSON

echo "Running 20-prompt smoke eval generation..."
.venv/bin/python evaluation/generate_responses.py \
  --models-config "${SMOKE_MODELS}" \
  --prompts "${SMOKE_PROMPTS}" \
  --out-dir "${RUN_ROOT}/responses" \
  --temperature 0.2 \
  --top-p 0.9 \
  --max-tokens 256 \
  --seed 42 \
  --progress-every 5 \
  --overwrite

echo "Checking response quality for full_ft_retrain..."
.venv/bin/python evaluation/check_response_quality.py \
  --responses "${RUN_ROOT}/responses/full_ft_retrain.jsonl" \
  --max-bad-rate 0.20

echo "Smoke eval passed. You can proceed to full 500-prompt run."
