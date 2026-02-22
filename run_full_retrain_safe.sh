#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-sanity}"
if [[ "${STAGE}" != "sanity" && "${STAGE}" != "scale" && "${STAGE}" != "epoch1" ]]; then
  echo "Usage: $0 [sanity|scale|epoch1]"
  exit 2
fi

if [[ ! -f ".venv/bin/python" ]]; then
  echo "Missing .venv. Create it first."
  exit 2
fi

if [[ ! -f "./data/train.jsonl" ]]; then
  echo "Missing ./data/train.jsonl"
  exit 2
fi

if [[ "${STAGE}" == "sanity" ]]; then
  CFG="experiments/full_safe_sanity.yaml"
elif [[ "${STAGE}" == "scale" ]]; then
  CFG="experiments/full_safe_scale.yaml"
else
  CFG="experiments/full_safe_epoch1.yaml"
fi

echo "Running full_ft retrain stage=${STAGE} config=${CFG}"
.venv/bin/python -m mlx_lm.lora --config "${CFG}"
echo "Finished full_ft retrain stage=${STAGE}. Output: ./mlx_best_models/full_retrain"
