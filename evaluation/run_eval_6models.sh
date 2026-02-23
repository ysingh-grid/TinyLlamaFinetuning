#!/usr/bin/env bash
set -euo pipefail

# Fixed pair set for fine-tuned models vs baselines.
PAIRS="full_ft:base,full_ft:qwen_1.8b,full_ft:phi_2,lora_ft:base,lora_ft:qwen_1.8b,lora_ft:phi_2,qlora_ft:base,qlora_ft:qwen_1.8b,qlora_ft:phi_2"

# Fixed generation params applied uniformly to every model.
TEMPERATURE="${TEMPERATURE:-0.2}"
TOP_P="${TOP_P:-0.9}"
MAX_TOKENS="${MAX_TOKENS:-256}"
SEED="${SEED:-42}"
MAX_PROMPTS="${MAX_PROMPTS:-50}"

.venv/bin/python evaluation/run_pipeline.py \
  --models-config evaluation/models.json \
  --prompts evaluation/eval_prompts.jsonl \
  --max-prompts "${MAX_PROMPTS}" \
  --pairs "${PAIRS}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --max-tokens "${MAX_TOKENS}" \
  --seed "${SEED}" \
  "$@"
