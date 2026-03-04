#!/usr/bin/env bash
set -euo pipefail

# Fixed pair set for fine-tuned models vs baselines.
PAIRS="full_ft:base,full_ft:qwen_1.8b,full_ft:phi_2,lora_ft:base,lora_ft:qwen_1.8b,lora_ft:phi_2,qlora_ft:base,qlora_ft:qwen_1.8b,qlora_ft:phi_2"

# Fixed generation params applied uniformly to every model.
TEMPERATURE="${TEMPERATURE:-0.2}"
TOP_P="${TOP_P:-0.9}"
MAX_TOKENS="${MAX_TOKENS:-512}"
SEED="${SEED:-42}"
MAX_PROMPTS="${MAX_PROMPTS:-200}"

# LM Studio judge params (used when JUDGE_MODE=lmstudio).
JUDGE_MODE="${JUDGE_MODE:-manual}"
LMSTUDIO_URL="${LMSTUDIO_URL:-http://127.0.0.1:1234/v1}"
LMSTUDIO_MODEL="${LMSTUDIO_MODEL:-qwen/qwen3-4b}"
LMSTUDIO_MAX_JUDGE_TOKENS="${LMSTUDIO_MAX_JUDGE_TOKENS:-3000}"
LMSTUDIO_MAX_RESPONSE_CHARS="${LMSTUDIO_MAX_RESPONSE_CHARS:-2000}"
LMSTUDIO_CONCURRENCY="${LMSTUDIO_CONCURRENCY:-1}"
LMSTUDIO_TIMEOUT_S="${LMSTUDIO_TIMEOUT_S:-180}"

# Fast local MLX judge params (used when JUDGE_MODE=model).
# Default: Qwen 1.5 1.8B 4-bit — fast, neutral relative to TinyLlama variants.
JUDGE_MODEL="${JUDGE_MODEL:-./models/qwen1.5-1.8b-chat-4bit}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-3}"
JUDGE_MAX_RESPONSE_CHARS="${JUDGE_MAX_RESPONSE_CHARS:-600}"

JUDGE_ARGS=()
if [ "${JUDGE_MODE}" = "lmstudio" ]; then
  JUDGE_ARGS+=(
    --judge-mode lmstudio
    --lmstudio-url "${LMSTUDIO_URL}"
    --lmstudio-model "${LMSTUDIO_MODEL}"
    --lmstudio-max-judge-tokens "${LMSTUDIO_MAX_JUDGE_TOKENS}"
    --lmstudio-max-response-chars "${LMSTUDIO_MAX_RESPONSE_CHARS}"
    --lmstudio-concurrency "${LMSTUDIO_CONCURRENCY}"
    --lmstudio-timeout-s "${LMSTUDIO_TIMEOUT_S}"
  )
elif [ "${JUDGE_MODE}" = "model" ]; then
  JUDGE_ARGS+=(
    --judge-mode model
    --judge-model "${JUDGE_MODEL}"
    --judge-max-tokens "${JUDGE_MAX_TOKENS}"
    --judge-max-response-chars "${JUDGE_MAX_RESPONSE_CHARS}"
  )
fi

.venv/bin/python evaluation/run_pipeline.py \
  --models-config evaluation/models.json \
  --prompts evaluation/eval_prompts.jsonl \
  --max-prompts "${MAX_PROMPTS}" \
  --pairs "${PAIRS}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --max-tokens "${MAX_TOKENS}" \
  --seed "${SEED}" \
  "${JUDGE_ARGS[@]}" \
  "$@"
