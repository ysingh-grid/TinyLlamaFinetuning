#!/bin/bash

# Default prompt if not provided
PROMPT="${1:-What is reinforcement learning?}"

echo "Testing TinyLlama LoRA adapter..."
echo "Prompt: $PROMPT"
echo "--------------------------------"

.venv/bin/python -m mlx_lm.generate \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter-path ./adapters/tinyllama-lora-alpaca \
  --prompt "$PROMPT" \
  --max-tokens 256

echo ""
echo "--------------------------------"
