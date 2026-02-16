#!/bin/bash
set -e

# Ensure data exists
if [ ! -f "./data/train.jsonl" ]; then
    echo "Dataset not found. Running preparation script..."
    python3 prepare_dataset.py
fi

echo "Starting TinyLlama Full Fine-Tuning..."
echo "Config: full_config.yaml"
echo "WARNING: This requires significantly more VRAM than LoRA."

python3 -m mlx_lm.lora --config full_config.yaml

echo "Full Fine-Tuning complete! Model saved to ./models/tinyllama-full-alpaca"
