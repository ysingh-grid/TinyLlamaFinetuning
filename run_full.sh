#!/bin/bash
set -e

# Ensure data exists
if [ ! -f "./data/train.jsonl" ]; then
    echo "Dataset not found. Running preparation script..."
    .venv/bin/python prepare_dataset.py
fi

echo "Starting TinyLlama Full Fine-Tuning..."
echo "Config: full_config.yaml"
echo "WARNING: This requires significantly more VRAM than LoRA."

.venv/bin/python smart_train.py --config full_config.yaml --patience 5

echo "Full Fine-Tuning complete! Model saved to ./models/tinyllama-full-alpaca"
