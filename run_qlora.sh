#!/bin/bash
set -e

# Ensure data exists
if [ ! -f "./data/train.jsonl" ]; then
    echo "Dataset not found. Running preparation script..."
    .venv/bin/python prepare_dataset.py
fi

echo "Starting TinyLlama QLoRA training (4-bit)..."
echo "Config: qlora_config.yaml"

.venv/bin/python -m mlx_lm.lora --config qlora_config.yaml

echo "QLoRA Training complete! Adapter saved to ./adapters/tinyllama-qlora-alpaca"
