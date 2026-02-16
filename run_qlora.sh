#!/bin/bash
set -e

# Ensure data exists
if [ ! -f "./data/train.jsonl" ]; then
    echo "Dataset not found. Running preparation script..."
    python3 prepare_dataset.py
fi

echo "Starting TinyLlama QLoRA training (4-bit)..."
echo "Config: qlora_config.yaml"

python3 -m mlx_lm.lora --config qlora_config.yaml

echo "QLoRA Training complete! Adapter saved to ./adapters/tinyllama-qlora-alpaca"
