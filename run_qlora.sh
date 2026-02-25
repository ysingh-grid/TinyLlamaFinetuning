#!/bin/bash
set -e

# Ensure data exists
if [ ! -f "./data/train.jsonl" ]; then
    echo "Dataset not found. Running preparation script..."
    .venv/bin/python prepare_dataset.py
fi

echo "Starting TinyLlama QLoRA training (4-bit)..."
echo "Config: qlora_config.yaml"

.venv/bin/python smart_train.py --config qlora_config.yaml --patience 5

echo "QLoRA Training complete! Adapter saved to ./adapters/tinyllama-qlora-alpaca"
