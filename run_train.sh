#!/bin/bash
set -e

# Ensure data exists
if [ ! -f "./data/train.jsonl" ]; then
    echo "Dataset not found. Running preparation script..."
    .venv/bin/python prepare_dataset.py
fi

echo "Starting TinyLlama LoRA training..."
echo "Config: lora_config.yaml"

.venv/bin/python smart_train.py --config lora_config.yaml --patience 5

echo "Training complete! Adapter saved to ./adapters/tinyllama-lora-alpaca"
