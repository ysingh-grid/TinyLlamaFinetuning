#!/bin/bash
set -e

# Activate virtual environment if needed (assuming user runs from venv or has python in path)
# But let's be explicit about using python3

echo "Starting Experiments: LoRA Ranks 8, 16, 32 (1200 iters)"

# Rank 8
echo "--------------------------------------------------"
echo "Running Rank 8 Experiment..."
python3 -m mlx_lm.lora --config experiments/rank8.yaml
echo "Rank 8 Complete"
sleep 2

# Rank 16
echo "--------------------------------------------------"
echo "Running Rank 16 Experiment..."
python3 -m mlx_lm.lora --config experiments/rank16.yaml
echo "Rank 16 Complete"
sleep 2

# Rank 32
echo "--------------------------------------------------"
echo "Running Rank 32 Experiment..."
python3 -m mlx_lm.lora --config experiments/rank32.yaml
echo "Rank 32 Complete"

echo "--------------------------------------------------"
echo "All Experiments Completed! Go to the Streamlit App to check them out."
