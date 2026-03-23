#!/bin/bash
# Start Task 2 adapter training in persistent background mode

cd /Users/ysingh/PyCharmMiscProject

echo "Starting Task 2 LoRA training (10k Alpaca)..."
echo "Estimated time: 60-90 minutes"
echo ""

# Start training with nohup
nohup .venv/bin/python -m mlx_lm.lora \
    --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --train \
    --data "./data/task2" \
    --adapter-path "./adapters/tinyllama-lora-alpaca-10k" \
    --config "./task2_lora_config.yaml" \
    --iters 5000 \
    --batch-size 2 \
    --num-layers 16 \
    --learning-rate 1e-4 \
    --val-batches 25 \
    --save-every 500 \
    --test-batches 10 \
    > /tmp/task2_training.log 2>&1 &

PID=$!
echo $PID > /tmp/task2_training.pid

# Wait a moment to see if it crashes immediately
sleep 3

if ps -p $PID > /dev/null 2>&1; then
    echo "✓ Training started successfully (PID: $PID)"
    echo "  Log: /tmp/task2_training.log"
    echo "  PID file: /tmp/task2_training.pid"
    echo ""
    echo "Monitor with:"
    echo "  tail -f /tmp/task2_training.log"
    echo "  ./monitor_task2.sh"
else
    echo "✗ Training failed to start. Check log:"
    echo "  tail /tmp/task2_training.log"
    exit 1
fi
