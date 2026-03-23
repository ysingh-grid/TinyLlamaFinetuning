#!/bin/bash
# Run Task 1 training in detached mode

cd /Users/ysingh/PyCharmMiscProject

echo "Starting Task 1 LoRA Rank Ablation training..."
echo "Log file: /tmp/task1_full_training.log"
echo "Started at: $(date)"

# Run with nohup to survive terminal close
nohup .venv/bin/python task1_lora_rank_ablation.py > /tmp/task1_full_training.log 2>&1 &

PID=$!
echo "Training process started with PID: $PID"
echo $PID > /tmp/task1_training.pid

# Wait a moment and check if it's running
sleep 3
if ps -p $PID > /dev/null; then
   echo "✓ Training is running successfully"
   echo "  Monitor: tail -f /tmp/task1_full_training.log"
   echo "  Progress: ./monitor_task1.sh"
else
   echo "✗ Training failed to start. Check log:"
   tail -20 /tmp/task1_full_training.log
fi
