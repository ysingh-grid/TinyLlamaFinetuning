#!/bin/bash
# Monitor Task 1 full training progress

LOG_FILE="/tmp/task1_full_training.log"

echo "════════════════════════════════════════════════════════"
echo "  Task 1: LoRA Rank Ablation - Full Training Monitor"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  - Ranks: 4, 8, 16, 32, 64"
echo "  - Iterations: 2000 per rank"
echo "  - Dataset: 4000 training samples"
echo "  - Coverage: 1 full epoch"
echo "  - Estimated time: 90-120 minutes total"
echo ""
echo "Progress:"
echo "────────────────────────────────────────────────────────"

if [ -f "$LOG_FILE" ]; then
    # Show current rank being trained
    current_rank=$(grep -o "rank=[0-9]*" "$LOG_FILE" | tail -1)
    echo "Current: $current_rank"
    echo ""
    
    # Show completed ranks
    echo "Completed ranks:"
    grep "tokens/sec=" "$LOG_FILE" | tail -5
    echo ""
    
    # Show recent progress
    echo "Recent activity:"
    tail -10 "$LOG_FILE"
else
    echo "Training log not found. Training may not have started yet."
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "To monitor live: tail -f /tmp/task1_full_training.log"
