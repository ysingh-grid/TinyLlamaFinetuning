#!/bin/bash
# Quick status check for Task 1 training

PID_FILE="/tmp/task1_training.pid"
LOG_FILE="/tmp/task1_full_training.log"

echo "════════════════════════════════════════════════════════"
echo "  Task 1 Training - Quick Status"
echo "════════════════════════════════════════════════════════"
echo ""

# Check if PID file exists
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "✓ Training is RUNNING (PID: $PID)"
        
        # Show progress
        if [ -f "$LOG_FILE" ]; then
            echo ""
            echo "Current Progress:"
            echo "----------------"
            grep -E "rank=|Iter|tokens/sec=" "$LOG_FILE" | tail -5
        fi
    else
        echo "✗ Training is NOT RUNNING (PID $PID is dead)"
        echo ""
        echo "To restart: cd /Users/ysingh/PyCharmMiscProject && ./run_task1_training.sh"
    fi
else
    echo "? No PID file found"
    
    # Check if process is running anyway
    if ps aux | grep -v grep | grep "task1_lora_rank_ablation.py" > /dev/null; then
        echo "  But training process is detected running"
        ps aux | grep -v grep | grep "task1_lora_rank_ablation.py"
    else
        echo "  Training is not running"
        echo ""
        echo "To start: cd /Users/ysingh/PyCharmMiscProject && ./run_task1_training.sh"
    fi
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "Commands:"
echo "  Full monitor: ./monitor_task1.sh"
echo "  Live log: tail -f $LOG_FILE"
echo "════════════════════════════════════════════════════════"
