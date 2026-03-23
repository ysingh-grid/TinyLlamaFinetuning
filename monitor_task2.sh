#!/bin/bash
# Monitor Task 2 training progress

echo "================================"
echo "Task 2 Training Monitor"
echo "================================"
echo ""

# Check if PID file exists
if [ ! -f /tmp/task2_training.pid ]; then
    echo "⚠ Training not started yet (no PID file)"
    echo "Start with: ./run_task2_training.sh"
    exit 1
fi

PID=$(cat /tmp/task2_training.pid)

# Check if process is running
if ps -p $PID > /dev/null 2>&1; then
    echo "Status: ✓ RUNNING (PID: $PID)"
else
    echo "Status: ✗ STOPPED"
    echo ""
    echo "Last log entries:"
    tail -20 /tmp/task2_training.log
    exit 1
fi

echo ""
echo "Recent progress:"
echo "----------------"
tail -30 /tmp/task2_training.log | grep -E "(Iter|Loss|Val|Tokens|saved)" | tail -15

echo ""
echo "----------------"
echo "Full log: tail -f /tmp/task2_training.log"
