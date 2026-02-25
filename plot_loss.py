#!/usr/bin/env python3
"""Parses training logs and generates a visual loss curve in the terminal.

Usage:
  .venv/bin/python plot_loss.py --log adapters/tinyllama-lora-alpaca/train.log
"""
import argparse
import re
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Terminal ASCII loss curve plots.")
    parser.add_argument("--log", type=str, required=True, help="Path to train.log file")
    parser.add_argument("--lines", type=int, default=15, help="Height of the ASCII chart")
    parser.add_argument("--width", type=int, default=60, help="Width of the ASCII chart")
    return parser.parse_args()

def extract_losses(log_file):
    train_losses = []
    val_losses = []
    log_path = Path(log_file)
    if not log_path.exists():
        print(f"Error: Log file not found at {log_path}")
        return [], []
        
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            t_match = re.search(r"Iter\s+(\d+).*Train loss ([0-9.]+)", line)
            v_match = re.search(r"Iter\s+(\d+).*Val(?:idation)? loss ([0-9.]+)", line)
            if t_match:
                train_losses.append((int(t_match.group(1)), float(t_match.group(2))))
            if v_match:
                val_losses.append((int(v_match.group(1)), float(v_match.group(2))))
    return train_losses, val_losses

def print_sparkline(data, width=60, height=15, title=""):
    if not data:
        print(f"\n--- {title} ---")
        print("No data found.")
        return
    
    iters = [d[0] for d in data]
    losses = [d[1] for d in data]
    
    min_loss, max_loss = min(losses), max(losses)
    if min_loss == max_loss: 
        max_loss = min_loss + 1.0
    
    print(f"\n--- {title} ---")
    print(f"Iters: {min(iters)} to {max(iters)}")
    print(f"Loss:  Min={min_loss:.4f}, Max={max_loss:.4f}\n")
    
    normalized = []
    for l in losses:
        # Scale loss to height constraint
        val = int((l - min_loss) / (max_loss - min_loss) * (height - 1))
        normalized.append(val)
        
    # Group into buckets (for chart width)
    buckets = []
    chunk_size = max(1, len(normalized) // width)
    for i in range(0, len(normalized), chunk_size):
        chunk = normalized[i:i+chunk_size]
        if chunk:
            buckets.append(sum(chunk) // len(chunk))
    buckets = buckets[:width]
    
    # Draw chart
    for y in range(height-1, -1, -1):
        line_chars = []
        for b in buckets:
            if b == y: line_chars.append("★")
            elif b > y: line_chars.append("│")
            else: line_chars.append(" ")
        
        # Approximate boundary labels
        val_at_y = max_loss - (max_loss - min_loss) * ((height-1-y) / max(1, height-1))
        print(f"{val_at_y:>6.2f} ┤ " + "".join(line_chars))
        
    print("       └" + "─" * len(buckets))
    print(f"         Iter {min(iters)}" + " "*(len(buckets)-14) + f"Iter {max(iters)}")

def main():
    args = parse_args()
    train, val = extract_losses(args.log)
    print_sparkline(train, args.width, args.lines, "Train Loss")
    print_sparkline(val, args.width, args.lines, "Validation Loss")

if __name__ == "__main__":
    main()
