#!/usr/bin/env python3
"""Smart training wrapper with early stopping for any mlx-lm config.

Wraps `mlx_lm.lora` and monitors validation loss in real-time.
If val loss doesn't improve for `--patience` consecutive evaluations,
training is terminated early and the best checkpoint is preserved.

Works for LoRA, QLoRA, and Full Fine-Tuning configs.

Usage:
    .venv/bin/python smart_train.py --config lora_config.yaml
    .venv/bin/python smart_train.py --config qlora_config.yaml --patience 3
    .venv/bin/python smart_train.py --config full_config.yaml --patience 5 --min-delta 0.01
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smart training wrapper with early stopping for mlx-lm."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the mlx-lm YAML config file (lora, qlora, or full).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Stop if val loss doesn't improve for this many consecutive evals. 0 disables early stopping.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Minimum improvement in val loss required to reset patience counter.",
    )
    return parser.parse_args()


def extract_val_loss(line: str) -> Optional[float]:
    """Extract validation loss from a training output line."""
    match = re.search(
        r"Iter\s+(\d+)\s*:\s*Val(?:idation)?\s+loss\s+([0-9]+(?:\.[0-9]+)?)",
        line,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return float(match.group(2))


def extract_iter_number(line: str) -> Optional[int]:
    """Extract the iteration number from a training output line."""
    match = re.search(r"Iter\s+(\d+)\s*:", line)
    if not match:
        return None
    return int(match.group(1))


def extract_adapter_path(config_path: Path) -> Optional[Path]:
    """Read the adapter_path from the config file."""
    try:
        import yaml
    except ImportError:
        return None
    with config_path.open() as f:
        cfg = yaml.safe_load(f)
    adapter_path = cfg.get("adapter_path")
    if adapter_path:
        return Path(adapter_path)
    return None


def find_best_checkpoint(adapter_dir: Path, best_iter: int) -> Optional[Path]:
    """Find the checkpoint file closest to (and not exceeding) the best iteration."""
    if not adapter_dir.exists():
        return None

    # Checkpoints are named like 0000200_adapters.safetensors
    checkpoints = sorted(adapter_dir.glob("[0-9]*_adapters.safetensors"))
    if not checkpoints:
        return None

    best_ckpt = None
    best_ckpt_iter = 0
    for ckpt in checkpoints:
        ckpt_iter = int(ckpt.stem.split("_")[0])
        if ckpt_iter <= best_iter and ckpt_iter > best_ckpt_iter:
            best_ckpt = ckpt
            best_ckpt_iter = ckpt_iter

    return best_ckpt


def main():
    args = parse_args()

    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    adapter_dir = extract_adapter_path(args.config)

    print(f"{'=' * 60}")
    print(f"  Smart Training Wrapper")
    print(f"  Config:    {args.config}")
    print(f"  Patience:  {args.patience} {'(disabled)' if args.patience == 0 else 'evals'}")
    print(f"  Min Delta: {args.min_delta}")
    if adapter_dir:
        print(f"  Adapter:   {adapter_dir}")
    print(f"{'=' * 60}\n")

    # Launch training
    cmd = [sys.executable, "-m", "mlx_lm.lora", "--config", str(args.config)]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    best_val: Optional[float] = None
    best_iter: int = 0
    stale_evals: int = 0
    early_stopped: bool = False
    current_iter: int = 0

    if proc.stdout is None:
        print("Error: Failed to capture training output.")
        sys.exit(1)

    for raw_line in proc.stdout:
        cleaned = ANSI_ESCAPE_RE.sub("", raw_line).replace("\r", "").strip()

        # Always print the training output
        if cleaned:
            print(cleaned, flush=True)

        # Track current iteration
        iter_num = extract_iter_number(raw_line)
        if iter_num is not None:
            current_iter = iter_num

        # Check for validation loss
        val_loss = extract_val_loss(raw_line)
        if val_loss is None or args.patience <= 0:
            continue

        # Skip NaN validation losses
        if val_loss != val_loss:  # NaN check
            continue

        improved = best_val is None or val_loss < (best_val - args.min_delta)
        if improved:
            best_val = val_loss
            best_iter = current_iter
            stale_evals = 0
            print(f"  ✅ New best val loss: {best_val:.6f} at iter {best_iter}", flush=True)
            continue

        stale_evals += 1
        remaining = args.patience - stale_evals
        print(
            f"  ⚠️  No improvement (best={best_val:.6f}, current={val_loss:.6f}). "
            f"Patience: {remaining} evals remaining.",
            flush=True,
        )

        if stale_evals >= args.patience:
            early_stopped = True
            print(
                f"\n{'=' * 60}\n"
                f"  🛑 EARLY STOPPING TRIGGERED\n"
                f"  Best val loss: {best_val:.6f} at iter {best_iter}\n"
                f"  Current val loss: {val_loss:.6f} at iter {current_iter}\n"
                f"  Patience exhausted after {args.patience} non-improving evals.\n"
                f"{'=' * 60}\n",
                flush=True,
            )
            proc.terminate()
            break

    proc.stdout.close()
    try:
        return_code = proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        return_code = proc.wait()

    # Post-training: restore best checkpoint if early stopped
    if early_stopped and adapter_dir and best_iter > 0:
        best_ckpt = find_best_checkpoint(adapter_dir, best_iter)
        if best_ckpt:
            final_adapter = adapter_dir / "adapters.safetensors"
            print(f"  Restoring best checkpoint: {best_ckpt.name} → adapters.safetensors")
            shutil.copy2(best_ckpt, final_adapter)
            print(f"  ✅ Best model (iter {best_iter}, val_loss={best_val:.6f}) is now the active adapter.\n")
        else:
            print(f"  ⚠️  Could not find checkpoint for iter {best_iter}. Last saved weights are active.\n")

    # Summary
    print(f"{'=' * 60}")
    print(f"  Training Summary")
    print(f"  {'Early stopped' if early_stopped else 'Completed normally'}")
    if best_val is not None:
        print(f"  Best val loss:  {best_val:.6f} (iter {best_iter})")
    print(f"  Exit code:      {return_code}")
    print(f"{'=' * 60}")

    sys.exit(0 if return_code == 0 or early_stopped else return_code)


if __name__ == "__main__":
    main()
