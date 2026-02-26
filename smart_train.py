#!/usr/bin/env python3
"""Smart training wrapper with early stopping for any mlx-lm config.

Wraps `mlx_lm.lora` and monitors validation loss in real-time.
If val loss doesn't improve for `--patience` consecutive evaluations,
training is terminated early and the best checkpoint is preserved.

Works for LoRA, QLoRA, and Full Fine-Tuning configs.

Usage:
    .venv/bin/python smart_train.py --config lora_config.yaml
    .venv/bin/python smart_train.py --config qlora_config.yaml --patience 3
    .venv/bin/python smart_train.py --config /path/to/full_ft_config.yaml --patience 5 --min-delta 0.01
"""
from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

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
    parser.add_argument(
        "--full-interval-divisor",
        type=int,
        default=10,
        help=(
            "For fine_tune_type=full, override steps_per_eval/save_every to "
            "max(1, steps_per_epoch // divisor). Set 0 to disable."
        ),
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


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def maybe_prepare_effective_config(
    config_path: Path,
    adapter_dir: Optional[Path],
    full_interval_divisor: int,
) -> Tuple[Path, Optional[Dict[str, int]]]:
    """Build an effective config for full FT with derived eval/save intervals.

    Returns (effective_config_path, interval_info). If no override was applied,
    effective_config_path is the original config_path and interval_info is None.
    """
    if full_interval_divisor <= 0:
        return config_path, None

    try:
        import yaml
    except ImportError:
        return config_path, None

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if cfg.get("fine_tune_type") != "full":
        return config_path, None

    batch_size = int(cfg.get("batch_size", 0) or 0)
    if batch_size < 1:
        return config_path, None

    data_dir = cfg.get("data")
    if not data_dir:
        return config_path, None

    data_dir_path = Path(data_dir)
    if not data_dir_path.is_absolute():
        data_dir_path = (config_path.parent / data_dir_path).resolve()
    train_file = data_dir_path / "train.jsonl"
    if not train_file.exists():
        return config_path, None

    train_rows = count_lines(train_file)
    if train_rows <= 0:
        return config_path, None

    steps_per_epoch = max(1, math.ceil(train_rows / batch_size))
    interval = max(1, steps_per_epoch // full_interval_divisor)
    cfg["steps_per_eval"] = interval
    cfg["save_every"] = interval

    out_dir = adapter_dir if adapter_dir is not None else config_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    effective_config = out_dir / "__smart_effective_config.yaml"
    effective_config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    return effective_config, {
        "train_rows": train_rows,
        "steps_per_epoch": steps_per_epoch,
        "interval": interval,
    }


def main():
    args = parse_args()

    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    adapter_dir = extract_adapter_path(args.config)
    effective_config_path, interval_info = maybe_prepare_effective_config(
        config_path=args.config,
        adapter_dir=adapter_dir,
        full_interval_divisor=args.full_interval_divisor,
    )

    log_file = None
    if adapter_dir:
        adapter_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(adapter_dir / "train.log", "w", encoding="utf-8")

    def log_print(msg: str):
        print(msg, flush=True)
        if log_file:
            log_file.write(msg + "\n")
            log_file.flush()

    log_print(f"{'=' * 60}")
    log_print(f"  Smart Training Wrapper")
    log_print(f"  Config:    {args.config}")
    if effective_config_path != args.config:
        log_print(f"  Effective: {effective_config_path}")
    log_print(f"  Patience:  {args.patience} {'(disabled)' if args.patience == 0 else 'evals'}")
    log_print(f"  Min Delta: {args.min_delta}")
    if interval_info is not None:
        log_print(
            "  Full FT intervals: "
            f"steps_per_eval=save_every={interval_info['interval']} "
            f"(train_rows={interval_info['train_rows']}, "
            f"steps_per_epoch={interval_info['steps_per_epoch']}, "
            f"divisor={args.full_interval_divisor})"
        )
    if adapter_dir:
        log_print(f"  Adapter:   {adapter_dir}")
    log_print(f"{'=' * 60}\n")

    # Launch training
    cmd = [sys.executable, "-m", "mlx_lm.lora", "--config", str(effective_config_path)]
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
        log_print("Error: Failed to capture training output.")
        sys.exit(1)

    for raw_line in proc.stdout:
        cleaned = ANSI_ESCAPE_RE.sub("", raw_line).replace("\r", "").strip()

        # Always print the training output
        if cleaned:
            log_print(cleaned)

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
            log_print(f"  ✅ New best val loss: {best_val:.6f} at iter {best_iter}")
            continue

        stale_evals += 1
        remaining = args.patience - stale_evals
        log_print(
            f"  ⚠️  No improvement (best={best_val:.6f}, current={val_loss:.6f}). "
            f"Patience: {remaining} evals remaining."
        )

        if stale_evals >= args.patience:
            early_stopped = True
            log_print(
                f"\n{'=' * 60}\n"
                f"  🛑 EARLY STOPPING TRIGGERED\n"
                f"  Best val loss: {best_val:.6f} at iter {best_iter}\n"
                f"  Current val loss: {val_loss:.6f} at iter {current_iter}\n"
                f"  Patience exhausted after {args.patience} non-improving evals.\n"
                f"{'=' * 60}\n"
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
            log_print(f"  Restoring best checkpoint: {best_ckpt.name} → adapters.safetensors")
            shutil.copy2(best_ckpt, final_adapter)
            log_print(f"  ✅ Best model (iter {best_iter}, val_loss={best_val:.6f}) is now the active adapter.\n")
        else:
            log_print(f"  ⚠️  Could not find checkpoint for iter {best_iter}. Last saved weights are active.\n")

    # Summary
    log_print(f"{'=' * 60}")
    log_print(f"  Training Summary")
    log_print(f"  {'Early stopped' if early_stopped else 'Completed normally'}")
    if best_val is not None:
        log_print(f"  Best val loss:  {best_val:.6f} (iter {best_iter})")
    log_print(f"  Exit code:      {return_code}")
    log_print(f"{'=' * 60}")
    
    if log_file:
        log_file.close()

    sys.exit(0 if return_code == 0 or early_stopped else return_code)


if __name__ == "__main__":
    main()
