#!/usr/bin/env python3
import argparse
import itertools
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

DEFAULT_FULL_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_STOCK_4BIT_MODEL = "./models/tinyllama-4bit-base"
DEFAULT_DATA_DIR = "./data"
DEFAULT_OUT_DIR = "./mlx_sweep_runs"
DEFAULT_BEST_DIR = "./mlx_best_models"
DEFAULT_KEYS = "self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj"

# Based on previous sweep results:
# Full FT best: lr=1e-5, ga=6
# LoRA best: rank=16, lr=2e-4, ga=6
# QLoRA best: rank=8, lr=2e-4, ga=4
FULL_LRS = [1e-5]
RANKS = [8, 16]
LRS = [2e-4]
EPOCHS = [2]
BATCH_SIZES = [4]
GRAD_ACCUMS = [4, 6]
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MLX sweep runner for Full FT, LoRA, and QLoRA."
    )
    parser.add_argument(
        "--technique",
        choices=["all", "full", "lora", "qlora"],
        default="all",
        help="Which technique to run. 'all' runs full, lora, qlora sequentially.",
    )
    parser.add_argument(
        "--search",
        choices=["grid", "tpe"],
        default="grid",
        help="Hyperparameter search strategy.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,  # Accelerated: Reduced from 50 to 10
        help="Number of trials for --search tpe (capped by search space size).",
    )
    parser.add_argument("--data-dir", type=Path, default=Path(DEFAULT_DATA_DIR))
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--best-dir",
        type=Path,
        default=Path(DEFAULT_BEST_DIR),
        help="Fresh per-technique best outputs are copied here.",
    )
    parser.add_argument(
        "--full-model",
        default=DEFAULT_FULL_MODEL,
        help="Non-quantized base model used for full fine-tuning.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_STOCK_4BIT_MODEL,
        help="4-bit stock base model used by lora and qlora.",
    )
    parser.add_argument(
        "--allow-local-model",
        action="store_true",
        help="Allow local model paths. Disabled by default to avoid reusing fine-tuned checkpoints.",
    )
    parser.add_argument("--keys", default=DEFAULT_KEYS)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lora-layers", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps-per-report", type=int, default=10)
    parser.add_argument("--val-batches", type=int, default=-1)
    parser.add_argument("--test-batches", type=int, default=-1)
    parser.add_argument(
        "--status",
        choices=["minimal", "all", "none"],
        default="minimal",
        help="Console status verbosity while trials run.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=3,
        help="Stop trial early after this many non-improving validation checks (0 disables).",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.005,
        help="Minimum validation-loss improvement required to reset patience.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max trials per technique (0 means run full grid).",
    )
    return parser.parse_args()


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def parse_keys(keys_csv: str) -> List[str]:
    keys = [key.strip() for key in keys_csv.split(",") if key.strip()]
    if not keys:
        raise ValueError("No LoRA keys provided.")
    return keys


def validate_model_sources(args: argparse.Namespace) -> None:
    if args.allow_local_model:
        return

    lora_model_path = Path(args.model)
    default_stock_path = Path(DEFAULT_STOCK_4BIT_MODEL)
    if lora_model_path.exists():
        # Allow the known stock local 4-bit base by default.
        if lora_model_path.resolve() == default_stock_path.resolve():
            model_file = lora_model_path / "model.safetensors"
            if not model_file.exists():
                raise ValueError(
                    f"Default stock model path '{lora_model_path}' is missing model.safetensors."
                )
        else:
            raise ValueError(
                f"--model points to local path '{args.model}'. "
                "Use the default stock path or pass --allow-local-model explicitly."
            )

    full_model_path = Path(args.full_model)
    if full_model_path.exists():
        raise ValueError(
            f"--full-model points to local path '{args.full_model}'. "
            "Use a hub model id or pass --allow-local-model explicitly."
        )


def model_for_technique(args: argparse.Namespace, technique: str) -> str:
    if technique == "full":
        return args.full_model
    if technique == "lora":
        # LoRA uses full-precision base (same as standalone lora_config.yaml).
        return args.full_model
    # QLoRA uses 4-bit quantized base.
    return args.model


def parse_loss(log_text: str) -> float:
    """Extract the best validation loss from training logs.

    Prefers test loss > validation loss > any loss mention.
    For validation losses, returns the *minimum* across all checkpoints
    rather than the last match, which gives a true best-checkpoint signal.
    """
    # 1. Prefer explicit test loss (single value at end of training).
    test_pattern = r"test loss[^0-9]*([0-9]+(?:\.[0-9]+)?)"
    test_matches = re.findall(test_pattern, log_text, flags=re.IGNORECASE)
    if test_matches:
        return float(test_matches[-1])

    # 2. Best (minimum) validation loss across all checkpoints.
    val_pattern = r"val(?:idation)? loss[^0-9]*([0-9]+(?:\.[0-9]+)?)"
    val_matches = re.findall(val_pattern, log_text, flags=re.IGNORECASE)
    if val_matches:
        return min(float(v) for v in val_matches)

    # 3. Fallback: any loss mention.
    fallback_pattern = r"\bloss[^0-9]*([0-9]+(?:\.[0-9]+)?)"
    fallback_matches = re.findall(fallback_pattern, log_text, flags=re.IGNORECASE)
    if fallback_matches:
        return float(fallback_matches[-1])

    return float("inf")


def extract_val_loss(line: str) -> Optional[float]:
    match = re.search(
        r"Iter\s+\d+\s*:\s*Val(?:idation)?\s+loss\s+([0-9]+(?:\.[0-9]+)?)",
        line,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return float(match.group(1))


def normalize_log_line(line: str) -> str:
    cleaned = ANSI_ESCAPE_RE.sub("", line)
    return cleaned.replace("\r", "").strip()


def should_echo_line(line: str, status_mode: str) -> bool:
    if status_mode == "none":
        return False
    if status_mode == "all":
        return True
    # minimal
    prefixes = (
        "Loading configuration file",
        "Loading pretrained model",
        "Loading datasets",
        "Training",
        "Starting training",
        "Iter ",
        "Test loss",
        "Saved final weights",
        "EARLY_STOP",
    )
    return line.startswith(prefixes)


def ensure_gpu_available_or_fail() -> None:
    probe_code = "\n".join(
        [
            "import mlx.core as mx",
            "ok = False",
            "try:",
            "    ok = bool(mx.is_available(mx.Device(mx.gpu, 0)))",
            "except Exception:",
            "    ok = False",
            "if not ok:",
            "    try:",
            "        ok = bool(mx.metal.is_available())",
            "    except Exception:",
            "        ok = False",
            "raise SystemExit(0 if ok else 1)",
        ]
    )
    probe = subprocess.run(
        [sys.executable, "-c", probe_code],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        details = ((probe.stderr or "") + "\n" + (probe.stdout or "")).strip()
        raise RuntimeError(
            "GPU-only mode is enabled, but no MLX Metal GPU is available. "
            f"Probe details: {details if details else 'no additional output'}"
        )


def build_gpu_only_cmd(config_path: Path) -> List[str]:
    config_literal = json.dumps(str(config_path))
    launcher = "\n".join(
        [
            "import runpy",
            "import sys",
            "import mlx.core as mx",
            "import transformers",
            "ok = False",
            "try:",
            "    ok = bool(mx.is_available(mx.Device(mx.gpu, 0)))",
            "except Exception:",
            "    ok = False",
            "if not ok:",
            "    try:",
            "        ok = bool(mx.metal.is_available())",
            "    except Exception:",
            "        ok = False",
            "if not ok:",
            "    raise SystemExit('MLX Metal GPU unavailable')",
            "mx.set_default_device(mx.Device(mx.gpu, 0))",
            "if not hasattr(transformers, 'TokenizersBackend'):",
            "    from transformers import PreTrainedTokenizerFast",
            "    class TokenizersBackend(PreTrainedTokenizerFast):",
            "        pass",
            "    transformers.TokenizersBackend = TokenizersBackend",
            f"sys.argv = ['mlx_lm.lora', '--config', {config_literal}]",
            "runpy.run_module('mlx_lm.lora', run_name='__main__')",
        ]
    )
    return [sys.executable, "-c", launcher]


def build_grid(technique: str) -> List[Tuple]:
    if technique == "full":
        return list(itertools.product(FULL_LRS, EPOCHS, BATCH_SIZES, GRAD_ACCUMS))
    # Alpha is derived as 2*rank; not an independent axis.
    return list(itertools.product(RANKS, LRS, EPOCHS, BATCH_SIZES, GRAD_ACCUMS))


def build_config(
    args: argparse.Namespace,
    technique: str,
    run_dir: Path,
    train_rows: int,
    params: Tuple,
    keys: List[str],
) -> Dict:
    if technique == "full":
        learning_rate, epochs, batch_size, grad_accum_steps = params
        rank = None
        alpha = None
    else:
        rank, learning_rate, epochs, batch_size, grad_accum_steps = params
        alpha = rank * 2  # scale fixed at 2.0 throughout

    steps_per_epoch = max(1, math.ceil(train_rows / batch_size))
    iters = steps_per_epoch * epochs
    # Run validation about 10 times per epoch (e.g., 100 when steps_per_epoch=1000).
    eval_interval = max(1, steps_per_epoch // 10)
    save_interval = max(1, steps_per_epoch // 10)

    config: Dict = {
        "model": model_for_technique(args, technique),
        "train": True,
        "data": str(args.data_dir),
        "seed": args.seed,
        "batch_size": batch_size,
        "iters": iters,
        "val_batches": args.val_batches,
        "learning_rate": learning_rate,
        "steps_per_report": args.steps_per_report,
        "steps_per_eval": eval_interval,
        "save_every": save_interval,
        "adapter_path": str(run_dir / "output"),
        "test": True,
        "test_batches": args.test_batches,
        "max_seq_length": args.max_seq_length,
        "grad_checkpoint": True,
        "grad_accumulation_steps": grad_accum_steps,
        "mask_prompt": True,
        "lr_schedule": {
            "name": "cosine_decay",
            "warmup": 50,
            "warmup_init": 0.0,
            "arguments": [learning_rate, iters - 50, learning_rate * 0.1],
        }
    }

    if technique == "full":
        config["fine_tune_type"] = "full"
    elif technique == "qlora":
        config["fine_tune_type"] = "lora"
        config["lora_layers"] = args.lora_layers
        config["lora_parameters"] = {
            "keys": keys,
            "rank": rank,
            "scale": alpha / rank,
            "dropout": args.dropout,
        }
    else:
        # technique == "lora"
        config["fine_tune_type"] = "lora"
        config["lora_layers"] = args.lora_layers
        config["lora_parameters"] = {
            "keys": keys,
            "rank": rank,
            "scale": alpha / rank,
            "dropout": args.dropout,
        }

    return config


def describe_params(technique: str, params: Tuple) -> str:
    if technique == "full":
        lr, epochs, batch_size, grad_accum_steps = params
        return f"lr={lr} ep={epochs} bs={batch_size} ga={grad_accum_steps}"

    rank, lr, epochs, batch_size, grad_accum_steps = params
    alpha = rank * 2
    return (
        f"rank={rank} alpha={alpha} scale=2.0 lr={lr} "
        f"ep={epochs} bs={batch_size} ga={grad_accum_steps}"
    )


def record_params(result: Dict, technique: str, params: Tuple) -> None:
    if technique == "full":
        lr, epochs, batch_size, grad_accum_steps = params
        result["learning_rate"] = lr
        result["epochs"] = epochs
        result["batch_size"] = batch_size
        result["grad_accumulation_steps"] = grad_accum_steps
        result["effective_batch_size"] = batch_size * grad_accum_steps
        return

    rank, lr, epochs, batch_size, grad_accum_steps = params
    result["rank"] = rank
    result["alpha"] = rank * 2  # always 2*rank
    result["scale"] = 2.0
    result["learning_rate"] = lr
    result["epochs"] = epochs
    result["batch_size"] = batch_size
    result["grad_accumulation_steps"] = grad_accum_steps
    result["effective_batch_size"] = batch_size * grad_accum_steps


def run_trial(
    args: argparse.Namespace,
    technique: str,
    trial_id: int,
    total_trials: int,
    params: Tuple,
    train_rows: int,
    keys: List[str],
    optuna_trial: Optional[Any] = None,
) -> Dict:
    run_dir = args.out_dir / technique / f"trial_{trial_id:04d}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[{technique} {trial_id}/{total_trials}] starting {describe_params(technique, params)}",
        flush=True,
    )

    config = build_config(args, technique, run_dir, train_rows, params, keys)
    config_path = run_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    cmd = build_gpu_only_cmd(config_path)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    log_lines: List[str] = []
    early_stopped = False
    pruned = False
    best_val: Optional[float] = None
    stale_evals = 0
    val_step = 0

    if proc.stdout is None:
        raise RuntimeError("Failed to capture training output stream.")

    for raw_line in proc.stdout:
        log_lines.append(raw_line)
        normalized = normalize_log_line(raw_line)
        if normalized and should_echo_line(normalized, args.status):
            print(f"[{technique} {trial_id}/{total_trials}] {normalized}", flush=True)
        val_loss = extract_val_loss(raw_line)

        if val_loss is not None:
            val_step += 1
            if optuna_trial is not None:
                optuna_trial.report(val_loss, val_step)
                if optuna_trial.should_prune():
                    pruned = True
                    log_lines.append(f"PRUNED: val_loss={val_loss:.6f}\n")
                    print(
                        f"[{technique} {trial_id}/{total_trials}] "
                        f"pruned by optuna at val_loss={val_loss:.6f}",
                        flush=True,
                    )
                    proc.terminate()
                    break

        if val_loss is None or args.early_stop_patience <= 0 or early_stopped or pruned:
            continue

        improved = best_val is None or val_loss < (best_val - args.early_stop_min_delta)
        if improved:
            best_val = val_loss
            stale_evals = 0
            continue

        stale_evals += 1
        if stale_evals >= args.early_stop_patience:
            early_stopped = True
            stop_msg = (
                f"EARLY_STOP: patience={args.early_stop_patience}, "
                f"best_val={best_val:.6f}, last_val={val_loss:.6f}\n"
            )
            log_lines.append(stop_msg)
            print(
                f"[{technique} {trial_id}/{total_trials}] "
                f"early-stopping triggered at val_loss={val_loss:.6f} "
                f"(best={best_val:.6f})",
                flush=True,
            )
            proc.terminate()

    proc.stdout.close()
    try:
        return_code = proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        return_code = proc.wait()
    log_text = "".join(log_lines)
    (run_dir / "train.log").write_text(log_text, encoding="utf-8")

    loss = parse_loss(log_text)
    ok = return_code == 0 or ((early_stopped or pruned) and loss != float("inf"))

    result = {
        "technique": technique,
        "trial": trial_id,
        "ok": ok,
        "loss": loss,
        "early_stopped": early_stopped,
        "pruned": pruned,
        "config_path": str(config_path),
        "output_path": str(run_dir / "output"),
    }
    record_params(result, technique, params)
    print(
        f"[{technique} {trial_id}/{total_trials}] "
        f"ok={result['ok']} loss={result['loss']:.6f} {describe_params(technique, params)}"
    )
    return result


def limit_grid(grid: List[Tuple], limit: int) -> Iterable[Tuple]:
    if limit > 0:
        return grid[:limit]
    return grid


def suggest_params_for_tpe(trial, technique: str) -> Tuple:
    if technique == "full":
        return (
            trial.suggest_categorical("learning_rate", FULL_LRS),
            trial.suggest_categorical("epochs", EPOCHS),
            trial.suggest_categorical("batch_size", BATCH_SIZES),
            trial.suggest_categorical("grad_accumulation_steps", GRAD_ACCUMS),
        )

    # alpha is always 2*rank — not a separate TPE dimension
    return (
        trial.suggest_categorical("rank", RANKS),
        trial.suggest_categorical("learning_rate", LRS),
        trial.suggest_categorical("epochs", EPOCHS),
        trial.suggest_categorical("batch_size", BATCH_SIZES),
        trial.suggest_categorical("grad_accumulation_steps", GRAD_ACCUMS),
    )


def run_tpe_trials(
    args: argparse.Namespace,
    technique: str,
    total_trials: int,
    train_rows: int,
    keys: List[str],
) -> List[Dict]:
    try:
        import optuna
        from optuna.pruners import MedianPruner
    except Exception as exc:
        raise RuntimeError(
            "Optuna is required for --search tpe. Install with: pip install optuna"
        ) from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2, interval_steps=1)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    results: List[Dict] = []

    def objective(trial):
        params = suggest_params_for_tpe(trial, technique)
        trial_id = len(results) + 1
        result = run_trial(
            args=args,
            technique=technique,
            trial_id=trial_id,
            total_trials=total_trials,
            params=params,
            train_rows=train_rows,
            keys=keys,
            optuna_trial=trial,
        )
        result["optuna_trial_number"] = trial.number
        results.append(result)
        
        if result.get("pruned"):
            raise optuna.TrialPruned()
            
        return result["loss"] if result["ok"] else float("inf")

    study.optimize(objective, n_trials=total_trials)
    return results


def run_technique(
    args: argparse.Namespace,
    technique: str,
    train_rows: int,
    keys: List[str],
) -> Dict:
    grid = build_grid(technique)
    planned_trials = len(grid)
    if args.search == "grid":
        selected_grid = list(limit_grid(grid, args.limit))
        total_trials = len(selected_grid)
    else:
        selected_grid = []
        requested_trials = args.n_trials if args.n_trials > 0 else planned_trials
        total_trials = min(requested_trials, planned_trials)

    print(
        f"\nRunning {technique.upper()} sweep: "
        f"{total_trials}/{planned_trials} trials (search={args.search})"
    )

    technique_dir = args.out_dir / technique
    technique_dir.mkdir(parents=True, exist_ok=True)
    best_output_dir = args.best_dir / technique
    if best_output_dir.is_symlink():
        best_output_dir.unlink()
    elif best_output_dir.exists():
        shutil.rmtree(best_output_dir)

    if args.search == "grid":
        results = []
        for i, params in enumerate(selected_grid, start=1):
            result = run_trial(args, technique, i, total_trials, params, train_rows, keys)
            results.append(result)
    else:
        results = run_tpe_trials(
            args=args,
            technique=technique,
            total_trials=total_trials,
            train_rows=train_rows,
            keys=keys,
        )

    results_path = technique_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    successful = [row for row in results if row["ok"]]
    summary: Dict = {
        "technique": technique,
        "search": args.search,
        "total_trials": total_trials,
        "successful_trials": len(successful),
        "results_path": str(results_path),
    }

    if successful:
        best = min(successful, key=lambda row: row["loss"])
        best_path = technique_dir / "best.json"
        best_path.write_text(json.dumps(best, indent=2), encoding="utf-8")
        best_trial_output = Path(best["output_path"])
        if best_trial_output.exists():
            best_output_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(best_trial_output, best_output_dir)
        else:
            raise FileNotFoundError(f"Best trial output path not found: {best_trial_output}")
        summary["best_path"] = str(best_path)
        summary["best_loss"] = best["loss"]
        summary["best_trial"] = best["trial"]
        summary["best_output_dir"] = str(best_output_dir)
    else:
        summary["best_path"] = None
        summary["best_loss"] = None
        summary["best_trial"] = None
        summary["best_output_dir"] = None

    return summary


def main() -> None:
    args = parse_args()
    validate_model_sources(args)

    # Auto-download the 4-bit base model if it is a local path that is missing.
    try:
        _root = Path(__file__).resolve().parent
        sys.path.insert(0, str(_root))
        from download_models import ensure_model_path
        if args.model.startswith("."):
            ensure_model_path(args.model)
    except Exception:
        pass

    ensure_gpu_available_or_fail()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.best_dir.mkdir(parents=True, exist_ok=True)

    train_file = args.data_dir / "train.jsonl"
    if not train_file.exists():
        raise FileNotFoundError(f"Missing training file: {train_file}")

    train_rows = count_lines(train_file)
    keys = parse_keys(args.keys)

    techniques = ["full", "lora", "qlora"] if args.technique == "all" else [args.technique]

    summaries = []
    for technique in techniques:
        summary = run_technique(args, technique, train_rows, keys)
        summaries.append(summary)

    print("\nSweep summary:")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
