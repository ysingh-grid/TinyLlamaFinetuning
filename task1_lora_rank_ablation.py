#!/usr/bin/env python3
"""
Task 1: LoRA Rank Ablation Study with MLX-LM

Trains 5 models with LoRA ranks 4, 8, 16, 32, 64.
alpha = rank * 2  →  scale = alpha/rank = 2.0  (alpha/2 ratio constant)

For each run, tracks:
  - training wall time
  - peak Metal memory (MB)
  - final test loss (accuracy proxy)
  - inference latency per token (ms/tok) on 100 Alpaca test prompts

Outputs
  - results/task1/comparison_matrix.json
  - results/task1/rank_ablation_metrics.png  (memory + speed + loss curves)
  - results/task1/pareto.json
"""
import argparse
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_DIR   = Path("./data")
RESULTS_DIR = Path("./results/task1")

# Keeping alpha/2 ratio constant means scale = alpha/rank = (rank*2)/rank = 2.0
ALPHA_RATIO = 2          # alpha = rank * ALPHA_RATIO
SCALE       = float(ALPHA_RATIO)   # scale = alpha / rank = 2.0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_config(rank: int, iters: int, adapter_path: Path) -> dict:
    alpha = rank * ALPHA_RATIO
    return {
        "model": BASE_MODEL,
        "train": True,
        "data": str(DATA_DIR),
        "seed": 0,
        "lora_layers": 16,
        "batch_size": 2,
        "iters": iters,
        "val_batches": 10 if iters > 10 else 1,
        "learning_rate": 2e-4,
        "steps_per_report": max(1, min(10, iters // 5)),
        "steps_per_eval": max(1, min(100, iters // 2)),
        "adapter_path": str(adapter_path),
        "save_every": max(1, iters // 2),
        "test": True,
        "test_batches": 25 if iters > 25 else 2,
        "max_seq_length": 512,
        "grad_checkpoint": True,
        "mask_prompt": True,
        "lora_parameters": {
            "keys": ["self_attn.q_proj", "self_attn.v_proj"],
            "rank": rank,
            "scale": SCALE,   # alpha/rank = 2.0
            "dropout": 0.05,
        },
    }


def train_adapter(rank: int, iters: int, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = out_dir / f"adapter_r{rank}"
    config_path  = out_dir / f"config_r{rank}.yaml"
    log_path     = out_dir / f"train_r{rank}.log"

    config = build_config(rank, iters, adapter_path)
    with config_path.open("w") as f:
        yaml.dump(config, f, sort_keys=False)

    logging.info(f"[rank={rank}] Training for {iters} iters  (alpha={rank*ALPHA_RATIO}, scale={SCALE})…")

    # Subprocess isolates Metal context so peak_memory is per-rank
    wrapper = f"""
import sys, mlx.core as mx, mlx_lm.lora as lm_lora
sys.argv = ['mlx_lm.lora', '--config', '{config_path}']
try:
    lm_lora.main()
except SystemExit:
    pass
if hasattr(mx, 'metal') and hasattr(mx.metal, 'get_peak_memory'):
    peak = mx.metal.get_peak_memory() / (1024*1024)
    print(f'__PEAK_MB__={{peak:.2f}}')
else:
    print('__PEAK_MB__=0')
"""
    wrapper_path = out_dir / f"_wrap_r{rank}.py"
    wrapper_path.write_text(wrapper)

    t0   = time.perf_counter()
    proc = subprocess.run([sys.executable, str(wrapper_path)],
                          capture_output=True, text=True)
    train_sec = time.perf_counter() - t0

    combined = proc.stdout + "\n" + proc.stderr
    log_path.write_text(combined)

    if proc.returncode not in (0, 1):          # mlx_lm exits 0 or 1 on normal run
        logging.error(f"[rank={rank}] Training subprocess failed (rc={proc.returncode}). See {log_path}")
        return {"error": True}

    peak_mb = 0.0
    for line in combined.splitlines():
        if line.startswith("__PEAK_MB__="):
            try:
                peak_mb = float(line.split("=")[1])
            except ValueError:
                pass

    # Parse test loss
    test_loss = float("inf")
    m = re.search(r"[Tt]est loss[^\d]*([0-9]+(?:\.[0-9]+)?)", combined)
    if m:
        test_loss = float(m.group(1))

    logging.info(f"[rank={rank}] done — loss={test_loss:.4f}  mem={peak_mb:.0f} MB  time={train_sec:.0f}s")
    return {
        "rank":           rank,
        "alpha":          rank * ALPHA_RATIO,
        "scale":          SCALE,
        "train_time_sec": round(train_sec, 1),
        "peak_mem_mb":    peak_mb if peak_mb > 0 else None,
        "test_loss":      test_loss,
        "adapter_path":   str(adapter_path),
    }


# ---------------------------------------------------------------------------
# Inference latency
# ---------------------------------------------------------------------------

def benchmark_inference(rank: int, adapter_path: str, num_prompts: int) -> Dict[str, float]:
    """
    Returns: {'ms_per_token': float, 'tokens_per_sec': float}
    Uses 100 Alpaca test prompts; generates up to 100 tokens each.
    """
    wrapper = f"""
import sys, time, json
from mlx_lm import load, generate

model, tokenizer = load('{BASE_MODEL}', adapter_path='{adapter_path}')

prompts = []
with open('{DATA_DIR}/test.jsonl') as fh:
    for line in fh:
        row = json.loads(line)
        msgs = row.get('messages', [])
        user = next((m['content'] for m in msgs if m['role'] == 'user'), 'Hello')
        prompts.append(user)
        if len(prompts) >= {num_prompts}: break

# Warm-up
generate(model, tokenizer, "Hello", max_tokens=5, verbose=False)

total_tokens = 0
total_sec    = 0.0

for user in prompts:
    if hasattr(tokenizer, 'apply_chat_template'):
        p = tokenizer.apply_chat_template(
            [{{"role":"user","content":user}}],
            tokenize=False, add_generation_prompt=True)
    else:
        p = user + "\\nAssistant: "
    t0  = time.perf_counter()
    out = generate(model, tokenizer, p, max_tokens=100, verbose=False)
    t1  = time.perf_counter()
    toks = len(tokenizer.encode(out))
    total_tokens += toks
    total_sec    += (t1 - t0)

if total_sec > 0:
    tps = total_tokens / total_sec
    mpt = (total_sec / total_tokens) * 1000
    print(f'__TPS__={{tps:.3f}}')
    print(f'__MPT__={{mpt:.3f}}')
"""
    wp   = RESULTS_DIR / f"_bench_r{rank}.py"
    wp.write_text(wrapper)
    proc = subprocess.run([sys.executable, str(wp)], capture_output=True, text=True)

    tps, mpt = 0.0, float("inf")
    if proc.returncode == 0:
        for line in proc.stdout.splitlines():
            if line.startswith("__TPS__="):
                tps = float(line.split("=")[1])
            elif line.startswith("__MPT__="):
                mpt = float(line.split("=")[1])
    else:
        logging.warning(f"[rank={rank}] inference benchmark failed:\n{proc.stderr[-500:]}")

    return {"tokens_per_sec": tps, "ms_per_token": mpt}


# ---------------------------------------------------------------------------
# Comparison Matrix
# ---------------------------------------------------------------------------

def build_comparison_matrix(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Each row: rank | tokens_per_sec | peak_mem_mb | test_loss (MMLU-style accuracy proxy)
    """
    matrix = []
    for r in results:
        matrix.append({
            "rank":           r["rank"],
            "alpha":          r.get("alpha"),
            "scale":          r.get("scale"),
            "tokens_per_sec": r.get("tokens_per_sec", 0),
            "ms_per_token":   r.get("ms_per_token", float("inf")),
            "peak_mem_mb":    r.get("peak_mem_mb") or 0,
            "test_loss":      r.get("test_loss", float("inf")),
            "train_time_sec": r.get("train_time_sec", 0),
        })
    return matrix


def find_pareto_optimal(matrix: List[Dict[str, Any]]) -> Optional[int]:
    """
    Pareto criterion: maximise accuracy (minimise loss) AND maximise speed (tokens/sec).
    Composite score = normalised_accuracy + normalised_speed
    """
    valid = [r for r in matrix if r["test_loss"] < float("inf") and r["tokens_per_sec"] > 0]
    if not valid:
        return None

    min_loss  = min(r["test_loss"]      for r in valid)
    max_loss  = max(r["test_loss"]      for r in valid)
    min_speed = min(r["tokens_per_sec"] for r in valid)
    max_speed = max(r["tokens_per_sec"] for r in valid)

    best_rank, best_score = None, -float("inf")
    for r in valid:
        acc_norm   = (max_loss - r["test_loss"])      / (max_loss - min_loss + 1e-9)
        speed_norm = (r["tokens_per_sec"] - min_speed) / (max_speed - min_speed + 1e-9)
        score = acc_norm + speed_norm
        if score > best_score:
            best_score, best_rank = score, r["rank"]

    return best_rank


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(matrix: List[Dict[str, Any]], pareto_rank: Optional[int]):
    if not HAS_PLOT:
        logging.warning("matplotlib not available – skipping plots.")
        return

    ranks  = [r["rank"]           for r in matrix]
    mem    = [r["peak_mem_mb"]    for r in matrix]
    speed  = [r["tokens_per_sec"] for r in matrix]
    losses = [r["test_loss"]      for r in matrix]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("TinyLlama LoRA Rank Ablation (alpha = rank × 2)", fontsize=13)

    def _add_pareto(ax, y_vals):
        if pareto_rank and pareto_rank in ranks:
            i = ranks.index(pareto_rank)
            ax.axvline(pareto_rank, color="gold", linewidth=1.5, linestyle="--", alpha=0.7)
            ax.annotate(f"Pareto r={pareto_rank}", xy=(pareto_rank, y_vals[i]),
                        xytext=(pareto_rank, y_vals[i] * 1.05),
                        fontsize=8, ha="center", color="darkorange")

    # Memory
    axs[0].plot(ranks, mem, marker="o", color="tab:red", linestyle="--")
    axs[0].set_title("Peak Memory vs Rank")
    axs[0].set_xlabel("LoRA Rank")
    axs[0].set_ylabel("Memory (MB)")
    axs[0].set_xscale("log", base=2)
    axs[0].set_xticks(ranks); axs[0].set_xticklabels(ranks)
    axs[0].grid(True, alpha=0.3)
    _add_pareto(axs[0], mem)

    # Speed
    axs[1].plot(ranks, speed, marker="s", color="tab:green", linestyle="--")
    axs[1].set_title("Inference Speed vs Rank")
    axs[1].set_xlabel("LoRA Rank")
    axs[1].set_ylabel("Tokens / Sec")
    axs[1].set_xscale("log", base=2)
    axs[1].set_xticks(ranks); axs[1].set_xticklabels(ranks)
    axs[1].grid(True, alpha=0.3)
    _add_pareto(axs[1], speed)

    # Test Loss (accuracy proxy)
    axs[2].plot(ranks, losses, marker="^", color="tab:blue", linestyle="--")
    axs[2].set_title("Test Loss vs Rank  (↓ = better)")
    axs[2].set_xlabel("LoRA Rank")
    axs[2].set_ylabel("Test Loss")
    axs[2].set_xscale("log", base=2)
    axs[2].set_xticks(ranks); axs[2].set_xticklabels(ranks)
    axs[2].grid(True, alpha=0.3)
    _add_pareto(axs[2], losses)

    plt.tight_layout()
    out = RESULTS_DIR / "rank_ablation_metrics.png"
    plt.savefig(out, dpi=150)
    plt.close()
    logging.info(f"Saved plots to {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Task 1: LoRA Rank Ablation Study")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick validation with 2 ranks and 5 iters")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        ranks       = [4, 8]
        iters       = 5
        num_prompts = 5
    else:
        ranks       = [4, 8, 16, 32, 64]
        iters       = 200       # ~10-15 min per rank on M-series
        num_prompts = 100       # 100 Alpaca test prompts

    logging.info(f"Rank Ablation — ranks={ranks}, iters={iters}, prompts={num_prompts}")

    all_results: List[Dict[str, Any]] = []

    for rank in ranks:
        res = train_adapter(rank, iters, RESULTS_DIR)
        if res.get("error"):
            logging.warning(f"[rank={rank}] Skipping inference bench (training failed).")
            continue

        inf = benchmark_inference(rank, res["adapter_path"], num_prompts)
        res.update(inf)
        logging.info(
            f"[rank={rank}] tokens/sec={inf['tokens_per_sec']:.1f}  "
            f"ms/tok={inf['ms_per_token']:.2f}  loss={res['test_loss']:.4f}"
        )
        all_results.append(res)

    if not all_results:
        logging.error("No successful runs – aborting.")
        sys.exit(1)

    matrix = build_comparison_matrix(all_results)

    # Save comparison matrix
    matrix_path = RESULTS_DIR / "comparison_matrix.json"
    matrix_path.write_text(json.dumps(matrix, indent=2))
    logging.info(f"Saved comparison matrix → {matrix_path}")

    # Pareto-optimal rank
    pareto_rank = find_pareto_optimal(matrix)
    pareto_path = RESULTS_DIR / "pareto.json"
    pareto_path.write_text(json.dumps({"pareto_optimal_rank": pareto_rank}, indent=2))
    logging.info(f"Pareto-optimal rank (best accuracy-per-latency): {pareto_rank}")

    # Print table
    print("\n=== Comparison Matrix ===")
    print(f"{'Rank':>6} {'Alpha':>6} {'Scale':>6} {'Tok/s':>9} {'ms/tok':>8} {'Mem MB':>8} {'TestLoss':>9}")
    for r in matrix:
        print(f"{r['rank']:>6} {r['alpha']:>6} {r['scale']:>6.1f} {r['tokens_per_sec']:>9.1f} "
              f"{r['ms_per_token']:>8.2f} {r['peak_mem_mb']:>8.0f} {r['test_loss']:>9.4f}")

    plot_results(matrix, pareto_rank)
    logging.info("Task 1 complete.")


if __name__ == "__main__":
    main()
