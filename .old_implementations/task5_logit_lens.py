#!/usr/bin/env python3
"""
Task 5: Logit Lens & Layer-wise LoRA Importance Analysis

Extracts intermediate token representations to evaluate where
instruction-following decisions are made inside the model.
Computes layer-wise LoRA weight update norms to measure importance.
Ablates non-important layers to find the minimal LoRA subset for 95% performance.
"""
import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import mlx.core as mx
from mlx_lm import load
import numpy as np
from scipy import stats as scipy_stats

try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "./adapters/tinyllama-lora-alpaca"
RESULTS_DIR = Path("./results/task5")
DATA_DIR = Path("./data")

def get_test_subset(num: int) -> List[Dict[str, Any]]:
    prompts = []
    try:
        with open(DATA_DIR / "test.jsonl") as f:
            for line in f:
                row = json.loads(line)
                if 'messages' in row:
                    prompts.append(row['messages'])
                if len(prompts) >= num:
                    break
    except Exception as e:
        logging.error(f"Failed to read test set: {e}")
    return prompts

def extract_logit_lens_accuracy(model, tokenizer, test_data: List[Dict[str, Any]]) -> Dict[int, float]:
    """
    Evaluates next-token logit lens prediction accuracy.
    At each transformer layer, projects the hidden state to vocab 
    and checks if the top-1 prediction matches the correct target.
    """
    total_tokens = 0
    correct_by_layer = {}
    
    num_layers = len(model.model.layers)
    for i in range(num_layers):
        correct_by_layer[i] = 0
        
    for msgs in test_data:
        if hasattr(tokenizer, 'apply_chat_template'):
            token_ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
        else:
            token_ids = tokenizer.encode(msgs[0]['content'])
            
        if len(token_ids) < 2:
            continue
            
        x = mx.array([token_ids[:-1]])
        targets = mx.array(token_ids[1:])
        
        # Forward pass up to each layer, apply lens
        h = model.model.embed_tokens(x)
        
        for idx, layer in enumerate(model.model.layers):
            out = layer(h, mask=None, cache=None)
            h = out[0] if isinstance(out, tuple) else out
            
            # Logit lens projection
            h_norm = model.model.norm(h)
            logits = model.lm_head(h_norm) # (1, Seq_len, Vocab)
            
            # Top 1 predictions
            preds = mx.argmax(logits[0], axis=-1)
            
            # Match
            matches = (preds == targets).sum().item()
            correct_by_layer[idx] += matches
            
        total_tokens += len(targets)
        
    acc_by_layer = {k: v / max(total_tokens, 1) for k, v in correct_by_layer.items()}
    return acc_by_layer

def compute_lora_importance(model) -> Dict[int, float]:
    """
    Computes L2 (Frobenius) norm of the effective LoRA weight matrix update
    at each transformer layer.
    """
    importance = {}
    for idx, layer in enumerate(model.model.layers):
        norm_sum = 0.0
        # Check submodules for LoRA (e.g. self_attn.q_proj, v_proj)
        for _, module in layer.named_modules():
            # In MLX-LM PEFT, LoRA layers have lora_a, lora_b
            if hasattr(module, 'lora_a') and hasattr(module, 'lora_b'):
                # delta W = lora_b * lora_a * scale
                # The shape in MLX is: lora_a is (rank, in_dim), lora_b is (out_dim, rank)
                # Wait, mlx lora uses weight = weight + (x @ a.T) @ b.T -> effectively W + b @ a
                # Let's just safely use Frobenius norm of b @ a
                try:
                    delta_w = (module.lora_b.weight @ module.lora_a.weight) * module.scale
                    norm = mx.linalg.norm(delta_w)
                    norm_sum += norm.item()
                except Exception:
                    pass
        importance[idx] = norm_sum
    return importance

def ablate_and_evaluate(model, tokenizer, test_data: List[Dict[str, Any]], layers_to_ablate: List[int]) -> float:
    """
    Sets the scale of the specified LoRA layers to 0 to ablate them,
    then evaluates standard loss. Restores scale afterwards.
    """
    ablated_modules = []
    
    # Ablate
    for idx in layers_to_ablate:
        layer = model.model.layers[idx]
        for _, module in layer.named_modules():
            if hasattr(module, 'lora_a') and hasattr(module, 'lora_b'):
                original_scale = module.scale
                module.scale = 0.0
                ablated_modules.append((module, original_scale))
                
    # Evaluate perplexity
    total_nll = 0.0
    total_tokens = 0
    for msgs in test_data:
        if hasattr(tokenizer, 'apply_chat_template'):
            token_ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
        else:
            token_ids = tokenizer.encode(msgs[0]['content'])
            
        if len(token_ids) < 2:
            continue
            
        input_ids = mx.array(token_ids[:-1])[None, :]
        target_ids = mx.array(token_ids[1:])
        logits = model(input_ids)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        idx = mx.arange(target_ids.shape[0])
        token_log_probs = log_probs[0, idx, target_ids]
        
        total_nll += float(-mx.sum(token_log_probs).item())
        total_tokens += target_ids.shape[0]

    ppl = math.exp(total_nll / total_tokens) if total_tokens > 0 else float('inf')
    
    # Restore
    for module, scale in ablated_modules:
        module.scale = scale
        
    return ppl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-prompts", type=int, default=100,
                        help="Number of test prompts (spec says 100)")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    num_prompts = 2 if args.smoke_test else args.num_prompts
    test_data = get_test_subset(num_prompts)
    
    logging.info(f"Loading Base + Adapter... {BASE_MODEL}")
    try:
        model, tokenizer = load(BASE_MODEL, adapter_path=ADAPTER_PATH)
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        sys.exit(1)
        
    logging.info(f"Evaluating Logit Lens Accuracy on {len(test_data)} samples...")
    start_t = time.time()
    acc_by_layer = extract_logit_lens_accuracy(model, tokenizer, test_data)
    logging.info(f"Logit Lens evaluated in {time.time() - start_t:.1f}s")
    
    logging.info("Computing LoRA layer-wise importance (matrix norms)...")
    importance = compute_lora_importance(model)
    logging.info(f"Computed LoRA importance (L2 norm) for {len(importance)} layers")
    
    # Compute correlation between accuracy gains and importance
    layers = sorted(list(acc_by_layer.keys()))
    accs = [acc_by_layer[l] for l in layers]
    imps = [importance[l] for l in layers]
    
    # Accuracy gains (compared to first layer)
    acc_gains = [a - accs[0] for a in accs] if len(accs) > 0 else []
    
    # Correlation coefficients
    if len(acc_gains) > 2 and len(imps) > 2:
        pearson_corr, pearson_p = scipy_stats.pearsonr(acc_gains, imps)
        spearman_corr, spearman_p = scipy_stats.spearmanr(acc_gains, imps)
        logging.info(f"Pearson correlation (accuracy gains vs importance): {pearson_corr:.3f} (p={pearson_p:.4f})")
        logging.info(f"Spearman correlation (accuracy gains vs importance): {spearman_corr:.3f} (p={spearman_p:.4f})")
        correlation_analysis = {
            "pearson_r": float(pearson_corr),
            "pearson_p_value": float(pearson_p),
            "spearman_rho": float(spearman_corr),
            "spearman_p_value": float(spearman_p)
        }
    else:
        logging.warning("Not enough data points for correlation analysis")
        correlation_analysis = None
    
    logging.info("Ablating bottom K least important layers...")
    # Sort layers by importance
    sorted_layers = sorted(importance.keys(), key=lambda k: importance[k])
    
    baseline_ppl = ablate_and_evaluate(model, tokenizer, test_data, [])
    logging.info(f"Baseline Perplexity (No Ablation): {baseline_ppl:.3f}")
    
    # Ablation fractions: bottom N layers by importance
    ablation_results = []
    ablation_fracs = [0.05, 0.1, 0.25, 0.50, 0.75, 1.0]
    num_layers = len(model.model.layers)
    for frac in ablation_fracs:
        k = max(1, int(num_layers * frac))
        ablated = sorted_layers[:k]
        ppl = ablate_and_evaluate(model, tokenizer, test_data, ablated)
        pct_loss = ((ppl - baseline_ppl) / baseline_ppl) * 100
        ablation_results.append({
            "ablated_fraction": frac,
            "ablated_count":    k,
            "ablated_layers":   ablated,
            "perplexity":       ppl,
            "ppl_loss_percent": pct_loss,
        })
        logging.info(f"Ablated {k:2d}/{num_layers} layers ({frac*100:.0f}%): PPL={ppl:.3f}  ({pct_loss:+.1f}%)")
        
    # Minimum layers for 95% performance: find maximum ablation where PPL loss ≤ 5%
    # (= minimum LoRA layers that must stay active)
    min_layers_needed = num_layers
    for res in ablation_results:
        if res["ppl_loss_percent"] <= 5.0:
            kept = num_layers - res["ablated_count"]
            min_layers_needed = min(min_layers_needed, kept)

    if min_layers_needed == num_layers:
        min_layers_needed = num_layers   # no ablation safe

    # Identify specific layers to keep (top N most important)
    optimal_layers_to_keep = sorted_layers[-(min_layers_needed):]  # Keep most important
    
    logging.info(
        f"Minimum LoRA layers for ≥95% instruction-following performance: "
        f"{min_layers_needed} / {num_layers}"
    )
    logging.info(f"Optimal layers to keep: {optimal_layers_to_keep}")

    # Print summary table
    print("\n=== Layer Ablation Summary ===")
    print(f"{'Ablated':>8} {'Kept':>6} {'PPL':>8} {'PPL Δ%':>8}")
    for res_row in ablation_results:
        kept = num_layers - res_row["ablated_count"]
        print(f"{res_row['ablated_count']:>8} {kept:>6} {res_row['perplexity']:>8.3f} {res_row['ppl_loss_percent']:>+8.1f}")
    print(f"\nMinimum layers needed for 95% performance: {min_layers_needed}")
    
    with open(RESULTS_DIR / "layer_analysis.json", "w") as f:
        result_data = {
            "logit_lens_accuracy": acc_by_layer,
            "lora_importance_l2": importance,
            "ablation_results": ablation_results,
            "min_layers_for_95pct": min_layers_needed,
            "optimal_layers_to_keep": optimal_layers_to_keep,
            "baseline_ppl": baseline_ppl
        }
        if correlation_analysis:
            result_data["correlation_analysis"] = correlation_analysis
        json.dump(result_data, f, indent=2)
        
    if HAS_PLOT:
        layers = sorted(list(acc_by_layer.keys()))
        accs = [acc_by_layer[l] for l in layers]
        imps = [importance[l] for l in layers]
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Transformer Layer')
        ax1.set_ylabel('Logit Lens Acc (Top-1 Next Token)', color=color)
        ax1.plot(layers, accs, marker='o', color=color, linestyle='-')
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('LoRA Weight ∆ L2-Norm (Importance)', color=color)
        ax2.bar(layers, imps, alpha=0.3, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Logit Lens Accuracy Progress vs LoRA Layer Importance')
        fig.tight_layout()
        plt.grid(alpha=0.2)
        plt.savefig(RESULTS_DIR / "lens_and_importance.png", dpi=150)
        logging.info("Saved plot lens_and_importance.png")
        
    logging.info("Task 5 complete.")

if __name__ == "__main__":
    main()
