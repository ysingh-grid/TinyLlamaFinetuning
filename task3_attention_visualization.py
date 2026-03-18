#!/usr/bin/env python3
"""
Task 3: Attention Visualization & Token Attribution

Extracts attention maps from all heads across all layers.
Implements attention head clustering using cosine similarity.
Implements attention rollout for token attribution.
Generates HTML heatmap visualizations for 5 example generations.
Compares base model vs LoRA-tuned model attention patterns.
"""
import argparse
import collections
import html
import json
import logging
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate

try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False

try:
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    from scipy.spatial.distance import pdist
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "./adapters/tinyllama-lora-alpaca"
RESULTS_DIR = Path("./results/task3")
DATA_DIR = Path("./data")

attention_store = []

def patched_attention_call(self, x, mask=None, cache=None, **kwargs):
    out = self._orig_call(x, mask=mask, cache=cache, **kwargs)
    
    if hasattr(self, "_store_layer_idx"):
        B, L, D = x.shape
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        
        if cache is None:
            queries = self.rope(queries)
            keys = self.rope(keys)
            
            # Handle Grouped Query Attention (GQA)
            if self.n_kv_heads < self.n_heads:
                repeats = self.n_heads // self.n_kv_heads
                keys = mx.repeat(keys, repeats, axis=1)
                
            scores = (queries @ keys.transpose(0, 1, 3, 2)) * self.scale
            if mask is not None and isinstance(mask, mx.array):
                scores = scores + mask
            
            attn_weights = mx.softmax(scores, axis=-1)
            mx.eval(attn_weights)
            
            attention_store.append({
                "layer": self._store_layer_idx,
                "weights": attn_weights
            })
            
    return out

def patch_model_attention(model):
    """Monkey-patch all attention layers to store attention weights."""
    AttnClass = type(model.model.layers[0].self_attn)
    if not hasattr(AttnClass, "_orig_call"):
        AttnClass._orig_call = AttnClass.__call__
        AttnClass.__call__ = patched_attention_call

    for i, layer in enumerate(model.model.layers):
        layer.self_attn._store_layer_idx = i

def get_prompts(num: int) -> List[str]:
    prompts = []
    try:
        with open(DATA_DIR / "test.jsonl") as f:
            for line in f:
                data = json.loads(line)
                msgs = data.get('messages', [])
                user_msg = next((m['content'] for m in msgs if m['role'] == 'user'), None)
                if user_msg:
                    prompts.append(user_msg)
                if len(prompts) >= num:
                    break
    except Exception as e:
        logging.warning("test.jsonl not found, using fallback prompts.")
        prompts = [
            "Explain the difference between a process and a thread.",
            "Write a Python function to calculate the Fibonacci sequence.",
            "What are the main causes of the French Revolution?",
            "Translate the following sentence into French: 'The weather is beautiful today.'",
            "Give me a recipe for chocolate chip cookies."
        ][:num]
    return prompts

def compute_attention_rollout(attentions: List[np.ndarray]) -> np.ndarray:
    """
    Computes attention rollout given a list of layer attention matrices.
    attentions: list of (L, L) arrays, one per layer, representing the mean attention across heads.
    """
    seq_len = attentions[0].shape[0]
    rollout = np.eye(seq_len)
    
    for layer_attn in attentions:
        # Add residual connection
        attn = layer_attn + np.eye(seq_len)
        # Normalize
        attn = attn / attn.sum(axis=-1, keepdims=True)
        # Multiply
        rollout = np.matmul(attn, rollout)
        
    return rollout

def generate_html_heatmap(tokens: List[str], rollout: np.ndarray, title: str) -> str:
    """
    Generates HTML string to highlight tokens based on their attribution to the last token.
    """
    # We look at how the last token attends to all previous tokens
    # rollout shape is (L, L). Last row is rollout[-1]
    attributions = rollout[-1, :]
    
    # Normalize to [0, 1] for coloring
    mx_val = max(1e-9, attributions.max())
    attributions = attributions / mx_val
    
    html_parts = [
        f"<html><head><style>",
        "body { font-family: monospace; line-height: 1.6; padding: 20px; background: #fafafa; }",
        ".token { padding: 2px 4px; border-radius: 3px; display: inline-block; margin: 2px 1px; }",
        "</style></head><body>",
        f"<h2>{html.escape(title)}</h2>",
        "<div style='border: 1px solid #ccc; padding: 15px; background: white; border-radius: 5px;'>"
    ]
    
    for tok, attr in zip(tokens, attributions):
        # Color from white (0) to red (1)
        r = 255
        g = int(255 * (1 - attr))
        b = int(255 * (1 - attr))
        
        # Text color
        color = "white" if attr > 0.5 else "black"
        escaped_tok = html.escape(str(tok).replace(" ", " "))
        html_parts.append(
            f"<span class='token' style='background-color: rgb({r},{g},{b}); color: {color};'>{escaped_tok}</span>"
        )
        
    html_parts.append("</div></body></html>")
    return "\n".join(html_parts)

def cluster_attention_heads(all_head_patterns: np.ndarray, model_name: str):
    """
    all_head_patterns: shape (num_heads_total, num_samples)
    Returns dict: head_index -> cluster_label
    """
    if not HAS_PLOT or not HAS_SCIPY:
        return {}

    logging.info(f"Clustering {all_head_patterns.shape[0]} attention heads for {model_name}…")

    dists = pdist(all_head_patterns, metric="cosine")
    Z = linkage(dists, method="ward")
    labels = fcluster(Z, t=3, criterion="maxclust")

    plt.figure(figsize=(12, 6))
    dendrogram(Z, truncate_mode="lastp", p=30, show_leaf_counts=True)
    plt.title(f"Attention Head Clustering — {model_name}")
    plt.xlabel("Cluster Size / Head Index")
    plt.ylabel("Ward Distance")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"head_clustering_{model_name.replace(' ', '_')}.png", dpi=150)
    plt.close()

    return {int(i): int(labels[i]) for i in range(len(labels))}

def extract_and_analyze(model, tokenizer, prompts, model_name):
    global attention_store
    
    html_outputs = []
    
    # Shape: list of patterns per head, per layer
    # We will compute the mean attention distance across sequence for each head to use as a signature
    # Signature per head: 1D vector of length = len(prompts)
    head_signatures = collections.defaultdict(list)
    
    for p_idx, prompt in enumerate(prompts):
        attention_store.clear()
        
        if hasattr(tokenizer, 'apply_chat_template'):
            text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        else:
            text = prompt + "\nAssistant: "
            
        tokens = tokenizer.encode(text)
        
        # Forward pass to capture attention
        x = mx.array([tokens])
        out = model(x)
        mx.eval(out)
        
        # We now have attention_store with layer weights: shape (1, n_heads, L, L)
        # Filter layers by index to ensure order
        layer_weights = {}
        for item in attention_store:
            l = item["layer"]
            # Convert bfloat16 to float32 before numpy conversion
            layer_weights[l] = np.array(item["weights"].astype(mx.float32))
            
        num_layers = max(layer_weights.keys()) + 1
        num_heads = layer_weights[0].shape[1]
        
        layer_means = []
        for l in range(num_layers):
            weights = layer_weights[l][0] # (n_heads, L, L)
            # Signature: mean attention entropy or diagonal mass for each head
            # Let's use average diagonal mass (local attention) as a signature
            for h in range(num_heads):
                diag_mass = np.trace(weights[h]) / weights.shape[-1]
                head_signatures[(l, h)].append(diag_mass)
                
            layer_means.append(weights.mean(axis=0)) # (L, L)
            
        # Compute rollout
        rollout = compute_attention_rollout(layer_means)
        
        if p_idx < 5:
            # Generate HTML
            # Get token strings
            tok_strs = [tokenizer.decode([t]) for t in tokens]
            html_str = generate_html_heatmap(tok_strs, rollout, f"Token Attribution - {model_name} (Prompt {p_idx+1})")
            html_outputs.append((p_idx, html_str))
            
    # Save HTMLs
    for p_idx, html_str in html_outputs:
        with open(RESULTS_DIR / f"attention_heatmap_{model_name.replace(' ', '_')}_p{p_idx}.html", "w") as f:
            f.write(html_str)
            
    # Clustering
    num_heads_total = len(head_signatures)
    signatures_matrix = np.zeros((num_heads_total, len(prompts)))
    for i, (l, h) in enumerate(sorted(head_signatures.keys())):
        signatures_matrix[i, :] = head_signatures[(l, h)]
        
    cluster_labels = cluster_attention_heads(signatures_matrix, model_name)

    # ---- Identify instruction-following vs completion heads ----
    # Instruction-following heads = high diagonal mass on PROMPT tokens (attend locally).
    # Completion heads            = low diagonal mass (attend far back = cross-token).
    # We classify each head by whether its mean signature exceeds the global median.
    inf_heads, comp_heads = [], []
    keys = sorted(head_signatures.keys())
    all_vals = [statistics.mean(head_signatures[k]) for k in keys]
    threshold = statistics.median(all_vals) if all_vals else 0.5
    for i, k in enumerate(keys):
        mean_val = statistics.mean(head_signatures[k])
        (inf_heads if mean_val >= threshold else comp_heads).append(k)

    head_json = {
        "instruction_following_heads": [list(h) for h in inf_heads],
        "completion_heads":            [list(h) for h in comp_heads],
        "cluster_labels": cluster_labels,
        "median_threshold": float(threshold),
    }
    with open(RESULTS_DIR / f"head_classification_{model_name.replace(' ', '_')}.json", "w") as f:
        json.dump(head_json, f, indent=2)
    logging.info(
        f"{model_name}: {len(inf_heads)} instruction-following heads, "
        f"{len(comp_heads)} completion heads  (threshold={threshold:.4f})"
    )
    return signatures_matrix

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=20)
    args = parser.parse_args()
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    prompts = get_prompts(args.num_samples)
    
    # 1. Base Model
    logging.info("Loading Base model...")
    base_model, base_tokenizer = load(BASE_MODEL)
    patch_model_attention(base_model)
    logging.info("Analyzing Base model attentions...")
    base_sigs = extract_and_analyze(base_model, base_tokenizer, prompts, "Base")
    del base_model
    
    # 2. LoRA Model
    logging.info("Loading LoRA model...")
    lora_model, lora_tokenizer = load(BASE_MODEL, adapter_path=ADAPTER_PATH)
    patch_model_attention(lora_model)
    logging.info("Analyzing LoRA model attentions...")
    lora_sigs = extract_and_analyze(lora_model, lora_tokenizer, prompts, "LoRA")
    
    # Compare
    if HAS_PLOT:
        logging.info("Generating base-vs-LoRA comparison heatmap…")
        if base_sigs.shape == lora_sigs.shape:
            diff = np.abs(base_sigs - lora_sigs).mean(axis=1)
        else:
            min_h = min(base_sigs.shape[0], lora_sigs.shape[0])
            diff = np.abs(base_sigs[:min_h] - lora_sigs[:min_h]).mean(axis=1)

        num_layers = len(lora_model.model.layers)
        num_heads  = len(diff) // num_layers
        diff_matrix = diff[:num_layers * num_heads].reshape((num_layers, num_heads))

        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(diff_matrix, aspect="auto", cmap="YlOrRd")
        plt.colorbar(im, ax=ax, label="Mean |Δ Diagonal Mass|")
        ax.set_title("Attention Head Changes: Base vs LoRA-tuned")
        ax.set_xlabel("Attention Head Index")
        ax.set_ylabel("Transformer Layer")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "base_vs_lora_attention_diff.png", dpi=150)
        plt.close()
        
    logging.info("Task 3 complete. Results saved to results/task3/")

if __name__ == "__main__":
    main()
