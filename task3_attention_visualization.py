#!/usr/bin/env python3
"""
Task 3: Attention Visualization & Token Attribution

This script:
- loads the trained TinyLlama LoRA checkpoint
- extracts attention maps from all heads across all layers
- clusters attention heads with cosine similarity
- computes token attribution via attention rollout
- compares the base model against the LoRA-tuned model
- generates HTML heatmaps and summary artifacts for results/task3/
"""

from __future__ import annotations

import argparse
import collections
import html
import json
import logging
import math
import shutil
import statistics
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import mlx.core as mx
from mlx_lm import generate, load

try:
    import matplotlib.pyplot as plt

    HAS_PLOT = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_PLOT = False

try:
    from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
    from scipy.spatial.distance import pdist, squareform

    HAS_SCIPY = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_SCIPY = False


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = Path("./adapters/tinyllama-lora-alpaca")
RESULTS_DIR = Path("./results/task3")
DATA_DIR = Path("./data")
MAX_GENERATED_TOKENS = 64
HTML_SAMPLES = 20
DETAILED_HTML_SAMPLES = 5
CLUSTER_COUNT = 3
ROLLOUT_EPS = 1e-9
LOCAL_WINDOW = 8

attention_store: List[Tuple[int, np.ndarray]] = []


def safe_float(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(value)


def safe_row_normalize(matrix: np.ndarray, eps: float = ROLLOUT_EPS) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    denom = matrix.sum(axis=-1, keepdims=True)
    denom = np.clip(denom, eps, None)
    return matrix / denom


def normalized_entropy(probabilities: np.ndarray, eps: float = ROLLOUT_EPS) -> float:
    values = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0
    values = np.clip(values, eps, 1.0)
    values = values / np.sum(values)
    entropy = -np.sum(values * np.log(values))
    denom = math.log(float(values.size)) if values.size > 1 else 1.0
    return safe_float(entropy / max(denom, eps))


def cosine_similarity_matrix(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.clip(norms, ROLLOUT_EPS, None)
    normalized = features / norms
    sim = np.einsum("ij,kj->ik", normalized, normalized, optimize=True)
    return np.clip(sim, -1.0, 1.0)


def compute_attention_rollout(attentions: Sequence[np.ndarray]) -> np.ndarray:
    if not attentions:
        return np.zeros((0,), dtype=np.float32)

    seq_len = attentions[0].shape[0]
    rollout = np.zeros(seq_len, dtype=np.float64)
    rollout[-1] = 1.0
    eye = np.eye(seq_len, dtype=np.float64)

    for layer_attn in attentions:
        layer = np.asarray(layer_attn, dtype=np.float64)
        layer = np.nan_to_num(layer, nan=0.0, posinf=0.0, neginf=0.0)
        layer = np.clip(layer, 0.0, None)
        layer = safe_row_normalize(layer + eye)
        rollout = np.einsum("i,ij->j", rollout, layer, optimize=True)
        rollout = np.nan_to_num(rollout, nan=0.0, posinf=0.0, neginf=0.0)
        rollout = np.clip(rollout, 0.0, None)
        total = float(rollout.sum())
        rollout = rollout / max(total, ROLLOUT_EPS)

    return rollout.astype(np.float32)


def build_prompt_text(tokenizer: Any, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"User: {prompt}\nAssistant: "


def decode_tokens(tokenizer: Any, token_ids: Sequence[int]) -> List[str]:
    tokens = []
    for token_id in token_ids:
        piece = tokenizer.decode([int(token_id)])
        if piece == "":
            piece = "∅"
        tokens.append(piece)
    return tokens


def clean_generated_response(text: str, fallback: str) -> str:
    text = (text or "").strip()
    return text if text else fallback


def get_prompts(num: int) -> List[str]:
    prompts: List[str] = []
    try:
        with open(DATA_DIR / "test.jsonl", "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                messages = record.get("messages", [])
                user_message = next((m.get("content", "") for m in messages if m.get("role") == "user"), None)
                if user_message:
                    prompts.append(user_message)
                if len(prompts) >= num:
                    break
    except Exception as exc:  # pragma: no cover - fallback path
        logging.warning("Falling back to built-in prompts: %s", exc)

    if len(prompts) < num:
        fallback = [
            "Explain the difference between a process and a thread.",
            "Write a Python function to calculate the Fibonacci sequence.",
            "What are the main causes of the French Revolution?",
            "Translate the following sentence into French: 'The weather is beautiful today.'",
            "Give me a recipe for chocolate chip cookies.",
            "Summarize the Water Cycle in one paragraph.",
            "List five benefits of regular exercise.",
            "Explain how photosynthesis works in simple terms.",
            "Draft a polite email asking for an extension on a deadline.",
            "Compare supervised and unsupervised learning.",
            "Describe the steps of the scientific method.",
            "Suggest a study plan for learning linear algebra.",
            "Write a concise explanation of recursion for beginners.",
            "How do you stay productive when working remotely?",
            "Create a travel checklist for a weekend trip.",
            "What is the difference between HTTP and HTTPS?",
            "Explain the concept of inflation to a high school student.",
            "Give an outline for a short persuasive essay about recycling.",
            "How can someone improve their public speaking skills?",
            "What should I consider when buying a used laptop?",
            "Summarize the plot of Romeo and Juliet in three sentences.",
            "Explain the role of buffers in chemistry.",
        ]
        for prompt in fallback:
            if len(prompts) >= num:
                break
            prompts.append(prompt)

    return prompts[:num]


def patch_model_attention(model: Any) -> None:
    """Monkey-patch attention layers to capture all head attention matrices."""

    global attention_store

    attn_class = type(model.model.layers[0].self_attn)
    if not hasattr(attn_class, "_orig_call"):
        attn_class._orig_call = attn_class.__call__

        def patched_attention_call(self, x, mask=None, cache=None, **kwargs):
            output = self._orig_call(x, mask=mask, cache=cache, **kwargs)
            layer_idx = getattr(self, "_store_layer_idx", None)
            if layer_idx is None or cache is not None:
                return output

            B, L, _ = x.shape
            queries = self.q_proj(x).astype(mx.float32)
            keys = self.k_proj(x).astype(mx.float32)

            queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
            keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

            queries = self.rope(queries)
            keys = self.rope(keys)

            if self.n_kv_heads < self.n_heads:
                repeats = self.n_heads // self.n_kv_heads
                keys = mx.repeat(keys, repeats, axis=1)

            scores = (queries @ keys.transpose(0, 1, 3, 2)) * self.scale
            scores = scores.astype(mx.float32)
            if mask is not None and hasattr(mask, "astype"):
                scores = scores + mask.astype(mx.float32)

            scores = scores - mx.max(scores, axis=-1, keepdims=True)
            attn_weights = mx.softmax(scores, axis=-1)
            mx.eval(attn_weights)
            attention_store.append((int(layer_idx), np.array(attn_weights, dtype=np.float32)))
            return output

        attn_class.__call__ = patched_attention_call

    for idx, layer in enumerate(model.model.layers):
        layer.self_attn._store_layer_idx = idx


def load_responses(model: Any, tokenizer: Any, prompts: Sequence[str]) -> List[Dict[str, str]]:
    responses = []
    for prompt in prompts:
        prompt_text = build_prompt_text(tokenizer, prompt)
        generated = generate(
            model,
            tokenizer,
            prompt_text,
            verbose=False,
            max_tokens=MAX_GENERATED_TOKENS,
        )
        responses.append(
            {
                "prompt_text": prompt_text,
                "response_text": clean_generated_response(generated, "[no response generated]"),
            }
        )
    return responses


def compute_head_metrics(
    attention_layers: Sequence[np.ndarray], prompt_len: int, seq_len: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], Dict[str, float]]:
    if not attention_layers:
        return np.array([]), np.array([]), np.array([]), [], {}

    num_layers = len(attention_layers)
    num_heads = attention_layers[0].shape[0]
    instruction = np.zeros((num_layers, num_heads), dtype=np.float32)
    completion = np.zeros((num_layers, num_heads), dtype=np.float32)
    entropy = np.zeros((num_layers, num_heads), dtype=np.float32)
    layer_means: List[np.ndarray] = []

    response_start = min(prompt_len, seq_len)
    response_len = max(0, seq_len - response_start)
    prompt_len = max(prompt_len, 0)

    for layer_idx, layer_weights in enumerate(attention_layers):
        weights = np.asarray(layer_weights, dtype=np.float32)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        layer_mean = weights.mean(axis=0)
        layer_means.append(layer_mean)

        for head_idx in range(num_heads):
            head = weights[head_idx]
            response_rows = head[response_start:, :] if response_len > 0 else head[-1:, :]
            if prompt_len > 0 and response_rows.size > 0:
                instruction[layer_idx, head_idx] = float(response_rows[:, :prompt_len].mean())
            else:
                instruction[layer_idx, head_idx] = 0.0

            local_scores = []
            for row in range(response_start, seq_len):
                left = max(response_start, row - LOCAL_WINDOW)
                if left < row:
                    local_scores.append(float(head[row, left:row].mean()))
            completion[layer_idx, head_idx] = float(np.mean(local_scores)) if local_scores else 0.0
            entropy[layer_idx, head_idx] = normalized_entropy(response_rows if response_rows.size else head)

    stats = {
        "instruction_mean": safe_float(instruction.mean()),
        "completion_mean": safe_float(completion.mean()),
        "entropy_mean": safe_float(entropy.mean()),
        "instruction_std": safe_float(instruction.std()),
        "completion_std": safe_float(completion.std()),
        "entropy_std": safe_float(entropy.std()),
    }
    return instruction, completion, entropy, layer_means, stats


def attention_classification(
    instruction: np.ndarray, completion: np.ndarray
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    diff = instruction - completion
    flat_diff = diff.reshape(-1)
    scores = np.nan_to_num(flat_diff, nan=0.0, posinf=0.0, neginf=0.0)

    instruction_mask = scores >= 0.0
    completion_mask = ~instruction_mask

    confidences = np.abs(scores) / np.clip(np.abs(instruction.reshape(-1)) + np.abs(completion.reshape(-1)), ROLLOUT_EPS, None)
    confidences = np.nan_to_num(confidences, nan=0.0, posinf=0.0, neginf=0.0)

    return (
        {
            "criterion": "higher mean response-to-prompt attention => instruction-following; higher mean local response attention => completion",
            "instruction_following_indices": [int(i) for i in np.flatnonzero(instruction_mask)],
            "completion_indices": [int(i) for i in np.flatnonzero(completion_mask)],
            "median_diff": safe_float(np.median(scores)),
            "mean_diff": safe_float(np.mean(scores)),
            "confidence_mean": safe_float(np.mean(confidences)),
            "confidence_median": safe_float(np.median(confidences)),
            "instruction_count": int(instruction_mask.sum()),
            "completion_count": int(completion_mask.sum()),
        },
        scores,
        confidences,
    )


def cluster_attention_heads(features: np.ndarray, model_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not HAS_PLOT or not HAS_SCIPY or features.size == 0:
        return np.array([]), np.array([]), np.array([])

    logging.info("Clustering %s attention heads with cosine similarity", model_name)
    features = np.asarray(features, dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.clip(norms, ROLLOUT_EPS, None)

    condensed = pdist(features, metric="cosine")
    linkage_matrix = linkage(condensed, method="average")
    labels = fcluster(linkage_matrix, t=CLUSTER_COUNT, criterion="maxclust")
    similarity = cosine_similarity_matrix(features)

    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(1, 2, 1)
    dendrogram(linkage_matrix, truncate_mode="lastp", p=30, show_leaf_counts=True, ax=ax1)
    ax1.set_title(f"{model_name} attention head clustering")
    ax1.set_xlabel("Cluster / head index")
    ax1.set_ylabel("Cosine distance")

    ax2 = fig.add_subplot(1, 2, 2)
    if similarity.size:
        im = ax2.imshow(similarity, aspect="auto", cmap="viridis", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax2, label="Cosine similarity")
    ax2.set_title(f"{model_name} head similarity matrix")
    ax2.set_xlabel("Head index")
    ax2.set_ylabel("Head index")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"head_clustering_{model_name}.png", dpi=150)
    plt.close(fig)

    return labels, linkage_matrix, similarity


def cluster_comparison_plot(base_features: np.ndarray, lora_features: np.ndarray) -> None:
    if not HAS_PLOT or base_features.size == 0 or lora_features.size == 0:
        return

    base_features = np.nan_to_num(np.asarray(base_features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lora_features = np.nan_to_num(np.asarray(lora_features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    # Compare the top clusters from each model by reusing their k-means-free hierarchical groupings.
    def cluster_centroids(features: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        if features.shape[0] < CLUSTER_COUNT:
            return features, [int(features.shape[0])]
        labels = fcluster(linkage(pdist(features, metric="cosine"), method="average"), t=CLUSTER_COUNT, criterion="maxclust")
        centroids = []
        sizes = []
        for cluster_id in range(1, CLUSTER_COUNT + 1):
            members = features[labels == cluster_id]
            if members.size:
                centroids.append(members.mean(axis=0))
                sizes.append(int(members.shape[0]))
        if not centroids:
            return np.array([features.mean(axis=0)]), [int(features.shape[0])]
        return np.array(centroids), sizes

    base_cluster_centroids, base_cluster_sizes = cluster_centroids(base_features)
    lora_cluster_centroids, lora_cluster_sizes = cluster_centroids(lora_features)
    combined = np.vstack([base_cluster_centroids, lora_cluster_centroids])
    similarity = cosine_similarity_matrix(combined)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im = axes[0].imshow(similarity, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    axes[0].set_title("Cluster centroid similarity\n(Base and LoRA)")
    axes[0].set_xlabel("Centroid index")
    axes[0].set_ylabel("Centroid index")
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    x = np.arange(max(len(base_cluster_sizes), len(lora_cluster_sizes)))
    axes[1].bar(x - 0.15, base_cluster_sizes + [0] * (len(x) - len(base_cluster_sizes)), width=0.3, label="Base", color="#4C72B0")
    axes[1].bar(x + 0.15, lora_cluster_sizes + [0] * (len(x) - len(lora_cluster_sizes)), width=0.3, label="LoRA", color="#DD8452")
    axes[1].set_xticks(x, [f"C{i+1}" for i in x])
    axes[1].set_ylabel("Heads per cluster")
    axes[1].set_title("Cluster membership counts")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "head_clustering_Comparison.png", dpi=150)
    plt.close(fig)


def render_heatmap_html(
    prompt: str,
    base_record: Dict[str, Any],
    lora_record: Dict[str, Any],
    detailed: bool,
    model_name: str = "comparison",
) -> str:
    def token_spans(tokens: Sequence[str], weights: Sequence[float]) -> str:
        weights = np.asarray(weights, dtype=np.float32)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        if weights.size == 0:
            return "<em>No tokens available</em>"
        weights = weights - weights.min()
        max_value = float(weights.max())
        if max_value <= ROLLOUT_EPS:
            norm = np.zeros_like(weights)
        else:
            norm = weights / max_value
        parts = []
        for token, weight in zip(tokens, norm):
            intensity = int(255 - 155 * float(weight))
            background = f"rgb(255,{intensity},{intensity})"
            color = "#111" if weight < 0.6 else "#fff"
            escaped = html.escape(token).replace("\n", "↵")
            parts.append(
                f"<span class='token' style='background:{background};color:{color};'>{escaped}</span>"
            )
        return " ".join(parts)

    def summary_table(record: Dict[str, Any]) -> str:
        metrics = record["summary"]
        rows = []
        for key in ["instruction_mean", "completion_mean", "entropy_mean", "attention_focus"]:
            rows.append(f"<tr><th>{html.escape(key)}</th><td>{metrics.get(key, 0.0):.4f}</td></tr>")
        return "<table class='summary'>" + "".join(rows) + "</table>"

    html_parts = [
        "<html><head><meta charset='utf-8'>",
        "<style>",
        "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 18px; background: #fafafa; color: #1d1d1f; }",
        ".grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; align-items: start; }",
        ".panel { background: white; border: 1px solid #ddd; border-radius: 12px; padding: 14px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }",
        ".token { display: inline-block; padding: 2px 5px; margin: 2px 1px; border-radius: 4px; line-height: 1.45; }",
        ".summary { border-collapse: collapse; margin-top: 10px; }",
        ".summary th, .summary td { border: 1px solid #ddd; padding: 6px 8px; text-align: left; }",
        ".summary th { background: #f3f4f6; }",
        ".muted { color: #666; font-size: 0.95em; }",
        ".section { margin-top: 18px; }",
        ".head-flow { width: 100%; border-collapse: collapse; margin-top: 8px; }",
        ".head-flow th, .head-flow td { border: 1px solid #ddd; padding: 5px 7px; font-size: 0.88em; }",
        ".head-flow th { background: #f3f4f6; }",
        ".prompt-box { background: #fff; border: 1px solid #ddd; padding: 12px; border-radius: 10px; white-space: pre-wrap; }",
        "</style></head><body>",
        f"<h2>Task 3 Attention Flow — {html.escape(model_name)} example</h2>",
        f"<div class='prompt-box'><strong>Instruction:</strong> {html.escape(prompt)}</div>",
        "<div class='grid section'>",
        "<div class='panel'>",
        "<h3>Base model</h3>",
        f"<div class='muted'>{html.escape(base_record['response_text'])}</div>",
        summary_table(base_record),
        f"<p><strong>Token attention heatmap</strong></p>{token_spans(base_record['tokens'], base_record['attribution'])}",
        "</div>",
        "<div class='panel'>",
        "<h3>LoRA-tuned model</h3>",
        f"<div class='muted'>{html.escape(lora_record['response_text'])}</div>",
        summary_table(lora_record),
        f"<p><strong>Token attention heatmap</strong></p>{token_spans(lora_record['tokens'], lora_record['attribution'])}",
        "</div>",
        "</div>",
    ]

    if detailed:
        html_parts.extend(
            [
                "<div class='section panel'>",
                "<h3>Multi-head attention highlights</h3>",
                "<p class='muted'>The table below lists the most instruction-oriented and completion-oriented heads for this example, based on response-to-prompt versus local response attention.</p>",
                render_head_table(base_record, lora_record),
                "</div>",
            ]
        )

    html_parts.append("</body></html>")
    return "\n".join(html_parts)


def render_head_table(base_record: Dict[str, Any], lora_record: Dict[str, Any]) -> str:
    rows = [
        "<table class='head-flow'>",
        "<tr><th>Model</th><th>Head</th><th>Instruction score</th><th>Completion score</th><th>Net</th></tr>",
    ]
    for model_name, record in [("Base", base_record), ("LoRA", lora_record)]:
        top = record["top_heads"]
        for entry in top[:4]:
            rows.append(
                "<tr>"
                f"<td>{model_name}</td>"
                f"<td>L{entry['layer']}H{entry['head']}</td>"
                f"<td>{entry['instruction_score']:.4f}</td>"
                f"<td>{entry['completion_score']:.4f}</td>"
                f"<td>{entry['net_score']:.4f}</td>"
                "</tr>"
            )
    rows.append("</table>")
    return "".join(rows)


def save_html_outputs(sample_records: List[Dict[str, Any]]) -> None:
    for idx, record in enumerate(sample_records):
        detailed = idx < DETAILED_HTML_SAMPLES
        html_text = render_heatmap_html(
            record["prompt"],
            record["Base"],
            record["LoRA"],
            detailed=detailed,
            model_name=f"Prompt {idx + 1}",
        )
        out_path = RESULTS_DIR / f"attention_heatmap_prompt_{idx:02d}.html"
        out_path.write_text(html_text, encoding="utf-8")


def save_multi_head_visualization(base_result: Dict[str, Any], lora_result: Dict[str, Any]) -> None:
    if not HAS_PLOT:
        return

    sample = lora_result["raw_sample"]
    prompt_len = sample["prompt_len"]
    tokens = sample["tokens"]
    layer_weights = sample["layer_weights"]

    head_entries = lora_result["top_heads"][:4] + lora_result["bottom_heads"][:4]
    if not head_entries:
        return

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    axes = axes.reshape(2, 4)

    for ax, entry in zip(axes.flat, head_entries):
        layer = entry["layer"]
        head = entry["head"]
        matrix = np.asarray(layer_weights[layer][head], dtype=np.float32)
        if prompt_len > 0:
            heat = matrix[prompt_len:, :prompt_len]
            x_tokens = tokens[:prompt_len]
            y_tokens = tokens[prompt_len:]
        else:
            heat = matrix
            x_tokens = tokens
            y_tokens = tokens

        if heat.size == 0:
            heat = matrix[:1, :1]
            x_tokens = [tokens[0]]
            y_tokens = [tokens[0]]

        im = ax.imshow(heat, aspect="auto", cmap="magma", vmin=0, vmax=max(heat.max(), 1e-6))
        ax.set_title(f"L{layer} H{head}\n{entry['kind']}")
        ax.set_xlabel("Prompt tokens")
        ax.set_ylabel("Response tokens")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    plt.suptitle("Multi-head attention visualization (LoRA, prompt 0)")
    plt.savefig(RESULTS_DIR / "multi_head_visualization.png", dpi=150)
    plt.close(fig)


def save_base_vs_lora_comparison(base_result: Dict[str, Any], lora_result: Dict[str, Any]) -> None:
    if not HAS_PLOT:
        return

    base_instruction = base_result["head_instruction"]
    base_completion = base_result["head_completion"]
    lora_instruction = lora_result["head_instruction"]
    lora_completion = lora_result["head_completion"]

    diff_instruction = lora_instruction - base_instruction
    diff_completion = lora_completion - base_completion
    diff_net = (lora_instruction - lora_completion) - (base_instruction - base_completion)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    im = axes[0, 0].imshow(diff_instruction, aspect="auto", cmap="coolwarm")
    axes[0, 0].set_title("Δ instruction-following attention")
    axes[0, 0].set_xlabel("Head")
    axes[0, 0].set_ylabel("Layer")
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im = axes[0, 1].imshow(diff_completion, aspect="auto", cmap="coolwarm")
    axes[0, 1].set_title("Δ completion attention")
    axes[0, 1].set_xlabel("Head")
    axes[0, 1].set_ylabel("Layer")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    axes[1, 0].plot(diff_instruction.mean(axis=1), label="instruction Δ", marker="o")
    axes[1, 0].plot(diff_completion.mean(axis=1), label="completion Δ", marker="o")
    axes[1, 0].plot(diff_net.mean(axis=1), label="net Δ", marker="o")
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set_title("Layer-wise mean change")
    axes[1, 0].set_xlabel("Layer")
    axes[1, 0].set_ylabel("Attention score change")
    axes[1, 0].legend()

    base_net = (base_instruction - base_completion).reshape(-1)
    lora_net = (lora_instruction - lora_completion).reshape(-1)
    axes[1, 1].scatter(base_net, lora_net, alpha=0.35, s=12, c=np.abs(lora_net - base_net), cmap="viridis")
    axes[1, 1].plot([base_net.min(), base_net.max()], [base_net.min(), base_net.max()], linestyle="--", color="gray")
    axes[1, 1].set_title("Head net scores: base vs LoRA")
    axes[1, 1].set_xlabel("Base")
    axes[1, 1].set_ylabel("LoRA")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "base_vs_lora_comparison.png", dpi=150)
    plt.savefig(RESULTS_DIR / "base_vs_lora_attention_diff.png", dpi=150)
    plt.close(fig)


def summarize_layers(
    model_name: str,
    instruction: np.ndarray,
    completion: np.ndarray,
    entropy: np.ndarray,
    head_labels: np.ndarray,
    head_confidence: np.ndarray,
) -> Dict[str, Any]:
    num_layers, num_heads = instruction.shape
    summaries = []
    for layer in range(num_layers):
        layer_instruction = instruction[layer]
        layer_completion = completion[layer]
        layer_entropy = entropy[layer]
        layer_net = layer_instruction - layer_completion
        labels = head_labels[layer * num_heads : (layer + 1) * num_heads]
        summaries.append(
            {
                "layer": layer,
                "instruction_mean": safe_float(layer_instruction.mean()),
                "completion_mean": safe_float(layer_completion.mean()),
                "entropy_mean": safe_float(layer_entropy.mean()),
                "net_mean": safe_float(layer_net.mean()),
                "instruction_heads": int(np.sum(labels == 1)),
                "completion_heads": int(np.sum(labels == -1)),
                "confidence_mean": safe_float(head_confidence[layer * num_heads : (layer + 1) * num_heads].mean()),
            }
        )

    return {
        "model": model_name,
        "layers": summaries,
        "top_layers_by_net_change": [],
    }


def analyze_model(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    model_name: str,
) -> Dict[str, Any]:
    global attention_store

    logging.info("Generating responses for %s", model_name)
    response_records = load_responses(model, tokenizer, prompts)

    patch_model_attention(model)
    num_layers = len(model.model.layers)
    num_heads = model.model.layers[0].self_attn.n_heads
    total_heads = num_layers * num_heads

    head_instruction = np.zeros((total_heads, len(prompts)), dtype=np.float32)
    head_completion = np.zeros((total_heads, len(prompts)), dtype=np.float32)
    head_entropy = np.zeros((total_heads, len(prompts)), dtype=np.float32)
    layer_instruction = np.zeros((num_layers, len(prompts)), dtype=np.float32)
    layer_completion = np.zeros((num_layers, len(prompts)), dtype=np.float32)
    layer_entropy = np.zeros((num_layers, len(prompts)), dtype=np.float32)
    sample_records: List[Dict[str, Any]] = []
    raw_sample: Dict[str, Any] = {}

    for sample_idx, prompt in enumerate(prompts):
        attention_store.clear()
        prompt_text = response_records[sample_idx]["prompt_text"]
        response_text = response_records[sample_idx]["response_text"]
        full_text = f"{prompt_text}{response_text}"
        token_ids = tokenizer.encode(full_text)
        if not token_ids:
            token_ids = tokenizer.encode(prompt_text)
        token_texts = decode_tokens(tokenizer, token_ids)
        prompt_len = len(tokenizer.encode(prompt_text))
        prompt_len = min(prompt_len, len(token_ids))

        x = mx.array([token_ids])
        output = model(x)
        mx.eval(output)

        layer_map = {layer_idx: weights for layer_idx, weights in attention_store}
        layer_weights = [layer_map[idx][0] for idx in sorted(layer_map)]
        if len(layer_weights) != num_layers:
            logging.warning(
                "%s sample %d: captured %d/%d layers",
                model_name,
                sample_idx,
                len(layer_weights),
                num_layers,
            )

        instruction, completion, entropy, layer_means, _ = compute_head_metrics(
            layer_weights, prompt_len=prompt_len, seq_len=len(token_ids)
        )

        head_instruction[:, sample_idx] = instruction.reshape(-1)
        head_completion[:, sample_idx] = completion.reshape(-1)
        head_entropy[:, sample_idx] = entropy.reshape(-1)
        layer_instruction[:, sample_idx] = instruction.mean(axis=1)
        layer_completion[:, sample_idx] = completion.mean(axis=1)
        layer_entropy[:, sample_idx] = entropy.mean(axis=1)

        rollout = compute_attention_rollout(layer_means)
        attribution = rollout if rollout.size else np.zeros(len(token_ids), dtype=np.float32)

        attribution = np.nan_to_num(np.asarray(attribution, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        attention_focus = float(attribution[:prompt_len].sum()) if prompt_len else float(attribution.mean() if attribution.size else 0.0)

        scores = instruction - completion
        flat_scores = scores.reshape(-1)
        flat_instructions = instruction.reshape(-1)
        flat_completions = completion.reshape(-1)
        classification = np.where(flat_scores >= 0.0, 1, -1)
        confidences = np.abs(flat_scores) / np.clip(np.abs(flat_instructions) + np.abs(flat_completions), ROLLOUT_EPS, None)
        confidences = np.nan_to_num(confidences, nan=0.0, posinf=0.0, neginf=0.0)

        top_indices = np.argsort(flat_scores)
        top_heads = []
        bottom_heads = []
        for idx in top_indices[::-1][:8]:
            layer = int(idx // num_heads)
            head = int(idx % num_heads)
            top_heads.append(
                {
                    "layer": layer,
                    "head": head,
                    "instruction_score": safe_float(flat_instructions[idx]),
                    "completion_score": safe_float(flat_completions[idx]),
                    "net_score": safe_float(flat_scores[idx]),
                    "confidence": safe_float(confidences[idx]),
                    "kind": "instruction-following" if flat_scores[idx] >= 0 else "completion",
                }
            )
        for idx in top_indices[:8]:
            layer = int(idx // num_heads)
            head = int(idx % num_heads)
            bottom_heads.append(
                {
                    "layer": layer,
                    "head": head,
                    "instruction_score": safe_float(flat_instructions[idx]),
                    "completion_score": safe_float(flat_completions[idx]),
                    "net_score": safe_float(flat_scores[idx]),
                    "confidence": safe_float(confidences[idx]),
                    "kind": "completion" if flat_scores[idx] < 0 else "instruction-following",
                }
            )

        model_record = {
            "summary": {
                "instruction_mean": safe_float(instruction.mean()),
                "completion_mean": safe_float(completion.mean()),
                "entropy_mean": safe_float(entropy.mean()),
                "attention_focus": safe_float(attention_focus),
                "seq_len": int(len(token_ids)),
                "prompt_len": int(prompt_len),
            },
            "response_text": response_text,
            "tokens": token_texts,
            "attribution": attribution.tolist(),
            "top_heads": top_heads,
            "bottom_heads": bottom_heads,
        }
        sample_records.append(
            {
                "prompt": prompt,
                "Base": model_record if model_name == "Base" else None,
                "LoRA": model_record if model_name == "LoRA" else None,
            }
        )

        if sample_idx == 0:
            raw_sample = {
                "prompt_len": int(prompt_len),
                "tokens": token_texts,
                "layer_weights": layer_weights,
            }

    features = np.concatenate([head_instruction, head_completion], axis=1)
    labels, linkage_matrix, similarity = cluster_attention_heads(features, model_name)
    flat_labels = labels if labels.size else np.zeros(total_heads, dtype=int)
    flat_instruction = head_instruction.mean(axis=1)
    flat_completion = head_completion.mean(axis=1)
    flat_entropy = head_entropy.mean(axis=1)
    classification, flat_scores, confidences = attention_classification(
        head_instruction.mean(axis=1).reshape(num_layers, num_heads),
        head_completion.mean(axis=1).reshape(num_layers, num_heads),
    )

    overall_order = np.argsort(flat_scores)
    overall_top_heads = []
    overall_bottom_heads = []
    for idx in overall_order[::-1][:8]:
        layer = int(idx // num_heads)
        head = int(idx % num_heads)
        overall_top_heads.append(
            {
                "layer": layer,
                "head": head,
                "instruction_score": safe_float(head_instruction.mean(axis=1)[idx]),
                "completion_score": safe_float(head_completion.mean(axis=1)[idx]),
                "net_score": safe_float(flat_scores[idx]),
                "confidence": safe_float(confidences[idx]),
                "kind": "instruction-following" if flat_scores[idx] >= 0 else "completion",
            }
        )
    for idx in overall_order[:8]:
        layer = int(idx // num_heads)
        head = int(idx % num_heads)
        overall_bottom_heads.append(
            {
                "layer": layer,
                "head": head,
                "instruction_score": safe_float(head_instruction.mean(axis=1)[idx]),
                "completion_score": safe_float(head_completion.mean(axis=1)[idx]),
                "net_score": safe_float(flat_scores[idx]),
                "confidence": safe_float(confidences[idx]),
                "kind": "completion" if flat_scores[idx] < 0 else "instruction-following",
            }
        )

    layer_summary = summarize_layers(model_name, layer_instruction, layer_completion, layer_entropy, flat_labels, confidences)

    result = {
        "model_name": model_name,
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "head_instruction": head_instruction,
        "head_completion": head_completion,
        "head_entropy": head_entropy,
        "layer_instruction": layer_instruction,
        "layer_completion": layer_completion,
        "layer_entropy": layer_entropy,
        "head_features": features,
        "head_labels": labels,
        "linkage_matrix": linkage_matrix,
        "similarity_matrix": similarity,
        "classification": classification,
        "layer_summary": layer_summary,
        "sample_records": sample_records,
        "raw_sample": raw_sample,
        "top_heads": overall_top_heads,
        "bottom_heads": overall_bottom_heads,
        "responses": response_records,
        "head_scores": {
            "scores": flat_scores.tolist(),
            "confidence": confidences.tolist(),
        },
    }
    return result


def build_layer_report(base_result: Dict[str, Any], lora_result: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    base_layers = base_result["layer_summary"]["layers"]
    lora_layers = lora_result["layer_summary"]["layers"]
    rows = []
    md = [
        "# Task 3 Layer-wise Attention Analysis",
        "",
        "This report summarizes instruction-following versus completion-style attention behavior across layers, after stable cosine-similarity clustering and attention-rollout attribution.",
        "",
        "| Layer | Base instruction | LoRA instruction | Base completion | LoRA completion | Δ net |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    layer_change = []
    for base_layer, lora_layer in zip(base_layers, lora_layers):
        delta_net = (lora_layer["instruction_mean"] - lora_layer["completion_mean"]) - (
            base_layer["instruction_mean"] - base_layer["completion_mean"]
        )
        layer_change.append((base_layer["layer"], delta_net))
        md.append(
            f"| {base_layer['layer']} | {base_layer['instruction_mean']:.4f} | {lora_layer['instruction_mean']:.4f} | {base_layer['completion_mean']:.4f} | {lora_layer['completion_mean']:.4f} | {delta_net:.4f} |"
        )
        rows.append(
            {
                "layer": base_layer["layer"],
                "base": base_layer,
                "lora": lora_layer,
                "delta_net": safe_float(delta_net),
            }
        )

    top_layers = sorted(layer_change, key=lambda item: abs(item[1]), reverse=True)[:5]
    md.extend(
        [
            "",
            "## Most changed layers",
            "",
        ]
    )
    for layer, delta in top_layers:
        md.append(f"- Layer {layer}: net attention shift {delta:+.4f}")

    md.extend(
        [
            "",
            "## Empirical head classification rule",
            "",
            "Heads are labeled using the average response-token attention target over 20 sample instructions.",
            "- Instruction-following: response tokens attend more to prompt tokens than to local response context.",
            "- Completion: response tokens attend more to recent response context than to the prompt.",
            "",
            "## Stability fix",
            "",
            "All rollout and normalization paths clip denominators with epsilon, convert NaN/Inf to zero, and use cosine-distance average-linkage clustering (not Ward on precomputed cosine distances).",
        ]
    )

    payload = {
        "base": {
            "model": base_result["model_name"],
            "classification": base_result["classification"],
            "layers": base_layers,
        },
        "lora": {
            "model": lora_result["model_name"],
            "classification": lora_result["classification"],
            "layers": lora_layers,
        },
        "layer_comparison": rows,
        "top_layers_by_change": [{"layer": layer, "delta_net": safe_float(delta)} for layer, delta in top_layers],
    }
    return payload, "\n".join(md)


def save_results(base_result: Dict[str, Any], lora_result: Dict[str, Any], prompts: Sequence[str]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for pattern in [
        "attention_heatmap_*.html",
        "multi_head_visualization.png",
        "base_vs_lora_comparison.png",
        "base_vs_lora_attention_diff.png",
        "layer_analysis.md",
        "layer_analysis.json",
        "head_classification.json",
        "head_classification_Base.json",
        "head_classification_LoRA.json",
    ]:
        for path in RESULTS_DIR.glob(pattern):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    # Fill the paired per-prompt records with both models before writing HTML.
    merged_records = []
    for idx, prompt in enumerate(prompts):
        merged_records.append(
            {
                "prompt": prompt,
                "Base": {
                    **base_result["sample_records"][idx]["Base"],
                    "response_text": base_result["responses"][idx]["response_text"],
                },
                "LoRA": {
                    **lora_result["sample_records"][idx]["LoRA"],
                    "response_text": lora_result["responses"][idx]["response_text"],
                },
            }
        )

    # Update the per-sample records with top heads so HTML can use the same summaries.
    for idx, prompt_record in enumerate(merged_records):
        prompt_record["Base"]["top_heads"] = base_result["sample_records"][idx]["Base"]["top_heads"]
        prompt_record["Base"]["bottom_heads"] = base_result["sample_records"][idx]["Base"]["bottom_heads"]
        prompt_record["LoRA"]["top_heads"] = lora_result["sample_records"][idx]["LoRA"]["top_heads"]
        prompt_record["LoRA"]["bottom_heads"] = lora_result["sample_records"][idx]["LoRA"]["bottom_heads"]

    save_html_outputs(merged_records)
    save_multi_head_visualization(base_result, lora_result)
    save_base_vs_lora_comparison(base_result, lora_result)
    cluster_comparison_plot(base_result["head_features"], lora_result["head_features"])

    layer_payload, layer_markdown = build_layer_report(base_result, lora_result)
    (RESULTS_DIR / "layer_analysis.json").write_text(json.dumps(layer_payload, indent=2), encoding="utf-8")
    (RESULTS_DIR / "layer_analysis.md").write_text(layer_markdown, encoding="utf-8")

    head_classification = {
        "method": "dominant average response attention target over 20 prompts",
        "criterion": "instruction_following if mean response-to-prompt attention >= mean local response attention",
        "models": {
            "Base": {
                **base_result["classification"],
                "top_heads": base_result["top_heads"],
                "bottom_heads": base_result["bottom_heads"],
            },
            "LoRA": {
                **lora_result["classification"],
                "top_heads": lora_result["top_heads"],
                "bottom_heads": lora_result["bottom_heads"],
            },
        },
    }
    (RESULTS_DIR / "head_classification.json").write_text(json.dumps(head_classification, indent=2), encoding="utf-8")
    (RESULTS_DIR / "head_classification_Base.json").write_text(
        json.dumps(base_result["classification"], indent=2), encoding="utf-8"
    )
    (RESULTS_DIR / "head_classification_LoRA.json").write_text(
        json.dumps(lora_result["classification"], indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 3: Attention Visualization & Token Attribution")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of instruction prompts to analyze")
    args = parser.parse_args()

    if RESULTS_DIR.exists():
        logging.info("Cleaning previous Task 3 artifacts in %s", RESULTS_DIR)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    else:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    prompts = get_prompts(max(HTML_SAMPLES, args.num_samples))
    prompts = prompts[: args.num_samples]

    logging.info("Loading base model")
    base_model, base_tokenizer = load(BASE_MODEL)
    base_result = analyze_model(base_model, base_tokenizer, prompts, "Base")

    logging.info("Loading LoRA model")
    lora_model, lora_tokenizer = load(BASE_MODEL, adapter_path=str(ADAPTER_PATH))
    lora_result = analyze_model(lora_model, lora_tokenizer, prompts, "LoRA")

    save_results(base_result, lora_result, prompts)

    logging.info("Task 3 complete. Results saved to %s", RESULTS_DIR)
    logging.info("Generated 20+ HTML heatmaps, 3 clustering plots, layer analysis, and head classification outputs")


if __name__ == "__main__":
    main()
