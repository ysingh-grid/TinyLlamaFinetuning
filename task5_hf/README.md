---
title: Logit Lens & LoRA Layer Importance (Task 5)
emoji: 🔭
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 5.10.0
app_file: app.py
pinned: false
license: apache-2.0
language:
  - en
base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
tags:
  - interpretability
  - logit-lens
  - lora
  - tinyllama
  - layer-analysis
---

# Task 5 — Logit lens & layer-wise LoRA importance

This repository implements **per-layer logit lens** analysis (hidden states → final norm → `lm_head`), **LoRA L₂ importance** per transformer layer, **correlation** between importance and accuracy / accuracy gain (LoRA − base), **layer ablation** (keep only top-importance LoRA layers), and optional **selective LoRA retraining** (MLX-LM) on a chosen subset of layers.

## Metrics

| Metric | Definition |
|--------|------------|
| Logit-lens accuracy (PyTorch plots) | At each depth \(L\), hidden state → final RMSNorm → `lm_head`. Teacher-forced forward on `full_ids[:-1]`; **mean** top-1 over up to **32** assistant positions (stable probe). Deepest layer uses the model’s own logits for alignment with training. |
| Ablation / 95% rule (MLX) | **Full MLX forward** on the prompt; top-1 at the last prompt position vs the **first** assistant token. LoRA scales are zeroed on all but the **kept** (highest L₂) layers. Minimum kept count such that accuracy ≥ 95% of the **full-adapter** baseline under this probe. |
| LoRA importance | Sum of L2 norms of all LoRA tensors (`lora_a`, `lora_b`, …) in layer \(L\) in `adapters.safetensors`. |
| Correlation | Pearson / Spearman between per-layer L2 importance and PyTorch logit-lens accuracy (or LoRA − base gain), over **adapted** layers only. |

### Reading the plots

- **`prediction_accuracy_by_layer.png`** — Top: **Base vs merged LoRA** (same unembedding). Near-zero accuracy in early layers is **expected** (representations are not vocabulary-aligned yet). Green band = layers that actually have LoRA weights. Bottom: **zoom** on those layers so differences are visible.
- **`prediction_gain_by_layer.png`** — Δ(LoRA − base) **only on LoRA layers** (avoids a long run of identical zeros).
- **`lens_and_importance.png`** — Top: **normalized** L₂ mass (0–1) **and** LoRA accuracy on the same scale. Bottom: **raw** ‖LoRA‖₂ sums per layer.
- **Scatter plots** — Each point is **labeled with layer index** so correlation is not anonymous.

## Data

- `data/test.jsonl` — 100 Alpaca-style chat rows (`messages`: user + assistant), built from held-out prompts.
- `data/train.jsonl` / `data/valid.jsonl` — copies/splits for optional **retraining** only.

## Run analysis

```bash
pip install -r requirements-dev.txt

# Default: 100 prompts, adapter at ../adapters/tinyllama-lora-alpaca (MLX), retrain bottom-k layers by depth
python run_analysis.py --adapter /path/to/tinyllama-lora-alpaca

# Skip SFT re-run (logit lens + ablation only)
python run_analysis.py --adapter ../adapters/tinyllama-lora-alpaca --skip-retrain

# Retrain only the k layers with **smallest** L₂ mass (alternative hypothesis)
python run_analysis.py --retrain-strategy least_importance --retrain-k 5
```

Artifacts are written to `results/` (JSON + PNG + `minimum_layers_analysis.md`).

## Hugging Face Space

The Space serves **precomputed** `results/` (no GPU required at runtime). Re-run `run_analysis.py` locally after changing the adapter or prompts, then upload.

**Space:** [ysingh-aiml/task5-logit-lens-lora](https://huggingface.co/spaces/ysingh-aiml/task5-logit-lens-lora)

## References

- Logit lens: [nostalgebraist’s lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) (conceptual).
- LoRA: [Hu et al., 2021](https://arxiv.org/abs/2106.09685).
