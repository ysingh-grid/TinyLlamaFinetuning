# Minimum layers for 95% instruction-following (proxy metrics)

**Ablation (MLX)** uses a **full forward** on the prompt and the **first assistant token**
top-1 at the last prompt position (same as `model(prompt_ids)`). The **95% rule** compares
ablation accuracy to the **full-adapter** baseline under that same definition.

**PyTorch logit-lens curves** in `results/*.png` use **mean top-1** over up to 32
teacher-forced assistant positions (more stable than a single token). Reference: deepest layer **0.632**.

- Baseline (full LoRA, MLX full forward, first assistant token): **0.570**
- 95% of that baseline: **0.541**
- Minimum number of kept (highest-importance) LoRA layers to reach threshold: **12**

## Layer ablation (keep top-importance LoRA layers, zero others)

| Kept layers | Accuracy |
|---:|---:|
| 1 | 0.440 |
| 2 | 0.440 |
| 4 | 0.450 |
| 6 | 0.480 |
| 8 | 0.540 |
| 10 | 0.490 |
| 12 | 0.570 |
| 16 | 0.570 |

## Selective retrain

- Strategy: **depth**
- Layers trained: `[]`
- Iterations: **100**
- Output: `/Users/ysingh/PyCharmMiscProject/task5_hf/results/retrained_adapter`
