# Evaluation Analysis (Run 20260223_143739)

## Executive Summary

This run is a regression for all fine-tuned variants.

- `full_ft`, `lora_ft`, and `qlora_ft` lose against `base`, `qwen_1.8b`, and `phi_2` in all 9 tracked pairings.
- No fine-tuned model is close to the expected promotion threshold.

## What The Report Shows

From `report.md`:

- FT vs `base` effective win rates:
  - `full_ft`: `0.305`
  - `lora_ft`: `0.305`
  - `qlora_ft`: `0.316`
- FT vs `qwen_1.8b` effective win rates:
  - `full_ft`: `0.248`
  - `lora_ft`: `0.235`
  - `qlora_ft`: `0.246`
- FT vs `phi_2` effective win rates:
  - `full_ft`: `0.172`
  - `lora_ft`: `0.208`
  - `qlora_ft`: `0.191`

Model rollup (effective win rate):

- `phi_2`: `0.810`
- `qwen_1.8b`: `0.757`
- `base`: `0.691`
- `qlora_ft`: `0.255`
- `lora_ft`: `0.252`
- `full_ft`: `0.248`

## Expected vs Actual

Expected for a successful run:

- At least one fine-tuned model should consistently beat `base`.
- Target decision rule: effective win rate `> 0.55` with CI lower bound `> 0.50`.

Actual:

- Best fine-tuned result is `0.316` effective win rate vs `base` (well below `0.55`).
- All fine-tuned models underperform baselines by a large margin.

## Interpretation

- The issue is systematic, not a single-model miss.
- Similar poor performance across `full_ft`, `lora_ft`, `qlora_ft` suggests data/config/training quality problems rather than random variance.
- High tie rates (especially vs `phi_2`) indicate many close calls, but decisive outcomes still strongly favor baselines.

## Recommended Next Steps

1. Do not promote any model from this run.
2. Re-check training data pipeline and prompt formatting alignment with inference formatting.
3. Re-run safe full-FT sequence (`sanity -> smoke -> scale -> epoch1`) before another full eval.
4. Re-run LoRA/QLoRA with conservative LR and fewer risky config changes per run.
5. Keep eval settings fixed (same prompts, pairs, seed, generation params) for comparability.
