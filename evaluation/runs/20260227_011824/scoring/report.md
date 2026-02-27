# Evaluation Report

## Pairwise Metrics

| Pair | Model 1 | Model 2 | Win Rate (M1) | Effective Win Rate (M1) | Tie Rate | 95% CI (Tie-Adj Score) |
|---|---|---|---:|---:|---:|---:|
| full_ft__vs__base | full_ft | base | 0.400 | 0.426 | 0.060 | [0.290, 0.570] |
| full_ft__vs__phi_2 | full_ft | phi_2 | 0.300 | 0.333 | 0.100 | [0.230, 0.480] |
| full_ft__vs__qwen_1.8b | full_ft | qwen_1.8b | 0.380 | 0.388 | 0.020 | [0.260, 0.530] |
| lora_ft__vs__base | lora_ft | base | 0.440 | 0.550 | 0.200 | [0.410, 0.660] |
| lora_ft__vs__phi_2 | lora_ft | phi_2 | 0.380 | 0.442 | 0.140 | [0.320, 0.580] |
| lora_ft__vs__qwen_1.8b | lora_ft | qwen_1.8b | 0.440 | 0.478 | 0.080 | [0.340, 0.610] |
| qlora_ft__vs__base | qlora_ft | base | 0.320 | 0.364 | 0.120 | [0.250, 0.500] |
| qlora_ft__vs__phi_2 | qlora_ft | phi_2 | 0.360 | 0.419 | 0.140 | [0.300, 0.560] |
| qlora_ft__vs__qwen_1.8b | qlora_ft | qwen_1.8b | 0.320 | 0.356 | 0.100 | [0.240, 0.500] |

## Model Rollup

| Model | Wins | Losses | Ties | Matches | Win Rate | Effective Win Rate |
|---|---:|---:|---:|---:|---:|---:|
| phi_2 | 79 | 52 | 19 | 150 | 0.527 | 0.603 |
| qwen_1.8b | 83 | 57 | 10 | 150 | 0.553 | 0.593 |
| base | 73 | 58 | 19 | 150 | 0.487 | 0.557 |
| lora_ft | 63 | 66 | 21 | 150 | 0.420 | 0.488 |
| full_ft | 54 | 87 | 9 | 150 | 0.360 | 0.383 |
| qlora_ft | 50 | 82 | 18 | 150 | 0.333 | 0.379 |
