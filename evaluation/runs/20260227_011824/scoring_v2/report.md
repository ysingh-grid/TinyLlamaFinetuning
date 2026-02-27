# Evaluation Report

## Pairwise Metrics

| Pair | Model 1 | Model 2 | Win Rate (M1) | Effective Win Rate (M1) | Tie Rate | 95% CI (Tie-Adj Score) |
|---|---|---|---:|---:|---:|---:|
| full_ft__vs__base | full_ft | base | 0.480 | 0.480 | 0.000 | [0.340, 0.620] |
| full_ft__vs__phi_2 | full_ft | phi_2 | 0.449 | 0.449 | 0.000 | [0.306, 0.592] |
| full_ft__vs__qwen_1.8b | full_ft | qwen_1.8b | 0.540 | 0.540 | 0.000 | [0.400, 0.680] |
| lora_ft__vs__base | lora_ft | base | 0.500 | 0.500 | 0.000 | [0.360, 0.620] |
| lora_ft__vs__phi_2 | lora_ft | phi_2 | 0.460 | 0.460 | 0.000 | [0.320, 0.600] |
| lora_ft__vs__qwen_1.8b | lora_ft | qwen_1.8b | 0.560 | 0.560 | 0.000 | [0.420, 0.700] |
| qlora_ft__vs__base | qlora_ft | base | 0.560 | 0.560 | 0.000 | [0.420, 0.700] |
| qlora_ft__vs__phi_2 | qlora_ft | phi_2 | 0.480 | 0.480 | 0.000 | [0.340, 0.620] |
| qlora_ft__vs__qwen_1.8b | qlora_ft | qwen_1.8b | 0.500 | 0.500 | 0.000 | [0.360, 0.640] |

## Model Rollup

| Model | Wins | Losses | Ties | Matches | Win Rate | Effective Win Rate |
|---|---:|---:|---:|---:|---:|---:|
| phi_2 | 80 | 69 | 0 | 149 | 0.537 | 0.537 |
| qlora_ft | 77 | 73 | 0 | 150 | 0.513 | 0.513 |
| lora_ft | 76 | 74 | 0 | 150 | 0.507 | 0.507 |
| full_ft | 73 | 76 | 0 | 149 | 0.490 | 0.490 |
| base | 73 | 77 | 0 | 150 | 0.487 | 0.487 |
| qwen_1.8b | 70 | 80 | 0 | 150 | 0.467 | 0.467 |
