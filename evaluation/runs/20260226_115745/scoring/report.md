# Evaluation Report

## Pairwise Metrics

| Pair | Model 1 | Model 2 | Win Rate (M1) | Effective Win Rate (M1) | Tie Rate | 95% CI (Tie-Adj Score) |
|---|---|---|---:|---:|---:|---:|
| full_ft__vs__base | full_ft | base | 0.200 | 0.333 | 0.400 | [0.200, 0.650] |
| full_ft__vs__phi_2 | full_ft | phi_2 | 0.000 | 0.000 | 0.600 | [0.150, 0.450] |
| full_ft__vs__qwen_1.8b | full_ft | qwen_1.8b | 0.000 | 0.000 | 0.300 | [0.000, 0.300] |
| lora_ft__vs__base | lora_ft | base | 0.200 | 0.333 | 0.400 | [0.200, 0.600] |
| lora_ft__vs__phi_2 | lora_ft | phi_2 | 0.000 | 0.000 | 0.600 | [0.150, 0.450] |
| lora_ft__vs__qwen_1.8b | lora_ft | qwen_1.8b | 0.000 | 0.000 | 0.400 | [0.050, 0.350] |
| qlora_ft__vs__base | qlora_ft | base | 0.200 | 0.400 | 0.500 | [0.250, 0.650] |
| qlora_ft__vs__phi_2 | qlora_ft | phi_2 | 0.000 | 0.000 | 0.700 | [0.200, 0.500] |
| qlora_ft__vs__qwen_1.8b | qlora_ft | qwen_1.8b | 0.000 | 0.000 | 0.400 | [0.050, 0.350] |

## Model Rollup

| Model | Wins | Losses | Ties | Matches | Win Rate | Effective Win Rate |
|---|---:|---:|---:|---:|---:|---:|
| phi_2 | 11 | 0 | 19 | 30 | 0.367 | 1.000 |
| qwen_1.8b | 19 | 0 | 11 | 30 | 0.633 | 1.000 |
| base | 11 | 6 | 13 | 30 | 0.367 | 0.647 |
| qlora_ft | 2 | 12 | 16 | 30 | 0.067 | 0.143 |
| lora_ft | 2 | 14 | 14 | 30 | 0.067 | 0.125 |
| full_ft | 2 | 15 | 13 | 30 | 0.067 | 0.118 |
