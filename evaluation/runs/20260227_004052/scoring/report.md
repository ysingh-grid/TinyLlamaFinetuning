# Evaluation Report

## Pairwise Metrics

| Pair | Model 1 | Model 2 | Win Rate (M1) | Effective Win Rate (M1) | Tie Rate | 95% CI (Tie-Adj Score) |
|---|---|---|---:|---:|---:|---:|
| full_ft__vs__base | full_ft | base | 0.240 | 0.400 | 0.400 | [0.280, 0.600] |
| full_ft__vs__phi_2 | full_ft | phi_2 | 0.080 | 0.182 | 0.560 | [0.260, 0.480] |
| full_ft__vs__qwen_1.8b | full_ft | qwen_1.8b | 0.160 | 0.211 | 0.240 | [0.140, 0.440] |
| lora_ft__vs__base | lora_ft | base | 0.280 | 0.467 | 0.400 | [0.340, 0.620] |
| lora_ft__vs__phi_2 | lora_ft | phi_2 | 0.000 | 0.000 | 0.680 | [0.240, 0.420] |
| lora_ft__vs__qwen_1.8b | lora_ft | qwen_1.8b | 0.120 | 0.167 | 0.280 | [0.140, 0.400] |
| qlora_ft__vs__base | qlora_ft | base | 0.200 | 0.333 | 0.400 | [0.260, 0.540] |
| qlora_ft__vs__phi_2 | qlora_ft | phi_2 | 0.000 | 0.000 | 0.600 | [0.200, 0.400] |
| qlora_ft__vs__qwen_1.8b | qlora_ft | qwen_1.8b | 0.200 | 0.294 | 0.320 | [0.200, 0.520] |

## Model Rollup

| Model | Wins | Losses | Ties | Matches | Win Rate | Effective Win Rate |
|---|---:|---:|---:|---:|---:|---:|
| phi_2 | 27 | 2 | 46 | 75 | 0.360 | 0.931 |
| qwen_1.8b | 42 | 12 | 21 | 75 | 0.560 | 0.778 |
| base | 27 | 18 | 30 | 75 | 0.360 | 0.600 |
| full_ft | 12 | 33 | 30 | 75 | 0.160 | 0.267 |
| lora_ft | 10 | 31 | 34 | 75 | 0.133 | 0.244 |
| qlora_ft | 10 | 32 | 33 | 75 | 0.133 | 0.238 |
