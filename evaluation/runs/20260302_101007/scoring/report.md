# Evaluation Report

## Pairwise Metrics

| Pair | Model 1 | Model 2 | Win Rate (M1) | Effective Win Rate (M1) | Net Margin (M1) | Tie Rate | 95% CI |
|---|---|---|---:|---:|---:|---:|---:|
| full_ft__vs__base | full_ft | base | 0.400 | 0.417 | -16.0% | 0.040 | [0.220, 0.600] |
| full_ft__vs__phi_2 | full_ft | phi_2 | 0.280 | 0.304 | -36.0% | 0.080 | [0.160, 0.500] |
| full_ft__vs__qwen_1.8b | full_ft | qwen_1.8b | 0.360 | 0.391 | -20.0% | 0.080 | [0.220, 0.580] |
| lora_ft__vs__base | lora_ft | base | 0.440 | 0.478 | -4.0% | 0.080 | [0.280, 0.680] |
| lora_ft__vs__phi_2 | lora_ft | phi_2 | 0.200 | 0.208 | -56.0% | 0.040 | [0.080, 0.380] |
| lora_ft__vs__qwen_1.8b | lora_ft | qwen_1.8b | 0.320 | 0.320 | -36.0% | 0.000 | [0.160, 0.520] |
| qlora_ft__vs__base | qlora_ft | base | 0.600 | 0.600 | +20.0% | 0.000 | [0.400, 0.800] |
| qlora_ft__vs__phi_2 | qlora_ft | phi_2 | 0.400 | 0.400 | -20.0% | 0.000 | [0.200, 0.600] |
| qlora_ft__vs__qwen_1.8b | qlora_ft | qwen_1.8b | 0.360 | 0.391 | -20.0% | 0.080 | [0.220, 0.580] |

## Model Rollup

| Model | Wins | Losses | Ties | Matches | Win Rate | Effective Win Rate | Net Margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| phi_2 | 50 | 22 | 3 | 75 | 0.667 | 0.694 | +37.3% |
| qwen_1.8b | 45 | 26 | 4 | 75 | 0.600 | 0.634 | +25.3% |
| base | 36 | 36 | 3 | 75 | 0.480 | 0.500 | +0.0% |
| qlora_ft | 34 | 39 | 2 | 75 | 0.453 | 0.466 | -6.7% |
| full_ft | 26 | 44 | 5 | 75 | 0.347 | 0.371 | -24.0% |
| lora_ft | 24 | 48 | 3 | 75 | 0.320 | 0.333 | -32.0% |
