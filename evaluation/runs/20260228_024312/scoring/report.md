# Evaluation Report

## Pairwise Metrics

| Pair | Model 1 | Model 2 | Win Rate (M1) | Effective Win Rate (M1) | Net Margin (M1) | Tie Rate | 95% CI |
|---|---|---|---:|---:|---:|---:|---:|
| full_ft__vs__base | full_ft | base | 0.420 | 0.438 | -12.0% | 0.040 | [0.290, 0.580] |
| full_ft__vs__phi_2 | full_ft | phi_2 | 0.400 | 0.408 | -18.0% | 0.020 | [0.280, 0.540] |
| full_ft__vs__qwen_1.8b | full_ft | qwen_1.8b | 0.400 | 0.408 | -18.0% | 0.020 | [0.280, 0.540] |
| lora_ft__vs__base | lora_ft | base | 0.380 | 0.388 | -22.0% | 0.020 | [0.250, 0.520] |
| lora_ft__vs__phi_2 | lora_ft | phi_2 | 0.300 | 0.306 | -38.0% | 0.020 | [0.180, 0.440] |
| lora_ft__vs__qwen_1.8b | lora_ft | qwen_1.8b | 0.380 | 0.413 | -16.0% | 0.080 | [0.290, 0.560] |
| qlora_ft__vs__base | qlora_ft | base | 0.420 | 0.438 | -12.0% | 0.040 | [0.300, 0.580] |
| qlora_ft__vs__phi_2 | qlora_ft | phi_2 | 0.400 | 0.417 | -16.0% | 0.040 | [0.290, 0.550] |
| qlora_ft__vs__qwen_1.8b | qlora_ft | qwen_1.8b | 0.340 | 0.347 | -30.0% | 0.020 | [0.230, 0.480] |

## Model Rollup

| Model | Wins | Losses | Ties | Matches | Win Rate | Effective Win Rate | Net Margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| phi_2 | 91 | 55 | 4 | 150 | 0.607 | 0.623 | +24.0% |
| qwen_1.8b | 88 | 56 | 6 | 150 | 0.587 | 0.611 | +21.3% |
| base | 84 | 61 | 5 | 150 | 0.560 | 0.579 | +15.3% |
| full_ft | 61 | 85 | 4 | 150 | 0.407 | 0.418 | -16.0% |
| qlora_ft | 58 | 87 | 5 | 150 | 0.387 | 0.400 | -19.3% |
| lora_ft | 53 | 91 | 6 | 150 | 0.353 | 0.368 | -25.3% |
