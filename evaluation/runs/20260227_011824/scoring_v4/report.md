# Evaluation Report

## Pairwise Metrics

| Pair | Model 1 | Model 2 | Win Rate (M1) | Effective Win Rate (M1) | Net Margin (M1) | Tie Rate | 95% CI |
|---|---|---|---:|---:|---:|---:|---:|
| full_ft__vs__base | full_ft | base | 0.480 | 0.490 | -2.0% | 0.020 | [0.350, 0.620] |
| full_ft__vs__phi_2 | full_ft | phi_2 | 0.320 | 0.320 | -36.0% | 0.000 | [0.200, 0.460] |
| full_ft__vs__qwen_1.8b | full_ft | qwen_1.8b | 0.340 | 0.347 | -30.0% | 0.020 | [0.220, 0.490] |
| lora_ft__vs__base | lora_ft | base | 0.380 | 0.396 | -20.0% | 0.040 | [0.270, 0.520] |
| lora_ft__vs__phi_2 | lora_ft | phi_2 | 0.380 | 0.380 | -24.0% | 0.000 | [0.260, 0.520] |
| lora_ft__vs__qwen_1.8b | lora_ft | qwen_1.8b | 0.560 | 0.560 | +12.0% | 0.000 | [0.420, 0.700] |
| qlora_ft__vs__base | qlora_ft | base | 0.340 | 0.354 | -28.0% | 0.040 | [0.230, 0.490] |
| qlora_ft__vs__phi_2 | qlora_ft | phi_2 | 0.400 | 0.408 | -18.0% | 0.020 | [0.280, 0.550] |
| qlora_ft__vs__qwen_1.8b | qlora_ft | qwen_1.8b | 0.380 | 0.380 | -24.0% | 0.000 | [0.240, 0.520] |

## Model Rollup

| Model | Wins | Losses | Ties | Matches | Win Rate | Effective Win Rate | Net Margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| phi_2 | 94 | 55 | 1 | 150 | 0.627 | 0.631 | +26.0% |
| base | 85 | 60 | 5 | 150 | 0.567 | 0.586 | +16.7% |
| qwen_1.8b | 85 | 64 | 1 | 150 | 0.567 | 0.570 | +14.0% |
| lora_ft | 66 | 82 | 2 | 150 | 0.440 | 0.446 | -10.7% |
| full_ft | 57 | 91 | 2 | 150 | 0.380 | 0.385 | -22.7% |
| qlora_ft | 56 | 91 | 3 | 150 | 0.373 | 0.381 | -23.3% |
