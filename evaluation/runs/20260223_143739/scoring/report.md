# Evaluation Report

## Pairwise Metrics

| Pair | Model 1 | Model 2 | Win Rate (M1) | Effective Win Rate (M1) | Tie Rate | 95% CI |
|---|---|---|---:|---:|---:|---:|
| full_ft__vs__base | full_ft | base | 0.230 | 0.305 | 0.246 | [0.318, 0.386] |
| full_ft__vs__phi_2 | full_ft | phi_2 | 0.100 | 0.172 | 0.420 | [0.282, 0.339] |
| full_ft__vs__qwen_1.8b | full_ft | qwen_1.8b | 0.204 | 0.248 | 0.176 | [0.258, 0.327] |
| lora_ft__vs__base | lora_ft | base | 0.228 | 0.305 | 0.252 | [0.319, 0.389] |
| lora_ft__vs__phi_2 | lora_ft | phi_2 | 0.118 | 0.208 | 0.432 | [0.304, 0.362] |
| lora_ft__vs__qwen_1.8b | lora_ft | qwen_1.8b | 0.184 | 0.235 | 0.216 | [0.259, 0.327] |
| qlora_ft__vs__base | qlora_ft | base | 0.238 | 0.316 | 0.248 | [0.327, 0.399] |
| qlora_ft__vs__phi_2 | qlora_ft | phi_2 | 0.114 | 0.191 | 0.404 | [0.285, 0.346] |
| qlora_ft__vs__qwen_1.8b | qlora_ft | qwen_1.8b | 0.196 | 0.246 | 0.202 | [0.260, 0.333] |

## Model Rollup

| Model | Wins | Losses | Ties | Matches | Win Rate | Effective Win Rate |
|---|---:|---:|---:|---:|---:|---:|
| phi_2 | 706 | 166 | 628 | 1500 | 0.471 | 0.810 |
| qwen_1.8b | 910 | 292 | 297 | 1499 | 0.607 | 0.757 |
| base | 779 | 348 | 373 | 1500 | 0.519 | 0.691 |
| qlora_ft | 274 | 799 | 427 | 1500 | 0.183 | 0.255 |
| lora_ft | 265 | 785 | 450 | 1500 | 0.177 | 0.252 |
| full_ft | 267 | 811 | 421 | 1499 | 0.178 | 0.248 |
