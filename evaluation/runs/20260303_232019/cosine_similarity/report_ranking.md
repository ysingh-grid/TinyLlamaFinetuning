# Similarity & Ranking Report (All vs All)

## Overall Metrics
Tested on 100 prompts.

| Model | Avg Sim | Avg Rank | Pairwise Matches | Win Rate (vs All) |
|---|---:|---:|---:|---:|
| phi_2 | 0.7893 | 2.01 | 500 | 0.792 |
| qwen_1.8b | 0.6650 | 3.42 | 500 | 0.502 |
| full_ft | 0.6240 | 3.70 | 500 | 0.460 |
| base | 0.6326 | 3.89 | 500 | 0.422 |
| qlora_ft | 0.6160 | 3.91 | 500 | 0.418 |
| lora_ft | 0.6235 | 4.07 | 500 | 0.386 |

## FT Models vs Base (Priority)
| FT Model | Base | FT Wins | Base Wins | Ties | FT Win % |
|---|---|---:|---:|---:|---:|
| full_ft | base | 56 | 44 | 0 | 0.560 |
| lora_ft | base | 46 | 54 | 0 | 0.460 |
| qlora_ft | base | 46 | 54 | 0 | 0.460 |

## Head-to-Head Pairwise Win Rates
| Model 1 | Model 2 | M1 Wins | M2 Wins | Ties | M1 Win % |
|---|---|---:|---:|---:|---:|
| full_ft | base | 56 | 44 | 0 | 0.560 |
| lora_ft | base | 46 | 54 | 0 | 0.460 |
| qlora_ft | base | 46 | 54 | 0 | 0.460 |
| phi_2 | base | 79 | 18 | 3 | 0.790 |
| phi_2 | full_ft | 78 | 22 | 0 | 0.780 |
| qwen_1.8b | base | 57 | 41 | 2 | 0.570 |
| qwen_1.8b | full_ft | 55 | 45 | 0 | 0.550 |
| qlora_ft | lora_ft | 52 | 48 | 0 | 0.520 |
| qlora_ft | full_ft | 48 | 52 | 0 | 0.480 |
| lora_ft | full_ft | 45 | 55 | 0 | 0.450 |
| qlora_ft | qwen_1.8b | 43 | 57 | 0 | 0.430 |
| lora_ft | qwen_1.8b | 37 | 63 | 0 | 0.370 |
| qlora_ft | phi_2 | 20 | 80 | 0 | 0.200 |
| qwen_1.8b | phi_2 | 19 | 76 | 5 | 0.190 |
| lora_ft | phi_2 | 17 | 83 | 0 | 0.170 |
