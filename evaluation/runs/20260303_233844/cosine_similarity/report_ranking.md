# Similarity & Ranking Report (All vs All)

## Overall Metrics
Tested on 100 prompts.

| Model | Avg Sim | Avg Rank | Pairwise Matches | Win Rate (vs All) |
|---|---:|---:|---:|---:|
| phi_2 | 0.7893 | 2.00 | 500 | 0.794 |
| qwen_1.8b | 0.6650 | 3.45 | 500 | 0.496 |
| qlora_ft | 0.6315 | 3.75 | 500 | 0.448 |
| lora_ft | 0.6437 | 3.83 | 500 | 0.434 |
| full_ft | 0.6323 | 3.95 | 500 | 0.410 |
| base | 0.6326 | 4.02 | 500 | 0.396 |

## FT Models vs Base (Priority)
| FT Model | Base | FT Wins | Base Wins | Ties | FT Win % |
|---|---|---:|---:|---:|---:|
| full_ft | base | 52 | 48 | 0 | 0.520 |
| lora_ft | base | 54 | 46 | 0 | 0.540 |
| qlora_ft | base | 55 | 45 | 0 | 0.550 |

## Head-to-Head Pairwise Win Rates
| Model 1 | Model 2 | M1 Wins | M2 Wins | Ties | M1 Win % |
|---|---|---:|---:|---:|---:|
| full_ft | base | 52 | 48 | 0 | 0.520 |
| lora_ft | base | 54 | 46 | 0 | 0.540 |
| qlora_ft | base | 55 | 45 | 0 | 0.550 |
| phi_2 | full_ft | 81 | 19 | 0 | 0.810 |
| phi_2 | base | 79 | 18 | 3 | 0.790 |
| qwen_1.8b | base | 57 | 41 | 2 | 0.570 |
| qwen_1.8b | full_ft | 57 | 43 | 0 | 0.570 |
| qlora_ft | full_ft | 55 | 44 | 1 | 0.550 |
| lora_ft | full_ft | 53 | 47 | 0 | 0.530 |
| qlora_ft | lora_ft | 50 | 50 | 0 | 0.500 |
| lora_ft | qwen_1.8b | 43 | 57 | 0 | 0.430 |
| qlora_ft | qwen_1.8b | 42 | 58 | 0 | 0.420 |
| qlora_ft | phi_2 | 22 | 78 | 0 | 0.220 |
| qwen_1.8b | phi_2 | 19 | 76 | 5 | 0.190 |
| lora_ft | phi_2 | 17 | 83 | 0 | 0.170 |
