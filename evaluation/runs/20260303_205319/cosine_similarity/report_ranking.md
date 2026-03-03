# Similarity & Ranking Report (All vs All)

## Overall Metrics
Tested on 100 prompts.

| Model | Avg Sim | Avg Rank | Pairwise Matches | Win Rate (vs All) |
|---|---:|---:|---:|---:|
| lora_ft | 0.6844 | 3.29 | 500 | 0.520 |
| qlora_ft | 0.6717 | 3.32 | 500 | 0.504 |
| full_ft | 0.6807 | 3.48 | 500 | 0.504 |
| qwen_1.8b | 0.6693 | 3.51 | 500 | 0.488 |
| phi_2 | 0.6706 | 3.66 | 500 | 0.464 |
| base | 0.6526 | 3.74 | 500 | 0.444 |

## FT Models vs Base (Priority)
| FT Model | Base | FT Wins | Base Wins | Ties | FT Win % |
|---|---|---:|---:|---:|---:|
| full_ft | base | 53 | 43 | 4 | 0.530 |
| lora_ft | base | 54 | 44 | 2 | 0.540 |
| qlora_ft | base | 54 | 44 | 2 | 0.540 |

## Head-to-Head Pairwise Win Rates
| Model 1 | Model 2 | M1 Wins | M2 Wins | Ties | M1 Win % |
|---|---|---:|---:|---:|---:|
| full_ft | base | 53 | 43 | 4 | 0.530 |
| lora_ft | base | 54 | 44 | 2 | 0.540 |
| qlora_ft | base | 54 | 44 | 2 | 0.540 |
| qlora_ft | phi_2 | 55 | 44 | 1 | 0.550 |
| lora_ft | qwen_1.8b | 55 | 44 | 1 | 0.550 |
| lora_ft | phi_2 | 55 | 45 | 0 | 0.550 |
| qwen_1.8b | base | 54 | 45 | 1 | 0.540 |
| phi_2 | base | 53 | 46 | 1 | 0.530 |
| qlora_ft | qwen_1.8b | 51 | 47 | 2 | 0.510 |
| qwen_1.8b | phi_2 | 50 | 47 | 3 | 0.500 |
| qwen_1.8b | full_ft | 49 | 50 | 1 | 0.490 |
| lora_ft | full_ft | 47 | 45 | 8 | 0.470 |
| qlora_ft | lora_ft | 46 | 49 | 5 | 0.460 |
| qlora_ft | full_ft | 46 | 48 | 6 | 0.460 |
| phi_2 | full_ft | 43 | 56 | 1 | 0.430 |
