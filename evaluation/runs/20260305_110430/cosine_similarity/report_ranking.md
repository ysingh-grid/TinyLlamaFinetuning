# Similarity & Ranking Report (All vs All)

## Overall Metrics
Tested on 25 prompts.

| Model | Avg Sim | Avg Rank | Pairwise Matches | Win Rate (vs All) |
|---|---:|---:|---:|---:|
| phi_2 | 0.7848 | 2.08 | 125 | 0.776 |
| base | 0.6409 | 3.60 | 125 | 0.480 |
| qwen_1.8b | 0.6486 | 3.64 | 125 | 0.456 |
| lora_ft | 0.6440 | 3.72 | 125 | 0.456 |
| full_ft | 0.6197 | 3.92 | 125 | 0.416 |
| qlora_ft | 0.6170 | 4.04 | 125 | 0.392 |

## FT Models vs Base (Priority)
| FT Model | Base | FT Wins | Base Wins | Ties | FT Win % |
|---|---|---:|---:|---:|---:|
| full_ft | base | 13 | 12 | 0 | 0.520 |
| lora_ft | base | 10 | 15 | 0 | 0.400 |
| qlora_ft | base | 10 | 15 | 0 | 0.400 |

## Head-to-Head Pairwise Win Rates
| Model 1 | Model 2 | M1 Wins | M2 Wins | Ties | M1 Win % |
|---|---|---:|---:|---:|---:|
| full_ft | base | 13 | 12 | 0 | 0.520 |
| lora_ft | base | 10 | 15 | 0 | 0.400 |
| qlora_ft | base | 10 | 15 | 0 | 0.400 |
| phi_2 | full_ft | 19 | 6 | 0 | 0.760 |
| phi_2 | base | 18 | 6 | 1 | 0.720 |
| lora_ft | full_ft | 17 | 8 | 0 | 0.680 |
| qlora_ft | lora_ft | 14 | 11 | 0 | 0.560 |
| qlora_ft | full_ft | 13 | 12 | 0 | 0.520 |
| lora_ft | qwen_1.8b | 13 | 12 | 0 | 0.520 |
| qwen_1.8b | base | 13 | 12 | 0 | 0.520 |
| qwen_1.8b | full_ft | 12 | 13 | 0 | 0.480 |
| qlora_ft | qwen_1.8b | 9 | 16 | 0 | 0.360 |
| lora_ft | phi_2 | 6 | 19 | 0 | 0.240 |
| qwen_1.8b | phi_2 | 4 | 19 | 2 | 0.160 |
| qlora_ft | phi_2 | 3 | 22 | 0 | 0.120 |
