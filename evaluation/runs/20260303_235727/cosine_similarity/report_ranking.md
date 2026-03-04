# Similarity & Ranking Report (All vs All)

## Overall Metrics
Tested on 100 prompts.

| Model | Avg Sim | Avg Rank | Pairwise Matches | Win Rate (vs All) |
|---|---:|---:|---:|---:|
| phi_2 | 0.7743 | 2.11 | 500 | 0.774 |
| full_ft | 0.6383 | 3.49 | 500 | 0.502 |
| lora_ft | 0.6413 | 3.62 | 500 | 0.474 |
| qlora_ft | 0.6248 | 3.79 | 500 | 0.442 |
| base | 0.6195 | 3.97 | 500 | 0.406 |
| qwen_1.8b | 0.6259 | 4.02 | 500 | 0.384 |

## FT Models vs Base (Priority)
| FT Model | Base | FT Wins | Base Wins | Ties | FT Win % |
|---|---|---:|---:|---:|---:|
| full_ft | base | 61 | 39 | 0 | 0.610 |
| lora_ft | base | 55 | 45 | 0 | 0.550 |
| qlora_ft | base | 55 | 45 | 0 | 0.550 |

## Head-to-Head Pairwise Win Rates
| Model 1 | Model 2 | M1 Wins | M2 Wins | Ties | M1 Win % |
|---|---|---:|---:|---:|---:|
| full_ft | base | 61 | 39 | 0 | 0.610 |
| lora_ft | base | 55 | 45 | 0 | 0.550 |
| qlora_ft | base | 55 | 45 | 0 | 0.550 |
| phi_2 | full_ft | 80 | 20 | 0 | 0.800 |
| phi_2 | base | 76 | 22 | 2 | 0.760 |
| lora_ft | qwen_1.8b | 58 | 42 | 0 | 0.580 |
| qlora_ft | lora_ft | 52 | 48 | 0 | 0.520 |
| qlora_ft | qwen_1.8b | 51 | 49 | 0 | 0.510 |
| lora_ft | full_ft | 50 | 49 | 1 | 0.500 |
| qwen_1.8b | base | 47 | 52 | 1 | 0.470 |
| qlora_ft | full_ft | 42 | 58 | 0 | 0.420 |
| qwen_1.8b | full_ft | 37 | 63 | 0 | 0.370 |
| lora_ft | phi_2 | 26 | 74 | 0 | 0.260 |
| qlora_ft | phi_2 | 21 | 79 | 0 | 0.210 |
| qwen_1.8b | phi_2 | 17 | 78 | 5 | 0.170 |
