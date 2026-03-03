# Similarity & Ranking Report (All vs All)

## Overall Metrics
Tested on 25 prompts.

| Model | Avg Sim | Avg Rank | Pairwise Matches | Win Rate (vs All) |
|---|---:|---:|---:|---:|
| lora_ft | 0.7129 | 3.12 | 125 | 0.536 |
| qwen_1.8b | 0.6893 | 3.40 | 125 | 0.512 |
| base | 0.6768 | 3.48 | 125 | 0.496 |
| full_ft | 0.6996 | 3.48 | 125 | 0.504 |
| qlora_ft | 0.6815 | 3.64 | 125 | 0.424 |
| phi_2 | 0.6467 | 3.88 | 125 | 0.424 |

## Head-to-Head Pairwise Win Rates
| Model 1 | Model 2 | M1 Wins | M2 Wins | Ties | M1 Win % |
|---|---|---:|---:|---:|---:|
| qwen_1.8b | phi_2 | 16 | 8 | 1 | 0.640 |
| lora_ft | phi_2 | 15 | 10 | 0 | 0.600 |
| lora_ft | full_ft | 14 | 8 | 3 | 0.560 |
| qlora_ft | phi_2 | 13 | 12 | 0 | 0.520 |
| lora_ft | qwen_1.8b | 13 | 11 | 1 | 0.520 |
| qwen_1.8b | base | 13 | 12 | 0 | 0.520 |
| phi_2 | base | 13 | 12 | 0 | 0.520 |
| lora_ft | base | 12 | 12 | 1 | 0.480 |
| base | full_ft | 12 | 12 | 1 | 0.480 |
| qlora_ft | base | 11 | 14 | 0 | 0.440 |
| qlora_ft | qwen_1.8b | 10 | 14 | 1 | 0.400 |
| qlora_ft | full_ft | 10 | 13 | 2 | 0.400 |
| qwen_1.8b | full_ft | 10 | 15 | 0 | 0.400 |
| phi_2 | full_ft | 10 | 15 | 0 | 0.400 |
| qlora_ft | lora_ft | 9 | 13 | 3 | 0.360 |
