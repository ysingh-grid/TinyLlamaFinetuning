# Similarity & Ranking Report (All vs All)

## Overall Metrics
Tested on 500 prompts.

| Model | Avg Sim | Avg Rank | Pairwise Matches | Win Rate (vs All) |
|---|---:|---:|---:|---:|
| phi_2 | 0.7572 | 2.28 | 2500 | 0.743 |
| full_ft | 0.6244 | 3.48 | 2500 | 0.504 |
| lora_ft | 0.6162 | 3.63 | 2500 | 0.470 |
| qlora_ft | 0.6142 | 3.67 | 2500 | 0.458 |
| base | 0.6007 | 3.82 | 2500 | 0.435 |
| qwen_1.8b | 0.6111 | 4.12 | 2500 | 0.369 |

## FT Models vs Base (Priority)
| FT Model | Base | FT Wins | Base Wins | Ties | FT Win % |
|---|---|---:|---:|---:|---:|
| full_ft | base | 283 | 215 | 2 | 0.566 |
| lora_ft | base | 261 | 238 | 1 | 0.522 |
| qlora_ft | base | 261 | 236 | 3 | 0.522 |

## Head-to-Head Pairwise Win Rates
| Model 1 | Model 2 | M1 Wins | M2 Wins | Ties | M1 Win % |
|---|---|---:|---:|---:|---:|
| full_ft | base | 283 | 215 | 2 | 0.566 |
| lora_ft | base | 261 | 238 | 1 | 0.522 |
| qlora_ft | base | 261 | 236 | 3 | 0.522 |
| phi_2 | base | 378 | 119 | 3 | 0.756 |
| phi_2 | full_ft | 358 | 141 | 1 | 0.716 |
| lora_ft | qwen_1.8b | 298 | 202 | 0 | 0.596 |
| qlora_ft | qwen_1.8b | 286 | 214 | 0 | 0.572 |
| qlora_ft | lora_ft | 252 | 240 | 8 | 0.504 |
| lora_ft | full_ft | 241 | 249 | 10 | 0.482 |
| qlora_ft | full_ft | 219 | 274 | 7 | 0.438 |
| qwen_1.8b | base | 219 | 279 | 2 | 0.438 |
| qwen_1.8b | full_ft | 188 | 312 | 0 | 0.376 |
| lora_ft | phi_2 | 135 | 364 | 1 | 0.270 |
| qlora_ft | phi_2 | 128 | 371 | 1 | 0.256 |
| qwen_1.8b | phi_2 | 99 | 386 | 15 | 0.198 |
