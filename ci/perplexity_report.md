# Test-set Perplexity Report

- **Data file:** `ci/data/test.jsonl`
- **Metric:** token-averaged cross-entropy loss over full chat sequence

| Model | Cross-Entropy (nats) | Perplexity | Tokens |
|-------|----------------------|-----------:|-------:|
| `full_ft` | 1.3112 | 3.711 | 10752 |
| `lora_ft` | 1.3426 | 3.829 | 10752 |
| `qlora_ft` | 1.3594 | 3.894 | 10752 |
| `base` | 1.4023 | 4.065 | 10752 |
| `qwen_1.8b` | 2.2686 | 9.666 | 9277 |
| `phi_2` | 1.7270 | 5.624 | 9142 |

> Note: Perplexities are computed over the entire chat sequence (user + assistant), using the ChatML-style template for tokenisation.