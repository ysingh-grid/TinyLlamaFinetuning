# Test-set Perplexity Report

- **Data file:** `/Users/ysingh/PyCharmMiscProject/data/test.jsonl`
- **Metric:** token-averaged cross-entropy loss over full chat sequence

| Model | Cross-Entropy (nats) | Perplexity | Tokens |
|-------|----------------------|-----------:|-------:|
| `full_ft` | 1.4011 | 4.060 | 122760 |
| `lora_ft` | 1.4273 | 4.168 | 122760 |
| `qlora_ft` | 1.4502 | 4.264 | 122760 |
| `base` | 1.4777 | 4.383 | 122760 |
| `qwen_1.8b` | 2.7972 | 16.398 | 105568 |
| `phi_2` | 1.4423 | 4.231 | 102489 |

> Note: Perplexities are computed over the entire chat sequence (user + assistant), using the ChatML-style template for tokenisation.