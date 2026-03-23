# Task 3 Layer-wise Attention Analysis

This report summarizes instruction-following versus completion-style attention behavior across layers, after stable cosine-similarity clustering and attention-rollout attribution.

| Layer | Base instruction | LoRA instruction | Base completion | LoRA completion | Δ net |
|---|---:|---:|---:|---:|---:|
| 0 | 0.0084 | 0.0099 | 0.0122 | 0.0141 | -0.0005 |
| 1 | 0.0081 | 0.0095 | 0.0266 | 0.0295 | -0.0016 |
| 2 | 0.0089 | 0.0101 | 0.0239 | 0.0272 | -0.0021 |
| 3 | 0.0220 | 0.0226 | 0.0074 | 0.0079 | -0.0000 |
| 4 | 0.0237 | 0.0241 | 0.0061 | 0.0065 | 0.0000 |
| 5 | 0.0218 | 0.0224 | 0.0078 | 0.0085 | -0.0001 |
| 6 | 0.0205 | 0.0211 | 0.0111 | 0.0122 | -0.0004 |
| 7 | 0.0184 | 0.0189 | 0.0087 | 0.0099 | -0.0007 |
| 8 | 0.0221 | 0.0226 | 0.0082 | 0.0091 | -0.0004 |
| 9 | 0.0180 | 0.0181 | 0.0145 | 0.0170 | -0.0024 |
| 10 | 0.0170 | 0.0174 | 0.0137 | 0.0155 | -0.0015 |
| 11 | 0.0182 | 0.0188 | 0.0112 | 0.0125 | -0.0008 |
| 12 | 0.0134 | 0.0138 | 0.0141 | 0.0170 | -0.0025 |
| 13 | 0.0137 | 0.0141 | 0.0155 | 0.0184 | -0.0024 |
| 14 | 0.0138 | 0.0144 | 0.0167 | 0.0193 | -0.0019 |
| 15 | 0.0168 | 0.0170 | 0.0103 | 0.0123 | -0.0018 |
| 16 | 0.0151 | 0.0152 | 0.0121 | 0.0152 | -0.0030 |
| 17 | 0.0172 | 0.0173 | 0.0104 | 0.0126 | -0.0021 |
| 18 | 0.0181 | 0.0180 | 0.0088 | 0.0103 | -0.0015 |
| 19 | 0.0185 | 0.0188 | 0.0089 | 0.0102 | -0.0011 |
| 20 | 0.0171 | 0.0171 | 0.0107 | 0.0127 | -0.0020 |
| 21 | 0.0152 | 0.0153 | 0.0103 | 0.0120 | -0.0016 |

## Most changed layers

- Layer 16: net attention shift -0.0030
- Layer 12: net attention shift -0.0025
- Layer 13: net attention shift -0.0024
- Layer 9: net attention shift -0.0024
- Layer 2: net attention shift -0.0021

## Empirical head classification rule

Heads are labeled using the average response-token attention target over 20 sample instructions.
- Instruction-following: response tokens attend more to prompt tokens than to local response context.
- Completion: response tokens attend more to recent response context than to the prompt.

## Stability fix

All rollout and normalization paths clip denominators with epsilon, convert NaN/Inf to zero, and use cosine-distance average-linkage clustering (not Ward on precomputed cosine distances).