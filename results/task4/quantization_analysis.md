# Task 4 Quantization & GGUF Export Analysis

## What was implemented
- Fused the LoRA adapter into the TinyLlama base model.
- Converted the fused model to GGUF with llama.cpp.
- Quantized the GGUF model to Q4_K_M, Q5_K_M, and Q8_0.
- Benchmarked 10 warm chat-completion generations per quantization.
- Measured peak RSS, latency, tokens/sec, output quality, and perplexity.
- Benchmarked batch sizes 1, 4, and 8 with llama-bench.
- Packaged a runnable llama.cpp server for the Q4_K_M model.

## Methodology
- Perplexity corpus: 80 Alpaca validation rows (14958 token estimate).
- Latency/quality runs used the first 10 validation rows via `POST /v1/chat/completions`.
- Memory footprint used per-request peak RSS sampled from the running llama-server process.
- Batched inference used `llama-bench` with batch sizes 1, 4, and 8, 10 repetitions, and 100 generated tokens.

## Perplexity summary
- Base GGUF perplexity: 4.2722
- Q4_K_M: 4.3399 (loss vs base: 0.0677)
- Q5_K_M: 4.2859 (loss vs base: 0.0137)
- Q8_0: 4.2725 (loss vs base: 0.0003)

## Latency and memory
- Q4_K_M: 214.529 tok/s, 0.4661s mean latency, 748.64 MB mean peak RSS, ROUGE-L 0.2011
- Q5_K_M: 166.378 tok/s, 0.6010s mean latency, 862.18 MB mean peak RSS, ROUGE-L 0.2005
- Q8_0: 157.148 tok/s, 0.6363s mean latency, 1224.17 MB mean peak RSS, ROUGE-L 0.2062

## Batched inference observations
- Q4_K_M: batch-1 197.074 tok/s, batch-4 198.918 tok/s, batch-8 193.096 tok/s
- Q5_K_M: batch-1 144.095 tok/s, batch-4 139.196 tok/s, batch-8 116.520 tok/s
- Q8_0: batch-1 95.942 tok/s, batch-4 80.914 tok/s, batch-8 80.567 tok/s

## Deployment recommendation
Q4_K_M is the best edge deployment choice because it is the smallest model, keeps the lowest peak RSS, and remains the fastest of the three quantizations in this run.

## Deployment comparison table
| Quantization | File Size (MB) | Tokens/sec | Mean Latency (s) | Peak RSS (MB) | Perplexity | Perplexity Loss | Batch 1 TPS | Batch 4 TPS | Batch 8 TPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q4_K_M | 636.88 | 214.529 | 0.4661 | 748.64 | 4.3399 | 0.0677 | 197.074 | 198.918 | 193.096 |
| Q5_K_M | 745.82 | 166.378 | 0.6010 | 862.18 | 4.2859 | 0.0137 | 144.095 | 139.196 | 116.520 |
| Q8_0 | 1115.62 | 157.148 | 0.6363 | 1224.17 | 4.2725 | 0.0003 | 95.942 | 80.914 | 80.567 |


## Packaging
The runnable server package is in `/Users/ysingh/PyCharmMiscProject/results/task4/inference_server` and serves `model-Q4_K_M.gguf` with llama-server.
