# Theoretical Foundation: Quantization and Edge Deployment

## 1. The Memory Bandwidth Bottleneck
In the context of Large Language Models (LLMs), the primary bottleneck for inference speed (token generation rate) is rarely the raw computational power (FLOPs) of the GPU or CPU. Instead, inference is overwhelmingly bounded by **Memory Bandwidth**.

Generating a single token requires the entire multi-gigabyte weight tensor of the model to be shuttled from physical RAM / VRAM into the processor's active cache to execute the matrix-vector multiplication against the current context. For a 7 Billion parameter model stored in 16-bit precision (FP16 or BF16), this requires moving roughly 14 Gigabytes of data for *every single word generated*. If a hardware bus is limited to 100 GB/sec, the absolute maximum theoretical limit of generation is mathematically capped at $\approx 7$ tokens per second, regardless of the ALU clock speed.

## 2. The Theory of Weight Quantization
Quantization solves the memory bottleneck by aggressively compressing the precision of the model's weights. 

Standard weights are stored as 16-bit floating-point numbers. Quantization algorithms map these continuous, high-precision numbers into a smaller, discrete set of integers—typically 8-bit, 6-bit, or 4-bit representations.

### Symmetric and Asymmetric Quantization
- In **Linear Symmetric Quantization**, the continuous weight range $[W_{min}, W_{max}]$ is scaled by a factor $S$ and mapped symmetrically around zero. For an 8-bit integer ($[-127, 127]$):
  $W_{quantized} = \text{Round}\left(\frac{W_{float}}{S}\right)$
- **Grouped Quantization** (used in formats like GGUF/Q4_K) recognizes that weight distributions vary drastically across the matrix. Instead of using one scale $S$ for the whole matrix, the matrix is divided into small blocks (e.g., 32 or 64 parameters), and a unique scaling factor is calculated for each block, preserving significantly more localized variance at the cost of slight metadata overhead.

By crushing 16-bit weights down to 4-bit integers, the physical payload of the model shrinks by 75%. The memory bus can now shuttle the same parameter count 400% faster, drastically increasing the Tokens/Second generation rate. 

## 3. Quantization Error and Perplexity Degradation
The cost of this compression is **Quantization Error**—the mathematical rounding delta between the original float and the recovered integer.
$Error = || W_{float} - (W_{quantized} \times S) ||$

As this error propagates through millions of sequential matrix multiplications in the transformer layers, the model's internal representations inevitably drift. This drift manifests as degraded reasoning, hallucination, or loss of nuanced factual recall.

This degradation is formally measured via **Perplexity (PPL)**, which evaluates the model's Cross-Entropy loss on a held-out dataset. A higher perplexity indicates the model is mathematically more "confused" and less confident in predicting the correct sequence of language.

The engineering challenge of edge deployment is plotting the theoretical degradation curve against the physical latency limits. A 4-bit model might suffer a 10% increase in Perplexity, but yield a 300% increase in inference speed, transitioning the model from an unusable academic artifact into a viable real-time application.
