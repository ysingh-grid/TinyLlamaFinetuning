---
license: apache-2.0
language:
  - en
base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
tags:
  - gguf
  - llama-cpp
  - quantized
  - tinyllama
  - lora
  - alpaca
  - text-generation
datasets:
  - tatsu-lab/alpaca
---

# TinyLlama 1.1B — LoRA (Alpaca) — GGUF quantizations

GGUF weights for **TinyLlama-1.1B-Chat** fine-tuned with **LoRA** on Alpaca-style instructions (fused HF checkpoint → F16 GGUF → `llama-quantize`).

## Files

| File | Quantization | ~Size |
|------|----------------|-------|
| `model-Q4_K_M.gguf` | Q4_K_M | ~637 MB |
| `model-Q5_K_M.gguf` | Q5_K_M | ~746 MB |
| `model-Q8_0.gguf` | Q8_0 | ~1.1 GB |

## Usage (llama.cpp)

```bash
llama-cli -m model-Q4_K_M.gguf -p "Hello" -n 128
# or
llama-server -m model-Q4_K_M.gguf --host 0.0.0.0 --port 8080
```

## Provenance

- Base: [`TinyLlama/TinyLlama-1.1B-Chat-v1.0`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- Conversion: `llama.cpp/convert_hf_to_gguf.py` (F16), then `llama-quantize`
- Chat template is embedded in the GGUF (TinyLlama chat format)

## Related

- Benchmark Space: [`ysingh-aiml/tinyllama-quantization-gguf`](https://huggingface.co/spaces/ysingh-aiml/tinyllama-quantization-gguf)
