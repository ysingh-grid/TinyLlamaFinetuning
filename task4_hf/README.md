---
title: TinyLlama Quantization & GGUF (Task 4)
emoji: 📦
colorFrom: gray
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# TinyLlama — Quantization & GGUF (Task 4)

Interactive **benchmark dashboard** for TinyLlama LoRA fusion → **GGUF** export and **llama.cpp** quantization (**Q4\_K\_M**, **Q5\_K\_M**, **Q8\_0**).

## What this Space shows

- **Try a quant** — in-browser GGUF chat powered by **`llama-cpp-python`** (pre-built CPU wheel, no compilation needed). First generation downloads the GGUF weights (~640 MB – 1.1 GB) and loads them on CPU — allow ~1 minute.
- Deployment comparison (size, throughput, memory, perplexity, batch TPS)
- Plots: throughput vs quantization, size vs perplexity, batched inference
- Instructions to run `llama-server` locally (`inference_server/`)

## GGUF weights on the Hub

Quantized **`.gguf`** files live in the model repo **[ysingh-aiml/tinyllama-alpaca-lora-gguf](https://huggingface.co/ysingh-aiml/tinyllama-alpaca-lora-gguf)**:

- [model-Q4_K_M.gguf](https://huggingface.co/ysingh-aiml/tinyllama-alpaca-lora-gguf/resolve/main/model-Q4_K_M.gguf)
- [model-Q5_K_M.gguf](https://huggingface.co/ysingh-aiml/tinyllama-alpaca-lora-gguf/resolve/main/model-Q5_K_M.gguf)
- [model-Q8_0.gguf](https://huggingface.co/ysingh-aiml/tinyllama-alpaca-lora-gguf/resolve/main/model-Q8_0.gguf)

You can also export locally with `task4_quantization_gguf.py` or copy from `results/task4/`.

## Repository layout

```
task4_hf/
├── app.py
├── gguf_chat.py
├── requirements.txt          # includes llama-cpp-python via pre-built CPU wheel
├── requirements-play.txt     # convenience alias (-r requirements.txt)
├── runtime.txt
├── results/
│   ├── deployment_comparison.json
│   ├── deployment_comparison.csv
│   ├── task4_summary.json
│   └── perplexity_results.json
├── plots/
│   ├── throughput_vs_quant.png
│   ├── size_vs_perplexity.png
│   └── batch_throughput.png
├── inference_server/
│   ├── server.py
│   ├── README.md
│   └── requirements.txt
└── models/
    ├── README.md
    └── *.gguf      # optional; inference_server/server.py downloads from the Hub if missing
```

## Hardware note

Benchmarks were collected on **Apple Silicon (M-series)** with local **llama.cpp** binaries. Numbers on other CPUs/GPUs will differ.

## Local: run "Try a quant" chat

```bash
cd task4_hf
pip install -r requirements.txt   # includes llama-cpp-python via pre-built CPU wheel
python app.py
```
