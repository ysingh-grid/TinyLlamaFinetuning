# Task 4 — GGUF inference server (llama.cpp)

Run **`llama-server`** against the quantized GGUFs from **[ysingh-aiml/tinyllama-alpaca-lora-gguf](https://huggingface.co/ysingh-aiml/tinyllama-alpaca-lora-gguf)**. If the file is not under `models/`, this script **downloads it from the Hub** (public repo; no token required).

## Prerequisites

- Python 3.10+ with deps: `pip install -r inference_server/requirements.txt`
- Built `llama-server` from [llama.cpp](https://github.com/ggerganov/llama.cpp), or a release on your `PATH`
- Optional: `export LLAMA_SERVER=/absolute/path/to/llama-server`

## Fetch and run (default quant: Q4_K_M)

From the `task4_hf` repo root:

```bash
cd task4_hf
pip install -r inference_server/requirements.txt
export LLAMA_SERVER=/path/to/llama-server   # or rely on PATH

python inference_server/server.py --port 8080 --n-gpu-layers 0
```

On first run, `models/model-Q4_K_M.gguf` is downloaded automatically, then `llama-server` starts.

## Choose quantization

```bash
python inference_server/server.py --quant q5_k_m --port 8080
python inference_server/server.py --quant q8_0 --port 8080
```

## Download only (no server)

```bash
python inference_server/server.py --fetch-only --quant q4_k_m
```

Prints the resolved path on success.

## Offline / local file only

Place a `.gguf` in `models/` or pass `--model /path/to/model.gguf` and use **`--no-fetch`** so nothing is pulled from the Hub.

## Override Hub repo

```bash
export TASK4_GGUF_REPO=your-org/your-gguf-repo
python inference_server/server.py
# or
python inference_server/server.py --hf-repo your-org/your-gguf-repo
```

## API

Then use the OpenAI-compatible HTTP API (e.g. `POST /v1/chat/completions`); see the llama.cpp server docs.
