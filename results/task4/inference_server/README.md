# Task 4 Inference Server

Serve the smallest working model with:

```bash
.venv/bin/python /Users/ysingh/PyCharmMiscProject/results/task4/inference_server/server.py --model /Users/ysingh/PyCharmMiscProject/results/task4/model-Q4_K_M.gguf
```

Benchmark the server with `requests` or curl against `POST /v1/chat/completions`.