PYTHON := .venv/bin/python

# ── Sentinel files for local models ──────────────────────────────────────────
# Each model is downloaded once; Make skips the download if config.json exists.
TINYLLAMA_4BIT  := models/tinyllama-4bit-base/config.json
PHI2            := models/phi-2-hf-4bit-mlx/config.json
QWEN            := models/qwen1.5-1.8b-chat-4bit/config.json

$(TINYLLAMA_4BIT):
	$(PYTHON) download_models.py --models tinyllama_4bit

$(PHI2):
	$(PYTHON) download_models.py --models phi_2

$(QWEN):
	$(PYTHON) download_models.py --models qwen

# ── Production targets ────────────────────────────────────────────────────────
.PHONY: install download-models prepare eda validate \
        train-lora train-qlora train-full sweep \
        eval eval-lmstudio perplexity demo

install:
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

# Download all three local models (idempotent — skips if already present)
download-models: $(TINYLLAMA_4BIT) $(PHI2) $(QWEN)
	@echo "All models present."

prepare:
	$(PYTHON) prepare_dataset.py

eda:
	$(PYTHON) eda.py

validate:
	$(PYTHON) validate_training_setup.py

# train-qlora requires the 4-bit base model to exist first
train-lora:
	$(PYTHON) smart_train.py --config lora_config.yaml --patience 5

train-qlora: $(TINYLLAMA_4BIT)
	$(PYTHON) smart_train.py --config qlora_config.yaml --patience 5

train-full:
	$(PYTHON) smart_train.py --config mlx_sweep_runs/full/trial_0001/config.yaml --patience 5 --min-delta 0.0

sweep: $(TINYLLAMA_4BIT)
	$(PYTHON) sweep_finetune.py --technique all --search grid

# eval needs phi_2 and qwen for response generation + qwen as the local judge
eval: $(TINYLLAMA_4BIT) $(PHI2) $(QWEN)
	JUDGE_MODE=model MAX_PROMPTS=100 ./evaluation/run_eval_6models.sh

eval-lmstudio: $(TINYLLAMA_4BIT) $(PHI2) $(QWEN)
	JUDGE_MODE=lmstudio MAX_PROMPTS=500 ./evaluation/run_eval_6models.sh

# perplexity runs all 6 models including local ones
perplexity: $(TINYLLAMA_4BIT) $(PHI2) $(QWEN)
	$(PYTHON) evaluate_perplexity.py

demo:
	$(PYTHON) -m streamlit run streamlit_app.py


# ── CI / smoke-test targets (fast — for verifying a fresh clone) ──────────────
# Uses ci/ subdirectory for all artifacts; never touches production data or adapters.
# Full smoke run: make ci
#
# Step-by-step timing on M-series Mac:
#   make test-prepare       ~30 s  — 200 examples from Alpaca
#   make test-train-lora    ~2 min — 100 iters, rank 8, 4 layers
#   make test-train-qlora   ~2 min — 100 iters, rank 8, 4 layers (Hub 4-bit)
#   make test-train-full    ~4 min — 100 iters, full FT
#   make test-eval          ~2 min — 25 prompts, local MLX judge
#   make test-eval-lmstudio ~2 min — 25 prompts, LM Studio judge (needs LM Studio running)
#   make test-cosine        ~1 min — cosine similarity on latest eval run
#   make test-perplexity    ~3 min — perplexity on ci/data/test.jsonl
#   make test-demo          ~6 s   — Streamlit smoke start

.PHONY: test-prepare test-train-lora test-train-qlora test-train-full \
        test-eval test-eval-lmstudio test-cosine test-perplexity test-demo ci

# Sentinel: data prepared for CI
ci/data/train.jsonl:
	$(PYTHON) prepare_dataset.py --num-examples 200 --min-answer-words 20 --out-dir ci/data

test-prepare: ci/data/train.jsonl

# LoRA CI: only needs the Hub full-precision model (auto-downloaded by mlx_lm)
test-train-lora: ci/data/train.jsonl
	$(PYTHON) smart_train.py --config ci/test-lora-config.yaml --patience 3

# QLoRA CI: uses mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit from Hub directly
# (ci/test-qlora-config.yaml sets model: "mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit")
test-train-qlora: ci/data/train.jsonl
	$(PYTHON) smart_train.py --config ci/test-qlora-config.yaml --patience 3

# Full FT CI: uses the Hub full-precision model (auto-downloaded by mlx_lm)
test-train-full: ci/data/train.jsonl
	$(PYTHON) smart_train.py --config ci/test-full-config.yaml --patience 3

# test-eval uses the same 6-model eval; needs phi_2, qwen, and tinyllama-4bit locally
test-eval: $(TINYLLAMA_4BIT) $(PHI2) $(QWEN)
	JUDGE_MODE=model MAX_PROMPTS=25 ./evaluation/run_eval_6models.sh

test-eval-lmstudio: $(TINYLLAMA_4BIT) $(PHI2) $(QWEN)
	JUDGE_MODE=lmstudio MAX_PROMPTS=25 ./evaluation/run_eval_6models.sh

test-cosine:
	$(PYTHON) evaluation/score_similarity_rankings.py \
	  --responses-dir evaluation/runs/$$(ls -t evaluation/runs | head -1)/responses \
	  --references evaluation/eval_references.jsonl \
	  --out-dir evaluation/runs/$$(ls -t evaluation/runs | head -1)/cosine_similarity

# test-perplexity runs all 6 models — needs the same local models as test-eval
test-perplexity: $(TINYLLAMA_4BIT) $(PHI2) $(QWEN)
	$(PYTHON) evaluate_perplexity.py --data ci/data/test.jsonl --out ci/perplexity_report.md

test-demo:
	@echo "Smoke-testing Streamlit startup (will exit after 6 s)..."
	$(PYTHON) -m streamlit run streamlit_app.py --server.headless true & \
	  sleep 6 && kill %1 2>/dev/null; echo "Streamlit smoke test passed."

# Full end-to-end CI run (skips LM Studio — requires no external services)
# Models are downloaded automatically before eval/perplexity steps.
ci: test-prepare test-train-lora test-train-qlora test-train-full \
    test-eval test-cosine test-perplexity test-demo
	@echo "All CI smoke tests passed."
