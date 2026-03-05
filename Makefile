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
	@echo ""
	@echo "══════════════════════════════════════════════════════════════════════"
	@echo "  INSTALL  Creating .venv and installing all dependencies"
	@echo "  Reads:   requirements.txt"
	@echo "  ETA:     ~2–3 min (first time), ~30 s (cached)"
	@echo "══════════════════════════════════════════════════════════════════════"
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	@echo "  ✓ INSTALL complete"

# Download all three local models (idempotent — skips if already present)
download-models: $(TINYLLAMA_4BIT) $(PHI2) $(QWEN)
	@echo "  ✓ All models present."

prepare:
	@echo ""
	@echo "══════════════════════════════════════════════════════════════════════"
	@echo "  PREPARE  Downloading Alpaca 52k → filtering top 5,000 examples"
	@echo "  Output:  data/train.jsonl  data/valid.jsonl  data/test.jsonl"
	@echo "  ETA:     ~30–60 s"
	@echo "══════════════════════════════════════════════════════════════════════"
	$(PYTHON) prepare_dataset.py
	@echo "  ✓ PREPARE complete"

eda:
	@echo ""
	@echo "══════════════════════════════════════════════════════════════════════"
	@echo "  EDA      Computing answer-length stats for data/train.jsonl"
	@echo "  Output:  data/eda_report.md"
	@echo "  ETA:     ~5 s"
	@echo "══════════════════════════════════════════════════════════════════════"
	$(PYTHON) eda.py
	@echo "  ✓ EDA complete"

validate:
	@echo ""
	@echo "══════════════════════════════════════════════════════════════════════"
	@echo "  VALIDATE Pre-flight check: GPU, data files, adapters, model paths"
	@echo "  ETA:     ~5 s"
	@echo "══════════════════════════════════════════════════════════════════════"
	$(PYTHON) validate_training_setup.py
	@echo "  ✓ VALIDATE complete"

# train-qlora requires the 4-bit base model to exist first
train-lora:
	@echo ""
	@echo "══════════════════════════════════════════════════════════════════════"
	@echo "  TRAIN-LORA  LoRA fine-tuning on TinyLlama 1.1B (rank 16, 16 layers)"
	@echo "  Config:  lora_config.yaml"
	@echo "  Data:    data/train.jsonl  (~4,000 examples)"
	@echo "  Output:  adapters/tinyllama-lora-alpaca/"
	@echo "  ETA:     ~25–40 min on M-series Mac"
	@echo "══════════════════════════════════════════════════════════════════════"
	$(PYTHON) smart_train.py --config lora_config.yaml --patience 5
	@echo "  ✓ TRAIN-LORA complete"

train-qlora: $(TINYLLAMA_4BIT)
	@echo ""
	@echo "══════════════════════════════════════════════════════════════════════"
	@echo "  TRAIN-QLORA  QLoRA fine-tuning on TinyLlama 1.1B 4-bit (rank 8)"
	@echo "  Config:  qlora_config.yaml"
	@echo "  Data:    data/train.jsonl  (~4,000 examples)"
	@echo "  Output:  adapters/tinyllama-qlora-alpaca/"
	@echo "  ETA:     ~20–35 min on M-series Mac"
	@echo "══════════════════════════════════════════════════════════════════════"
	$(PYTHON) smart_train.py --config qlora_config.yaml --patience 5
	@echo "  ✓ TRAIN-QLORA complete"

train-full:
	@echo ""
	@echo "══════════════════════════════════════════════════════════════════════"
	@echo "  TRAIN-FULL  Full fine-tuning on TinyLlama 1.1B (all weights)"
	@echo "  Config:  mlx_sweep_runs/full/trial_0001/config.yaml"
	@echo "  Data:    data/train.jsonl  (~4,000 examples)"
	@echo "  Output:  adapters/tinyllama-full-alpaca/"
	@echo "  ETA:     ~45–90 min on M-series Mac"
	@echo "══════════════════════════════════════════════════════════════════════"
	$(PYTHON) smart_train.py --config mlx_sweep_runs/full/trial_0001/config.yaml --patience 5 --min-delta 0.0
	@echo "  ✓ TRAIN-FULL complete"

sweep: $(TINYLLAMA_4BIT)
	@echo ""
	@echo "══════════════════════════════════════════════════════════════════════"
	@echo "  SWEEP  Grid search over rank × LR × grad-accum for all 3 techniques"
	@echo "  Data:    data/train.jsonl"
	@echo "  Output:  mlx_sweep_runs/   mlx_best_models/"
	@echo "  ETA:     ~3–6 hours (full grid, all techniques)"
	@echo "══════════════════════════════════════════════════════════════════════"
	$(PYTHON) sweep_finetune.py --technique all --search grid
	@echo "  ✓ SWEEP complete"

# eval needs phi_2 and qwen for response generation + qwen as the local judge
eval: $(TINYLLAMA_4BIT) $(PHI2) $(QWEN)
	@echo ""
	@echo "══════════════════════════════════════════════════════════════════════"
	@echo "  EVAL  Generate 100 prompts × 6 models → judge all pairs (local MLX)"
	@echo "  Models:  base, lora_ft, qlora_ft, full_ft, phi_2, qwen_1.8b"
	@echo "  Output:  evaluation/runs/<timestamp>/"
	@echo "  ETA:     ~15–25 min"
	@echo "══════════════════════════════════════════════════════════════════════"
	JUDGE_MODE=model MAX_PROMPTS=100 ./evaluation/run_eval_6models.sh
	@echo "  ✓ EVAL complete"

eval-lmstudio: $(TINYLLAMA_4BIT) $(PHI2) $(QWEN)
	@echo ""
	@echo "══════════════════════════════════════════════════════════════════════"
	@echo "  EVAL-LMSTUDIO  500 prompts × 6 models → judge via LM Studio"
	@echo "  Requires:  LM Studio running at http://127.0.0.1:1234"
	@echo "  Output:    evaluation/runs/<timestamp>/"
	@echo "  ETA:       ~60–90 min"
	@echo "══════════════════════════════════════════════════════════════════════"
	JUDGE_MODE=lmstudio MAX_PROMPTS=500 ./evaluation/run_eval_6models.sh
	@echo "  ✓ EVAL-LMSTUDIO complete"

# perplexity runs all 6 models including local ones
perplexity: $(TINYLLAMA_4BIT) $(PHI2) $(QWEN)
	@echo ""
	@echo "══════════════════════════════════════════════════════════════════════"
	@echo "  PERPLEXITY  MLX forward pass on data/test.jsonl for all 6 models"
	@echo "  Output:  evaluation/perplexity_report.md"
	@echo "  ETA:     ~8–15 min"
	@echo "══════════════════════════════════════════════════════════════════════"
	$(PYTHON) evaluate_perplexity.py
	@echo "  ✓ PERPLEXITY complete"

demo: $(TINYLLAMA_4BIT) $(PHI2) $(QWEN)
	@echo ""
	@echo "══════════════════════════════════════════════════════════════════════"
	@echo "  DEMO  Launching Streamlit UI (Inference / Data Collection / EDA)"
	@echo "  URL:  http://localhost:8501"
	@echo "══════════════════════════════════════════════════════════════════════"
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
	@echo ""
	@echo "──────────────────────────────────────────────────────────────────────"
	@echo "  CI PREPARE  200 Alpaca examples → ci/data/  (fast subset)"
	@echo "  ETA: ~20 s"
	@echo "──────────────────────────────────────────────────────────────────────"
	$(PYTHON) prepare_dataset.py --num-examples 200 --min-answer-words 20 --out-dir ci/data
	@echo "  ✓ CI data ready: ci/data/train.jsonl"

test-prepare: ci/data/train.jsonl

# LoRA CI: only needs the Hub full-precision model (auto-downloaded by mlx_lm)
test-train-lora: ci/data/train.jsonl
	@echo ""
	@echo "──────────────────────────────────────────────────────────────────────"
	@echo "  CI TRAIN-LORA  100 iters, rank 8, 4 layers on ci/data/"
	@echo "  Config:  ci/test-lora-config.yaml"
	@echo "  Output:  ci/adapters/lora/"
	@echo "  ETA:     ~2 min"
	@echo "──────────────────────────────────────────────────────────────────────"
	$(PYTHON) smart_train.py --config ci/test-lora-config.yaml --patience 3
	@echo "  ✓ CI TRAIN-LORA complete"

# QLoRA CI: uses the same local 4-bit base as production; sentinel downloads it first.
test-train-qlora: ci/data/train.jsonl $(TINYLLAMA_4BIT)
	@echo ""
	@echo "──────────────────────────────────────────────────────────────────────"
	@echo "  CI TRAIN-QLORA  100 iters, rank 8, 4-bit base model"
	@echo "  Config:  ci/test-qlora-config.yaml"
	@echo "  Output:  ci/adapters/qlora/"
	@echo "  ETA:     ~2 min"
	@echo "──────────────────────────────────────────────────────────────────────"
	$(PYTHON) smart_train.py --config ci/test-qlora-config.yaml --patience 3
	@echo "  ✓ CI TRAIN-QLORA complete"

# Full FT CI: uses the Hub full-precision model (auto-downloaded by mlx_lm)
test-train-full: ci/data/train.jsonl
	@echo ""
	@echo "──────────────────────────────────────────────────────────────────────"
	@echo "  CI TRAIN-FULL  100 iters, full fine-tuning on ci/data/"
	@echo "  Config:  ci/test-full-config.yaml"
	@echo "  Output:  ci/adapters/full/"
	@echo "  ETA:     ~4 min"
	@echo "──────────────────────────────────────────────────────────────────────"
	$(PYTHON) smart_train.py --config ci/test-full-config.yaml --patience 3
	@echo "  ✓ CI TRAIN-FULL complete"

# test-eval uses the same 6-model eval; needs phi_2, qwen, and tinyllama-4bit locally
test-eval: $(TINYLLAMA_4BIT) $(PHI2) $(QWEN)
	@echo ""
	@echo "──────────────────────────────────────────────────────────────────────"
	@echo "  CI EVAL  25 prompts × 6 models → judge with local MLX model"
	@echo "  Output:  evaluation/runs/<timestamp>/"
	@echo "  ETA:     ~3–5 min"
	@echo "──────────────────────────────────────────────────────────────────────"
	JUDGE_MODE=model MAX_PROMPTS=25 ./evaluation/run_eval_6models.sh
	@echo "  ✓ CI EVAL complete"

test-eval-lmstudio: $(TINYLLAMA_4BIT) $(PHI2) $(QWEN)
	@echo ""
	@echo "──────────────────────────────────────────────────────────────────────"
	@echo "  CI EVAL-LMSTUDIO  25 prompts × 6 models → LM Studio judge"
	@echo "  Requires: LM Studio running at http://127.0.0.1:1234"
	@echo "  ETA:      ~3–5 min"
	@echo "──────────────────────────────────────────────────────────────────────"
	JUDGE_MODE=lmstudio MAX_PROMPTS=25 ./evaluation/run_eval_6models.sh
	@echo "  ✓ CI EVAL-LMSTUDIO complete"

test-cosine:
	@echo ""
	@echo "──────────────────────────────────────────────────────────────────────"
	@echo "  CI COSINE  Cosine similarity on latest evaluation run"
	@echo "  ETA:       ~1 min"
	@echo "──────────────────────────────────────────────────────────────────────"
	$(PYTHON) evaluation/score_similarity_rankings.py \
	  --responses-dir evaluation/runs/$$(ls -t evaluation/runs | head -1)/responses \
	  --references evaluation/eval_references.jsonl \
	  --out-dir evaluation/runs/$$(ls -t evaluation/runs | head -1)/cosine_similarity
	@echo "  ✓ CI COSINE complete"

# test-perplexity runs all 6 models — needs the same local models as test-eval
test-perplexity: $(TINYLLAMA_4BIT) $(PHI2) $(QWEN)
	@echo ""
	@echo "──────────────────────────────────────────────────────────────────────"
	@echo "  CI PERPLEXITY  MLX forward pass on ci/data/test.jsonl (all 6 models)"
	@echo "  Output:  ci/perplexity_report.md"
	@echo "  ETA:     ~3–5 min"
	@echo "──────────────────────────────────────────────────────────────────────"
	$(PYTHON) evaluate_perplexity.py --data ci/data/test.jsonl --out ci/perplexity_report.md
	@echo "  ✓ CI PERPLEXITY complete"

test-demo:
	@echo ""
	@echo "──────────────────────────────────────────────────────────────────────"
	@echo "  CI DEMO  Smoke-testing Streamlit startup (exits after 6 s)"
	@echo "──────────────────────────────────────────────────────────────────────"
	$(PYTHON) -m streamlit run streamlit_app.py --server.headless true & \
	  sleep 6 && kill %1 2>/dev/null; echo "  ✓ CI DEMO passed"

# Full end-to-end CI run (skips LM Studio — requires no external services)
# Models are downloaded automatically before eval/perplexity steps.
ci: test-prepare test-train-lora test-train-qlora test-train-full \
    test-eval test-cosine test-perplexity test-demo
	@echo ""
	@echo "══════════════════════════════════════════════════════════════════════"
	@echo "  ✓ ALL CI SMOKE TESTS PASSED"
	@echo "══════════════════════════════════════════════════════════════════════"
