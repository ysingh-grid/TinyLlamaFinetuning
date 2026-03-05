PYTHON := .venv/bin/python

# ── Production targets ────────────────────────────────────────────────────────
.PHONY: install prepare eda validate train-lora train-qlora train-full sweep \
        eval eval-lmstudio perplexity demo

install:
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

prepare:
	$(PYTHON) prepare_dataset.py

eda:
	$(PYTHON) eda.py

validate:
	$(PYTHON) validate_training_setup.py

train-lora:
	$(PYTHON) smart_train.py --config lora_config.yaml --patience 5

train-qlora:
	$(PYTHON) smart_train.py --config qlora_config.yaml --patience 5

train-full:
	$(PYTHON) smart_train.py --config mlx_sweep_runs/full/trial_0001/config.yaml --patience 5 --min-delta 0.0

sweep:
	$(PYTHON) sweep_finetune.py --technique all --search grid

eval:
	JUDGE_MODE=model MAX_PROMPTS=100 ./evaluation/run_eval_6models.sh

eval-lmstudio:
	JUDGE_MODE=lmstudio MAX_PROMPTS=500 ./evaluation/run_eval_6models.sh

perplexity:
	$(PYTHON) evaluate_perplexity.py

demo:
	$(PYTHON) -m streamlit run streamlit_app.py


# ── CI / smoke-test targets (fast — for verifying a fresh clone) ──────────────
# Uses ci/ subdirectory for all artifacts; never touches production data or adapters.
# Full smoke run: make ci
#
# Individual steps:
#   make test-prepare       ~30 s  — 200 examples from Alpaca
#   make test-train-lora    ~2 min — 100 iters, rank 8, 4 layers
#   make test-train-qlora   ~2 min — 100 iters, rank 8, 4 layers (4-bit base)
#   make test-train-full    ~4 min — 100 iters, full FT
#   make test-eval          ~2 min — 25 prompts, local MLX judge
#   make test-eval-lmstudio ~2 min — 25 prompts, LM Studio judge (needs LM Studio running)
#   make test-cosine        ~1 min — cosine similarity on latest eval run
#   make test-perplexity    ~3 min — perplexity on ci/data/test.jsonl (all 6 models)
#   make test-demo          ~5 s   — Streamlit smoke start (prints URL then exits)

.PHONY: test-prepare test-train-lora test-train-qlora test-train-full \
        test-eval test-eval-lmstudio test-cosine test-perplexity test-demo ci

# sentinel file: created by test-prepare, depended on by all test-train-* targets
ci/data/train.jsonl:
	$(PYTHON) prepare_dataset.py --num-examples 200 --min-answer-words 20 --out-dir ci/data

test-prepare: ci/data/train.jsonl

test-train-lora: ci/data/train.jsonl
	$(PYTHON) smart_train.py --config ci/test-lora-config.yaml --patience 3

test-train-qlora: ci/data/train.jsonl
	$(PYTHON) smart_train.py --config ci/test-qlora-config.yaml --patience 3

test-train-full: ci/data/train.jsonl
	$(PYTHON) smart_train.py --config ci/test-full-config.yaml --patience 3

test-eval:
	JUDGE_MODE=model MAX_PROMPTS=25 ./evaluation/run_eval_6models.sh

test-eval-lmstudio:
	JUDGE_MODE=lmstudio MAX_PROMPTS=25 ./evaluation/run_eval_6models.sh

test-cosine:
	$(PYTHON) evaluation/score_similarity_rankings.py \
	  --responses-dir evaluation/runs/$$(ls -t evaluation/runs | head -1)/responses \
	  --references evaluation/eval_references.jsonl \
	  --out-dir evaluation/runs/$$(ls -t evaluation/runs | head -1)/cosine_similarity

test-perplexity:
	$(PYTHON) evaluate_perplexity.py --data ci/data/test.jsonl --out ci/perplexity_report.md

test-demo:
	@echo "Smoke-testing Streamlit startup (will exit after 6 s)..."
	$(PYTHON) -m streamlit run streamlit_app.py --server.headless true & \
	  sleep 6 && kill %1 2>/dev/null; echo "Streamlit smoke test passed."

# Full end-to-end CI run (skips LM Studio — requires no external services)
ci: test-prepare test-train-lora test-train-qlora test-train-full \
    test-eval test-cosine test-perplexity test-demo
	@echo "All CI smoke tests passed."

