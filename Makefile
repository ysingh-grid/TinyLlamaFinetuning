PYTHON := .venv/bin/python

.PHONY: install prepare eda validate train-lora train-qlora train-full sweep eval eval-lmstudio perplexity demo

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

