# Copilot Instructions: TinyLlama Instruction Tuning (MLX)

Fine-tunes TinyLlama 1.1B using LoRA, QLoRA, and Full Fine-Tuning on Apple Silicon via MLX.

## Build and Test Commands

```bash
make install           # Install dependencies
make prepare           # Prepare Alpaca dataset
make train-lora        # LoRA fine-tune (~25-40 min)
make eval              # Full evaluation
make ci                # Run all smoke tests (~15 min)

# Advanced tasks (interpretability & optimization)
make task1             # LoRA rank ablation (~2-4 hrs)
make task2             # Activation steering (~30 min)
make task3             # Attention visualization (~20 min)
make task4             # GGUF quantization (~45 min)
make task5             # Logit lens analysis (~30 min)
make tasks-all         # Run all 5 tasks
```

## Architecture

- **smart_train.py**: Training wrapper with early stopping
- **sweep_finetune.py**: Hyperparameter grid/TPE search
- **evaluation/**: Pairwise judging pipeline (6 models)
- **task{1-5}.py**: Advanced ML tasks for analysis

## Key Conventions

- Always use `.venv/bin/python` (not system python)
- Config files: alpha = rank * 2 for LoRA (constant scaling)
- Production uses `data/`, `adapters/`, CI uses `ci/` subdirectory
- Task outputs go to `results/task{1-5}/`
- Three evaluation modes: manual, lmstudio, model (via JUDGE_MODE env var)

See README.md for full documentation.
