# Colab Prompt — TinyLlama Fine-Tuning: All Critical Fixes

Copy this entire block as instructions for your Colab notebook or AI assistant.

---

## Context

Fine-tuning TinyLlama-1.1B-Chat-v1.0 with LoRA/QLoRA/Full FT on Alpaca dataset.
Evaluating against base model via pairwise LLM-as-judge.
Local pipeline uses MLX (Apple Silicon). Colab should use PyTorch + HuggingFace PEFT.

## 8 Critical Changes (From Deep Audit)

### 1. TRAINING DATA: Filter for Long Answers ONLY

Raw Alpaca has 53% answers under 30 words (including empty and 1-word answers).
This teaches the model to be TERSE → loses to base model which is already verbose.

```python
from datasets import load_dataset
import random, json

ds = load_dataset("tatsu-lab/alpaca", split="train")
# Remove empty, sort by answer word count descending, take top 5000
ds_filtered = [ex for ex in ds if len(ex["output"].strip()) > 0]
ds_sorted = sorted(ds_filtered, key=lambda x: len(x["output"].split()), reverse=True)
selected = ds_sorted[:5000]
# Result: min ~100 words, avg ~141 words, max 717 words per answer

# Format as chat messages (NO system prompt!)
def format_chat(ex):
    user = ex["instruction"]
    if ex.get("input", ""):
        user += f"\n\nInput:\n{ex['input']}"
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": ex["output"]}
    ]}

samples = [format_chat(ex) for ex in selected]
random.seed(42)
random.shuffle(samples)

# 80/10/10 split
n = len(samples)
train = samples[:int(0.8*n)]      # 4000
valid = samples[int(0.8*n):int(0.9*n)]  # 500
test  = samples[int(0.9*n):]      # 500
```

### 2. LoRA: ALWAYS use alpha = 2 * rank (scale=2.0)

scale=1.0 under-applies adapter patterns. This was a key bug.

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,                # sweep: [8, 16]
    lora_alpha=16,      # ALWAYS 2 * r
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
```

### 3. max_seq_length = 512 (NOT 256)

Long answers need room. 256 truncated training examples causing repetition.

```python
max_seq_length = 512
```

### 4. Early Stopping: patience=2, min_delta=0.005

```python
from transformers import EarlyStoppingCallback, TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    max_steps=1000,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # sweep: [4, 6]
    learning_rate=1e-4,             # sweep: [1e-4, 2e-4]
    warmup_steps=50,
    logging_steps=10,
    eval_steps=100,
    save_steps=100,
    eval_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,
    gradient_checkpointing=True,
)

early_stop = EarlyStoppingCallback(
    early_stopping_patience=2,
    early_stopping_threshold=0.005,
)
```

### 5. mask_prompt=True (Train Only on Assistant Response)

```python
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# TinyLlama chat template uses "<|assistant|>\n" before responses
response_template = "<|assistant|>\n"
data_collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    max_seq_length=512,
    callbacks=[early_stop],
)
```

### 6. Upgrade: Move from SFT to DPO (Direct Preference Optimization)

Standard SFT (Supervised Fine-Tuning) forces the model to mimic text completions. Because TinyLlama is already highly optimized with RLHF, mimicking the relatively basic Alpaca dataset often *regresses* its performance vs the base model. 

Instead, use **DPO** to teach the model **preferences** (what *not* to do vs what *to* do). DPO requires a dataset with `prompt`, `chosen`, and `rejected` fields (e.g., `Intel/orca_dpo_pairs` or your own ranked outputs).

```python
from trl import DPOTrainer
from datasets import load_dataset

# 1. Load a preference dataset (requires 'prompt', 'chosen', and 'rejected' columns)
dpo_dataset = load_dataset("Intel/orca_dpo_pairs", split="train[:5000]")

# 2. Convert standard Trainer to DPOTrainer
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None, # PEFT handles the reference model automatically
    args=training_args,
    train_dataset=dpo_dataset,
    eval_dataset=dpo_valid_dataset,
    tokenizer=tokenizer,
    beta=0.1,       # The temperature parameter for the DPO loss, typically 0.1 to 0.5
    max_length=512,
    max_prompt_length=256,
)

# 3. Train
dpo_trainer.train()
```

### 7. Inference: repetition_penalty=1.2 + min_new_tokens=50

Without these, FT models either:
- Stop too early (3.5x shorter than base) → lose on length
- Or loop endlessly when forced to continue → lose on quality

```python
outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    min_new_tokens=50,
    repetition_penalty=1.2,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
```

### 8. NO System Prompt at Inference

Training data has 0/4000 system prompts. Injecting one at eval is OOD.
Evaluate ALL models (FT and base) WITHOUT system prompt for fairness.

### 9. QLoRA Specifics

For QLoRA, quantize the base model to 4-bit first:

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    quantization_config=bnb_config,
    device_map="auto",
)
```

Then apply the same LoRA config as #2 above.

---

## Sweep Grid Summary

| Param | Values |
|-------|--------|
| Technique | LoRA, QLoRA, Full FT |
| Rank (LoRA/QLoRA) | 8, 16 |
| Alpha | always 2 * rank |
| Learning Rate | 1e-4, 2e-4 (LoRA/QLoRA), 1e-5, 2e-5 (Full) |
| Batch Size | 4 |
| Grad Accum | 4, 6 |
| max_seq_length | 512 |
| Epochs | 1 (= 1000 iters with 4000 train examples) |
| Early Stop | patience=2, delta=0.005 |
| Dropout | 0.05 |
| Target Modules | q_proj, k_proj, v_proj, o_proj |

## Evaluation Pipeline

1. Generate responses from all models (FT + base) on 50 held-out prompts
2. Create pairwise matchups (FT vs base, FT vs other models)
3. Use an LLM judge (e.g., GPT-4, Mistral, or local model) for pairwise comparison
4. Judge outputs: LEFT/RIGHT/TIE for each pair
5. Score: effective_win_rate = wins / (wins + losses) — target > 55%
6. Check judge bias: left/right split should be near 50/50

## Key Differences from MLX Pipeline

| MLX (Local) | PyTorch (Colab) |
|-------------|-----------------|
| mlx-lm-lora CLI | HuggingFace SFTTrainer |
| mlx_lm.stream_generate | model.generate() |
| logits_processors for min_tokens | min_new_tokens parameter |
| Custom repetition penalty processor | repetition_penalty parameter (built in) |
| make_sampler(temp, top_p) | temperature, top_p in generate() |
| 4-bit via mlx quantize | 4-bit via BitsAndBytesConfig |
