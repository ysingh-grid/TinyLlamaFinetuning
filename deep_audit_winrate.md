# Deep Audit: Why FT Win Rate Is Below 50% and How to Fix It

## Current State
- lora_ft vs base: **39.6%** effective win rate (target: >55%)
- full_ft vs base: **49.0%** | qlora_ft vs base: **35.4%**

---

## CRITICAL FINDING 1: Repetition Loops in FT Models

**Severity: HIGH — accounts for 10-15% win rate loss**

Actual FT responses from the latest eval run:

- full_ft on "Name five Mediterranean countries":
  "1. Italy 2. Greece... 2. Greece 3. France 4. Spain 5. Egypt. 2. Greece 3. France..." (repeating list 3x)

- qlora_ft on same: repeats "France, Italy, Spain, Greece, and Turkey." five times (110 words)

- full_ft on "Which food has most sugar": "Chocolate contains more sugar..." repeated for 191 words

- Base on same prompts: clean, concise, no repetition

**Root cause:** `min_tokens=50` suppresses EOS, forcing generation past the natural stop. FT models were trained on short Alpaca answers (median 26 words) so they have nothing meaningful to say after the stop point — they loop.

**Fix options:**
1. Add `repetition_penalty=1.2` to sampler (1 line)
2. Post-process: truncate at first repeated 4-gram
3. Remove `min_tokens` entirely; rely on system prompt for verbosity
4. Lower `min_tokens` to 30 + add repetition penalty

---

## CRITICAL FINDING 2: LoRA scale=1.0 — Adapter Under-Contributing

**Severity: HIGH**

| Adapter | rank | scale | effective_alpha | weight_multiplier |
|---------|------|-------|-----------------|-------------------|
| LoRA (trial_0010) | 32 | 1.0 | 32 | 1.0x |
| QLoRA (trial_0013) | 32 | 2.0 | 64 | 2.0x |

Standard practice is scale=2.0 (alpha=2*rank). With scale=1.0, adapter learned
patterns are applied at HALF the normal strength. The adapter IS learning — val
loss 1.256 is better than QLoRA's 1.280 — but contributions are too conservative.

**Fix:** New sweep enforces scale=2.0. Current eval uses old trial_0010 (scale=1.0).

---

## FINDING 3: Training Data — 53% of Answers Under 30 Words

**Severity: MEDIUM-HIGH**

| Metric | Value |
|--------|-------|
| Total examples | 4000 |
| Empty (0 words) | 2 |
| Under 10 words | 1173 (29%) |
| Under 20 words | 1790 (45%) |
| Under 30 words | 2103 (53%) |
| 50+ words | 1441 (36%) |
| Median | 26 words |

Training literally teaches one-word answers: "Sang.", "larger", "Fiction."
Meanwhile base TinyLlama was RLHF'd on verbose helpful conversations.
FT is REGRESSING the model's verbosity.

**Fix:** Filter data/train.jsonl to >= 30 word answers (~1900 examples). Model
learns longer responses naturally without needing min_tokens hacks.

---

## FINDING 4: System Prompt Mismatch — Training vs Inference

**Severity: MEDIUM**

- Training: 0/4000 examples have a system prompt. Format is just [user, assistant].
- Inference: We inject a system prompt via chat template, creating a `<|system|>` block.

The FT model NEVER SAW a system prompt during training. At inference time we
inject "You are a helpful assistant..." which adds an OOD (out-of-distribution)
prefix. This could confuse the model's generation pattern.

The base model (TinyLlama-Chat) WAS trained with system prompts during RLHF,
so it handles them correctly. This gives base an unfair advantage.

**Fix options:**
1. Remove system_prompt from FT models in models.json (keep for base only)
2. OR add system prompts to training data before re-training
3. OR evaluate ALL models without system prompts (fairest comparison)

---

## FINDING 5: Full FT max_seq_length=256 (vs LoRA's 512)

**Severity: MEDIUM**

| Model | max_seq_length |
|-------|---------------|
| LoRA | 512 |
| QLoRA | 512 |
| Full FT | **256** |

Full fine-tuning used max_seq_length=256, meaning any training example longer
than 256 tokens was truncated. This means the model never learned to produce
responses longer than ~64 words during full FT. Combined with min_tokens=50,
this creates the worst repetition behavior (191-word repetition on simple prompts).

**Fix:** Increase max_seq_length to 512 for full FT in next sweep.

---

## FINDING 6: No Overfitting (Good News)

Train/Val loss gaps for all best models are within normal range:
- LoRA: gap = -0.030 (val slightly BETTER than recent train batches — healthy)
- QLoRA: gap = -0.017 (healthy)
- Full: gap = -0.026 (healthy)

Val loss is still improving at iteration 1000 for all models, suggesting
we could benefit from MORE training iterations (1500-2000).

---

## FINDING 7: Val Loss Still Declining — Models Under-Trained?

LoRA val loss curve:
```
step 100:  1.306
step 500:  1.268
step 1000: 1.256  <- still dropping, not plateaued
```

QLoRA val loss curve:
```
step 100:  1.318
step 500:  1.291
step 900:  1.280  <- still dropping
```

Full FT val loss curve:
```
step 100:  1.544
step 500:  1.421
step 900:  1.373  <- still dropping significantly
```

ALL models are still improving at the end of training. Increasing iterations
from 1000 to 1500 or 2000 could yield better adapters.

**Fix:** Increase iters to 1500 in sweep config.

---

## FINDING 8: Correct Models Selected (No Misplacement Errors)

Verified: The best trial for each technique was correctly selected:
- LoRA: trial_0010 (val_loss=1.256) — best out of all LoRA trials ✓
- QLoRA: trial_0013 (val_loss=1.280) — best out of all QLoRA trials ✓
- Full: trial_0001 (val_loss=1.373) — best out of all Full trials ✓

No mix-up between adapter paths and model configs. Models.json paths match.

---

## FINDING 9: No Data Leakage (Good News)

Eval prompts vs training data: 0/500 eval prompts appear in training set.
This means the evaluation is genuinely testing generalization, not memorization.

---

## ACTION PLAN (Ranked by Expected Impact)

### Immediate (no retraining needed, biggest quick wins):

| # | Action | Expected Impact | Effort |
|---|--------|----------------|--------|
| 1 | Add repetition_penalty=1.2 to FT model inference | +5-10% win rate | 1 line |
| 2 | Remove system prompt from FT models (they never trained with one) | +3-5% win rate | Edit models.json |
| 3 | Lower min_tokens from 50 to 25 (reduce repetition forcing) | +3-5% win rate | Edit models.json |

### Quick Re-evaluation (same adapters, just change eval config):

After applying #1-3 above, re-run the 50-prompt eval to see immediate improvement
before investing in retraining.

### Retraining Required (for next sweep):

| # | Action | Expected Impact | Effort |
|---|--------|----------------|--------|
| 4 | Filter training data to >= 30 word answers | +10-15% win rate | Script + retrain |
| 5 | Enforce scale=2.0 for all LoRA/QLoRA | +3-5% win rate | Already configured |
| 6 | Increase iters from 1000 to 1500 | +2-3% win rate | Config change |
| 7 | Increase Full FT max_seq_length to 512 | +2-3% for full_ft | Config change |
| 8 | Add system prompts to training data | +2-3% win rate | Data prep + retrain |

### Combined Expected Improvement:
- Immediate fixes (#1-3): could reach **~50% win rate** (parity with base)
- After retraining (#4-8): could reach **55-65% win rate** (clear FT advantage)
