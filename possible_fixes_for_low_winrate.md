# Possible Fixes for Low Fine-Tuned Model Win Rate

> Last updated: 2026-02-27
> Context: All three fine-tuned variants (lora_ft, qlora_ft, full_ft) consistently
> lose to the untuned base model in pairwise evaluation, with effective win rates
> of ~24–27% vs base's ~60–69%.

---

## Root Cause Summary

Two compounding problems were identified from data:

| # | Problem | Evidence |
|---|---|---|
| 1 | **FT models produce much shorter responses** | avg 18–41 words vs base's 72–105 words; 7–16% near-empty outputs |
| 2 | **Alpaca training data is inherently terse** | 45% of answers < 20 words; median answer = 26 words |
| 3 | **LM judge has length bias** | Longer responses are systematically preferred regardless of correctness |
| 4 | **Judge positional (right-side) bias** | Previous runs showed 63.7% right-side preference (now fixed: 51–54%) |
| 5 | **`scale=1.0` in some LoRA trials** | rank=32, alpha=32 → scale=1.0 instead of intended 2.0 (now fixed) |
| 6 | **Full FT repetition loops** | `5e-5` LR caused catastrophic token repetition (now fixed: `1e-5 / 2e-5`) |
| 7 | **1000 iters may not be enough** | Val loss still declining at end of training — never plateaued |

---

## Fix Options

### Option 1 — Data Filtering (Best ROI) ✅ Recommended
**Filter training data to keep only examples with answers ≥ 30 words.**

```python
# In prepare_dataset.py or as a pre-processing step
filtered = [ex for ex in examples if len(ex["messages"][1]["content"].split()) >= 30]
```

- **Effect**: Raises the learned output-length distribution without touching training code
- **Downside**: Reduces training set from ~4,000 to ~2,000 examples (58% kept)
- **When to apply**: Before the next sweep run
- **Effort**: Low — one-line filter in `prepare_dataset.py`

---

### Option 2 — Min-Tokens at Inference (No Retraining) ✅ Recommended
**Add `min_tokens=30` to generation for FT models in `pipeline.py`.**

```python
# In evaluation/pipeline.py → run_generation()
response = generate(
    model, tokenizer,
    prompt=formatted,
    min_tokens=30,    # ← add this
    max_tokens=max_tokens,
    ...
)
```

- **Effect**: Forces FT models to produce at least 30-token responses; eliminates near-empty outputs
- **Downside**: Can feel padded on genuinely short-answer prompts ("What is 2+2?")
- **Fairness note**: Apply the same `min_tokens` to all models for a fair comparison, or apply to none
- **When to apply**: Immediately — no retraining required
- **Effort**: Very low — one argument in `pipeline.py`

---

### Option 3 — System Prompt Engineering (No Retraining)
**Inject a verbosity-nudging system prompt via `models.json`.**

```json
{
  "name": "lora_ft",
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "adapter_path": "./mlx_best_models/lora",
  "system_prompt": "You are a helpful assistant. Always provide thorough, detailed explanations."
}
```

- **Effect**: Encourages longer generation without retraining
- **Downside**: Base model benefits equally from the same prompt; net advantage ~0
- **Fairness note**: Must apply the same system prompt to ALL models (including base) for a valid comparison
- **When to apply**: Immediately — edit `evaluation/models.json`
- **Effort**: Trivial

---

### Option 4 — Mix in Longer-Answer Data
**Supplement Alpaca with a dataset that has naturally longer, more detailed answers.**

Candidates:
- `databricks/databricks-dolly-15k` — longer, human-written answers
- `openhermes` — diverse, detailed instruction responses
- Filter Alpaca to the top 25% longest answers (> 97 words) and oversample them

```python
# Rough sketch: oversample long Alpaca answers
long_examples = [ex for ex in examples if len(ex["messages"][1]["content"].split()) > 80]
examples = examples + long_examples * 3  # 3× oversample long answers
```

- **Effect**: Model learns that longer answers are normal, not exceptional
- **Downside**: Changes the data distribution; may require re-validating eval prompts for leakage
- **When to apply**: Next major sweep
- **Effort**: Medium

---

### Option 5 — Increase Training Iterations
**Val loss was still declining at end of training — train for 2 epochs or 2000 iters.**

```yaml
# In sweep config
EPOCHS = [2]   # or iters: 2000
```

- **Effect**: More learning signal; model may develop richer generation patterns
- **Downside**: Doubles sweep runtime; risk of overfitting on small dataset
- **When to apply**: After Option 1 (data filter) — more data reduces overfit risk
- **Effort**: Low — change one constant in `sweep_finetune.py`

---

### Option 6 — Fix the Evaluation Metric (Fairness, Not Training)
**Accept that win rate vs. a verbose base model is a biased metric. Add reference-match scoring instead.**

Use ROUGE or token-overlap against the Alpaca reference answers in `eval_prompts.jsonl`:

```bash
.venv/bin/python evaluation/score_rouge.py \
  --responses evaluation/runs/<run_id>/responses/ \
  --references evaluation/eval_references.jsonl
```

- **Effect**: Measures factual correctness, not verbosity; levels the playing field
- **Downside**: Requires implementing `score_rouge.py`; ROUGE has its own limitations
- **When to apply**: Can run in parallel with existing pairwise eval
- **Effort**: Medium

---

## Priority Order

| Priority | Option | Cost | Retraining? |
|---|---|---|---|
| 🥇 1 | Data filtering (≥30 words) | Low | Yes |
| 🥇 1 | Min-tokens=30 at inference | Very Low | No |
| 🥈 2 | Increase to 2 epochs | Low | Yes |
| 🥉 3 | Mix in longer-answer data | Medium | Yes |
| 4 | System prompt (all models) | Trivial | No |
| 5 | ROUGE reference scoring | Medium | No |

---

## Already Fixed

| Issue | Fix Applied |
|---|---|
| `max_seq_length: 256` during training | ✅ Changed to 512 in current runs |
| `scale=1.0` in some LoRA trials | ✅ Alpha removed from grid; `scale=2.0` enforced |
| Full FT LR `5e-5` causing repetition | ✅ FULL_LRS changed to `[1e-5, 2e-5]` |
| Judge right-side positional bias | ✅ Confirmed neutral (51–54%) in latest runs |
| `TokenizersBackend` crash on QLoRA eval | ✅ Shim added to `pipeline.py` |
| LoRA rank=32 (overkill for 1.1B) | ✅ RANKS changed to `[8, 16]` |
