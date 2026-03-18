
import sys
import json
import math
import mlx.core as mx
from mlx_lm import load

model, tokenizer = load('results/task4/model_q8')

total_nll = 0.0
total_tokens = 0

with open('data/valid.jsonl') as f:
    for i, line in enumerate(f):
        if i >= 20: break
        row = json.loads(line)
        msgs = row.get('messages', [])
        if hasattr(tokenizer, 'apply_chat_template'):
            token_ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
        else:
            token_ids = tokenizer.encode(msgs[0]['content'])
            
        if len(token_ids) < 2: continue
        
        input_ids = mx.array(token_ids[:-1])[None, :]
        target_ids = mx.array(token_ids[1:])
        logits = model(input_ids)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        idx = mx.arange(target_ids.shape[0])
        token_log_probs = log_probs[0, idx, target_ids]
        
        total_nll += float(-mx.sum(token_log_probs).item())
        total_tokens += target_ids.shape[0]

ppl = math.exp(total_nll / total_tokens) if total_tokens > 0 else 0
print(f"\n__PPL__={ppl}")
