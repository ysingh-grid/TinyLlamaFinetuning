
import sys, time, json
from mlx_lm import load, generate

model, tokenizer = load('TinyLlama/TinyLlama-1.1B-Chat-v1.0', adapter_path='results/task1/adapter_r16')

prompts = []
with open('data/test.jsonl') as fh:
    for line in fh:
        row = json.loads(line)
        msgs = row.get('messages', [])
        user = next((m['content'] for m in msgs if m['role'] == 'user'), 'Hello')
        prompts.append(user)
        if len(prompts) >= 100: break

# Warm-up
generate(model, tokenizer, "Hello", max_tokens=5, verbose=False)

total_tokens = 0
total_sec    = 0.0

for user in prompts:
    if hasattr(tokenizer, 'apply_chat_template'):
        p = tokenizer.apply_chat_template(
            [{"role":"user","content":user}],
            tokenize=False, add_generation_prompt=True)
    else:
        p = user + "\nAssistant: "
    t0  = time.perf_counter()
    out = generate(model, tokenizer, p, max_tokens=100, verbose=False)
    t1  = time.perf_counter()
    toks = len(tokenizer.encode(out))
    total_tokens += toks
    total_sec    += (t1 - t0)

if total_sec > 0:
    tps = total_tokens / total_sec
    mpt = (total_sec / total_tokens) * 1000
    print(f'__TPS__={tps:.3f}')
    print(f'__MPT__={mpt:.3f}')
