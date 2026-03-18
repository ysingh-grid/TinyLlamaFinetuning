
import sys
import time
import json
import mlx.core as mx
from mlx_lm import load, generate

try:
    model, tokenizer = load('TinyLlama/TinyLlama-1.1B-Chat-v1.0', adapter_path='results/task1/adapter_r8')
except Exception as e:
    print(f"\n__ERROR__={e}")
    sys.exit(1)

prompts = []
with open('data/test.jsonl') as f:
    for line in f:
        prompts.append(json.loads(line))
        if len(prompts) >= 2: break

total_tokens = 0
total_time = 0.0

generate(model, tokenizer, "Hello", max_tokens=2, verbose=False)

for p in prompts:
    msgs = p.get('messages', [])
    user_msg = next((m['content'] for m in msgs if m['role'] == 'user'), "Hello")
    
    if hasattr(tokenizer, 'apply_chat_template'):
        prompt = tokenizer.apply_chat_template([
            {"role": "user", "content": user_msg}
        ], tokenize=False, add_generation_prompt=True)
    else:
        prompt = user_msg + "\nAssistant: "
        
    start = time.perf_counter()
    res = generate(model, tokenizer, prompt, max_tokens=100, verbose=False)
    elapsed = time.perf_counter() - start
    
    total_time += elapsed
    total_tokens += len(tokenizer.encode(res))

if total_time > 0:
    tok_per_sec = total_tokens / total_time
    print(f"\n__TOK_PER_SEC__={tok_per_sec:.2f}")
