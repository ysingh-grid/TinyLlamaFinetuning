
import sys, time, json, statistics
from mlx_lm import load, generate

try:
    model, tokenizer = load('results/task4/model_q8')
except Exception as e:
    print(e)
    sys.exit(1)

prompts = ['Explain machine learning in simple terms.', 'Write a Python script to parse a CSV.', 'How do quantum computers work?', 'Explain machine learning in simple terms.', 'Write a Python script to parse a CSV.', 'How do quantum computers work?', 'Explain machine learning in simple terms.', 'Write a Python script to parse a CSV.', 'How do quantum computers work?', 'Explain machine learning in simple terms.', 'Write a Python script to parse a CSV.', 'How do quantum computers work?', 'Explain machine learning in simple terms.', 'Write a Python script to parse a CSV.', 'How do quantum computers work?', 'Explain machine learning in simple terms.', 'Write a Python script to parse a CSV.', 'How do quantum computers work?', 'Explain machine learning in simple terms.', 'Write a Python script to parse a CSV.']
batch_sizes = [1, 4, 8]
results = {}

# warm-up
generate(model, tokenizer, prompts[0], max_tokens=5, verbose=False)

for bs in batch_sizes:
    run_times = []
    run_tokens = []
    for run_i in range(10):   # 10 generations for latency variance
        p = prompts[run_i % len(prompts)]
        t0 = time.perf_counter()
        for _ in range(bs):
            out = generate(model, tokenizer, p, max_tokens=100, verbose=False)
            run_tokens.append(len(tokenizer.encode(out)))
        run_times.append(time.perf_counter() - t0)
    total_tokens = sum(run_tokens)
    total_sec = sum(run_times)
    results[f"batch_{bs}_tok_sec"] = total_tokens / total_sec if total_sec > 0 else 0
    results[f"batch_{bs}_latency_mean_sec"] = statistics.mean(run_times)
    results[f"batch_{bs}_latency_var_sec"]  = statistics.variance(run_times) if len(run_times)>1 else 0

print(f"\n__BENCH__={json.dumps(results)}")
