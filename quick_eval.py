#!/usr/bin/env python3
"""Quick sanity check evaluation for an adapter vs base model.

Usage:
  .venv/bin/python quick_eval.py --adapter ./adapters/tinyllama-lora-alpaca --n-prompts 10
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

def parse_args():
    parser = argparse.ArgumentParser(description="Quickly eyeball base vs fine-tuned adapter generation.")
    parser.add_argument("--adapter", type=str, required=True, help="Path to adapter directory")
    parser.add_argument("--base-model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--prompts-file", type=str, default="evaluation/eval_prompts.jsonl")
    parser.add_argument("--n-prompts", type=int, default=10, help="Number of prompts to evaluate")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    return parser.parse_args()

def format_prompt(prompt: str, tokenizer) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"User: {prompt}\nAssistant:"

def load_prompts(file_path: str, n: int) -> list[str]:
    prompts = []
    path = Path(file_path)
    if not path.exists():
        print(f"Warning: {file_path} not found.")
        return []
        
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n: break
            prompts.append(json.loads(line)["prompt"])
    return prompts

def get_stats(responses: list[str]) -> dict:
    if not responses:
        return {"avg_len": 0}
    avg_len = sum(len(r) for r in responses) / len(responses)
    return {"avg_len": avg_len}

def main():
    args = parse_args()
    prompts = load_prompts(args.prompts_file, args.n_prompts)
    if not prompts:
        print("No prompts loaded. Exiting.")
        return

    print(f"\n=========================================")
    print(f"  QUICK EVAL: BASE vs ADAPTER          ")
    print(f"=========================================\n")

    sampler = make_sampler(args.temperature)
    
    # 1. Base Model Check
    print(f"Loading Base Model: {args.base_model}...")
    base_model, base_tokenizer = load(args.base_model)
    
    print(f"Generating base responses... ({args.n_prompts} prompts)")
    base_responses = []
    start_time = time.time()
    for p in prompts:
        fmt = format_prompt(p, base_tokenizer)
        res = generate(base_model, base_tokenizer, prompt=fmt, max_tokens=args.max_tokens, sampler=sampler, verbose=False)
        base_responses.append(res.strip())
    print(f"  [Base generation took {time.time() - start_time:.1f}s]\n")

    # Clear memory explicitly
    del base_model
    if hasattr(mx, "metal"):
        mx.metal.clear_cache()
        
    # 2. Adapter Model Check
    print(f"Loading Adapter: {args.adapter}...")
    ft_model, ft_tokenizer = load(args.base_model, adapter_path=args.adapter)
    
    print(f"Generating adapter responses... ({args.n_prompts} prompts)")
    ft_responses = []
    start_time = time.time()
    for p in prompts:
        fmt = format_prompt(p, ft_tokenizer)
        res = generate(ft_model, ft_tokenizer, prompt=fmt, max_tokens=args.max_tokens, sampler=sampler, verbose=False)
        ft_responses.append(res.strip())
    print(f"  [Adapter generation took {time.time() - start_time:.1f}s]\n")

    # 3. Present Results
    print("=" * 80)
    print("  SIDE-BY-SIDE COMPARISON")
    print("=" * 80)
    for i, (p, b_res, f_res) in enumerate(zip(prompts, base_responses, ft_responses)):
        print(f"\n{'-'*80}")
        print(f"💬 PROMPT {i+1}:\n{p}")
        print(f"{'-'*80}")
        print(f"[BASE MODEL] :\n{b_res}")
        print(f"\n[FINE-TUNED]: \n{f_res}")
        
    # 4. Global Stats
    b_stats = get_stats(base_responses)
    f_stats = get_stats(ft_responses)
    
    print("\n" + "=" * 80)
    print("  QUICK STATS")
    print("=" * 80)
    print(f"Base avg response length:    {b_stats['avg_len']:.1f} chars")
    print(f"Adapter avg response length: {f_stats['avg_len']:.1f} chars")
    print(f"\nTip: If adapter outputs are extremely short or repetitive, the model collapsed.")

if __name__ == "__main__":
    main()
