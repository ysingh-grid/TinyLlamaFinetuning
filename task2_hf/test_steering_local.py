#!/usr/bin/env python3
"""Local smoke test: baseline vs steered (norm-relative steering + anti-repetition)."""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _repetition_score(text: str) -> float:
    """Higher = more suspicious repetition (best-effort)."""
    s = re.sub(r"\s+", " ", text).strip()
    if len(s) < 80:
        return 0.0
    # Count max occurrences of any 40-char substring
    worst = 0
    step = 20
    for i in range(0, min(len(s) - 40, 400), step):
        chunk = s[i : i + 40]
        worst = max(worst, s.count(chunk))
    return float(worst)


def main() -> int:
    sys.path.insert(0, str(ROOT))

    import torch
    from model_utils import format_prompt, generate, load_model_and_tokenizer
    from steering import ActivationSteerer, load_steering_vectors

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)

    print("Loading model…", flush=True)
    model, tok = load_model_and_tokenizer(device=device)
    vecs = load_steering_vectors(ROOT / "steering_vectors")
    steerer = ActivationSteerer(model, vecs)
    steerer.register_hooks()

    prompt = "How do I stay productive while working from home?"
    full = format_prompt(tok, prompt)
    max_new = 96

    steerer.active = False
    print("Generating baseline…", flush=True)
    baseline, t0 = generate(model, tok, full, device, max_new_tokens=max_new)

    print("Generating steered (α=0.5)…", flush=True)
    with steerer.steering_context(scale=0.5):
        steered, t1 = generate(
            model,
            tok,
            full,
            device,
            max_new_tokens=max_new,
            repetition_penalty=1.22,
            no_repeat_ngram_size=4,
        )

    print("\n=== BASELINE ===\n", baseline, "\n")
    print("=== STEERED ===\n", steered, "\n")
    print(f"latency baseline={t0}s steered={t1}s", flush=True)

    rb = _repetition_score(baseline)
    rs = _repetition_score(steered)
    print(f"repetition heuristic: baseline={rb:.1f} steered={rs:.1f}", flush=True)

    # Fail if steered looks much more repetitive than baseline
    if rs >= 5 and rs > rb + 2:
        print("FAIL: steered output looks overly repetitive.", file=sys.stderr)
        return 1
    print("OK (heuristic).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
