#!/usr/bin/env python3
"""
evaluate_perplexity.py

Compute token-averaged cross-entropy loss and perplexity for all models
listed in `evaluation/models.json` on a JSONL test set (default:
`data/test.jsonl`) using an MLX forward pass.

Usage (from project root):

    .venv/bin/python evaluate_perplexity.py \
        --data data/test.jsonl \
        --models-config evaluation/models.json \
        --out evaluation/perplexity_report.md
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).parent


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model: str
    adapter_path: Optional[str] = None
    system_prompt: str = ""


def load_models_config(path: Path) -> List[ModelSpec]:
    if not path.exists():
        raise FileNotFoundError(f"Models config not found: {path}")
    raw = json.loads(path.read_text())
    specs: List[ModelSpec] = []
    for item in raw:
        specs.append(
            ModelSpec(
                name=item["name"],
                model=item["model"],
                adapter_path=item.get("adapter_path"),
                system_prompt=item.get("system_prompt", "") or "",
            )
        )
    return specs


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as exc:
                raise ValueError(f"Failed to parse JSONL line in {path}: {line!r}") from exc


def extract_chat_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    The test set is stored in the same MLX chat JSONL format as training:

        {"messages":[{"role":"user","content":...},{"role":"assistant","content":...}]}

    This function normalises the shape and validates basic structure.
    """
    msgs = row.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise ValueError(f"Missing or invalid 'messages' field: {row!r}")
    out: List[Dict[str, str]] = []
    for m in msgs:
        role = str(m.get("role", "")).strip()
        content = str(m.get("content", ""))
        if not role:
            raise ValueError(f"Message without role in row: {row!r}")
        out.append({"role": role, "content": content})
    return out


def build_tokens_for_example(
    tokenizer: Any,
    messages: List[Dict[str, str]],
) -> List[int]:
    """
    Convert chat messages to token IDs using the same ChatML-style template
    used throughout the project. We treat the entire sequence (user + assistant)
    as the target for perplexity, which is sufficient for comparing models.
    """
    try:
        token_ids: List[int] = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
    except Exception:
        # Fallback: simple concatenation if chat template is unavailable.
        text_parts: List[str] = []
        for m in messages:
            prefix = f"{m['role'].capitalize()}: "
            text_parts.append(prefix + m["content"] + "\n")
        text = "".join(text_parts)
        token_ids = tokenizer.encode(text)
    return token_ids


def compute_nll_for_tokens(model: Any, tokens: List[int]) -> Tuple[float, int]:
    """
    Compute negative log-likelihood for a single token sequence.

    nll = -sum_t log p(x_t | x_<t)
    Returns (nll, token_count) where token_count is the number of predicted tokens.
    """
    import mlx.core as mx  # type: ignore

    if len(tokens) < 2:
        return 0.0, 0

    # Predict token t given tokens up to t-1.
    input_ids = mx.array(tokens[:-1])[None, :]  # (1, T-1)
    target_ids = mx.array(tokens[1:])  # (T-1,)

    logits = model(input_ids)  # (1, T-1, V)
    # Convert to log-probabilities.
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    # Gather log-probabilities of the actual next tokens.
    idx = mx.arange(target_ids.shape[0])
    token_log_probs = log_probs[0, idx, target_ids]
    total_log_prob = mx.sum(token_log_probs).item()

    # NLL is -log p(x).
    nll = -float(total_log_prob)
    return nll, int(target_ids.shape[0])


def compute_perplexity_for_model(
    spec: ModelSpec,
    rows: Sequence[Dict[str, Any]],
) -> Tuple[float, float, int]:
    """
    Compute (cross_entropy, perplexity, total_tokens) for a single model.
    """
    try:
        import mlx.core as mx  # type: ignore
        from mlx_lm import load  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "Perplexity evaluation requires mlx and mlx_lm. "
            "Install them in your virtualenv first."
        ) from exc

    # TokenizersBackend shim (mirrors evaluation/pipeline.py) for quantised models.
    try:
        import transformers  # type: ignore

        if not hasattr(transformers, "TokenizersBackend"):
            from transformers import PreTrainedTokenizerFast  # type: ignore

            class TokenizersBackend(PreTrainedTokenizerFast):  # type: ignore
                pass

            transformers.TokenizersBackend = TokenizersBackend  # type: ignore
    except Exception:
        pass

    load_kwargs: Dict[str, Any] = {}
    if spec.adapter_path:
        load_kwargs["adapter_path"] = spec.adapter_path

    print(f"Loading model: {spec.name} ({spec.model})", flush=True)
    model, tokenizer = load(spec.model, **load_kwargs)

    total_nll = 0.0
    total_tokens = 0

    for idx, row in enumerate(rows, start=1):
        try:
            messages = extract_chat_messages(row)
        except ValueError as exc:
            raise ValueError(f"Invalid example at index {idx}: {exc}") from exc

        token_ids = build_tokens_for_example(tokenizer, messages)
        if not token_ids:
            continue

        nll, token_count = compute_nll_for_tokens(model, token_ids)
        if token_count == 0:
            continue

        total_nll += nll
        total_tokens += token_count

        if idx % 50 == 0:
            avg_loss = total_nll / max(total_tokens, 1)
            print(
                f"[{spec.name}] processed {idx} examples — "
                f"running cross-entropy {avg_loss:.4f}",
                flush=True,
            )

    # Free GPU / unified memory.
    del model
    del tokenizer
    try:
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal"):
            mx.metal.clear_cache()
    except Exception:
        pass

    if total_tokens == 0:
        raise RuntimeError(f"No tokens processed for model {spec.name}")

    cross_entropy = total_nll / total_tokens
    perplexity = math.exp(cross_entropy)
    return cross_entropy, perplexity, total_tokens


def write_markdown_report(
    out_path: Path,
    results: List[Tuple[ModelSpec, float, float, int]],
    data_path: Path,
) -> None:
    lines: List[str] = []
    lines.append("# Test-set Perplexity Report")
    lines.append("")
    lines.append(f"- **Data file:** `{data_path}`")
    lines.append("- **Metric:** token-averaged cross-entropy loss over full chat sequence")
    lines.append("")
    lines.append("| Model | Cross-Entropy (nats) | Perplexity | Tokens |")
    lines.append("|-------|----------------------|-----------:|-------:|")
    for spec, ce, ppl, tokens in results:
        lines.append(
            f"| `{spec.name}` | {ce:.4f} | {ppl:.3f} | {tokens} |"
        )
    lines.append("")
    lines.append(
        "> Note: Perplexities are computed over the entire chat sequence "
        "(user + assistant), using the ChatML-style template for tokenisation."
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"Wrote report to {out_path}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Compute test-set perplexity for all models.")
    parser.add_argument(
        "--data",
        type=str,
        default=str(ROOT / "data" / "test.jsonl"),
        help="Path to test JSONL file (default: data/test.jsonl).",
    )
    parser.add_argument(
        "--models-config",
        type=str,
        default=str(ROOT / "evaluation" / "models.json"),
        help="Path to evaluation/models.json (default: evaluation/models.json).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "evaluation" / "perplexity_report.md"),
        help="Path to write markdown report (default: evaluation/perplexity_report.md).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    data_path = Path(args.data)
    models_path = Path(args.models_config)
    out_path = Path(args.out)

    specs = load_models_config(models_path)
    rows = list(iter_jsonl(data_path))
    if not rows:
        raise RuntimeError(f"No rows found in {data_path}")

    results: List[Tuple[ModelSpec, float, float, int]] = []
    for spec in specs:
        ce, ppl, tokens = compute_perplexity_for_model(spec, rows)
        print(
            f"[{spec.name}] cross-entropy={ce:.4f}, perplexity={ppl:.3f}, tokens={tokens}",
            flush=True,
        )
        results.append((spec, ce, ppl, tokens))

    write_markdown_report(out_path, results, data_path)


if __name__ == "__main__":
    main()

