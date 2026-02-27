#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List
from urllib import error, request


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_num}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_num}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _parse_winner(text: str) -> str:
    """Parse winner from judge output, handling both direct and chain-of-thought outputs.

    Reasoning models (e.g. ministral-3b-reasoning) emit a long thinking block
    before the final verdict. We therefore scan the LAST 300 chars first for a
    clean verdict before falling back to full-text keyword counting.
    """
    import re

    normalized = text.strip().lower()
    if not normalized:
        return "invalid"

    verdict_map = {
        "left": "left", "right": "right", "tie": "tie",
        "draw": "tie", "equal": "tie", "a": "left", "b": "right",
    }

    # 1. Check the last 300 chars — reasoning models put verdict at the end.
    tail = normalized[-300:]
    # Look for an explicit "verdict: X" or "winner: X" pattern first.
    verdict_match = re.search(
        r"(?:verdict|winner|answer)[\s:]*([\w]+)", tail
    )
    if verdict_match:
        word = verdict_match.group(1).lower()
        if word in verdict_map:
            return verdict_map[word]

    # 2. Check the last clear standalone word in the tail.
    tail_words = re.split(r"[\s:.,;!\[\]()\"']+", tail.strip())
    for word in reversed(tail_words):
        if word in verdict_map:
            return verdict_map[word]

    # 3. Check the very first substantive word (direct / non-reasoning output).
    first_word = re.split(r"[\s:.,;!]+", normalized)[0]
    if first_word in verdict_map:
        return verdict_map[first_word]

    # 4. Keyword count over full text — only assign when one side dominates.
    left_count  = len(re.findall(r"\bleft\b", normalized))  + len(re.findall(r"\bresponse a\b", normalized))
    right_count = len(re.findall(r"\bright\b", normalized)) + len(re.findall(r"\bresponse b\b", normalized))
    tie_count   = (len(re.findall(r"\btie\b", normalized))
                   + len(re.findall(r"\bdraw\b", normalized))
                   + len(re.findall(r"\bequal\b", normalized)))

    counts   = {"left": left_count, "right": right_count, "tie": tie_count}
    non_zero = {k: v for k, v in counts.items() if v > 0}
    if len(non_zero) == 1:
        return next(iter(non_zero))
    if non_zero:
        return max(non_zero, key=non_zero.get)  # majority vote

    return "invalid"


def _truncate(text: str, max_chars: int) -> str:
    """Hard-truncate a response to max_chars, appending an ellipsis if cut."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ...[truncated]"


def _judge_prompt(prompt: str, left: str, right: str, max_response_chars: int = 800) -> str:
    left  = _truncate(left,  max_response_chars)
    right = _truncate(right, max_response_chars)
    return (
        "You are an impartial evaluator. Compare two responses to a user prompt. "
        "Judge correctness, instruction-following, relevance, and helpfulness. "
        "Think step-by-step, then end with: Verdict: LEFT, Verdict: RIGHT, or Verdict: TIE.\n\n"
        f"Prompt:\n{prompt}\n\n"
        f"LEFT RESPONSE:\n{left}\n\n"
        f"RIGHT RESPONSE:\n{right}\n\n"
        "Your evaluation:"
    )


def _chat_completion(
    base_url: str,
    model: str,
    user_prompt: str,
    timeout_s: int,
    max_tokens: int = 1024,
) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an impartial judge. Evaluate carefully, then end your response "
                    "with exactly: Verdict: LEFT, Verdict: RIGHT, or Verdict: TIE."
                ),
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LM Studio HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Unable to reach LM Studio at {url}: {exc}") from exc

    try:
        parsed = json.loads(body)
        return str(parsed["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unexpected LM Studio response: {body}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Judge pairwise tasks using a local LM Studio OpenAI-compatible server."
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="gpt-oss-20b")
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--sleep-ms", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--max-judge-tokens", type=int, default=1024,
        help="Max tokens for judge completion (raise for reasoning models). Default: 1024"
    )
    parser.add_argument(
        "--max-response-chars", type=int, default=800,
        help="Truncate each response to this many chars before sending to judge. Default: 800"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = read_jsonl(args.tasks)
    if not tasks:
        raise ValueError(f"No judging tasks found in {args.tasks}")

    rows: List[Dict] = []
    total = len(tasks)
    for idx, task in enumerate(tasks, start=1):
        prompt = _judge_prompt(
            prompt=str(task["prompt"]),
            left=str(task["left_response"]),
            right=str(task["right_response"]),
            max_response_chars=args.max_response_chars,
        )
        raw = _chat_completion(
            base_url=args.base_url,
            model=args.model,
            user_prompt=prompt,
            timeout_s=args.timeout_s,
            max_tokens=args.max_judge_tokens,
        )
        winner = _parse_winner(raw)
        rows.append(
            {
                "item_id": task["item_id"],
                "winner": winner,
                "raw_judge_output": raw,
                "judge_model": args.model,
                "judge_endpoint": args.base_url,
            }
        )

        if args.progress_every > 0 and (idx % args.progress_every == 0 or idx == total):
            print(f"Judged {idx}/{total}", flush=True)

        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    write_jsonl(args.out, rows)
    print(f"Wrote judgments: {args.out}")


if __name__ == "__main__":
    main()
