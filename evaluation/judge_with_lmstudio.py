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
    """Parse winner from judge output, avoiding order-dependent substring bias.

    Strategy:
    1. Check if the first token is an unambiguous verdict (LEFT/RIGHT/TIE/A/B).
    2. If the output is verbose, count keyword occurrences and only assign
       when a single verdict keyword dominates.
    3. Otherwise return "invalid".
    """
    import re

    normalized = text.strip().lower()
    if not normalized:
        return "invalid"

    # 1. Check the first substantive word (most reliable signal).
    first_word = re.split(r"[\s:.,;!]+", normalized)[0]
    first_word_map = {
        "left": "left",
        "right": "right",
        "tie": "tie",
        "draw": "tie",
        "equal": "tie",
        "a": "left",
        "b": "right",
    }
    if first_word in first_word_map:
        return first_word_map[first_word]

    # 2. Count keyword occurrences — only assign if one side dominates.
    left_count = len(re.findall(r"\bleft\b", normalized)) + len(re.findall(r"\bresponse a\b", normalized))
    right_count = len(re.findall(r"\bright\b", normalized)) + len(re.findall(r"\bresponse b\b", normalized))
    tie_count = len(re.findall(r"\btie\b", normalized)) + len(re.findall(r"\bdraw\b", normalized)) + len(re.findall(r"\bequal\b", normalized))

    counts = {"left": left_count, "right": right_count, "tie": tie_count}
    non_zero = {k: v for k, v in counts.items() if v > 0}

    if len(non_zero) == 1:
        return next(iter(non_zero))

    # Ambiguous or no verdict keywords found.
    return "invalid"


def _judge_prompt(prompt: str, left: str, right: str) -> str:
    return (
        "You are an impartial evaluator. Compare two responses to a user prompt. "
        "Judge correctness, instruction-following, relevance/helpfulness, and safety. "
        "Respond with exactly one token: LEFT, RIGHT, or TIE.\n\n"
        f"Prompt:\n{prompt}\n\n"
        f"LEFT RESPONSE:\n{left}\n\n"
        f"RIGHT RESPONSE:\n{right}\n\n"
        "Verdict (LEFT/RIGHT/TIE):"
    )


def _chat_completion(base_url: str, model: str, user_prompt: str, timeout_s: int) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Return only LEFT, RIGHT, or TIE.",
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "temperature": 0,
        "max_tokens": 4,
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
    parser.add_argument("--timeout-s", type=int, default=60)
    parser.add_argument("--sleep-ms", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100)
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
        )
        raw = _chat_completion(
            base_url=args.base_url,
            model=args.model,
            user_prompt=prompt,
            timeout_s=args.timeout_s,
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
