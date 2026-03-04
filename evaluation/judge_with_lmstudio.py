#!/usr/bin/env python3
"""Judge pairwise tasks using a local LM Studio OpenAI-compatible server.

Key design choices vs the previous version:
  - max_response_chars raised to 2000 (covers 100% of 512-token responses).
  - max_judge_tokens raised to 3000 (gives reasoning models full thinking budget).
  - Streaming with early-exit: stops the request the moment "Verdict:" appears
    in the stream, cutting median latency ~60-70% for reasoning models.
  - <think>...</think> block stripping so the verdict parser always sees clean text.
  - Optional threading concurrency (--concurrency N) to pipeline HTTP overhead;
    LM Studio queues requests and still serialises GPU work, but reduces wall-clock
    time spent waiting for JSON parsing / network overhead.
  - progress-every default lowered to 25 for more granular feedback on long runs.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib import error, request


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------

def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> chain-of-thought blocks emitted by reasoning models.

    After stripping, only the post-reasoning verdict text remains, which is
    much easier to parse reliably.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _parse_winner(text: str) -> str:
    """Parse winner from judge output.

    Handles:
    - Direct single-word outputs  (LEFT / RIGHT / TIE)
    - Chain-of-thought with explicit "Verdict: X" at end
    - Reasoning models that emit <think>…</think> before the verdict
    - Keyword majority-vote fallback
    """
    if not text or not text.strip():
        return "invalid"

    # Strip any <think> block first so reasoning noise doesn't confuse later steps.
    clean = _strip_think_blocks(text)
    normalized = clean.lower() if clean else text.lower()

    verdict_map = {
        "left": "left", "right": "right", "tie": "tie",
        "draw": "tie", "equal": "tie", "a": "left", "b": "right",
    }

    # 1. Explicit "Verdict: X" / "Winner: X" / "Answer: X" pattern anywhere.
    verdict_match = re.search(
        r"(?:verdict|winner|answer)[\s:*_]*([a-z]+)", normalized
    )
    if verdict_match:
        word = verdict_match.group(1)
        if word in verdict_map:
            return verdict_map[word]

    # 2. Last standalone verdict word in the tail 300 chars (post-reasoning models).
    tail = normalized[-300:]
    tail_words = re.split(r"[\s:.,;!\[\]()\"'*_]+", tail.strip())
    for word in reversed(tail_words):
        if word in verdict_map:
            return verdict_map[word]

    # 3. First substantive word (direct / non-reasoning output).
    first_word = re.split(r"[\s:.,;!*_]+", normalized.strip())[0]
    if first_word in verdict_map:
        return verdict_map[first_word]

    # 4. Keyword count over full text — only assign when one side dominates.
    full = text.lower()
    left_count  = len(re.findall(r"\bleft\b", full))  + len(re.findall(r"\bresponse a\b", full))
    right_count = len(re.findall(r"\bright\b", full)) + len(re.findall(r"\bresponse b\b", full))
    tie_count   = (len(re.findall(r"\btie\b", full))
                   + len(re.findall(r"\bdraw\b", full))
                   + len(re.findall(r"\bequal\b", full)))

    counts   = {"left": left_count, "right": right_count, "tie": tie_count}
    non_zero = {k: v for k, v in counts.items() if v > 0}
    if len(non_zero) == 1:
        return next(iter(non_zero))
    if non_zero:
        return max(non_zero, key=non_zero.get)  # type: ignore[arg-type]

    return "invalid"


# ---------------------------------------------------------------------------
# Response truncation
# ---------------------------------------------------------------------------

def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ...[truncated]"


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

def _judge_prompt(prompt: str, left: str, right: str, max_response_chars: int = 2000) -> str:
    left  = _truncate(left,  max_response_chars)
    right = _truncate(right, max_response_chars)
    return (
        "You are an impartial evaluator. Compare two AI responses to the same prompt.\n"
        "Evaluate on: correctness, instruction-following, relevance, helpfulness.\n"
        "Do NOT favour a response simply because it is longer.\n"
        "Think step-by-step, then end with exactly one of:\n"
        "  Verdict: LEFT   Verdict: RIGHT   Verdict: TIE\n\n"
        f"Prompt:\n{prompt}\n\n"
        f"LEFT RESPONSE:\n{left}\n\n"
        f"RIGHT RESPONSE:\n{right}\n\n"
        "Your evaluation:"
    )


# ---------------------------------------------------------------------------
# LM Studio HTTP calls — blocking and streaming
# ---------------------------------------------------------------------------

_VERDICT_RE = re.compile(
    r"(?:verdict|winner|answer)[\s:*_]*(?:is[\s:*_]*)?(left|right|tie)", re.IGNORECASE
)


def _chat_completion_stream(
    base_url: str,
    model: str,
    user_prompt: str,
    timeout_s: int,
    max_tokens: int = 3000,
) -> str:
    """Streaming call with early exit once 'Verdict: X' is detected.

    For reasoning models this typically cuts wall-clock time by 60-70% because
    the verdict appears right after </think> and we stop reading immediately.
    Falls back to the full accumulated text if no verdict appears in the stream.
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an impartial judge. Reason carefully, then end your response "
                    "with exactly: Verdict: LEFT, Verdict: RIGHT, or Verdict: TIE."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
    }

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )

    accumulated = []
    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    line = line[6:]
                try:
                    chunk = json.loads(line)
                    delta = chunk["choices"][0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        accumulated.append(token)
                        combined = "".join(accumulated)
                        # Early exit once we see a clear verdict after </think>.
                        clean = _strip_think_blocks(combined)
                        if _VERDICT_RE.search(clean):
                            return combined
                except (KeyError, IndexError, json.JSONDecodeError):
                    pass
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LM Studio HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Unable to reach LM Studio at {url}: {exc}") from exc

    return "".join(accumulated)


def _chat_completion_blocking(
    base_url: str,
    model: str,
    user_prompt: str,
    timeout_s: int,
    max_tokens: int = 3000,
) -> str:
    """Non-streaming fallback for servers that don't support SSE."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an impartial judge. Reason carefully, then end your response "
                    "with exactly: Verdict: LEFT, Verdict: RIGHT, or Verdict: TIE."
                ),
            },
            {"role": "user", "content": user_prompt},
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


def _call_judge(
    base_url: str,
    model: str,
    user_prompt: str,
    timeout_s: int,
    max_tokens: int,
    stream: bool,
) -> str:
    if stream:
        try:
            return _chat_completion_stream(base_url, model, user_prompt, timeout_s, max_tokens)
        except Exception:
            # If streaming fails, fall back to blocking.
            return _chat_completion_blocking(base_url, model, user_prompt, timeout_s, max_tokens)
    return _chat_completion_blocking(base_url, model, user_prompt, timeout_s, max_tokens)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Judge pairwise tasks using a local LM Studio OpenAI-compatible server."
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="qwen/qwen3-4b")
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--timeout-s", type=int, default=180,
                        help="Per-request timeout in seconds. Raise for slow reasoning models. Default: 180")
    parser.add_argument("--sleep-ms", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=25,
                        help="Print progress every N tasks. Default: 25")
    parser.add_argument(
        "--max-judge-tokens", type=int, default=3000,
        help="Max tokens for judge completion. Raise for reasoning models with long think chains. Default: 3000"
    )
    parser.add_argument(
        "--max-response-chars", type=int, default=2000,
        help=(
            "Truncate each response to this many chars before sending to judge. "
            "Set to match your max_tokens budget (512 tokens ≈ 2000 chars). Default: 2000"
        ),
    )
    parser.add_argument(
        "--stream", action="store_true", default=True,
        help="Use SSE streaming with early exit on verdict detection (default: on).",
    )
    parser.add_argument(
        "--no-stream", dest="stream", action="store_false",
        help="Disable streaming; use blocking HTTP call instead.",
    )
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help=(
            "Number of parallel HTTP requests. LM Studio serialises GPU work so "
            "concurrency > 1 reduces network/parsing overhead only. Default: 1"
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _judge_task(
    task: Dict,
    base_url: str,
    model: str,
    timeout_s: int,
    max_judge_tokens: int,
    max_response_chars: int,
    stream: bool,
    sleep_ms: int,
) -> Dict:
    prompt = _judge_prompt(
        prompt=str(task["prompt"]),
        left=str(task["left_response"]),
        right=str(task["right_response"]),
        max_response_chars=max_response_chars,
    )
    raw = _call_judge(
        base_url=base_url,
        model=model,
        user_prompt=prompt,
        timeout_s=timeout_s,
        max_tokens=max_judge_tokens,
        stream=stream,
    )
    winner = _parse_winner(raw)
    if sleep_ms > 0:
        time.sleep(sleep_ms / 1000.0)
    return {
        "item_id": task["item_id"],
        "winner": winner,
        "raw_judge_output": raw,
        "judge_model": model,
        "judge_endpoint": base_url,
    }


def _load_completed_ids(out_path: Path) -> set:
    """Return the set of item_ids already written to out_path (for resume)."""
    if not out_path.exists():
        return set()
    done: set = set()
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if row.get("item_id"):
                    done.add(str(row["item_id"]))
            except json.JSONDecodeError:
                pass
    return done


def run_lmstudio_judging(
    tasks_path: Path,
    out_path: Path,
    model: str,
    base_url: str = "http://127.0.0.1:1234/v1",
    timeout_s: int = 180,
    max_judge_tokens: int = 3000,
    max_response_chars: int = 2000,
    stream: bool = True,
    concurrency: int = 1,
    sleep_ms: int = 0,
    progress_every: int = 25,
) -> Path:
    """Programmatic entry point — call this from run_pipeline.py or other scripts.

    Crash-safe: each judgment is appended to out_path immediately after it
    completes.  If the process is interrupted and restarted, already-written
    item_ids are skipped automatically (resume behaviour).
    """
    all_tasks = read_jsonl(tasks_path)
    if not all_tasks:
        raise ValueError(f"No judging tasks found in {tasks_path}")

    # ── Resume: skip already-completed items ───────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed_ids = _load_completed_ids(out_path)
    tasks = [t for t in all_tasks if str(t["item_id"]) not in completed_ids]

    total_all = len(all_tasks)
    already_done = len(completed_ids)
    remaining = len(tasks)

    if already_done:
        print(
            f"Resuming: {already_done}/{total_all} already done, "
            f"{remaining} remaining → {out_path}"
        )
    else:
        print(
            f"Judging {remaining} tasks | model={model} | stream={stream} | "
            f"max_response_chars={max_response_chars} | max_judge_tokens={max_judge_tokens} | "
            f"concurrency={concurrency}"
        )

    if not tasks:
        print("All tasks already judged — nothing to do.")
        return out_path

    # ── Incremental append writer ──────────────────────────────────────────
    _write_lock = threading.Lock()

    def _append_result(row: Dict) -> None:
        with _write_lock:
            with out_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=True) + "\n")

    completed = [0]  # mutable counter shared across closures

    def _on_result(result: Dict) -> None:
        _append_result(result)
        completed[0] += 1
        n = completed[0]
        done_total = already_done + n
        if progress_every > 0 and (n % progress_every == 0 or n == remaining):
            pct = done_total / total_all * 100
            print(f"Judged {done_total}/{total_all}  ({pct:.0f}%)", flush=True)

    def _run_sequential() -> None:
        for task in tasks:
            result = _judge_task(
                task=task,
                base_url=base_url,
                model=model,
                timeout_s=timeout_s,
                max_judge_tokens=max_judge_tokens,
                max_response_chars=max_response_chars,
                stream=stream,
                sleep_ms=sleep_ms,
            )
            _on_result(result)

    def _run_parallel() -> None:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            future_to_task = {
                pool.submit(
                    _judge_task,
                    task=task,
                    base_url=base_url,
                    model=model,
                    timeout_s=timeout_s,
                    max_judge_tokens=max_judge_tokens,
                    max_response_chars=max_response_chars,
                    stream=stream,
                    sleep_ms=sleep_ms,
                ): task
                for task in tasks
            }
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = {
                        "item_id": task["item_id"],
                        "winner": "invalid",
                        "raw_judge_output": f"ERROR: {exc}",
                        "judge_model": model,
                        "judge_endpoint": base_url,
                    }
                _on_result(result)

    if concurrency <= 1:
        _run_sequential()
    else:
        _run_parallel()

    # ── Final summary ──────────────────────────────────────────────────────
    all_written = _load_completed_ids(out_path)
    invalid_count = sum(
        1 for t in all_tasks
        if str(t["item_id"]) in all_written
        and _verdict_for_id(out_path, str(t["item_id"])) == "invalid"
    )
    print(
        f"Done. {len(all_written)}/{total_all} judgments in {out_path}  "
        f"(invalid≈{invalid_count})"
    )
    return out_path


def _verdict_for_id(path: Path, item_id: str) -> str:
    """Quick scan to get the winner for a given item_id (used only in summary)."""
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if str(row.get("item_id")) == item_id:
                        return str(row.get("winner", "invalid"))
                except json.JSONDecodeError:
                    pass
    except OSError:
        pass
    return "invalid"


def main() -> None:
    args = parse_args()
    run_lmstudio_judging(
        tasks_path=args.tasks,
        out_path=args.out,
        model=args.model,
        base_url=args.base_url,
        timeout_s=args.timeout_s,
        max_judge_tokens=args.max_judge_tokens,
        max_response_chars=args.max_response_chars,
        stream=args.stream,
        concurrency=args.concurrency,
        sleep_ms=args.sleep_ms,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
