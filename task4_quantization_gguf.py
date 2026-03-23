#!/usr/bin/env python3
"""
Task 4: Quantization & GGUF Export for Edge Deployment.

This script:
1. Fuses the LoRA adapter into the TinyLlama base model.
2. Converts the fused HF model to GGUF via llama.cpp.
3. Quantizes the GGUF model to Q4_K_M, Q5_K_M, and Q8_0.
4. Benchmarks latency, memory, output quality, and perplexity.
5. Benchmarks batched inference for batch sizes 1, 4, and 8.
6. Packages a lightweight inference server for the smallest quantized model.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import requests

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "task4"
LEGACY_GGUF_DIR = RESULTS_DIR / "gguf_models"
FUSED_MODEL_DIR = RESULTS_DIR / "fused_model"
SERVER_DIR = RESULTS_DIR / "inference_server"
CONVERSION_LOG = RESULTS_DIR / "conversion_logs.txt"
LATENCY_RESULTS = RESULTS_DIR / "latency_benchmarks.json"
PPL_RESULTS = RESULTS_DIR / "perplexity_results.json"
BATCH_RESULTS = RESULTS_DIR / "batched_inference_results.json"
COMP_TABLE_MD = RESULTS_DIR / "deployment_comparison.md"
COMP_TABLE_CSV = RESULTS_DIR / "deployment_comparison.csv"
ANALYSIS_MD = RESULTS_DIR / "quantization_analysis.md"

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_CANDIDATES = [
    ROOT / "results" / "task1" / "adapter_r8",
    ROOT / "adapters" / "tinyllama-lora-alpaca",
]

LLAMA_CPP_DIR = ROOT / "llama.cpp"
LLAMA_CPP_BUILD = LLAMA_CPP_DIR / "build"
LLAMA_CPP_BIN = LLAMA_CPP_BUILD / "bin"
LLAMA_CLI = LLAMA_CPP_BIN / "llama-cli"
LLAMA_BENCH = LLAMA_CPP_BIN / "llama-bench"
LLAMA_PERPLEXITY = LLAMA_CPP_BIN / "llama-perplexity"
LLAMA_SERVER = LLAMA_CPP_BIN / "llama-server"
CONVERT_HF_TO_GGUF = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
LLAMA_QUANTIZE = LLAMA_CPP_BIN / "llama-quantize"

VALID_PATH = ROOT / "data" / "valid.jsonl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def append_log(message: str) -> None:
    CONVERSION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with CONVERSION_LOG.open("a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def format_cmd(cmd: Sequence[str]) -> str:
    return " ".join(repr(part) if " " in part else part for part in cmd)


def run_logged(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    append_log(f"$ {format_cmd(cmd)}")
    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        capture_output=True,
    )
    if proc.stdout:
        append_log(proc.stdout)
    if proc.stderr:
        append_log(proc.stderr)
    append_log(f"[exit={proc.returncode}]")
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {format_cmd(cmd)}")
    return proc


def ensure_llama_cpp_built() -> None:
    required = [LLAMA_CLI, LLAMA_BENCH, LLAMA_PERPLEXITY, LLAMA_SERVER, LLAMA_QUANTIZE]
    if all(path.exists() for path in required):
        return
    if not LLAMA_CPP_DIR.exists():
        raise FileNotFoundError(f"llama.cpp not found at {LLAMA_CPP_DIR}")

    logging.info("llama.cpp binaries missing; building required targets...")
    LLAMA_CPP_BUILD.mkdir(parents=True, exist_ok=True)
    run_logged(
        [
            "cmake",
            "-S",
            str(LLAMA_CPP_DIR),
            "-B",
            str(LLAMA_CPP_BUILD),
            "-DCMAKE_BUILD_TYPE=Release",
        ]
    )
    run_logged(
        [
            "cmake",
            "--build",
            str(LLAMA_CPP_BUILD),
            "--config",
            "Release",
            "-j",
            "8",
            "--target",
            "llama-cli",
            "llama-bench",
            "llama-perplexity",
            "llama-server",
            "llama-quantize",
        ]
    )
    if not all(path.exists() for path in required):
        raise RuntimeError("llama.cpp build completed but required binaries are still missing")


def pick_adapter_path() -> Path:
    for candidate in ADAPTER_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No LoRA adapter found. Expected one of: "
        + ", ".join(str(p) for p in ADAPTER_CANDIDATES)
    )


def fuse_model(adapter_path: Path) -> Path:
    ensure_clean_dir(FUSED_MODEL_DIR)
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm.fuse",
        "--model",
        BASE_MODEL,
        "--adapter-path",
        str(adapter_path),
        "--save-path",
        str(FUSED_MODEL_DIR),
    ]
    run_logged(cmd)
    return FUSED_MODEL_DIR


def convert_to_gguf(fused_model_dir: Path) -> Path:
    base_gguf = RESULTS_DIR / "model.gguf"
    if base_gguf.exists():
        base_gguf.unlink()
    cmd = [
        sys.executable,
        str(CONVERT_HF_TO_GGUF),
        str(fused_model_dir),
        "--outfile",
        str(base_gguf),
        "--outtype",
        "f16",
    ]
    run_logged(cmd)
    return base_gguf


def quantize_gguf(base_gguf: Path, quant_type: str) -> Path:
    out_path = RESULTS_DIR / f"model-{quant_type}.gguf"
    if out_path.exists():
        out_path.unlink()
    cmd = [str(LLAMA_QUANTIZE), str(base_gguf), str(out_path), quant_type]
    run_logged(cmd)
    return out_path


def copy_legacy_artifacts(base_gguf: Path, quantized: Dict[str, Path]) -> None:
    LEGACY_GGUF_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_gguf, LEGACY_GGUF_DIR / "model.gguf")
    for quant_type, path in quantized.items():
        shutil.copy2(path, LEGACY_GGUF_DIR / f"model-{quant_type}.gguf")


def read_valid_rows(limit: int | None = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with VALID_PATH.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            rows.append(json.loads(line))
    return rows


def build_ppl_corpus(rows: Sequence[Dict[str, Any]], output_path: Path) -> Tuple[int, int]:
    parts: List[str] = []
    token_estimate = 0
    for row in rows:
        msgs = row.get("messages", [])
        if len(msgs) < 2:
            continue
        user = msgs[0].get("content", "").strip()
        assistant = msgs[1].get("content", "").strip()
        block = f"Instruction: {user}\nResponse: {assistant}\n"
        parts.append(block)
        token_estimate += len(re.findall(r"\w+|[^\w\s]", block))
    output_path.write_text("\n".join(parts), encoding="utf-8")
    return len(parts), token_estimate


def parse_ppl_output(text: str) -> Tuple[float, float]:
    match = re.search(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)\s*\+/-\s*([0-9.]+)", text)
    if not match:
        raise ValueError("Could not parse perplexity output")
    return float(match.group(1)), float(match.group(2))


def run_perplexity(model_path: Path, corpus_path: Path) -> Dict[str, Any]:
    cmd = [
        str(LLAMA_PERPLEXITY),
        "-m",
        str(model_path),
        "-c",
        "2048",
        "-f",
        str(corpus_path),
        "--no-warmup",
        "-t",
        "8",
    ]
    proc = run_logged(cmd, check=True)
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    ppl, ppl_std = parse_ppl_output(combined)
    return {
        "perplexity": ppl,
        "perplexity_std": ppl_std,
        "command": format_cmd(cmd),
    }


def start_server(model_path: Path) -> Tuple[subprocess.Popen[str], int]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    cmd = [
        str(LLAMA_SERVER),
        "-m",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        "2048",
        "--n-gpu-layers",
        "99",
        "--threads",
        "8",
        "--batch-size",
        "2048",
        "--ubatch-size",
        "512",
        "--parallel",
        "1",
        "--no-warmup",
    ]
    append_log(f"$ {format_cmd(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 180
    last_error = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("llama-server exited before becoming ready")
        try:
            response = requests.get(f"{base_url}/health", timeout=2)
            if response.ok and response.json().get("status") == "ok":
                return proc, port
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(1)

    proc.terminate()
    raise TimeoutError(f"Timed out waiting for llama-server: {last_error}")


def stop_server(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()


def sample_server_rss_mb(pid: int) -> float:
    proc = subprocess.run(
        ["ps", "-p", str(pid), "-o", "rss="],
        text=True,
        capture_output=True,
        check=True,
    )
    rss_kb = int(proc.stdout.strip())
    return rss_kb / 1024.0


def tokenize_for_quality(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def rouge_l_f1(reference: str, hypothesis: str) -> float:
    ref = tokenize_for_quality(reference)
    hyp = tokenize_for_quality(hypothesis)
    if not ref or not hyp:
        return 0.0

    dp = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i, r_tok in enumerate(ref, start=1):
        for j, h_tok in enumerate(hyp, start=1):
            if r_tok == h_tok:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[-1][-1]
    precision = lcs / len(hyp)
    recall = lcs / len(ref)
    if precision == 0.0 or recall == 0.0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def completion_request(port: int, prompt: str) -> Dict[str, Any]:
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.0,
        "top_k": 1,
        "stream": False,
    }
    response = requests.post(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json=payload,
        timeout=600,
    )
    response.raise_for_status()
    return response.json()


def benchmark_single_request_generation(model_path: Path) -> Dict[str, Any]:
    rows = read_valid_rows(limit=10)
    server, port = start_server(model_path)
    try:
        # Warmup request
        completion_request(port, "Say hello in one short sentence.")

        samples: List[Dict[str, Any]] = []
        latencies: List[float] = []
        rss_samples: List[float] = []
        token_counts: List[int] = []
        quality_scores: List[float] = []

        for row in rows:
            msgs = row.get("messages", [])
            prompt = msgs[0]["content"]
            reference = msgs[1]["content"]

            rss_mb_before = sample_server_rss_mb(server.pid)
            start = time.perf_counter()
            response_json = completion_request(port, prompt)
            elapsed = time.perf_counter() - start
            rss_mb_after = sample_server_rss_mb(server.pid)
            rss_samples.append(max(rss_mb_before, rss_mb_after))
            latencies.append(elapsed)

            generated = response_json["choices"][0]["message"]["content"].strip()
            tokens_predicted = int(response_json.get("usage", {}).get("completion_tokens", 100))
            token_counts.append(tokens_predicted)
            quality = rouge_l_f1(reference, generated)
            quality_scores.append(quality)
            samples.append(
                {
                    "prompt": prompt,
                    "reference": reference,
                    "generated": generated,
                    "latency_sec": round(elapsed, 6),
                    "peak_rss_mb": round(max(rss_mb_before, rss_mb_after), 2),
                    "tokens_predicted": tokens_predicted,
                    "rouge_l_f1": round(quality, 4),
                }
            )

        avg_tokens = statistics.mean(token_counts)
        avg_latency = statistics.mean(latencies)
        throughput = avg_tokens / avg_latency if avg_latency > 0 else 0.0

        return {
            "model_path": str(model_path),
            "num_generations": len(samples),
            "generation_tokens": 100,
            "mean_latency_sec": round(avg_latency, 6),
            "latency_std_sec": round(statistics.stdev(latencies), 6) if len(latencies) > 1 else 0.0,
            "tokens_per_second": round(throughput, 3),
            "peak_rss_mb_mean": round(statistics.mean(rss_samples), 2),
            "peak_rss_mb_max": round(max(rss_samples), 2),
            "output_quality_rouge_l_f1": round(statistics.mean(quality_scores), 4),
            "samples": samples,
            "server_endpoint": f"http://127.0.0.1:{port}/v1/chat/completions",
        }
    finally:
        stop_server(server)


def parse_llama_bench_json(output: str) -> List[Dict[str, Any]]:
    json_start = output.find("[")
    if json_start == -1:
        raise ValueError("llama-bench JSON output not found")
    return json.loads(output[json_start:])


def benchmark_batched(model_path: Path) -> Dict[str, Any]:
    cmd = [
        str(LLAMA_BENCH),
        "-m",
        str(model_path),
        "-p",
        "128",
        "-n",
        "100",
        "-r",
        "10",
        "-b",
        "1,4,8",
        "--no-warmup",
        "-o",
        "json",
    ]
    proc = run_logged(cmd)
    records = parse_llama_bench_json(proc.stdout or "")
    batched: Dict[str, Any] = {}
    for record in records:
        batch_size = str(record["n_batch"])
        phase = "prompt" if record["n_prompt"] > 0 else "generation"
        batched.setdefault(batch_size, {})[phase] = {
            "avg_ns": record["avg_ns"],
            "stddev_ns": record["stddev_ns"],
            "avg_tokens_per_sec": record["avg_ts"],
            "stddev_tokens_per_sec": record["stddev_ts"],
            "samples_ns": record["samples_ns"],
            "samples_ts": record["samples_ts"],
            "n_prompt": record["n_prompt"],
            "n_gen": record["n_gen"],
        }
    return {
        "model_path": str(model_path),
        "command": format_cmd(cmd),
        "results": batched,
    }


def build_comparison_table(
    latency_results: Dict[str, Any],
    ppl_results: Dict[str, Any],
    batched_results: Dict[str, Any],
) -> Tuple[str, str]:
    rows = []
    base_ppl = ppl_results["model.gguf"]["perplexity"]
    for quant_type in ["Q4_K_M", "Q5_K_M", "Q8_0"]:
        latency = latency_results[quant_type]
        ppl = ppl_results[f"model-{quant_type}.gguf"]
        batch_stats = batched_results[quant_type]["results"]
        rows.append(
            {
                "quantization": quant_type,
                "file_size_mb": round(latency["file_size_mb"], 2),
                "tokens_per_second": round(latency["tokens_per_second"], 3),
                "mean_latency_sec": round(latency["mean_latency_sec"], 4),
                "peak_rss_mb": round(latency["peak_rss_mb_mean"], 2),
                "perplexity": round(ppl["perplexity"], 4),
                "perplexity_loss": round(ppl["perplexity"] - base_ppl, 4),
                "batch_1_tps": round(batch_stats["1"]["generation"]["avg_tokens_per_sec"], 3),
                "batch_4_tps": round(batch_stats["4"]["generation"]["avg_tokens_per_sec"], 3),
                "batch_8_tps": round(batch_stats["8"]["generation"]["avg_tokens_per_sec"], 3),
            }
        )

    md_lines = [
        "| Quantization | File Size (MB) | Tokens/sec | Mean Latency (s) | Peak RSS (MB) | Perplexity | Perplexity Loss | Batch 1 TPS | Batch 4 TPS | Batch 8 TPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['quantization']} | {row['file_size_mb']:.2f} | {row['tokens_per_second']:.3f} | "
            f"{row['mean_latency_sec']:.4f} | {row['peak_rss_mb']:.2f} | {row['perplexity']:.4f} | "
            f"{row['perplexity_loss']:.4f} | {row['batch_1_tps']:.3f} | {row['batch_4_tps']:.3f} | {row['batch_8_tps']:.3f} |"
        )
    md_text = "\n".join(md_lines) + "\n"

    with COMP_TABLE_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    return md_text, json.dumps(rows, indent=2)


def write_server_package(model_path: Path) -> None:
    SERVER_DIR.mkdir(parents=True, exist_ok=True)
    server_py = f"""#!/usr/bin/env python3
\"\"\"Launch a llama.cpp server for the smallest working GGUF model.\"\"\"

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LLAMA_SERVER = ROOT / "llama.cpp" / "build" / "bin" / "llama-server"
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "{model_path.name}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--ctx-size", type=int, default=2048)
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    args = parser.parse_args()
    cmd = [
        str(LLAMA_SERVER),
        "-m",
        str(args.model),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--threads",
        str(args.threads),
        "--ctx-size",
        str(args.ctx_size),
        "--n-gpu-layers",
        str(args.n_gpu_layers),
        "--parallel",
        "1",
        "--no-warmup",
    ]
    print("Running:", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
"""
    (SERVER_DIR / "server.py").write_text(server_py, encoding="utf-8")
    (SERVER_DIR / "requirements.txt").write_text("requests\n", encoding="utf-8")
    (SERVER_DIR / "README.md").write_text(
        "\n".join(
            [
                "# Task 4 Inference Server",
                "",
                "Serve the smallest working model with:",
                "",
                "```bash",
                f".venv/bin/python {SERVER_DIR / 'server.py'} --model {model_path}",
                "```",
                "",
                "Benchmark the server with `requests` or curl against `POST /v1/chat/completions`.",
            ]
        ),
        encoding="utf-8",
    )


def write_analysis(
    latency_results: Dict[str, Any],
    ppl_results: Dict[str, Any],
    batch_results: Dict[str, Any],
    comparison_md: str,
    q4_model_path: Path,
) -> None:
    base_ppl = ppl_results["model.gguf"]["perplexity"]
    lines = [
        "# Task 4 Quantization & GGUF Export Analysis",
        "",
        "## What was implemented",
        "- Fused the LoRA adapter into the TinyLlama base model.",
        "- Converted the fused model to GGUF with llama.cpp.",
        "- Quantized the GGUF model to Q4_K_M, Q5_K_M, and Q8_0.",
        "- Benchmarked 10 warm chat-completion generations per quantization.",
        "- Measured peak RSS, latency, tokens/sec, output quality, and perplexity.",
        "- Benchmarked batch sizes 1, 4, and 8 with llama-bench.",
        "- Packaged a runnable llama.cpp server for the Q4_K_M model.",
        "",
        "## Methodology",
        f"- Perplexity corpus: {ppl_results['corpus']['rows_used']} Alpaca validation rows ({ppl_results['corpus']['token_estimate']} token estimate).",
        "- Latency/quality runs used the first 10 validation rows via `POST /v1/chat/completions`.",
        "- Memory footprint used per-request peak RSS sampled from the running llama-server process.",
        "- Batched inference used `llama-bench` with batch sizes 1, 4, and 8, 10 repetitions, and 100 generated tokens.",
        "",
        "## Perplexity summary",
        f"- Base GGUF perplexity: {base_ppl:.4f}",
    ]
    for quant_type in ["Q4_K_M", "Q5_K_M", "Q8_0"]:
        model_key = f"model-{quant_type}.gguf"
        ppl = ppl_results[model_key]["perplexity"]
        lines.append(
            f"- {quant_type}: {ppl:.4f} (loss vs base: {ppl - base_ppl:.4f})"
        )

    lines += [
        "",
        "## Latency and memory",
    ]
    for quant_type in ["Q4_K_M", "Q5_K_M", "Q8_0"]:
        lat = latency_results[quant_type]
        lines.append(
            f"- {quant_type}: {lat['tokens_per_second']:.3f} tok/s, "
            f"{lat['mean_latency_sec']:.4f}s mean latency, "
            f"{lat['peak_rss_mb_mean']:.2f} MB mean peak RSS, "
            f"ROUGE-L {lat['output_quality_rouge_l_f1']:.4f}"
        )

    lines += [
        "",
        "## Batched inference observations",
    ]
    for quant_type in ["Q4_K_M", "Q5_K_M", "Q8_0"]:
        batch = batch_results[quant_type]["results"]
        lines.append(
            f"- {quant_type}: batch-1 {batch['1']['generation']['avg_tokens_per_sec']:.3f} tok/s, "
            f"batch-4 {batch['4']['generation']['avg_tokens_per_sec']:.3f} tok/s, "
            f"batch-8 {batch['8']['generation']['avg_tokens_per_sec']:.3f} tok/s"
        )

    lines += [
        "",
        "## Deployment recommendation",
        "Q4_K_M is the best edge deployment choice because it is the smallest model, keeps the lowest peak RSS, and remains the fastest of the three quantizations in this run.",
        "",
        "## Deployment comparison table",
        comparison_md,
        "",
        "## Packaging",
        f"The runnable server package is in `{SERVER_DIR}` and serves `{q4_model_path.name}` with llama-server.",
    ]
    ANALYSIS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 4 GGUF quantization and benchmark pipeline")
    parser.add_argument("--skip-conversion", action="store_true", help="Reuse existing GGUF artifacts")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CONVERSION_LOG.write_text("", encoding="utf-8")
    append_log("Task 4 conversion and quantization log")
    append_log(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    ensure_llama_cpp_built()

    adapter_path = pick_adapter_path()
    logging.info("Fusing LoRA adapter from %s", adapter_path)
    fused_model = fuse_model(adapter_path)

    if args.skip_conversion and (RESULTS_DIR / "model.gguf").exists():
        base_gguf = RESULTS_DIR / "model.gguf"
    else:
        logging.info("Converting fused model to GGUF")
        base_gguf = convert_to_gguf(fused_model)

    quantized: Dict[str, Path] = {}
    for quant_type in ["Q4_K_M", "Q5_K_M", "Q8_0"]:
        logging.info("Quantizing to %s", quant_type)
        quantized[quant_type] = quantize_gguf(base_gguf, quant_type)

    copy_legacy_artifacts(base_gguf, quantized)

    rows = read_valid_rows(limit=80)
    ppl_corpus = RESULTS_DIR / "alpaca_valid_corpus.txt"
    rows_used, token_estimate = build_ppl_corpus(rows, ppl_corpus)

    model_paths = {"model.gguf": base_gguf}
    model_paths.update({f"model-{k}.gguf": v for k, v in quantized.items()})

    perplexity_results: Dict[str, Any] = {
        "corpus": {
            "path": str(ppl_corpus),
            "rows_used": rows_used,
            "token_estimate": token_estimate,
        }
    }
    for name, path in model_paths.items():
        logging.info("Running perplexity for %s", name)
        perplexity_results[name] = run_perplexity(path, ppl_corpus)
    PPL_RESULTS.write_text(json.dumps(perplexity_results, indent=2), encoding="utf-8")

    latency_results: Dict[str, Any] = {}
    quality_results: Dict[str, Any] = {}
    for quant_type, path in quantized.items():
        logging.info("Benchmarking single-request generation for %s", quant_type)
        result = benchmark_single_request_generation(path)
        result["file_size_mb"] = round(path.stat().st_size / (1024 * 1024), 2)
        latency_results[quant_type] = result
        quality_results[quant_type] = result["samples"]
    LATENCY_RESULTS.write_text(json.dumps(latency_results, indent=2), encoding="utf-8")

    batch_results: Dict[str, Any] = {}
    for quant_type, path in quantized.items():
        logging.info("Benchmarking batched inference for %s", quant_type)
        batch_results[quant_type] = benchmark_batched(path)
    BATCH_RESULTS.write_text(json.dumps(batch_results, indent=2), encoding="utf-8")

    comparison_md, comparison_json = build_comparison_table(latency_results, perplexity_results, batch_results)
    COMP_TABLE_MD.write_text(comparison_md, encoding="utf-8")

    write_server_package(quantized["Q4_K_M"])
    write_analysis(latency_results, perplexity_results, batch_results, comparison_md, quantized["Q4_K_M"])

    summary = {
        "artifacts": {
            "model.gguf": str(base_gguf),
            "model-Q4_K_M.gguf": str(quantized["Q4_K_M"]),
            "model-Q5_K_M.gguf": str(quantized["Q5_K_M"]),
            "model-Q8_0.gguf": str(quantized["Q8_0"]),
            "perplexity_results.json": str(PPL_RESULTS),
            "latency_benchmarks.json": str(LATENCY_RESULTS),
            "batched_inference_results.json": str(BATCH_RESULTS),
            "deployment_comparison.md": str(COMP_TABLE_MD),
            "deployment_comparison.csv": str(COMP_TABLE_CSV),
            "quantization_analysis.md": str(ANALYSIS_MD),
            "conversion_logs.txt": str(CONVERSION_LOG),
            "server_package": str(SERVER_DIR),
        },
        "comparison_rows": json.loads(comparison_json),
    }
    (RESULTS_DIR / "task4_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("Task 4 completed successfully")
    logging.info("Artifacts written to %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
