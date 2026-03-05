"""Prepare training data from Alpaca — keep only examples whose answers contain
at least MIN_ANSWER_WORDS words (removes terse/one-word answers that regress
verbosity), then select the top NUM_EXAMPLES by answer length."""

import argparse
import json
import random
import time
from datasets import load_dataset
from pathlib import Path


def _ts() -> str:
    """Return a short HH:MM:SS timestamp for progress lines."""
    return time.strftime("%H:%M:%S")

# Defaults (overridable via CLI flags)
DATASET_NAME = "tatsu-lab/alpaca"
NUM_EXAMPLES = 5000
OUTPUT_DIR = Path("./data")
TRAIN_SPLIT = 0.8
VALID_SPLIT = 0.1
TEST_SPLIT = 0.1
# Hard floor: only keep examples whose answers are at least this many words.
# Eliminates one-word and terse answers (e.g. "Sang.", "larger", "Fiction.")
# that caused FT models to regress TinyLlama Chat's RLHF-trained verbosity.
# ~24k of Alpaca's 52k examples pass this threshold, giving plenty of headroom.
MIN_ANSWER_WORDS = 30


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Alpaca training data.")
    parser.add_argument("--num-examples", type=int, default=NUM_EXAMPLES,
                        help=f"Number of top examples to select by answer length (default: {NUM_EXAMPLES})")
    parser.add_argument("--min-answer-words", type=int, default=MIN_ANSWER_WORDS,
                        help=f"Hard floor on answer word count (default: {MIN_ANSWER_WORDS})")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR,
                        help=f"Output directory for JSONL files (default: {OUTPUT_DIR})")
    return parser.parse_args()


def format_chat(example):
    """Convert Alpaca example to MLX-LM chat format."""
    instruction = example["instruction"]
    inp = example.get("input", "")
    output = example["output"]

    if inp:
        user_content = f"{instruction}\n\nInput:\n{inp}"
    else:
        user_content = instruction

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]
    }


def answer_word_count(example):
    return len(example["output"].split())


def main():
    wall_start = time.monotonic()
    args = parse_args()
    num_examples = args.num_examples
    min_answer_words = args.min_answer_words
    output_dir = args.out_dir

    print(f"[{_ts()}] PREPARE  target={num_examples} examples, min_words={min_answer_words}, out={output_dir}")

    # Step 1 — download/cache dataset
    t0 = time.monotonic()
    print(f"[{_ts()}] Step 1/5  Downloading {DATASET_NAME} from HuggingFace Hub (cached after first run)…")
    ds = load_dataset(DATASET_NAME, split="train")
    print(f"[{_ts()}]           Loaded {len(ds):,} examples  ({time.monotonic()-t0:.1f}s)")

    # Step 2 — filter
    t0 = time.monotonic()
    print(f"[{_ts()}] Step 2/5  Filtering: remove empty answers, apply ≥{min_answer_words}-word floor…")
    ds_filtered = [ex for ex in ds if len(ex["output"].strip()) > 0]
    ds_filtered = [ex for ex in ds_filtered if answer_word_count(ex) >= min_answer_words]
    print(f"[{_ts()}]           {len(ds_filtered):,} examples pass filter  ({time.monotonic()-t0:.1f}s)")

    # Step 3 — sort and select
    t0 = time.monotonic()
    print(f"[{_ts()}] Step 3/5  Sorting by answer length, selecting top {num_examples:,}…")
    ds_sorted = sorted(ds_filtered, key=answer_word_count, reverse=True)
    selected = ds_sorted[:num_examples]
    min_wc = answer_word_count(selected[-1])
    max_wc = answer_word_count(selected[0])
    avg_wc = sum(answer_word_count(ex) for ex in selected) / len(selected)
    print(f"[{_ts()}]           Selected {len(selected):,}  word-count: min={min_wc} avg={avg_wc:.0f} max={max_wc}  ({time.monotonic()-t0:.1f}s)")

    # Step 4 — format, shuffle, split
    t0 = time.monotonic()
    print(f"[{_ts()}] Step 4/5  Formatting to ChatML JSONL, shuffling, splitting 80/10/10…")
    samples = [format_chat(ex) for ex in selected]
    random.seed(42)
    random.shuffle(samples)
    n = len(samples)
    n_train = int(n * TRAIN_SPLIT)
    n_valid = int(n * VALID_SPLIT)
    train_data = samples[:n_train]
    valid_data = samples[n_train : n_train + n_valid]
    test_data  = samples[n_train + n_valid :]
    print(f"[{_ts()}]           train={len(train_data):,}  valid={len(valid_data):,}  test={len(test_data):,}  ({time.monotonic()-t0:.1f}s)")

    # Step 5 — save
    t0 = time.monotonic()
    print(f"[{_ts()}] Step 5/5  Writing JSONL files to {output_dir}/…")
    backup_dir = output_dir / "backup_v1"
    if not backup_dir.exists():
        backup_dir.mkdir(parents=True)
        for f in ["train.jsonl", "valid.jsonl", "test.jsonl"]:
            src = output_dir / f
            if src.exists():
                src.rename(backup_dir / f)
                print(f"[{_ts()}]           Backed up {f} → {backup_dir / f}")
    output_dir.mkdir(exist_ok=True)
    save_jsonl(train_data, output_dir / "train.jsonl")
    save_jsonl(valid_data, output_dir / "valid.jsonl")
    save_jsonl(test_data,  output_dir / "test.jsonl")
    print(f"[{_ts()}]           Wrote train/valid/test.jsonl  ({time.monotonic()-t0:.1f}s)")

    ans_lens = sorted(len(s["messages"][1]["content"].split()) for s in train_data)
    n = len(ans_lens)
    total_elapsed = time.monotonic() - wall_start
    print(f"[{_ts()}] PREPARE done in {total_elapsed:.1f}s")
    print(f"           Answer length (train): min={ans_lens[0]}  median={ans_lens[n//2]}  mean={sum(ans_lens)/n:.0f}  max={ans_lens[-1]}")
    print(f"           ≥50 words: {sum(1 for l in ans_lens if l >= 50):,} / {n:,} ({100*sum(1 for l in ans_lens if l >= 50)/n:.0f}%)")


def save_jsonl(data, path):
    with open(path, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
