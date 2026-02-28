"""Prepare training data from Alpaca — select the 5000 examples with the LONGEST
answers so the model learns to produce verbose, detailed responses."""

import json
import random
from datasets import load_dataset
from pathlib import Path

# Config
DATASET_NAME = "tatsu-lab/alpaca"
NUM_EXAMPLES = 5000
OUTPUT_DIR = Path("./data")
TRAIN_SPLIT = 0.8
VALID_SPLIT = 0.1
TEST_SPLIT = 0.1
MIN_ANSWER_WORDS = 0  # no hard floor — we just take the top N by length


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
    print(f"Loading FULL dataset from {DATASET_NAME}...")
    ds = load_dataset(DATASET_NAME, split="train")
    print(f"Total examples in Alpaca: {len(ds)}")

    # Filter out empty answers
    ds_filtered = [ex for ex in ds if len(ex["output"].strip()) > 0]
    print(f"After removing empty answers: {len(ds_filtered)}")

    # Sort by answer length (longest first)
    ds_sorted = sorted(ds_filtered, key=answer_word_count, reverse=True)

    # Take top N
    selected = ds_sorted[:NUM_EXAMPLES]

    min_wc = answer_word_count(selected[-1])
    max_wc = answer_word_count(selected[0])
    avg_wc = sum(answer_word_count(ex) for ex in selected) / len(selected)
    print(f"Selected {len(selected)} longest examples:")
    print(f"  Word count range: {min_wc} - {max_wc}")
    print(f"  Average answer length: {avg_wc:.1f} words")

    # Format to chat
    print("Formatting to chat JSONL...")
    samples = [format_chat(ex) for ex in selected]

    # Shuffle
    random.seed(42)
    random.shuffle(samples)

    # Split
    n = len(samples)
    n_train = int(n * TRAIN_SPLIT)
    n_valid = int(n * VALID_SPLIT)

    train_data = samples[:n_train]
    valid_data = samples[n_train : n_train + n_valid]
    test_data = samples[n_train + n_valid :]

    # Back up old data
    backup_dir = OUTPUT_DIR / "backup_v1"
    if not backup_dir.exists():
        backup_dir.mkdir(parents=True)
        for f in ["train.jsonl", "valid.jsonl", "test.jsonl"]:
            src = OUTPUT_DIR / f
            if src.exists():
                src.rename(backup_dir / f)
                print(f"Backed up {f} -> {backup_dir / f}")

    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Saving to {OUTPUT_DIR}...")
    save_jsonl(train_data, OUTPUT_DIR / "train.jsonl")
    save_jsonl(valid_data, OUTPUT_DIR / "valid.jsonl")
    save_jsonl(test_data, OUTPUT_DIR / "test.jsonl")

    print("Done!")
    print(f"Train: {len(train_data)}")
    print(f"Valid: {len(valid_data)}")
    print(f"Test:  {len(test_data)}")

    # Print distribution
    ans_lens = [len(s["messages"][1]["content"].split()) for s in train_data]
    ans_lens.sort()
    n = len(ans_lens)
    print(f"\nNew training answer length stats:")
    print(f"  min={ans_lens[0]}  median={ans_lens[n//2]}  mean={sum(ans_lens)/n:.0f}  max={ans_lens[-1]}")
    print(f"  <30 words: {sum(1 for l in ans_lens if l < 30)} ({100*sum(1 for l in ans_lens if l < 30)/n:.0f}%)")
    print(f"  >=50 words: {sum(1 for l in ans_lens if l >= 50)} ({100*sum(1 for l in ans_lens if l >= 50)/n:.0f}%)")


def save_jsonl(data, path):
    with open(path, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
