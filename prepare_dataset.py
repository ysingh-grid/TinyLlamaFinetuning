
import json
import os
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

def format_chat(example):
    """
    Convert Alpaca example to MLX-LM chat format:
    {
      "messages": [
        {"role": "user", "content": "instruction + input"},
        {"role": "assistant", "content": "output"}
      ]
    }
    """
    instruction = example['instruction']
    inp = example.get('input', '')
    output = example['output']

    if inp:
        user_content = f"{instruction}\n\nInput:\n{inp}"
    else:
        user_content = instruction

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output}
        ]
    }

def main():
    print(f"Loading {NUM_EXAMPLES} examples from {DATASET_NAME}...")
    # Load dataset
    ds = load_dataset(DATASET_NAME, split=f"train[:{NUM_EXAMPLES}]")
    
    # Format
    print("Formatting to chat JSONL...")
    samples = [format_chat(ex) for ex in ds]
    
    # Shuffle (optional but good practice)
    random.seed(42)
    random.shuffle(samples)
    
    # Split
    n = len(samples)
    n_train = int(n * TRAIN_SPLIT)
    n_valid = int(n * VALID_SPLIT)
    
    train_data = samples[:n_train]
    valid_data = samples[n_train:n_train+n_valid]
    test_data = samples[n_train+n_valid:]
    
    # Ensure output dir exists
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Save
    print(f"Saving to {OUTPUT_DIR}...")
    save_jsonl(train_data, OUTPUT_DIR / "train.jsonl")
    save_jsonl(valid_data, OUTPUT_DIR / "valid.jsonl")
    save_jsonl(test_data, OUTPUT_DIR / "test.jsonl")
    
    print("Done!")
    print(f"Train: {len(train_data)}")
    print(f"Valid: {len(valid_data)}")
    print(f"Test:  {len(test_data)}")

def save_jsonl(data, path):
    with open(path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

if __name__ == "__main__":
    main()
