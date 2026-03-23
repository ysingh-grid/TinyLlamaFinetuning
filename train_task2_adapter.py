#!/usr/bin/env python3
"""
Train Task 2 LoRA Adapter on 10k Alpaca Examples

Trains a dedicated LoRA adapter for Task 2 activation steering:
- 10k Alpaca training examples
- LoRA rank 16, alpha 32
- ~5000 iterations (2+ epochs)
- Saves to adapters/tinyllama-lora-alpaca-10k/
"""
import logging
import subprocess
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Training configuration
MODEL = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
DATA_DIR = "./data/task2"
ADAPTER_DIR = "./adapters/tinyllama-lora-alpaca-10k"
TRAIN_FILE = f"{DATA_DIR}/alpaca_10k_train.jsonl"
VAL_FILE = f"{DATA_DIR}/alpaca_10k_val.jsonl"

# LoRA config
LORA_RANK = 16
LORA_ALPHA = 32  # alpha = rank * 2
LORA_LAYERS = 16  # Apply to first 16 layers

# Training hyperparameters
BATCH_SIZE = 2
ITERS = 5000  # 5000 iters * batch 2 = 10k samples = 1.25 epochs
LEARNING_RATE = 1e-4
VAL_BATCHES = 25


def verify_data_exists():
    """Verify training data is prepared."""
    train_path = Path(TRAIN_FILE)
    val_path = Path(VAL_FILE)
    
    if not train_path.exists():
        logging.error(f"Training data not found: {train_path}")
        logging.error("Run: python prepare_task2_dataset.py first")
        return False
    
    if not val_path.exists():
        logging.error(f"Validation data not found: {val_path}")
        return False
    
    # Count lines
    train_count = sum(1 for _ in open(train_path))
    val_count = sum(1 for _ in open(val_path))
    
    logging.info(f"Found training data:")
    logging.info(f"  Train: {train_count} examples")
    logging.info(f"  Val:   {val_count} examples")
    
    if train_count < 8000:
        logging.warning(f"Expected ~8000 train examples, found {train_count}")
    
    return True


def train_adapter():
    """Train LoRA adapter using MLX-LM."""
    logging.info("="*70)
    logging.info("Training Task 2 LoRA Adapter (10k Alpaca)")
    logging.info("="*70)
    
    logging.info(f"\nConfiguration:")
    logging.info(f"  Model:         {MODEL}")
    logging.info(f"  LoRA rank:     {LORA_RANK}")
    logging.info(f"  LoRA alpha:    {LORA_ALPHA}")
    logging.info(f"  LoRA layers:   {LORA_LAYERS}")
    logging.info(f"  Batch size:    {BATCH_SIZE}")
    logging.info(f"  Iterations:    {ITERS}")
    logging.info(f"  Learning rate: {LEARNING_RATE}")
    logging.info(f"  Train file:    {TRAIN_FILE}")
    logging.info(f"  Val file:      {VAL_FILE}")
    logging.info(f"  Output:        {ADAPTER_DIR}")
    
    # Build command
    cmd = [
        ".venv/bin/python", "-m", "mlx_lm.lora",
        "--model", MODEL,
        "--train",
        "--data", DATA_DIR,
        "--adapter-path", ADAPTER_DIR,
        "--iters", str(ITERS),
        "--batch-size", str(BATCH_SIZE),
        "--lora-layers", str(LORA_LAYERS),
        "--learning-rate", str(LEARNING_RATE),
        "--val-batches", str(VAL_BATCHES),
        "--save-every", "500",
        "--test-batches", "10",
        # LoRA config
        "--lora-parameters", json.dumps({
            "rank": LORA_RANK,
            "alpha": LORA_ALPHA,
            "dropout": 0.05,
            "scale": LORA_ALPHA / LORA_RANK,  # 2.0
        })
    ]
    
    logging.info(f"\nStarting training...")
    logging.info(f"Estimated time: ~60-90 minutes")
    logging.info(f"(Training {ITERS} iterations on 8000 examples = 1.25 epochs)\n")
    
    start_time = time.time()
    
    try:
        # Run training
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        
        elapsed = time.time() - start_time
        logging.info(f"\n✓ Training completed in {elapsed/60:.1f} minutes")
        logging.info(f"  Adapter saved to: {ADAPTER_DIR}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Training failed: {e}")
        return False
    except KeyboardInterrupt:
        logging.warning("\nTraining interrupted by user")
        return False


def verify_adapter():
    """Verify adapter was created successfully."""
    adapter_path = Path(ADAPTER_DIR)
    
    if not adapter_path.exists():
        logging.error(f"Adapter directory not found: {adapter_path}")
        return False
    
    # Check for key files
    required_files = [
        "adapters.safetensors",
        "adapter_config.json",
    ]
    
    all_good = True
    for fname in required_files:
        fpath = adapter_path / fname
        if fpath.exists():
            size = fpath.stat().st_size / 1024 / 1024  # MB
            logging.info(f"  ✓ {fname} ({size:.1f} MB)")
        else:
            logging.error(f"  ✗ {fname} MISSING")
            all_good = False
    
    return all_good


# Need json for lora params
import json


def main():
    # Verify data
    if not verify_data_exists():
        return 1
    
    # Train adapter
    if not train_adapter():
        return 1
    
    # Verify output
    logging.info("\nVerifying adapter files...")
    if verify_adapter():
        logging.info("\n✓ Training complete and verified!")
        logging.info(f"\nNext step:")
        logging.info(f"  python task2_activation_steering_complete.py")
    else:
        logging.error("\n✗ Adapter verification failed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
