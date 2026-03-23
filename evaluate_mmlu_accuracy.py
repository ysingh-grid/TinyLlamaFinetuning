#!/usr/bin/env python3
"""
MMLU-Style Accuracy Evaluation for Task 1 Models

Evaluates trained LoRA adapters using genuine MMLU-style multiple-choice scoring:
- Each test question is presented with 4 answer choices (A/B/C/D)
- Correct answer is one of the choices; 3 distractors are sampled from other test answers
- Model must output the letter of the correct choice (A, B, C, or D)
- Primary metric: MCQ accuracy = (correct letter predictions / total questions) * 100
"""
import argparse
import csv
import json
import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

from mlx_lm import load, generate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_DIR = Path("./data")
RESULTS_DIR = Path("./results/task1")

_LETTERS = ["A", "B", "C", "D"]
_LETTER_RE = re.compile(r"\b([A-D])\b")


def _load_test_rows(test_file: Path) -> List[Tuple[str, str]]:
    """Return list of (instruction, answer) pairs from the test split."""
    pairs: List[Tuple[str, str]] = []
    with test_file.open() as f:
        for line in f:
            row = json.loads(line)
            msgs = row.get("messages", [])
            instruction = next((m["content"] for m in msgs if m["role"] == "user"), None)
            answer = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
            if instruction and answer:
                pairs.append((instruction, answer))
    return pairs


def _build_mcq_prompt(tokenizer, instruction: str, choices: List[str]) -> str:
    """Format a 4-choice MCQ prompt from an instruction and shuffled answer choices."""
    choice_block = "\n".join(
        f"{letter}. {choice[:120].strip()}"
        for letter, choice in zip(_LETTERS, choices)
    )
    user_content = (
        f"Instruction: {instruction}\n\n"
        f"Which of the following responses best follows the instruction?\n\n"
        f"{choice_block}\n\n"
        f"Answer with only the letter (A, B, C, or D):"
    )
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return user_content + "\nAssistant: "


def _extract_letter(response: str) -> str:
    """Return the first A/B/C/D letter found in the model's response, else ''."""
    cleaned = response.strip()
    if cleaned and cleaned[0].upper() in _LETTERS:
        return cleaned[0].upper()
    m = _LETTER_RE.search(cleaned.upper())
    return m.group(1) if m else ""


def evaluate_accuracy(
    adapter_path: str,
    num_samples: int = 100,
    similarity_threshold: float = 0.2,   # kept for API compatibility, unused in MCQ
    max_tokens: int = 10,
) -> Dict:
    """
    MMLU-style multiple-choice accuracy evaluation.

    For each of `num_samples` test instructions:
      1. The correct assistant answer is placed at a random choice position (A/B/C/D).
      2. Three distractor answers are sampled from other test examples.
      3. The model must output the letter of the correct choice.

    Returns dict with keys:
      accuracy        – MCQ letter accuracy (0-100 %)   ← primary metric
      exact_match     – same as accuracy for MCQ
      fuzzy_match     – same as accuracy for MCQ
      avg_token_f1    – 0.0 (not applicable to MCQ)
      total_evaluated – number of questions attempted
    """
    logging.info("Loading model with adapter: %s", adapter_path)
    model, tokenizer = load(BASE_MODEL, adapter_path=adapter_path)

    test_file = DATA_DIR / "test.jsonl"
    all_pairs = _load_test_rows(test_file)
    if not all_pairs:
        logging.error("No test pairs found in %s", test_file)
        return {"accuracy": 0.0, "exact_match": 0.0, "fuzzy_match": 0.0, "avg_token_f1": 0.0, "total_evaluated": 0}

    eval_pairs = all_pairs[:num_samples]
    answer_pool = [ans for _, ans in all_pairs]

    correct = 0
    total = 0

    for idx, (instruction, correct_answer) in enumerate(eval_pairs):
        rng = random.Random(idx)
        # Sample 3 distractors that are different from the correct answer
        distractors = []
        candidates = [a for i, a in enumerate(answer_pool) if i != idx]
        rng.shuffle(candidates)
        for candidate in candidates:
            if candidate.strip() != correct_answer.strip():
                distractors.append(candidate)
            if len(distractors) == 3:
                break
        # Pad with truncated versions of the correct answer if pool is tiny
        while len(distractors) < 3:
            distractors.append(correct_answer[:80] + "...")

        choices = distractors + [correct_answer]
        rng.shuffle(choices)
        correct_letter = _LETTERS[choices.index(correct_answer)]

        prompt = _build_mcq_prompt(tokenizer, instruction, choices)
        try:
            response = generate(model, tokenizer, prompt, max_tokens=max_tokens, verbose=False)
        except Exception as exc:
            logging.warning("Generation failed for sample %d: %s", idx, exc)
            total += 1
            continue

        predicted_letter = _extract_letter(response)
        if predicted_letter == correct_letter:
            correct += 1
        total += 1

        if total % 10 == 0:
            logging.info("MCQ progress: %d/%d  running accuracy=%.1f%%", total, num_samples, correct / total * 100)

    if total == 0:
        return {"accuracy": 0.0, "exact_match": 0.0, "fuzzy_match": 0.0, "avg_token_f1": 0.0, "total_evaluated": 0}

    mcq_accuracy = (correct / total) * 100.0
    logging.info("MCQ evaluation complete: %d/%d correct → %.2f%%", correct, total, mcq_accuracy)
    return {
        "accuracy":        mcq_accuracy,
        "exact_match":     mcq_accuracy,   # letter match is exact
        "fuzzy_match":     mcq_accuracy,   # no partial credit in MCQ
        "avg_token_f1":    0.0,            # N/A for multiple-choice
        "total_evaluated": total,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate MMLU-style accuracy for Task 1 models")
    parser.add_argument("--num-samples", type=int, default=100, 
                        help="Number of test samples to evaluate (default: 100)")
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="Similarity threshold for fuzzy matching (default: 0.2)")
    parser.add_argument("--max-tokens", type=int, default=32,
                        help="Max generation tokens per sample (default: 32)")
    parser.add_argument("--csv-out", type=str, default=str(RESULTS_DIR / "accuracy_results.csv"),
                        help="Write per-rank accuracy summary to CSV")
    args = parser.parse_args()
    
    ranks = [4, 8, 16, 32, 64]
    all_accuracies = {}
    
    for rank in ranks:
        adapter_path = RESULTS_DIR / f"adapter_r{rank}"
        if not adapter_path.exists():
            logging.warning(f"Adapter not found: {adapter_path}, skipping")
            continue
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Evaluating Rank {rank}")
        logging.info(f"{'='*60}")
        
        accuracy_results = evaluate_accuracy(
            str(adapter_path), 
            num_samples=args.num_samples,
            similarity_threshold=args.threshold,
            max_tokens=args.max_tokens,
        )
        
        all_accuracies[rank] = accuracy_results
        
        logging.info(f"Rank {rank} Results:")
        logging.info(f"  Accuracy: {accuracy_results['accuracy']:.2f}%")
        logging.info(f"  Exact Match: {accuracy_results['exact_match']:.2f}%")
        logging.info(f"  Fuzzy Match: {accuracy_results['fuzzy_match']:.2f}%")
    
    # Update comparison matrix
    output_file = RESULTS_DIR / "accuracy_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_accuracies, f, indent=2)
    logging.info(f"\nSaved accuracy results to {output_file}")

    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["rank", "accuracy", "exact_match", "fuzzy_match", "avg_token_f1", "total_evaluated"],
        )
        writer.writeheader()
        for rank, results in sorted(all_accuracies.items()):
            writer.writerow({
                "rank": rank,
                "accuracy": round(results["accuracy"], 4),
                "exact_match": round(results["exact_match"], 4),
                "fuzzy_match": round(results["fuzzy_match"], 4),
                "avg_token_f1": round(results["avg_token_f1"], 4),
                "total_evaluated": results["total_evaluated"],
            })
    logging.info(f"Saved accuracy CSV to {csv_path}")
    
    # Update existing comparison matrix
    comparison_file = RESULTS_DIR / "comparison_matrix.json"
    if comparison_file.exists():
        with open(comparison_file, 'r') as f:
            comparison = json.load(f)
        
        # Add accuracy to each entry
        for entry in comparison:
            rank = entry['rank']
            if rank in all_accuracies:
                entry['accuracy'] = all_accuracies[rank]['accuracy']
                entry['exact_match'] = all_accuracies[rank]['exact_match']
                entry['fuzzy_match'] = all_accuracies[rank]['fuzzy_match']
                entry['avg_token_f1'] = all_accuracies[rank]['avg_token_f1']

        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        logging.info(f"Updated {comparison_file} with accuracy metrics")
    
    # Print summary table
    print("\n" + "="*70)
    print("MMLU-Style Accuracy Summary")
    print("="*70)
    print(f"{'Rank':<8} {'Accuracy':<12} {'Exact Match':<15} {'Samples':<10}")
    print("-"*70)
    for rank, results in sorted(all_accuracies.items()):
        print(f"{rank:<8} {results['accuracy']:>10.2f}% {results['exact_match']:>13.2f}% {results['total_evaluated']:>8}")
    print("="*70)


if __name__ == "__main__":
    main()
