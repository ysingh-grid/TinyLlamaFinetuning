#!/usr/bin/env python3
import argparse
import json
import re
from itertools import combinations
from pathlib import Path


def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)

    # Lazy NLTK imports/downloads so the script can run standalone.
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    processed_tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stop_words]
    return " ".join(processed_tokens)


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank models by cosine similarity against reference answers."
    )
    parser.add_argument("--responses-dir", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer, util

    references_raw = read_jsonl(args.references)
    references = {row["prompt_id"]: row["reference"] for row in references_raw}

    model_responses = {}
    for file in args.responses_dir.glob("*.jsonl"):
        model_name = file.stem
        lines = read_jsonl(file)
        model_responses[model_name] = {
            row["prompt_id"]: row["response"]
            for row in lines
            if "prompt_id" in row and "response" in row
        }

    models = list(model_responses.keys())
    print(f"Loaded models: {models}")

    st_model = SentenceTransformer(args.model)

    common_prompts = set(references.keys())
    for responses in model_responses.values():
        common_prompts &= set(responses.keys())
    common_prompts = sorted(common_prompts)
    print(f"Shared prompts: {len(common_prompts)}")

    sims = {prompt_id: {} for prompt_id in common_prompts}
    for idx, prompt_id in enumerate(common_prompts, start=1):
        ref_text = preprocess_text(references[prompt_id])
        ref_emb = st_model.encode(ref_text, convert_to_tensor=True)
        for model in models:
            resp_text = preprocess_text(model_responses[model][prompt_id])
            resp_emb = st_model.encode(resp_text, convert_to_tensor=True)
            sims[prompt_id][model] = util.cos_sim(ref_emb, resp_emb).item()
        if idx % 10 == 0:
            print(f"Scored {idx}/{len(common_prompts)} prompts")

    model_sum_sim = {model: 0.0 for model in models}
    model_sum_rank = {model: 0 for model in models}
    pairwise_wins = {(a, b): 0 for a in models for b in models if a != b}
    pairwise_ties = {tuple(sorted([a, b])): 0 for a in models for b in models if a != b}

    for prompt_id in common_prompts:
        prompt_sims = sims[prompt_id]
        sorted_models = sorted(models, key=lambda m: prompt_sims[m], reverse=True)

        for rank, model in enumerate(sorted_models, start=1):
            model_sum_sim[model] += prompt_sims[model]
            model_sum_rank[model] += rank

        for m1, m2 in combinations(models, 2):
            s1 = prompt_sims[m1]
            s2 = prompt_sims[m2]
            if s1 > s2 + 1e-5:
                pairwise_wins[(m1, m2)] += 1
            elif s2 > s1 + 1e-5:
                pairwise_wins[(m2, m1)] += 1
            else:
                pairwise_ties[tuple(sorted([m1, m2]))] += 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    lines = ["# Similarity & Ranking Report (All vs All)", "", "## Overall Metrics"]
    lines.append(f"Tested on {len(common_prompts)} prompts.")
    lines.append("")
    lines.append("| Model | Avg Sim | Avg Rank | Pairwise Matches | Win Rate (vs All) |")
    lines.append("|---|---:|---:|---:|---:|")

    overall = []
    for model in models:
        avg_sim = model_sum_sim[model] / len(common_prompts)
        avg_rank = model_sum_rank[model] / len(common_prompts)
        wins = sum(pairwise_wins[(model, other)] for other in models if other != model)
        losses = sum(pairwise_wins[(other, model)] for other in models if other != model)
        ties = sum(pairwise_ties[tuple(sorted([model, other]))] for other in models if other != model)
        matches = wins + losses + ties
        win_rate = wins / matches if matches else 0.0
        overall.append((model, avg_sim, avg_rank, matches, win_rate))

    overall.sort(key=lambda x: x[2])
    for model, avg_sim, avg_rank, matches, win_rate in overall:
        lines.append(f"| {model} | {avg_sim:.4f} | {avg_rank:.2f} | {matches} | {win_rate:.3f} |")

    lines.append("")
    lines.append("## FT Models vs Base (Priority)")
    lines.append("| FT Model | Base | FT Wins | Base Wins | Ties | FT Win % |")
    lines.append("|---|---|---:|---:|---:|---:|")

    priority_ft_models = ["full_ft", "lora_ft", "qlora_ft"]
    for ft_model in priority_ft_models:
        if ft_model not in models or "base" not in models:
            continue
        ft_wins = pairwise_wins[(ft_model, "base")]
        base_wins = pairwise_wins[("base", ft_model)]
        ties = pairwise_ties[tuple(sorted([ft_model, "base"]))]
        total = ft_wins + base_wins + ties
        ft_win_pct = ft_wins / total if total else 0.0
        lines.append(
            f"| {ft_model} | base | {ft_wins} | {base_wins} | {ties} | {ft_win_pct:.3f} |"
        )

    lines.append("")
    lines.append("## Head-to-Head Pairwise Win Rates")
    lines.append("| Model 1 | Model 2 | M1 Wins | M2 Wins | Ties | M1 Win % |")
    lines.append("|---|---|---:|---:|---:|---:|")

    pairs = list(combinations(models, 2))
    priority_pairs = [("full_ft", "base"), ("lora_ft", "base"), ("qlora_ft", "base")]
    ordered_pairs = []
    seen_unordered = set()
    for m1, m2 in priority_pairs:
        if m1 in models and m2 in models and m1 != m2:
            unordered = tuple(sorted((m1, m2)))
            if unordered not in seen_unordered:
                ordered_pairs.append((m1, m2))
                seen_unordered.add(unordered)

    remaining_pairs = []
    for m1, m2 in pairs:
        unordered = tuple(sorted((m1, m2)))
        if unordered not in seen_unordered:
            remaining_pairs.append((m1, m2))
            seen_unordered.add(unordered)

    remaining_pairs.sort(key=lambda p: pairwise_wins[(p[0], p[1])], reverse=True)
    pairs = ordered_pairs + remaining_pairs

    for m1, m2 in pairs:
        w1 = pairwise_wins[(m1, m2)]
        w2 = pairwise_wins[(m2, m1)]
        ties = pairwise_ties[tuple(sorted([m1, m2]))]
        total = w1 + w2 + ties
        win_pct = w1 / total if total else 0.0
        lines.append(f"| {m1} | {m2} | {w1} | {w2} | {ties} | {win_pct:.3f} |")

    report = args.out_dir / "report_ranking.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote ranking report to {report}")


if __name__ == "__main__":
    main()
