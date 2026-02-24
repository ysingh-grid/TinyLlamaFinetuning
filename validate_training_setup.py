#!/usr/bin/env python3
"""Validation checklist script to catch training/eval misconfigurations.

Run this before launching a sweep or evaluation to ensure the codebase
does not regress into the issues identified in the deep_search_audit.md.

Usage:
    .venv/bin/python validate_training_setup.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ERRORS: list[str] = []
WARNINGS: list[str] = []


def error(msg: str) -> None:
    ERRORS.append(msg)


def warn(msg: str) -> None:
    WARNINGS.append(msg)


# ---------------------------------------------------------------------------
# 1. LoRA vs QLoRA config distinctness
# ---------------------------------------------------------------------------
def check_lora_qlora_configs() -> None:
    lora_cfg = ROOT / "lora_config.yaml"
    qlora_cfg = ROOT / "qlora_config.yaml"

    if not lora_cfg.exists() or not qlora_cfg.exists():
        warn("Cannot verify LoRA/QLoRA configs — one or both files missing.")
        return

    try:
        import yaml
    except ImportError:
        warn("PyYAML not installed — skipping YAML config checks.")
        return

    with lora_cfg.open() as f:
        lora = yaml.safe_load(f)
    with qlora_cfg.open() as f:
        qlora = yaml.safe_load(f)

    # Check that base models are different
    lora_model = lora.get("model", "")
    qlora_model = qlora.get("model", "")
    if lora_model == qlora_model:
        error(
            f"LoRA and QLoRA configs use the same base model: '{lora_model}'. "
            "QLoRA should use the 4-bit quantized base."
        )

    # Check mask_prompt
    if not lora.get("mask_prompt", False):
        error("lora_config.yaml: mask_prompt is not true (recommended for instruction tuning).")
    if not qlora.get("mask_prompt", False):
        error("qlora_config.yaml: mask_prompt is not true (recommended for instruction tuning).")


# ---------------------------------------------------------------------------
# 2. Adapter weight identity check (if both exist)
# ---------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def check_adapter_distinctness() -> None:
    lora_dir = ROOT / "mlx_best_models" / "lora"
    qlora_dir = ROOT / "mlx_best_models" / "qlora"

    if not lora_dir.exists() or not qlora_dir.exists():
        warn("Cannot compare adapter weights — one or both adapter dirs missing.")
        return

    lora_weights = sorted(lora_dir.glob("*.safetensors"))
    qlora_weights = sorted(qlora_dir.glob("*.safetensors"))

    if not lora_weights or not qlora_weights:
        warn("No .safetensors files found in one or both adapter dirs.")
        return

    lora_hashes = {p.name: _sha256(p) for p in lora_weights}
    qlora_hashes = {p.name: _sha256(p) for p in qlora_weights}

    identical = 0
    total = 0
    for name in set(lora_hashes) & set(qlora_hashes):
        total += 1
        if lora_hashes[name] == qlora_hashes[name]:
            identical += 1

    if total > 0 and identical == total:
        error(
            f"LoRA and QLoRA adapter weights are IDENTICAL ({identical}/{total} files). "
            "This means the two techniques produced the same model — something is wrong."
        )
    elif total > 0 and identical > 0:
        warn(
            f"{identical}/{total} adapter weight files are identical between LoRA and QLoRA. "
            "Verify this is expected."
        )


# ---------------------------------------------------------------------------
# 3. Eval model.json base model alignment
# ---------------------------------------------------------------------------
def check_eval_model_alignment() -> None:
    models_json = ROOT / "evaluation" / "models.json"
    if not models_json.exists():
        warn("evaluation/models.json not found — skipping eval base model check.")
        return

    models = json.loads(models_json.read_text())

    for entry in models:
        name = entry.get("name", "")
        adapter_path = entry.get("adapter_path")
        if not adapter_path:
            continue

        adapter_config = Path(adapter_path) / "adapter_config.json"
        # Try relative to ROOT
        if not adapter_config.is_absolute():
            adapter_config = ROOT / adapter_config

        if not adapter_config.exists():
            warn(f"Adapter config not found for {name}: {adapter_config}")
            continue

        ac = json.loads(adapter_config.read_text())
        trained_model = ac.get("model", "")
        eval_model = entry.get("model", "")

        # Normalize: strip trailing slashes and resolve relative paths
        if trained_model and eval_model:
            train_norm = trained_model.rstrip("/")
            eval_norm = eval_model.rstrip("/")

            # If trained model is a local path, compare resolved paths
            train_path = (ROOT / train_norm).resolve() if train_norm.startswith(".") else None
            eval_path = (ROOT / eval_norm).resolve() if eval_norm.startswith(".") else None

            # Only flag if both are resolvable and different, or both are strings and different
            if train_path and eval_path:
                if train_path != eval_path:
                    error(
                        f"Eval model mismatch for '{name}': "
                        f"trained on '{trained_model}', eval loads '{eval_model}'."
                    )
            elif train_norm != eval_norm:
                # One is hub ID, one is local — could be intentional but worth flagging
                warn(
                    f"Eval base model for '{name}' differs from training base: "
                    f"trained='{trained_model}', eval='{eval_model}'. "
                    "Verify this is intentional."
                )


# ---------------------------------------------------------------------------
# 4. Path contract: full retrain vs smoke eval
# ---------------------------------------------------------------------------
def check_retrain_path_contract() -> None:
    smoke_script = ROOT / "run_full_smoke_eval.sh"
    if not smoke_script.exists():
        return

    smoke_text = smoke_script.read_text()

    retrain_configs = list((ROOT / "experiments").glob("full_safe_*.yaml"))
    if not retrain_configs:
        return

    try:
        import yaml
    except ImportError:
        warn("PyYAML not installed — skipping retrain path contract check.")
        return

    for cfg_path in retrain_configs:
        with cfg_path.open() as f:
            cfg = yaml.safe_load(f)
        adapter_path = cfg.get("adapter_path", "")
        # Normalize path
        adapter_dir = Path(adapter_path).name if adapter_path else ""

        if adapter_dir and adapter_dir not in smoke_text:
            error(
                f"Retrain config {cfg_path.name} writes to '{adapter_path}', "
                f"but run_full_smoke_eval.sh does not reference '{adapter_dir}'."
            )


# ---------------------------------------------------------------------------
# 5. Shell scripts use venv
# ---------------------------------------------------------------------------
def check_venv_usage() -> None:
    for script_name in ["run_train.sh", "run_qlora.sh", "run_full.sh"]:
        script = ROOT / script_name
        if not script.exists():
            continue
        text = script.read_text()
        # Check for bare python3 usage (not .venv/bin/python)
        lines = text.split("\n")
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "python3 " in stripped and ".venv" not in stripped:
                error(f"{script_name}:{i} uses system 'python3' instead of '.venv/bin/python'.")


# ---------------------------------------------------------------------------
# 6. Data count sanity
# ---------------------------------------------------------------------------
def check_data_counts() -> None:
    data_dir = ROOT / "data"
    prep_script = ROOT / "prepare_dataset.py"

    if not prep_script.exists():
        return

    text = prep_script.read_text()
    import re
    match = re.search(r"NUM_EXAMPLES\s*=\s*(\d+)", text)
    if not match:
        return

    declared = int(match.group(1))

    total_lines = 0
    for split in ["train.jsonl", "valid.jsonl", "test.jsonl"]:
        split_path = data_dir / split
        if split_path.exists():
            with split_path.open() as f:
                total_lines += sum(1 for _ in f)

    if total_lines > 0 and declared != total_lines:
        warn(
            f"prepare_dataset.py declares NUM_EXAMPLES={declared}, "
            f"but data/ contains {total_lines} total rows. "
            "Consider rerunning prepare_dataset.py if you changed this value."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("  Training/Eval Setup Validation")
    print("=" * 60)

    check_lora_qlora_configs()
    check_adapter_distinctness()
    check_eval_model_alignment()
    check_retrain_path_contract()
    check_venv_usage()
    check_data_counts()

    print()
    if WARNINGS:
        print(f"⚠️  {len(WARNINGS)} warning(s):")
        for w in WARNINGS:
            print(f"   WARN: {w}")

    if ERRORS:
        print(f"\n❌ {len(ERRORS)} error(s):")
        for e in ERRORS:
            print(f"   ERROR: {e}")
        print(f"\nValidation FAILED with {len(ERRORS)} error(s).")
        sys.exit(1)
    else:
        print("\n✅ All checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
