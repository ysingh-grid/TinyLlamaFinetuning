import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import sweep_mlx_lora as sweep


class _FakeStdout:
    def __init__(self, lines):
        self._iter = iter(lines)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iter)

    def close(self):
        self.closed = True


class _FakeProcess:
    def __init__(self, lines, return_code=0):
        self.stdout = _FakeStdout(lines)
        self.return_code = return_code
        self.terminated = False
        self.killed = False

    def terminate(self):
        self.terminated = True
        self.return_code = 1

    def wait(self, timeout=None):
        return self.return_code

    def kill(self):
        self.killed = True
        self.return_code = 9


def _make_args(tmp_path: Path, patience: int, min_delta: float) -> Namespace:
    return Namespace(
        out_dir=tmp_path / "out",
        data_dir=tmp_path / "data",
        full_model=sweep.DEFAULT_FULL_MODEL,
        model=sweep.DEFAULT_STOCK_4BIT_MODEL,
        seed=0,
        val_batches=-1,
        steps_per_report=10,
        test_batches=-1,
        max_seq_length=512,
        lora_layers=16,
        dropout=0.05,
        status="none",
        early_stop_patience=patience,
        early_stop_min_delta=min_delta,
    )


class SweepMlxLoraTests(unittest.TestCase):
    def test_build_grid_counts(self):
        self.assertEqual(len(sweep.build_grid("full")), 4)
        self.assertEqual(len(sweep.build_grid("lora")), 16)
        self.assertEqual(len(sweep.build_grid("qlora")), 16)

    def test_extract_val_loss_and_parse_loss(self):
        line = "Iter 123: Val loss 1.234, Val took 0.123s"
        self.assertAlmostEqual(sweep.extract_val_loss(line), 1.234)

        log_text = "Iter 1: Val loss 2.000\nTest loss 1.500, Test ppl 4.48\n"
        self.assertAlmostEqual(sweep.parse_loss(log_text), 1.5)

    def test_model_for_technique(self):
        args = Namespace(full_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", model="./models/tinyllama-4bit-base")
        self.assertEqual(sweep.model_for_technique(args, "full"), args.full_model)
        self.assertEqual(sweep.model_for_technique(args, "lora"), args.model)
        self.assertEqual(sweep.model_for_technique(args, "qlora"), args.model)

    def test_validate_model_sources_rejects_local_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_model = Path(tmp_dir) / "local_model"
            local_model.mkdir(parents=True, exist_ok=True)
            args = Namespace(
                full_model=sweep.DEFAULT_FULL_MODEL,
                model=str(local_model),
                allow_local_model=False,
            )
            with self.assertRaises(ValueError):
                sweep.validate_model_sources(args)

    def test_validate_model_sources_allows_default_stock_local_path(self):
        args = Namespace(
            full_model=sweep.DEFAULT_FULL_MODEL,
            model=sweep.DEFAULT_STOCK_4BIT_MODEL,
            allow_local_model=False,
        )
        sweep.validate_model_sources(args)

    def test_run_trial_early_stops_when_patience_exhausted(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            args = _make_args(tmp_path, patience=1, min_delta=0.0)

            fake_proc = _FakeProcess(
                lines=[
                    "Iter 1: Val loss 1.000, Val took 0.100s\n",
                    "Iter 2: Val loss 1.100, Val took 0.100s\n",
                ],
                return_code=0,
            )

            with patch("sweep_mlx_lora.subprocess.Popen", return_value=fake_proc):
                result = sweep.run_trial(
                    args=args,
                    technique="lora",
                    trial_id=1,
                    total_trials=1,
                    params=(8, 16, 1e-4, 1, 1, 4),
                    train_rows=16,
                    keys=["self_attn.q_proj", "self_attn.v_proj"],
                )

            self.assertTrue(result["early_stopped"])
            self.assertTrue(result["ok"])
            self.assertTrue(fake_proc.terminated)
            self.assertAlmostEqual(result["loss"], 1.1)
            log_path = tmp_path / "out" / "lora" / "trial_0001" / "train.log"
            self.assertIn("EARLY_STOP:", log_path.read_text(encoding="utf-8"))

    def test_run_trial_no_early_stop_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            args = _make_args(tmp_path, patience=0, min_delta=0.0)

            fake_proc = _FakeProcess(
                lines=[
                    "Iter 1: Val loss 1.000, Val took 0.100s\n",
                    "Iter 2: Val loss 1.200, Val took 0.100s\n",
                ],
                return_code=0,
            )

            with patch("sweep_mlx_lora.subprocess.Popen", return_value=fake_proc):
                result = sweep.run_trial(
                    args=args,
                    technique="full",
                    trial_id=1,
                    total_trials=1,
                    params=(1e-4, 1, 1, 4),
                    train_rows=16,
                    keys=["self_attn.q_proj"],
                )

            self.assertFalse(result["early_stopped"])
            self.assertTrue(result["ok"])
            self.assertFalse(fake_proc.terminated)


if __name__ == "__main__":
    unittest.main()
