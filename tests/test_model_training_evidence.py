from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.agent_runtime import build_fama_training_history, build_model_training_summary
from core.errors import ContractError


def _record(epoch: int, train_ic: float, train_valid_ic: float) -> dict:
    return {
        "epoch": epoch,
        "training_loss": 1.0 - epoch / 100.0,
        "train": {
            "ic": train_ic,
            "icir": train_ic / 2,
            "rank_ic": train_ic / 3,
            "rank_icir": train_ic / 4,
        },
        "train_valid": {
            "ic": train_valid_ic,
            "icir": train_valid_ic / 2,
            "rank_ic": train_valid_ic / 3,
            "rank_icir": train_valid_ic / 4,
        },
        "excluded_dates": {
            "train": {"too_few_pairs": 0, "constant_vector": 0},
            "train_valid": {"too_few_pairs": 0, "constant_vector": 0},
        },
    }


class ModelTrainingEvidenceTests(unittest.TestCase):
    def _experiment(self, root: Path) -> Path:
        experiment = root / "EXP_003"
        (experiment / "logs").mkdir(parents=True)
        (experiment / "config.json").write_text(
            json.dumps({"z_model": {"base_params": {"n_epochs": 10}}}),
            encoding="utf-8",
        )
        records = [
            _record(0, 0.10004, 0.08004),
            _record(1, 0.20006, 0.15006),
            _record(2, 0.25006, 0.12006),
            {"event": "checkpoint_selected", "metric": "train_valid.ic", "mode": "max", "epoch": 1, "score": 0.15006},
        ]
        (experiment / "logs" / "training_metrics.jsonl").write_text(
            "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
            encoding="utf-8",
        )
        return experiment

    def test_clem_summary_is_compact_and_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            summary = build_model_training_summary(self._experiment(Path(temporary_dir)))
        self.assertEqual(summary["epochs_completed"], 3)
        self.assertEqual(summary["selected_epoch"], 1)
        self.assertEqual(summary["best_train_valid_score"], 0.1501)
        self.assertAlmostEqual(summary["generalization_gap_at_best"], 0.05)
        self.assertAlmostEqual(summary["generalization_gap_last"], 0.13)
        self.assertAlmostEqual(summary["post_best_train_valid_drop"], 0.03)
        self.assertAlmostEqual(summary["post_best_train_change"], 0.05)
        self.assertEqual(summary["epochs_after_best"], 1)
        self.assertTrue(summary["early_stopped"])
        self.assertNotIn("rows", summary)

    def test_fama_receives_every_epoch_in_order_without_test_fields(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            history = build_fama_training_history(self._experiment(Path(temporary_dir)))
        self.assertEqual([row[0] for row in history["rows"]], [0, 1, 2])
        self.assertEqual(history["rows"][1][2:4], [0.2001, 0.1501])
        self.assertEqual(len(history["rows"]), history["derived_summary"]["epochs_completed"])
        serialized = json.dumps(history, sort_keys=True).lower()
        self.assertNotIn('"test_ic"', serialized)
        self.assertNotIn('"test_metrics"', serialized)
        self.assertNotIn("/test/", serialized)
        self.assertFalse(history["contains_test_derived_data"])

    def test_fama_follows_nested_checkpoint_reuse_to_training_source(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            experiments = Path(temporary_dir) / "run" / "experiments"
            source = self._experiment(experiments)
            middle = experiments / "EXP_007"
            candidate = experiments / "EXP_008"
            for experiment, reused_from in ((middle, source.name), (candidate, middle.name)):
                (experiment / "artifacts").mkdir(parents=True)
                (experiment / "artifacts" / "selected_checkpoint.json").write_text(
                    json.dumps({"reused_from": reused_from}), encoding="utf-8"
                )

            history = build_fama_training_history(candidate)

        self.assertEqual(history["model_source_experiment_id"], "EXP_003")
        self.assertTrue(history["model_reused"])
        self.assertEqual(history["derived_summary"]["model_source_experiment_id"], "EXP_003")
        self.assertTrue(history["derived_summary"]["model_reused"])

    def test_fama_rejects_checkpoint_reuse_cycle(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            experiments = Path(temporary_dir) / "run" / "experiments"
            first = experiments / "EXP_007"
            second = experiments / "EXP_008"
            for experiment, reused_from in ((first, second.name), (second, first.name)):
                (experiment / "artifacts").mkdir(parents=True)
                (experiment / "artifacts" / "selected_checkpoint.json").write_text(
                    json.dumps({"reused_from": reused_from}), encoding="utf-8"
                )

            with self.assertRaisesRegex(ContractError, "MODEL_TRAINING_SOURCE_CYCLE"):
                build_fama_training_history(first)

    def test_fama_rejects_noncanonical_reuse_id(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            experiment = Path(temporary_dir) / "run" / "experiments" / "EXP_007"
            (experiment / "artifacts").mkdir(parents=True)
            (experiment / "artifacts" / "selected_checkpoint.json").write_text(
                json.dumps({"reused_from": "../EXP_001"}), encoding="utf-8"
            )

            with self.assertRaisesRegex(ContractError, "MODEL_TRAINING_SOURCE_ID_INVALID"):
                build_fama_training_history(experiment)


if __name__ == "__main__":
    unittest.main()
