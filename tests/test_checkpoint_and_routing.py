from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from core.agent_runtime import AgentRuntime
from core.errors import ContractError
from training.model_trainer import (
    _atomic_torch_save,
    checkpoint_score,
    is_checkpoint_improvement,
    persisted_epoch_record,
    persisted_metric,
    select_checkpoint_epoch,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class CheckpointTests(unittest.TestCase):
    def test_all_validation_metrics_and_default(self):
        history = [
            {"epoch": 0, "valid": {"ic": 3.0, "icir": 1.0, "rank_ic": 1.0, "rank_icir": 1.0}},
            {"epoch": 1, "valid": {"ic": 2.0, "icir": 4.0, "rank_ic": 2.0, "rank_icir": 2.0}},
            {"epoch": 2, "valid": {"ic": 1.0, "icir": 2.0, "rank_ic": 5.0, "rank_icir": 6.0}},
        ]
        self.assertEqual(select_checkpoint_epoch(history)[0], 0)
        self.assertEqual(select_checkpoint_epoch(history, "ic")[0], 0)
        self.assertEqual(select_checkpoint_epoch(history, "icir")[0], 1)
        self.assertEqual(select_checkpoint_epoch(history, "rank_ic")[0], 2)
        self.assertEqual(select_checkpoint_epoch(history, "rank_icir")[0], 2)
        with self.assertRaises(ContractError):
            select_checkpoint_epoch(history, "sharpe")

    def test_best_checkpoint_requires_strict_finite_improvement(self):
        self.assertTrue(is_checkpoint_improvement(0.10, float("-inf")))
        self.assertTrue(is_checkpoint_improvement(0.11, 0.10))
        self.assertFalse(is_checkpoint_improvement(0.10, 0.10))
        self.assertFalse(is_checkpoint_improvement(0.09, 0.10))
        self.assertFalse(is_checkpoint_improvement(None, 0.10))
        self.assertFalse(is_checkpoint_improvement(float("nan"), 0.10))
        self.assertEqual(checkpoint_score(None), float("-inf"))

    def test_best_checkpoint_atomically_replaces_one_file(self):
        with TemporaryDirectory() as temporary_dir:
            checkpoint = Path(temporary_dir) / "checkpoints" / "best.pt"
            _atomic_torch_save({"epoch": 0}, checkpoint)
            _atomic_torch_save({"epoch": 3}, checkpoint)
            self.assertEqual([path.name for path in checkpoint.parent.iterdir()], ["best.pt"])
            self.assertEqual(torch.load(checkpoint, weights_only=False)["epoch"], 3)

    def test_training_artifact_metrics_are_rounded_to_four_decimals(self):
        self.assertEqual(persisted_metric(0.123456), 0.1235)
        self.assertIsNone(persisted_metric(float("nan")))
        record = persisted_epoch_record({
            "epoch": 0,
            "training_loss": 0.987654,
            "train": {"ic": 0.111149},
            "valid": {"ic": 0.222251},
            "excluded_dates": {},
        })
        self.assertEqual(record["training_loss"], 0.9877)
        self.assertEqual(record["train"]["ic"], 0.1111)
        self.assertEqual(record["valid"]["ic"], 0.2223)


class RoutingTests(unittest.TestCase):
    def runtime(self, alpha_frozen: bool):
        runtime = object.__new__(AgentRuntime)
        runtime.repo_root = REPO_ROOT
        runtime.config = {"z_alpha": {"selected_features": list(range(10 if alpha_frozen else 6)), "alpha_frozen": alpha_frozen}}
        return runtime

    def test_mandatory_rass_and_permanent_freeze(self):
        wrong = {
            "selected_agent": "FAMA", "selected_layer": "model", "failure_summary": "model gap",
            "supporting_evidence": ["six features"], "why_not_other_layers": ["none"],
            "intervention_goal": "learn", "confidence": 0.8,
        }
        with self.assertRaises(ContractError):
            self.runtime(False).validate_clem(wrong, 1)
        rass = dict(wrong, selected_agent="RASS", selected_layer="alpha")
        with self.assertRaises(ContractError):
            self.runtime(True).validate_clem(rass, 2)

    def test_no_intervention_shape(self):
        decision = {
            "selected_agent": "NO_INTERVENTION", "selected_layer": "none", "failure_summary": "no dominant failure",
            "supporting_evidence": ["all changes within noise"], "why_not_other_layers": ["unsupported"],
            "intervention_goal": None, "confidence": 0.4,
        }
        self.runtime(True).validate_clem(decision, 2)


if __name__ == "__main__":
    unittest.main()
