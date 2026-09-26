from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import jsonschema

from core.agent_runtime import AgentRuntime, build_rapa_search_state, consecutive_unaccepted_layer
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
            {
                "epoch": 0,
                "train_valid": {"ic": 3.0, "icir": 1.0, "rank_ic": 1.0, "rank_icir": 1.0},
                "agent_valid": {"ic": -9.0},
            },
            {
                "epoch": 1,
                "train_valid": {"ic": 2.0, "icir": 4.0, "rank_ic": 2.0, "rank_icir": 2.0},
                "agent_valid": {"ic": 99.0},
            },
            {
                "epoch": 2,
                "train_valid": {"ic": 1.0, "icir": 2.0, "rank_ic": 5.0, "rank_icir": 6.0},
                "agent_valid": {"ic": 100.0},
            },
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
            "train_valid": {"ic": 0.222251},
            "excluded_dates": {},
        })
        self.assertEqual(record["training_loss"], 0.9877)
        self.assertEqual(record["train"]["ic"], 0.1111)
        self.assertEqual(record["train_valid"]["ic"], 0.2223)
        self.assertNotIn("agent_valid", record)


class RoutingTests(unittest.TestCase):
    class Memory:
        def __init__(self, records: list[dict] | None = None):
            self.records = records or []

        def projection(self, role: str) -> list[dict]:
            self.assert_role(role)
            return self.records

        def assert_role(self, role: str) -> None:
            if role != "cross_layer":
                raise AssertionError(role)

    def runtime(self, alpha_frozen: bool, memory: list[dict] | None = None):
        runtime = object.__new__(AgentRuntime)
        runtime.repo_root = REPO_ROOT
        runtime.config = {"z_alpha": {"selected_features": list(range(10 if alpha_frozen else 5)), "alpha_frozen": alpha_frozen}}
        runtime.memory = self.Memory(memory)
        return runtime

    def test_three_consecutive_unaccepted_same_layer_detection(self):
        rejected_portfolio = [
            {"experiment_id": f"EXP_{index:03d}", "layer": "portfolio", "accepted": False}
            for index in range(1, 4)
        ]
        self.assertEqual(consecutive_unaccepted_layer(rejected_portfolio), "RAPA")
        self.assertIsNone(consecutive_unaccepted_layer(rejected_portfolio[:2]))
        self.assertIsNone(consecutive_unaccepted_layer([
            rejected_portfolio[0],
            {**rejected_portfolio[1], "accepted": True},
            rejected_portfolio[2],
        ]))
        self.assertIsNone(consecutive_unaccepted_layer([
            *rejected_portfolio,
            {"experiment_id": "EXP_004", "layer": "model", "accepted": False},
        ]))

    def test_three_consecutive_unaccepted_layer_is_blocked_for_one_round(self):
        memory = [
            {"experiment_id": "EXP_000", "layer": None, "accepted": True},
            {"experiment_id": "EXP_001", "layer": "portfolio", "accepted": False},
            {"experiment_id": "EXP_002", "layer": "portfolio", "accepted": False},
            {
                "experiment_id": "EXP_003", "layer": "portfolio", "accepted": False,
                "verdict": "CONTRACT_REJECTED",
            },
        ]
        rapa = {
            "selected_agent": "RAPA", "selected_layer": "portfolio",
            "failure_summary": "portfolio translation remains inefficient",
            "supporting_evidence": ["validation sharpe_ratio is 1.248"],
            "why_not_other_layers": ["portfolio evidence is otherwise strongest"],
            "intervention_goal": "improve absolute validation Sharpe", "confidence": 0.8,
            "primary_objective_assessment": {
                "metric": "validation_sharpe_ratio", "observed_value": 1.248,
                "failure_supported": True, "rationale": "portfolio diagnosis is supported",
            },
        }
        evidence = {"result": {"sharpe_ratio": 1.248}}
        with self.assertRaisesRegex(ContractError, "THREE_CONSECUTIVE_UNACCEPTED_LAYER_COOLDOWN"):
            self.runtime(True, memory).validate_clem(rapa, 4, evidence)

        fama = {
            "selected_agent": "FAMA", "selected_layer": "model",
            "failure_summary": "model generalization is now comparatively strongest",
            "supporting_evidence": ["three portfolio interventions were not accepted"],
            "why_not_other_layers": ["portfolio is on deterministic cooldown"],
            "intervention_goal": "improve model generalization", "confidence": 0.8,
        }
        self.runtime(True, memory).validate_clem(fama, 4, evidence)

    def test_rass_retries_until_alpha_resolution_then_is_disabled(self):
        wrong = {
            "selected_agent": "FAMA", "selected_layer": "model", "failure_summary": "model gap",
            "supporting_evidence": ["five features"], "why_not_other_layers": ["none"],
            "intervention_goal": "learn", "confidence": 0.8,
        }
        with self.assertRaises(ContractError):
            self.runtime(False).validate_clem(wrong, 1)
        rass = dict(wrong, selected_agent="RASS", selected_layer="alpha")
        self.runtime(False).validate_clem(rass, 1)
        one_failure = [{
            "experiment_id": "EXP_001", "round": 1, "layer": "alpha",
            "group": "initial_feature_bootstrap", "accepted": False,
            "verdict": "FALSIFIED", "intervention": {"added_features": [
                {"expression": "A"}, {"expression": "B"}, {"expression": "C"},
            ]},
        }]
        self.runtime(False, one_failure).validate_clem(rass, 2)
        with self.assertRaisesRegex(ContractError, "RASS_ONLY_LEGAL_DURING_BOOTSTRAP_SEARCH"):
            self.runtime(True).validate_clem(rass, 2)

    def test_clem_schema_only_allows_fama_or_rapa_after_alpha_freeze(self):
        runtime = self.runtime(True)
        schema, eligible = runtime._clem_route_schema(
            alpha_frozen=True,
            forced_agent=None,
        )
        self.assertEqual(eligible, ["FAMA", "RAPA"])
        self.assertEqual(
            schema["properties"]["selected_agent"]["enum"],
            ["FAMA", "RAPA"],
        )

    def test_clem_schema_honors_deterministic_layer_cooldown(self):
        runtime = self.runtime(True)
        schema, eligible = runtime._clem_route_schema(
            alpha_frozen=True,
            forced_agent="RAPA",
        )
        self.assertEqual(eligible, ["RAPA"])
        self.assertEqual(
            schema["properties"]["selected_agent"]["enum"],
            ["RAPA"],
        )

    def test_rapa_search_state_tracks_only_two_parameters_and_four_attempts(self):
        config = {"z_portfolio": {"risk_aversion": 1.1, "turnover_penalty": 1.0}}
        memory = [{
            "experiment_id": "EXP_002", "layer": "portfolio",
            "intervention": [{
                "path": "z_portfolio.risk_aversion",
                "old_value": 1.0, "new_value": 1.1,
            }],
            "accepted": True, "verdict": "SUPPORTED",
        }]
        state = build_rapa_search_state(memory, config)
        self.assertEqual(set(state["parameters"]), {"risk_aversion", "turnover_penalty"})
        self.assertEqual(state["parameters"]["risk_aversion"]["phase"], "EXPAND")
        self.assertEqual(state["parameters"]["risk_aversion"]["remaining_attempts"], 3)

    def test_three_fama_failures_force_rapa_even_without_preexisting_opportunity(self):
        memory = [
            {"experiment_id": f"EXP_{index:03d}", "layer": "model", "accepted": False}
            for index in range(2, 5)
        ]
        decision = {
            "selected_agent": "RAPA", "selected_layer": "portfolio",
            "failure_summary": "forced cross-layer portfolio reassessment",
            "supporting_evidence": ["three consecutive FAMA attempts were not accepted"],
            "why_not_other_layers": ["FAMA is deterministically blocked for this round"],
            "intervention_goal": "probe one bounded portfolio mechanism", "confidence": 0.6,
            "primary_objective_assessment": {
                "metric": "validation_sharpe_ratio", "observed_value": 1.248,
                "failure_supported": False,
                "rationale": "RAPA is forced as a routing-diversity reassessment",
            },
        }
        self.runtime(True, memory).validate_clem(
            decision, 5, {"result": {"sharpe_ratio": 1.248}}
        )

    def test_no_intervention_is_not_a_legal_route(self):
        decision = {
            "selected_agent": "NO_INTERVENTION", "selected_layer": "none", "failure_summary": "no dominant failure",
            "supporting_evidence": ["all changes within noise"], "why_not_other_layers": ["unsupported"],
            "intervention_goal": None, "confidence": 0.4,
        }
        with self.assertRaises(jsonschema.ValidationError):
            self.runtime(True).validate_clem(decision, 2)

    def test_rapa_allows_positive_sharpe_with_matching_portfolio_opportunity(self):
        decision = {
            "selected_agent": "RAPA", "selected_layer": "portfolio",
            "failure_summary": "absolute validation Sharpe is not translated efficiently",
            "supporting_evidence": ["validation sharpe_ratio is 1.248"],
            "why_not_other_layers": ["signal remains usable"],
            "intervention_goal": "improve absolute validation Sharpe", "confidence": 0.8,
            "primary_objective_assessment": {
                "metric": "validation_sharpe_ratio", "observed_value": 1.248,
                "failure_supported": True, "rationale": "absolute Sharpe supports portfolio diagnosis",
            },
        }
        evidence = {"result": {"sharpe_ratio": 1.248}}
        self.runtime(True).validate_clem(decision, 2, evidence)

        unsupported = {**decision, "primary_objective_assessment": {
            **decision["primary_objective_assessment"], "failure_supported": False,
        }}
        with self.assertRaisesRegex(ContractError, "PORTFOLIO_OPPORTUNITY_NOT_SUPPORTED"):
            self.runtime(True).validate_clem(unsupported, 2, evidence)

        mismatched = {**decision, "primary_objective_assessment": {
            **decision["primary_objective_assessment"], "observed_value": 1.5,
        }}
        with self.assertRaisesRegex(ContractError, "VALIDATION_SHARPE_MISMATCH"):
            self.runtime(True).validate_clem(mismatched, 2, evidence)


if __name__ == "__main__":
    unittest.main()
