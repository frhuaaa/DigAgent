from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.agent_runtime import AgentRuntime
from core.contract_validator import validate_proposal
from core.errors import ContractError
from core.executor import _expected_checkpoint_config_hash
from core.evidence_builder import build_rass_shortlist_evidence
from core.io_utils import atomic_write_json
from core.orchestrator import _memory_candidate, _memory_contract_rejection
from scripts.audit_researcher_outputs import REQUIRED_TEST_ARTIFACTS, build_researcher_outputs_audit


REPO_ROOT = Path(__file__).resolve().parents[1]


class RassTwoStageTests(unittest.TestCase):
    def test_shortlist_is_exactly_twelve_unique_eligible_nonanchors(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            records = [
                {"id": f"F{i:02d}", "expression": f"Expr{i:02d}", "eligible": True}
                for i in range(12)
            ]
            atomic_write_json(run_root / "configs" / "rass_train_evidence.json", {
                "evidence_hash": "base-hash",
                "records": records,
            })
            runtime = object.__new__(AgentRuntime)
            runtime.repo_root = REPO_ROOT
            runtime.run_root = run_root
            runtime.config = {"z_alpha": {"selected_features": ["Anchor"]}}
            proposal = {
                "agent": "RASS",
                "stage": "train_shortlist",
                "data_split": "train",
                "shortlist_size": 12,
                "shortlist": [{"id": item["id"], "expression": item["expression"]} for item in records],
                "reasoning": "diverse train-only candidates",
                "diversity_rationale": "multiple nonredundant factor families",
                "evidence_hash": "base-hash",
            }
            runtime.validate_rass_shortlist(proposal)
            records[0]["eligible"] = False
            atomic_write_json(run_root / "configs" / "rass_train_evidence.json", {
                "evidence_hash": "base-hash",
                "records": records,
            })
            with self.assertRaises(ContractError):
                runtime.validate_rass_shortlist(proposal)

    def test_joint_shortlist_evidence_uses_train_only_daily_cross_sections(self):
        anchors = [f"Anchor{i}" for i in range(6)]
        shortlist = [{"id": f"F{i:02d}", "expression": f"Expr{i:02d}"} for i in range(12)]
        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        instruments = [f"S{i:03d}" for i in range(25)]
        index = pd.MultiIndex.from_product([instruments, dates], names=["instrument", "datetime"])
        rng = np.random.default_rng(0)
        columns = anchors + [item["expression"] for item in shortlist] + ["Ref($close, -2) / Ref($close, -1) - 1"]
        frame = pd.DataFrame(rng.normal(size=(len(index), len(columns))), index=index, columns=columns)
        config = {
            "task": {
                "train_start_time": "2020-01-01",
                "train_end_time": "2020-01-03",
            },
            "z_alpha": {"selected_features": anchors},
        }
        with patch("core.evidence_builder.initialize_qlib"), patch(
            "core.evidence_builder.resolve_market_instruments",
            return_value=({name: [] for name in instruments}, []),
        ), patch("core.evidence_builder.D.features", return_value=frame, create=True):
            evidence = build_rass_shortlist_evidence(REPO_ROOT, config, shortlist, "base-hash")
        self.assertEqual(evidence["data_split"], "train")
        self.assertEqual(len(evidence["pairwise_daily_cross_sectional_correlation"]), 66)
        self.assertEqual(len(evidence["marginal_information_after_anchor_residualization"]), 12)
        self.assertTrue(evidence["evidence_hash"])


class RapaRecoveryTests(unittest.TestCase):
    def test_rapa_recovery_uses_accepted_parent_model_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / "EXP_001"
            atomic_write_json(parent / "config.json", {"provenance": {"effective_config_hash": "parent-model-hash"}})
            candidate = {"provenance": {"effective_config_hash": "candidate-portfolio-hash"}}
            selected = {"reused_from": "EXP_001"}
            self.assertEqual(
                _expected_checkpoint_config_hash(candidate, "RAPA", parent, selected),
                "parent-model-hash",
            )
            with self.assertRaises(ContractError):
                _expected_checkpoint_config_hash(candidate, "RAPA", parent, {"reused_from": "EXP_000"})


class AntiRepetitionTests(unittest.TestCase):
    def test_rejected_parameter_cannot_repeat_from_same_parent(self):
        accepted = {
            "z_portfolio": {"risk_aversion": 1.0},
            "z_alpha": {"selected_features": [], "alpha_frozen": True},
            "z_model": {"derived": {"d_feat": 0}},
        }
        clem = {"selected_agent": "RAPA", "selected_layer": "portfolio", "confidence": 0.8}
        proposal = {
            "agent": "RAPA",
            "selected_group": "portfolio_objective",
            "local_diagnosis": "risk translation remains unresolved",
            "hypothesis": "lower risk aversion improves expression",
            "diff": [{
                "op": "replace",
                "path": "z_portfolio.risk_aversion",
                "old_value": 1.0,
                "new_value": 0.5,
            }],
            "reasoning": "validation-only portfolio evidence",
            "expected_signature": [{"metric": "risk_term_mean", "direction": "decrease"}],
            "falsification_condition": "risk expression does not improve",
            "confidence": 0.8,
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            atomic_write_json(run_root / "configs" / "state.json", {"accepted_experiment_id": "EXP_001"})
            memory = run_root / "memory" / "experiments.jsonl"
            memory.parent.mkdir(parents=True)
            memory.write_text(json.dumps({
                "experiment_id": "EXP_002",
                "round": 2,
                "parent": "EXP_001",
                "layer": "portfolio",
                "group": "portfolio_objective",
                "intervention": [{"path": "z_portfolio.risk_aversion", "new_value": 0.1}],
                "verdict": "FALSIFIED",
                "accepted": False,
            }) + "\n", encoding="utf-8")
            outcome = validate_proposal(REPO_ROOT, run_root, accepted, "EXP_003", clem, proposal)
            self.assertFalse(outcome.passed)
            self.assertIn("REPEATED_REJECTED_PARAMETER_SAME_PARENT", outcome.reason_codes)


class MemoryStateTests(unittest.TestCase):
    def test_rejected_round_preserves_parent_alpha_frozen_state(self):
        record = _memory_candidate(
            "EXP_003",
            3,
            "EXP_001",
            {"failure_summary": "model failure", "selected_layer": "model"},
            {"selected_group": "train_params", "diff": [], "expected_signature": []},
            {
                "promotion": False,
                "promotion_reason": "rollback",
                "gate_verdict": "UNCERTAIN",
                "sharpe": {"state": "UNCHANGED"},
                "mechanism_state": "INCONCLUSIVE",
                "tradeoff_state": "NO_MATERIAL_DEGRADATION",
                "alpha_frozen_after_promotion": False,
            },
            {"ic": 0.01, "sharpe_ratio": 0.5},
            {"z_alpha": {"alpha_frozen": True}},
        )
        self.assertTrue(record["alpha_frozen"])

    def test_contract_rejection_redacts_isolation_unsafe_proposal(self):
        record = _memory_contract_rejection(
            "EXP_002",
            2,
            "EXP_001",
            {"failure_summary": "model failure", "selected_layer": "model"},
            {
                "selected_group": "train_params",
                "diff": [{"path": "experiments/EXP_000/test/result.json"}],
                "expected_signature": [],
            },
            ["TEST_REFERENCE_FORBIDDEN"],
            {"z_alpha": {"alpha_frozen": True}},
        )
        self.assertEqual(record["intervention"], {"redacted": True, "reason": "validation_isolation"})
        self.assertIsNone(record["expected_signature"])


class ResearcherAuditTests(unittest.TestCase):
    def test_audit_reports_missing_outputs_without_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            complete = run_root / "experiments" / "EXP_000"
            incomplete = run_root / "experiments" / "EXP_001"
            atomic_write_json(complete / "logs" / "execution_complete.json", {"status": "COMPLETE"})
            atomic_write_json(incomplete / "logs" / "execution_complete.json", {"status": "COMPLETE"})
            for name in REQUIRED_TEST_ARTIFACTS:
                path = complete / "test" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                if name == "researcher_complete.json":
                    atomic_write_json(path, {"status": "COMPLETE"})
                else:
                    path.write_text("nonempty\n", encoding="utf-8")
            audit = build_researcher_outputs_audit(run_root)
            self.assertFalse(audit["researcher_outputs_complete"])
            self.assertEqual(audit["incomplete_experiments"], ["EXP_001"])
            self.assertEqual(
                set(audit),
                {
                    "researcher_only",
                    "adaptive_state_affected",
                    "executed_experiment_count",
                    "complete_experiment_count",
                    "researcher_outputs_complete",
                    "incomplete_experiments",
                    "experiments",
                },
            )
            self.assertNotIn("sharpe", json.dumps(audit).lower())
            self.assertNotIn("icir", json.dumps(audit).lower())


if __name__ == "__main__":
    unittest.main()
