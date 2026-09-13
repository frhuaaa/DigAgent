from __future__ import annotations

import copy
import json
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

from core.contract_validator import apply_validated_proposal, validate_proposal
from core.io_utils import atomic_write_json, canonical_json_bytes, load_json, sha256_bytes
from evaluation.result_writer import (
    RESULT_KEYS,
    format_portfolio_report,
    serialize_result,
    validate_result,
    write_result,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def accepted_config() -> dict:
    config = load_json(REPO_ROOT / "submit" / "lstm_csi500_deepseek.json")
    config["z_alpha"]["alpha_frozen"] = False
    config["z_model"]["derived"] = {"d_feat": 6}
    config["provenance"] = {"effective_config_hash": "a"}
    return config


class ResultContractTests(unittest.TestCase):
    def test_exact_keys_nulls_types_and_three_decimals(self):
        result = OrderedDict((key, None) for key in RESULT_KEYS)
        result["ic"] = 0.1
        result["max_drawdown"] = -0.0
        result["n_periods"] = 3
        validate_result(result, REPO_ROOT / "schemas" / "result.schema.json")
        text = serialize_result(result).decode("utf-8")
        self.assertIn('"ic": 0.100', text)
        self.assertIn('"max_drawdown": -0.000', text)
        self.assertIn('"n_periods": 3', text)
        self.assertIn('"icir": null', text)
        invalid = OrderedDict(result)
        invalid["extra"] = 1.0
        with self.assertRaises(Exception):
            validate_result(invalid, REPO_ROOT / "schemas" / "result.schema.json")

    def test_write_result_argument_contract(self):
        result = OrderedDict((key, None) for key in RESULT_KEYS)
        result["n_periods"] = 0
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            write_result(result, path, REPO_ROOT / "schemas" / "result.schema.json")
            self.assertEqual(tuple(json.loads(path.read_text(encoding="utf-8")).keys()), RESULT_KEYS)

    def test_portfolio_report_keeps_net_excess_turnover_and_other_metrics(self):
        result = OrderedDict((key, None) for key in RESULT_KEYS)
        result.update({
            "annual_return": 0.071,
            "sharpe_ratio": 0.369,
            "information_ratio": 0.682,
            "excess_win_rate": 0.517,
            "one_way_turnover_mean": 0.051,
            "ic": 0.041,
            "n_periods": 240,
        })
        report = format_portfolio_report("EXP_000", result)
        self.assertIn("Net return", report)
        self.assertIn("Excess return", report)
        self.assertIn("Sharpe (IR)", report)
        self.assertIn("One-way", report)
        self.assertIn("Other metrics", report)
        self.assertIn("7.100%", report)
        self.assertIn("51.700%", report)
        self.assertIn("240", report)


class ContractValidatorTests(unittest.TestCase):
    def setUp(self):
        self.config = accepted_config()
        self.clem_fama = {"selected_agent": "FAMA", "selected_layer": "model", "confidence": 0.8}
        self.clem_rapa = {"selected_agent": "RAPA", "selected_layer": "portfolio", "confidence": 0.8}

    def _fama(self, group="train_params", diff=None):
        return {
            "agent": "FAMA", "selected_group": group, "local_diagnosis": "gap",
            "hypothesis": "bounded capacity change", "reasoning": "validation learning evidence",
            "diff": diff or [{"op": "replace", "path": "z_model.train_params.lr", "old_value": 0.001, "new_value": 0.0005}],
            "expected_signature": [{"metric": "ic", "direction": "increase"}],
            "falsification_condition": "IC moves materially opposite", "confidence": 0.7,
        }

    def _rapa(self, diff):
        return {
            "agent": "RAPA", "selected_group": "portfolio_objective", "local_diagnosis": "translation",
            "hypothesis": "reduce churn", "reasoning": "cost evidence", "diff": diff,
            "expected_signature": [{"metric": "one_way_turnover_mean", "direction": "decrease"}],
            "falsification_condition": "turnover rises", "confidence": 0.8,
        }

    def test_fama_legal_and_apply(self):
        outcome = validate_proposal(REPO_ROOT, REPO_ROOT, self.config, "EXP_002", self.clem_fama, self._fama())
        self.assertTrue(outcome.passed, outcome.reason_codes)
        candidate = apply_validated_proposal(self.config, outcome, "FAMA")
        self.assertEqual(candidate["z_model"]["train_params"]["lr"], 0.0005)

    def test_fama_unambiguous_json_pointer_is_canonicalized(self):
        proposal = self._fama(diff=[{
            "op": "replace", "path": "/z_model/train_params/dropout",
            "old_value": 0.0, "new_value": 0.2,
        }])
        outcome = validate_proposal(REPO_ROOT, REPO_ROOT, self.config, "EXP_002", self.clem_fama, proposal)
        self.assertTrue(outcome.passed, outcome.reason_codes)
        self.assertEqual(outcome.candidate_diff[0]["path"], "z_model.train_params.dropout")
        candidate = apply_validated_proposal(self.config, outcome, "FAMA")
        self.assertEqual(candidate["z_model"]["train_params"]["dropout"], 0.2)

    def test_fama_escaped_json_pointer_is_rejected(self):
        proposal = self._fama(diff=[{
            "op": "replace", "path": "/z_model/train_params/drop~1out",
            "old_value": 0.0, "new_value": 0.2,
        }])
        outcome = validate_proposal(REPO_ROOT, REPO_ROOT, self.config, "EXP_002", self.clem_fama, proposal)
        self.assertFalse(outcome.passed)
        self.assertIn("FAMA_CROSS_GROUP_PATH", outcome.reason_codes)

    def test_fama_two_parameter_cap_and_cross_group(self):
        three = [
            {"op": "replace", "path": "z_model.train_params.lr", "old_value": 0.001, "new_value": 0.0005},
            {"op": "replace", "path": "z_model.train_params.dropout", "old_value": 0.0, "new_value": 0.1},
            {"op": "replace", "path": "z_model.train_params.hidden_size", "old_value": 64, "new_value": 72},
        ]
        outcome = validate_proposal(REPO_ROOT, REPO_ROOT, self.config, "EXP_002", self.clem_fama, self._fama(diff=three))
        self.assertIn("SCHEMA_INVALID", outcome.reason_codes)
        cross = [{"op": "replace", "path": "z_model.model_params.model_layer", "old_value": 2, "new_value": 3}]
        outcome = validate_proposal(REPO_ROOT, REPO_ROOT, self.config, "EXP_002", self.clem_fama, self._fama(diff=cross))
        self.assertIn("FAMA_CROSS_GROUP_PATH", outcome.reason_codes)

    def test_frozen_field_and_rapa_exactly_one(self):
        illegal = [{"op": "replace", "path": "z_portfolio.max_weight", "old_value": 0.02, "new_value": 0.03}]
        outcome = validate_proposal(REPO_ROOT, REPO_ROOT, self.config, "EXP_002", self.clem_rapa, self._rapa(illegal))
        self.assertFalse(outcome.passed)
        self.assertTrue({"SCHEMA_INVALID", "RAPA_PATH_OUTSIDE_SPACE"} & set(outcome.reason_codes))
        two = [
            {"op": "replace", "path": "z_portfolio.risk_aversion", "old_value": 1, "new_value": 0.8},
            {"op": "replace", "path": "z_portfolio.turnover_penalty", "old_value": 1, "new_value": 1.2},
        ]
        outcome = validate_proposal(REPO_ROOT, REPO_ROOT, self.config, "EXP_002", self.clem_rapa, self._rapa(two))
        self.assertIn("SCHEMA_INVALID", outcome.reason_codes)

    def test_rass_exact_append_and_freeze_preconditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            entries = [{"id": f"F{i}", "expression": f"Expr{i}", "category": "rolling", "eligible": True} for i in range(4)]
            evidence = {
                "records": entries, "input_data_hash": "data", "catalog_hash": "cat",
                "label_hash": "label", "effective_config_hash": "a", "evidence_code_hash": "code",
                "evidence_hash": "evidence",
            }
            atomic_write_json(run_root / "configs" / "frozen_feature_catalog.json", {"entries": entries})
            atomic_write_json(run_root / "configs" / "rass_train_evidence.json", evidence)
            shortlist_evidence = {
                "method": "agent_shortlist_joint_train_evidence_v1",
                "base_evidence_hash": "evidence",
                "shortlist": [{"id": item["id"], "expression": item["expression"]} for item in entries],
                "contains_test_derived_data": False,
            }
            shortlist_hash = sha256_bytes(canonical_json_bytes(shortlist_evidence))
            shortlist_evidence["evidence_hash"] = shortlist_hash
            atomic_write_json(run_root / "experiments" / "EXP_001" / "logs" / "rass_shortlist_evidence.json", shortlist_evidence)
            proposal = {
                "agent": "RASS", "method": "agent_factor_exploration", "data_split": "train",
                "selected_group": "initial_feature_bootstrap", "target_subset_size": 10,
                "initial_features": self.config["z_alpha"]["selected_features"],
                "added_features": [{"id": item["id"], "expression": item["expression"]} for item in entries],
                "selected_features": self.config["z_alpha"]["selected_features"] + [item["expression"] for item in entries],
                "hypothesis": "complementary breadth", "local_diagnosis": "narrow anchors",
                "reasoning": "train evidence supports complementary signals", "alternatives_considered": ["other"],
                "expected_signature": [{"metric": "ic", "direction": "stable_or_improve"}],
                "falsification_condition": "mechanism materially opposite", "confidence": 0.8,
                "alpha_frozen_after_success": True, "evidence_refs": ["configs/rass_train_evidence.json", "logs/rass_shortlist_evidence.json", f"evidence_hash: {shortlist_hash}"],
                "provenance": {"data_hash": "data", "catalog_hash": "cat", "label_hash": "label", "config_hash": "a", "evidence_code_hash": "code", "shortlist_evidence_hash": shortlist_hash},
            }
            clem = {"selected_agent": "RASS", "selected_layer": "alpha", "confidence": 1.0}
            outcome = validate_proposal(REPO_ROOT, run_root, self.config, "EXP_001", clem, proposal)
            self.assertTrue(outcome.passed, outcome.reason_codes)
            proposal["evidence_refs"].append("evidence_hash: evidence")
            outcome = validate_proposal(REPO_ROOT, run_root, self.config, "EXP_001", clem, proposal)
            self.assertTrue(outcome.passed, outcome.reason_codes)
            proposal["selected_features"] = proposal["selected_features"][1:]
            outcome = validate_proposal(REPO_ROOT, run_root, self.config, "EXP_001", clem, proposal)
            self.assertIn("SCHEMA_INVALID", outcome.reason_codes)


if __name__ == "__main__":
    unittest.main()
