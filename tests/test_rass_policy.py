from __future__ import annotations

import json
import unittest
from pathlib import Path

import yaml

from core.rass_policy import (
    RASS_MAX_FAILED_ATTEMPTS,
    rass_attempt_summaries,
    rass_attempted_addition_signatures,
    rass_failed_attempt_count,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def record(index: int, features: list[str], verdict: str, accepted: bool) -> dict:
    return {
        "experiment_id": f"EXP_{index:03d}",
        "round": index,
        "layer": "alpha",
        "group": "initial_feature_bootstrap",
        "intervention": {
            "added_features": [{"expression": expression} for expression in features],
        },
        "observed_signature": {
            "validation_ic": 0.01 * index,
            "validation_sharpe": 0.1 * index,
        },
        "verdict": verdict,
        "accepted": accepted,
    }


class RassPolicyTests(unittest.TestCase):
    def test_failure_limit_is_two_complete_rejections(self):
        self.assertEqual(RASS_MAX_FAILED_ATTEMPTS, 2)
        contract = json.loads((REPO_ROOT / "configs" / "frozen_contract.json").read_text(encoding="utf-8"))
        space = yaml.safe_load((REPO_ROOT / "agents" / "RASS" / "intervention_space.yaml").read_text(encoding="utf-8"))
        self.assertEqual(contract["validation_gate"]["rass_bounded_initial_search"]["maximum_failed_executed_candidates"], 2)
        self.assertEqual(space["groups"]["initial_feature_bootstrap"]["maximum_failed_executed_candidates"], 2)

    def test_counts_only_complete_gate_failures(self):
        records = [
            record(1, ["A", "B", "C"], "FALSIFIED", False),
            {**record(2, ["A", "B", "D"], "CONTRACT_REJECTED", False)},
            record(3, ["A", "C", "D"], "UNCERTAIN", False),
        ]
        self.assertEqual(rass_failed_attempt_count(records), 2)

    def test_addition_signature_is_order_independent_and_overlap_is_allowed(self):
        records = [record(1, ["A", "B", "C"], "FALSIFIED", False)]
        signatures = rass_attempted_addition_signatures(records)
        self.assertIn(("A", "B", "C"), signatures)
        self.assertNotIn(("A", "B", "D"), signatures)

    def test_retry_context_contains_validation_safe_outcome(self):
        summaries = rass_attempt_summaries([
            record(1, ["A", "B", "C"], "FALSIFIED", False),
        ])
        self.assertEqual(summaries[0]["added_features"], ["A", "B", "C"])
        self.assertEqual(summaries[0]["validation_sharpe"], 0.1)
        self.assertFalse(summaries[0]["accepted"])


if __name__ == "__main__":
    unittest.main()
