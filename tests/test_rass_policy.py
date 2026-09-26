from __future__ import annotations

import unittest

from core.rass_policy import (
    rass_attempt_summaries,
    rass_attempted_addition_signatures,
    rass_failed_attempt_count,
)


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
