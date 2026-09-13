from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from core.io_utils import atomic_write_json, load_json
from core.state_store import StateStore
from core.validation_gate import evaluate_gate
from memory.memory_store import MemoryStore


def diagnostics(scale: float = 1.0) -> pd.DataFrame:
    n = 80
    base = np.sin(np.arange(n) / 5.0) * 0.005 + 0.0005
    return pd.DataFrame({
        "signal_date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "net_return": base * scale,
        "one_way_turn": 0.02,
        "transaction_cost_drag": 0.0001,
        "portfolio_concentration": 0.02,
        "invested_weight_shortfall": 0.0,
    })


class GateTests(unittest.TestCase):
    def test_machine_precision_noise_is_not_material_degradation(self):
        parent = diagnostics(1.0)
        candidate = diagnostics(1.0)
        candidate["net_return"] *= 1.0 + 1e-14
        for column in [
            "one_way_turn",
            "transaction_cost_drag",
            "portfolio_concentration",
            "invested_weight_shortfall",
        ]:
            candidate[column] += 5e-16
        signal = pd.DataFrame(
            {
                "ic": np.linspace(0.01, 0.02, 80),
                "rank_ic": np.linspace(0.01, 0.02, 80),
            },
            index=pd.date_range("2024-01-01", periods=80),
        )

        gate = evaluate_gate(
            parent,
            candidate,
            signal,
            signal,
            [{"metric": "ic", "direction": "increase"}],
        )

        self.assertEqual(gate["sharpe"]["state"], "UNCHANGED")
        self.assertEqual(gate["tradeoff_state"], "NO_MATERIAL_DEGRADATION")
        self.assertTrue(all(item["state"] == "NO_MATERIAL_DEGRADATION" for item in gate["guardrails"]))

    def test_materially_worse_sharpe_falsifies(self):
        parent = diagnostics(1.0)
        candidate = diagnostics(1.0)
        parent["net_return"] = 0.001 + np.sin(np.arange(len(parent)) / 5.0) * 0.0001
        candidate["net_return"] = -0.001 + np.sin(np.arange(len(candidate)) / 5.0) * 0.0001
        signal = pd.DataFrame({"ic": np.linspace(0.01, 0.02, 80), "rank_ic": np.linspace(0.01, 0.02, 80)}, index=pd.date_range("2024-01-01", periods=80))
        gate = evaluate_gate(parent, candidate, signal, signal, [{"metric": "ic", "direction": "increase"}])
        self.assertEqual(gate["gate_verdict"], "FALSIFIED")
        self.assertFalse(gate["promotion"])

    def test_structural_rass_promotes_audit_verdict(self):
        frame = diagnostics(1.0)
        signal = pd.DataFrame({"ic": np.linspace(0.01, 0.02, 80), "rank_ic": np.linspace(0.01, 0.02, 80)}, index=pd.date_range("2024-01-01", periods=80))
        gate = evaluate_gate(frame, frame, signal, signal, [{"metric": "ic", "direction": "increase"}], structural_rass_promotion=True)
        self.assertTrue(gate["promotion"])
        self.assertEqual(gate["promotion_reason"], "mandatory_rass_structural_initialization")
        self.assertIn(gate["gate_verdict"], {"SUPPORTED", "PARTIALLY_SUPPORTED", "UNCERTAIN", "FALSIFIED"})


class StateAndMemoryTests(unittest.TestCase):
    def test_promotion_rollback_and_alpha_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {"z_alpha": {"alpha_frozen": False, "selected_features": list(range(10))}, "z_model": {"derived": {"d_feat": 10}}, "provenance": {}}
            atomic_write_json(root / "configs" / "current.json", config)
            atomic_write_json(root / "configs" / "state.json", {"accepted_experiment_id": "EXP_000", "next_round": 1, "status": "ADAPTING"})
            store = StateStore(root)
            store.rollback("EXP_001", 2)
            self.assertEqual(load_json(root / "configs" / "current.json"), config)
            store.promote(config, "EXP_001", 2, structural_rass=True)
            self.assertTrue(load_json(root / "configs" / "current.json")["z_alpha"]["alpha_frozen"])
            self.assertEqual(store.state()["accepted_experiment_id"], "EXP_001")

    def test_memory_keeps_rejected_records_and_all_projections(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))
            for index, accepted in enumerate([True, False]):
                store.append({
                    "experiment_id": f"EXP_{index:03d}", "round": index, "layer": "model",
                    "state": "s", "clem_diagnosis": "d", "group": "train_params", "intervention": {},
                    "expected_signature": {}, "observed_signature": {}, "verdict": "BASELINE" if index == 0 else "FALSIFIED",
                    "accepted": accepted, "test_derived": False,
                })
            self.assertEqual(len(store.projection("model")), 2)
            self.assertFalse(store.projection("model")[1]["accepted"])


if __name__ == "__main__":
    unittest.main()
