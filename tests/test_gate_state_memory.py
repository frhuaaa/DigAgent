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


def diagnostics_with_sharpe(target: float) -> pd.DataFrame:
    frame = diagnostics()
    centered = np.sin(np.arange(len(frame)) / 5.0)
    centered = (centered - centered.mean()) / centered.std(ddof=1)
    volatility = 0.01
    frame["net_return"] = centered * volatility + target / np.sqrt(252.0) * volatility
    return frame


def ensemble_metrics(parent: float, deltas: tuple[float, float, float]) -> tuple[dict, dict]:
    return (
        {"seed_signal_metrics": [{"seed": seed, "ic": parent} for seed in (0, 1, 2)]},
        {
            "seed_signal_metrics": [
                {"seed": seed, "ic": parent + delta}
                for seed, delta in zip((0, 1, 2), deltas)
            ]
        },
    )


class GateTests(unittest.TestCase):
    def test_fama_ic_improvement_cannot_override_falsified_global_gate(self):
        parent = diagnostics_with_sharpe(1.0)
        candidate = diagnostics_with_sharpe(0.5)
        parent_signal = pd.DataFrame(
            {"ic": np.full(80, 0.05), "rank_ic": np.full(80, 0.02)},
            index=pd.date_range("2024-01-01", periods=80),
        )
        candidate_signal = parent_signal.copy()
        candidate_signal["ic"] += 0.0011
        gate = evaluate_gate(
            parent,
            candidate,
            parent_signal,
            candidate_signal,
            [{"metric": "ic", "direction": "increase"}],
            fama_ic_override=True,
        )
        self.assertEqual(gate["gate_verdict"], "FALSIFIED")
        self.assertFalse(gate["promotion"])
        self.assertEqual(gate["promotion_reason"], "rollback")
        self.assertFalse(gate["fama_ic_override"]["applied"])

    def test_fama_ic_override_requires_ensemble_and_seed_consistency(self):
        parent = diagnostics_with_sharpe(1.0)
        candidate = diagnostics_with_sharpe(1.0)
        parent_signal = pd.DataFrame(
            {"ic": np.full(80, 0.05), "rank_ic": np.full(80, 0.02)},
            index=pd.date_range("2024-01-01", periods=80),
        )
        candidate_signal = parent_signal.copy()
        candidate_signal["ic"] += 0.0012
        parent_ensemble = {
            "seed_signal_metrics": [{"seed": seed, "ic": 0.05} for seed in (0, 1, 2)]
        }
        candidate_ensemble = {
            "seed_signal_metrics": [
                {"seed": 0, "ic": 0.0513},
                {"seed": 1, "ic": 0.0512},
                {"seed": 2, "ic": 0.0498},
            ]
        }
        gate = evaluate_gate(
            parent,
            candidate,
            parent_signal,
            candidate_signal,
            [{"metric": "unsupported_metric", "direction": "increase"}],
            fama_ic_override=True,
            parent_ensemble_diagnostics=parent_ensemble,
            candidate_ensemble_diagnostics=candidate_ensemble,
        )
        self.assertEqual(gate["gate_verdict"], "UNCERTAIN")
        self.assertTrue(gate["promotion"])
        self.assertTrue(gate["fama_ic_override"]["seed_consistent"])
        self.assertTrue(gate["fama_ic_override"]["applied"])

    def test_fama_ic_override_cannot_hide_temporal_sharpe_reversal(self):
        parent = diagnostics_with_sharpe(1.0)
        candidate = parent.copy()
        midpoint = len(candidate) // 2
        candidate.loc[candidate.index[:midpoint], "net_return"] += 0.0002
        candidate.loc[candidate.index[midpoint:], "net_return"] -= 0.0002
        parent_signal = pd.DataFrame(
            {"ic": np.full(80, 0.05), "rank_ic": np.full(80, 0.02)},
            index=pd.date_range("2024-01-01", periods=80),
        )
        candidate_signal = parent_signal.copy()
        candidate_signal["ic"] += 0.0012
        parent_ensemble, candidate_ensemble = ensemble_metrics(0.05, (0.0012, 0.0013, 0.0014))
        gate = evaluate_gate(
            parent,
            candidate,
            parent_signal,
            candidate_signal,
            [{"metric": "unsupported_metric", "direction": "increase"}],
            fama_ic_override=True,
            parent_ensemble_diagnostics=parent_ensemble,
            candidate_ensemble_diagnostics=candidate_ensemble,
        )
        self.assertFalse(gate["partial_promotion_stability"]["temporal"]["no_material_reversal"])
        self.assertFalse(gate["fama_ic_override"]["applied"])
        self.assertFalse(gate["promotion"])

    def test_fama_ic_change_at_or_below_threshold_does_not_override(self):
        parent = diagnostics_with_sharpe(1.0)
        candidate = diagnostics_with_sharpe(0.5)
        parent_signal = pd.DataFrame(
            {"ic": np.full(80, 0.05), "rank_ic": np.full(80, 0.02)},
            index=pd.date_range("2024-01-01", periods=80),
        )
        candidate_signal = parent_signal.copy()
        candidate_signal["ic"] += 0.001
        gate = evaluate_gate(
            parent,
            candidate,
            parent_signal,
            candidate_signal,
            [{"metric": "ic", "direction": "increase"}],
            fama_ic_override=True,
        )
        self.assertFalse(gate["promotion"])
        self.assertFalse(gate["fama_ic_override"]["applied"])

    def test_sharpe_delta_strictly_above_point_zero_zero_two_promotes(self):
        signal = pd.DataFrame({"ic": np.linspace(0.01, 0.02, 80), "rank_ic": np.linspace(0.01, 0.02, 80)}, index=pd.date_range("2024-01-01", periods=80))
        accepted = evaluate_gate(diagnostics_with_sharpe(1.0), diagnostics_with_sharpe(1.0021), signal, signal, [{"metric": "ic", "direction": "increase"}])
        rejected = evaluate_gate(diagnostics_with_sharpe(1.0), diagnostics_with_sharpe(1.0019), signal, signal, [{"metric": "ic", "direction": "increase"}])
        self.assertTrue(accepted["promotion"])
        self.assertEqual(accepted["gate_verdict"], "PARTIALLY_SUPPORTED")
        self.assertEqual(accepted["sharpe"]["acceptance_delta"], 0.002)
        self.assertFalse(rejected["promotion"])
        self.assertEqual(rejected["gate_verdict"], "UNCERTAIN")

    def test_partial_support_requires_both_chronological_halves_to_be_stable(self):
        parent = diagnostics_with_sharpe(1.0)
        candidate = parent.copy()
        midpoint = len(candidate) // 2
        candidate.loc[candidate.index[:midpoint], "net_return"] += 0.0008
        candidate.loc[candidate.index[midpoint:], "net_return"] -= 0.0002
        signal = pd.DataFrame(
            {"ic": np.linspace(0.01, 0.02, 80), "rank_ic": np.linspace(0.01, 0.02, 80)},
            index=pd.date_range("2024-01-01", periods=80),
        )
        gate = evaluate_gate(
            parent,
            candidate,
            signal,
            signal,
            [{"metric": "ic", "direction": "increase"}],
        )
        self.assertEqual(gate["partial_promotion_stability"]["temporal"]["periods"][1]["state"], "WORSE")
        self.assertEqual(gate["gate_verdict"], "UNCERTAIN")
        self.assertFalse(gate["promotion"])

    def test_model_partial_support_requires_majority_seed_stability(self):
        parent = diagnostics_with_sharpe(1.0)
        candidate = diagnostics_with_sharpe(1.01)
        signal = pd.DataFrame(
            {"ic": np.linspace(0.01, 0.02, 80), "rank_ic": np.linspace(0.01, 0.02, 80)},
            index=pd.date_range("2024-01-01", periods=80),
        )
        parent_ensemble, candidate_ensemble = ensemble_metrics(0.05, (0.001, -0.002, -0.001))
        gate = evaluate_gate(
            parent,
            candidate,
            signal,
            signal,
            [{"metric": "ic", "direction": "increase"}],
            fama_ic_override=True,
            parent_ensemble_diagnostics=parent_ensemble,
            candidate_ensemble_diagnostics=candidate_ensemble,
        )
        self.assertFalse(gate["partial_promotion_stability"]["model_seed_stable"])
        self.assertEqual(gate["gate_verdict"], "UNCERTAIN")
        self.assertFalse(gate["promotion"])

    def test_fixed_sharpe_improvement_does_not_override_opposite_mechanism(self):
        parent_signal = pd.DataFrame({"ic": np.linspace(0.03, 0.04, 80), "rank_ic": np.linspace(0.03, 0.04, 80)}, index=pd.date_range("2024-01-01", periods=80))
        candidate_signal = pd.DataFrame({"ic": np.linspace(-0.04, -0.03, 80), "rank_ic": np.linspace(-0.04, -0.03, 80)}, index=parent_signal.index)
        gate = evaluate_gate(diagnostics_with_sharpe(1.0), diagnostics_with_sharpe(1.3), parent_signal, candidate_signal, [{"metric": "ic", "direction": "increase"}])
        self.assertEqual(gate["sharpe"]["state"], "IMPROVED")
        self.assertEqual(gate["mechanism_state"], "OPPOSITE")
        self.assertEqual(gate["gate_verdict"], "FALSIFIED")
        self.assertFalse(gate["promotion"])

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

    def test_turnover_increase_is_not_an_independent_tradeoff_failure(self):
        parent = diagnostics(1.0)
        candidate = diagnostics(1.0)
        candidate["one_way_turn"] = 0.04
        signal = pd.DataFrame({"ic": np.linspace(0.01, 0.02, 80), "rank_ic": np.linspace(0.01, 0.02, 80)}, index=pd.date_range("2024-01-01", periods=80))
        gate = evaluate_gate(
            parent,
            candidate,
            signal,
            signal,
            [{"metric": "one_way_turnover_mean", "direction": "increase"}],
        )
        self.assertNotEqual(gate["tradeoff_state"], "MATERIAL_DEGRADATION")

    def test_rass_bootstrap_flag_does_not_override_gate(self):
        frame = diagnostics(1.0)
        signal = pd.DataFrame({"ic": np.linspace(0.01, 0.02, 80), "rank_ic": np.linspace(0.01, 0.02, 80)}, index=pd.date_range("2024-01-01", periods=80))
        gate = evaluate_gate(frame, frame, signal, signal, [{"metric": "ic", "direction": "increase"}], freeze_alpha_on_promotion=True)
        self.assertFalse(gate["promotion"])
        self.assertEqual(gate["promotion_reason"], "rollback")
        self.assertFalse(gate["alpha_frozen_after_promotion"])


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
            store.promote(config, "EXP_001", 2, freeze_alpha_on_promotion=True)
            self.assertTrue(load_json(root / "configs" / "current.json")["z_alpha"]["alpha_frozen"])
            self.assertEqual(store.state()["accepted_experiment_id"], "EXP_001")

    def test_three_failure_fallback_freezes_original_five_factor_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {
                "z_alpha": {"alpha_frozen": False, "selected_features": list(range(5))},
                "z_model": {"derived": {"d_feat": 5}},
                "provenance": {},
            }
            atomic_write_json(root / "configs" / "current.json", config)
            atomic_write_json(root / "configs" / "state.json", {
                "accepted_experiment_id": "EXP_000", "next_round": 3, "status": "ADAPTING",
            })
            store = StateStore(root)
            frozen = store.freeze_five_factor_alpha_after_rass_failures("EXP_003", 4, 3)
            self.assertTrue(frozen["z_alpha"]["alpha_frozen"])
            self.assertEqual(frozen["z_alpha"]["selected_features"], list(range(5)))
            self.assertEqual(frozen["z_model"]["derived"]["d_feat"], 5)
            self.assertEqual(store.state()["accepted_experiment_id"], "EXP_000")
            self.assertEqual(store.state()["rass_failed_attempts"], 3)

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
