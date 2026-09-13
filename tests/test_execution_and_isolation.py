from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from adapters.portfolio_runner import preprocess_prediction_cross_section, run_mean_variance
from core.executor import dependency_plan
from core.isolation import AdaptiveReader, IsolationError, assert_validation_safe_value
from core.split_guard import SplitWindow, aligned_trade_return_dates, purged_signal_dates


class IsolationTests(unittest.TestCase):
    def test_physical_test_path_denial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "experiments" / "EXP_000" / "test" / "result.json"
            path.parent.mkdir(parents=True)
            path.write_text("{}", encoding="utf-8")
            with self.assertRaises(IsolationError):
                AdaptiveReader([root]).read_json(path)
        with self.assertRaises(IsolationError):
            assert_validation_safe_value({"payload": "experiments/EXP_000/test/result.json"})


class AlignmentTests(unittest.TestCase):
    def test_t_t1_t2_and_boundary_purge(self):
        calendar = pd.date_range("2024-01-01", periods=5, freq="D")
        window = SplitWindow("valid", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-04"))
        signal = purged_signal_dates(calendar, window)
        self.assertEqual(signal, list(calendar[:2]))
        alignment = aligned_trade_return_dates(calendar, signal)
        self.assertEqual(alignment[calendar[0]], (calendar[1], calendar[2]))


class PortfolioTests(unittest.TestCase):
    def _config(self):
        return {"z_portfolio": {
            "lookback": 60, "min_lookback": 20, "risk_model_mode": "diagonal",
            "variance_floor": 1e-6, "cov_shrink_to_diag": 0.5, "risk_aversion": 1.0,
            "turnover_penalty": 1.0, "max_iter": 100, "tol": 1e-10,
            "weight_threshold_for_drop": 1e-8, "max_weight": 0.6, "min_weight": 0.0,
            "full_position": True, "candidate_count": 2, "alpha_scale": 0.001,
            "buy_commission": 0.0001, "buy_slippage": 0.0005,
            "sell_commission": 0.0001, "sell_tax": 0.0005, "sell_slippage": 0.0005,
        }}

    def test_prediction_preprocessing_constant_maps_zero(self):
        result = preprocess_prediction_cross_section(pd.Series([2.0, 2.0], index=["A", "B"]))
        self.assertTrue((result == 0).all())

    def test_directional_limit_up_blocks_buy(self):
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        alpha = pd.DataFrame([[3.0, 2.0]], index=[dates[0]], columns=["A", "B"])
        returns = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
        up = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
        down = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
        up.loc[dates[1], "A"] = 1.0
        run = run_mean_variance(self._config(), alpha, returns, up, down, {dates[0]: (dates[1], dates[2])}, pd.Series({dates[0]: 0.0}))
        buys = run.orders[run.orders["side"] == "BUY"]
        self.assertNotIn("A", set(buys["instrument"]))
        self.assertLessEqual(float(run.diagnostics.iloc[0]["invested_weight"]), 0.6 + 1e-10)


class DependencyTests(unittest.TestCase):
    def test_dependency_aware_reruns(self):
        self.assertNotIn("model", dependency_plan("RAPA")[0])
        self.assertEqual(dependency_plan("FAMA")[0], ["model", "predictions", "portfolio", "validation"])
        self.assertEqual(dependency_plan("RASS")[0][0], "factors")


if __name__ == "__main__":
    unittest.main()

