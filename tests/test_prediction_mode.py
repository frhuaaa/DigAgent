from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from adapters.qlib_runner import benchmark_returns
from core.errors import ContractError
from core.prediction_mode import apply_prediction_mode, prediction_mode


class PredictionModeTests(unittest.TestCase):
    @staticmethod
    def config(mode: str) -> dict:
        return {
            "task": {"prediction_mode": mode, "benchmark": "SH000852"},
            "z_portfolio": {
                "ret_path": "stale.csv",
                "limit_up_mask_path": "stale_up.csv",
                "limit_down_mask_path": "stale_down.csv",
                "buy_slippage": 9.0,
                "sell_slippage": 9.0,
            },
        }

    def test_c2c_is_default_and_binds_complete_execution_contract(self):
        config = {"task": {}, "z_portfolio": {}}
        apply_prediction_mode(config)
        self.assertEqual(prediction_mode(config), "c2c")
        self.assertEqual(config["task"]["target"], "Ref($close, -2) / Ref($close, -1) - 1")
        self.assertEqual(config["z_portfolio"]["ret_path"], "./data/portfolio/c_2_c_1D.csv")
        self.assertEqual(config["z_portfolio"]["buy_slippage"], 0.0005)

    def test_o2o_binds_label_returns_masks_and_open_slippage(self):
        config = self.config("O2O")
        apply_prediction_mode(config)
        self.assertEqual(config["task"]["prediction_mode"], "o2o")
        self.assertEqual(config["task"]["target"], "Ref($open, -2) / Ref($open, -1) - 1")
        self.assertEqual(config["z_portfolio"]["ret_path"], "./data/portfolio/o_2_o_1D.csv")
        self.assertEqual(config["z_portfolio"]["limit_up_mask_path"], "./data/portfolio/mask_limit_up_open_1D.csv")
        self.assertEqual(config["z_portfolio"]["limit_down_mask_path"], "./data/portfolio/mask_limit_down_open_1D.csv")
        self.assertEqual(config["z_portfolio"]["execution_price"], "open")
        self.assertEqual(config["z_portfolio"]["buy_slippage"], 0.001)
        self.assertEqual(config["z_portfolio"]["sell_slippage"], 0.001)

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ContractError):
            apply_prediction_mode(self.config("vwap2vwap"))

    def test_o2o_benchmark_uses_open_prices_with_t1_t2_alignment(self):
        dates = pd.date_range("2025-01-02", periods=3, freq="D")
        index = pd.MultiIndex.from_product(
            [["SH000852"], dates], names=["instrument", "datetime"]
        )
        prices = pd.DataFrame({"$open": [10.0, 11.0, 12.0]}, index=index)
        config = self.config("o2o")
        apply_prediction_mode(config)
        with patch("adapters.qlib_runner.D") as qlib_data:
            qlib_data.features.return_value = prices
            result = benchmark_returns(config, list(dates), [dates[0]])
        self.assertAlmostEqual(float(result.iloc[0]), 12.0 / 11.0 - 1.0)
        self.assertEqual(qlib_data.features.call_args.args[1], ["$open"])


if __name__ == "__main__":
    unittest.main()
