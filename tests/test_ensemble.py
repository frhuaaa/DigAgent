from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from core.errors import ContractError
from training.ensemble import ensemble_predictions, ensemble_seeds


class EnsembleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = pd.MultiIndex.from_product(
            [pd.to_datetime(["2024-01-02", "2024-01-03"]), ["A", "B", "C", "D"]],
            names=["datetime", "instrument"],
        )

    def test_equal_weight_cross_sectional_ensemble_is_normalized(self):
        first = pd.Series([1, 2, 3, 100, 2, 4, 6, 8], index=self.index, dtype=float)
        second = pd.Series([4, 3, 2, 1, 8, 6, 4, 2], index=self.index, dtype=float)
        result = ensemble_predictions([first, second])
        self.assertTrue(result.index.equals(self.index))
        for _, values in result.groupby(level="datetime"):
            self.assertAlmostEqual(float(values.mean()), 0.0, places=12)
            if float(values.std(ddof=0)) > 0:
                self.assertAlmostEqual(float(values.std(ddof=0)), 1.0, places=12)

    def test_member_index_mismatch_is_rejected(self):
        first = pd.Series(np.arange(len(self.index)), index=self.index, dtype=float)
        second = first.iloc[:-1]
        with self.assertRaisesRegex(ContractError, "ENSEMBLE_PREDICTION_INDEX_MISMATCH"):
            ensemble_predictions([first, second])

    def test_frozen_seed_contract(self):
        config = {
            "task": {"seed": 0},
            "ensemble": {
                "enabled": True,
                "seeds": [0, 1, 2],
                "method": "cross_sectional_zscore_mean",
            },
        }
        self.assertEqual(ensemble_seeds(config), (0, 1, 2))


if __name__ == "__main__":
    unittest.main()
