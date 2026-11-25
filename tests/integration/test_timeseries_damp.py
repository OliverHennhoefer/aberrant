"""Integration test for X-Lag Amnesic DAMP on a synthetic time series."""

import unittest

import numpy as np

from aberrant.model.timeseries import DAMP
from tests.integration._quality import assert_detects

# Measured on main at ROC-AUC 1.000 and PR-AUC 1.000; the floors leave a margin.
ROC_FLOOR = 0.99
PR_FLOOR = 0.95


class TestXLagDAMPIntegration(unittest.TestCase):
    """Verify that DAMP locates a novel subsequence in a periodic stream."""

    def test_synthetic_discord_pr_auc(self) -> None:
        rng = np.random.default_rng(42)
        subsequence_length = 24
        period = np.sin(
            2.0 * np.pi * np.arange(subsequence_length) / subsequence_length
        )
        series = np.tile(period, 50).astype(np.float64)
        series += rng.normal(0.0, 0.01, series.size)

        discord_start = 25 * subsequence_length
        discord = np.sign(
            np.sin(6.0 * np.pi * np.arange(subsequence_length) / subsequence_length)
        ) + np.linspace(-1.0, 1.0, subsequence_length)
        series[discord_start : discord_start + subsequence_length] = discord

        model = DAMP(
            subsequence_length=subsequence_length,
            x_lag=16 * subsequence_length,
            start_index=4 * subsequence_length,
        )
        labels: list[int] = []
        scores: list[float] = []

        for index, value in enumerate(series):
            scores.append(model.score_one({"value": float(value)}))
            model.learn_one({"value": float(value)})
            labels.append(
                int(
                    discord_start <= index <= discord_start + 2 * subsequence_length - 2
                )
            )

        evaluation_start = 5 * subsequence_length
        assert_detects(
            labels[evaluation_start:],
            scores[evaluation_start:],
            roc_auc_floor=ROC_FLOOR,
            pr_auc_floor=PR_FLOOR,
        )


if __name__ == "__main__":
    unittest.main()
