"""Integration test for X-Lag Amnesic DAMP on a synthetic time series."""

import unittest

import numpy as np
from sklearn.metrics import average_precision_score

from aberrant.model.timeseries import XLagDAMP


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
            np.sin(
                6.0
                * np.pi
                * np.arange(subsequence_length)
                / subsequence_length
            )
        ) + np.linspace(-1.0, 1.0, subsequence_length)
        series[discord_start : discord_start + subsequence_length] = discord

        model = XLagDAMP(
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
                    discord_start
                    <= index
                    <= discord_start + 2 * subsequence_length - 2
                )
            )

        evaluation_start = 5 * subsequence_length
        pr_auc = average_precision_score(
            labels[evaluation_start:],
            scores[evaluation_start:],
        )
        self.assertGreaterEqual(pr_auc, 0.95)
        self.assertLessEqual(pr_auc, 1.0)


if __name__ == "__main__":
    unittest.main()
