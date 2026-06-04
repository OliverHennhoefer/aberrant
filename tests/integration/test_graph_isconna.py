"""Integration test for the ISCONNA graph model."""

import unittest

from aberrant.model.graph import ISCONNA


class TestISCONNA(unittest.TestCase):
    """Test ISCONNA on a stream with an anomalous consecutive-pattern change."""

    def test_long_gap_and_burst_scores_above_regular_pattern(self) -> None:
        """
        Exercise all three author-defined signals on a deterministic edge stream.

        The target edge first follows a regular alternating presence/absence
        pattern. It then disappears for an unusually long gap and returns in a
        burst. ISCONNA is specifically designed to score that combined
        frequency/width/gap pattern change.
        """
        model = ISCONNA(
            count_min_rows=2,
            count_min_cols=10_007,
            time_decay_factor=0.7,
            include_endpoints=False,
            warm_up_samples=0,
            seed=42,
        )
        target = {"src": 1.0, "dst": 2.0}
        filler = {"src": 3.0, "dst": 4.0}
        regular_scores: list[float] = []

        for timestamp in range(1, 25):
            edge = target if timestamp % 2 else filler
            sample = {**edge, "t": float(timestamp)}
            score = model.score_one(sample)
            model.learn_one(sample)
            if timestamp > 8 and timestamp % 2:
                regular_scores.append(score)

        for timestamp in range(25, 37):
            model.learn_one({**filler, "t": float(timestamp)})

        first_return = {**target, "t": 37.0}
        anomaly_score = model.score_one(first_return)
        model.learn_one(first_return)
        burst_score = model.score_one(first_return)

        self.assertTrue(any(score > 0.0 for score in regular_scores))
        self.assertGreater(anomaly_score, max(regular_scores))
        self.assertGreater(burst_score, max(regular_scores))


if __name__ == "__main__":
    unittest.main()
