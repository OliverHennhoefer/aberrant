"""Integration test for the MStream model."""

import unittest

from aberrant.model.sketch import MStream


class TestMStream(unittest.TestCase):
    """Test MStream's complete-record interaction signal."""

    def test_novel_record_interaction_scores_above_familiar_record(self) -> None:
        """
        Detect a novel combination whose individual attribute values are familiar.

        Original MStream adds a complete-record sketch specifically so this case
        is detectable even when every singleton attribute value is common.
        """
        model = MStream(
            rows=4,
            buckets=2048,
            alpha=0.5,
            time_key="t",
            seed=42,
        )

        for timestamp in range(1, 31):
            for _ in range(8):
                model.learn_one({"x": 0.0, "y": 0.0, "t": float(timestamp)})
                model.learn_one({"x": 10.0, "y": 10.0, "t": float(timestamp)})

        familiar_score = model.score_one({"x": 0.0, "y": 0.0, "t": 31.0})
        novel_interaction_score = model.score_one({"x": 0.0, "y": 10.0, "t": 31.0})

        self.assertGreater(novel_interaction_score, familiar_score)


if __name__ == "__main__":
    unittest.main()
