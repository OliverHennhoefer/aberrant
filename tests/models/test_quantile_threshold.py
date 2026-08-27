"""Tests for QuantileThreshold adaptive threshold model."""

import unittest

from aberrant.model.quantile_threshold import QuantileThreshold


class TestQuantileThreshold(unittest.TestCase):
    """Test suite for QuantileThreshold model."""

    def test_initialization_default_params(self):
        """Test QuantileThreshold initialization with default parameters."""
        qt = QuantileThreshold()
        self.assertEqual(qt.quantile, 0.95)
        self.assertEqual(qt.window_size, 1000)
        self.assertEqual(qt.score_key, "score")
        self.assertIsNone(qt.threshold)
        self.assertEqual(qt.n_scores, 0)

    def test_initialization_custom_params(self):
        """Test QuantileThreshold initialization with custom parameters."""
        qt = QuantileThreshold(quantile=0.9, window_size=500, score_key="anomaly_score")
        self.assertEqual(qt.quantile, 0.9)
        self.assertEqual(qt.window_size, 500)
        self.assertEqual(qt.score_key, "anomaly_score")

    def test_invalid_quantile(self):
        """Test that invalid quantile values raise ValueError."""
        with self.assertRaises(ValueError) as context:
            QuantileThreshold(quantile=0)
        self.assertIn("quantile must be in (0, 1)", str(context.exception))

        with self.assertRaises(ValueError):
            QuantileThreshold(quantile=1)

        with self.assertRaises(ValueError):
            QuantileThreshold(quantile=-0.1)

        with self.assertRaises(ValueError):
            QuantileThreshold(quantile=1.1)

    def test_invalid_window_size(self):
        """Test that invalid window_size raises ValueError."""
        with self.assertRaises(ValueError) as context:
            QuantileThreshold(window_size=0)
        self.assertIn("window_size must be positive", str(context.exception))

        with self.assertRaises(ValueError):
            QuantileThreshold(window_size=-1)

    def test_invalid_score_key(self):
        """Test that empty score_key raises ValueError."""
        with self.assertRaises(ValueError) as context:
            QuantileThreshold(score_key="")
        self.assertIn("score_key cannot be empty", str(context.exception))

    def test_learn_one_missing_key(self):
        """Test learn_one raises ValueError for missing score key."""
        qt = QuantileThreshold()
        with self.assertRaises(ValueError) as context:
            qt.learn_one({"other_key": 0.5})
        self.assertIn("score", str(context.exception))

    def test_score_one_missing_key(self):
        """Test score_one raises ValueError for missing score key."""
        qt = QuantileThreshold()
        qt.learn_one({"score": 0.5})
        with self.assertRaises(ValueError) as context:
            qt.score_one({"other_key": 0.5})
        self.assertIn("score", str(context.exception))

    def test_learn_one_increases_count(self):
        """Test that learn_one increases score count."""
        qt = QuantileThreshold()
        for i in range(10):
            qt.learn_one({"score": float(i)})
        self.assertEqual(qt.n_scores, 10)

    def test_threshold_none_during_warmup(self):
        """Test threshold is None during warmup period."""
        qt = QuantileThreshold(window_size=1000)
        # Warmup requires at least 10% of window_size
        for i in range(5):
            qt.learn_one({"score": float(i)})
        self.assertIsNone(qt.threshold)

    def test_threshold_computed_after_warmup(self):
        """Test threshold is computed after warmup."""
        qt = QuantileThreshold(window_size=100)  # min_samples = 10
        for i in range(15):
            qt.learn_one({"score": float(i)})
        self.assertIsNotNone(qt.threshold)

    def test_score_one_during_warmup(self):
        """Test score_one returns 0 during warmup."""
        qt = QuantileThreshold(window_size=1000)
        qt.learn_one({"score": 0.5})
        score = qt.score_one({"score": 0.9})
        self.assertEqual(score, 0.0)

    def test_score_one_anomaly_detection(self):
        """Test that scores above threshold return 1.0."""
        qt = QuantileThreshold(quantile=0.9, window_size=100)
        # Add scores from 0 to 99
        for i in range(100):
            qt.learn_one({"score": float(i)})
        # 90th percentile should be around 90
        # Score above threshold should return 1.0
        score = qt.score_one({"score": 95.0})
        self.assertEqual(score, 1.0)

    def test_score_one_normal_normalized(self):
        """Test that scores below threshold are normalized."""
        qt = QuantileThreshold(quantile=0.9, window_size=100)
        for i in range(100):
            qt.learn_one({"score": float(i)})
        # Score below threshold should return normalized value
        # If threshold is ~90, then score of 45 should return ~0.5
        score = qt.score_one({"score": 45.0})
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_negative_scores_are_clipped_to_normalized_lower_bound(self):
        qt = QuantileThreshold(quantile=0.9, window_size=10)
        for score in range(10):
            qt.learn_one({"score": float(score)})

        self.assertEqual(qt.score_one({"score": -1.0}), 0.0)

    def test_threshold_updates_with_new_data(self):
        """Test that threshold updates as new data arrives."""
        qt = QuantileThreshold(quantile=0.9, window_size=100)
        # Initial data: scores 0-99
        for i in range(100):
            qt.learn_one({"score": float(i)})
        initial_threshold = qt.threshold
        # Add higher scores
        for i in range(100, 200):
            qt.learn_one({"score": float(i)})
        # Threshold should have increased
        self.assertGreater(qt.threshold, initial_threshold)

    def test_window_size_limit(self):
        """Test that window size is enforced."""
        qt = QuantileThreshold(window_size=50)
        for i in range(100):
            qt.learn_one({"score": float(i)})
        self.assertEqual(qt.n_scores, 50)

    def test_small_window_computes_threshold(self):
        """Test small windows still complete warmup and set a threshold."""
        qt = QuantileThreshold(window_size=5)
        for i in range(20):
            qt.learn_one({"score": float(i)})
        self.assertEqual(qt.n_scores, 5)
        self.assertIsNotNone(qt.threshold)

    def test_reset(self):
        """Test reset clears all state."""
        qt = QuantileThreshold()
        for i in range(100):
            qt.learn_one({"score": float(i)})
        qt.reset()
        self.assertEqual(qt.n_scores, 0)
        self.assertIsNone(qt.threshold)

    def test_custom_score_key(self):
        """Test using a custom score key."""
        qt = QuantileThreshold(score_key="my_score", window_size=50)
        for i in range(50):
            qt.learn_one({"my_score": float(i)})
        score = qt.score_one({"my_score": 45.0})
        self.assertIsInstance(score, float)

    def test_repr(self):
        """Test string representation."""
        qt = QuantileThreshold(quantile=0.9, window_size=500)
        repr_str = repr(qt)
        self.assertIn("QuantileThreshold", repr_str)
        self.assertIn("quantile=0.9", repr_str)
        self.assertIn("window_size=500", repr_str)


class TestQuantileThresholdEdgeCases(unittest.TestCase):
    """Test QuantileThreshold edge cases."""

    def test_non_finite_scores_are_rejected_without_poisoning_window(self):
        qt = QuantileThreshold(window_size=50)

        with self.assertRaisesRegex(ValueError, "score must be finite"):
            qt.learn_one({"score": float("nan")})
        with self.assertRaisesRegex(ValueError, "score must be finite"):
            qt.score_one({"score": float("inf")})

        self.assertEqual(qt.n_scores, 0)
        self.assertIsNone(qt.threshold)

    def test_zero_threshold_handling(self):
        """Test handling when all scores are zero (threshold would be 0)."""
        qt = QuantileThreshold(window_size=50)
        for _ in range(50):
            qt.learn_one({"score": 0.0})
        # Threshold should be 0 or very small
        # Score of 0 should not be detected as anomaly
        score = qt.score_one({"score": 0.0})
        self.assertEqual(score, 0.0)
        # Positive score should be anomaly
        score = qt.score_one({"score": 0.1})
        self.assertEqual(score, 1.0)

    def test_constant_scores(self):
        """Test with constant scores."""
        qt = QuantileThreshold(window_size=50)
        for _ in range(50):
            qt.learn_one({"score": 5.0})
        # Threshold should equal the constant value
        self.assertAlmostEqual(qt.threshold, 5.0, places=5)
        # Score equal to threshold should be anomaly (>=)
        score = qt.score_one({"score": 5.0})
        self.assertEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
