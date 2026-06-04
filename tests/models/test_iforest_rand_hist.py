"""Unit tests for Stream Random Histogram Forest - Edge cases and initialization only.

Real dataset tests are in tests/integration/test_iforest_models.py
"""

import random
import unittest

from aberrant.model.iforest.rand_hist import StreamRandomHistogramForest


class TestStreamRandomHistogramForestEdgeCases(unittest.TestCase):
    """Test Stream Random Histogram Forest edge cases and initialization."""

    def create_model(self):
        return StreamRandomHistogramForest(
            n_estimators=5,  # Small for fast testing
            max_bins=5,
            window_size=50,
            seed=42,
        )

    def setUp(self):
        self.model = self.create_model()

    def test_initialization_valid_parameters(self):
        """Test initialization with valid parameters."""
        model = StreamRandomHistogramForest(
            n_estimators=10, max_bins=8, window_size=100, seed=123
        )

        self.assertEqual(model.n_estimators, 10)
        self.assertEqual(model.max_bins, 8)
        self.assertEqual(model.window_size, 100)
        self.assertEqual(model.seed, 123)

    def test_initialization_invalid_parameters(self):
        """Test initialization with invalid parameters raises errors."""
        # Invalid n_estimators
        with self.assertRaises((ValueError, AssertionError)):
            StreamRandomHistogramForest(n_estimators=0)

        # Invalid max_bins
        with self.assertRaises((ValueError, AssertionError)):
            StreamRandomHistogramForest(max_bins=0)

        # Invalid window_size
        with self.assertRaises((ValueError, AssertionError)):
            StreamRandomHistogramForest(window_size=0)

    def test_window_size_behavior(self):
        """Test window size constraint behavior."""
        model = StreamRandomHistogramForest(
            n_estimators=3,
            max_bins=4,
            window_size=10,  # Small window
            seed=42,
        )

        # Add more points than window size
        for i in range(20):
            point = {"feature1": float(i), "feature2": float(i * 2)}
            model.learn_one(point)

        # Test that model still functions
        test_point = {"feature1": 15.0, "feature2": 30.0}
        score = model.score_one(test_point)

        self.assertIsInstance(score, (int, float))
        self.assertGreaterEqual(score, 0.0)

    def test_different_bin_sizes(self):
        """Test different max_bins configurations."""
        for max_bins in [3, 5, 10]:
            with self.subTest(max_bins=max_bins):
                model = StreamRandomHistogramForest(
                    n_estimators=3, max_bins=max_bins, window_size=50, seed=42
                )

                # Should initialize without error
                self.assertEqual(model.max_bins, max_bins)

                # Basic functionality test
                point = {"feature1": 1.0, "feature2": 2.0}
                model.learn_one(point)
                score = model.score_one(point)

                self.assertIsInstance(score, (int, float))
                self.assertGreaterEqual(score, 0.0)

    def test_empty_dict_handling(self):
        """Test handling of empty feature dictionary."""
        model = self.create_model()

        try:
            model.learn_one({})
            score = model.score_one({})
            self.assertIsInstance(score, (int, float))
        except (ValueError, KeyError):
            # Acceptable to reject empty dict
            pass

    def test_single_feature_data(self):
        """Test with single feature data."""
        model = self.create_model()

        # Train with single feature
        for i in range(20):
            point = {"feature": float(i) % 10}  # Values 0-9 repeated
            model.learn_one(point)

        # Test scoring
        test_point = {"feature": 5.0}
        score = model.score_one(test_point)

        self.assertIsInstance(score, (int, float))
        self.assertGreaterEqual(score, 0.0)

    def test_deterministic_behavior(self):
        """Test deterministic behavior with same seed."""
        model1 = StreamRandomHistogramForest(
            n_estimators=3, max_bins=5, window_size=20, seed=42
        )
        model2 = StreamRandomHistogramForest(
            n_estimators=3, max_bins=5, window_size=20, seed=42
        )

        # Train both models identically
        training_points = [
            {"feature1": 1.0, "feature2": 2.0},
            {"feature1": 3.0, "feature2": 4.0},
            {"feature1": 5.0, "feature2": 6.0},
        ]

        for point in training_points:
            model1.learn_one(point.copy())
            model2.learn_one(point.copy())

        # Scores should be identical with same seed
        test_point = {"feature1": 2.5, "feature2": 3.5}
        score1 = model1.score_one(test_point)
        score2 = model2.score_one(test_point)

        self.assertEqual(score1, score2, "Same seed should produce identical results")

    def test_model_randomness_does_not_mutate_global_random_state(self):
        random.seed(123)
        expected_next = random.random()
        random.seed(123)

        model = self.create_model()
        model.learn_one({"feature": 1.0})

        self.assertEqual(random.random(), expected_next)

    def test_repeated_values(self):
        """Test behavior with repeated values (histogram binning edge case)."""
        model = self.create_model()

        # Train with repeated values
        repeated_point = {"feature1": 5.0, "feature2": 10.0}
        for _ in range(15):
            model.learn_one(repeated_point.copy())

        # Test with same value
        score1 = model.score_one(repeated_point)

        # Test with different value
        different_point = {"feature1": 25.0, "feature2": 50.0}
        score2 = model.score_one(different_point)

        self.assertIsInstance(score1, (int, float))
        self.assertIsInstance(score2, (int, float))
        self.assertGreaterEqual(score1, 0.0)
        self.assertGreaterEqual(score2, 0.0)

    def test_schema_change_raises_instead_of_zero_imputation(self):
        model = self.create_model()
        model.learn_one({"x": 0.1, "y": 0.2})

        with self.assertRaises(ValueError):
            model.learn_one({"x": 0.1})
        with self.assertRaises(ValueError):
            model.score_one({"x": 0.1})


if __name__ == "__main__":
    unittest.main()
