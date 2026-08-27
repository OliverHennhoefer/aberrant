"""Tests for ASD Isolation Forest anomaly detection model."""

import unittest

import numpy as np

from aberrant.model.iforest.asd import ASDIsolationForest, _ASDBranch
from tests.utils import DataGenerator


class TestASDIsolationForest(unittest.TestCase):
    """Test suite for ASDIsolationForest."""

    def create_model(self) -> ASDIsolationForest:
        """Create ASDIsolationForest instance for testing."""
        return ASDIsolationForest(n_estimators=10, max_samples=20, seed=42)

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.data_generator = DataGenerator(seed=42)

    def test_initialization_with_default_params(self):
        """Test ASDIsolationForest initialization with default parameters."""
        forest = ASDIsolationForest()

        self.assertEqual(forest.n_estimators, 100)
        self.assertEqual(forest.max_samples, 256)
        self.assertEqual(forest.window_size, 256)
        self.assertEqual(forest.retrain_interval, 256)
        self.assertIsNone(forest.seed)
        self.assertIsNone(forest._schema.names)
        self.assertEqual(len(forest.window), 0)
        self.assertEqual(len(forest.trees), 0)

    def test_initialization_with_custom_params(self):
        """Test ASDIsolationForest initialization with custom parameters."""
        forest = ASDIsolationForest(n_estimators=50, max_samples=128, seed=42)

        self.assertEqual(forest.n_estimators, 50)
        self.assertEqual(forest.max_samples, 128)
        self.assertEqual(forest.window_size, 128)
        self.assertEqual(forest.seed, 42)

    def test_initialization_validation(self):
        """Test parameter validation during initialization."""
        # Test invalid n_estimators
        with self.assertRaises(ValueError) as context:
            ASDIsolationForest(n_estimators=0)
        self.assertIn("n_estimators must be positive", str(context.exception))

        with self.assertRaises(ValueError) as context:
            ASDIsolationForest(n_estimators=-1)
        self.assertIn("n_estimators must be positive", str(context.exception))

        # Test invalid max_samples
        with self.assertRaises(ValueError) as context:
            ASDIsolationForest(max_samples=0)
        self.assertIn("max_samples must be greater than 1", str(context.exception))

        with self.assertRaises(ValueError) as context:
            ASDIsolationForest(max_samples=-1)
        self.assertIn("max_samples must be greater than 1", str(context.exception))

        with self.assertRaises(ValueError):
            ASDIsolationForest(max_samples=1)
        with self.assertRaises(ValueError):
            ASDIsolationForest(window_size=1)
        with self.assertRaises(ValueError):
            ASDIsolationForest(retrain_interval=0)

    def test_compute_c_static_method(self):
        """Test the static _compute_c method."""
        # Test edge cases
        self.assertEqual(ASDIsolationForest._compute_c(0), 0.0)
        self.assertEqual(ASDIsolationForest._compute_c(1), 0.0)
        self.assertEqual(ASDIsolationForest._compute_c(2), 1.0)

        # Test normal cases
        c_2 = ASDIsolationForest._compute_c(2)
        c_10 = ASDIsolationForest._compute_c(10)
        c_100 = ASDIsolationForest._compute_c(100)

        self.assertIsInstance(c_2, float)
        self.assertIsInstance(c_10, float)
        self.assertIsInstance(c_100, float)

        # c should increase with sample size
        self.assertGreater(c_10, c_2)
        self.assertGreater(c_100, c_10)

    def test_periodically_retrains_full_forest_on_recent_window(self):
        forest = ASDIsolationForest(
            n_estimators=3,
            max_samples=3,
            window_size=4,
            retrain_interval=2,
            seed=42,
        )
        for value in [1.0, 2.0, 3.0, 4.0]:
            forest.learn_one({"feature": value})

        self.assertEqual(len(forest.trees), 3)
        self.assertEqual(forest._fits_completed, 1)
        np.testing.assert_array_equal(
            forest._last_fit_window[:, 0],  # type: ignore[index]
            np.asarray([1.0, 2.0, 3.0, 4.0]),
        )

        forest.learn_one({"feature": 5.0})
        self.assertEqual(forest._fits_completed, 1)
        forest.learn_one({"feature": 6.0})
        self.assertEqual(forest._fits_completed, 2)
        np.testing.assert_array_equal(
            forest._last_fit_window[:, 0],  # type: ignore[index]
            np.asarray([3.0, 4.0, 5.0, 6.0]),
        )

    def test_tree_selects_from_variable_features(self):
        forest = ASDIsolationForest(max_samples=4, seed=42)
        data = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])

        tree = forest._build_tree(data)

        self.assertIsInstance(tree, _ASDBranch)
        assert isinstance(tree, _ASDBranch)
        self.assertEqual(tree.feature, 1)

    def test_feature_names_initialization(self):
        """Test feature names establishment on first sample."""
        forest = ASDIsolationForest(max_samples=5)

        # Before learning, no features established
        self.assertIsNone(forest._schema.names)
        self.assertEqual(len(forest.window), 0)

        # First data point establishes features
        first_point = {"c": 3.0, "a": 1.0, "b": 2.0}
        forest.learn_one(first_point)

        # Features should be sorted alphabetically
        expected_order = ["a", "b", "c"]
        self.assertEqual(forest._schema.names, tuple(expected_order))
        self.assertEqual(len(forest.window), 1)
        self.assertEqual(forest.window[0].shape, (3,))

    def test_empty_input_validation(self):
        """Test validation of empty input dictionaries."""
        forest = ASDIsolationForest()

        # Empty dictionary should raise error for learn_one
        with self.assertRaises(ValueError) as context:
            forest.learn_one({})
        self.assertIn("Input dictionary cannot be empty", str(context.exception))

        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            forest.score_one({})

    def test_tree_limit_enforcement(self):
        """Test that number of trees doesn't exceed n_estimators."""
        forest = ASDIsolationForest(n_estimators=3, max_samples=2, seed=42)

        # Generate enough data to create more trees than n_estimators
        for i in range(20):  # This should trigger multiple tree creations
            forest.learn_one({"feature": float(i)})

        # Every replacement is a complete forest.
        self.assertEqual(len(forest.trees), forest.n_estimators)

    def test_score_before_trees_created(self):
        """Test scoring before any trees are created."""
        forest = ASDIsolationForest(max_samples=10)

        # Score before learning anything
        score = forest.score_one({"feature": 1.0})
        self.assertEqual(score, 0.0)

        # Learn a few points (not enough to create tree)
        for i in range(5):
            forest.learn_one({"feature": float(i)})

        # Still no trees created, should return 0.0
        score = forest.score_one({"feature": 999.0})
        self.assertEqual(score, 0.0)

    def test_missing_features_handling(self):
        """Test missing features are rejected instead of silently imputed."""
        forest = ASDIsolationForest(max_samples=5, seed=42)

        # Learn with certain features
        training_data = [
            {"a": 1.0, "b": 2.0, "c": 3.0},
            {"a": 2.0, "b": 3.0, "c": 4.0},
            {"a": 3.0, "b": 4.0, "c": 5.0},
            {"a": 4.0, "b": 5.0, "c": 6.0},
            {"a": 5.0, "b": 6.0, "c": 7.0},  # This creates first tree
        ]

        for point in training_data:
            forest.learn_one(point)

        with self.assertRaises(ValueError):
            forest.score_one({"a": 1.0, "b": 2.0})  # Missing "c"

    def test_reproducibility_with_seed(self):
        """Test that same seed produces reproducible results."""
        # Create two forests with same seed
        forest1 = ASDIsolationForest(n_estimators=5, max_samples=10, seed=42)
        forest2 = ASDIsolationForest(n_estimators=5, max_samples=10, seed=42)

        # Generate test data
        test_data = self.data_generator.generate_streaming_data(n=50, n_features=2)

        scores1 = []
        scores2 = []

        # Train and score both forests identically
        for point in test_data:
            forest1.learn_one(point.copy())
            forest2.learn_one(point.copy())

            score1 = forest1.score_one(point)
            score2 = forest2.score_one(point)

            scores1.append(score1)
            scores2.append(score2)

        # Scores should be identical with same seed
        for s1, s2 in zip(scores1, scores2, strict=False):
            self.assertAlmostEqual(
                s1, s2, places=10, msg=f"Scores differ with same seed: {s1} != {s2}"
            )

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        forest1 = ASDIsolationForest(n_estimators=8, max_samples=15, seed=42)
        forest2 = ASDIsolationForest(n_estimators=8, max_samples=15, seed=123)

        test_data = self.data_generator.generate_streaming_data(n=80, n_features=2)

        scores1 = []
        scores2 = []

        for point in test_data:
            forest1.learn_one(point.copy())
            forest2.learn_one(point.copy())

            score1 = forest1.score_one(point)
            score2 = forest2.score_one(point)

            scores1.append(score1)
            scores2.append(score2)

        # Most scores should differ with different seeds
        different_count = sum(
            1 for s1, s2 in zip(scores1, scores2, strict=False) if abs(s1 - s2) > 1e-10
        )
        self.assertGreater(
            different_count,
            len(scores1) * 0.3,
            "Expected many scores to differ with different seeds",
        )

    def test_max_samples_effect(self):
        """Test that different max_samples values affect behavior."""
        forest_small = ASDIsolationForest(n_estimators=5, max_samples=10, seed=42)
        forest_large = ASDIsolationForest(n_estimators=5, max_samples=50, seed=42)

        # Generate enough data for both forests
        training_data = self.data_generator.generate_streaming_data(n=100, n_features=2)

        for point in training_data:
            forest_small.learn_one(point.copy())
            forest_large.learn_one(point.copy())

        # Test scoring
        test_point = {"feature_0": 5.0, "feature_1": 5.0}

        score_small = forest_small.score_one(test_point)
        score_large = forest_large.score_one(test_point)

        # Both should produce valid scores
        self.assertIsInstance(score_small, float)
        self.assertIsInstance(score_large, float)
        self.assertGreaterEqual(score_small, 0.0)
        self.assertLessEqual(score_small, 1.0)
        self.assertGreaterEqual(score_large, 0.0)
        self.assertLessEqual(score_large, 1.0)

    def test_n_estimators_effect(self):
        """Test that different numbers of estimators affect results."""
        forest_few = ASDIsolationForest(n_estimators=3, max_samples=15, seed=42)
        forest_many = ASDIsolationForest(n_estimators=20, max_samples=15, seed=42)

        # Train on same data
        training_data = self.data_generator.generate_streaming_data(n=100, n_features=3)

        for point in training_data:
            forest_few.learn_one(point.copy())
            forest_many.learn_one(point.copy())

        # Both should have complete replacement forests.
        self.assertGreater(len(forest_few.trees), 0)
        self.assertGreater(len(forest_many.trees), 0)
        self.assertEqual(len(forest_few.trees), 3)
        self.assertEqual(len(forest_many.trees), 20)

        # Score test points
        test_point = {"feature_0": 10.0, "feature_1": 10.0, "feature_2": 10.0}

        score_few = forest_few.score_one(test_point)
        score_many = forest_many.score_one(test_point)

        # Both should be valid scores
        self.assertIsInstance(score_few, float)
        self.assertIsInstance(score_many, float)
        self.assertGreaterEqual(score_few, 0.0)
        self.assertLessEqual(score_few, 1.0)
        self.assertGreaterEqual(score_many, 0.0)
        self.assertLessEqual(score_many, 1.0)

    def test_repr_string(self):
        """Test string representation includes key parameters."""
        forest = ASDIsolationForest(n_estimators=25, max_samples=64, seed=999)

        # Add some trees to test n_trees display
        for i in range(70):  # Enough to create some trees
            forest.learn_one({"feature": float(i)})

        repr_str = repr(forest)
        self.assertIn("ASDIsolationForest", repr_str)
        self.assertIn("n_estimators=25", repr_str)
        self.assertIn("max_samples=64", repr_str)
        self.assertIn("window_size=64", repr_str)
        self.assertIn("seed=999", repr_str)
        self.assertIn(f"n_trees={len(forest.trees)}", repr_str)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        forest = ASDIsolationForest(n_estimators=3, max_samples=5, seed=42)

        # Test with zero values
        zero_point = {"a": 0.0, "b": 0.0}
        forest.learn_one(zero_point)
        score = forest.score_one(zero_point)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

        # Test with negative values
        negative_point = {"a": -5.0, "b": -10.0}
        forest.learn_one(negative_point)
        score = forest.score_one(negative_point)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

        # Test with very large values
        large_point = {"a": 1e6, "b": 1e6}
        forest.learn_one(large_point)
        score = forest.score_one(large_point)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

        # Test with very small values
        small_point = {"a": 1e-6, "b": 1e-6}
        forest.learn_one(small_point)
        score = forest.score_one(small_point)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_tree_building_with_identical_values(self):
        """Test tree building when all values in a feature are identical."""
        forest = ASDIsolationForest(n_estimators=2, max_samples=3, seed=42)

        # Add identical points to create a tree
        identical_points = [
            {"feature": 5.0},
            {"feature": 5.0},
            {"feature": 5.0},  # This should create a tree with identical values
        ]

        for point in identical_points:
            forest.learn_one(point)

        # Should handle identical values gracefully
        self.assertEqual(len(forest.trees), 2)

        # Scoring should work
        score = forest.score_one({"feature": 5.0})
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_feature_conversion_efficiency(self):
        """Test that feature conversion is handled efficiently."""
        forest = ASDIsolationForest(max_samples=5)

        # First point establishes feature order
        forest.learn_one({"z": 3.0, "a": 1.0, "m": 2.0})

        # Verify feature order is alphabetical
        expected_order = ["a", "m", "z"]
        self.assertEqual(forest._schema.names, tuple(expected_order))

        # Subsequent points should work with same feature order
        forest.learn_one({"z": 6.0, "a": 4.0, "m": 5.0})
        score = forest.score_one({"a": 1.0, "m": 2.0, "z": 3.0})

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)

    def test_schema_change_raises_instead_of_zero_imputation(self):
        forest = ASDIsolationForest(n_estimators=1, max_samples=2, seed=42)
        forest.learn_one({"x": 1.0, "y": 2.0})
        forest.learn_one({"x": 2.0, "y": 3.0})

        with self.assertRaises(ValueError):
            forest.learn_one({"x": 3.0})
        with self.assertRaises(ValueError):
            forest.score_one({"x": 3.0})


if __name__ == "__main__":
    unittest.main()
