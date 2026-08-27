import unittest

import numpy as np

from aberrant.transform.preprocessing.scaler import MinMaxScaler, StandardScaler
from aberrant.transform.projection.random_projection import RandomProjection


class TestMinMaxScaler(unittest.TestCase):
    """Comprehensive tests for MinMaxScaler with tight validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.scaler = MinMaxScaler()
        self.test_data = [
            {"feature1": 10.0, "feature2": 5.0},
            {"feature1": 20.0, "feature2": 15.0},
            {"feature1": 30.0, "feature2": 25.0},
        ]

    def test_basic_scaling_default_range(self):
        """Test basic scaling with default (0, 1) range."""
        # Learn from data
        for data in self.test_data:
            self.scaler.learn_one(data)

        # Test first point: min values should scale to 0
        result = self.scaler.transform_one({"feature1": 10.0, "feature2": 5.0})
        self.assertAlmostEqual(result["feature1"], 0.0, places=10)
        self.assertAlmostEqual(result["feature2"], 0.0, places=10)

        # Test last point: max values should scale to 1
        result = self.scaler.transform_one({"feature1": 30.0, "feature2": 25.0})
        self.assertAlmostEqual(result["feature1"], 1.0, places=10)
        self.assertAlmostEqual(result["feature2"], 1.0, places=10)

        # Test middle point: should scale to 0.5
        result = self.scaler.transform_one({"feature1": 20.0, "feature2": 15.0})
        self.assertAlmostEqual(result["feature1"], 0.5, places=10)
        self.assertAlmostEqual(result["feature2"], 0.5, places=10)

    def test_custom_feature_range(self):
        """Test scaling with custom feature range."""
        scaler = MinMaxScaler(feature_range=(-1, 1))

        for data in self.test_data:
            scaler.learn_one(data)

        # Min values should scale to -1
        result = scaler.transform_one({"feature1": 10.0, "feature2": 5.0})
        self.assertAlmostEqual(result["feature1"], -1.0, places=10)
        self.assertAlmostEqual(result["feature2"], -1.0, places=10)

        # Max values should scale to 1
        result = scaler.transform_one({"feature1": 30.0, "feature2": 25.0})
        self.assertAlmostEqual(result["feature1"], 1.0, places=10)
        self.assertAlmostEqual(result["feature2"], 1.0, places=10)

        # Middle values should scale to 0
        result = scaler.transform_one({"feature1": 20.0, "feature2": 15.0})
        self.assertAlmostEqual(result["feature1"], 0.0, places=10)
        self.assertAlmostEqual(result["feature2"], 0.0, places=10)

    def test_invalid_feature_range_and_non_finite_learning_value(self):
        with self.assertRaisesRegex(ValueError, "lower bound must be less"):
            MinMaxScaler(feature_range=(1.0, 1.0))

        with self.assertRaisesRegex(ValueError, "must be finite"):
            self.scaler.learn_one({"x": float("nan")})
        self.assertNotIn("x", self.scaler.min)

    def test_single_data_point(self):
        """Test behavior with single data point."""
        single_data = {"feature1": 42.0}
        self.scaler.learn_one(single_data)

        # With single point, min == max, should return feature_range[0]
        result = self.scaler.transform_one(single_data)
        self.assertAlmostEqual(result["feature1"], 0.0, places=10)

    def test_constant_feature_values(self):
        """Test scaling when all feature values are identical."""
        constant_data = [
            {"feature1": 5.0},
            {"feature1": 5.0},
            {"feature1": 5.0},
        ]

        for data in constant_data:
            self.scaler.learn_one(data)

        # All constant values should scale to feature_range[0]
        result = self.scaler.transform_one({"feature1": 5.0})
        self.assertAlmostEqual(result["feature1"], 0.0, places=10)

    def test_incremental_learning_correctness(self):
        """Test that incremental learning produces correct min/max values."""
        data_points = [{"x": 1.0}, {"x": 5.0}, {"x": 3.0}, {"x": 9.0}, {"x": 2.0}]

        for data in data_points:
            self.scaler.learn_one(data)

        # Verify internal state
        self.assertEqual(self.scaler.min["x"], 1.0)
        self.assertEqual(self.scaler.max["x"], 9.0)

        # Test scaling correctness
        result = self.scaler.transform_one({"x": 1.0})  # min
        self.assertAlmostEqual(result["x"], 0.0, places=10)

        result = self.scaler.transform_one({"x": 9.0})  # max
        self.assertAlmostEqual(result["x"], 1.0, places=10)

        result = self.scaler.transform_one({"x": 5.0})  # (5-1)/(9-1) = 0.5
        self.assertAlmostEqual(result["x"], 0.5, places=10)

    def test_numpy_float64_handling(self):
        """Test proper handling of numpy float64 values."""
        data = {"feature1": np.float64(15.5)}
        self.scaler.learn_one(data)
        self.scaler.learn_one({"feature1": np.float64(25.5)})

        result = self.scaler.transform_one({"feature1": np.float64(20.5)})
        expected = (20.5 - 15.5) / (25.5 - 15.5)  # Should be 0.5
        self.assertAlmostEqual(result["feature1"], expected, places=10)
        self.assertIsInstance(result["feature1"], float)

    def test_unseen_feature_error(self):
        """Test error handling for unseen features during transform."""
        self.scaler.learn_one({"known_feature": 10.0})

        with self.assertRaises(ValueError) as context:
            self.scaler.transform_one({"unknown_feature": 5.0})

        self.assertIn(
            "Feature 'unknown_feature' has not been seen during learning",
            str(context.exception),
        )

    def test_mixed_features_error(self):
        """Test error when transform includes both known and unknown features."""
        self.scaler.learn_one({"feature1": 10.0})

        with self.assertRaises(ValueError):
            self.scaler.transform_one({"feature1": 15.0, "unknown": 20.0})

    def test_empty_feature_dict(self):
        """Test behavior with empty feature dictionary."""
        result = self.scaler.transform_one({})
        self.assertEqual(result, {})


class TestStandardScaler(unittest.TestCase):
    """Comprehensive tests for StandardScaler with tight validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.scaler = StandardScaler()
        self.test_values = [2.0, 4.0, 6.0, 8.0, 10.0]  # mean=6, var=8
        self.test_data = [{"x": val} for val in self.test_values]

    def test_basic_standardization_with_std(self):
        """Test basic standardization with standard deviation (default behavior)."""
        for data in self.test_data:
            self.scaler.learn_one(data)

        # Verify learned statistics
        expected_mean = 6.0
        expected_var = 8.0  # population variance

        self.assertAlmostEqual(self.scaler.means["x"], expected_mean, places=10)

        computed_var = self.scaler.sum_sq_diffs["x"] / self.scaler.counts["x"]
        self.assertAlmostEqual(computed_var, expected_var, places=10)

        # Test transformations
        # For x=2: z = (2-6)/sqrt(8) = -4/2.828... ≈ -1.414
        result = self.scaler.transform_one({"x": 2.0})
        expected = (2.0 - 6.0) / (8.0**0.5)
        self.assertAlmostEqual(result["x"], expected, places=10)

        # For x=10: z = (10-6)/sqrt(8) = 4/2.828... ≈ 1.414
        result = self.scaler.transform_one({"x": 10.0})
        expected = (10.0 - 6.0) / (8.0**0.5)
        self.assertAlmostEqual(result["x"], expected, places=10)

        # Mean should transform to 0
        result = self.scaler.transform_one({"x": 6.0})
        self.assertAlmostEqual(result["x"], 0.0, places=10)

    def test_standardization_without_std(self):
        """Test standardization without standard deviation (mean centering only)."""
        scaler = StandardScaler(with_std=False)

        for data in self.test_data:
            scaler.learn_one(data)

        # Should only subtract mean, not divide by std
        result = scaler.transform_one({"x": 8.0})
        expected = 8.0 - 6.0  # No division by std
        self.assertAlmostEqual(result["x"], expected, places=10)

        result = scaler.transform_one({"x": 4.0})
        expected = 4.0 - 6.0
        self.assertAlmostEqual(result["x"], expected, places=10)

    def test_single_data_point(self):
        """Test behavior with single data point."""
        single_data = {"feature1": 42.0}
        self.scaler.learn_one(single_data)

        # With single point, variance is 0, should use safe division
        result = self.scaler.transform_one(single_data)
        self.assertAlmostEqual(result["feature1"], 0.0, places=10)  # Safe div returns 0

    def test_non_finite_learning_value_does_not_poison_statistics(self):
        with self.assertRaisesRegex(ValueError, "must be finite"):
            self.scaler.learn_one({"x": float("inf")})

        self.assertEqual(self.scaler.counts["x"], 0)

    def test_zero_variance_feature(self):
        """Test handling of features with zero variance."""
        constant_data = [{"x": 5.0}, {"x": 5.0}, {"x": 5.0}]

        for data in constant_data:
            self.scaler.learn_one(data)

        # Zero variance should use safe division (return 0)
        result = self.scaler.transform_one({"x": 5.0})
        self.assertAlmostEqual(result["x"], 0.0, places=10)

    def test_incremental_mean_calculation(self):
        """Test correctness of incremental mean calculation."""
        values = [1.0, 3.0, 5.0, 7.0, 9.0]
        scaler = StandardScaler()

        for val in values:
            scaler.learn_one({"x": val})

        expected_mean = sum(values) / len(values)
        self.assertAlmostEqual(scaler.means["x"], expected_mean, places=10)

    def test_incremental_variance_calculation(self):
        """Test correctness of incremental variance calculation using Welford's algorithm."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        scaler = StandardScaler()

        for val in values:
            scaler.learn_one({"x": val})

        # Verify against numpy variance (population variance)
        expected_var = np.var(values)
        computed_var = scaler.sum_sq_diffs["x"] / scaler.counts["x"]
        self.assertAlmostEqual(computed_var, expected_var, places=10)

    def test_multiple_features(self):
        """Test scaling with multiple features simultaneously."""
        multi_data = [
            {"x": 1.0, "y": 10.0},
            {"x": 2.0, "y": 20.0},
            {"x": 3.0, "y": 30.0},
        ]

        for data in multi_data:
            self.scaler.learn_one(data)

        # Test that each feature is scaled independently
        result = self.scaler.transform_one({"x": 2.0, "y": 20.0})

        # x: mean=2, var=2/3, std=sqrt(2/3), z=(2-2)/sqrt(2/3)=0
        self.assertAlmostEqual(result["x"], 0.0, places=10)

        # y: mean=20, var=200/3, std=sqrt(200/3), z=(20-20)/sqrt(200/3)=0
        self.assertAlmostEqual(result["y"], 0.0, places=10)

    def test_numpy_float64_handling(self):
        """Test proper handling of numpy float64 values."""
        data = [{"x": np.float64(val)} for val in [1.0, 2.0, 3.0]]

        for d in data:
            self.scaler.learn_one(d)

        result = self.scaler.transform_one({"x": np.float64(2.0)})
        self.assertAlmostEqual(result["x"], 0.0, places=10)  # Mean value
        self.assertIsInstance(result["x"], float)

    def test_unseen_feature_error(self):
        """Test error handling for unseen features during transform."""
        self.scaler.learn_one({"known_feature": 10.0})

        with self.assertRaises(ValueError) as context:
            self.scaler.transform_one({"unknown_feature": 5.0})

        self.assertIn(
            "Feature 'unknown_feature' has not been seen during learning",
            str(context.exception),
        )

    def test_mixed_features_error(self):
        """Test error when transform includes both known and unknown features."""
        self.scaler.learn_one({"feature1": 10.0})

        with self.assertRaises(ValueError):
            self.scaler.transform_one({"feature1": 15.0, "unknown": 20.0})

    def test_empty_feature_dict(self):
        """Test behavior with empty feature dictionary."""
        result = self.scaler.transform_one({})
        self.assertEqual(result, {})

    def test_safe_div_functionality(self):
        """Test the _safe_div helper method directly."""
        # Normal division
        self.assertAlmostEqual(self.scaler._safe_div(10.0, 2.0), 5.0, places=10)

        # Division by zero
        self.assertEqual(self.scaler._safe_div(10.0, 0.0), 0.0)

        # Division by False (falsy value)
        self.assertEqual(self.scaler._safe_div(10.0, False), 0.0)


class TestRandomProjections(unittest.TestCase):
    def setUp(self):
        self.n_components = 4
        self.feature_keys = [f"feature_{i}" for i in range(10)]

    def test_initialization_without_keys(self):
        rp = RandomProjection(n_components=self.n_components)
        self.assertIsNone(
            rp.feature_names,
            "Feature names should be None when initialized without keys.",
        )
        self.assertEqual(
            rp.random_matrix.size, 0, "Random matrix should not be initialized."
        )

    def test_initialization_with_keys(self):
        rp = RandomProjection(n_components=self.n_components, keys=self.feature_keys)
        np.testing.assert_array_equal(
            rp.feature_names,
            self.feature_keys,
            "Feature names do not match the provided keys.",
        )
        expected_shape = (len(self.feature_keys), self.n_components)
        self.assertEqual(
            rp.random_matrix.shape,
            expected_shape,
            f"Random matrix shape should be {expected_shape}, got {rp.random_matrix.shape}.",
        )

    def test_initialization_learn_one_without_keys(self):
        rp = RandomProjection(n_components=self.n_components)
        sample_data = {f"feature_{i}": np.random.rand() for i in range(10)}
        rp.learn_one(sample_data)

        expected_feature_names = list(sample_data.keys())
        np.testing.assert_array_equal(
            rp.feature_names,
            expected_feature_names,
            "Feature names should be set to the keys of the first learned sample.",
        )

        expected_shape = (len(expected_feature_names), self.n_components)
        self.assertEqual(
            rp.random_matrix.shape,
            expected_shape,
            f"Random matrix shape should be {expected_shape}, got {rp.random_matrix.shape}.",
        )

    def test_transform_one(self):
        rp = RandomProjection(n_components=self.n_components)
        sample_data = {f"feature_{i}": np.random.rand() for i in range(10)}
        rp.learn_one(sample_data)

        transformed_data = rp.transform_one(sample_data)
        self.assertEqual(
            len(transformed_data),
            self.n_components,
            "Transformed data should have the same number of components as n_components.",
        )

    def test_transform_rejects_extra_features(self):
        projection = RandomProjection(1, keys=["x", "y"], seed=42)

        with self.assertRaisesRegex(ValueError, "unexpected feature"):
            projection.transform_one({"x": 1.0, "y": 2.0, "z": 3.0})

    def test_error_with_too_many_n_components(self):
        with self.assertRaises(ValueError) as context:
            RandomProjection(n_components=15, keys=self.feature_keys[:10])

        expected_msg = "The number of n_components (15) has to be less or equal to the number of features (10)"
        self.assertTrue(
            expected_msg in str(context.exception),
            f"Expected error message '{expected_msg}' but got {context.exception}",
        )

    def test_transform_one_with_none_feature_names(self):
        rp = RandomProjection(n_components=self.n_components)
        with self.assertRaises(RuntimeError) as context:
            sample_data = {f"feature_{i}": np.random.rand() for i in range(10)}
            rp.transform_one(sample_data)

        expected_msg = (
            "Cannot transform before learning. Call learn_one() first or provide keys."
        )
        self.assertTrue(
            expected_msg in str(context.exception),
            f"Expected error message '{expected_msg}' but got {context.exception}",
        )

    def test_error_with_zero_or_negative_n_components(self):
        with self.assertRaises(ValueError) as context:
            RandomProjection(n_components=0, keys=self.feature_keys[:5])

        expected_msg_zero = "n_components must be greater than 0"
        self.assertTrue(
            expected_msg_zero in str(context.exception),
            f"Expected error message '{expected_msg_zero}' but got {context.exception}",
        )

        with self.assertRaises(ValueError) as context:
            RandomProjection(n_components=-3, keys=self.feature_keys[:5])

        expected_msg_negative = "n_components must be greater than 0"
        self.assertTrue(
            expected_msg_negative in str(context.exception),
            f"Expected error message '{expected_msg_negative}' but got {context.exception}",
        )

    def test_duplicate_feature_names_initialization(self):
        duplicate_keys = ["feature_1", "feature_1"]
        with self.assertRaises(ValueError) as context:
            RandomProjection(n_components=self.n_components, keys=duplicate_keys)

        expected_msg = "Feature names cannot contain duplicates"
        self.assertTrue(
            expected_msg in str(context.exception),
            f"Expected error message '{expected_msg}' but got {context.exception}",
        )

    def test_empty_data_point_in_learn_one(self):
        rp = RandomProjection(n_components=self.n_components)
        rp.learn_one({})
        self.assertEqual(rp.random_matrix.size, 0)

        rp.learn_one({f"feature_{i}": np.random.rand() for i in range(5)})
        self.assertEqual(rp.random_matrix.shape, (5, 4))

        rp.learn_one({})
        self.assertEqual(rp.random_matrix.shape, (5, 4))

    def test_random_state(self):
        rp = RandomProjection(3, seed=42)
        rng = np.random.default_rng(42)
        datapoint = {f"feature_{i}": rng.random() for i in range(5)}
        rp.learn_one(datapoint)
        expected_random_matrix = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        )
        expected_random_matrix = np.round(expected_random_matrix, 7)
        random_matrix = np.round(rp.random_matrix, 7)
        random_matrix_equal = expected_random_matrix == random_matrix
        self.assertTrue(
            random_matrix_equal.all()
        )  # test creating random matrix (alomost equal)

        dict_values = rp.transform_one(datapoint)
        expected_transformed = expected_random_matrix.T @ np.fromiter(
            datapoint.values(), dtype=float
        )
        transformed_values = [
            dict_values["component_0"],
            dict_values["component_1"],
            dict_values["component_2"],
        ]
        np.testing.assert_allclose(
            expected_transformed, transformed_values, rtol=1e-14, atol=1e-14
        )

    def test_projection_owns_configured_feature_names(self):
        keys = ["a", "b", "c"]
        projection = RandomProjection(2, keys=keys, seed=42)
        keys[0] = "changed"

        self.assertEqual(projection.feature_names, ["a", "b", "c"])

    def test_projection_rejects_non_finite_values_before_learning_schema(self):
        projection = RandomProjection(1)

        with self.assertRaisesRegex(ValueError, "must be finite"):
            projection.learn_one({"x": float("nan")})

        self.assertIsNone(projection.feature_names)


if __name__ == "__main__":
    unittest.main()
