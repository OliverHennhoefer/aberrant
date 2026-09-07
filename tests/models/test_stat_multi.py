import unittest
from collections import deque

import numpy as np

from aberrant.model.stat.multi import (
    MovingCorrelationCoefficient,
    MovingCovariance,
    MovingMahalanobisDistance,
)


class TestMovingCovariance(unittest.TestCase):
    def test_initialization_with_positive_window_size(self):
        model = MovingCovariance(window_size=5)
        self.assertEqual(model.window_size, 5)

    def test_initialization_with_negative_window_size(self):
        with self.assertRaises(ValueError):
            MovingCovariance(window_size=-1)

    def test_learn_one_with_valid_input(self):
        model = MovingCovariance(window_size=3)
        point = {"x": 1.0, "y": 2.0}
        model.learn_one(point)
        self.assertEqual(model.window["x"], deque([1]))
        self.assertEqual(model.window["y"], deque([2]))

    def test_keys_initialization_creates_windows(self):
        model = MovingCovariance(window_size=3, keys=["x", "y"])
        self.assertIn("x", model.window)
        self.assertIn("y", model.window)
        self.assertEqual(model.window["x"], deque([], maxlen=3))
        self.assertEqual(model.window["y"], deque([], maxlen=3))

    def test_keys_initialization_rejects_missing_feature(self):
        model = MovingCovariance(window_size=3, keys=["x", "y"])
        with self.assertRaises(ValueError):
            model.learn_one({"x": 1.0, "z": 2.0})

    def test_rejects_duplicate_keys_and_invalid_points_atomically(self):
        with self.assertRaisesRegex(ValueError, "cannot contain duplicates"):
            MovingCovariance(window_size=3, keys=["x", "x"])

        model = MovingCovariance(window_size=3)
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            model.learn_one({"x": 1.0, "y": "bad"})  # type: ignore[dict-item]
        self.assertIsNone(model._schema.names)
        self.assertEqual(model.window, {})

    def test_learn_one_with_multiple_points(self):
        model = MovingCovariance(window_size=3)
        points = [{"x": float(i), "y": float(i) + 1} for i in range(5)]
        for point in points:
            model.learn_one(point)
        self.assertEqual(len(model.window["x"]), 3)
        self.assertEqual(len(model.window["y"]), 3)

    def test_score_one_with_empty_window(self):
        model = MovingCovariance(window_size=2)
        self.assertEqual(model.score_one({"x": 1, "y": 1}), 0)

    def test_score_one_with_zero_window(self):
        model = MovingCovariance(window_size=5)
        for _ in range(4):
            model.learn_one({"a": 0, "b": 0})
        self.assertEqual(model.score_one({"a": 0, "b": 0}), 0)

    def test_score_one_without_bessel(self):
        model = MovingCovariance(window_size=3, bias=True)
        points = [{"x": float(i), "y": i**2 - 1} for i in range(1, 4)]
        for point in points:
            model.learn_one(point)
        cov_old = np.cov([1, 2, 3], [0, 3, 8], bias=True)[0][1]
        cov_new = np.cov([1, 2, 3, 4], [0, 3, 8, 15], bias=True)[0][1]
        expected_diff = float(cov_new - cov_old)
        self.assertAlmostEqual(model.score_one({"x": 4, "y": 15}), expected_diff)

    def test_score_one_with_bessel_correction(self):
        model = MovingCovariance(window_size=3, bias=False)
        points = [{"x": float(i), "y": i**2 - 1} for i in range(1, 4)]
        for point in points:
            model.learn_one(point)
        cov_old = np.cov([1, 2, 3], [0, 3, 8], bias=False)[0][1]
        cov_new = np.cov([1, 2, 3, 4], [0, 3, 8, 15], bias=False)[0][1]
        expected_diff = float(cov_new - cov_old)
        self.assertAlmostEqual(model.score_one({"x": 4, "y": 15}), expected_diff)


class TestMovingCorrelationCoefficient(unittest.TestCase):
    def test_initialization_with_positive_window_size(self):
        model = MovingCorrelationCoefficient(window_size=5)
        self.assertEqual(model.window_size, 5)

    def test_initialization_with_negative_window_size(self):
        with self.assertRaises(ValueError):
            MovingCorrelationCoefficient(window_size=-1)

    def test_learn_one_with_valid_input(self):
        model = MovingCorrelationCoefficient(window_size=3)
        point = {"x": 1.0, "y": 2.0}
        model.learn_one(point)
        self.assertEqual(model.window["x"], deque([1]))
        self.assertEqual(model.window["y"], deque([2]))

    def test_keys_initialization_creates_windows(self):
        model = MovingCorrelationCoefficient(window_size=3, keys=["x", "y"])
        self.assertIn("x", model.window)
        self.assertIn("y", model.window)
        self.assertEqual(model.window["x"], deque([], maxlen=3))
        self.assertEqual(model.window["y"], deque([], maxlen=3))

    def test_keys_initialization_rejects_missing_feature(self):
        model = MovingCorrelationCoefficient(window_size=3, keys=["x", "y"])
        with self.assertRaises(ValueError):
            model.learn_one({"x": 1.0, "z": 2.0})

    def test_learn_one_with_multiple_points(self):
        model = MovingCorrelationCoefficient(window_size=3)
        points = [{"x": float(i), "y": float(i) + 1} for i in range(5)]
        for point in points:
            model.learn_one(point)
        self.assertEqual(len(model.window["x"]), 3)
        self.assertEqual(len(model.window["y"]), 3)

    def test_score_one_with_empty_window(self):
        model = MovingCorrelationCoefficient(window_size=2)
        self.assertEqual(model.score_one({"x": 3, "y": 6}), 0)

    def test_score_one_with_zero_window(self):
        model = MovingCorrelationCoefficient(window_size=5)
        for _ in range(4):
            model.learn_one({"a": 0, "b": 0})
        self.assertEqual(model.score_one({"a": 0, "b": 0}), 0)

    def test_score_one_without_bessel(self):
        model = MovingCorrelationCoefficient(window_size=3, bias=True, abs_diff=False)
        points = [{"x": float(i), "y": i**2 - 1} for i in range(1, 4)]
        for point in points:
            model.learn_one(point)

        cov_old = np.cov([1, 2, 3], [0, 3, 8], bias=True)[0][1]
        cov_new = np.cov([1, 2, 3, 4], [0, 3, 8, 15], bias=True)[0][1]
        std_old_0 = np.std([1, 2, 3], ddof=0)
        std_old_1 = np.std([0, 3, 8], ddof=0)
        std_new_0 = np.std([1, 2, 3, 4], ddof=0)
        std_new_1 = np.std([0, 3, 8, 15], ddof=0)
        cor_coef_old = cov_old / (std_old_0 * std_old_1)
        cor_coef_new = cov_new / (std_new_0 * std_new_1)
        cor_coef_diff = float(cor_coef_new - cor_coef_old)

        self.assertAlmostEqual(model.score_one({"x": 4, "y": 15}), cor_coef_diff)

    def test_score_one_with_bessel_correction(self):
        model = MovingCorrelationCoefficient(window_size=3, bias=False, abs_diff=False)
        points = [{"x": float(i), "y": i**2 - 1} for i in range(1, 4)]
        for point in points:
            model.learn_one(point)

        cov_old = np.cov([1, 2, 3], [0, 3, 8], bias=False)[0][1]
        cov_new = np.cov([1, 2, 3, 4], [0, 3, 8, 15], bias=False)[0][1]
        std_old_0 = np.std([1, 2, 3], ddof=1)
        std_old_1 = np.std([0, 3, 8], ddof=1)
        std_new_0 = np.std([1, 2, 3, 4], ddof=1)
        std_new_1 = np.std([0, 3, 8, 15], ddof=1)
        cor_coef_old = cov_old / (std_old_0 * std_old_1)
        cor_coef_new = cov_new / (std_new_0 * std_new_1)
        cor_coef_diff = float(cor_coef_new - cor_coef_old)

        self.assertAlmostEqual(model.score_one({"x": 4, "y": 15}), cor_coef_diff)


class TestMovingMahalanobisDistance(unittest.TestCase):
    def test_initialization(self):
        # Test valid initialization
        mmd = MovingMahalanobisDistance(window_size=3)
        self.assertEqual(mmd.window_size, 3)

        # Test invalid window size
        with self.assertRaises(ValueError):
            MovingMahalanobisDistance(window_size=-1)

        with self.assertRaisesRegex(ValueError, "cannot contain duplicates"):
            MovingMahalanobisDistance(window_size=3, keys=["x", "x"])

    def test_learn_one(self):
        mmd = MovingMahalanobisDistance(window_size=3)

        # Test learning a single data point
        mmd.learn_one({"feature1": 1.0, "feature2": 2.0})
        self.assertEqual(mmd.window, deque([[1.0, 2.0]]))

        # Test updating with another point
        mmd.learn_one({"feature1": 3.0, "feature2": 4.0})
        self.assertEqual(len(mmd.window), 2)

        # Invalid observations are rejected atomically instead of being ignored.
        with self.assertRaises(ValueError):
            mmd.learn_one({"feature1": "a", "feature2": 5.0})  # type: ignore[arg-type]
        self.assertEqual(len(mmd.window), 2)

    def test_score_one_insufficient_data_points(self):
        mmd = MovingMahalanobisDistance(window_size=2)

        # Test scoring with insufficient data points
        self.assertEqual(mmd.score_one({"feature1": 1.0, "feature2": 2.0}), 0)

        # Add two valid data points and check the score
        mmd.learn_one({"feature1": 1.0, "feature2": 2.0})
        mmd.learn_one({"feature1": 3.0, "feature2": 4.0})

        # Test scoring with insufficient data points
        score = mmd.score_one({"feature1": 1.0, "feature2": 2.0})
        self.assertGreaterEqual(score, 0)

    def test_score_one_singular_matrix(self):
        mmd = MovingMahalanobisDistance(window_size=5)
        mmd.learn_one(
            {"feature1": 1.0, "feature2": 1.0, "feature3": 3.0, "feature4": 4.0}
        )
        mmd.learn_one(
            {"feature1": 2.0, "feature2": 4.0, "feature3": 6.0, "feature4": 8.0}
        )
        mmd.learn_one(
            {"feature1": 3.0, "feature2": 6.0, "feature3": 9.0, "feature4": 12.0}
        )
        self.assertGreaterEqual(
            mmd.score_one(
                {"feature1": 4.0, "feature2": 5.0, "feature3": 6.0, "feature4": 7.0}
            ),
            1,
        )

    def test_score_one(self):
        mmd = MovingMahalanobisDistance(window_size=10, bias=False)
        values = np.array([[1, 2], [2, 3], [2, 3.5], [3, 5], [5, 10]])
        for point in values:
            mmd.learn_one({"a": point[0], "b": point[1]})
        scored = mmd.score_one({"a": 6, "b": 11})

        previous_points = np.array(list(values))
        cov_matrix = np.cov(previous_points, rowvar=False)
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        feature_mean = np.mean(previous_points, axis=0)
        x = np.array([6, 11])
        diff = x - feature_mean
        score = float(diff.T @ inv_cov_matrix @ diff)

        self.assertAlmostEqual(scored, score)

    def test_rank_deficient_covariance_has_positive_regularized_distance(self):
        model = MovingMahalanobisDistance(window_size=3)
        values = np.array([[2.0, 2.0, 3.0], [4.0, 0.0, 0.0], [4.0, 4.0, 1.0]])
        query = np.array([1.0, 2.0, 3.0])
        for point in values:
            model.learn_one(dict(zip("abc", point, strict=True)))

        # Three observations in three dimensions cannot give full covariance
        # rank, even if the numerical inverse happens to succeed.
        covariance = np.cov(values, rowvar=False, bias=True)
        ridge = 1e-6 * np.trace(covariance) / 3
        diff = query - np.mean(values, axis=0)
        expected = float(diff @ np.linalg.solve(covariance + ridge * np.eye(3), diff))
        score = model.score_one(dict(zip("abc", query, strict=True)))
        self.assertGreater(score, 1.0)
        np.testing.assert_allclose(score, expected, rtol=1e-9)

    def test_correlated_features_preserve_distance_across_numeric_scales(self):
        # In the (1, 1) and (1, -1) directions the covariance eigenvalues
        # are 4/3 and zero; the diagonal ridge is (2/3) * 1e-6.
        ridge = (2.0 / 3.0) * 1e-6
        expected = 0.5 / (4.0 / 3.0 + ridge) + 0.5 / ridge
        for scale in (1e-200, 1.0, 1e200):
            with self.subTest(scale=scale):
                model = MovingMahalanobisDistance(window_size=3)
                for value in (1.0, 2.0, 3.0):
                    model.learn_one({"a": scale * value, "b": scale * value})
                score = model.score_one({"a": scale * 2.0, "b": scale * 3.0})
                np.testing.assert_allclose(score, expected, rtol=1e-9)

    def test_nearly_correlated_features_are_regularized(self):
        model = MovingMahalanobisDistance(window_size=3)
        for a, b in ((1.0, 1.0), (2.0, 2.0 + 1e-7), (3.0, 3.0)):
            model.learn_one({"a": a, "b": b})

        # An arbitrarily small perturbation of the correlated reference must
        # not produce the enormous distance of an unstable unregularized solve.
        np.testing.assert_allclose(
            model.score_one({"a": 2.0, "b": 3.0}),
            750000.375,
            rtol=1e-6,
        )

    def test_large_constant_coordinate_does_not_erase_other_variance(self):
        constant = float(2**700)
        for scale in (1e-200, 1.0, 1e200):
            with self.subTest(scale=scale):
                model = MovingMahalanobisDistance(window_size=3)
                for value in (-1.0, 0.0, 1.0):
                    model.learn_one({"x": constant, "y": value * scale})

                # The changing coordinate has variance 2/3 and receives a
                # relative ridge of 1e-6/3, regardless of the constant's size.
                expected = 4.0 / (2.0 / 3.0 + 1e-6 / 3.0)
                self.assertAlmostEqual(
                    model.score_one({"x": constant, "y": 2.0 * scale}), expected
                )

    def test_centering_handles_opposite_reference_extremes(self):
        extreme = np.finfo(float).max
        model = MovingMahalanobisDistance(window_size=3)
        for value in (-extreme, 0.0, extreme):
            model.learn_one({"x": value})

        with np.errstate(over="raise", invalid="raise"):
            self.assertAlmostEqual(model.score_one({"x": extreme / 2.0}), 0.375)

    def test_centering_handles_opposite_query_extreme(self):
        model = MovingMahalanobisDistance(window_size=3)
        for value in (1e308, 1.1e308, 1.2e308):
            model.learn_one({"x": value})

        with np.errstate(over="raise", invalid="raise"):
            self.assertAlmostEqual(model.score_one({"x": -1e308}), 661.5)

    def test_well_conditioned_distance_is_unchanged_across_scales_and_bias(self):
        for scale in (1e-200, 1.0, 1e200):
            for bias in (False, True):
                with self.subTest(scale=scale, bias=bias):
                    model = MovingMahalanobisDistance(window_size=4, bias=bias)
                    for a, b in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)):
                        model.learn_one({"a": a * scale, "b": b * scale})

                    # Population covariance is I/2; sample covariance is 2I/3.
                    expected = 10.0 if bias else 7.5
                    self.assertAlmostEqual(
                        model.score_one({"a": scale, "b": 2.0 * scale}), expected
                    )

    def test_constant_windows_keep_absolute_covariance_floor(self):
        for value in (0.0, 0.1, 1.0, 1000.0):
            with self.subTest(value=value):
                model = MovingMahalanobisDistance(window_size=3)
                for _ in range(3):
                    model.learn_one({"a": value, "b": 1.0})
                self.assertEqual(model.score_one({"a": value, "b": 1.0}), 0.0)
                self.assertAlmostEqual(
                    model.score_one({"a": value + 1.0, "b": 1.0}), 1e6
                )

    def test_score_one_supports_one_dimensional_input_and_bias(self):
        biased = MovingMahalanobisDistance(window_size=3, bias=True)
        unbiased = MovingMahalanobisDistance(window_size=3, bias=False)
        for value in [1.0, 2.0, 3.0]:
            biased.learn_one({"value": value})
            unbiased.learn_one({"value": value})

        self.assertAlmostEqual(biased.score_one({"value": 4.0}), 6.0)
        self.assertAlmostEqual(unbiased.score_one({"value": 4.0}), 4.0)


if __name__ == "__main__":
    unittest.main()
