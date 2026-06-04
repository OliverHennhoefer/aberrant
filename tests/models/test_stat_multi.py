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

    def test_learn_one(self):
        mmd = MovingMahalanobisDistance(window_size=3)

        # Test learning a single data point
        mmd.learn_one({"feature1": 1.0, "feature2": 2.0})
        self.assertEqual(mmd.window, deque([[1.0, 2.0]]))

        # Test updating with another point
        mmd.learn_one({"feature1": 3.0, "feature2": 4.0})
        self.assertEqual(len(mmd.window), 2)

        # Test handling of non-numeric data#
        mmd.learn_one({"feature1": "a", "feature2": 5.0})  # type: ignore
        self.assertEqual(len(mmd.window), 2)  # The invalid entry should not be added

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

        self.assertEqual(scored, score)

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
