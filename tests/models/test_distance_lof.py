"""Tests for Local Outlier Factor (LOF) anomaly detection model."""

import math
import random
import unittest

import numpy as np

from aberrant.model.distance.lof import LocalOutlierFactor


class TestLocalOutlierFactor(unittest.TestCase):
    """Test suite for LocalOutlierFactor model."""

    def test_initialization_default_params(self):
        """Test LOF initialization with default parameters."""
        lof = LocalOutlierFactor()
        self.assertEqual(lof.k, 10)
        self.assertEqual(lof.window_size, 1000)
        self.assertEqual(lof.distance, "euclidean")
        self.assertEqual(lof.n_points, 0)

    def test_initialization_custom_params(self):
        """Test LOF initialization with custom parameters."""
        lof = LocalOutlierFactor(k=5, window_size=500, distance="manhattan")
        self.assertEqual(lof.k, 5)
        self.assertEqual(lof.window_size, 500)
        self.assertEqual(lof.distance, "manhattan")

    def test_invalid_k(self):
        """Test that invalid k values raise ValueError."""
        with self.assertRaises(ValueError) as context:
            LocalOutlierFactor(k=0)
        self.assertIn("k must be positive", str(context.exception))

        with self.assertRaises(ValueError):
            LocalOutlierFactor(k=-1)

    def test_invalid_window_size(self):
        """Test that invalid window_size raises ValueError."""
        with self.assertRaises(ValueError) as context:
            LocalOutlierFactor(window_size=0)
        self.assertIn("window_size must be positive", str(context.exception))

    def test_k_greater_than_window(self):
        """Test that k >= window_size raises ValueError."""
        with self.assertRaises(ValueError) as context:
            LocalOutlierFactor(k=100, window_size=50)
        self.assertIn("k must be less than window_size", str(context.exception))

    def test_invalid_distance(self):
        """Test that invalid distance metric raises ValueError."""
        with self.assertRaises(ValueError) as context:
            LocalOutlierFactor(distance="cosine")
        self.assertIn("distance must be", str(context.exception))

    def test_learn_one_empty_dict(self):
        """Test that empty input raises ValueError."""
        lof = LocalOutlierFactor()
        with self.assertRaises(ValueError) as context:
            lof.learn_one({})
        self.assertIn("empty", str(context.exception))

    def test_learn_one_increases_count(self):
        """Test that learn_one increases point count."""
        lof = LocalOutlierFactor()
        for i in range(10):
            lof.learn_one({"x": float(i), "y": float(i)})
        self.assertEqual(lof.n_points, 10)

    def test_score_one_insufficient_data(self):
        """Test score_one returns 0 with insufficient data."""
        lof = LocalOutlierFactor(k=5)
        # Add fewer than k+1 points
        for i in range(4):
            lof.learn_one({"x": float(i)})
        score = lof.score_one({"x": 5.0})
        self.assertEqual(score, 0.0)

    def test_score_one_returns_float(self):
        """Test that score_one returns a float."""
        lof = LocalOutlierFactor(k=3)
        # Add enough points
        for i in range(20):
            lof.learn_one({"x": float(i), "y": float(i)})
        score = lof.score_one({"x": 10.0, "y": 10.0})
        self.assertIsInstance(score, float)

    def test_normal_point_low_lof(self):
        """Test that points in dense regions have LOF close to 1."""
        lof = LocalOutlierFactor(k=5, window_size=100)
        # Create a dense cluster
        for i in range(50):
            lof.learn_one({"x": float(i % 5), "y": float(i % 5)})
        # Point in the cluster should have LOF close to 1
        score = lof.score_one({"x": 2.0, "y": 2.0})
        self.assertGreater(score, 0.0)
        # Should be relatively close to 1 (normal)
        self.assertLess(score, 5.0)

    def test_outlier_high_lof(self):
        """Test that outliers have higher LOF scores."""
        rng = random.Random(42)
        lof = LocalOutlierFactor(k=5, window_size=100)
        # Create a cluster with some variation around origin
        for _ in range(50):
            lof.learn_one({"x": rng.gauss(0, 0.1), "y": rng.gauss(0, 0.1)})
        # Score a point in the cluster
        normal_score = lof.score_one({"x": 0.0, "y": 0.0})
        # Score an outlier far from the cluster
        outlier_score = lof.score_one({"x": 100.0, "y": 100.0})
        # Both scores should be valid floats
        self.assertIsInstance(normal_score, float)
        self.assertIsInstance(outlier_score, float)
        # Outlier should have higher LOF (or be inf which is greater)
        self.assertGreaterEqual(outlier_score, normal_score)

    def test_manhattan_distance(self):
        """Test LOF with Manhattan distance."""
        lof = LocalOutlierFactor(k=3, distance="manhattan")
        for i in range(20):
            lof.learn_one({"x": float(i), "y": float(i)})
        score = lof.score_one({"x": 10.0, "y": 10.0})
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)

    def test_window_size_limit(self):
        """Test that window size is enforced."""
        lof = LocalOutlierFactor(k=3, window_size=10)
        # Add more points than window_size
        for i in range(20):
            lof.learn_one({"x": float(i)})
        # Should only keep window_size points
        self.assertEqual(lof.n_points, 10)

    def test_feature_names_sorted(self):
        """Test that feature names are sorted consistently."""
        lof = LocalOutlierFactor()
        lof.learn_one({"z": 1.0, "a": 2.0, "m": 3.0})
        self.assertEqual(lof.feature_names, ["a", "m", "z"])

    def test_duplicate_points_do_not_produce_nan(self):
        """Test duplicate neighborhoods produce finite/non-NaN LOF."""
        lof = LocalOutlierFactor(k=3, window_size=10)
        for _ in range(10):
            lof.learn_one({"x": 1.0, "y": 1.0})

        score = lof.score_one({"x": 1.0, "y": 1.0})
        self.assertIsInstance(score, float)
        self.assertFalse(math.isnan(score))

    def test_neighbor_selection_excludes_self_but_keeps_duplicate_points(self):
        lof = LocalOutlierFactor(k=2, window_size=10)
        for value in [0.0, 0.0, 1.0, 2.0]:
            lof.learn_one({"x": value})

        distances = np.array([0.0, 0.0, 1.0, 2.0])
        self.assertEqual(lof._get_k_neighbors(distances, exclude_index=0), [1, 2])
        self.assertEqual(lof._get_k_neighbors(distances), [0, 1])

    def test_neighbor_selection_includes_all_k_distance_ties(self):
        lof = LocalOutlierFactor(k=2, window_size=10)
        distances = np.array([0.0, 1.0, 1.0, 1.0, 2.0])

        self.assertEqual(lof._get_k_neighbors(distances, exclude_index=0), [1, 2, 3])

    def test_repr(self):
        """Test string representation."""
        lof = LocalOutlierFactor(k=5, window_size=500, distance="manhattan")
        repr_str = repr(lof)
        self.assertIn("LocalOutlierFactor", repr_str)
        self.assertIn("k=5", repr_str)
        self.assertIn("window_size=500", repr_str)
        self.assertIn("manhattan", repr_str)


class TestLocalOutlierFactorStreaming(unittest.TestCase):
    """Test LOF streaming behavior."""

    def test_streaming_updates(self):
        """Test that LOF handles streaming data correctly."""
        lof = LocalOutlierFactor(k=5, window_size=50)

        scores = []
        for i in range(100):
            point = {"x": float(i % 10), "y": float(i % 10)}
            lof.learn_one(point)
            if lof.n_points > lof.k:
                score = lof.score_one(point)
                scores.append(score)

        # All scores should be valid floats
        for score in scores:
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertFalse(math.isnan(score))


if __name__ == "__main__":
    unittest.main()
