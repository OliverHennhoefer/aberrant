"""Tests for Local Outlier Factor (LOF) anomaly detection model."""

import math
import random
import time
import unittest
from collections import deque

import numpy as np

from aberrant.model.distance.lof import LocalOutlierFactor, _sum_rows
from tests import score_sequences


def _reference_scores(events, k, window_size, distance="euclidean"):
    """Score then learn with the aberrant 1.4.0 algorithm, searching everything.

    Every distance is measured against the whole window and every neighborhood
    is sorted with ``np.argsort``, exactly as the original implementation did,
    so tied distances keep the platform's sort order.
    """
    points = deque(maxlen=window_size)

    def distances_to(point):
        diff = np.array(points) - point
        if distance == "euclidean":
            return np.sqrt(np.sum(diff**2, axis=1))
        return np.sum(np.abs(diff), axis=1)

    def neighborhood(distances, exclude=None):
        ordered = [int(i) for i in np.argsort(distances) if int(i) != exclude]
        if len(ordered) <= k:
            return ordered
        k_distance = distances[ordered[k - 1]]
        return [i for i in ordered if distances[i] <= k_distance]

    def k_distance(index):
        distances = distances_to(points[index])
        members = neighborhood(distances, exclude=index)
        return math.inf if len(members) < k else float(distances[members[-1]])

    def lrd(members, distances):
        total = 0.0
        for index in members:
            total += max(k_distance(index), distances[index])
        return math.inf if total <= 0 else len(members) / total

    scores = []
    for event in events:
        query = np.array([event[name] for name in sorted(event)])
        if len(points) <= k:
            scores.append(0.0)
        else:
            distances = distances_to(query)
            members = neighborhood(distances)
            lrd_query = lrd(members, distances)
            if lrd_query <= 0:
                scores.append(0.0)
            else:
                lof_sum = 0.0
                for member in members:
                    member_distances = distances_to(points[member])
                    lrd_member = lrd(
                        neighborhood(member_distances, exclude=member),
                        member_distances,
                    )
                    if lrd_member > 0:
                        if math.isinf(lrd_member) and math.isinf(lrd_query):
                            lof_sum += 1.0
                        elif not math.isinf(lrd_query):
                            lof_sum += lrd_member / lrd_query
                scores.append(lof_sum / len(members))
        points.append(query)
    return scores


def _tied_stream(seed, n_events, n_features, levels):
    """Draw coordinates from a few integer levels, so distances tie often."""
    rng = np.random.default_rng(seed)
    data = rng.integers(0, levels, size=(n_events, n_features)).astype(float)
    return [
        {f"x{index}": value for index, value in enumerate(row)} for row in data.tolist()
    ]


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
        self.assertEqual(lof._schema.names, ("a", "m", "z"))

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


class TestLocalOutlierFactorExactness(unittest.TestCase):
    """The incremental window must reproduce the original algorithm exactly."""

    def test_score_sequences_match_recorded_reference(self):
        for case in score_sequences.load("local_outlier_factor"):
            with self.subTest(case=case["name"]):
                model = LocalOutlierFactor(**case["params"])
                events = score_sequences.make_stream(**case["stream"])
                scores = score_sequences.score_then_learn(model, events)
                # The tolerance only absorbs last-bit libm differences between
                # platforms; the recording platform reproduces every bit.
                np.testing.assert_allclose(scores, case["scores"], rtol=1e-12, atol=0)

    def test_tied_and_duplicate_points_match_the_original_algorithm(self):
        cases = [
            ({"k": 3, "window_size": 20}, _tied_stream(1, 150, 2, 4)),
            (
                {"k": 3, "window_size": 20, "distance": "manhattan"},
                _tied_stream(2, 150, 2, 4),
            ),
            ({"k": 2, "window_size": 3}, _tied_stream(3, 80, 1, 3)),
            ({"k": 5, "window_size": 30}, _tied_stream(4, 150, 3, 3)),
            ({"k": 4, "window_size": 25}, _tied_stream(5, 120, 2, 2)),
            ({"k": 3, "window_size": 15}, score_sequences.make_stream(6, 120, 3)),
        ]
        for params, events in cases:
            with self.subTest(params=params, n_features=len(events[0])):
                model = LocalOutlierFactor(**params)
                scores = score_sequences.score_then_learn(model, events)
                self.assertEqual(scores, _reference_scores(events, **params))

    def test_neighbor_lists_hold_the_nearest_window_points(self):
        model = LocalOutlierFactor(k=3, window_size=25)
        for step, event in enumerate(_tied_stream(8, 400, 2, 5)):
            model.learn_one(event)
            if step % 17:
                continue
            n = model.n_points
            points = model._window[:, :n].T
            for slot in range(n):
                distances = np.sqrt(np.sum((points - points[slot]) ** 2, axis=1))
                listed = model._near_slots[slot]
                left_out = np.delete(distances, [*listed, slot])
                self.assertEqual(model._near[slot], distances[listed].tolist())
                self.assertEqual(model._near[slot], sorted(model._near[slot]))
                self.assertNotIn(slot, listed)
                if model._complete[slot]:
                    self.assertEqual(left_out.size, 0)
                else:
                    self.assertGreaterEqual(len(listed), model.k)
                    self.assertTrue(np.all(left_out >= model._near[slot][-1]))
                others = np.delete(distances, slot)
                expected = np.sort(others)[model.k - 1] if n > model.k else math.inf
                self.assertEqual(model._k_distances[slot], expected)
                for other in listed:
                    self.assertIn(slot, model._holders[other])
            self.assertEqual(
                sum(len(holders) for holders in model._holders),
                sum(len(listed) for listed in model._near_slots),
            )

    def test_row_sums_follow_numpy_pairwise_order(self):
        rng = np.random.default_rng(9)
        for n_features in [*range(1, 40), 63, 64, 65, 127, 128, 129, 200, 300]:
            values = rng.random((50, n_features)) * np.exp(
                rng.normal(scale=8.0, size=(50, n_features))
            )
            with self.subTest(n_features=n_features):
                np.testing.assert_array_equal(
                    _sum_rows(np.ascontiguousarray(values.T)), np.sum(values, axis=1)
                )

    def test_default_model_processes_two_thousand_events_quickly(self):
        events = score_sequences.make_stream(seed=1, n_events=2000, n_features=8)
        model = LocalOutlierFactor()

        start = time.perf_counter()
        score_sequences.score_then_learn(model, events)

        # Generous bound against large regressions; typical runs take ~0.4 s.
        self.assertLess(time.perf_counter() - start, 5.0)


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
