"""Unit tests for the NETS distance-based anomaly detector."""

import pickle
import unittest
from collections import deque

import numpy as np
import pytest

from aberrant.model.distance import CellNeighborhoodDetector as NETS
from aberrant.model.distance import StationaryRegionNeighborDetector as STARE


class TestNETS(unittest.TestCase):
    """Test suite for NETS."""

    def create_model(self, **overrides: object) -> NETS:
        defaults: dict[str, object] = {
            "k": 6,
            "radius": 1.25,
            "window_size": 64,
            "slide_size": 4,
            "subspace_dim": 2,
            "time_key": "t",
            "warm_up_slides": 2,
            "predict_threshold": 0.5,
            "seed": 42,
            "eps": 1e-9,
        }
        defaults.update(overrides)
        return NETS(**defaults)

    def test_initialization_defaults(self) -> None:
        model = NETS()
        self.assertEqual(model.k, 50)
        self.assertEqual(model.radius, 1.5)
        self.assertEqual(model.window_size, 10_000)
        self.assertEqual(model.slide_size, 500)
        self.assertIsNone(model.subspace_dim)
        self.assertEqual(model.warm_up_slides, 1)
        self.assertEqual(model.predict_threshold, 0.5)

    def test_invalid_parameters(self) -> None:
        with self.assertRaises(ValueError):
            NETS(k=0)
        with self.assertRaises(ValueError):
            NETS(radius=0.0)
        with self.assertRaises(ValueError):
            NETS(window_size=0)
        with self.assertRaises(ValueError):
            NETS(window_size=10, k=10)
        with self.assertRaises(ValueError):
            NETS(slide_size=0)
        with self.assertRaises(ValueError):
            NETS(subspace_dim=0)
        with self.assertRaises(ValueError):
            NETS(time_key="")
        with self.assertRaises(ValueError):
            NETS(warm_up_slides=0)
        with self.assertRaises(ValueError):
            NETS(predict_threshold=-0.1)
        with self.assertRaises(ValueError):
            NETS(predict_threshold=1.1)
        with self.assertRaises(ValueError):
            NETS(eps=0.0)

    def test_input_validation(self) -> None:
        model = self.create_model()

        with self.assertRaises(ValueError):
            model.learn_one({})
        with self.assertRaises(ValueError):
            model.learn_one({"t": 1.0, "x": "bad"})  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            model.learn_one({"x": 1.0})
        with self.assertRaises(ValueError):
            model.learn_one({"t": float("nan"), "x": 1.0})
        with self.assertRaises(ValueError):
            model.score_one({"t": float("inf"), "x": 1.0})

    def test_score_is_zero_before_warmup(self) -> None:
        model = self.create_model(k=4, warm_up_slides=3, slide_size=4)
        for i in range(8):
            model.learn_one({"t": float(i), "x": float(i % 3), "y": float(i % 5)})

        score = model.score_one({"t": 9.0, "x": 0.3, "y": 0.4})
        self.assertEqual(score, 0.0)

    def test_feature_schema_mismatch_raises(self) -> None:
        model = self.create_model()
        model.learn_one({"t": 1.0, "a": 1.0, "b": 2.0})

        with self.assertRaises(ValueError):
            model.learn_one({"t": 2.0, "a": 1.0, "c": 2.0})

    def test_non_monotonic_timestamp_raises(self) -> None:
        model = self.create_model()
        model.learn_one({"t": 1.0, "a": 1.0, "b": 2.0})
        model.learn_one({"t": 2.0, "a": 1.2, "b": 2.1})

        with self.assertRaises(ValueError):
            model.score_one({"t": 1.5, "a": 1.0, "b": 2.0})

        with self.assertRaises(ValueError):
            model.learn_one({"t": 1.0, "a": 1.0, "b": 2.0})

    def test_internal_clock_fallback_without_time_key(self) -> None:
        model = self.create_model(time_key=None, warm_up_slides=1, slide_size=3, k=2)

        first_point = {"a": 1.0, "b": 2.0}
        second_point = {"a": 1.2, "b": 2.1}
        third_point = {"a": 1.3, "b": 2.2}

        self.assertEqual(model.score_one(first_point), 0.0)
        model.learn_one(first_point)
        model.learn_one(second_point)
        model.learn_one(third_point)

        score = model.score_one({"a": 1.1, "b": 2.0})
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_subspace_dim_cannot_exceed_feature_count(self) -> None:
        model = self.create_model(subspace_dim=3)
        with self.assertRaises(ValueError):
            model.learn_one({"t": 1.0, "a": 1.0, "b": 2.0})

    def test_deterministic_behavior(self) -> None:
        model1 = self.create_model(k=4, warm_up_slides=2, slide_size=4, seed=7)
        model2 = self.create_model(k=4, warm_up_slides=2, slide_size=4, seed=7)

        for i in range(40):
            point = {
                "t": float(i),
                "a": float((i % 7) * 0.2),
                "b": float((i % 5) * 0.3),
            }
            model1.learn_one(point)
            model2.learn_one(point)

        query = {"t": 41.0, "a": 0.7, "b": 0.9}
        self.assertAlmostEqual(
            model1.score_one(query),
            model2.score_one(query),
            places=12,
        )

    def test_outlier_scores_higher_than_normal(self) -> None:
        model = self.create_model(k=8, radius=0.75, warm_up_slides=3, slide_size=4)
        rng = np.random.default_rng(123)

        for i in range(250):
            model.learn_one(
                {
                    "t": float(i),
                    "a": float(rng.normal(0.0, 0.15)),
                    "b": float(rng.normal(0.0, 0.15)),
                }
            )

        normal_score = model.score_one({"t": 251.0, "a": 0.02, "b": -0.03})
        outlier_score = model.score_one({"t": 251.0, "a": 6.0, "b": 6.0})
        self.assertGreaterEqual(normal_score, 0.0)
        self.assertGreater(outlier_score, normal_score)

    def test_scores_are_bounded_and_predict_is_binary(self) -> None:
        model = self.create_model(k=3, warm_up_slides=1, slide_size=3)
        for i in range(20):
            model.learn_one({"t": float(i), "a": float(i % 4), "b": float(i % 3)})

        score = model.score_one({"t": 21.0, "a": 10.0, "b": 10.0})
        prediction = model.predict_one({"t": 21.0, "a": 10.0, "b": 10.0})

        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertIn(prediction, (0, 1))

    def test_window_and_cell_state_is_bounded(self) -> None:
        model = self.create_model(
            k=5,
            radius=1.0,
            window_size=12,
            slide_size=3,
            warm_up_slides=1,
        )

        for i in range(50):
            model.learn_one(
                {
                    "t": float(i),
                    "a": float(i % 11),
                    "b": float((i * 2) % 13),
                }
            )

        # Intentional private-state checks: these invariants validate bounded-memory
        # behavior and internal count consistency that are not exposed publicly.
        self.assertLessEqual(len(model._window_entries), 12)
        self.assertEqual(
            sum(map(len, model._cell_members.values())), len(model._window_entries)
        )

    def test_score_only_queries_do_not_grow_neighbor_cache(self) -> None:
        model = self.create_model(
            time_key=None,
            k=2,
            radius=0.1,
            window_size=10,
            slide_size=2,
            warm_up_slides=1,
            subspace_dim=1,
        )

        for i in range(30):
            model.learn_one({"a": float(i), "b": float(i)})

        model.score_one({"a": 29.0, "b": 29.0})
        state_before = pickle.dumps(model)

        for i in range(500):
            value = float(10_000 + i * 10)
            model.score_one({"a": value, "b": value})

        self.assertEqual(pickle.dumps(model), state_before)

    def test_neighbor_score_refreshes_after_learning(self) -> None:
        model = self.create_model(
            time_key=None,
            k=2,
            radius=1.0,
            window_size=50,
            slide_size=10,
            warm_up_slides=1,
            subspace_dim=1,
        )

        for i in range(8):
            value = float(5 + i)
            model.learn_one({"a": value, "b": value})
        model.learn_one({"a": 0.1, "b": 0.1})
        model.learn_one({"a": 20.0, "b": 20.0})

        query = {"a": 0.0, "b": 0.0}
        self.assertEqual(model.score_one(query), 0.5)

        model.learn_one({"a": 0.2, "b": 0.1})
        refreshed_score = model.score_one(query)
        self.assertEqual(refreshed_score, 0.0)

    def test_offset_lookup_matches_radius_counts(self) -> None:
        model = self.create_model(
            time_key=None, k=20, radius=1.0, slide_size=1, warm_up_slides=1
        )
        points = [
            np.array([a + 0.2, b + 0.2]) for a in range(-3, 4) for b in range(-3, 4)
        ]
        for point in points:
            model.learn_one({"a": point[0], "b": point[1]})
        query = np.array([0.2, 0.2])
        count = sum(
            np.dot(point - query, point - query) <= 1.0 + model.eps for point in points
        )
        self.assertEqual(
            model.score_one({"a": query[0], "b": query[1]}), 1.0 - count / model.k
        )
        self.assertEqual(len(model._neighbor_offsets_cache[2]), 25)

    def test_high_dimensional_tolerance_does_not_enumerate_offsets(self) -> None:
        model = self.create_model(
            time_key=None, k=2, radius=1.0, eps=100.0, slide_size=1, warm_up_slides=1
        )
        for value in (0.0, 1.0, 20.0):
            model.learn_one({str(dim): value for dim in range(20)})
        query = {str(dim): 0.0 for dim in range(20)}
        self.assertEqual(model.score_one(query), 0.0)
        self.assertEqual(model._neighbor_offsets_cache, {})

    def test_reset_restores_cold_state(self) -> None:
        model = self.create_model(k=3, warm_up_slides=1, slide_size=3)
        for i in range(20):
            model.learn_one({"t": float(i), "a": float(i), "b": float(i + 1)})

        self.assertGreater(model.n_samples_seen, 0)
        model.reset()

        self.assertEqual(model.n_samples_seen, 0)
        self.assertIsNone(model._boundary.schema.names)
        self.assertEqual(len(model._window_entries), 0)
        self.assertEqual(model.score_one({"t": 1.0, "a": 1.0, "b": 2.0}), 0.0)

    def test_repr_contains_key_config(self) -> None:
        model = self.create_model(k=9, window_size=128, slide_size=16, subspace_dim=1)
        output = repr(model)
        self.assertIn("CellNeighborhoodDetector", output)
        self.assertIn("k=9", output)
        self.assertIn("window_size=128", output)
        self.assertIn("slide_size=16", output)
        self.assertIn("subspace_dim=1", output)


if __name__ == "__main__":
    unittest.main()


def test_continuous_score_matches_brute_force_through_evictions():
    model = NETS(k=4, radius=1.0, window_size=16, slide_size=1, seed=42)
    window = deque(maxlen=16)
    rng = np.random.default_rng(42)
    for point in rng.normal(size=(100, 2)):
        model.learn_one(dict(zip(("a", "b"), point, strict=True)))
        window.append(point)
        if len(window) <= model.k:
            continue
        for query in rng.normal(size=(3, 2)):
            neighbors = sum(
                np.dot(point - query, point - query) <= 1.0 + model.eps
                for point in window
            )
            expected = 1.0 - min(neighbors / model.k, 1.0)
            assert (
                model.score_one(dict(zip(("a", "b"), query, strict=True))) == expected
            )


def test_score_only_queries_in_shared_subspace_do_not_accumulate_state():
    model = NETS(k=2, radius=1.0, window_size=5, slide_size=1, subspace_dim=1, seed=42)
    for value in (0.1, 0.2, 0.3):
        model.learn_one({"a": value, "b": value})
    model.score_one({"a": 0.1, "b": 0.1})
    before = pickle.dumps(model)
    for i in range(1000):
        model.score_one({"a": 0.1, "b": float(10 + i * 2)})
        model.score_one({"a": float(10 + i * 2), "b": 0.1})
    assert pickle.dumps(model) == before


@pytest.mark.parametrize("detector", [NETS, STARE])
@pytest.mark.parametrize(
    ("point", "query", "eps"),
    [
        (0.9999999999, 2.0, 1e-9),
        (-1.0000000001, 0.0, 1e-9),
        (float(np.nextafter(1.0, 0.0)), 2.0, 1e-20),
        (2.7, 0.0, 8.0),
    ],
)
def test_tolerance_neighbors_across_nonadjacent_cells(detector, point, query, eps):
    model = detector(k=2, radius=1.0, window_size=5, slide_size=1, eps=eps)
    for value in (point, 10.0, 20.0):
        model.learn_one({"x": value})
    assert model.score_one({"x": query}) == 0.5


@pytest.mark.parametrize("detector", [NETS, STARE])
@pytest.mark.parametrize("offset", [1e16, -1e16])
def test_neighbor_candidates_preserve_distances_at_large_offsets(detector, offset):
    # This tolerance makes radius² + eps exactly 4.0 in float arithmetic.
    eps = float(np.nextafter(3.51, np.inf))
    model = detector(k=2, radius=0.7, eps=eps, window_size=5, slide_size=1)
    for value in (offset, offset + 100.0, offset + 200.0):
        model.learn_one({"x": value})
    assert model.score_one({"x": offset + 2.0}) == 0.5


@pytest.mark.parametrize("detector", [NETS, STARE])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_small_radius_accepts_finite_points_without_quotient_overflow(detector, sign):
    model = detector(k=2, radius=1e-320, window_size=5, slide_size=1)
    with np.errstate(over="raise", invalid="raise"):
        for value in (1.0, 2.0, 3.0):
            model.learn_one({"x": sign * value})
        assert model.score_one({"x": sign}) == 0.5


@pytest.mark.parametrize("detector", [NETS, STARE])
@pytest.mark.parametrize("radius", [float("nan"), float("inf"), -float("inf")])
def test_radius_must_be_finite(detector, radius):
    with pytest.raises(ValueError, match="radius must be finite and positive"):
        detector(radius=radius)


@pytest.mark.parametrize("detector", [NETS, STARE])
def test_subnormal_squared_distance_rounding_preserves_candidate_coverage(detector):
    model = detector(k=2, radius=1e-200, eps=5e-324, window_size=5, slide_size=1)
    for value in (2.5e-162, 1.0, 2.0):
        model.learn_one({"x": value})
    point = np.array([2.5e-162])
    assert float(np.dot(point, point)) == model.eps
    assert model.score_one({"x": 0.0}) == 0.5


@pytest.mark.parametrize("n_features", [1, 2, 8])
@pytest.mark.parametrize("eps", [1e-9, 8.0])
def test_both_detectors_match_window_reference_through_eviction(n_features, eps):
    config = {
        "k": 4,
        "radius": 1.0,
        "window_size": 13,
        "slide_size": 2,
        "warm_up_slides": 3,
        "time_key": "t",
        "eps": eps,
    }
    models = [NETS(**config), STARE(**config)]
    window = deque(maxlen=13)
    rng = np.random.default_rng(719)
    names = tuple(str(dim) for dim in range(n_features))
    for sample_index in range(80):
        timestamp = float(sample_index // 3)
        for query in rng.normal(size=(3, n_features)):
            event = dict(zip(reversed(names), reversed(query), strict=True))
            event["t"] = timestamp
            count = sum(
                np.dot(point - query, point - query) <= 1.0 + eps for point in window
            )
            expected = 0.0 if sample_index < 6 else 1.0 - min(count / 4, 1.0)
            for model in models:
                assert model.score_one(event) == expected
                assert model.predict_one(event) == int(expected >= 0.5)
        point = rng.normal(size=n_features)
        event = dict(zip(names, point, strict=True))
        event["t"] = timestamp
        for model in models:
            model.learn_one(event)
        window.append(point)
    for model in models:
        model.reset()
        assert model.n_samples_seen == 0
        assert model.score_one({"new": 1.0, "t": 0.0}) == 0.0
        model.learn_one({"new": 1.0, "t": 0.0})
