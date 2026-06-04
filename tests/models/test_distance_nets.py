"""Unit tests for the NETS distance-based anomaly detector."""

import unittest

import numpy as np

from aberrant.model.distance import NETS


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
            sum(model._full_cell_counts.values()), len(model._window_entries)
        )
        self.assertEqual(
            sum(model._sub_cell_counts.values()), len(model._window_entries)
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
        cache_size_before = len(model._upper_bound_cache)

        for i in range(500):
            value = float(10_000 + i * 10)
            model.score_one({"a": value, "b": value})

        self.assertEqual(len(model._upper_bound_cache), cache_size_before)

    def test_upper_bound_cache_refreshes_after_learning(self) -> None:
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
        self.assertEqual(model.score_one(query), 1.0)

        model.learn_one({"a": 0.2, "b": 0.1})
        refreshed_score = model.score_one(query)
        self.assertLess(refreshed_score, 1.0)

    def test_neighbor_cells_uses_offset_lookup_when_beneficial(self) -> None:
        model = self.create_model()
        center = (0, 0)
        cell_counts = {
            (-1, -1): 1,
            (-1, 0): 1,
            (-1, 1): 1,
            (0, -1): 1,
            (0, 0): 2,
            (0, 1): 1,
            (1, -1): 1,
            (1, 0): 1,
            (1, 1): 1,
            (3, 3): 1,
            (5, 0): 1,
            (0, 5): 1,
            (-4, 2): 1,
            (7, -7): 1,
        }

        expected = {
            existing
            for existing in cell_counts
            if model._are_neighbor_cells(existing, center)
        }
        actual = set(model._neighbor_cells(center, cell_counts))

        self.assertEqual(actual, expected)
        self.assertIn(2, model._neighbor_offsets_cache)
        self.assertEqual(len(model._neighbor_offsets_cache[2]), 9)

    def test_neighbor_cells_falls_back_to_scan_for_large_offset_space(self) -> None:
        model = self.create_model()
        center = (0, 0, 0, 0, 0, 0)
        cell_counts = {
            center: 1,
            (1, 0, 0, 0, 0, 0): 1,
            (-1, 1, 0, 0, 0, 0): 1,
            (0, -1, 1, 0, 0, 0): 1,
            (2, 0, 0, 0, 0, 0): 1,
            (0, 0, 0, 0, 0, 3): 1,
            (5, 5, 5, 5, 5, 5): 1,
            (-7, 0, 0, 0, 0, 0): 1,
        }

        expected = {
            existing
            for existing in cell_counts
            if model._are_neighbor_cells(existing, center)
        }
        actual = set(model._neighbor_cells(center, cell_counts))

        self.assertEqual(actual, expected)
        self.assertNotIn(6, model._neighbor_offsets_cache)

    def test_reset_restores_cold_state(self) -> None:
        model = self.create_model(k=3, warm_up_slides=1, slide_size=3)
        for i in range(20):
            model.learn_one({"t": float(i), "a": float(i), "b": float(i + 1)})

        self.assertGreater(model.n_samples_seen, 0)
        model.reset()

        self.assertEqual(model.n_samples_seen, 0)
        self.assertIsNone(model._feature_order)
        self.assertEqual(len(model._window_entries), 0)
        self.assertEqual(model.score_one({"t": 1.0, "a": 1.0, "b": 2.0}), 0.0)

    def test_repr_contains_key_config(self) -> None:
        model = self.create_model(k=9, window_size=128, slide_size=16, subspace_dim=1)
        output = repr(model)
        self.assertIn("NETS", output)
        self.assertIn("k=9", output)
        self.assertIn("window_size=128", output)
        self.assertIn("slide_size=16", output)
        self.assertIn("subspace_dim=1", output)


if __name__ == "__main__":
    unittest.main()
