"""Unit tests for the STARE distance-based anomaly detector."""

import unittest

import numpy as np

from aberrant.model.distance import StationaryRegionNeighborDetector as STARE


class TestSTARE(unittest.TestCase):
    """Test suite for STARE."""

    def create_model(self, **overrides: object) -> STARE:
        defaults: dict[str, object] = {
            "k": 6,
            "radius": 1.25,
            "window_size": 64,
            "slide_size": 4,
            "skip_threshold": 0.2,
            "time_key": "t",
            "warm_up_slides": 2,
            "predict_threshold": 0.5,
            "eps": 1e-9,
        }
        defaults.update(overrides)
        return STARE(**defaults)

    def test_initialization_defaults(self) -> None:
        model = STARE()
        self.assertEqual(model.k, 50)
        self.assertEqual(model.radius, 1.0)
        self.assertEqual(model.window_size, 2048)
        self.assertEqual(model.slide_size, 128)
        self.assertEqual(model.skip_threshold, 0.1)
        self.assertEqual(model.warm_up_slides, 1)
        self.assertEqual(model.predict_threshold, 0.5)

    def test_invalid_parameters(self) -> None:
        with self.assertRaises(ValueError):
            STARE(k=0)
        with self.assertRaises(ValueError):
            STARE(radius=0.0)
        with self.assertRaises(ValueError):
            STARE(window_size=0)
        with self.assertRaises(ValueError):
            STARE(window_size=10, k=10)
        with self.assertRaises(ValueError):
            STARE(slide_size=0)
        with self.assertRaises(ValueError):
            STARE(skip_threshold=-0.1)
        with self.assertRaises(ValueError):
            STARE(skip_threshold=1.1)
        with self.assertRaises(ValueError):
            STARE(time_key="")
        with self.assertRaises(ValueError):
            STARE(warm_up_slides=0)
        with self.assertRaises(ValueError):
            STARE(predict_threshold=-0.1)
        with self.assertRaises(ValueError):
            STARE(predict_threshold=1.1)
        with self.assertRaises(ValueError):
            STARE(eps=0.0)

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

    def test_deterministic_behavior(self) -> None:
        model1 = self.create_model(k=4, warm_up_slides=2, slide_size=4)
        model2 = self.create_model(k=4, warm_up_slides=2, slide_size=4)

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
            model1.score_one(query), model2.score_one(query), places=12
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

    def test_window_state_is_bounded(self) -> None:
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

        self.assertLessEqual(len(model._window_entries), 12)
        self.assertEqual(sum(model._cell_counts.values()), len(model._window_entries))

    def test_dirty_cells_remain_bounded_after_evictions(self) -> None:
        model = self.create_model(
            time_key=None,
            k=2,
            radius=0.1,
            window_size=10,
            slide_size=2,
            warm_up_slides=1,
        )

        for i in range(200):
            model.learn_one({"a": float(i * 10), "b": float(i * 10)})

        self.assertLessEqual(len(model._window_entries), model.window_size)
        self.assertLessEqual(len(model._dirty_cells), len(model._cell_counts))

    def test_score_only_queries_do_not_grow_neighbor_cache(self) -> None:
        model = self.create_model(
            time_key=None,
            k=2,
            radius=0.1,
            window_size=10,
            slide_size=2,
            warm_up_slides=1,
        )

        for i in range(30):
            model.learn_one({"a": float(i), "b": float(i)})

        model.score_one({"a": 29.0, "b": 29.0})
        cache_size_before = len(model._neighbor_cache)

        for i in range(500):
            value = float(10_000 + i * 10)
            model.score_one({"a": value, "b": value})

        self.assertEqual(len(model._neighbor_cache), cache_size_before)

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
        model = self.create_model(k=9, window_size=128, slide_size=16)
        output = repr(model)
        self.assertIn("StationaryRegionNeighborDetector", output)
        self.assertIn("k=9", output)
        self.assertIn("window_size=128", output)
        self.assertIn("slide_size=16", output)


if __name__ == "__main__":
    unittest.main()
