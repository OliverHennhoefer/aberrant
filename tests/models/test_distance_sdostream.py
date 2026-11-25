"""Unit tests for the SDOStream distance-based anomaly detector."""

import unittest

import numpy as np

from aberrant.model.distance import SDOStream


class TestSDOStream(unittest.TestCase):
    """Test suite for SDOStream."""

    def create_model(self, **overrides: object) -> SDOStream:
        defaults: dict[str, object] = {
            "k": 64,
            "T": 128.0,
            "qv": 0.3,
            "x_neighbors": 4,
            "distance": "euclidean",
            "time_key": "t",
            "warm_up_observers": 4,
            "seed": 42,
        }
        defaults.update(overrides)
        return SDOStream(**defaults)

    def test_initialization_defaults(self) -> None:
        model = SDOStream()
        self.assertEqual(model.k, 256)
        self.assertEqual(model.T, 512.0)
        self.assertEqual(model.qv, 0.3)
        self.assertEqual(model.x_neighbors, 6)
        self.assertEqual(model.distance, "euclidean")
        self.assertEqual(model.warm_up_observers, 6)

    def test_invalid_parameters(self) -> None:
        with self.assertRaises(ValueError):
            SDOStream(k=0)
        with self.assertRaises(ValueError):
            SDOStream(T=0.0)
        with self.assertRaises(ValueError):
            SDOStream(qv=-0.1)
        with self.assertRaises(ValueError):
            SDOStream(qv=1.0)
        with self.assertRaises(ValueError):
            SDOStream(x_neighbors=0)
        with self.assertRaises(ValueError):
            SDOStream(k=4, x_neighbors=5)
        with self.assertRaises(ValueError):
            SDOStream(distance="cosine")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            SDOStream(distance="minkowski", minkowski_p=0.0)
        with self.assertRaises(ValueError):
            SDOStream(time_key="")
        with self.assertRaises(ValueError):
            SDOStream(warm_up_observers=0)

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
        model = self.create_model(warm_up_observers=6)
        for i in range(3):
            model.learn_one({"t": float(i), "x": float(i), "y": float(i + 1)})

        score = model.score_one({"t": 4.0, "x": 0.1, "y": 0.2})
        self.assertEqual(score, 0.0)

    def test_feature_schema_mismatch_raises(self) -> None:
        model = self.create_model()
        model.learn_one({"t": 1.0, "a": 1.0, "b": 2.0})

        with self.assertRaises(ValueError):
            model.learn_one({"t": 2.0, "a": 1.0, "c": 2.0})

    def test_non_monotonic_timestamp_raises(self) -> None:
        model = self.create_model()
        model.learn_one({"t": 1.0, "a": 1.0, "b": 2.0})
        model.learn_one({"t": 2.0, "a": 1.1, "b": 1.9})

        with self.assertRaises(ValueError):
            model.score_one({"t": 1.5, "a": 1.0, "b": 2.0})

        with self.assertRaises(ValueError):
            model.learn_one({"t": 1.0, "a": 1.0, "b": 2.0})

    def test_internal_clock_fallback_without_time_key(self) -> None:
        model = self.create_model(time_key=None, warm_up_observers=3)

        first_point = {"a": 1.0, "b": 2.0}
        second_point = {"a": 1.1, "b": 2.1}
        third_point = {"a": 1.2, "b": 2.2}

        self.assertEqual(model.score_one(first_point), 0.0)
        model.learn_one(first_point)
        model.learn_one(second_point)
        model.learn_one(third_point)

        score = model.score_one({"a": 1.1, "b": 2.0})
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)

    def test_deterministic_with_seed(self) -> None:
        model1 = self.create_model(seed=7)
        model2 = self.create_model(seed=7)

        for i in range(120):
            point = {
                "t": float(i),
                "a": float((i % 9) * 0.1),
                "b": float((i % 7) * 0.2),
            }
            model1.learn_one(point)
            model2.learn_one(point)

        query = {"t": 121.0, "a": 0.25, "b": 0.85}
        self.assertAlmostEqual(
            model1.score_one(query),
            model2.score_one(query),
            places=12,
        )

    def test_replacement_evicts_the_lowest_fading_normalized_activity(self) -> None:
        # The reference compares observations / (1 - fading**age): an old
        # observer that stays popular outranks a young one that is barely hit.
        model = self.create_model(k=2, T=10.0, x_neighbors=1, warm_up_observers=2)
        model.learn_one({"t": 0.0, "a": 0.0, "b": 0.0})
        assert model._observations is not None
        assert model._time_added is not None and model._time_touched is not None
        model._n_observers = 2
        now = 100.0
        model._observations[:2] = [8.0, 1.0]
        model._time_added[:2] = [0.0, 95.0]
        model._time_touched[:2] = [now, now]
        fading = np.exp(-1.0 / model.T)
        normalized = [8.0 / (1 - fading**100.0), 1.0 / (1 - fading**5.0)]
        self.assertEqual(int(np.argmin(normalized)), 1)
        self.assertEqual(model._choose_replacement_index(now), 1)

    def test_observer_sampling_does_not_depend_on_the_time_unit(self) -> None:
        rng = np.random.default_rng(3)
        points = rng.normal(size=(400, 2))

        def scores(unit: float) -> list[float]:
            model = self.create_model(k=16, T=32.0 * unit, seed=11)
            out = []
            for i, (a, b) in enumerate(points):
                point = {"t": i * unit, "a": float(a), "b": float(b)}
                out.append(model.score_one(point))
                model.learn_one(point)
            return out

        reference = scores(1.0)
        for unit in (0.01, 60.0):
            with self.subTest(unit=unit):
                np.testing.assert_allclose(scores(unit), reference, rtol=1e-9)

    def test_no_observer_is_sampled_without_elapsed_time(self) -> None:
        # The sampling prefactor k**2 / (x_neighbors * T) is 4096 here, so any
        # elapsed time at all would admit nearly every point.
        model = self.create_model(k=64, T=1.0, x_neighbors=1, warm_up_observers=2)
        for i in range(50):
            model.learn_one({"t": 5.0, "a": float(i), "b": float(-i)})
        self.assertEqual(model.n_observers, 1)

    def test_outlier_scores_higher_than_normal(self) -> None:
        model = self.create_model(
            k=96,
            T=64.0,
            qv=0.2,
            x_neighbors=6,
            warm_up_observers=10,
            seed=11,
        )
        rng = np.random.default_rng(123)

        for i in range(400):
            point = {
                "t": float(i),
                "a": float(rng.normal(0.0, 0.25)),
                "b": float(rng.normal(0.0, 0.25)),
            }
            model.learn_one(point)

        normal_score = model.score_one({"t": 401.0, "a": 0.05, "b": -0.02})
        outlier_score = model.score_one({"t": 401.0, "a": 6.0, "b": 6.0})

        self.assertGreaterEqual(normal_score, 0.0)
        self.assertGreater(outlier_score, normal_score)

    def test_observer_count_is_bounded(self) -> None:
        model = self.create_model(k=32, x_neighbors=4, warm_up_observers=4, seed=99)

        for i in range(1_000):
            model.learn_one(
                {
                    "t": float(i),
                    "a": float((i * 3) % 17),
                    "b": float((i * 5) % 19),
                }
            )

        self.assertLessEqual(model.n_observers, 32)

    def test_reset_restores_cold_state(self) -> None:
        model = self.create_model()
        for i in range(80):
            model.learn_one({"t": float(i), "a": float(i), "b": float(i + 1)})

        self.assertGreater(model.n_observers, 0)
        model.reset()

        self.assertEqual(model.n_observers, 0)
        self.assertIsNone(model._boundary.schema.names)
        self.assertEqual(model.score_one({"t": 1.0, "a": 1.0, "b": 2.0}), 0.0)

    def test_repr_contains_key_config(self) -> None:
        model = self.create_model(k=72, x_neighbors=5)
        output = repr(model)
        self.assertIn("SDOStream", output)
        self.assertIn("k=72", output)
        self.assertIn("x_neighbors=5", output)

    def test_distance_variants_return_valid_scores(self) -> None:
        for metric in ("euclidean", "manhattan", "chebyshev", "minkowski"):
            with self.subTest(metric=metric):
                model = self.create_model(
                    distance=metric,
                    minkowski_p=3.0,
                    warm_up_observers=4,
                )
                for i in range(20):
                    model.learn_one(
                        {
                            "t": float(i),
                            "a": float(i % 4),
                            "b": float((i * 2) % 5),
                        }
                    )

                score = model.score_one({"t": 21.0, "a": 1.5, "b": 2.5})
                self.assertIsInstance(score, float)
                self.assertGreaterEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()
