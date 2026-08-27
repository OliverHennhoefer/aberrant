"""Unit tests for the LODA sketch-based anomaly detector."""

import unittest

import numpy as np

from aberrant.model.sketch import StreamingLODA as LODA


class TestLODA(unittest.TestCase):
    """Test suite for LODA."""

    def create_model(self, **overrides: object) -> LODA:
        defaults: dict[str, object] = {
            "n_projections": 32,
            "n_bins": 24,
            "sparsity": 0.4,
            "warm_up_samples": 32,
            "decay": 0.99,
            "time_key": "t",
            "pseudocount": 0.5,
            "predict_threshold": 0.75,
            "seed": 42,
            "eps": 1e-12,
        }
        defaults.update(overrides)
        return LODA(**defaults)

    def test_initialization_defaults(self) -> None:
        model = LODA()
        self.assertEqual(model.n_projections, 100)
        self.assertEqual(model.n_bins, 32)
        self.assertIsNone(model.sparsity)
        self.assertEqual(model.warm_up_samples, 256)
        self.assertEqual(model.decay, 1.0)
        self.assertEqual(model.pseudocount, 1.0)
        self.assertEqual(model.predict_threshold, 0.5)

    def test_invalid_parameters(self) -> None:
        with self.assertRaises(ValueError):
            LODA(n_projections=0)
        with self.assertRaises(ValueError):
            LODA(n_bins=1)
        with self.assertRaises(ValueError):
            LODA(sparsity=0.0)
        with self.assertRaises(ValueError):
            LODA(sparsity=1.5)
        with self.assertRaises(ValueError):
            LODA(warm_up_samples=0)
        with self.assertRaises(ValueError):
            LODA(decay=0.0)
        with self.assertRaises(ValueError):
            LODA(decay=1.1)
        with self.assertRaises(ValueError):
            LODA(time_key="")
        with self.assertRaises(ValueError):
            LODA(pseudocount=0.0)
        with self.assertRaises(ValueError):
            LODA(predict_threshold=-1.0)
        with self.assertRaises(ValueError):
            LODA(eps=0.0)

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
        model = self.create_model(warm_up_samples=20)
        for i in range(10):
            model.learn_one({"t": float(i), "a": float(i), "b": float(i + 1)})

        score = model.score_one({"t": 11.0, "a": 0.1, "b": 0.2})
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
            model.score_one({"t": 1.5, "a": 1.1, "b": 2.0})
        with self.assertRaises(ValueError):
            model.learn_one({"t": 1.0, "a": 1.0, "b": 2.0})

    def test_internal_clock_fallback_without_time_key(self) -> None:
        model = self.create_model(
            time_key=None,
            warm_up_samples=4,
            n_projections=16,
            n_bins=12,
        )

        first_point = {"a": 1.0, "b": 2.0}
        self.assertEqual(model.score_one(first_point), 0.0)

        model.learn_one(first_point)
        model.learn_one({"a": 1.1, "b": 2.1})
        model.learn_one({"a": 1.2, "b": 2.2})
        model.learn_one({"a": 1.3, "b": 2.3})

        score = model.score_one({"a": 1.1, "b": 2.0})
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)

    def test_deterministic_with_seed(self) -> None:
        model1 = self.create_model(seed=7)
        model2 = self.create_model(seed=7)

        for i in range(120):
            point = {
                "t": float(i),
                "a": float((i % 11) * 0.1),
                "b": float((i % 9) * 0.15),
            }
            model1.learn_one(point)
            model2.learn_one(point)

        query = {"t": 121.0, "a": 0.37, "b": 0.56}
        self.assertAlmostEqual(
            model1.score_one(query),
            model2.score_one(query),
            places=12,
        )

    def test_outlier_scores_higher_than_normal(self) -> None:
        model = self.create_model(
            n_projections=48,
            n_bins=32,
            warm_up_samples=64,
            decay=0.995,
            seed=11,
        )
        rng = np.random.default_rng(123)

        for i in range(500):
            model.learn_one(
                {
                    "t": float(i),
                    "a": float(rng.normal(0.0, 0.2)),
                    "b": float(rng.normal(0.0, 0.2)),
                }
            )

        normal_score = model.score_one({"t": 501.0, "a": 0.02, "b": -0.03})
        outlier_score = model.score_one({"t": 501.0, "a": 6.0, "b": 6.0})

        self.assertGreaterEqual(normal_score, 0.0)
        self.assertGreater(outlier_score, normal_score)

    def test_state_shapes_are_bounded(self) -> None:
        model = self.create_model(
            n_projections=12,
            n_bins=16,
            warm_up_samples=10,
            time_key=None,
        )

        for i in range(250):
            model.learn_one(
                {
                    "a": float(i % 7),
                    "b": float((i * 2) % 11),
                    "c": float((i * 3) % 13),
                }
            )

        if (
            model._projection_matrix is None
            or model._bin_edges is None
            or model._bin_counts is None
            or model._bin_totals is None
        ):
            self.fail("LODA learned state was not initialized")

        self.assertEqual(model._projection_matrix.shape, (12, 3))
        self.assertEqual(model._bin_edges.shape, (12, 17))
        self.assertEqual(model._bin_counts.shape, (12, 16))
        self.assertEqual(model._bin_totals.shape, (12,))
        self.assertEqual(len(model._warmup_buffer), 0)

    def test_score_queries_do_not_mutate_histograms(self) -> None:
        model = self.create_model(
            n_projections=24,
            n_bins=20,
            warm_up_samples=12,
            decay=1.0,
            time_key=None,
        )
        for i in range(100):
            model.learn_one({"a": float(i % 10), "b": float((i * 2) % 9)})

        if model._bin_counts is None or model._bin_totals is None:
            self.fail("Histogram state was not initialized")

        before_counts = model._bin_counts.copy()
        before_totals = model._bin_totals.copy()

        for i in range(500):
            value = float(10_000 + i)
            model.score_one({"a": value, "b": value})

        np.testing.assert_allclose(model._bin_counts, before_counts)
        np.testing.assert_allclose(model._bin_totals, before_totals)

    def test_predict_is_binary(self) -> None:
        model = self.create_model(warm_up_samples=4)
        for i in range(50):
            model.learn_one(
                {
                    "t": float(i),
                    "a": float(i % 4),
                    "b": float(i % 3),
                }
            )

        prediction = model.predict_one({"t": 51.0, "a": 10.0, "b": 10.0})
        self.assertIn(prediction, (0, 1))

    def test_reset_restores_cold_state(self) -> None:
        model = self.create_model()
        for i in range(100):
            model.learn_one({"t": float(i), "a": float(i), "b": float(i + 1)})

        self.assertGreater(model.n_samples_seen, 0)
        model.reset()

        self.assertEqual(model.n_samples_seen, 0)
        self.assertIsNone(model._boundary.schema.names)
        self.assertIsNone(model._projection_matrix)
        self.assertEqual(model.score_one({"t": 1.0, "a": 1.0, "b": 2.0}), 0.0)

    def test_repr_contains_key_config(self) -> None:
        model = self.create_model(n_projections=18, n_bins=14, sparsity=0.25)
        output = repr(model)
        self.assertIn("StreamingLODA", output)
        self.assertIn("n_projections=18", output)
        self.assertIn("n_bins=14", output)
        self.assertIn("sparsity=0.25", output)


if __name__ == "__main__":
    unittest.main()
