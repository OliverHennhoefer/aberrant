"""Unit tests for the RSHash sketch-based anomaly detector."""

import unittest

import numpy as np

from aberrant.model.sketch import StreamingRSHash as RSHash


class TestRSHash(unittest.TestCase):
    """Test suite for RSHash."""

    def create_model(self, **overrides: object) -> RSHash:
        defaults: dict[str, object] = {
            "components_num": 12,
            "hash_num": 4,
            "bins": 128,
            "subspace_size": 2,
            "bin_width": 1.0,
            "decay": 0.01,
            "warm_up_samples": 16,
            "time_key": "t",
            "seed": 42,
        }
        defaults.update(overrides)
        return RSHash(**defaults)

    def test_initialization_defaults(self) -> None:
        model = RSHash()
        self.assertEqual(model.components_num, 24)
        self.assertEqual(model.hash_num, 4)
        self.assertEqual(model.bins, 256)
        self.assertEqual(model.subspace_size, None)
        self.assertEqual(model.bin_width, 1.0)
        self.assertEqual(model.decay, 0.01)
        self.assertEqual(model.warm_up_samples, 64)

    def test_invalid_parameters(self) -> None:
        with self.assertRaises(ValueError):
            RSHash(components_num=0)
        with self.assertRaises(ValueError):
            RSHash(hash_num=0)
        with self.assertRaises(ValueError):
            RSHash(bins=0)
        with self.assertRaises(ValueError):
            RSHash(subspace_size=0)
        with self.assertRaises(ValueError):
            RSHash(bin_width=0.0)
        with self.assertRaises(ValueError):
            RSHash(decay=-0.1)
        with self.assertRaises(ValueError):
            RSHash(warm_up_samples=0)
        with self.assertRaises(ValueError):
            RSHash(time_key="")

    def test_input_validation(self) -> None:
        model = self.create_model()

        with self.assertRaises(ValueError):
            model.learn_one({})
        with self.assertRaises(ValueError):
            model.learn_one({"t": 1.0, "x": "bad"})  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            model.learn_one({"x": 1.0})  # Missing time key.
        with self.assertRaises(ValueError):
            model.learn_one({"t": float("nan"), "x": 1.0})
        with self.assertRaises(ValueError):
            model.score_one({"t": float("inf"), "x": 1.0})

    def test_score_is_zero_before_warmup(self) -> None:
        model = self.create_model(warm_up_samples=32)
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
        model.learn_one({"t": 2.0, "a": 1.1, "b": 1.9})

        with self.assertRaises(ValueError):
            model.score_one({"t": 1.5, "a": 1.0, "b": 2.0})
        with self.assertRaises(ValueError):
            model.learn_one({"t": 1.0, "a": 1.0, "b": 2.0})

    def test_internal_clock_fallback_without_time_key(self) -> None:
        model = self.create_model(time_key=None, warm_up_samples=4)

        first_point = {"a": 1.0, "b": 2.0}
        second_point = {"a": 1.1, "b": 2.1}
        third_point = {"a": 1.2, "b": 2.2}
        fourth_point = {"a": 1.3, "b": 2.3}

        self.assertEqual(model.score_one(first_point), 0.0)
        model.learn_one(first_point)
        model.learn_one(second_point)
        model.learn_one(third_point)
        model.learn_one(fourth_point)

        score = model.score_one({"a": 1.1, "b": 2.0})
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_subspace_size_larger_than_features_raises(self) -> None:
        model = self.create_model(subspace_size=3, time_key=None)
        with self.assertRaises(ValueError):
            model.learn_one({"a": 1.0, "b": 2.0})

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

    def test_outlier_scores_higher_than_normal(self) -> None:
        model = self.create_model(
            components_num=20,
            hash_num=4,
            bins=256,
            subspace_size=2,
            warm_up_samples=32,
            decay=0.005,
            seed=11,
        )
        rng = np.random.default_rng(123)

        for i in range(500):
            point = {
                "t": float(i),
                "a": float(rng.normal(0.0, 0.35)),
                "b": float(rng.normal(0.0, 0.35)),
            }
            model.learn_one(point)

        normal_score = model.score_one({"t": 501.0, "a": 0.05, "b": -0.02})
        outlier_score = model.score_one({"t": 501.0, "a": 6.0, "b": 6.0})

        self.assertGreaterEqual(normal_score, 0.0)
        self.assertGreater(outlier_score, normal_score)

    def test_state_shapes_are_bounded(self) -> None:
        model = self.create_model(
            components_num=8,
            hash_num=3,
            bins=64,
            subspace_size=2,
        )
        model.learn_one({"t": 0.0, "a": 1.0, "b": 2.0, "c": 3.0})

        if model._counts is None:
            self.fail("Counts tensor was not initialized")

        counts_shape = model._counts.shape
        for i in range(1, 2_000):
            model.learn_one(
                {
                    "t": float(i),
                    "a": float(i % 11),
                    "b": float((i * 2) % 13),
                    "c": float((i * 3) % 17),
                }
            )

        self.assertEqual(model._counts.shape, counts_shape)

    def test_reset_restores_cold_state(self) -> None:
        model = self.create_model()
        for i in range(100):
            model.learn_one({"t": float(i), "a": float(i), "b": float(i + 1)})

        self.assertGreater(model.n_samples_seen, 0)
        model.reset()

        self.assertEqual(model.n_samples_seen, 0)
        self.assertIsNone(model._boundary.schema.names)
        self.assertEqual(model.score_one({"t": 1.0, "a": 1.0, "b": 2.0}), 0.0)

    def test_repr_contains_key_config(self) -> None:
        model = self.create_model(components_num=18, hash_num=5)
        output = repr(model)
        self.assertIn("StreamingRSHash", output)
        self.assertIn("components_num=18", output)
        self.assertIn("hash_num=5", output)


if __name__ == "__main__":
    unittest.main()
