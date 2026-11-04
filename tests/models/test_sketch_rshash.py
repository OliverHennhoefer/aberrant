"""Unit tests for the RSHash sketch-based anomaly detector."""

import math
import time
import unittest
from unittest.mock import patch

import numpy as np

from aberrant.model.sketch import StreamingRSHash as RSHash
from tests import score_sequences

_MODULUS = 2_147_483_647


def _reference_buckets(model: RSHash, normalized: np.ndarray) -> np.ndarray:
    """Hash like aberrant 1.4.0, one Python-integer sum per table."""
    assert model._subspace_indices is not None
    assert model._subspace_shifts is not None
    assert model._hash_a is not None and model._hash_b is not None
    sub_values = normalized[model._subspace_indices]
    shifted = (sub_values + model._subspace_shifts) / model.bin_width
    quantized = np.floor(shifted).astype(np.int64)
    buckets = np.zeros((model.components_num, model.hash_num), dtype=np.intp)
    for component in range(model.components_num):
        for table in range(model.hash_num):
            acc = int(model._hash_b[component, table])
            for coefficient, value in zip(
                model._hash_a[component, table].tolist(),
                quantized[component].tolist(),
                strict=True,
            ):
                acc = (acc + coefficient * (value % _MODULUS)) % _MODULUS
            buckets[component, table] = acc % model.bins
    return buckets


def _reference_score(model: RSHash, buckets: np.ndarray, query_scale: float) -> float:
    """Score like aberrant 1.4.0, one minimum and one clip per component."""
    assert model._counts is not None
    normalizer = math.log(1 + max(model.n_samples_seen, 1))
    component_scores = np.empty(model.components_num, dtype=np.float64)
    for component in range(model.components_num):
        occupancy_min = float("inf")
        for table in range(model.hash_num):
            bucket = int(buckets[component, table])
            occupancy = model._counts[component, table, bucket] * query_scale
            occupancy_min = min(occupancy_min, float(occupancy))
        score = 1.0 - (np.log1p(max(occupancy_min, 0.0)) / normalizer)
        component_scores[component] = np.clip(score, 0.0, 1.0)
    return float(np.mean(component_scores))


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

    def test_large_time_gap_forgets_counts_without_poisoning_state(self) -> None:
        model = self.create_model(warm_up_samples=1)
        model.learn_one({"t": 0.0, "a": 1.0, "b": 2.0})
        counts_before = model._counts.copy()
        query = {"t": 100_000.0, "a": 1.0, "b": 2.0}

        self.assertEqual(model.score_one(query), 1.0)
        np.testing.assert_array_equal(model._counts, counts_before)
        self.assertEqual(model._boundary.clock.max_time, 0.0)
        self.assertEqual(model._scale, 1.0)

        model.learn_one(query)
        np.testing.assert_array_equal(
            model._counts.sum(axis=2),
            np.ones((model.components_num, model.hash_num)),
        )
        self.assertEqual(model._scale, 1.0)
        self.assertEqual(model._boundary.clock.max_time, query["t"])
        model.learn_one({"t": 100_001.0, "a": 1.1, "b": 2.1})
        self.assertEqual(model.n_samples_seen, 3)

    def test_failed_hashing_does_not_apply_decay_or_advance_time(self) -> None:
        model = self.create_model()
        control = self.create_model()
        initial = {"t": 0.0, "a": 1.0, "b": 2.0}
        model.learn_one(initial)
        control.learn_one(initial)

        with (
            patch.object(model, "_bucket_indices", side_effect=RuntimeError("hash")),
            self.assertRaisesRegex(RuntimeError, "hash"),
        ):
            model.learn_one({"t": 10.0, "a": 1.5, "b": 2.5})

        self.assertEqual(model._scale, control._scale)
        self.assertEqual(model._boundary.clock.max_time, 0.0)
        next_point = {"t": 1.0, "a": 1.2, "b": 2.2}
        model.learn_one(next_point)
        control.learn_one(next_point)
        np.testing.assert_array_equal(model._counts, control._counts)
        self.assertEqual(model._scale, control._scale)

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


class TestRSHashExactness(unittest.TestCase):
    """Array hashing and scoring must reproduce the 1.4.0 buckets and scores."""

    CONFIGS: tuple[dict[str, object], ...] = (
        {"seed": 1},
        {"components_num": 5, "hash_num": 1, "bins": 1, "subspace_size": 1},
        {"components_num": 9, "hash_num": 6, "bins": 1_000, "subspace_size": 7},
        {"components_num": 30, "hash_num": 3, "bins": 17, "bin_width": 0.05},
    )

    def _learned(self, config: dict[str, object], n_features: int) -> RSHash:
        model = RSHash(**{"warm_up_samples": 1, "seed": 2, **config})
        events = score_sequences.make_stream(
            seed=3, n_events=200, n_features=n_features
        )
        for event in events:
            model.learn_one(event)
        return model

    def test_score_sequences_match_recorded_reference(self) -> None:
        for case in score_sequences.load("streaming_rs_hash"):
            with self.subTest(case=case["name"]):
                model = RSHash(**case["params"])
                scores = score_sequences.replay(model, case)
                # The tolerance only absorbs last-bit libm differences in exp
                # and log1p between platforms; the recording platform
                # reproduces every bit.
                np.testing.assert_allclose(scores, case["scores"], rtol=1e-12, atol=0)

    def test_buckets_match_reference_hash(self) -> None:
        rng = np.random.default_rng(4)
        for config in self.CONFIGS:
            with self.subTest(config=config):
                model = self._learned(config, n_features=12)
                # Quantized values span the whole int64 range, both signs and
                # zero, so residues and products reach their largest values.
                bound = 9.2e18 * float(model.bin_width)
                vectors = [
                    rng.normal(size=12),
                    rng.normal(size=12) * 1e15,
                    rng.uniform(-bound, bound, size=12),
                    np.zeros(12),
                    -np.arange(12.0),
                ]
                for normalized in vectors:
                    np.testing.assert_array_equal(
                        model._bucket_indices(normalized),
                        _reference_buckets(model, normalized),
                    )

    def test_scores_match_reference_component_loop(self) -> None:
        rng = np.random.default_rng(5)
        configs = (*self.CONFIGS, {"decay": 0.5}, {"decay": 0.0})
        for config in configs:
            with self.subTest(config=config):
                model = self._learned(config, n_features=12)
                for query_scale in (0.0, 1e-300, 0.37, model._scale, 1.0):
                    for _ in range(5):
                        buckets = rng.integers(
                            0, model.bins, size=(model.components_num, model.hash_num)
                        )
                        self.assertEqual(
                            model._score_buckets(buckets, query_scale),
                            _reference_score(model, buckets, query_scale),
                        )

    def test_learning_a_scored_point_reuses_its_buckets(self) -> None:
        events = score_sequences.make_stream(seed=6, n_events=60, n_features=4)
        model = RSHash(warm_up_samples=4, seed=7)
        reference = RSHash(warm_up_samples=4, seed=7)
        for event in events[:20]:
            model.learn_one(event)
            reference.learn_one(event)

        with patch.object(
            model, "_bucket_indices", wraps=model._bucket_indices
        ) as hashing:
            for event in events[20:40]:
                model.score_one(event)
                model.learn_one(event)
                reference.learn_one(event)
            self.assertEqual(hashing.call_count, 20)

            # Learning a scored point twice hashes it again with the updated
            # statistics, and scoring one point and learning another hashes
            # both.
            model.score_one(events[40])
            model.learn_one(events[40])
            model.learn_one(events[40])
            model.score_one(events[41])
            model.learn_one(events[42])
            for event in (events[40], events[40], events[42]):
                reference.learn_one(event)
            self.assertEqual(hashing.call_count, 24)

        np.testing.assert_array_equal(model._counts, reference._counts)
        self.assertEqual(model.score_one(events[43]), reference.score_one(events[43]))

        model.reset()
        fresh = RSHash(warm_up_samples=4, seed=7)
        for event in events[43:]:
            model.learn_one(event)
            fresh.learn_one(event)
        np.testing.assert_array_equal(model._counts, fresh._counts)

    def test_default_model_speed_smoke(self) -> None:
        events = score_sequences.make_stream(seed=8, n_events=2000, n_features=8)
        model = RSHash(seed=8)

        start = time.perf_counter()
        score_sequences.score_then_learn(model, events)

        # Generous bound against large regressions; typical runs take ~0.15 s.
        self.assertLess(time.perf_counter() - start, 2.0)


if __name__ == "__main__":
    unittest.main()
