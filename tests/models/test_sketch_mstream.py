"""Unit tests for the MStream sketch-based anomaly detector."""

import unittest

import numpy as np

from aberrant.model.sketch import MStream


class TestMStream(unittest.TestCase):
    """Test suite for the author-grounded MStream implementation."""

    def create_model(self, **overrides: object) -> MStream:
        defaults: dict[str, object] = {
            "rows": 2,
            "buckets": 128,
            "alpha": 0.7,
            "time_key": "t",
            "categorical_features": ("category",),
            "warm_up_buckets": 0,
            "seed": 42,
        }
        defaults.update(overrides)
        return MStream(**defaults)

    def test_initialization_defaults_match_author_parameters(self) -> None:
        model = MStream()
        self.assertEqual(model.rows, 2)
        self.assertEqual(model.buckets, 1024)
        self.assertEqual(model.alpha, 0.6)
        self.assertEqual(model.categorical_features, ())
        self.assertEqual(model.warm_up_buckets, 0)

    def test_invalid_parameters(self) -> None:
        with self.assertRaises(ValueError):
            MStream(rows=0)
        with self.assertRaises(ValueError):
            MStream(buckets=0)
        with self.assertRaises(ValueError):
            MStream(alpha=0.0)
        with self.assertRaises(ValueError):
            MStream(alpha=1.1)
        with self.assertRaises(ValueError):
            MStream(time_key="")
        with self.assertRaises(ValueError):
            MStream(categorical_features=("",))
        with self.assertRaises(ValueError):
            MStream(categorical_features=("a", "a"))
        with self.assertRaises(ValueError):
            MStream(time_key="t", categorical_features=("t",))
        with self.assertRaises(ValueError):
            MStream(warm_up_buckets=-1)

    def test_input_validation(self) -> None:
        model = self.create_model()
        with self.assertRaises(ValueError):
            model.learn_one({})
        with self.assertRaises(ValueError):
            model.learn_one({"t": 1.0, "x": "bad", "category": 1.0})  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            model.learn_one({"x": 1.0, "category": 1.0})
        with self.assertRaises(ValueError):
            model.learn_one({"t": float("nan"), "x": 1.0, "category": 1.0})
        with self.assertRaises(ValueError):
            model.learn_one({"t": 1.0, "x": -1.0, "category": 1.0})
        with self.assertRaises(ValueError):
            model.learn_one({"t": 1.0, "x": 1.0, "category": 1.5})

    def test_counts_to_anomaly_matches_author_formula(self) -> None:
        total = 8.0
        current = 4.0
        time_index = 5
        mean = total / time_index
        squared_error = max(0.0, current - mean) ** 2
        expected = squared_error / mean + squared_error / (mean * 4.0)
        self.assertAlmostEqual(
            MStream._counts_to_anomaly(total, current, time_index),
            expected,
        )
        self.assertEqual(MStream._counts_to_anomaly(8.0, 1.0, 5), 0.0)

    def test_numeric_transform_and_online_min_max_match_author_code(self) -> None:
        model = MStream(rows=1, buckets=11, time_key="t", seed=1)
        model.learn_one({"t": 1.0, "x": 0.0})
        model.learn_one({"t": 1.0, "x": 99.0})
        self.assertIsNotNone(model._state)
        assert model._state is not None

        normalized = model._normalize_numeric(
            model._state.numeric,
            np.asarray([9.0], dtype=np.float64),
            update=False,
        )
        expected = np.log10(10.0) / np.log10(100.0)
        self.assertAlmostEqual(float(normalized[0]), expected)

    def test_creates_individual_and_complete_record_sketches(self) -> None:
        model = self.create_model(rows=3, buckets=64)
        model.learn_one({"t": 1.0, "x": 1.0, "y": 2.0, "category": 3.0})
        self.assertIsNotNone(model._state)
        assert model._state is not None

        self.assertEqual(model._state.numeric.counts.current.shape, (2, 64))
        self.assertEqual(model._state.categorical.counts.current.shape, (1, 3, 64))
        self.assertEqual(model._state.record.counts.current.shape, (3, 64))

    def test_full_record_hash_uses_all_attributes(self) -> None:
        model = MStream(rows=1, buckets=4, time_key="t", seed=1)
        model.learn_one({"t": 1.0, "x": 1.0, "y": 1.0})
        self.assertIsNotNone(model._state)
        assert model._state is not None
        model._state.record.numeric_planes[:] = np.asarray(
            [[[1.0, -1.0], [-1.0, 1.0]]],
            dtype=np.float64,
        )

        first = model._record_bins(
            model._state.record,
            np.asarray([1.0, 0.0]),
            np.asarray([], dtype=np.int64),
        )
        second = model._record_bins(
            model._state.record,
            np.asarray([0.0, 1.0]),
            np.asarray([], dtype=np.int64),
        )
        self.assertNotEqual(int(first[0]), int(second[0]))

    def test_score_matches_sum_of_attribute_and_record_contributions(self) -> None:
        model = MStream(rows=1, buckets=128, time_key="t", seed=3)
        sample = {"t": 1.0, "x": 1.0}
        model.learn_one(sample)
        self.assertIsNotNone(model._state)
        assert model._state is not None
        state = model._state
        normalized = model._normalize_numeric(
            state.numeric,
            np.asarray([1.0], dtype=np.float64),
            update=False,
        )
        numeric_bin = int(model._numeric_bins(normalized)[0])
        record_bin = int(
            model._record_bins(
                state.record,
                normalized,
                np.asarray([], dtype=np.int64),
            )[0]
        )
        state.numeric.counts.current[0, numeric_bin] = 3.0
        state.numeric.counts.total[0, numeric_bin] = 7.0
        state.record.counts.current[0, record_bin] = 4.0
        state.record.counts.total[0, record_bin] = 8.0

        query = {"t": 5.0, "x": 1.0}
        expected = np.log1p(
            MStream._counts_to_anomaly(8.0, 3.0 * model.alpha + 1.0, 5)
            + MStream._counts_to_anomaly(9.0, 4.0 * model.alpha + 1.0, 5)
        )
        self.assertAlmostEqual(model.score_one(query), float(expected))

    def test_rollover_decays_current_counts_once_like_author_code(self) -> None:
        model = self.create_model(rows=1, buckets=1024)
        sample = {"t": 1.0, "x": 1.0, "category": 2.0}
        model.learn_one(sample)
        self.assertIsNotNone(model._state)
        assert model._state is not None
        current_sum = float(np.sum(model._state.record.counts.current))

        model.learn_one({"t": 100.0, "x": 2.0, "category": 3.0})
        expected = current_sum * model.alpha + 1.0
        self.assertAlmostEqual(
            float(np.sum(model._state.record.counts.current)),
            expected,
        )

    def test_score_one_does_not_advance_decay_or_normalize_state(self) -> None:
        model = self.create_model()
        model.learn_one({"t": 1.0, "x": 1.0, "category": 2.0})
        self.assertIsNotNone(model._state)
        assert model._state is not None
        state = model._state
        current_bucket = state.current_bucket
        numeric_min = state.numeric.minimum.copy()
        current = state.record.counts.current.copy()

        first = model.score_one({"t": 10.0, "x": 100.0, "category": 9.0})
        second = model.score_one({"t": 10.0, "x": 100.0, "category": 9.0})

        self.assertEqual(first, second)
        self.assertEqual(state.current_bucket, current_bucket)
        np.testing.assert_array_equal(state.numeric.minimum, numeric_min)
        np.testing.assert_array_equal(state.record.counts.current, current)

    def test_score_before_learning_does_not_initialize_schema_or_rng_state(
        self,
    ) -> None:
        model = self.create_model()
        rng_state = model._rng.bit_generator.state

        score = model.score_one({"t": 1.0, "x": 2.0, "category": 3.0})

        self.assertEqual(score, 0.0)
        self.assertIsNone(model._boundary.schema.names)
        self.assertIsNone(model._state)
        self.assertEqual(model._rng.bit_generator.state, rng_state)

    def test_score_is_zero_before_optional_warmup(self) -> None:
        model = self.create_model(warm_up_buckets=2)
        model.learn_one({"t": 1.0, "x": 1.0, "category": 2.0})
        self.assertEqual(
            model.score_one({"t": 2.0, "x": 1.0, "category": 2.0}),
            0.0,
        )

    def test_feature_schema_mismatch_raises(self) -> None:
        model = self.create_model()
        model.learn_one({"t": 1.0, "x": 1.0, "category": 2.0})
        with self.assertRaises(ValueError):
            model.learn_one({"t": 1.0, "y": 1.0, "category": 2.0})

    def test_missing_configured_categorical_feature_raises(self) -> None:
        model = self.create_model()
        with self.assertRaises(ValueError):
            model.learn_one({"t": 1.0, "x": 1.0})

    def test_non_monotonic_timestamp_raises(self) -> None:
        model = self.create_model()
        model.learn_one({"t": 2.0, "x": 1.0, "category": 2.0})
        with self.assertRaises(ValueError):
            model.score_one({"t": 1.0, "x": 1.0, "category": 2.0})

    def test_internal_clock_fallback_without_time_key(self) -> None:
        model = MStream(time_key=None, seed=3)
        sample = {"x": 1.0, "y": 2.0}
        model.learn_one(sample)
        model.learn_one(sample)
        self.assertIsNotNone(model._state)
        assert model._state is not None
        self.assertEqual(model._state.current_bucket, 2)

    def test_deterministic_with_seed(self) -> None:
        model1 = self.create_model(seed=7)
        model2 = self.create_model(seed=7)

        for i in range(60):
            point = {
                "t": float(i // 4),
                "x": float(i % 5),
                "category": float((i * 3) % 7),
            }
            model1.learn_one(point)
            model2.learn_one(point)

        query = {"t": 30.0, "x": 2.5, "category": 4.0}
        self.assertAlmostEqual(
            model1.score_one(query),
            model2.score_one(query),
            places=12,
        )

    def test_sketch_shapes_are_bounded(self) -> None:
        model = self.create_model(buckets=64)
        model.learn_one({"t": 0.0, "x": 1.0, "category": 3.0})
        self.assertIsNotNone(model._state)
        assert model._state is not None
        numeric_shape = model._state.numeric.counts.current.shape
        categorical_shape = model._state.categorical.counts.current.shape
        record_shape = model._state.record.counts.current.shape

        for i in range(1, 300):
            model.learn_one(
                {
                    "t": float(i // 3),
                    "x": float(i % 10),
                    "category": float((i * 2) % 11),
                }
            )

        self.assertEqual(model._state.numeric.counts.current.shape, numeric_shape)
        self.assertEqual(
            model._state.categorical.counts.current.shape,
            categorical_shape,
        )
        self.assertEqual(model._state.record.counts.current.shape, record_shape)

    def test_reset_restores_cold_state(self) -> None:
        model = self.create_model()
        model.learn_one({"t": 1.0, "x": 1.0, "category": 2.0})
        model.reset()
        self.assertEqual(model.n_samples_seen, 0)
        self.assertIsNone(model._boundary.schema.names)
        self.assertIsNone(model._state)

    def test_repr_contains_key_config(self) -> None:
        model = self.create_model(rows=3, buckets=256)
        output = repr(model)
        self.assertIn("MStream", output)
        self.assertIn("rows=3", output)
        self.assertIn("buckets=256", output)
        self.assertIn("categorical_features=('category',)", output)


if __name__ == "__main__":
    unittest.main()
