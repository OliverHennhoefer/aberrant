"""Unit tests for the MStream sketch-based anomaly detector."""

import math
import time
import unittest

import numpy as np

from aberrant.model.sketch import MStream
from tests import score_sequences


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

    @staticmethod
    def author_statistic(total: float, current: float, time_index: int) -> float:
        """Evaluate the published raw statistic at ordinary magnitudes."""
        mean = total / time_index
        squared_error = max(0.0, current - mean) ** 2
        return squared_error / mean + squared_error / (mean * max(1, time_index - 1))

    def test_log_statistic_matches_author_formula(self) -> None:
        for total, current, time_index in (
            (8.0, 4.0, 5),
            (8.0, 1.0, 5),
            (8.0, 9.0, 1),
            (8.0, 8.0, 1),
        ):
            with self.subTest(total=total, current=current, time_index=time_index):
                statistic = self.author_statistic(total, current, time_index)
                expected = math.log(statistic) if statistic > 0.0 else -math.inf
                self.assertAlmostEqual(
                    MStream._counts_to_log_anomaly(total, current, time_index),
                    expected,
                )

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
            self.author_statistic(8.0, 3.0 * model.alpha + 1.0, 5)
            + self.author_statistic(9.0, 4.0 * model.alpha + 1.0, 5)
        )
        self.assertAlmostEqual(model.score_one(query), float(expected))

    def test_all_zero_statistics_return_zero_score(self) -> None:
        model = self.create_model()
        sample = {"t": 0, "x": 1.0, "category": 2.0}
        model.learn_one(sample)
        self.assertEqual(model.score_one(sample), 0.0)

    def test_huge_time_score_keeps_every_attribute_and_record_contribution(
        self,
    ) -> None:
        for numeric in ({}, {"x": 1.0, "y": 1.0}):
            with self.subTest(numeric=numeric):
                model = self.create_model()
                sample = {"t": 0, "category": 2.0, **numeric}
                model.learn_one(sample)
                expected = 400 * math.log(10.0) + math.log(
                    (len(numeric) + 2) * (model.alpha + 1.0) ** 2 / 2.0
                )
                query = {**sample, "t": 10**400}
                self.assertAlmostEqual(model.score_one(query), expected)

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


class TestMStreamExactness(unittest.TestCase):
    def test_score_sequences_match_recorded_reference(self) -> None:
        for case in score_sequences.load("m_stream"):
            with self.subTest(case=case["name"]):
                scores = score_sequences.replay(MStream(**case["params"]), case)
                # The tolerance only absorbs last-bit libm differences in log
                # between platforms; the recording platform reproduces every bit.
                np.testing.assert_allclose(scores, case["scores"], rtol=1e-12, atol=0)

    def test_integer_check_matches_allclose(self) -> None:
        model = MStream(buckets=16, categorical_features=("c",))
        offsets = [0.0, -0.0, 5e-10, -5e-10, 9.9e-10, 1e-9, -1e-9, 1.01e-9, 2e-9]
        for base in (0.0, 3.0, -7.0, 2.0**40, -(2.0**52), 1e15, 0.5):
            for offset in offsets:
                value = base + offset
                with self.subTest(value=value):
                    values = np.array([1.0, value])
                    expected = np.allclose(value, np.rint(value), rtol=0.0, atol=1e-9)
                    try:
                        model._split_values(values, np.array([0]), np.array([1]))
                    except ValueError:
                        accepted = False
                    else:
                        accepted = True
                    self.assertEqual(accepted, expected)

    def test_numeric_only_events_have_no_categorical_hashes(self) -> None:
        model = MStream()
        numeric, categorical = model._split_values(
            np.array([0.5, 2.0]), np.array([0, 1]), np.array([], dtype=np.intp)
        )
        np.testing.assert_array_equal(numeric, [0.5, 2.0])
        self.assertEqual(categorical.shape, (0,))
        self.assertEqual(categorical.dtype, np.int64)

    def test_learning_increments_one_cell_per_feature_and_row(self) -> None:
        model = MStream(
            rows=3, buckets=8, time_key="t", categorical_features=("a", "b"), seed=4
        )
        model.learn_one({"x": 1.0, "y": 0.0, "a": 2.0, "b": 2.0, "t": 0})
        model.learn_one({"x": 4.0, "y": 0.5, "a": 2.0, "b": 9.0, "t": 0})
        state = model._state
        assert state is not None
        for counts in (state.numeric.counts, state.categorical.counts):
            # Each learned event adds one count per feature (and row).
            self.assertEqual(counts.total.sum(axis=-1).max(), 2.0)
            self.assertEqual(counts.total.sum(), 2.0 * counts.total[..., 0].size)
            np.testing.assert_array_equal(counts.current, counts.total)
        categorical_total = state.categorical.counts.total
        # Both events hash category 2 of feature ``a`` to the same cells.
        self.assertEqual(categorical_total[0].max(), 2.0)

    def test_default_model_speed_smoke(self) -> None:
        events = score_sequences.make_multi_aspect_stream(
            seed=17, n_events=2000, n_numeric=8, n_categorical=0
        )
        model = MStream(seed=17)

        start = time.perf_counter()
        score_sequences.score_then_learn(model, events)

        # Generous bound against large regressions; typical runs take ~0.3 s.
        self.assertLess(time.perf_counter() - start, 4.0)


if __name__ == "__main__":
    unittest.main()
