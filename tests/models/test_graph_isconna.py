"""Unit tests for the ISCONNA graph-stream anomaly detector."""

import unittest

import numpy as np

from aberrant.model.graph import ISCONNA


class TestISCONNA(unittest.TestCase):
    """Test suite for the author-grounded ISCONNA implementation."""

    def create_model(self, **overrides: object) -> ISCONNA:
        defaults: dict[str, object] = {
            "source_key": "src",
            "destination_key": "dst",
            "time_key": "t",
            "count_min_rows": 2,
            "count_min_cols": 257,
            "time_decay_factor": 0.7,
            "alpha": 1.0,
            "beta": 1.0,
            "gamma": 0.5,
            "include_endpoints": True,
            "warm_up_samples": 0,
            "normalize_score": False,
            "seed": 42,
        }
        defaults.update(overrides)
        return ISCONNA(**defaults)

    def test_initialization_defaults_match_author_demo(self) -> None:
        model = ISCONNA()
        self.assertEqual(model.source_key, "src")
        self.assertEqual(model.destination_key, "dst")
        self.assertEqual(model.time_key, "t")
        self.assertEqual(model.count_min_rows, 2)
        self.assertEqual(model.count_min_cols, 3000)
        self.assertEqual(model.time_decay_factor, 0.7)
        self.assertEqual(model.alpha, 1.0)
        self.assertEqual(model.beta, 1.0)
        self.assertEqual(model.gamma, 0.5)
        self.assertTrue(model.include_endpoints)
        self.assertEqual(model.warm_up_samples, 0)

    def test_invalid_parameters(self) -> None:
        with self.assertRaises(ValueError):
            ISCONNA(source_key="")
        with self.assertRaises(ValueError):
            ISCONNA(destination_key="")
        with self.assertRaises(ValueError):
            ISCONNA(source_key="x", destination_key="x")
        with self.assertRaises(ValueError):
            ISCONNA(time_key="")
        with self.assertRaises(ValueError):
            ISCONNA(time_key="src")
        with self.assertRaises(ValueError):
            ISCONNA(count_min_rows=0)
        with self.assertRaises(ValueError):
            ISCONNA(count_min_cols=0)
        with self.assertRaises(ValueError):
            ISCONNA(time_decay_factor=0.0)
        with self.assertRaises(ValueError):
            ISCONNA(alpha=-1.0)
        with self.assertRaises(ValueError):
            ISCONNA(beta=-1.0)
        with self.assertRaises(ValueError):
            ISCONNA(gamma=-1.0)
        with self.assertRaises(ValueError):
            ISCONNA(warm_up_samples=-1)

    def test_input_validation(self) -> None:
        model = self.create_model()
        with self.assertRaises(ValueError):
            model.learn_one({})
        with self.assertRaises(ValueError):
            model.learn_one({"dst": 1.0, "t": 1.0})
        with self.assertRaises(ValueError):
            model.learn_one({"src": 1.0, "t": 1.0})
        with self.assertRaises(ValueError):
            model.learn_one({"src": 1.0, "dst": 2.0})
        with self.assertRaises(ValueError):
            model.learn_one({"src": "bad", "dst": 2.0, "t": 1.0})  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            model.learn_one({"src": 1.5, "dst": 2.0, "t": 1.0})
        with self.assertRaises(ValueError):
            model.score_one({"src": 1.0, "dst": 2.0, "t": float("inf")})

    def test_g_test_matches_author_formula(self) -> None:
        self.assertEqual(ISCONNA._g_test(0.0, 4.0, 3), 0.0)
        self.assertEqual(ISCONNA._g_test(2.0, 4.0, 1), 0.0)
        expected = 2.0 * 3.0 * abs(np.log(3.0 * 4.0 / 7.0))
        self.assertAlmostEqual(ISCONNA._g_test(3.0, 7.0, 5), expected)

    def test_component_preview_matches_author_update_formula(self) -> None:
        model = self.create_model(
            count_min_rows=1,
            count_min_cols=101,
            include_endpoints=False,
        )
        model.learn_one({"src": 1.0, "dst": 2.0, "t": 1.0})
        index = int(model._indices(7, 8)[0])
        group = model._edge

        group.busy_current[0, index] = True
        group.frequency_current[0, index] = 3.0
        group.frequency_accumulated[0, index] = 7.0
        group.width_current[0, index] = 2.0
        group.width_accumulated[0, index] = 2.0
        group.width_time[0, index] = 3
        group.gap_current[0, index] = 2.0
        group.gap_accumulated[0, index] = 3.0
        group.gap_time[0, index] = 4

        frequency, width, gap = model._component_scores(5, 7, 8)
        self.assertAlmostEqual(frequency, ISCONNA._g_test(3.1, 8.0, 5))
        self.assertAlmostEqual(width, ISCONNA._g_test(3.0, 2.0, 3))
        self.assertAlmostEqual(gap, ISCONNA._g_test(2.0, 3.0, 4))

    def test_reset_tracks_consecutive_absence_gap(self) -> None:
        model = self.create_model(
            count_min_rows=1,
            count_min_cols=10_007,
            include_endpoints=False,
        )
        target = {"src": 1.0, "dst": 2.0, "t": 1.0}
        target_index = int(model._indices(1, 2)[0])
        model.learn_one(target)
        model.learn_one({"src": 3.0, "dst": 4.0, "t": 2.0})
        model.learn_one({"src": 3.0, "dst": 4.0, "t": 3.0})
        model.learn_one({"src": 3.0, "dst": 4.0, "t": 4.0})

        self.assertEqual(model._edge.gap_time[0, target_index], 2)
        self.assertEqual(model._edge.gap_current[0, target_index], 2.0)
        self.assertFalse(model._edge.busy_current[0, target_index])

    def test_edge_node_aggregation_uses_componentwise_maxima(self) -> None:
        model = self.create_model(count_min_rows=1, count_min_cols=10_007)
        for timestamp in range(1, 8):
            model.learn_one(
                {"src": 1.0, "dst": float(timestamp + 10), "t": float(timestamp)}
            )

        bucket, src, dst = model._prepare_sample({"src": 1.0, "dst": 99.0, "t": 8.0})
        edge = model._preview_group(
            model._edge,
            model._indices(src, dst),
            rollover=True,
            time_index=model._time_index(bucket),
        )
        self.assertIsNotNone(model._source)
        self.assertIsNotNone(model._destination)
        source = model._preview_group(
            model._source,  # type: ignore[arg-type]
            model._indices(src, 0),
            rollover=True,
            time_index=model._time_index(bucket),
        )
        destination = model._preview_group(
            model._destination,  # type: ignore[arg-type]
            model._indices(dst, 0),
            rollover=True,
            time_index=model._time_index(bucket),
        )
        expected = tuple(
            max(values) for values in zip(edge, source, destination, strict=True)
        )
        self.assertEqual(model._component_scores(bucket, src, dst), expected)

    def test_score_one_does_not_advance_or_decay_state(self) -> None:
        model = self.create_model(include_endpoints=False)
        for timestamp in range(1, 8):
            model.learn_one({"src": 1.0, "dst": 2.0, "t": float(timestamp)})

        current_bucket = model._current_bucket
        frequency_before = model._edge.frequency_current.copy()
        gap_before = model._edge.gap_current.copy()

        first = model.score_one({"src": 1.0, "dst": 2.0, "t": 20.0})
        second = model.score_one({"src": 1.0, "dst": 2.0, "t": 20.0})

        self.assertEqual(first, second)
        self.assertEqual(model._current_bucket, current_bucket)
        np.testing.assert_array_equal(model._edge.frequency_current, frequency_before)
        np.testing.assert_array_equal(model._edge.gap_current, gap_before)

    def test_score_is_zero_before_optional_warmup(self) -> None:
        model = self.create_model(warm_up_samples=4)
        for timestamp in range(1, 4):
            model.learn_one({"src": 1.0, "dst": 2.0, "t": float(timestamp)})
        self.assertEqual(
            model.score_one({"src": 1.0, "dst": 2.0, "t": 4.0}),
            0.0,
        )

    def test_internal_clock_fallback_without_time_key(self) -> None:
        model = self.create_model(time_key=None, include_endpoints=False)
        sample = {"src": 1.0, "dst": 2.0}

        self.assertIsInstance(model.score_one(sample), float)
        model.learn_one(sample)
        model.learn_one(sample)
        self.assertEqual(model._current_bucket, 2)

    def test_non_monotonic_timestamp_raises(self) -> None:
        model = self.create_model()
        model.learn_one({"src": 1.0, "dst": 2.0, "t": 2.0})
        with self.assertRaises(ValueError):
            model.score_one({"src": 1.0, "dst": 2.0, "t": 1.0})

    def test_deterministic_with_seed(self) -> None:
        model1 = self.create_model(seed=7)
        model2 = self.create_model(seed=7)

        for i in range(200):
            point = {
                "src": float((i * 3) % 23),
                "dst": float((i * 5) % 29),
                "t": float(i + 1),
            }
            model1.learn_one(point)
            model2.learn_one(point)

        query = {"src": 3.0, "dst": 17.0, "t": 201.0}
        self.assertAlmostEqual(
            model1.score_one(query), model2.score_one(query), places=12
        )

    def test_state_shapes_are_bounded(self) -> None:
        model = self.create_model(count_min_rows=3, count_min_cols=128)
        baseline_shape = model._edge.frequency_current.shape

        for i in range(500):
            model.learn_one(
                {
                    "src": float(i % 37),
                    "dst": float((i * 2) % 41),
                    "t": float(i + 1),
                }
            )

        self.assertEqual(model._edge.frequency_current.shape, baseline_shape)
        self.assertEqual(model._edge.width_current.shape, baseline_shape)
        self.assertEqual(model._edge.gap_current.shape, baseline_shape)

    def test_reset_restores_cold_state(self) -> None:
        model = self.create_model()
        model.learn_one({"src": 1.0, "dst": 2.0, "t": 1.0})
        model.reset()
        self.assertEqual(model.n_samples_seen, 0)
        self.assertIsNone(model._current_bucket)
        self.assertFalse(np.any(model._edge.busy_current))

    def test_normalize_score_bounds_output(self) -> None:
        model = self.create_model(normalize_score=True, include_endpoints=False)
        for i in range(1, 20):
            model.learn_one({"src": 1.0, "dst": 2.0, "t": float(i)})
        score = model.score_one({"src": 1.0, "dst": 2.0, "t": 20.0})
        self.assertGreaterEqual(score, 0.0)
        self.assertLess(score, 1.0)

    def test_repr_contains_key_config(self) -> None:
        model = self.create_model(count_min_rows=9, count_min_cols=257)
        output = repr(model)
        self.assertIn("ISCONNA", output)
        self.assertIn("count_min_rows=9", output)
        self.assertIn("count_min_cols=257", output)
        self.assertIn("gamma=0.5", output)


if __name__ == "__main__":
    unittest.main()
