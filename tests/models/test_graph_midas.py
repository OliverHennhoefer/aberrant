"""Unit tests for the MIDAS graph-stream anomaly detector."""

import unittest

import numpy as np

from aberrant.model.graph import MIDAS


class TestMIDAS(unittest.TestCase):
    """Test suite for the author-grounded MIDAS and MIDAS-R implementations."""

    def create_model(self, **overrides: object) -> MIDAS:
        defaults: dict[str, object] = {
            "source_key": "src",
            "destination_key": "dst",
            "time_key": "t",
            "count_min_rows": 2,
            "count_min_cols": 257,
            "time_decay_factor": 0.5,
            "warm_up_samples": 0,
            "use_relational": True,
            "normalize_score": False,
            "seed": 42,
        }
        defaults.update(overrides)
        return MIDAS(**defaults)

    def test_initialization_defaults_match_author_parameters(self) -> None:
        model = MIDAS()
        self.assertEqual(model.source_key, "src")
        self.assertEqual(model.destination_key, "dst")
        self.assertEqual(model.time_key, "t")
        self.assertEqual(model.count_min_rows, 2)
        self.assertEqual(model.count_min_cols, 1024)
        self.assertEqual(model.time_decay_factor, 0.5)
        self.assertEqual(model.warm_up_samples, 0)
        self.assertTrue(model.use_relational)

    def test_invalid_parameters(self) -> None:
        with self.assertRaises(ValueError):
            MIDAS(source_key="")
        with self.assertRaises(ValueError):
            MIDAS(destination_key="")
        with self.assertRaises(ValueError):
            MIDAS(source_key="x", destination_key="x")
        with self.assertRaises(ValueError):
            MIDAS(time_key="")
        with self.assertRaises(ValueError):
            MIDAS(time_key="src")
        with self.assertRaises(ValueError):
            MIDAS(count_min_rows=0)
        with self.assertRaises(ValueError):
            MIDAS(count_min_cols=0)
        with self.assertRaises(ValueError):
            MIDAS(time_decay_factor=0.0)
        with self.assertRaises(ValueError):
            MIDAS(warm_up_samples=-1)

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

    def test_compute_score_matches_author_formula(self) -> None:
        current = 4.0
        total = 8.0
        timestamp = 5
        expected = ((current - total / timestamp) * timestamp) ** 2 / (
            total * (timestamp - 1)
        )
        self.assertAlmostEqual(
            MIDAS._compute_score(current, total, timestamp), expected
        )
        self.assertEqual(MIDAS._compute_score(1.0, 1.0, 1), 0.0)

    def test_current_and_total_sketches_share_hash_functions(self) -> None:
        model = self.create_model()
        payload = model._edge_payload(1, 2)
        np.testing.assert_array_equal(
            model._edge_current.indices(payload),
            model._edge_total.indices(payload),
        )

    def test_normal_core_clears_current_counts_on_new_timestamp(self) -> None:
        model = self.create_model(use_relational=False)
        sample = {"src": 1.0, "dst": 2.0, "t": 1.0}
        model.learn_one(sample)
        model.learn_one(sample)
        self.assertEqual(float(np.sum(model._edge_current.table)), 4.0)

        model.learn_one({"src": 3.0, "dst": 4.0, "t": 2.0})
        self.assertEqual(float(np.sum(model._edge_current.table)), 2.0)
        self.assertEqual(float(np.sum(model._edge_total.table)), 6.0)

    def test_relational_core_decays_current_counts_on_new_timestamp(self) -> None:
        model = self.create_model(use_relational=True, time_decay_factor=0.5)
        sample = {"src": 1.0, "dst": 2.0, "t": 1.0}
        model.learn_one(sample)
        model.learn_one(sample)
        model.learn_one({"src": 3.0, "dst": 4.0, "t": 100.0})

        self.assertEqual(float(np.sum(model._edge_current.table)), 4.0)
        self.assertEqual(float(np.sum(model._edge_total.table)), 6.0)

    def test_relational_score_is_componentwise_maximum(self) -> None:
        model = self.create_model(
            count_min_rows=1,
            count_min_cols=10_007,
            use_relational=True,
        )
        for timestamp in range(1, 8):
            for _ in range(timestamp):
                model.learn_one({"src": 1.0, "dst": 2.0, "t": float(timestamp)})

        event = model._boundary.preview({"src": 1.0, "dst": 9.0, "t": 8.0})
        bucket, src, dst = event.bucket, event.source, event.destination
        rollover = bucket > model._current_bucket  # type: ignore[operator]
        time_index = model._time_index(bucket)
        edge = model._candidate_score(
            model._edge_current,
            model._edge_total,
            model._edge_payload(src, dst),
            rollover=rollover,
            time_index=time_index,
        )
        source = model._candidate_score(
            model._source_current,  # type: ignore[arg-type]
            model._source_total,  # type: ignore[arg-type]
            model._source_payload(src),
            rollover=rollover,
            time_index=time_index,
        )
        destination = model._candidate_score(
            model._destination_current,  # type: ignore[arg-type]
            model._destination_total,  # type: ignore[arg-type]
            model._destination_payload(dst),
            rollover=rollover,
            time_index=time_index,
        )
        self.assertEqual(
            model.score_one({"src": 1.0, "dst": 9.0, "t": 8.0}),
            max(edge, source, destination),
        )

    def test_score_one_does_not_advance_or_decay_state(self) -> None:
        model = self.create_model()
        for _ in range(8):
            model.learn_one({"src": 1.0, "dst": 2.0, "t": 1.0})
        bucket = model._current_bucket
        edge_current = model._edge_current.table.copy()
        source_current = model._source_current.table.copy()  # type: ignore[union-attr]

        first = model.score_one({"src": 1.0, "dst": 2.0, "t": 10.0})
        second = model.score_one({"src": 1.0, "dst": 2.0, "t": 10.0})
        self.assertEqual(first, second)
        self.assertEqual(model._current_bucket, bucket)
        np.testing.assert_array_equal(model._edge_current.table, edge_current)
        np.testing.assert_array_equal(
            model._source_current.table,  # type: ignore[union-attr]
            source_current,
        )

    def test_relational_toggle_disables_node_sketches(self) -> None:
        model = self.create_model(use_relational=False)
        self.assertIsNone(model._source_current)
        self.assertIsNone(model._source_total)
        self.assertIsNone(model._destination_current)
        self.assertIsNone(model._destination_total)

    def test_score_is_zero_before_optional_warmup(self) -> None:
        model = self.create_model(warm_up_samples=4)
        for _ in range(3):
            model.learn_one({"src": 1.0, "dst": 2.0, "t": 1.0})
        self.assertEqual(
            model.score_one({"src": 1.0, "dst": 2.0, "t": 1.0}),
            0.0,
        )

    def test_non_monotonic_timestamp_raises(self) -> None:
        model = self.create_model()
        model.learn_one({"src": 1.0, "dst": 2.0, "t": 2.0})
        with self.assertRaises(ValueError):
            model.score_one({"src": 1.0, "dst": 2.0, "t": 1.0})

    def test_internal_clock_fallback_without_time_key(self) -> None:
        model = self.create_model(time_key=None)
        model.learn_one({"src": 1.0, "dst": 2.0})
        model.learn_one({"src": 1.0, "dst": 2.0})
        self.assertEqual(model._current_bucket, 2)

    def test_deterministic_with_seed(self) -> None:
        model1 = self.create_model(seed=7)
        model2 = self.create_model(seed=7)
        for i in range(240):
            point = {
                "src": float((i * 3) % 23),
                "dst": float((i * 5) % 29),
                "t": float(i // 8),
            }
            model1.learn_one(point)
            model2.learn_one(point)

        query = {"src": 3.0, "dst": 17.0, "t": 31.0}
        self.assertAlmostEqual(
            model1.score_one(query), model2.score_one(query), places=12
        )

    def test_state_shapes_are_bounded(self) -> None:
        model = self.create_model(count_min_rows=5, count_min_cols=128)
        shape = model._edge_current.table.shape
        for i in range(500):
            model.learn_one(
                {
                    "src": float(i % 37),
                    "dst": float((i * 2) % 41),
                    "t": float(i // 4),
                }
            )
        self.assertEqual(model._edge_current.table.shape, shape)
        self.assertEqual(model._edge_total.table.shape, shape)

    def test_reset_restores_cold_state(self) -> None:
        model = self.create_model()
        model.learn_one({"src": 1.0, "dst": 2.0, "t": 1.0})
        model.reset()
        self.assertEqual(model.n_samples_seen, 0)
        self.assertIsNone(model._current_bucket)

    def test_normalize_score_bounds_output(self) -> None:
        model = self.create_model(normalize_score=True)
        for timestamp in range(1, 8):
            model.learn_one({"src": 1.0, "dst": 2.0, "t": float(timestamp)})
        score = model.score_one({"src": 1.0, "dst": 2.0, "t": 8.0})
        self.assertGreaterEqual(score, 0.0)
        self.assertLess(score, 1.0)

    def test_repr_contains_key_config(self) -> None:
        model = self.create_model(count_min_rows=9, count_min_cols=257)
        output = repr(model)
        self.assertIn("MIDAS", output)
        self.assertIn("count_min_rows=9", output)
        self.assertIn("count_min_cols=257", output)
        self.assertIn("time_decay_factor=0.5", output)


if __name__ == "__main__":
    unittest.main()
