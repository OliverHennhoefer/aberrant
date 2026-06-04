"""Unit tests for the AnoEdge-L graph-stream anomaly detector."""

import unittest

import numpy as np

from aberrant.model.graph import AnoEdgeL
from aberrant.model.graph.anoedge import _DenseSubmatrix


class TestDenseSubmatrix(unittest.TestCase):
    """Directly verify the authors' local dense-submatrix operations."""

    def test_adds_rows_and_columns_only_when_density_improves(self) -> None:
        matrix = np.asarray([[5.0, 5.0, 0.0], [5.0, 5.0, 0.0], [0.0, 0.0, 1.0]])

        def value(row: int, col: int) -> float:
            return float(matrix[row, col])

        submatrix = _DenseSubmatrix(0, 0, 5.0)

        self.assertTrue(submatrix.check_and_add(1, 1, value))
        self.assertEqual(set(submatrix.row_sums), {0, 1})
        self.assertEqual(set(submatrix.col_sums), {0, 1})
        self.assertEqual(submatrix.total, 20.0)
        self.assertFalse(submatrix.check_and_add(2, 2, value))
        self.assertEqual(set(submatrix.row_sums), {0, 1})
        self.assertEqual(set(submatrix.col_sums), {0, 1})

    def test_deletes_weakest_dimension_when_density_improves(self) -> None:
        matrix = np.asarray([[5.0, 5.0], [5.0, 5.0], [0.5, 0.5]])

        def value(row: int, col: int) -> float:
            return float(matrix[row, col])

        submatrix = _DenseSubmatrix(0, 0)
        submatrix.total = 21.0
        submatrix.row_sums = {0: 10.0, 1: 10.0, 2: 1.0}
        submatrix.col_sums = {0: 10.5, 1: 10.5}

        self.assertTrue(submatrix.check_and_delete(value))
        self.assertEqual(set(submatrix.row_sums), {0, 1})
        self.assertEqual(submatrix.total, 20.0)

    def test_likelihood_matches_author_formula(self) -> None:
        matrix = np.asarray([[5.0, 5.0], [5.0, 5.0]])

        def value(row: int, col: int) -> float:
            return float(matrix[row, col])

        submatrix = _DenseSubmatrix(0, 0)
        submatrix.total = 20.0
        submatrix.row_sums = {0: 10.0, 1: 10.0}
        submatrix.col_sums = {0: 10.0, 1: 10.0}

        self.assertEqual(submatrix.likelihood(1, 1, value), 5.0)


class TestAnoEdgeL(unittest.TestCase):
    """Test suite for the author-grounded AnoEdge-L detector."""

    def create_model(self, **overrides: object) -> AnoEdgeL:
        defaults: dict[str, object] = {
            "source_key": "src",
            "destination_key": "dst",
            "time_key": "t",
            "count_min_rows": 64,
            "count_min_cols": 64,
            "num_hashes": 4,
            "num_dense_submatrices": 1,
            "time_decay_factor": 1.0,
            "warm_up_samples": 0,
            "normalize_score": False,
            "predict_threshold": 0.5,
            "seed": 42,
        }
        defaults.update(overrides)
        return AnoEdgeL(**defaults)

    def test_initialization_defaults(self) -> None:
        model = AnoEdgeL()
        self.assertEqual(model.source_key, "src")
        self.assertEqual(model.destination_key, "dst")
        self.assertEqual(model.time_key, "t")
        self.assertEqual(model.count_min_rows, 256)
        self.assertEqual(model.count_min_cols, 256)
        self.assertEqual(model.num_hashes, 4)
        self.assertEqual(model.num_dense_submatrices, 1)
        self.assertEqual(model.time_decay_factor, 1.0)
        self.assertEqual(model.warm_up_samples, 0)

    def test_invalid_parameters(self) -> None:
        with self.assertRaises(ValueError):
            AnoEdgeL(source_key="")
        with self.assertRaises(ValueError):
            AnoEdgeL(destination_key="")
        with self.assertRaises(ValueError):
            AnoEdgeL(source_key="x", destination_key="x")
        with self.assertRaises(ValueError):
            AnoEdgeL(time_key="")
        with self.assertRaises(ValueError):
            AnoEdgeL(time_key="src")
        with self.assertRaises(ValueError):
            AnoEdgeL(count_min_rows=0)
        with self.assertRaises(ValueError):
            AnoEdgeL(count_min_cols=0)
        with self.assertRaises(ValueError):
            AnoEdgeL(num_hashes=0)
        with self.assertRaises(ValueError):
            AnoEdgeL(num_dense_submatrices=0)
        with self.assertRaises(ValueError):
            AnoEdgeL(
                count_min_rows=2,
                count_min_cols=2,
                num_dense_submatrices=3,
            )
        with self.assertRaises(ValueError):
            AnoEdgeL(time_decay_factor=0.0)
        with self.assertRaises(ValueError):
            AnoEdgeL(warm_up_samples=-1)
        with self.assertRaises(ValueError):
            AnoEdgeL(normalize_score=True, predict_threshold=1.1)
        with self.assertRaises(ValueError):
            AnoEdgeL(normalize_score=False, predict_threshold=-0.1)

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

    def test_dense_block_edge_scores_above_sparse_edge(self) -> None:
        model = self.create_model(
            count_min_rows=32,
            count_min_cols=32,
            num_hashes=1,
            seed=1,
        )
        model._hash_a[:] = 1
        model._hash_b[:] = 0
        for _ in range(20):
            for src in (0.0, 1.0):
                for dst in (0.0, 1.0):
                    model.learn_one({"src": src, "dst": dst, "t": 1.0})

        dense_score = model.score_one({"src": 0.0, "dst": 1.0, "t": 1.0})
        sparse_score = model.score_one({"src": 10.0, "dst": 11.0, "t": 1.0})
        self.assertGreater(dense_score, sparse_score)

    def test_score_one_does_not_mutate_or_advance_state(self) -> None:
        model = self.create_model(num_hashes=1, time_decay_factor=0.5)
        for _ in range(10):
            model.learn_one({"src": 1.0, "dst": 2.0, "t": 1.0})

        bucket = model._current_bucket
        sketch = model._sketch.copy()
        submatrix = model._dense_submatrices[0][0].copy()
        first = model.score_one({"src": 1.0, "dst": 2.0, "t": 5.0})
        second = model.score_one({"src": 1.0, "dst": 2.0, "t": 5.0})

        self.assertEqual(first, second)
        self.assertEqual(model._current_bucket, bucket)
        np.testing.assert_array_equal(model._sketch, sketch)
        self.assertEqual(model._dense_submatrices[0][0].total, submatrix.total)
        self.assertEqual(
            model._dense_submatrices[0][0].row_sums,
            submatrix.row_sums,
        )

    def test_rollover_decays_once_on_bucket_jump(self) -> None:
        model = self.create_model(num_hashes=1, time_decay_factor=0.5)
        model.learn_one({"src": 1.0, "dst": 2.0, "t": 1.0})
        mass_before = float(np.sum(model._sketch))
        model.learn_one({"src": 3.0, "dst": 4.0, "t": 100.0})
        self.assertAlmostEqual(float(np.sum(model._sketch)), mass_before * 0.5 + 1.0)

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
        for i in range(200):
            point = {
                "src": float((i * 3) % 31),
                "dst": float((i * 5) % 29),
                "t": float(i // 4),
            }
            model1.learn_one(point)
            model2.learn_one(point)

        query = {"src": 3.0, "dst": 17.0, "t": 50.0}
        self.assertAlmostEqual(
            model1.score_one(query),
            model2.score_one(query),
            places=12,
        )

    def test_state_shapes_are_bounded(self) -> None:
        model = self.create_model(
            count_min_rows=12,
            count_min_cols=13,
            num_hashes=5,
            num_dense_submatrices=2,
        )
        sketch_shape = model._sketch.shape
        for i in range(500):
            model.learn_one(
                {
                    "src": float(i % 101),
                    "dst": float((i * 7) % 103),
                    "t": float(i // 5),
                }
            )

        self.assertEqual(model._sketch.shape, sketch_shape)
        self.assertEqual(len(model._dense_submatrices), 5)
        self.assertTrue(
            all(len(submatrices) == 2 for submatrices in model._dense_submatrices)
        )

    def test_reset_restores_cold_state(self) -> None:
        model = self.create_model()
        model.learn_one({"src": 1.0, "dst": 2.0, "t": 1.0})
        model.reset()
        self.assertEqual(model.n_samples_seen, 0)
        self.assertIsNone(model._current_bucket)
        self.assertEqual(float(np.sum(model._sketch)), 0.0)

    def test_normalize_score_bounds_output(self) -> None:
        model = self.create_model(normalize_score=True)
        for _ in range(20):
            model.learn_one({"src": 1.0, "dst": 2.0, "t": 1.0})
        score = model.score_one({"src": 1.0, "dst": 2.0, "t": 1.0})
        self.assertGreaterEqual(score, 0.0)
        self.assertLess(score, 1.0)

    def test_predict_one_is_binary(self) -> None:
        model = self.create_model(normalize_score=True, predict_threshold=0.2)
        prediction = model.predict_one({"src": 1.0, "dst": 2.0, "t": 1.0})
        self.assertIn(prediction, (0, 1))

    def test_repr_contains_key_config(self) -> None:
        model = self.create_model(
            count_min_rows=15,
            count_min_cols=17,
            num_hashes=6,
            num_dense_submatrices=2,
        )
        output = repr(model)
        self.assertIn("AnoEdgeL", output)
        self.assertIn("count_min_rows=15", output)
        self.assertIn("count_min_cols=17", output)
        self.assertIn("num_dense_submatrices=2", output)


if __name__ == "__main__":
    unittest.main()
