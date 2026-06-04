"""Unit tests for the StreamSpot graph-stream anomaly detector."""

import unittest

from aberrant.model.graph import StreamSpot


class TestStreamSpot(unittest.TestCase):
    """Test suite for StreamSpot."""

    def create_model(self, **overrides: object) -> StreamSpot:
        defaults: dict[str, object] = {
            "graph_key": "graph",
            "source_key": "src",
            "destination_key": "dst",
            "edge_type_key": "etype",
            "time_key": "t",
            "sketch_dim": 64,
            "shingle_size": 2,
            "num_clusters": 4,
            "max_graphs": 32,
            "warm_up_graphs": 4,
            "normalize_score": False,
            "predict_threshold": 0.5,
            "seed": 42,
            "eps": 1e-9,
        }
        defaults.update(overrides)
        return StreamSpot(**defaults)

    def test_initialization_defaults(self) -> None:
        model = StreamSpot()
        self.assertEqual(model.graph_key, "graph")
        self.assertEqual(model.source_key, "src")
        self.assertEqual(model.destination_key, "dst")
        self.assertIsNone(model.edge_type_key)
        self.assertEqual(model.time_key, "t")
        self.assertEqual(model.sketch_dim, 1024)
        self.assertEqual(model.shingle_size, 2)
        self.assertEqual(model.num_clusters, 8)
        self.assertEqual(model.max_graphs, 4096)
        self.assertEqual(model.warm_up_graphs, 32)
        self.assertFalse(model.normalize_score)

    def test_invalid_parameters(self) -> None:
        with self.assertRaises(ValueError):
            StreamSpot(graph_key="")
        with self.assertRaises(ValueError):
            StreamSpot(source_key="")
        with self.assertRaises(ValueError):
            StreamSpot(destination_key="")
        with self.assertRaises(ValueError):
            StreamSpot(edge_type_key="")
        with self.assertRaises(ValueError):
            StreamSpot(time_key="")
        with self.assertRaises(ValueError):
            StreamSpot(graph_key="x", source_key="x")
        with self.assertRaises(ValueError):
            StreamSpot(sketch_dim=0)
        with self.assertRaises(ValueError):
            StreamSpot(shingle_size=0)
        with self.assertRaises(ValueError):
            StreamSpot(num_clusters=0)
        with self.assertRaises(ValueError):
            StreamSpot(max_graphs=0)
        with self.assertRaises(ValueError):
            StreamSpot(num_clusters=5, max_graphs=4)
        with self.assertRaises(ValueError):
            StreamSpot(warm_up_graphs=0)
        with self.assertRaises(ValueError):
            StreamSpot(warm_up_graphs=5, max_graphs=4)
        with self.assertRaises(ValueError):
            StreamSpot(normalize_score=True, predict_threshold=-0.1)
        with self.assertRaises(ValueError):
            StreamSpot(normalize_score=True, predict_threshold=1.1)
        with self.assertRaises(ValueError):
            StreamSpot(normalize_score=False, predict_threshold=-0.1)
        with self.assertRaises(ValueError):
            StreamSpot(eps=0.0)

    def test_input_validation(self) -> None:
        model = self.create_model()
        with self.assertRaises(ValueError):
            model.learn_one({})
        with self.assertRaises(ValueError):
            model.learn_one({"src": 1.0, "dst": 2.0, "etype": 0.0, "t": 1.0})
        with self.assertRaises(ValueError):
            model.learn_one({"graph": 1.0, "dst": 2.0, "etype": 0.0, "t": 1.0})
        with self.assertRaises(ValueError):
            model.learn_one({"graph": 1.0, "src": 1.0, "etype": 0.0, "t": 1.0})
        with self.assertRaises(ValueError):
            model.learn_one({"graph": 1.0, "src": 1.0, "dst": 2.0, "t": 1.0})
        with self.assertRaises(ValueError):
            model.learn_one(
                {"graph": "bad", "src": 1.0, "dst": 2.0, "etype": 0.0, "t": 1.0}
            )  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            model.score_one(
                {"graph": 1.0, "src": 1.0, "dst": 2.0, "etype": 0.0, "t": float("inf")}
            )
        with self.assertRaises(ValueError):
            model.score_one({"graph": 1.0, "src": 1.0, "dst": 2.0, "etype": 0.0, "t": 1.5})

    def test_score_is_zero_before_warmup(self) -> None:
        model = self.create_model(warm_up_graphs=6, shingle_size=1, edge_type_key=None)
        for i in range(20):
            model.learn_one(
                {
                    "graph": float(i % 3),
                    "src": float(i % 5),
                    "dst": float((i + 1) % 5),
                    "t": float(i),
                }
            )
        self.assertEqual(
            model.score_one({"graph": 1.0, "src": 1.0, "dst": 2.0, "t": 21.0}),
            0.0,
        )

    def test_non_monotonic_timestamp_raises(self) -> None:
        model = self.create_model(edge_type_key=None, shingle_size=1, warm_up_graphs=1)
        model.learn_one({"graph": 1.0, "src": 1.0, "dst": 2.0, "t": 1.0})
        model.learn_one({"graph": 1.0, "src": 1.0, "dst": 2.0, "t": 2.0})
        with self.assertRaises(ValueError):
            model.score_one({"graph": 1.0, "src": 1.0, "dst": 2.0, "t": 1.0})
        with self.assertRaises(ValueError):
            model.learn_one({"graph": 1.0, "src": 1.0, "dst": 2.0, "t": 1.0})

    def test_score_one_does_not_advance_timestamp(self) -> None:
        model = self.create_model(edge_type_key=None, shingle_size=1, warm_up_graphs=1)
        model.learn_one({"graph": 1.0, "src": 1.0, "dst": 2.0, "t": 1.0})
        current_bucket = model._current_bucket

        model.score_one({"graph": 1.0, "src": 1.0, "dst": 2.0, "t": 10.0})

        self.assertEqual(model._current_bucket, current_bucket)

    def test_internal_clock_fallback_without_time_key(self) -> None:
        model = self.create_model(
            time_key=None,
            edge_type_key=None,
            shingle_size=1,
            warm_up_graphs=2,
        )
        sample = {"graph": 1.0, "src": 1.0, "dst": 2.0}
        self.assertEqual(model.score_one(sample), 0.0)

        model.learn_one(sample)
        model.learn_one({"graph": 2.0, "src": 1.0, "dst": 2.0})
        model.learn_one({"graph": 1.0, "src": 2.0, "dst": 3.0})

        score = model.score_one({"graph": 1.0, "src": 1.0, "dst": 2.0})
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)

    def test_deterministic_with_seed(self) -> None:
        model1 = self.create_model(edge_type_key=None, shingle_size=1, warm_up_graphs=4)
        model2 = self.create_model(edge_type_key=None, shingle_size=1, warm_up_graphs=4)

        for i in range(400):
            point = {
                "graph": float(i % 12),
                "src": float((i * 3) % 31),
                "dst": float((i * 5) % 37),
                "t": float(i // 4),
            }
            model1.learn_one(point)
            model2.learn_one(point)

        query = {"graph": 7.0, "src": 9.0, "dst": 11.0, "t": 101.0}
        self.assertAlmostEqual(
            model1.score_one(query),
            model2.score_one(query),
            places=12,
        )

    def test_outlier_scores_higher_than_normal(self) -> None:
        model = self.create_model(
            edge_type_key="etype",
            shingle_size=1,
            num_clusters=1,
            warm_up_graphs=1,
            max_graphs=8,
        )
        for i in range(300):
            model.learn_one(
                {"graph": 1.0, "src": 1.0, "dst": 2.0, "etype": 0.0, "t": float(i)}
            )

        normal_score = model.score_one(
            {"graph": 1.0, "src": 1.0, "dst": 2.0, "etype": 0.0, "t": 301.0}
        )
        outlier_score = model.score_one(
            {"graph": 99.0, "src": 900.0, "dst": 901.0, "etype": 1.0, "t": 301.0}
        )
        self.assertGreaterEqual(normal_score, 0.0)
        self.assertGreater(outlier_score, normal_score)

    def test_graph_state_is_bounded(self) -> None:
        model = self.create_model(
            edge_type_key=None,
            shingle_size=1,
            max_graphs=5,
            num_clusters=2,
            warm_up_graphs=1,
        )
        for i in range(100):
            model.learn_one(
                {
                    "graph": float(i),
                    "src": float(i % 7),
                    "dst": float((i + 1) % 7),
                    "t": float(i),
                }
            )

        self.assertLessEqual(len(model._graph_states), 5)
        self.assertLessEqual(len(model._graph_last_seen), 5)

    def test_reset_restores_cold_state(self) -> None:
        model = self.create_model(edge_type_key=None, shingle_size=1, warm_up_graphs=2)
        for i in range(100):
            model.learn_one(
                {
                    "graph": float(i % 4),
                    "src": float(i % 9),
                    "dst": float((i + 1) % 9),
                    "t": float(i),
                }
            )

        self.assertGreater(model.n_samples_seen, 0)
        model.reset()
        self.assertEqual(model.n_samples_seen, 0)
        self.assertEqual(
            model.score_one({"graph": 1.0, "src": 1.0, "dst": 2.0, "t": 1.0}),
            0.0,
        )

    def test_normalize_score_bounds_output(self) -> None:
        model = self.create_model(
            edge_type_key=None,
            shingle_size=1,
            warm_up_graphs=2,
            normalize_score=True,
        )
        for i in range(120):
            model.learn_one(
                {
                    "graph": float(i % 6),
                    "src": float(i % 11),
                    "dst": float((i + 1) % 11),
                    "t": float(i),
                }
            )

        score = model.score_one({"graph": 777.0, "src": 888.0, "dst": 999.0, "t": 120.0})
        self.assertGreaterEqual(score, 0.0)
        self.assertLess(score, 1.0)

    def test_predict_is_binary(self) -> None:
        model = self.create_model(
            edge_type_key=None,
            shingle_size=1,
            warm_up_graphs=2,
            normalize_score=True,
            predict_threshold=0.3,
        )
        for i in range(120):
            model.learn_one(
                {
                    "graph": float(i % 5),
                    "src": float(i % 13),
                    "dst": float((i + 1) % 13),
                    "t": float(i),
                }
            )

        prediction = model.predict_one(
            {"graph": 444.0, "src": 555.0, "dst": 666.0, "t": 121.0}
        )
        self.assertIn(prediction, (0, 1))

    def test_repr_contains_key_config(self) -> None:
        model = self.create_model(sketch_dim=128, num_clusters=6, max_graphs=64)
        output = repr(model)
        self.assertIn("StreamSpot", output)
        self.assertIn("sketch_dim=128", output)
        self.assertIn("num_clusters=6", output)
        self.assertIn("max_graphs=64", output)


if __name__ == "__main__":
    unittest.main()
