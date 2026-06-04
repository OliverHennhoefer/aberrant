"""Unit tests for Random Cut Forest anomaly detection model."""

import unittest

import numpy as np

from aberrant.model.iforest.random_cut import RandomCutForest, _RandomCutTree


class TestRandomCutForest(unittest.TestCase):
    """Test suite for RandomCutForest model."""

    def create_model(self) -> RandomCutForest:
        return RandomCutForest(
            n_trees=8,
            sample_size=32,
            shingle_size=1,
            warmup_samples=16,
            normalize_score=True,
            score_scale=8.0,
            seed=42,
        )

    def test_initialization_defaults(self):
        """Test default initialization."""
        model = RandomCutForest()
        self.assertEqual(model.n_trees, 40)
        self.assertEqual(model.sample_size, 256)
        self.assertEqual(model.shingle_size, 1)
        self.assertEqual(model.warmup_samples, 256)
        self.assertFalse(model.normalize_score)
        self.assertFalse(model._ready)

    def test_invalid_parameters(self):
        """Test parameter validation."""
        with self.assertRaises(ValueError):
            RandomCutForest(n_trees=0)
        with self.assertRaises(ValueError):
            RandomCutForest(sample_size=1)
        with self.assertRaises(ValueError):
            RandomCutForest(shingle_size=0)
        with self.assertRaises(ValueError):
            RandomCutForest(warmup_samples=0)
        with self.assertRaises(ValueError):
            RandomCutForest(score_scale=0.0)

    def test_score_zero_before_warmup(self):
        """score_one should return 0.0 before warm-up is complete."""
        model = self.create_model()
        for i in range(15):
            model.learn_one({"x": float(i), "y": float(i)})
        score = model.score_one({"x": 0.0, "y": 0.0})
        self.assertEqual(score, 0.0)

    def test_empty_and_non_numeric_input(self):
        """Model should reject empty and non-numeric inputs."""
        model = self.create_model()
        with self.assertRaises(ValueError):
            model.learn_one({})
        with self.assertRaises(ValueError):
            model.score_one({})
        with self.assertRaises(ValueError):
            model.learn_one({"x": "bad"})  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            model.score_one({"x": float("inf")})
        with self.assertRaises(ValueError):
            model.score_one({"x": float("nan")})

    def test_feature_key_consistency(self):
        """Model should enforce consistent feature keys."""
        model = self.create_model()
        for i in range(20):
            model.learn_one({"a": float(i), "b": float(i + 1)})

        with self.assertRaises(ValueError):
            model.score_one({"a": 1.0, "c": 2.0})
        with self.assertRaises(ValueError):
            model.learn_one({"a": 1.0, "c": 2.0})

    def test_deterministic_with_seed(self):
        """Same seed and data should produce identical scores."""
        model_1 = self.create_model()
        model_2 = self.create_model()

        data = [{"x": float(i % 7), "y": float(i % 3)} for i in range(64)]
        for point in data:
            model_1.learn_one(point)
            model_2.learn_one(point)

        test_point = {"x": 2.0, "y": 1.0}
        self.assertAlmostEqual(
            model_1.score_one(test_point),
            model_2.score_one(test_point),
            places=12,
        )

    def test_bounded_memory(self):
        """Stored point count should be bounded by sample_size."""
        model = RandomCutForest(
            n_trees=4,
            sample_size=20,
            warmup_samples=10,
            seed=7,
        )
        for i in range(200):
            model.learn_one({"x": float(i), "y": float(i * 2)})
            self.assertLessEqual(len(model._id_window), 20)
            for tree in model._trees:
                self.assertLessEqual(tree.size, 20)

    def test_outlier_scores_higher_than_normal(self):
        """Outlier should score higher than near-cluster point."""
        model = RandomCutForest(
            n_trees=16,
            sample_size=64,
            warmup_samples=32,
            seed=11,
        )
        rng = np.random.default_rng(123)
        for _ in range(200):
            model.learn_one(
                {
                    "x": float(rng.normal(0.0, 0.25)),
                    "y": float(rng.normal(0.0, 0.25)),
                }
            )

        normal_score = model.score_one({"x": 0.05, "y": -0.02})
        outlier_score = model.score_one({"x": 6.0, "y": 6.0})
        self.assertGreater(outlier_score, normal_score)

    def test_tree_score_matches_reference_insert_codisp_remove(self):
        """score_point should equal candidate-inclusive reference CoDisp."""
        points = [
            np.asarray([0.0, 0.0]),
            np.asarray([0.1, 0.0]),
            np.asarray([0.0, 0.1]),
            np.asarray([3.0, 3.0]),
        ]
        preview_tree = _RandomCutTree(np.random.default_rng(42))
        reference_tree = _RandomCutTree(np.random.default_rng(42))
        for point_id, point in enumerate(points):
            preview_tree.insert(point_id, point)
            reference_tree.insert(point_id, point)

        query = np.asarray([8.0, 8.0])
        preview_score = preview_tree.score_point(query)
        reference_tree.insert(99, query)
        reference_score = reference_tree.codisp(99)

        self.assertEqual(preview_score, reference_score)
        self.assertEqual(preview_tree.size, len(points))
        self.assertNotIn(-1, preview_tree._id_to_leaf)

    def test_tree_score_preview_restores_rng_state(self):
        """Scoring must not change the random cuts used by future inserts."""
        scored_tree = _RandomCutTree(np.random.default_rng(7))
        control_tree = _RandomCutTree(np.random.default_rng(7))
        for point_id in range(8):
            point = np.asarray([float(point_id), float(point_id % 3)])
            scored_tree.insert(point_id, point)
            control_tree.insert(point_id, point)

        scored_tree.score_point(np.asarray([20.0, 20.0]))
        new_point = np.asarray([9.0, 4.0])
        scored_tree.insert(9, new_point)
        control_tree.insert(9, new_point)

        self.assertEqual(scored_tree.codisp(9), control_tree.codisp(9))

    def test_shingle_support(self):
        """Model should support shingled streaming state."""
        model = RandomCutForest(
            n_trees=6,
            sample_size=24,
            shingle_size=3,
            warmup_samples=8,
            seed=5,
        )
        for i in range(50):
            model.learn_one({"x": float(i), "y": float(i % 4)})

        score = model.score_one({"x": 51.0, "y": 1.0})
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)

    def test_raw_score_mode(self):
        """Raw score mode should return a non-negative continuous score."""
        model = RandomCutForest(
            n_trees=8,
            sample_size=32,
            warmup_samples=16,
            normalize_score=False,
            seed=42,
        )
        for i in range(50):
            model.learn_one({"x": float(i % 5), "y": float(i % 3)})

        score = model.score_one({"x": 10.0, "y": -10.0})
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)

    def test_reset_restores_state(self):
        """reset() should clear learned state."""
        model = self.create_model()
        for i in range(40):
            model.learn_one({"x": float(i), "y": float(i + 1)})

        self.assertTrue(model._ready)
        model.reset()
        self.assertFalse(model._ready)
        self.assertEqual(model.score_one({"x": 1.0, "y": 2.0}), 0.0)

    def test_repr_contains_key_config(self):
        """repr should include key hyperparameters."""
        model = self.create_model()
        output = repr(model)
        self.assertIn("RandomCutForest", output)
        self.assertIn("n_trees=8", output)
        self.assertIn("sample_size=32", output)


if __name__ == "__main__":
    unittest.main()
