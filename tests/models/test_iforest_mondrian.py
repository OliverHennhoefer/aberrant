"""Tests for the online Mondrian Forest anomaly detection model."""

import math
import unittest

import numpy as np

from aberrant.model.iforest.mondrian import (
    MondrianIsolationForest as MondrianForest,
)
from aberrant.model.iforest.mondrian import (
    MondrianNode,
    MondrianTree,
)
from tests.utils import DataGenerator


class TestMondrianNode(unittest.TestCase):
    """Test suite for MondrianNode."""

    def test_initialization(self):
        """Node should initialize as an empty leaf."""
        node = MondrianNode()

        self.assertTrue(node.is_leaf())
        self.assertIsNone(node.split_feature)
        self.assertIsNone(node.split_threshold)
        self.assertIsNone(node.left_child)
        self.assertIsNone(node.right_child)
        self.assertEqual(node.count, 0)
        self.assertIsNone(node.min)
        self.assertIsNone(node.max)
        self.assertTrue(math.isinf(node.split_time))

    def test_update_stats(self):
        """Node stats should track min/max bounds and count."""
        node = MondrianNode()
        node.update_stats(np.array([3.0, 2.0, 1.0]))
        node.update_stats(np.array([1.0, 5.0, 4.0]))

        self.assertEqual(node.count, 2)
        np.testing.assert_array_equal(node.min, np.array([1.0, 2.0, 1.0]))
        np.testing.assert_array_equal(node.max, np.array([3.0, 5.0, 4.0]))

    def test_recompute_from_children(self):
        """Internal node should derive bounds/count from both children."""
        left = MondrianNode()
        left.update_stats(np.array([0.0, 1.0]))
        left.update_stats(np.array([1.0, 2.0]))

        right = MondrianNode()
        right.update_stats(np.array([3.0, 4.0]))

        parent = MondrianNode(split_time=0.2)
        parent.is_leaf_ = False
        parent.left_child = left
        parent.right_child = right
        parent.recompute_from_children()

        self.assertEqual(parent.count, 3)
        np.testing.assert_array_equal(parent.min, np.array([0.0, 1.0]))
        np.testing.assert_array_equal(parent.max, np.array([3.0, 4.0]))


class TestMondrianTree(unittest.TestCase):
    """Test suite for MondrianTree."""

    def test_initialization(self):
        """Tree should initialize with empty root block and zero samples."""
        selected_indices = np.array([0, 2, 4])
        tree = MondrianTree(
            selected_indices=selected_indices,
            lambda_=1.5,
            rng=np.random.default_rng(42),
        )

        np.testing.assert_array_equal(tree.selected_indices, selected_indices)
        self.assertEqual(tree.lambda_, 1.5)
        self.assertEqual(tree.n_samples, 0)
        self.assertTrue(tree.root.is_leaf())
        self.assertEqual(tree.root.count, 0)
        self.assertEqual(tree.root.split_time, 1.5)

    def test_first_sample_learned_once(self):
        """First update should increment root count exactly once."""
        tree = MondrianTree(
            selected_indices=np.array([0, 1]),
            lambda_=1.0,
            rng=np.random.default_rng(42),
        )

        tree.learn_one(np.array([1.0, 2.0]))
        self.assertEqual(tree.n_samples, 1)
        self.assertEqual(tree.root.count, 1)

    def test_outside_point_can_create_new_parent(self):
        """A far outside point should typically create a parent split."""
        tree = MondrianTree(
            selected_indices=np.array([0, 1]),
            lambda_=25.0,
            rng=np.random.default_rng(0),
        )

        tree.learn_one(np.array([0.0, 0.0]))
        tree.learn_one(np.array([10.0, 10.0]))

        self.assertEqual(tree.n_samples, 2)
        self.assertFalse(tree.root.is_leaf())
        self.assertEqual(tree.root.count, 2)
        self.assertIsNotNone(tree.root.left_child)
        self.assertIsNotNone(tree.root.right_child)
        self.assertIsNotNone(tree.root.split_feature)
        self.assertIsNotNone(tree.root.split_threshold)
        self.assertLess(tree.root.split_time, tree.lambda_)

    def test_score_without_learning(self):
        """Scoring empty tree should return 0."""
        tree = MondrianTree(
            selected_indices=np.array([0, 1]),
            lambda_=1.0,
            rng=np.random.default_rng(42),
        )
        self.assertEqual(tree.score_one(np.array([1.0, 2.0])), 0.0)

    def test_score_uses_leaf_size_adjustment(self):
        """Repeated identical points in one leaf should increase path length."""
        tree = MondrianTree(
            selected_indices=np.array([0, 1]),
            lambda_=0.1,
            rng=np.random.default_rng(123),
        )
        point = np.array([1.0, 1.0])

        tree.learn_one(point)
        tree.learn_one(point)

        score = tree.score_one(point)
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)


class TestMondrianForest(unittest.TestCase):
    """Test suite for MondrianForest."""

    def setUp(self):
        self.data_generator = DataGenerator(seed=42)

    def test_initialization_defaults(self):
        """Forest should initialize with expected defaults."""
        forest = MondrianForest()
        self.assertEqual(forest.n_estimators, 100)
        self.assertEqual(forest.subspace_size, 256)
        self.assertEqual(forest.lambda_, 1.0)
        self.assertIsNone(forest.seed)
        self.assertEqual(forest.n_samples, 0)
        self.assertEqual(len(forest.trees), 0)
        self.assertIsNone(forest._schema.names)

    def test_initialization_validation(self):
        """Invalid constructor parameters should fail fast."""
        with self.assertRaises(ValueError):
            MondrianForest(n_estimators=0)
        with self.assertRaises(ValueError):
            MondrianForest(subspace_size=0)
        with self.assertRaises(ValueError):
            MondrianForest(lambda_=0)

    def test_feature_initialization_and_no_double_learn(self):
        """First sample should initialize features and be learned once per tree."""
        forest = MondrianForest(n_estimators=5, subspace_size=2, seed=42)
        first_point = {"c": 3.0, "a": 1.0, "b": 2.0}

        forest.learn_one(first_point)

        self.assertEqual(forest._schema.names, ("a", "b", "c"))
        self.assertEqual(len(forest.trees), 5)
        self.assertEqual(forest.n_samples, 1)
        self.assertTrue(all(tree.n_samples == 1 for tree in forest.trees))

    def test_subspace_size_adapts_to_feature_count(self):
        """Requested subspace should be clipped by feature dimensionality."""
        forest = MondrianForest(n_estimators=3, subspace_size=10, seed=42)
        forest.learn_one({"x": 1.0, "y": 2.0})
        self.assertEqual(forest.subspace_size, 2)

    def test_input_validation(self):
        """Forest should reject empty, non-numeric, and non-finite input."""
        forest = MondrianForest(n_estimators=3, subspace_size=2, seed=42)

        with self.assertRaises(ValueError):
            forest.learn_one({})
        with self.assertRaises(ValueError):
            forest.score_one({})
        with self.assertRaises(ValueError):
            forest.learn_one({"x": "bad"})  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            forest.learn_one({"x": float("inf")})
        with self.assertRaises(ValueError):
            forest.score_one({"x": float("nan")})

    def test_feature_key_consistency(self):
        """Forest should enforce consistent feature keys after initialization."""
        forest = MondrianForest(n_estimators=3, subspace_size=2, seed=42)
        forest.learn_one({"a": 1.0, "b": 2.0})

        with self.assertRaises(ValueError):
            forest.learn_one({"a": 1.0, "c": 2.0})
        with self.assertRaises(ValueError):
            forest.score_one({"a": 1.0, "c": 2.0})

    def test_score_before_learning(self):
        """Scoring before any learn should return 0."""
        forest = MondrianForest(n_estimators=5, subspace_size=2)
        self.assertEqual(forest.score_one({"feature": 1.0}), 0.0)

    def test_score_normalization_bounds(self):
        """Scores should stay in [0, 1] once the forest is trained."""
        forest = MondrianForest(n_estimators=10, subspace_size=3, seed=42)
        training_data = self.data_generator.generate_streaming_data(n=200, n_features=4)

        for point in training_data:
            forest.learn_one(point)

        test_points = [
            {"feature_0": 0.0, "feature_1": 0.0, "feature_2": 0.0, "feature_3": 0.0},
            {
                "feature_0": 100.0,
                "feature_1": 100.0,
                "feature_2": 100.0,
                "feature_3": 100.0,
            },
            training_data[0],
        ]

        for point in test_points:
            score = forest.score_one(point)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_reproducibility_with_seed(self):
        """Same seed and data stream should produce equal scores."""
        forest1 = MondrianForest(n_estimators=6, subspace_size=2, seed=42)
        forest2 = MondrianForest(n_estimators=6, subspace_size=2, seed=42)
        test_data = self.data_generator.generate_streaming_data(n=50, n_features=3)

        scores1 = []
        scores2 = []
        for point in test_data:
            forest1.learn_one(point.copy())
            forest2.learn_one(point.copy())
            scores1.append(forest1.score_one(point))
            scores2.append(forest2.score_one(point))

        for score1, score2 in zip(scores1, scores2, strict=False):
            self.assertAlmostEqual(score1, score2, places=12)

    def test_different_seeds_produce_different_scores(self):
        """Different seeds should lead to different score trajectories."""
        forest1 = MondrianForest(n_estimators=10, subspace_size=3, seed=42)
        forest2 = MondrianForest(n_estimators=10, subspace_size=3, seed=123)
        test_data = self.data_generator.generate_streaming_data(n=50, n_features=4)

        scores1 = []
        scores2 = []
        for point in test_data:
            forest1.learn_one(point.copy())
            forest2.learn_one(point.copy())
            scores1.append(forest1.score_one(point))
            scores2.append(forest2.score_one(point))

        different_count = sum(
            1
            for score1, score2 in zip(scores1, scores2, strict=False)
            if abs(score1 - score2) > 1e-10
        )
        self.assertGreater(different_count, len(scores1) * 0.4)

    def test_n_estimators_effect(self):
        """Different ensemble sizes should still return valid scores."""
        forest_few = MondrianForest(n_estimators=5, subspace_size=2, seed=42)
        forest_many = MondrianForest(n_estimators=40, subspace_size=2, seed=42)
        training_data = self.data_generator.generate_streaming_data(n=100, n_features=3)

        for point in training_data:
            forest_few.learn_one(point.copy())
            forest_many.learn_one(point.copy())

        test_point = {"feature_0": 10.0, "feature_1": 10.0, "feature_2": 10.0}
        score_few = forest_few.score_one(test_point)
        score_many = forest_many.score_one(test_point)

        self.assertIsInstance(score_few, float)
        self.assertIsInstance(score_many, float)
        self.assertGreaterEqual(score_few, 0.0)
        self.assertLessEqual(score_few, 1.0)
        self.assertGreaterEqual(score_many, 0.0)
        self.assertLessEqual(score_many, 1.0)

    def test_c_factor_computation(self):
        """c-factor should match isolation-forest semantics."""
        forest = MondrianForest(n_estimators=1, subspace_size=1)

        forest.n_samples = 1
        self.assertEqual(forest._compute_c_factor(), 0.0)

        forest.n_samples = 2
        c_2 = forest._compute_c_factor()
        self.assertEqual(c_2, 1.0)

        forest.n_samples = 100
        c_100 = forest._compute_c_factor()
        self.assertGreater(c_100, c_2)

    def test_repr_string(self):
        """repr should include key hyperparameters."""
        forest = MondrianForest(n_estimators=25, subspace_size=8, lambda_=2.5, seed=999)
        repr_string = repr(forest)
        self.assertIn("MondrianIsolationForest", repr_string)
        self.assertIn("n_estimators=25", repr_string)
        self.assertIn("subspace_size=8", repr_string)
        self.assertIn("lambda_=2.5", repr_string)
        self.assertIn("seed=999", repr_string)


if __name__ == "__main__":
    unittest.main()
