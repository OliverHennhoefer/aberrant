"""Unit tests for Stream Random Histogram Forest - Edge cases and initialization only.

Real dataset tests are in tests/integration/test_iforest_models.py
"""

import copy
import random
import unittest

import numpy as np

from aberrant.model.iforest.rand_hist import (
    StreamRandomHistogramForest,
    _RandomHistogramTree,
)


class TestStreamRandomHistogramForestEdgeCases(unittest.TestCase):
    """Test Stream Random Histogram Forest edge cases and initialization."""

    def create_model(self):
        return StreamRandomHistogramForest(
            n_estimators=5,  # Small for fast testing
            max_depth=5,
            window_size=50,
            seed=42,
        )

    def setUp(self):
        self.model = self.create_model()

    def test_initialization_valid_parameters(self):
        """Test initialization with valid parameters."""
        model = StreamRandomHistogramForest(
            n_estimators=10, max_depth=8, window_size=100, seed=123
        )

        self.assertEqual(model.n_estimators, 10)
        self.assertEqual(model.max_depth, 8)
        self.assertFalse(hasattr(model, "max_bins"))
        self.assertEqual(model.window_size, 100)
        self.assertEqual(model.seed, 123)

    def test_initialization_invalid_parameters(self):
        """Test initialization with invalid parameters raises errors."""
        # Invalid n_estimators
        with self.assertRaises((ValueError, AssertionError)):
            StreamRandomHistogramForest(n_estimators=0)

        # Invalid max_depth
        with self.assertRaises((ValueError, AssertionError)):
            StreamRandomHistogramForest(max_depth=0)

        # Invalid window_size
        with self.assertRaises((ValueError, AssertionError)):
            StreamRandomHistogramForest(window_size=0)

    def test_max_bins_is_no_longer_accepted(self):
        with self.assertRaises(TypeError):
            StreamRandomHistogramForest(max_bins=5)  # type: ignore[call-arg]

    def test_window_size_behavior(self):
        """Test window size constraint behavior."""
        model = StreamRandomHistogramForest(
            n_estimators=3,
            max_depth=4,
            window_size=10,  # Small window
            seed=42,
        )

        # Add more points than window size
        for i in range(20):
            point = {"feature1": float(i), "feature2": float(i * 2)}
            model.learn_one(point)

        # Test that model still functions
        test_point = {"feature1": 15.0, "feature2": 30.0}
        score = model.score_one(test_point)

        self.assertIsInstance(score, (int, float))
        self.assertGreaterEqual(score, 0.0)

    def test_different_depths(self):
        """Test different max_depth configurations."""
        for max_depth in [3, 5, 10]:
            with self.subTest(max_depth=max_depth):
                model = StreamRandomHistogramForest(
                    n_estimators=3,
                    max_depth=max_depth,
                    window_size=50,
                    seed=42,
                )

                # Should initialize without error
                self.assertEqual(model.max_depth, max_depth)

                # Basic functionality test
                point = {"feature1": 1.0, "feature2": 2.0}
                model.learn_one(point)
                score = model.score_one(point)

                self.assertIsInstance(score, (int, float))
                self.assertGreaterEqual(score, 0.0)

    def test_empty_dict_handling(self):
        """Test handling of empty feature dictionary."""
        model = self.create_model()

        try:
            model.learn_one({})
            score = model.score_one({})
            self.assertIsInstance(score, (int, float))
        except (ValueError, KeyError):
            # Acceptable to reject empty dict
            pass

    def test_single_feature_data(self):
        """Test with single feature data."""
        model = self.create_model()

        # Train with single feature
        for i in range(20):
            point = {"feature": float(i) % 10}  # Values 0-9 repeated
            model.learn_one(point)

        # Test scoring
        test_point = {"feature": 5.0}
        score = model.score_one(test_point)

        self.assertIsInstance(score, (int, float))
        self.assertGreaterEqual(score, 0.0)

    def test_deterministic_behavior(self):
        """Test deterministic behavior with same seed."""
        model1 = StreamRandomHistogramForest(
            n_estimators=3, max_depth=5, window_size=20, seed=42
        )
        model2 = StreamRandomHistogramForest(
            n_estimators=3, max_depth=5, window_size=20, seed=42
        )

        # Train both models identically
        training_points = [
            {"feature1": 1.0, "feature2": 2.0},
            {"feature1": 3.0, "feature2": 4.0},
            {"feature1": 5.0, "feature2": 6.0},
        ]

        for point in training_points:
            model1.learn_one(point.copy())
            model2.learn_one(point.copy())

        # Scores should be identical with same seed
        test_point = {"feature1": 2.5, "feature2": 3.5}
        score1 = model1.score_one(test_point)
        score2 = model2.score_one(test_point)

        self.assertEqual(score1, score2, "Same seed should produce identical results")

    def test_different_seeds_produce_different_node_random_values(self):
        model1 = StreamRandomHistogramForest(
            n_estimators=2, max_depth=5, window_size=4, seed=42
        )
        model2 = StreamRandomHistogramForest(
            n_estimators=2, max_depth=5, window_size=4, seed=43
        )
        points = [
            {"x": 0.0, "y": 4.0},
            {"x": 1.0, "y": 1.0},
            {"x": 2.0, "y": 3.0},
            {"x": 3.0, "y": 0.0},
        ]
        for point in points:
            model1.learn_one(point)
            model2.learn_one(point)

        caches1 = [tree._node_random for tree in model1._trees]
        caches2 = [tree._node_random for tree in model2._trees]
        self.assertNotEqual(caches1, caches2)

    def test_large_depth_allocates_random_values_only_for_visited_nodes(self):
        model = StreamRandomHistogramForest(
            n_estimators=2,
            max_depth=64,
            window_size=4,
            seed=42,
        )
        for point in [
            {"x": 0.0, "y": 4.0},
            {"x": 1.0, "y": 1.0},
            {"x": 2.0, "y": 3.0},
            {"x": 3.0, "y": 0.0},
        ]:
            model.learn_one(point)

        cached_nodes = sum(len(tree._node_random) for tree in model._trees)
        self.assertGreater(cached_nodes, 0)
        self.assertLessEqual(cached_nodes, model.n_estimators * (2 * model.window_size - 1))
        self.assertIsInstance(model.score_one({"x": 1.5, "y": 2.0}), float)

    def test_node_random_values_are_independent_of_visit_order(self):
        nodes = [0, 7, 2**32, 2**40 + 7]
        tree1 = _RandomHistogramTree(64, 2, np.random.SeedSequence(42))
        tree2 = _RandomHistogramTree(64, 2, np.random.SeedSequence(42))

        values1 = {node: tree1._random_values(node) for node in nodes}
        values2 = {node: tree2._random_values(node) for node in reversed(nodes)}

        self.assertEqual(values1, values2)

    def test_score_preview_preserves_cache_and_matches_later_learning(self):
        model = StreamRandomHistogramForest(
            n_estimators=3,
            max_depth=8,
            window_size=4,
            seed=42,
        )
        for point in [
            {"x": 0.0, "y": 4.0},
            {"x": 1.0, "y": 1.0},
            {"x": 2.0, "y": 3.0},
            {"x": 3.0, "y": 0.0},
        ]:
            model.learn_one(point)

        query = {"x": 1.5, "y": 2.0}
        point = model._schema.preview(query).values
        preview_trees = copy.deepcopy(model._trees)
        for tree in preview_trees:
            tree.insert(point)
        preview_caches = [tree._node_random for tree in preview_trees]
        learned_caches = [copy.deepcopy(tree._node_random) for tree in model._trees]

        model.score_one(query)
        self.assertEqual(
            [tree._node_random for tree in model._trees],
            learned_caches,
        )

        model.learn_one(query)
        self.assertEqual(
            [tree._node_random for tree in model._trees],
            preview_caches,
        )

    def test_score_preview_matches_deepcopy_insert_reference(self):
        model = StreamRandomHistogramForest(
            n_estimators=4,
            max_depth=8,
            window_size=8,
            seed=42,
        )
        for index in range(20):
            model.learn_one(
                {
                    "x": float(index % 7),
                    "y": float((index * index) % 11),
                }
            )

        query = {"x": 2.5, "y": 7.5}
        point = model._schema.preview(query).values
        candidate_size = model._forest_size + 1
        expected = 0.0
        for learned_tree in model._trees:
            reference_tree = copy.deepcopy(learned_tree)
            leaf_size = reference_tree.insert(point)
            if leaf_size > 0:
                expected += np.log(float(candidate_size) / float(leaf_size))

        self.assertAlmostEqual(model.score_one(query), expected)

    def test_model_randomness_does_not_mutate_global_random_state(self):
        random.seed(123)
        expected_next = random.random()
        random.seed(123)

        model = self.create_model()
        model.learn_one({"feature": 1.0})

        self.assertEqual(random.random(), expected_next)

    def test_repeated_values(self):
        """Test behavior with repeated values (histogram binning edge case)."""
        model = self.create_model()

        # Train with repeated values
        repeated_point = {"feature1": 5.0, "feature2": 10.0}
        for _ in range(15):
            model.learn_one(repeated_point.copy())

        # Test with same value
        score1 = model.score_one(repeated_point)

        # Test with different value
        different_point = {"feature1": 25.0, "feature2": 50.0}
        score2 = model.score_one(different_point)

        self.assertIsInstance(score1, (int, float))
        self.assertIsInstance(score2, (int, float))
        self.assertGreaterEqual(score1, 0.0)
        self.assertGreaterEqual(score2, 0.0)

    def test_schema_change_raises_instead_of_zero_imputation(self):
        model = self.create_model()
        model.learn_one({"x": 0.1, "y": 0.2})

        with self.assertRaises(ValueError):
            model.learn_one({"x": 0.1})
        with self.assertRaises(ValueError):
            model.score_one({"x": 0.1})


if __name__ == "__main__":
    unittest.main()
