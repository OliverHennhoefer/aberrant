"""Unit tests for Stream Random Histogram Forest - Edge cases and initialization only.

Real dataset tests are in tests/integration/test_iforest_models.py
"""

import copy
import pickle
import random
import time
import unittest
import warnings

import numpy as np

from aberrant.model.iforest.rand_hist import (
    StreamRandomHistogramForest,
    _fold_moments,
    _moments_with,
    _RandomHistogramTree,
    _RHFBranch,
    _RHFNode,
    _selected_feature,
    _selected_features,
)
from tests import score_sequences


def _tree_shape(node: _RHFNode | None) -> tuple:
    """Return everything that defines a tree, with arrays as bytes."""
    if node is None:
        return ()
    if isinstance(node, _RHFBranch):
        return (
            node.depth,
            node.node_id,
            node.split_feature,
            node.split_value,
            node.moments.tobytes(),
            _tree_shape(node.left),
            _tree_shape(node.right),
        )
    return (node.depth, node.node_id, [point.tobytes() for point in node.points])


def _forest_shape(model: StreamRandomHistogramForest) -> list[tuple]:
    return [(_tree_shape(tree.root), tree._node_random) for tree in model._trees]


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
        self.assertLessEqual(
            cached_nodes, model.n_estimators * (2 * model.window_size - 1)
        )
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


class TestStreamRandomHistogramForestExactness(unittest.TestCase):
    """Planned insertions must reproduce the 1.4.0 trees and scores."""

    def test_score_sequences_match_recorded_reference(self):
        for case in score_sequences.load("stream_random_histogram_forest"):
            with self.subTest(case=case["name"]), warnings.catch_warnings():
                # The overflowing case warns where NumPy moments overflow.
                warnings.simplefilter("ignore", RuntimeWarning)
                scores = score_sequences.replay(
                    StreamRandomHistogramForest(**case["params"]), case
                )
                np.testing.assert_allclose(scores, case["scores"], rtol=1e-12, atol=0)

    def test_moments_with_matches_folding_one_value_at_a_time(self):
        rng = np.random.default_rng(3)
        for scale in (1.0, 1e-160, 1e80):
            columns = (rng.standard_cauchy(size=(12, 3, 9)) * scale).tolist()
            point = rng.normal(size=3) * scale
            moments = np.array(
                [[_fold_moments(values) for values in branch] for branch in columns]
            )
            expected = np.array(
                [
                    [
                        _fold_moments([*values, value])
                        for values, value in zip(branch, point.tolist(), strict=True)
                    ]
                    for branch in columns
                ]
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                updated = _moments_with(moments, point)
            np.testing.assert_array_equal(updated, expected)

    def test_selected_features_match_the_scalar_selection(self):
        rng = np.random.default_rng(4)
        weights = rng.exponential(size=(64, 7))
        weights[rng.random(weights.shape) < 0.2] = 0.0
        weights[1] = 0.0
        weights[2, 3] = np.inf
        weights[3, [1, 5]] = np.nan
        weights[4, 0] = np.nan
        weights[5, 6] = np.nan
        weights[6] = [np.inf, 0.0, np.nan, 1.0, 0.0, np.inf, 2.0]
        randoms = rng.uniform(np.nextafter(0.0, 1.0), 1.0, size=len(weights))
        randoms[7] = np.nextafter(1.0, 0.0)

        selected = _selected_features(weights, randoms).tolist()

        for row, (row_weights, row_random) in enumerate(
            zip(weights, randoms.tolist(), strict=True)
        ):
            total = float(np.sum(row_weights))
            expected = (
                -1
                if total <= 0.0
                else _selected_feature(row_weights, total, row_random)
            )
            self.assertEqual(selected[row], expected, msg=f"row {row}")

    def test_learning_a_scored_point_matches_learning_it_directly(self):
        events = score_sequences.make_stream(seed=7, n_events=260, n_features=3)
        params = {"n_estimators": 4, "max_depth": 6, "window_size": 40, "seed": 8}
        scored = StreamRandomHistogramForest(**params)
        learned = StreamRandomHistogramForest(**params)
        scored_elsewhere = StreamRandomHistogramForest(**params)
        for index, event in enumerate(events):
            scored.score_one(event)
            scored.learn_one(event)
            learned.learn_one(event)
            # A preview of another point must not be applied to this one.
            scored_elsewhere.score_one(events[-1 - index])
            scored_elsewhere.learn_one(event)

        self.assertEqual(_forest_shape(scored), _forest_shape(learned))
        self.assertEqual(_forest_shape(scored_elsewhere), _forest_shape(learned))

    def test_pickles_leave_out_the_preview(self):
        events = score_sequences.make_stream(seed=9, n_events=80, n_features=2)
        model = StreamRandomHistogramForest(
            n_estimators=3, max_depth=5, window_size=30, seed=10
        )
        for event in events[:60]:
            model.learn_one(event)
        before = pickle.dumps(model)

        score = model.score_one(events[60])

        self.assertEqual(pickle.dumps(model), before)
        restored = pickle.loads(pickle.dumps(model))
        self.assertIsNone(restored._previewed)
        self.assertEqual(restored.score_one(events[60]), score)
        model.learn_one(events[60])
        restored.learn_one(events[60])
        self.assertEqual(_forest_shape(restored), _forest_shape(model))

    def test_default_forest_speed_smoke(self):
        events = score_sequences.make_stream(seed=1, n_events=1000, n_features=8)
        model = StreamRandomHistogramForest(seed=1)

        start = time.perf_counter()
        score_sequences.score_then_learn(model, events)

        # Generous bound against large regressions; typical runs take ~1 s.
        self.assertLess(time.perf_counter() - start, 10.0)


if __name__ == "__main__":
    unittest.main()
