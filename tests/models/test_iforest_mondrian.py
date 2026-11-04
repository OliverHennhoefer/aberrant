"""Tests for the online Mondrian Forest anomaly detection model."""

import copy
import math
import pickle
import time
import unittest

import numpy as np

from aberrant.model.iforest.mondrian import (
    MondrianBranch,
    MondrianLeaf,
    MondrianNode,
    MondrianTree,
    _bottom_rates,
)
from aberrant.model.iforest.mondrian import (
    MondrianIsolationForest as MondrianForest,
)
from tests import score_sequences
from tests.utils import DataGenerator


def _tree_shape(root: MondrianNode | None) -> list[object]:
    """Return a comparable description of every block of a tree."""
    shape: list[object] = []
    pending = [root]
    while pending:
        node = pending.pop()
        if node is None:
            shape.append(None)
            continue
        shape.append(
            (node.count, node.split_time, node.min.tolist(), node.max.tolist())
        )
        if isinstance(node, MondrianBranch):
            shape.append((node.split_feature, node.split_threshold))
            pending += [node.right_child, node.left_child]
    return shape


def _forest_shape(forest: MondrianForest) -> object:
    """Return trees, sample counts and generator states of a forest."""
    return (
        forest.n_samples,
        [
            (tree.n_samples, tree.rng.bit_generator.state, _tree_shape(tree.root))
            for tree in forest.trees
        ],
    )


class TestMondrianNode(unittest.TestCase):
    """Test suite for MondrianNode."""

    def test_initialization(self):
        """A leaf starts occupied and carries no branch-only metadata."""
        point = np.array([3.0, 2.0, 1.0])
        node = MondrianLeaf.from_point(point, math.inf)

        self.assertTrue(node.is_leaf())
        self.assertFalse(hasattr(node, "split_feature"))
        self.assertFalse(hasattr(node, "left_child"))
        self.assertEqual(node.count, 1)
        np.testing.assert_array_equal(node.min, point)
        np.testing.assert_array_equal(node.max, point)
        self.assertTrue(math.isinf(node.split_time))
        point[:] = -1.0
        np.testing.assert_array_equal(node.min, [3.0, 2.0, 1.0])

    def test_update_stats(self):
        """Node stats should track min/max bounds and count."""
        node = MondrianLeaf.from_point(np.array([3.0, 2.0, 1.0]), math.inf)
        node.update_stats(np.array([1.0, 5.0, 4.0]))

        self.assertEqual(node.count, 2)
        np.testing.assert_array_equal(node.min, np.array([1.0, 2.0, 1.0]))
        np.testing.assert_array_equal(node.max, np.array([3.0, 5.0, 4.0]))

    def test_recompute_from_children(self):
        """Internal node should derive bounds/count from both children."""
        left = MondrianLeaf.from_point(np.array([0.0, 1.0]), math.inf)
        left.update_stats(np.array([1.0, 2.0]))

        right = MondrianLeaf.from_point(np.array([3.0, 4.0]), math.inf)

        parent = MondrianBranch.join(left, right, 0, 2.0, 0.2)
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
        self.assertIsNone(tree.root)

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


class TestMondrianForestExactness(unittest.TestCase):
    """The batched insertion must reproduce lone per-tree insertions exactly."""

    def test_score_sequences_match_recorded_reference(self):
        for case in score_sequences.load("mondrian_isolation_forest"):
            with self.subTest(case=case["name"]):
                model = MondrianForest(**case["params"])
                scores = score_sequences.replay(model, case)
                self.assertEqual(scores, case["scores"])

    def test_scaled_standard_exponential_matches_exponential(self):
        rng = np.random.default_rng(3)
        scales = rng.exponential(size=2000) * 10.0 ** rng.integers(-300, 300, 2000)
        scales = [*scales.tolist(), 0.0, 5e-324, 1.7e308, math.inf]
        drawn = np.random.default_rng(4)
        scaled = np.random.default_rng(4)
        for scale in scales:
            self.assertEqual(
                float(drawn.exponential(scale=scale)),
                scale * scaled.standard_exponential(),
            )
        self.assertEqual(drawn.bit_generator.state, scaled.bit_generator.state)

    def test_stacked_rates_match_lone_blocks(self):
        rng = np.random.default_rng(5)
        for width in (1, 3, 8, 9, 17):
            with self.subTest(width=width):
                paths = []
                for _ in range(6):
                    path = []
                    for _ in range(int(rng.integers(1, 9))):
                        leaf = MondrianLeaf.from_point(rng.normal(size=width), 1.0)
                        leaf.update_stats(rng.normal(size=width))
                        path.append(leaf)
                    paths.append(path)
                points = rng.normal(size=(len(paths), width))
                lows = [int(rng.integers(0, len(path))) for path in paths]
                for path, point, low, rates in zip(
                    paths, points, lows, _bottom_rates(paths, points, lows), strict=True
                ):
                    for level in range(low, len(path)):
                        node = path[level]
                        weights = np.maximum(node.min - point, 0.0) + np.maximum(
                            point - node.max, 0.0
                        )
                        row = level + rates.offset
                        self.assertEqual(rates.weights[row].tolist(), weights.tolist())
                        self.assertEqual(rates.sums[row], float(np.sum(weights)))

    def test_learning_a_scored_point_matches_learning_it_directly(self):
        for window_size in (None, 24):
            with self.subTest(window_size=window_size):
                events = score_sequences.make_stream(seed=7, n_events=260, n_features=4)
                params = {"n_estimators": 6, "subspace_size": 3, "seed": 8}
                scored = MondrianForest(**params, window_size=window_size)
                learned = MondrianForest(**params, window_size=window_size)
                scored_elsewhere = MondrianForest(**params, window_size=window_size)
                for index, event in enumerate(events):
                    scored.score_one(event)
                    scored.learn_one(event)
                    learned.learn_one(event)
                    # Walks planned for another point must not be reused.
                    scored_elsewhere.score_one(events[-1 - index])
                    scored_elsewhere.learn_one(event)

                self.assertEqual(_forest_shape(scored), _forest_shape(learned))
                self.assertEqual(
                    _forest_shape(scored_elsewhere), _forest_shape(learned)
                )

    def test_forest_matches_lone_tree_insertions(self):
        events = score_sequences.make_stream(seed=9, n_events=200, n_features=3)
        forest = MondrianForest(n_estimators=5, subspace_size=2, lambda_=4.0, seed=10)
        forest.learn_one(events[0])
        trees = copy.deepcopy(forest.trees)
        for event in events[1:]:
            forest.score_one(event)
            forest.learn_one(event)
            values = np.array([event[name] for name in sorted(event)])
            for tree in trees:
                tree.learn_one_from_global(values)

        for alone, batched in zip(trees, forest.trees, strict=True):
            self.assertEqual(alone.n_samples, batched.n_samples)
            self.assertEqual(
                alone.rng.bit_generator.state, batched.rng.bit_generator.state
            )
            self.assertEqual(_tree_shape(alone.root), _tree_shape(batched.root))

    def test_blocks_nest_and_bound_their_points(self):
        events = score_sequences.make_stream(seed=11, n_events=300, n_features=3)
        model = MondrianForest(n_estimators=3, subspace_size=3, lambda_=5.0, seed=12)
        for event in events:
            model.score_one(event)
            model.learn_one(event)
        for tree in model.trees:
            pending = [tree.root]
            while pending:
                node = pending.pop()
                if isinstance(node, MondrianBranch):
                    children = (node.left_child, node.right_child)
                    np.testing.assert_array_equal(
                        node.min, np.minimum(children[0].min, children[1].min)
                    )
                    np.testing.assert_array_equal(
                        node.max, np.maximum(children[0].max, children[1].max)
                    )
                    self.assertEqual(node.count, children[0].count + children[1].count)
                    for child in children:
                        self.assertGreater(child.split_time, node.split_time)
                    pending.extend(children)
                else:
                    self.assertEqual(node.min.dtype, np.float64)

    def test_failed_rebuild_keeps_the_trees_after_the_failing_one(self):
        events = score_sequences.make_stream(seed=13, n_events=8, n_features=2)
        model = MondrianForest(n_estimators=4, subspace_size=2, window_size=4, seed=14)
        for event in events[:7]:
            model.learn_one(event)
        expected = copy.deepcopy(model)

        failing = model.trees[1]
        calls = []
        insert = failing._extend_block

        def fail_on_third_rebuild_point(*args):
            calls.append(None)
            # One call learns the eighth event, the next ones rebuild.
            if len(calls) == 4:
                raise OverflowError("range exceeds valid bounds")
            insert(*args)

        failing._extend_block = fail_on_third_rebuild_point
        with self.assertRaises(OverflowError):
            model.learn_one(events[7])
        del failing._extend_block

        # Rebuilding one tree after another stops inside the failing tree.
        values = np.array([events[7][name] for name in sorted(events[7])])
        for tree in expected.trees:
            tree.learn_one_from_global(values)
        window = [*expected._window, values][-4:]
        for index, tree in enumerate(expected.trees[:2]):
            tree.root = None
            tree.n_samples = 0
            for point in window[: 4 if index == 0 else 2]:
                tree.learn_one_from_global(point)

        self.assertEqual(model.n_samples, 8)
        for alone, batched in zip(expected.trees, model.trees, strict=True):
            self.assertEqual(alone.n_samples, batched.n_samples)
            self.assertEqual(
                alone.rng.bit_generator.state, batched.rng.bit_generator.state
            )
            self.assertEqual(_tree_shape(alone.root), _tree_shape(batched.root))

    def test_pickles_leave_out_the_plan(self):
        events = score_sequences.make_stream(seed=15, n_events=60, n_features=3)
        model = MondrianForest(n_estimators=3, subspace_size=2, seed=16)
        for event in events:
            model.learn_one(event)
        before = pickle.dumps(model)
        model.score_one(events[0])
        self.assertIsNotNone(model._planned)
        self.assertEqual(pickle.dumps(model), before)
        restored = pickle.loads(pickle.dumps(model))
        self.assertIsNone(restored._planned)

    def test_default_forest_speed_smoke(self):
        events = score_sequences.make_stream(seed=1, n_events=600, n_features=8)
        model = MondrianForest(seed=1)

        start = time.perf_counter()
        score_sequences.score_then_learn(model, events)

        # Generous bound against large regressions; typical runs take ~2 s.
        self.assertLess(time.perf_counter() - start, 20.0)


if __name__ == "__main__":
    unittest.main()
