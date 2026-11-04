"""Unit tests for Online Isolation Forest - Edge cases and initialization only.

Real dataset tests are in tests/integration/test_iforest_models.py
"""

import time
import unittest
from unittest.mock import patch

import numpy as np
import pytest

from aberrant.model.iforest._online_tree import (
    OnlineIsolationTree,
    _Bounds,
    _EmptyLeaf,
    _OnlineBranch,
    _OnlineLeaf,
    _OnlineNode,
)
from aberrant.model.iforest.online import OnlineIsolationForest
from tests import score_sequences


class TestOnlineIsolationForestEdgeCases(unittest.TestCase):
    """Test Online Isolation Forest edge cases and initialization."""

    def create_model(self):
        return OnlineIsolationForest(
            num_trees=3,  # Small for fast testing
            max_leaf_samples=8,
            tree_type="fixed",
            subsample=1.0,
            window_size=50,
            branching_factor=2,
            metric="axisparallel",
            n_jobs=1,
            seed=42,
        )

    def setUp(self):
        self.model = self.create_model()

    def test_initialization_valid_parameters(self):
        """Test initialization with valid parameters."""
        model = OnlineIsolationForest(
            num_trees=5,
            max_leaf_samples=16,
            tree_type="fixed",
            subsample=0.8,
            window_size=256,
            branching_factor=3,
            metric="axisparallel",
            n_jobs=1,
            seed=42,
        )

        self.assertEqual(model.num_trees, 5)
        self.assertEqual(model.max_leaf_samples, 16)
        self.assertEqual(model.tree_type, "fixed")
        self.assertEqual(model.subsample, 0.8)
        self.assertEqual(model.window_size, 256)
        self.assertEqual(model.branching_factor, 3)
        self.assertEqual(model.seed, 42)

    def test_seed_is_reproducible_and_does_not_mutate_global_rng(self):
        np.random.seed(123)
        expected = np.random.random(5)
        np.random.seed(123)

        model_1 = OnlineIsolationForest(
            num_trees=3,
            max_leaf_samples=4,
            window_size=20,
            seed=7,
        )
        model_2 = OnlineIsolationForest(
            num_trees=3,
            max_leaf_samples=4,
            window_size=20,
            seed=7,
        )
        for i in range(30):
            point = {"x": float(i % 7), "y": float((i * 3) % 11)}
            model_1.learn_one(point)
            model_2.learn_one(point)

        self.assertEqual(
            model_1.score_one({"x": 2.0, "y": 4.0}),
            model_2.score_one({"x": 2.0, "y": 4.0}),
        )
        np.testing.assert_array_equal(np.random.random(5), expected)

    def test_initialization_invalid_parameters(self):
        """Test initialization with invalid parameters raises errors."""
        # Invalid num_trees
        with self.assertRaises(ValueError):
            OnlineIsolationForest(num_trees=0)

        # Invalid max_leaf_samples
        with self.assertRaises(ValueError):
            OnlineIsolationForest(max_leaf_samples=0)

        # Invalid type
        with self.assertRaises(ValueError):
            OnlineIsolationForest(tree_type="invalid_type")

        # Invalid subsample
        with self.assertRaises(ValueError):
            OnlineIsolationForest(subsample=0.0)

        with self.assertRaises(ValueError):
            OnlineIsolationForest(subsample=1.5)

        # Invalid branching_factor
        with self.assertRaises(ValueError):
            OnlineIsolationForest(branching_factor=1)

        with self.assertRaisesRegex(ValueError, "window_size must be positive"):
            OnlineIsolationForest(window_size=0)

        with self.assertRaisesRegex(ValueError, "metric must be 'axisparallel'"):
            OnlineIsolationForest(metric="unsupported")

        with self.assertRaisesRegex(ValueError, "n_jobs must be -1 or a positive"):
            OnlineIsolationForest(n_jobs=0)

    def test_empty_dict_handling(self):
        """Test handling of empty feature dictionary."""
        model = self.create_model()

        # Test with empty dict - should handle gracefully
        try:
            model.learn_one({})
            score = model.score_one({})
            self.assertEqual(score, 0.0)
        except (ValueError, KeyError):
            # Acceptable to reject empty dict
            pass

    def test_n_jobs_minus_one_uses_all_available_cpus(self):
        with patch("aberrant.model.iforest.online.os.cpu_count", return_value=3):
            model = OnlineIsolationForest(num_trees=2, n_jobs=-1, seed=42)

        self.assertEqual(model.n_jobs, -1)
        self.assertEqual(model._max_workers, 3)

    def test_nan_inf_handling(self):
        """Test handling of NaN and Inf values."""
        model = self.create_model()

        # Test with NaN values
        nan_point = {"feature1": float("nan"), "feature2": 1.0}
        try:
            model.learn_one(nan_point)
            score = model.score_one(nan_point)
            # If it doesn't raise exception, score should be valid
            self.assertFalse(np.isnan(score))
        except (ValueError, RuntimeError):
            # Acceptable to reject NaN values
            pass

        # Test with Inf values
        inf_point = {"feature1": float("inf"), "feature2": 1.0}
        try:
            model.learn_one(inf_point)
            score = model.score_one(inf_point)
            self.assertFalse(np.isnan(score))
        except (ValueError, RuntimeError, OverflowError):
            # Acceptable to reject Inf values
            pass

    def test_window_size_boundary(self):
        """Test window size boundary behavior."""
        model = OnlineIsolationForest(
            num_trees=2,
            max_leaf_samples=4,
            tree_type="fixed",
            window_size=10,  # Very small window for testing
            n_jobs=1,
        )

        # Add points up to window size
        for i in range(10):
            point = {"feature1": float(i), "feature2": float(i * 2)}
            model.learn_one(point)

        # Verify window size constraint
        self.assertLessEqual(model.data_size, 10)

        # Add more points (should trigger window sliding)
        for i in range(5):
            point = {"feature1": float(i + 10), "feature2": float((i + 10) * 2)}
            model.learn_one(point)

        # Window size should still be constrained
        self.assertLessEqual(model.data_size, 10)

        # Model should still function
        test_point = {"feature1": 99.0, "feature2": 198.0}
        score = model.score_one(test_point)
        self.assertIsInstance(score, (int, float))
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_batch_input_is_not_retained_by_reference(self):
        model = OnlineIsolationForest(
            num_trees=2,
            max_leaf_samples=4,
            window_size=10,
            n_jobs=1,
            seed=42,
        )
        batch = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        model.learn_batch(batch)

        batch[:] = 99.0

        np.testing.assert_array_equal(model.data_window[0], [1.0, 2.0])
        np.testing.assert_array_equal(model.data_window[1], [3.0, 4.0])

    def test_type_variants(self):
        """Test different multiplier types (fixed vs adaptive)."""
        for type_name in ["fixed", "adaptive"]:
            with self.subTest(type=type_name):
                model = OnlineIsolationForest(
                    num_trees=2,
                    max_leaf_samples=8,
                    tree_type=type_name,
                    window_size=50,
                    n_jobs=1,
                )

                # Should initialize without error
                self.assertEqual(model.tree_type, type_name)

                # Basic functionality test
                point = {"feature1": 1.0, "feature2": 2.0}
                model.learn_one(point)
                score = model.score_one(point)

                self.assertIsInstance(score, (int, float))
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)

    def test_feature_consistency(self):
        """Test behavior with inconsistent feature sets."""
        model = self.create_model()

        # Train with one feature set
        point1 = {"feature1": 1.0, "feature2": 2.0}
        model.learn_one(point1)

        # Score with different feature set should fail fast
        point2 = {"feature3": 3.0, "feature4": 4.0}
        with self.assertRaises(ValueError):
            model.score_one(point2)

    def test_feature_order_stability(self):
        """Test feature-order invariance across different dict insertion orders."""
        model = self.create_model()
        p1 = {"b": 2.0, "a": 1.0}
        p2 = {"a": 1.0, "b": 2.0}
        p3 = {"b": 3.0, "a": 4.0}

        model.learn_one(p1)
        model.learn_one(p2)

        score1 = model.score_one(p1)
        score2 = model.score_one(p2)
        score3 = model.score_one(p3)

        self.assertIsInstance(score1, float)
        self.assertIsInstance(score2, float)
        self.assertIsInstance(score3, float)
        self.assertAlmostEqual(score1, score2, places=12)

    def test_batch_then_mismatched_dict_does_not_corrupt_feature_width(self):
        """Test dict conversion after batch training preserves learned feature width."""
        model = self.create_model()
        batch = np.array(
            [[1.0, 2.0, 3.0, 4.0], [1.5, 2.5, 3.5, 4.5]],
            dtype=np.float32,
        )
        model.learn_batch(batch)
        self.assertEqual(model._n_features, 4)

        with self.assertRaises(ValueError):
            model.score_one({"a": 1.0, "b": 2.0, "c": 3.0})

        # Ensure failed dict scoring does not mutate learned feature count.
        self.assertEqual(model._n_features, 4)
        scores = model.score_batch(np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32))
        self.assertEqual(scores.shape, (1,))

    def test_extreme_values(self):
        """Test with extreme numeric values."""
        model = self.create_model()

        # Very large values
        large_point = {"feature1": 1e10, "feature2": -1e10}
        try:
            model.learn_one(large_point)
            score = model.score_one(large_point)
            self.assertIsInstance(score, (int, float))
        except (ValueError, OverflowError):
            # Acceptable to have issues with extreme values
            pass

        # Very small values
        small_point = {"feature1": 1e-10, "feature2": -1e-10}
        try:
            model.learn_one(small_point)
            score = model.score_one(small_point)
            self.assertIsInstance(score, (int, float))
        except ValueError:
            # Acceptable to have issues with extreme values
            pass


if __name__ == "__main__":
    unittest.main()


def test_subsampled_tree_forgets_exactly_the_learned_membership():
    tree = OnlineIsolationTree(
        max_leaf_samples=32,
        tree_type="fixed",
        subsample=0.4,
        branching_factor=2,
        data_size=0,
        rng=np.random.default_rng(42),
    )
    batch = np.arange(40.0).reshape(-1, 1)
    tree.learn(batch)
    retained = tree.data_size
    assert 0 < retained < len(batch)
    tree.unlearn(batch[:13])
    tree.unlearn(batch[13:])
    assert tree.data_size == 0
    assert tree.root.data_size == 0


def test_subsampled_forest_population_stays_within_window():
    model = OnlineIsolationForest(num_trees=2, window_size=8, subsample=0.5, seed=42)
    for i in range(1000):
        model.learn_one({"x": float(i)})
        for tree in model.trees:
            assert 0 <= tree.data_size <= model.window_size


def test_score_sequences_match_recorded_reference():
    for case in score_sequences.load("online_isolation_forest"):
        model = OnlineIsolationForest(**case["params"])
        scores = score_sequences.replay(model, case)
        # The tolerance only absorbs last-bit libm differences between
        # platforms; the recording platform reproduces every bit.
        np.testing.assert_allclose(
            scores, case["scores"], rtol=1e-12, atol=0, err_msg=case["name"]
        )


def _structure(node: _OnlineNode | None) -> object:
    """Describe a subtree completely, bounds included, for comparisons."""
    if node is None:
        return None
    if isinstance(node, _EmptyLeaf):
        return ("empty", node.depth, node.node_index)
    if isinstance(node, _OnlineLeaf):
        return (
            "leaf",
            node.data_size,
            node.depth,
            node.node_index,
            node.bounds.minimum.tolist(),
            node.bounds.maximum.tolist(),
        )
    return (
        "branch",
        node.data_size,
        node.depth,
        node.node_index,
        node.feature,
        node.split_values,
        OnlineIsolationTree.node_bounds(node).minimum.tolist(),
        OnlineIsolationTree.node_bounds(node).maximum.tolist(),
        [_structure(child) for child in node.children],
    )


@pytest.mark.parametrize(
    "params",
    [
        {"max_leaf_samples": 4, "window_size": 30},
        {"max_leaf_samples": 2, "tree_type": "fixed", "window_size": 25},
        {"max_leaf_samples": 3, "subsample": 0.6, "window_size": 40},
        {"max_leaf_samples": 1, "branching_factor": 3, "window_size": 20},
    ],
)
def test_single_rows_update_trees_like_one_row_batches(params):
    events = score_sequences.make_stream(seed=3, n_events=150, n_features=3)
    rows = np.array([list(event.values()) for event in events], dtype=np.float32)
    by_row = OnlineIsolationForest(num_trees=4, seed=11, **params)
    by_batch = OnlineIsolationForest(num_trees=4, seed=11, **params)

    for event, row in zip(events, rows, strict=True):
        assert by_row.score_one(event) == by_batch.score_batch(row[np.newaxis])[0]
        by_row.learn_one(event)
        by_batch.learn_batch(row[np.newaxis])

        assert by_row.normalization_factor == by_batch.normalization_factor
        for tree, reference in zip(by_row.trees, by_batch.trees, strict=True):
            assert tree.data_size == reference.data_size
            assert tree.next_node_index == reference.next_node_index
            assert tree._sample_membership == reference._sample_membership
            assert _structure(tree.root) == _structure(reference.root)
    np.testing.assert_array_equal(by_row.data_window, by_batch.data_window)


@pytest.mark.parametrize("by_row", [True, False])
def test_collapsing_branch_keeps_bounds_of_leaf_split_since_last_unlearn(by_row):
    tree = OnlineIsolationTree(
        max_leaf_samples=3,
        tree_type="fixed",
        subsample=1.0,
        branching_factor=2,
        data_size=100,
        rng=np.random.default_rng(0),
    )
    leaf = _OnlineLeaf(
        data_size=2,
        depth=1,
        node_index=0,
        bounds=_Bounds.from_data(np.array([[0.0], [1.0]])),
    )
    far_leaf = _OnlineLeaf(
        data_size=1,
        depth=1,
        node_index=1,
        bounds=_Bounds.from_data(np.array([[7.0]])),
    )
    # Unlearning can leave a branch with fewer points than its children hold.
    tree.root = _OnlineBranch(
        data_size=2,
        depth=0,
        node_index=2,
        children=[leaf, far_leaf],
        feature=0,
        split_values=(5.0,),
    )

    # The leaf fills up and is rebuilt from points sampled inside [0, 1].
    if by_row:
        point = _Bounds.from_data(np.array([[0.5]]))
        tree.learn_point(point, [0.5])
    else:
        tree.learn(np.array([[0.5]]))
    assert isinstance(tree.root, _OnlineBranch)
    rebuilt = tree.root.children[0]
    assert isinstance(rebuilt, _OnlineBranch)
    sampled = OnlineIsolationTree.node_bounds(rebuilt)
    assert sampled.minimum[0] > 0.0
    assert sampled.maximum[0] < 1.0

    # One unlearned point drops the root below the split threshold. The leaf it
    # becomes still bounds everything the root covered, as in 1.4.0.
    if by_row:
        tree.unlearn_point([7.0])
    else:
        tree.unlearn(np.array([[7.0]]))
    assert isinstance(tree.root, _OnlineLeaf)
    assert tree.root.data_size == 2
    assert tree.root.bounds.minimum.tolist() == [0.0]
    assert tree.root.bounds.maximum.tolist() == [7.0]


@pytest.mark.filterwarnings("ignore:overflow encountered in cast:RuntimeWarning")
def test_single_rows_reject_float32_overflow():
    model = OnlineIsolationForest(num_trees=2, window_size=8, seed=1)
    with pytest.raises(ValueError, match="only finite values"):
        model.learn_one({"a": 1e39, "b": 0.0})
    assert model._n_features is None
    assert model.data_size == 0
    assert all(len(tree._sample_membership) == 0 for tree in model.trees)

    model.learn_one({"a": 1.0, "b": 0.0})
    with pytest.raises(ValueError, match="only finite values"):
        model.score_one({"a": -1e39, "b": 0.0})
    assert model.data_size == 1


def test_model_processes_two_thousand_events_quickly():
    events = score_sequences.make_stream(seed=1, n_events=2000, n_features=8)
    model = OnlineIsolationForest(num_trees=25, window_size=256, seed=1)

    start = time.perf_counter()
    score_sequences.score_then_learn(model, events)

    # Generous bound against large regressions; typical runs take ~0.3 s.
    assert time.perf_counter() - start < 3.0
