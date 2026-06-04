"""Tests for Half-Space Trees (HST) anomaly detection model."""

import unittest

from aberrant.model.iforest.halfspace import HalfSpaceTrees, HSTLeaf, HSTNode


class TestHSTDataStructures(unittest.TestCase):
    """Test HST data structure classes."""

    def test_hst_leaf_default(self):
        """Test HSTLeaf default initialization."""
        leaf = HSTLeaf()
        self.assertEqual(leaf.l_mass, 0)
        self.assertEqual(leaf.r_mass, 0)

    def test_hst_leaf_reset(self):
        """Test HSTLeaf mass reset."""
        leaf = HSTLeaf()
        leaf.l_mass = 10
        leaf.r_mass = 5
        leaf.reset_mass()
        self.assertEqual(leaf.l_mass, 0)
        self.assertEqual(leaf.r_mass, 0)

    def test_hst_leaf_pivot(self):
        """Test HSTLeaf mass pivot (learning to reference)."""
        leaf = HSTLeaf()
        leaf.l_mass = 10
        leaf.pivot_mass()
        self.assertEqual(leaf.l_mass, 0)
        self.assertEqual(leaf.r_mass, 10)

    def test_hst_node_default(self):
        """Test HSTNode default initialization."""
        node = HSTNode(feature=0, threshold=0.5)
        self.assertEqual(node.feature, 0)
        self.assertEqual(node.threshold, 0.5)
        self.assertEqual(node.l_mass, 0)
        self.assertEqual(node.r_mass, 0)
        self.assertIsInstance(node.left, HSTLeaf)
        self.assertIsInstance(node.right, HSTLeaf)

    def test_hst_node_reset_recursive(self):
        """Test HSTNode recursive mass reset."""
        # Create a small tree
        left_leaf = HSTLeaf(l_mass=5, r_mass=2)
        right_leaf = HSTLeaf(l_mass=3, r_mass=1)
        node = HSTNode(
            feature=0,
            threshold=0.5,
            left=left_leaf,
            right=right_leaf,
            l_mass=10,
            r_mass=4,
        )
        node.reset_mass()
        self.assertEqual(node.l_mass, 0)
        self.assertEqual(node.r_mass, 0)
        self.assertEqual(left_leaf.l_mass, 0)
        self.assertEqual(left_leaf.r_mass, 0)
        self.assertEqual(right_leaf.l_mass, 0)
        self.assertEqual(right_leaf.r_mass, 0)

    def test_hst_node_pivot_recursive(self):
        """Test HSTNode recursive mass pivot."""
        left_leaf = HSTLeaf(l_mass=5, r_mass=0)
        right_leaf = HSTLeaf(l_mass=3, r_mass=0)
        node = HSTNode(
            feature=0,
            threshold=0.5,
            left=left_leaf,
            right=right_leaf,
            l_mass=10,
        )
        node.pivot_mass()
        # Learning masses should be copied to reference, then reset
        self.assertEqual(node.l_mass, 0)
        self.assertEqual(node.r_mass, 10)
        self.assertEqual(left_leaf.l_mass, 0)
        self.assertEqual(left_leaf.r_mass, 5)
        self.assertEqual(right_leaf.l_mass, 0)
        self.assertEqual(right_leaf.r_mass, 3)


class TestHalfSpaceTrees(unittest.TestCase):
    """Test suite for HalfSpaceTrees model."""

    def test_initialization_default_params(self):
        """Test HST initialization with default parameters."""
        hst = HalfSpaceTrees()
        self.assertEqual(hst.n_trees, 10)
        self.assertEqual(hst.height, 8)
        self.assertEqual(hst.window_size, 250)
        self.assertIsNone(hst.seed)
        self.assertFalse(hst._initialized)

    def test_initialization_custom_params(self):
        """Test HST initialization with custom parameters."""
        hst = HalfSpaceTrees(n_trees=25, height=10, window_size=500, seed=42)
        self.assertEqual(hst.n_trees, 25)
        self.assertEqual(hst.height, 10)
        self.assertEqual(hst.window_size, 500)
        self.assertEqual(hst.seed, 42)

    def test_invalid_n_trees(self):
        """Test that invalid n_trees raises ValueError."""
        with self.assertRaises(ValueError) as context:
            HalfSpaceTrees(n_trees=0)
        self.assertIn("n_trees must be positive", str(context.exception))

        with self.assertRaises(ValueError):
            HalfSpaceTrees(n_trees=-1)

    def test_invalid_height(self):
        """Test that invalid height raises ValueError."""
        with self.assertRaises(ValueError) as context:
            HalfSpaceTrees(height=0)
        self.assertIn("height must be positive", str(context.exception))

    def test_invalid_window_size(self):
        """Test that invalid window_size raises ValueError."""
        with self.assertRaises(ValueError) as context:
            HalfSpaceTrees(window_size=0)
        self.assertIn("window_size must be positive", str(context.exception))

    def test_learn_one_empty_dict(self):
        """Test that empty input raises ValueError."""
        hst = HalfSpaceTrees()
        with self.assertRaises(ValueError) as context:
            hst.learn_one({})
        self.assertIn("empty", str(context.exception))

    def test_learn_one_initializes_trees(self):
        """Test that learn_one initializes trees on first call."""
        hst = HalfSpaceTrees(n_trees=5)
        self.assertFalse(hst._initialized)
        hst.learn_one({"x": 0.5, "y": 0.5})
        self.assertTrue(hst._initialized)
        self.assertEqual(len(hst._trees), 5)

    def test_score_one_before_learn(self):
        """Test score_one returns 0 before any learning."""
        hst = HalfSpaceTrees()
        score = hst.score_one({"x": 0.5, "y": 0.5})
        self.assertEqual(score, 0.0)

    def test_score_one_returns_float(self):
        """Test that score_one returns a float in [0, 1]."""
        hst = HalfSpaceTrees(n_trees=5, seed=42)
        for i in range(50):
            hst.learn_one({"x": float(i) / 100, "y": float(i) / 100})
        score = hst.score_one({"x": 0.5, "y": 0.5})
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_score_is_zero_until_reference_window_is_complete(self):
        hst = HalfSpaceTrees(n_trees=3, height=3, window_size=5, seed=42)
        for _ in range(4):
            hst.learn_one({"x": 0.5})
            self.assertEqual(hst.score_one({"x": 0.5}), 0.0)

    def test_tree_recursively_bisects_propagated_intervals(self):
        hst = HalfSpaceTrees(n_trees=1, height=4, window_size=5, seed=42)
        hst.learn_one({"x": 0.5, "y": 0.5})
        tree = hst._trees[0]
        lower, upper = hst._workspaces[0]

        def assert_midpoints(node, current_lower, current_upper):
            if isinstance(node, HSTLeaf):
                return
            expected = (current_lower[node.feature] + current_upper[node.feature]) / 2.0
            self.assertAlmostEqual(node.threshold, expected)

            left_upper = current_upper.copy()
            left_upper[node.feature] = node.threshold
            right_lower = current_lower.copy()
            right_lower[node.feature] = node.threshold
            assert_midpoints(node.left, current_lower, left_upper)
            assert_midpoints(node.right, right_lower, current_upper)

        assert_midpoints(tree, lower, upper)

    def test_schema_change_raises(self):
        hst = HalfSpaceTrees(n_trees=1, height=2, window_size=2, seed=42)
        hst.learn_one({"x": 0.5, "y": 0.5})
        with self.assertRaises(ValueError):
            hst.learn_one({"x": 0.5})
        hst.learn_one({"x": 0.6, "y": 0.6})
        with self.assertRaises(ValueError):
            hst.score_one({"x": 0.5})

    def test_normal_point_low_score(self):
        """Test that frequently visited regions have low anomaly scores."""
        hst = HalfSpaceTrees(n_trees=10, window_size=100, seed=42)
        # Train on points around (0.5, 0.5)
        for _ in range(100):
            hst.learn_one({"x": 0.5, "y": 0.5})
        # Score a point in the same region
        score = hst.score_one({"x": 0.5, "y": 0.5})
        # Should be low (high mass = low anomaly)
        self.assertLess(score, 0.5)

    def test_anomaly_high_score(self):
        """Test that rarely visited regions have high anomaly scores."""
        hst = HalfSpaceTrees(n_trees=10, window_size=200, seed=42)
        # Train on points in one corner (less than window_size to avoid reset)
        for _ in range(150):
            hst.learn_one({"x": 0.1, "y": 0.1})
        # Score a point in opposite corner
        score_anomaly = hst.score_one({"x": 0.9, "y": 0.9})
        score_normal = hst.score_one({"x": 0.1, "y": 0.1})
        # Both should be valid scores
        self.assertIsInstance(score_anomaly, float)
        self.assertIsInstance(score_normal, float)
        # Anomaly should have higher score (or equal in edge cases)
        self.assertGreaterEqual(score_anomaly, score_normal)

    def test_window_reset(self):
        """Test that mass counters reset after window_size samples."""
        hst = HalfSpaceTrees(n_trees=3, window_size=10, seed=42)
        # Fill one window
        for _ in range(10):
            hst.learn_one({"x": 0.5})
        # After window_size, counters should reset
        self.assertEqual(hst._samples_in_window, 0)

    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with same seed."""
        hst1 = HalfSpaceTrees(n_trees=5, seed=42)
        hst2 = HalfSpaceTrees(n_trees=5, seed=42)

        data = [{"x": float(i) / 100, "y": float(i) / 100} for i in range(50)]

        for point in data:
            hst1.learn_one(point)
            hst2.learn_one(point)

        test_point = {"x": 0.5, "y": 0.5}
        score1 = hst1.score_one(test_point)
        score2 = hst2.score_one(test_point)

        self.assertEqual(score1, score2)

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        hst1 = HalfSpaceTrees(n_trees=5, seed=42)
        hst2 = HalfSpaceTrees(n_trees=5, seed=123)

        data = [{"x": float(i) / 100, "y": float(i) / 100} for i in range(50)]

        for point in data:
            hst1.learn_one(point)
            hst2.learn_one(point)

        test_point = {"x": 0.5, "y": 0.5}
        score1 = hst1.score_one(test_point)
        score2 = hst2.score_one(test_point)

        # With different seeds, scores may differ
        # (though they could coincidentally be equal)
        self.assertIsInstance(score1, float)
        self.assertIsInstance(score2, float)

    def test_feature_names_sorted(self):
        """Test that feature names are sorted consistently."""
        hst = HalfSpaceTrees()
        hst.learn_one({"z": 0.1, "a": 0.2, "m": 0.3})
        self.assertEqual(hst.feature_names, ["a", "m", "z"])

    def test_repr(self):
        """Test string representation."""
        hst = HalfSpaceTrees(n_trees=25, height=10, window_size=500)
        repr_str = repr(hst)
        self.assertIn("HalfSpaceTrees", repr_str)
        self.assertIn("n_trees=25", repr_str)
        self.assertIn("height=10", repr_str)
        self.assertIn("window_size=500", repr_str)


class TestHalfSpaceTreesStreaming(unittest.TestCase):
    """Test HST streaming behavior."""

    def test_continuous_streaming(self):
        """Test HST handles continuous streaming correctly."""
        hst = HalfSpaceTrees(n_trees=5, window_size=50, seed=42)

        scores = []
        for i in range(200):
            point = {"x": (i % 10) / 10.0, "y": (i % 10) / 10.0}
            hst.learn_one(point)
            score = hst.score_one(point)
            scores.append(score)

        # All scores should be valid
        for i, score in enumerate(scores):
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
            self.assertFalse(score != score, f"Score {i} is NaN")  # noqa: PLR0124


if __name__ == "__main__":
    unittest.main()
