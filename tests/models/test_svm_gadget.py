"""Tests for GADGET SVM anomaly detection model."""

import unittest

import numpy as np

from aberrant.model.svm.gadget import (
    GraphGatedOneClassSVM,
    IncrementalOneClassSVM,
)
from tests.utils import DataGenerator, TestAssertions


class TestIncrementalOneClassSVM(unittest.TestCase):
    """Test suite for the underlying IncrementalOneClassSVM component."""

    def test_initialization_with_default_params(self):
        """Test SVM initialization with default parameters."""
        svm = IncrementalOneClassSVM()

        self.assertIsNone(svm.w)
        self.assertEqual(svm.rho, 0.0)
        self.assertEqual(svm.learning_rate, 0.01)
        self.assertEqual(svm.nu, 0.5)
        self.assertEqual(svm.lambda_reg, 0.01)

    def test_initialization_with_custom_params(self):
        """Test SVM initialization with custom parameters."""
        svm = IncrementalOneClassSVM(learning_rate=0.1, nu=0.8, lambda_reg=0.05)

        self.assertEqual(svm.learning_rate, 0.1)
        self.assertEqual(svm.nu, 0.8)
        self.assertEqual(svm.lambda_reg, 0.05)

    def test_learn_one_initializes_weights(self):
        """Test that learn_one initializes weight vector on first call."""
        svm = IncrementalOneClassSVM()

        # Initially no weights
        self.assertIsNone(svm.w)

        # First learn_one should initialize weights
        x_vec = np.array([1.0, 2.0, 3.0])
        svm.learn_one(x_vec)

        self.assertIsNotNone(svm.w)
        self.assertEqual(svm.w.shape, x_vec.shape)

    def test_score_one_with_uninitialized_weights(self):
        """Test score_one returns 0.0 when weights uninitialized."""
        svm = IncrementalOneClassSVM()

        x_vec = np.array([1.0, 2.0, 3.0])
        score = svm.score_one(x_vec)

        self.assertEqual(score, 0.0)

    def test_learn_and_score_basic_functionality(self):
        """Test basic learning and scoring functionality."""
        svm = IncrementalOneClassSVM(learning_rate=0.1)

        # Train on some data
        training_data = [
            np.array([1.0, 2.0]),
            np.array([2.0, 3.0]),
            np.array([3.0, 4.0]),
        ]

        for x_vec in training_data:
            svm.learn_one(x_vec)

        # Score the same data
        scores = [svm.score_one(x_vec) for x_vec in training_data]

        # All scores should be numeric
        for score in scores:
            self.assertIsInstance(score, (int, float))

    def test_gradient_updates_modify_parameters(self):
        """Test that learning updates modify model parameters."""
        svm = IncrementalOneClassSVM(
            learning_rate=0.5
        )  # High learning rate for visible changes

        x_vec = np.array([1.0, 1.0])

        # Learn first time to initialize
        svm.learn_one(x_vec)
        initial_w = svm.w.copy()

        # Learn multiple times
        for _ in range(10):
            svm.learn_one(x_vec)

        # Parameters should change
        self.assertFalse(
            np.array_equal(initial_w, svm.w), "Weight vector should change"
        )


class TestGADGETSVM(unittest.TestCase):
    """Test suite for GADGET SVM model."""

    def create_model(self) -> GraphGatedOneClassSVM:
        """Create GADGET SVM instance for testing."""
        # Simple linear graph: 0 -> 1 -> 2
        graph = {0: [1], 1: [2], 2: []}
        return GraphGatedOneClassSVM(
            graph=graph, threshold=0.1, learning_rate=0.01
        )

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.data_generator = DataGenerator(seed=42)

    def test_initialization_with_default_graph(self):
        """Test GADGET SVM initialization with default graph."""
        model = GraphGatedOneClassSVM()

        # Should create default graph
        expected_graph = {0: [1], 1: [2], 2: []}
        self.assertEqual(model.graph, expected_graph)

        # Should have SVMs for all nodes
        self.assertIn(0, model.svms)
        self.assertIn(1, model.svms)
        self.assertIn(2, model.svms)

        # Should identify root nodes correctly
        self.assertEqual(model.root_nodes, [0])

    def test_initialization_with_custom_graph(self):
        """Test GADGET SVM initialization with custom graph."""
        # More complex graph: multiple roots and branches
        graph = {0: [2, 3], 1: [3, 4], 2: [], 3: [5], 4: [], 5: []}

        model = GraphGatedOneClassSVM(graph=graph)

        self.assertEqual(model.graph, graph)

        # Should have SVMs for all nodes
        for node in [0, 1, 2, 3, 4, 5]:
            self.assertIn(node, model.svms)

        # Should identify root nodes correctly (0 and 1)
        self.assertCountEqual(model.root_nodes, [0, 1])

    def test_feature_order_consistency(self):
        """Test that feature order is established and maintained."""
        model = GraphGatedOneClassSVM()

        # First data point establishes feature order
        first_point = {"c": 3.0, "a": 1.0, "b": 2.0}
        model.learn_one(first_point)

        # Feature order should be alphabetical (tuple for fast comparison)
        expected_order = ("a", "b", "c")
        self.assertEqual(model._schema.names, expected_order)

        # Subsequent data with same keys should work
        model.learn_one({"b": 5.0, "c": 6.0, "a": 4.0})
        score = model.score_one({"a": 1.0, "b": 2.0, "c": 3.0})

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)

    def test_inconsistent_features_raise_error(self):
        """Test that inconsistent features raise ValueError."""
        model = GraphGatedOneClassSVM()

        # Establish feature order
        model.learn_one({"a": 1.0, "b": 2.0})

        # Try to learn with different features
        with self.assertRaises(ValueError) as context:
            model.learn_one({"a": 1.0, "c": 3.0})  # Different key 'c'

        self.assertIn("Inconsistent feature keys", str(context.exception))

    def test_graph_traversal_with_threshold(self):
        """Test that graph traversal respects threshold parameter."""
        # Create graph where high threshold prevents deep traversal
        graph = {0: [1], 1: [2], 2: [3], 3: []}
        model = GraphGatedOneClassSVM(
            graph=graph, threshold=999.0
        )  # Very high threshold

        # Train on data
        training_data = self.data_generator.generate_streaming_data(n=50, n_features=2)
        for point in training_data:
            model.learn_one(point)

        # Score should still work (uses root nodes)
        test_point = {"feature_0": 1.0, "feature_1": 2.0}
        score = model.score_one(test_point)

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)

    def test_multiple_root_nodes(self):
        """Test GADGET SVM with multiple root nodes."""
        # Graph with two separate components
        graph = {0: [2], 1: [3], 2: [], 3: []}

        model = GraphGatedOneClassSVM(graph=graph)

        # Should identify both 0 and 1 as roots
        self.assertCountEqual(model.root_nodes, [0, 1])

        # Should work with streaming data
        data = self.data_generator.generate_streaming_data(n=50, n_features=2)

        scores = []
        for point in data:
            model.learn_one(point)
            score = model.score_one(point)
            scores.append(score)

        TestAssertions.assert_scores_valid(scores)

    def test_score_before_learning(self):
        """Test that scoring before learning returns 0.0."""
        model = GraphGatedOneClassSVM()

        score = model.score_one({"feature": 1.0})
        self.assertEqual(score, 0.0)

    def test_different_threshold_values(self):
        """Test GADGET SVM behavior with different threshold values."""
        thresholds = [0.0, 0.5, 1.0, 2.0]

        training_data = self.data_generator.generate_streaming_data(n=30, n_features=2)

        for threshold in thresholds:
            with self.subTest(threshold=threshold):
                # Create separate graph for each test to avoid interference
                graph = {0: [1], 1: [2], 2: []}
                model = GraphGatedOneClassSVM(graph=graph, threshold=threshold)

                # Train
                for point in training_data:
                    model.learn_one(point)

                # Score
                test_point = {"feature_0": 1.0, "feature_1": 2.0}
                score = model.score_one(test_point)

                self.assertIsInstance(score, float)
                self.assertGreaterEqual(score, 0.0)

    def test_learning_rate_effect(self):
        """Test that different learning rates affect model behavior."""
        learning_rates = [0.001, 0.01, 0.1]

        training_data = self.data_generator.generate_streaming_data(n=20, n_features=2)

        models = []
        for lr in learning_rates:
            graph = {0: [1], 1: []}  # Simple graph
            model = GraphGatedOneClassSVM(graph=graph, learning_rate=lr)
            models.append(model)

        # Train all models on same data
        for point in training_data:
            for model in models:
                model.learn_one(point.copy())

        # Score test point with all models
        test_point = {"feature_0": 5.0, "feature_1": 5.0}
        scores = [model.score_one(test_point) for model in models]

        # All should produce valid scores
        for score in scores:
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)

    def test_complex_graph_structure(self):
        """Test GADGET SVM with complex graph structure."""
        # Diamond-shaped graph
        graph = {
            0: [1, 2],  # Root splits to two paths
            1: [3],  # Left path
            2: [3],  # Right path
            3: [4],  # Paths converge
            4: [],  # End
        }

        model = GraphGatedOneClassSVM(graph=graph, threshold=0.1)

        # Should handle complex traversal
        data = self.data_generator.generate_streaming_data(n=80, n_features=3)

        scores = []
        for point in data:
            model.learn_one(point)
            score = model.score_one(point)
            scores.append(score)

        TestAssertions.assert_scores_valid(scores)

        # Should have correct root identification
        self.assertEqual(model.root_nodes, [0])

        # All nodes should have SVMs
        for node in [0, 1, 2, 3, 4]:
            self.assertIn(node, model.svms)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        model = GraphGatedOneClassSVM()

        # Test with zero values
        zero_point = {"a": 0.0, "b": 0.0}
        model.learn_one(zero_point)
        score = model.score_one(zero_point)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)

        # Test with negative values
        negative_point = {"a": -5.0, "b": -10.0}
        model.learn_one(negative_point)
        score = model.score_one(negative_point)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)

        # Test with very large values
        large_point = {"a": 1000.0, "b": 2000.0}
        model.learn_one(large_point)
        score = model.score_one(large_point)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)


class TestGADGETSVMSpecialCases(unittest.TestCase):
    """Test special cases and configurations for GADGET SVM."""

    def test_single_node_graph(self):
        """Test GADGET SVM with single node graph."""
        graph = {0: []}
        model = GraphGatedOneClassSVM(graph=graph)

        self.assertEqual(model.root_nodes, [0])
        self.assertEqual(len(model.svms), 1)

        # Should work normally
        model.learn_one({"feature": 1.0})
        score = model.score_one({"feature": 2.0})

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)

    def test_empty_graph(self):
        """Test GADGET SVM with empty graph."""
        graph = {}
        model = GraphGatedOneClassSVM(graph=graph)

        # Should handle empty graph gracefully
        self.assertEqual(len(model.svms), 0)
        self.assertEqual(model.root_nodes, [])

        # Score before learning should return 0
        score = model.score_one({"feature": 1.0})
        self.assertEqual(score, 0.0)

    def test_disconnected_nodes_in_graph(self):
        """Test graph that has nodes mentioned only in neighbor lists."""
        graph = {0: [1, 2]}  # Nodes 1 and 2 are not keys but are neighbors

        model = GraphGatedOneClassSVM(graph=graph)

        # Should create entries for all nodes
        expected_graph = {0: [1, 2], 1: [], 2: []}
        self.assertEqual(model.graph, expected_graph)

        # Should have SVMs for all nodes
        for node in [0, 1, 2]:
            self.assertIn(node, model.svms)

        # Should identify correct root
        self.assertEqual(model.root_nodes, [0])


if __name__ == "__main__":
    unittest.main()
