"""Tests for k-Nearest Neighbors (kNN) anomaly detection model."""

import unittest
from unittest.mock import Mock

import pytest

pytest.importorskip("faiss")

from aberrant.model.distance.knn import KNN
from aberrant.utils.similar.faiss_engine import FaissSimilaritySearchEngine
from tests.utils import DataGenerator


class TestKNN(unittest.TestCase):
    """Test suite for KNN model."""

    def create_model(self) -> KNN:
        """Create KNN instance for testing."""
        engine = FaissSimilaritySearchEngine(window_size=100, warm_up=5)
        return KNN(k=5, similarity_engine=engine)

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.data_generator = DataGenerator(seed=42)

    def test_initialization_valid_params(self):
        """Test KNN initialization with valid parameters."""
        engine = FaissSimilaritySearchEngine(window_size=50, warm_up=3)
        model = KNN(k=3, similarity_engine=engine)

        self.assertEqual(model.k, 3)
        self.assertIs(model.engine, engine)

    def test_initialization_invalid_k(self):
        """Test that invalid k values raise ValueError."""
        engine = FaissSimilaritySearchEngine(window_size=50, warm_up=3)

        with self.assertRaises(ValueError) as context:
            KNN(k=0, similarity_engine=engine)
        self.assertIn("k must be a positive integer", str(context.exception))

        with self.assertRaises(ValueError) as context:
            KNN(k=-1, similarity_engine=engine)
        self.assertIn("k must be a positive integer", str(context.exception))

    def test_learn_one_calls_engine_append(self):
        """Test that learn_one properly delegates to similarity engine."""
        # Create mock engine to verify calls
        mock_engine = Mock()
        model = KNN(k=3, similarity_engine=mock_engine)

        test_point = {"feature1": 1.0, "feature2": 2.0}
        model.learn_one(test_point)

        mock_engine.append.assert_called_once_with(test_point)

    def test_score_one_calls_engine_search(self):
        """Test that score_one properly delegates to similarity engine."""
        # Create mock engine
        mock_engine = Mock()
        mock_engine.search.return_value = 0.5

        model = KNN(k=3, similarity_engine=mock_engine)
        test_point = {"feature1": 1.0, "feature2": 2.0}

        score = model.score_one(test_point)

        mock_engine.search.assert_called_once_with(test_point, n_neighbors=3)
        self.assertEqual(score, 0.5)

    def test_with_faiss_engine_basic_functionality(self):
        """Test KNN with real FAISS engine for basic functionality."""
        engine = FaissSimilaritySearchEngine(window_size=100, warm_up=5)
        model = KNN(k=3, similarity_engine=engine)

        # Generate training data
        training_data = self.data_generator.generate_streaming_data(n=20, n_features=2)

        # Learn training data
        for point in training_data:
            model.learn_one(point)

        # Test scoring
        test_point = {"feature_0": 0.0, "feature_1": 0.0}
        score = model.score_one(test_point)

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)

    def test_faiss_engine_returns_mean_l2_distance(self):
        engine = FaissSimilaritySearchEngine(window_size=4, warm_up=2)
        engine.append({"x": 0.0})
        engine.append({"x": 3.0})

        # IndexFlatL2 reports squared distances [0, 9]; the public engine
        # contract is the mean Euclidean distance [0, 3].
        self.assertAlmostEqual(engine.search({"x": 0.0}, n_neighbors=2), 1.5)

    def test_faiss_engine_owns_window_samples(self):
        engine = FaissSimilaritySearchEngine(window_size=2, warm_up=1)
        point = {"x": 1.0}
        engine.append(point)
        point["x"] = 99.0

        self.assertEqual(engine.window[0]["x"], 1.0)

    def test_with_insufficient_data(self):
        """Test behavior when there's insufficient data for nearest neighbors."""
        engine = FaissSimilaritySearchEngine(window_size=100, warm_up=10)
        model = KNN(k=5, similarity_engine=engine)

        # Add less data than warm_up requirement
        for i in range(5):
            model.learn_one({"feature": float(i)})

        # Should return 0.0 when insufficient data
        score = model.score_one({"feature": 999.0})
        self.assertEqual(score, 0.0)

    def test_k_parameter_effect(self):
        """Test that different k values produce different behaviors."""
        engine1 = FaissSimilaritySearchEngine(window_size=100, warm_up=10)
        engine2 = FaissSimilaritySearchEngine(window_size=100, warm_up=10)

        model_k1 = KNN(k=1, similarity_engine=engine1)
        model_k5 = KNN(k=5, similarity_engine=engine2)

        # Generate same training data for both models
        training_data = self.data_generator.generate_streaming_data(n=50, n_features=2)

        for point in training_data:
            model_k1.learn_one(point.copy())
            model_k5.learn_one(point.copy())

        # Test on same points - should generally give different scores
        test_data = self.data_generator.generate_streaming_data(n=20, n_features=2)

        scores_k1 = []
        scores_k5 = []

        for point in test_data:
            score1 = model_k1.score_one(point)
            score5 = model_k5.score_one(point)
            scores_k1.append(score1)
            scores_k5.append(score5)

        # Should produce different score distributions
        # (k=1 uses only nearest neighbor, k=5 uses average of 5)
        different_count = sum(
            1 for s1, s5 in zip(scores_k1, scores_k5, strict=False) if s1 != s5
        )
        self.assertGreater(
            different_count,
            10,
            "Expected different k values to produce different scores",
        )

    def test_repr_string(self):
        """Test string representation includes key parameters."""
        engine = FaissSimilaritySearchEngine(window_size=100, warm_up=5)
        model = KNN(k=7, similarity_engine=engine)

        repr_str = repr(model)
        self.assertIn("k=7", repr_str)
        self.assertIn("FaissSimilaritySearchEngine", repr_str)
        self.assertIn("KNN", repr_str)

    def test_window_size_effect(self):
        """Test that different window sizes affect behavior."""
        # Small window
        small_engine = FaissSimilaritySearchEngine(window_size=20, warm_up=5)
        small_model = KNN(k=3, similarity_engine=small_engine)

        # Large window
        large_engine = FaissSimilaritySearchEngine(window_size=100, warm_up=5)
        large_model = KNN(k=3, similarity_engine=large_engine)

        # Generate long data stream
        data = self.data_generator.generate_streaming_data(n=80, n_features=2)

        # Train both models
        for point in data:
            small_model.learn_one(point.copy())
            large_model.learn_one(point.copy())

        # Test point should see different neighborhoods due to window size
        test_point = {"feature_0": 999.0, "feature_1": 999.0}  # Outlier point

        score_small = small_model.score_one(test_point)
        score_large = large_model.score_one(test_point)

        # Both should be valid scores
        self.assertIsInstance(score_small, float)
        self.assertIsInstance(score_large, float)
        self.assertGreaterEqual(score_small, 0.0)
        self.assertGreaterEqual(score_large, 0.0)


class TestKNNWithMockedEngine(unittest.TestCase):
    """Test KNN with mocked similarity engine for isolated testing."""

    def test_delegates_correctly_to_engine(self):
        """Test that KNN correctly delegates operations to its engine."""
        mock_engine = Mock()
        mock_engine.search.return_value = 1.5

        model = KNN(k=3, similarity_engine=mock_engine)

        # Test learn_one delegation
        test_point1 = {"x": 1.0, "y": 2.0}
        model.learn_one(test_point1)
        mock_engine.append.assert_called_once_with(test_point1)

        # Test score_one delegation
        test_point2 = {"x": 3.0, "y": 4.0}
        result = model.score_one(test_point2)

        mock_engine.search.assert_called_once_with(test_point2, n_neighbors=3)
        self.assertEqual(result, 1.5)

    def test_passes_k_parameter_correctly(self):
        """Test that the k parameter is passed correctly to engine search."""
        mock_engine = Mock()
        mock_engine.search.return_value = 2.0

        # Test different k values
        for k in [1, 3, 5, 10]:
            with self.subTest(k=k):
                model = KNN(k=k, similarity_engine=mock_engine)
                model.score_one({"feature": 1.0})

                # Verify k was passed correctly
                calls = mock_engine.search.call_args_list
                last_call = calls[-1]
                self.assertEqual(last_call[1]["n_neighbors"], k)

        mock_engine.reset_mock()


if __name__ == "__main__":
    unittest.main()
