"""Tests for RandomModel baseline anomaly detector."""

import unittest

import numpy as np

from aberrant.model.random import RandomModel
from tests.utils import DataGenerator


class TestRandomModel(unittest.TestCase):
    """Test suite for RandomModel."""

    def create_model(self) -> RandomModel:
        """Create RandomModel instance for testing."""
        return RandomModel(seed=42)

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.data_generator = DataGenerator(seed=42)

    def test_returns_random_scores_in_range(self):
        """Test that RandomModel returns scores in [0, 1] range."""
        model = RandomModel(seed=42)

        # Generate test data
        test_data = self.data_generator.generate_streaming_data(n=100)
        scores = []

        for point in test_data:
            model.learn_one(point)
            score = model.score_one(point)
            scores.append(score)

        # All scores should be in [0, 1]
        for score in scores:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
            self.assertIsInstance(score, float)

        # Scores should be random (not all the same)
        unique_scores = set(scores)
        self.assertGreater(
            len(unique_scores), 50, "Expected more variety in random scores"
        )

    def test_reproducibility_with_same_seed(self):
        """Test that same seed produces reproducible results."""
        model1 = RandomModel(seed=42)
        model2 = RandomModel(seed=42)

        test_data = self.data_generator.generate_streaming_data(n=100)

        scores1 = []
        scores2 = []

        for point in test_data:
            model1.learn_one(point)
            model2.learn_one(point)

            score1 = model1.score_one(point)
            score2 = model2.score_one(point)

            scores1.append(score1)
            scores2.append(score2)

        # Scores should be identical with same seed
        for s1, s2 in zip(scores1, scores2, strict=False):
            self.assertEqual(s1, s2)

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        model1 = RandomModel(seed=42)
        model2 = RandomModel(seed=123)

        test_data = self.data_generator.generate_streaming_data(n=100)

        scores1 = []
        scores2 = []

        for point in test_data:
            model1.learn_one(point)
            model2.learn_one(point)

            score1 = model1.score_one(point)
            score2 = model2.score_one(point)

            scores1.append(score1)
            scores2.append(score2)

        # Most scores should be different
        different_count = sum(
            1 for s1, s2 in zip(scores1, scores2, strict=False) if s1 != s2
        )
        self.assertGreater(
            different_count,
            95,
            "Expected most scores to be different with different seeds",
        )

    def test_learn_one_does_not_affect_randomness(self):
        """Test that learn_one doesn't affect the random number generation."""
        model = RandomModel(seed=42)

        # Get scores without learning
        scores_without_learning = []
        for _ in range(10):
            score = model.score_one({"feature": 1.0})
            scores_without_learning.append(score)

        # Reset model with same seed
        model = RandomModel(seed=42)

        # Get scores with learning
        scores_with_learning = []
        for i in range(10):
            model.learn_one({"feature": float(i)})  # Learn different data
            score = model.score_one({"feature": 1.0})
            scores_with_learning.append(score)

        # Should be identical (learning doesn't affect randomness)
        self.assertEqual(scores_without_learning, scores_with_learning)

    def test_uniform_distribution_properties(self):
        """Test that scores follow uniform distribution properties."""
        model = RandomModel(seed=42)

        # Generate many scores
        scores = []
        for i in range(1000):
            model.learn_one({"feature": float(i)})
            score = model.score_one({"feature": float(i)})
            scores.append(score)

        # Basic statistics for uniform distribution [0, 1]
        mean = np.mean(scores)
        variance = np.var(scores)

        # Uniform distribution [0, 1] has mean=0.5, variance=1/12≈0.083
        self.assertAlmostEqual(mean, 0.5, places=1)
        self.assertAlmostEqual(variance, 1 / 12, places=1)

        # All values should be in [0, 1]
        self.assertGreaterEqual(min(scores), 0.0)
        self.assertLessEqual(max(scores), 1.0)

    def test_consistency_across_different_inputs(self):
        """Test that model produces scores regardless of input."""
        model = RandomModel(seed=42)

        test_cases = [
            {"single_feature": 1.0},
            {"x": 0.0, "y": 100.0},
            {"a": -999.0, "b": 0.001, "c": 1e6},
            {"small": 1e-10},
            {"large": 1e10},
            {},  # Empty dict
        ]

        for test_case in test_cases:
            with self.subTest(test_case=test_case):
                score = model.score_one(test_case)
                self.assertIsInstance(score, float)
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)

    def test_score_stability_after_learning(self):
        """Override base test - RandomModel scores are not deterministic."""
        # For RandomModel, scores should NOT be deterministic
        model = RandomModel(seed=42)

        # Learn sufficient data
        learning_data = [{"x": float(i)} for i in range(500)]
        for data in learning_data:
            model.learn_one(data)

        # Score the same point multiple times - should be DIFFERENT
        test_point = {"x": 100.0}
        score1 = model.score_one(test_point)
        score2 = model.score_one(test_point)

        # Random model should produce different scores each time
        self.assertNotEqual(
            score1, score2, "Random model should produce different scores"
        )

        # But both should be valid random values
        self.assertIsInstance(score1, float)
        self.assertIsInstance(score2, float)
        self.assertGreaterEqual(score1, 0.0)
        self.assertLessEqual(score1, 1.0)
        self.assertGreaterEqual(score2, 0.0)
        self.assertLessEqual(score2, 1.0)

    def test_repr_string(self):
        """Test string representation includes seed."""
        model = RandomModel(seed=42)
        repr_str = repr(model)

        self.assertEqual(repr_str, "RandomModel(seed=42)")

    def test_default_seed(self):
        """Test that default seed is used when none provided."""
        model = RandomModel()

        self.assertEqual(repr(model), "RandomModel()")
        self.assertEqual(model.seed, 1)

    def test_reproducibility_over_multiple_calls(self):
        """Test reproducibility when calling score_one multiple times."""
        # Test that calling score_one advances the RNG state
        model1 = RandomModel(seed=42)
        model2 = RandomModel(seed=42)

        # First call on both models
        score1_1 = model1.score_one({"feature": 1.0})
        score2_1 = model2.score_one({"feature": 1.0})
        self.assertEqual(score1_1, score2_1)

        # Second call should produce different but identical scores
        score1_2 = model1.score_one({"feature": 1.0})
        score2_2 = model2.score_one({"feature": 1.0})
        self.assertEqual(score1_2, score2_2)
        self.assertNotEqual(score1_1, score1_2)  # Different from first call


if __name__ == "__main__":
    unittest.main()
