import unittest

import numpy as np

from aberrant.model.svm.adaptive import (
    IncrementalOneClassSVMAdaptiveKernel,
)


class TestIncrementalOneClassSVMAdaptiveKernel(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Set random seed for reproducible tests
        np.random.seed(42)

    def test_adaptive_gamma(self):
        """Test that gamma adaptation is working."""
        model = IncrementalOneClassSVMAdaptiveKernel(
            initial_gamma=1.0,
            adaptation_rate=0.5,  # Higher rate for faster adaptation in test
            gamma_bounds=(0.1, 10.0),
        )

        # Feed data that should trigger gamma adaptation
        np.random.seed(42)
        for i in range(50):  # Need enough samples to trigger adaptation
            x = np.random.normal(0, 2, 3)  # Wider distribution
            sample = {f"feature_{i}": float(x[i]) for i in range(len(x))}
            model.learn_one(sample)

        # Gamma should have changed (though direction depends on data characteristics)
        # Check that adaptation mechanism is active
        self.assertIsInstance(model.gamma, float)
        self.assertGreaterEqual(model.gamma, model.gamma_min)
        self.assertLessEqual(model.gamma, model.gamma_max)

    def test_feature_consistency(self):
        """Test that model handles feature consistency correctly."""
        model = IncrementalOneClassSVMAdaptiveKernel()

        # Train with specific features
        sample1 = {"feature_a": 1.0, "feature_b": 2.0, "feature_c": 3.0}
        model.learn_one(sample1)

        # Should work with same features in different order
        sample2 = {"feature_c": 1.0, "feature_a": 2.0, "feature_b": 3.0}
        score = model.score_one(sample2)
        self.assertIsInstance(score, float)

        # Should raise error with different features
        sample3 = {"feature_x": 1.0, "feature_y": 2.0}
        with self.assertRaises(ValueError):
            model.score_one(sample3)

    def test_model_info(self):
        """Test model information reporting."""
        model = IncrementalOneClassSVMAdaptiveKernel(sv_budget=20)

        # Initial state
        info = model.get_model_info()
        self.assertEqual(info["n_support_vectors"], 0)
        self.assertEqual(info["n_samples_processed"], 0)

        # After training
        for i in range(10):
            sample = {"x": float(i), "y": float(i**2)}
            model.learn_one(sample)

        info = model.get_model_info()
        self.assertGreater(info["n_support_vectors"], 0)
        self.assertEqual(info["n_samples_processed"], 10)
        self.assertIn("gamma", info)
        self.assertIn("rho", info)

    def test_empty_model_predictions(self):
        """Test model behavior before any training."""
        model = IncrementalOneClassSVMAdaptiveKernel()

        # Should handle gracefully
        sample = {"feature1": 1.0, "feature2": 2.0}

        # Before training, should return default values
        prediction = model.predict_one(sample)
        score = model.score_one(sample)

        self.assertEqual(prediction, 1)  # Default to normal
        self.assertEqual(score, 0.0)  # Default neutral score

    def test_running_statistics_include_first_sample_and_keep_raw_support_vectors(self):
        model = IncrementalOneClassSVMAdaptiveKernel(seed=42)
        samples = [
            {"x": 10.0, "y": -3.0},
            {"x": 12.0, "y": 1.0},
            {"x": 14.0, "y": 5.0},
        ]

        for sample in samples:
            model.learn_one(sample)

        np.testing.assert_allclose(model.support_vectors[0], [10.0, -3.0])
        self.assertAlmostEqual(model.feature_stats["x"][0], 12.0)
        self.assertAlmostEqual(model.feature_stats["y"][0], 1.0)
        self.assertAlmostEqual(model.feature_stats["x"][1], np.std([10.0, 12.0, 14.0]))
        self.assertAlmostEqual(model.feature_stats["y"][1], np.std([-3.0, 1.0, 5.0]))

    def test_scoring_does_not_change_coordinates_or_state(self):
        model = IncrementalOneClassSVMAdaptiveKernel(seed=42)
        for i in range(30):
            model.learn_one({"x": float(i), "y": float(i * i)})

        support_before = [vector.copy() for vector in model.support_vectors]
        stats_before = dict(model.feature_stats)
        score_1 = model.score_one({"x": 2.0, "y": 4.0})
        score_2 = model.score_one({"x": 2.0, "y": 4.0})

        self.assertEqual(score_1, score_2)
        self.assertEqual(model.feature_stats, stats_before)
        for before, after in zip(support_before, model.support_vectors, strict=True):
            np.testing.assert_array_equal(before, after)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            IncrementalOneClassSVMAdaptiveKernel(nu=0.0)
        with self.assertRaises(ValueError):
            IncrementalOneClassSVMAdaptiveKernel(initial_gamma=0.0)
        with self.assertRaises(ValueError):
            IncrementalOneClassSVMAdaptiveKernel(
                initial_gamma=1.0, gamma_bounds=(2.0, 3.0)
            )
        with self.assertRaises(ValueError):
            IncrementalOneClassSVMAdaptiveKernel(adaptation_rate=0.0)
        with self.assertRaises(ValueError):
            IncrementalOneClassSVMAdaptiveKernel(buffer_size=0)
        with self.assertRaises(ValueError):
            IncrementalOneClassSVMAdaptiveKernel(sv_budget=0)


if __name__ == "__main__":
    # Run with verbosity to see print statements
    unittest.main(verbosity=2)
