import copy
import time
import unittest

import numpy as np

from aberrant.model.svm.adaptive import (
    IncrementalOneClassSVMAdaptiveKernel,
)
from tests import score_sequences


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


def _pairwise_gamma_estimate(
    model: IncrementalOneClassSVMAdaptiveKernel, std: np.ndarray
) -> float:
    """Estimate gamma with the pair-by-pair loop of aberrant 1.4.0."""
    if len(model.data_buffer) < 10:
        return model.gamma
    mean = model._feature_mean
    standardized = np.vstack([(row - mean) / std for row in model.data_buffer])
    n_samples = min(50, len(standardized))
    indices = model.rng.choice(len(standardized), size=n_samples, replace=False)
    sampled = standardized[indices]
    distances = []
    for i in range(n_samples):
        for j in range(i + 1, min(i + 10, n_samples)):
            distance = float(np.linalg.norm(sampled[i] - sampled[j]))
            if distance > 1e-10:
                distances.append(distance)
    if not distances:
        return model.gamma
    gamma = 1.0 / (2.0 * float(np.median(distances)) ** 2)
    return float(np.clip(gamma, model.gamma_min, model.gamma_max))


class TestIncrementalOneClassSVMAdaptiveKernelExactness(unittest.TestCase):
    """Vectorized kernels must reproduce the 1.4.0 scores and generator draws."""

    def test_score_sequences_match_recorded_reference(self):
        for case in score_sequences.load("incremental_one_class_svm_adaptive_kernel"):
            with self.subTest(case=case["name"]):
                model = IncrementalOneClassSVMAdaptiveKernel(**case["params"])
                scores = score_sequences.replay(model, case)
                # Scores are differences of kernel sums, so last-bit exp and
                # BLAS differences between platforms surface as absolute
                # errors near 1e-14; the recording platform reproduces every bit.
                np.testing.assert_allclose(scores, case["scores"], rtol=0, atol=1e-12)

    def test_kernel_row_matches_kernels_of_single_pairs(self):
        events = score_sequences.make_stream(seed=1, n_events=150, n_features=5)
        model = IncrementalOneClassSVMAdaptiveKernel(seed=2)
        for event in events:
            model.learn_one(event)
        self.assertGreater(len(model.support_vectors), 10)
        mean = model._feature_mean
        std = model._feature_std()

        for query in np.random.default_rng(3).normal(size=(20, 5)) * 3.0:
            row = model._compute_kernel_row(query, std)
            for vector, kernel in zip(model.support_vectors, row, strict=True):
                # The 1.4.0 kernel of one standardized pair.
                difference = (query - mean) / std - (vector - mean) / std
                distance = float(np.dot(difference, difference))
                self.assertEqual(kernel, float(np.exp(-model.gamma * distance)))

    def test_row_dot_products_match_dot_of_each_row(self):
        rng = np.random.default_rng(4)
        for n_features in (1, 2, 3, 5, 8, 9, 16, 33, 64):
            scales = 10.0 ** rng.integers(-150, 150, size=(30, 1))
            rows = rng.normal(size=(30, n_features)) * scales
            products = np.vecdot(rows, rows)
            for row, product in zip(rows, products, strict=True):
                self.assertEqual(product, np.dot(row, row))
                self.assertEqual(np.sqrt(product), np.linalg.norm(row))

    def test_gamma_estimate_matches_pairwise_loop(self):
        streams = {
            "too_short": score_sequences.make_stream(seed=5, n_events=9, n_features=3),
            "short": score_sequences.make_stream(seed=6, n_events=23, n_features=3),
            "repeated": score_sequences.make_stream(
                seed=7, n_events=70, n_features=2, kind="grid"
            ),
            "full_buffer": score_sequences.make_stream(
                seed=8, n_events=260, n_features=4
            ),
            "wide": score_sequences.make_stream(seed=11, n_events=80, n_features=24),
            "identical": [{"x0": 1.5, "x1": -2.0}] * 30,
        }
        for name, events in streams.items():
            with self.subTest(stream=name):
                model = IncrementalOneClassSVMAdaptiveKernel(seed=9)
                for event in events:
                    model.learn_one(event)
                reference = copy.deepcopy(model)
                std = model._feature_std()

                self.assertEqual(
                    model._estimate_optimal_gamma(std),
                    _pairwise_gamma_estimate(reference, std),
                )
                self.assertEqual(
                    model.rng.bit_generator.state, reference.rng.bit_generator.state
                )

    def test_default_model_speed_smoke(self):
        events = score_sequences.make_stream(seed=10, n_events=2000, n_features=8)
        model = IncrementalOneClassSVMAdaptiveKernel(seed=10)

        start = time.perf_counter()
        score_sequences.score_then_learn(model, events)

        # Generous bound against large regressions; typical runs take ~0.4 s.
        self.assertLess(time.perf_counter() - start, 4.0)


if __name__ == "__main__":
    # Run with verbosity to see print statements
    unittest.main(verbosity=2)
