"""Unit tests for xStream anomaly detection model."""

import unittest

import numpy as np

from aberrant.model.iforest.xstream import XStream


class TestXStream(unittest.TestCase):
    """Test suite for XStream model."""

    def create_model(self) -> XStream:
        return XStream(
            k=16,
            n_chains=8,
            depth=6,
            cms_width=128,
            cms_num_hashes=3,
            window_size=16,
            init_sample_size=16,
            density=0.25,
            max_feature_cache_size=32,
            seed=42,
        )

    def test_initialization_defaults(self):
        """Test default initialization."""
        model = XStream()
        self.assertEqual(model.k, 100)
        self.assertEqual(model.n_chains, 100)
        self.assertEqual(model.depth, 15)
        self.assertEqual(model.window_size, 256)
        self.assertEqual(model.max_feature_cache_size, 10_000)
        self.assertIsNone(model._state)

    def test_invalid_parameters(self):
        """Test parameter validation."""
        with self.assertRaises(ValueError):
            XStream(k=0)
        with self.assertRaises(ValueError):
            XStream(n_chains=0)
        with self.assertRaises(ValueError):
            XStream(depth=0)
        with self.assertRaises(ValueError):
            XStream(cms_width=0)
        with self.assertRaises(ValueError):
            XStream(cms_num_hashes=0)
        with self.assertRaises(ValueError):
            XStream(window_size=0)
        with self.assertRaises(ValueError):
            XStream(init_sample_size=0)
        with self.assertRaises(ValueError):
            XStream(density=0.0)
        with self.assertRaises(ValueError):
            XStream(density=1.5)
        with self.assertRaises(ValueError):
            XStream(max_feature_cache_size=0)

    def test_score_zero_before_ready(self):
        """score_one should return 0.0 before model warm-up is complete."""
        model = self.create_model()
        score = model.score_one({"x": 1.0, "y": 2.0})
        self.assertEqual(score, 0.0)

    def test_empty_and_non_numeric_input(self):
        """Model should reject empty and non-numeric inputs."""
        model = self.create_model()
        with self.assertRaises(ValueError):
            model.learn_one({})
        with self.assertRaises(ValueError):
            model.score_one({})
        with self.assertRaises(ValueError):
            model.learn_one({"x": "bad"})  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            model.score_one({"x": float("inf")})
        with self.assertRaises(ValueError):
            model.score_one({"x": float("nan")})

    def test_large_finite_values_do_not_overflow(self):
        """Large finite values should not trigger integer overflow."""
        model = XStream(
            k=8,
            n_chains=4,
            depth=4,
            cms_width=64,
            cms_num_hashes=2,
            window_size=8,
            init_sample_size=8,
            density=0.5,
            seed=7,
        )
        for i in range(16):
            model.learn_one({"x": float(i), "y": 0.0})

        large_point = {"x": 1e30, "y": 0.0}
        score = model.score_one(large_point)
        model.learn_one(large_point)

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_deterministic_with_seed(self):
        """Same seed and data should produce identical scores."""
        model1 = self.create_model()
        model2 = self.create_model()

        data = [{"x": float(i % 5), "y": float(i % 3)} for i in range(64)]
        for point in data:
            model1.learn_one(point)
            model2.learn_one(point)

        test_point = {"x": 2.0, "y": 1.0}
        self.assertAlmostEqual(
            model1.score_one(test_point), model2.score_one(test_point), places=12
        )

    def test_feature_evolving_support(self):
        """Model should handle changing feature keys over time."""
        model = self.create_model()
        for i in range(80):
            point = {"base": float(i % 5)}
            if i % 3 == 0:
                point["f_a"] = float(i)
            if i % 5 == 0:
                point["f_b"] = float(i * 0.1)
            model.learn_one(point)

        score = model.score_one({"base": 1.0, "f_new": 8.0})
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_feature_cache_is_bounded_lru(self):
        """Feature cache should evict least recently used mappings."""
        model = XStream(
            k=8,
            n_chains=4,
            depth=4,
            cms_width=64,
            cms_num_hashes=2,
            window_size=8,
            init_sample_size=8,
            density=0.5,
            max_feature_cache_size=3,
            seed=7,
        )

        model._feature_projection("a")
        model._feature_projection("b")
        model._feature_projection("c")
        model._feature_projection("a")  # refresh "a"
        model._feature_projection("d")  # should evict "b"

        self.assertEqual(len(model._feature_cache), 3)
        self.assertNotIn("b", model._feature_cache)
        self.assertIn("a", model._feature_cache)
        self.assertIn("c", model._feature_cache)
        self.assertIn("d", model._feature_cache)

    def test_window_rollover_sets_reference(self):
        """After enough samples, reference sketch should become available."""
        model = XStream(
            k=8,
            n_chains=4,
            depth=4,
            cms_width=64,
            cms_num_hashes=2,
            window_size=8,
            init_sample_size=8,
            density=0.5,
            seed=7,
        )
        for i in range(8):
            model.learn_one({"x": float(i), "y": float(i + 1)})

        self.assertIsNotNone(model._state)
        assert model._state is not None
        self.assertTrue(model._state.reference_ready)
        self.assertEqual(model._state.samples_in_window, 0)

    def test_outlier_scores_higher_than_normal(self):
        """Outlier should score higher than near-cluster point."""
        model = self.create_model()

        rng = np.random.default_rng(123)
        for _ in range(200):
            normal = {
                "x": float(rng.normal(0.0, 0.3)),
                "y": float(rng.normal(0.0, 0.3)),
            }
            model.learn_one(normal)

        normal_score = model.score_one({"x": 0.05, "y": -0.02})
        outlier_score = model.score_one({"x": 6.0, "y": 6.0})

        self.assertGreater(outlier_score, normal_score)

    def test_reset_restores_state(self):
        """reset() should clear learned state."""
        model = self.create_model()
        for i in range(50):
            model.learn_one({"x": float(i), "y": float(i * 2)})

        self.assertIsNotNone(model._state)
        model.reset()
        self.assertIsNone(model._state)
        self.assertEqual(model.score_one({"x": 1.0, "y": 2.0}), 0.0)

    def test_repr_contains_key_config(self):
        """repr should include key hyperparameters."""
        model = self.create_model()
        output = repr(model)
        self.assertIn("XStream", output)
        self.assertIn("k=16", output)
        self.assertIn("n_chains=8", output)


if __name__ == "__main__":
    unittest.main()
