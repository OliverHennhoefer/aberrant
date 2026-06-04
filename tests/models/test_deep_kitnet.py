"""Unit tests for KitNET anomaly detection model."""

import unittest

import numpy as np

from aberrant.model.deep.kitnet import OnlineAutoencoderEnsemble as KitNET


class TestKitNET(unittest.TestCase):
    """Test suite for KitNET model."""

    def create_model(self) -> KitNET:
        return KitNET(
            max_ae_size=2,
            feature_map_grace=12,
            ad_grace=24,
            learning_rate=0.05,
            hidden_ratio=0.75,
            seed=42,
        )

    def test_initialization_defaults(self) -> None:
        """Default constructor values are valid."""
        model = KitNET()
        self.assertEqual(model.max_ae_size, 10)
        self.assertEqual(model.feature_map_grace, 5000)
        self.assertEqual(model.ad_grace, 50000)
        self.assertEqual(model.learning_rate, 0.1)
        self.assertEqual(model.hidden_ratio, 0.75)
        self.assertEqual(model.phase, "feature_map_warmup")
        self.assertFalse(model.is_ready)

    def test_invalid_parameters(self) -> None:
        """Invalid constructor parameters raise ValueError."""
        with self.assertRaises(ValueError):
            KitNET(max_ae_size=0)
        with self.assertRaises(ValueError):
            KitNET(feature_map_grace=0)
        with self.assertRaises(ValueError):
            KitNET(ad_grace=-1)
        with self.assertRaises(ValueError):
            KitNET(learning_rate=0.0)
        with self.assertRaises(ValueError):
            KitNET(hidden_ratio=0.0)
        with self.assertRaises(ValueError):
            KitNET(hidden_ratio=1.5)

    def test_rejects_invalid_input(self) -> None:
        """Model rejects empty, non-numeric, and non-finite inputs."""
        model = self.create_model()

        with self.assertRaises(ValueError):
            model.learn_one({})
        with self.assertRaises(ValueError):
            model.score_one({})

        with self.assertRaises(ValueError):
            model.learn_one({"x": "bad"})  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            model.score_one({"x": float("nan")})
        with self.assertRaises(ValueError):
            model.learn_one({"x": float("inf")})

    def test_score_before_learning_does_not_initialize_schema(self) -> None:
        """score_one before learn_one should return 0.0 and keep schema unset."""
        model = self.create_model()
        score = model.score_one({"x": 1.0, "y": 2.0})

        self.assertEqual(score, 0.0)
        self.assertIsNone(model._feature_order)
        self.assertEqual(model.phase, "feature_map_warmup")

    def test_phase_transitions(self) -> None:
        """Model transitions through warm-up phases deterministically."""
        model = self.create_model()

        for i in range(11):
            model.learn_one({"x": float(i), "y": float(i * 2)})
        self.assertEqual(model.phase, "feature_map_warmup")

        model.learn_one({"x": 11.0, "y": 22.0})
        self.assertEqual(model.phase, "detector_warmup")
        self.assertGreater(len(model._feature_groups), 0)

        for i in range(24):
            model.learn_one({"x": float(i), "y": float(i * 2)})
        self.assertTrue(model.is_ready)
        self.assertEqual(model.phase, "ready")

    def test_ad_grace_zero_trains_before_ready(self) -> None:
        """ad_grace=0 should still run one detector training step."""
        model = KitNET(
            max_ae_size=2,
            feature_map_grace=3,
            ad_grace=0,
            learning_rate=0.05,
            hidden_ratio=0.75,
            adaptive_after_warmup=False,
            seed=42,
        )

        model.learn_one({"x": 0.0, "y": 0.0})
        model.learn_one({"x": 1.0, "y": 1.0})
        self.assertEqual(model.phase, "feature_map_warmup")
        self.assertEqual(model._detector_samples, 0)

        model.learn_one({"x": 2.0, "y": 2.0})
        self.assertEqual(model.phase, "ready")
        self.assertEqual(model._detector_samples, 1)

    def test_score_zero_during_warmups(self) -> None:
        """Scores remain zero until ready phase."""
        model = self.create_model()
        x = {"x": 1.0, "y": 2.0}

        for _ in range(10):
            model.learn_one(x)
            self.assertEqual(model.score_one(x), 0.0)

    def test_feature_mismatch_raises(self) -> None:
        """Feature key mismatches should fail fast."""
        model = self.create_model()
        model.learn_one({"a": 1.0, "b": 2.0})

        with self.assertRaises(ValueError):
            model.learn_one({"a": 1.0, "c": 2.0})
        with self.assertRaises(ValueError):
            model.score_one({"a": 1.0, "c": 2.0})

    def test_deterministic_with_seed(self) -> None:
        """Two models with same seed and stream should produce equal scores."""
        model_1 = self.create_model()
        model_2 = self.create_model()

        data = [{"x": float(i % 5), "y": float((i * 3) % 7)} for i in range(64)]
        for point in data:
            model_1.learn_one(point)
            model_2.learn_one(point)

        test_point = {"x": 1.25, "y": -0.75}
        self.assertAlmostEqual(
            model_1.score_one(test_point),
            model_2.score_one(test_point),
            places=12,
        )

    def test_score_is_non_mutating(self) -> None:
        """score_one should not mutate learning counters."""
        model = self.create_model()
        for i in range(64):
            model.learn_one({"x": float(i), "y": float(i + 1)})

        samples_before = model._samples_seen
        phase_before = model.phase

        _ = model.score_one({"x": 1.0, "y": 2.0})

        self.assertEqual(model._samples_seen, samples_before)
        self.assertEqual(model.phase, phase_before)

    def test_outlier_scores_higher_than_normal(self) -> None:
        """Far point should score above in-cluster point after training."""
        model = self.create_model()
        rng = np.random.default_rng(1234)

        for _ in range(200):
            point = {
                "x": float(rng.normal(0.0, 0.15)),
                "y": float(rng.normal(0.0, 0.15)),
            }
            model.learn_one(point)

        normal_score = model.score_one({"x": 0.05, "y": -0.02})
        outlier_score = model.score_one({"x": 3.5, "y": 3.0})

        self.assertIsInstance(normal_score, float)
        self.assertIsInstance(outlier_score, float)
        self.assertGreaterEqual(normal_score, 0.0)
        self.assertGreaterEqual(outlier_score, 0.0)
        self.assertGreater(outlier_score, normal_score)

    def test_reset_restores_cold_start(self) -> None:
        """reset() should clear all learned state."""
        model = self.create_model()
        for i in range(40):
            model.learn_one({"x": float(i), "y": float(i + 1)})
        self.assertIsNotNone(model._feature_order)

        model.reset()
        self.assertIsNone(model._feature_order)
        self.assertEqual(model.phase, "feature_map_warmup")
        self.assertFalse(model.is_ready)
        self.assertEqual(model.score_one({"x": 0.0, "y": 1.0}), 0.0)

    def test_repr_contains_key_settings(self) -> None:
        """repr should include main hyperparameters."""
        model = self.create_model()
        representation = repr(model)

        self.assertIn("OnlineAutoencoderEnsemble", representation)
        self.assertIn("max_ae_size=2", representation)
        self.assertIn("feature_map_grace=12", representation)
        self.assertIn("ad_grace=24", representation)


if __name__ == "__main__":
    unittest.main()
