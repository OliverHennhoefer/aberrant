"""Tests for KSWIN drift detector."""

import random
import unittest

from aberrant.drift.kswin import KSWIN


class TestKSWIN(unittest.TestCase):
    """Test suite for KSWIN drift detector."""

    def test_initialization_default_params(self):
        """Test KSWIN initialization with default parameters."""
        detector = KSWIN()
        self.assertEqual(detector.alpha, 0.005)
        self.assertEqual(detector.window_size, 100)
        self.assertEqual(detector.stat_size, 30)
        self.assertIsNone(detector.seed)
        self.assertFalse(detector.drift_detected)
        self.assertEqual(detector.n_detections, 0)

    def test_initialization_custom_params(self):
        """Test KSWIN initialization with custom parameters."""
        detector = KSWIN(
            alpha=0.01,
            window_size=200,
            stat_size=50,
            seed=42,
        )
        self.assertEqual(detector.alpha, 0.01)
        self.assertEqual(detector.window_size, 200)
        self.assertEqual(detector.stat_size, 50)
        self.assertEqual(detector.seed, 42)

    def test_invalid_alpha(self):
        """Test that invalid alpha values raise ValueError."""
        with self.assertRaises(ValueError) as context:
            KSWIN(alpha=0)
        self.assertIn("alpha must be in (0, 1)", str(context.exception))

        with self.assertRaises(ValueError):
            KSWIN(alpha=1)

        with self.assertRaises(ValueError):
            KSWIN(alpha=-0.1)

    def test_invalid_window_size(self):
        """Test that invalid window_size raises ValueError."""
        with self.assertRaises(ValueError) as context:
            KSWIN(window_size=0)
        self.assertIn("window_size must be positive", str(context.exception))

        with self.assertRaises(ValueError):
            KSWIN(window_size=-1)

    def test_invalid_stat_size(self):
        """Test that invalid stat_size raises ValueError."""
        with self.assertRaises(ValueError) as context:
            KSWIN(stat_size=0)
        self.assertIn("stat_size must be positive", str(context.exception))

        # stat_size must be less than window_size / 2
        with self.assertRaises(ValueError) as context:
            KSWIN(window_size=100, stat_size=60)
        self.assertIn(
            "stat_size must be less than window_size / 2", str(context.exception)
        )

    def test_update_returns_self(self):
        """Test that update returns self for method chaining."""
        detector = KSWIN()
        result = detector.update(1.0)
        self.assertIs(result, detector)

    def test_non_finite_observation_is_rejected_without_mutating_window(self):
        detector = KSWIN()
        detector.update(1.0)

        with self.assertRaisesRegex(ValueError, "observation must be finite"):
            detector.update(float("inf"))

        self.assertEqual(list(detector._window), [1.0])

    def test_no_drift_before_full_window(self):
        """Test that no drift is detected before window is full."""
        detector = KSWIN(window_size=100)
        for i in range(99):
            detector.update(float(i))
            self.assertFalse(detector.drift_detected)
        self.assertEqual(detector.n_detections, 0)

    def test_no_drift_on_stationary_data(self):
        """Test that no drift is detected on stationary data."""
        rng = random.Random(42)
        detector = KSWIN(alpha=0.005, window_size=100, stat_size=30, seed=123)

        # Generate stationary data
        for _ in range(300):
            detector.update(rng.gauss(0, 1))

        # Should not detect drift (or very few false positives)
        self.assertLess(detector.n_detections, 3)

    def test_drift_detected_on_distribution_change(self):
        """Test that drift is detected when distribution changes."""
        detector = KSWIN(alpha=0.01, window_size=100, stat_size=30, seed=42)

        # First phase: mean = 0
        for _ in range(100):
            detector.update(0.0)

        # Second phase: mean = 10 (clear shift)
        detected = False
        for _ in range(200):
            detector.update(10.0)
            if detector.drift_detected:
                detected = True
                break

        self.assertTrue(detected, "Drift should be detected on distribution change")
        self.assertGreater(detector.n_detections, 0)

    def test_p_value_and_statistic_properties(self):
        """Test p_value and statistic properties."""
        detector = KSWIN(window_size=50, stat_size=15)

        # Before any check
        self.assertIsNone(detector.p_value)
        self.assertIsNone(detector.statistic)

        # After filling window
        for i in range(60):
            detector.update(float(i))

        # Properties should be set after check
        self.assertIsNotNone(detector.p_value)
        self.assertIsNotNone(detector.statistic)
        self.assertGreaterEqual(detector.p_value, 0.0)
        self.assertLessEqual(detector.p_value, 1.0)
        self.assertGreaterEqual(detector.statistic, 0.0)

    def test_reset(self):
        """Test reset clears all state."""
        detector = KSWIN(seed=42)
        for i in range(150):
            detector.update(float(i))

        detector.reset()
        self.assertEqual(len(detector._window), 0)
        self.assertEqual(detector.n_detections, 0)
        self.assertFalse(detector.drift_detected)
        self.assertIsNone(detector.p_value)
        self.assertIsNone(detector.statistic)

    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with same seed."""
        results1 = []
        detector1 = KSWIN(seed=42)
        rng1 = random.Random(123)
        for _ in range(200):
            detector1.update(rng1.gauss(0, 1))
            results1.append(detector1.drift_detected)

        results2 = []
        detector2 = KSWIN(seed=42)
        rng2 = random.Random(123)
        for _ in range(200):
            detector2.update(rng2.gauss(0, 1))
            results2.append(detector2.drift_detected)

        self.assertEqual(results1, results2)

    def test_method_chaining(self):
        """Test that methods can be chained."""
        detector = KSWIN()
        result = detector.update(1.0).update(2.0).update(3.0)
        self.assertIs(result, detector)

    def test_repr(self):
        """Test string representation."""
        detector = KSWIN(alpha=0.01, window_size=200, stat_size=50)
        repr_str = repr(detector)
        self.assertIn("KSWIN", repr_str)
        self.assertIn("alpha=0.01", repr_str)
        self.assertIn("window_size=200", repr_str)
        self.assertIn("stat_size=50", repr_str)


if __name__ == "__main__":
    unittest.main()
