"""Tests for ADWIN drift detector."""

import random
import unittest

from aberrant.drift.adwin import ADWIN


class TestADWIN(unittest.TestCase):
    """Test suite for ADWIN drift detector."""

    def test_initialization_default_params(self):
        """Test ADWIN initialization with default parameters."""
        detector = ADWIN()
        self.assertEqual(detector.delta, 0.002)
        self.assertEqual(detector.clock, 32)
        self.assertEqual(detector.max_buckets, 5)
        self.assertEqual(detector.min_window_length, 5)
        self.assertEqual(detector.grace_period, 10)
        self.assertEqual(detector.width, 0)
        self.assertEqual(detector.n_detections, 0)
        self.assertFalse(detector.drift_detected)

    def test_initialization_custom_params(self):
        """Test ADWIN initialization with custom parameters."""
        detector = ADWIN(
            delta=0.01,
            clock=16,
            max_buckets=10,
            min_window_length=10,
            grace_period=20,
        )
        self.assertEqual(detector.delta, 0.01)
        self.assertEqual(detector.clock, 16)
        self.assertEqual(detector.max_buckets, 10)
        self.assertEqual(detector.min_window_length, 10)
        self.assertEqual(detector.grace_period, 20)

    def test_invalid_delta(self):
        """Test that invalid delta values raise ValueError."""
        with self.assertRaises(ValueError) as context:
            ADWIN(delta=0)
        self.assertIn("delta must be in (0, 1)", str(context.exception))

        with self.assertRaises(ValueError):
            ADWIN(delta=1)

        with self.assertRaises(ValueError):
            ADWIN(delta=-0.1)

    def test_invalid_clock(self):
        """Test that invalid clock values raise ValueError."""
        with self.assertRaises(ValueError) as context:
            ADWIN(clock=0)
        self.assertIn("clock must be positive", str(context.exception))

        with self.assertRaises(ValueError):
            ADWIN(clock=-1)

    def test_invalid_max_buckets(self):
        """Test that invalid max_buckets values raise ValueError."""
        with self.assertRaises(ValueError):
            ADWIN(max_buckets=0)

    def test_invalid_min_window_length(self):
        """Test that invalid min_window_length values raise ValueError."""
        with self.assertRaises(ValueError):
            ADWIN(min_window_length=0)

    def test_invalid_grace_period(self):
        """Test that invalid grace_period values raise ValueError."""
        with self.assertRaises(ValueError):
            ADWIN(grace_period=-1)

    def test_update_returns_self(self):
        """Test that update returns self for method chaining."""
        detector = ADWIN()
        result = detector.update(1.0)
        self.assertIs(result, detector)

    def test_update_increases_width(self):
        """Test that update increases window width."""
        detector = ADWIN()
        for i in range(10):
            detector.update(float(i))
        self.assertEqual(detector.width, 10)

    def test_non_finite_observation_is_rejected_without_mutating_state(self):
        detector = ADWIN()
        detector.update(1.0)

        with self.assertRaisesRegex(ValueError, "observation must be finite"):
            detector.update(float("nan"))

        self.assertEqual(detector.width, 1)
        self.assertEqual(detector.estimation, 1.0)

    def test_estimation_property(self):
        """Test the estimation (mean) property."""
        detector = ADWIN()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values:
            detector.update(v)
        self.assertAlmostEqual(detector.estimation, 3.0, places=5)

    def test_estimation_empty_window(self):
        """Test estimation returns 0 for empty window."""
        detector = ADWIN()
        self.assertEqual(detector.estimation, 0.0)

    def test_variance_property(self):
        """Test the variance property."""
        detector = ADWIN()
        # Add constant values - should have zero variance
        for _ in range(10):
            detector.update(5.0)
        self.assertAlmostEqual(detector.variance, 0.0, places=5)

    def test_variance_matches_two_sample_population_variance(self):
        """Test Welford update matches expected population variance for 2 samples."""
        detector = ADWIN(clock=1000)
        detector.update(1.0)
        detector.update(2.0)
        self.assertAlmostEqual(detector.variance, 0.25, places=10)

    def test_variance_remains_exact_after_oldest_bucket_removal(self):
        """Removing a subwindow must remove within- and between-group variance."""
        detector = ADWIN(clock=1000, max_buckets=10)
        for value in [1.0, 2.0, 10.0, 11.0]:
            detector.update(value)

        detector._remove_oldest(2)

        self.assertEqual(detector.width, 2)
        self.assertAlmostEqual(detector.estimation, 10.5, places=12)
        self.assertAlmostEqual(detector.variance, 0.25, places=12)

    def test_variance_empty_window(self):
        """Test variance returns 0 for empty/single-element window."""
        detector = ADWIN()
        self.assertEqual(detector.variance, 0.0)
        detector.update(1.0)
        self.assertEqual(detector.variance, 0.0)

    def test_no_drift_on_stationary_data(self):
        """Test that no drift is detected on stationary data."""
        detector = ADWIN(delta=0.002)
        # Generate stationary data around mean 0
        rng = random.Random(42)
        for _ in range(500):
            detector.update(rng.gauss(0, 1))
        # Should not detect drift (or very few false positives)
        self.assertLess(detector.n_detections, 3)

    def test_drift_detected_on_mean_shift(self):
        """Test that drift is detected when mean shifts."""
        detector = ADWIN(delta=0.002, grace_period=5)
        # First phase: mean = 0
        for _ in range(200):
            detector.update(0.0)
        # Second phase: mean = 10 (large shift)
        detected = False
        for _ in range(200):
            detector.update(10.0)
            if detector.drift_detected:
                detected = True
                break
        self.assertTrue(detected, "Drift should be detected on mean shift")
        self.assertGreater(detector.n_detections, 0)

    def test_reset(self):
        """Test reset clears all state."""
        detector = ADWIN()
        for i in range(100):
            detector.update(float(i))
        self.assertGreater(detector.width, 0)
        detector.reset()
        self.assertEqual(detector.width, 0)
        self.assertEqual(detector.n_detections, 0)
        self.assertFalse(detector.drift_detected)
        self.assertEqual(detector.estimation, 0.0)

    def test_method_chaining(self):
        """Test that methods can be chained."""
        detector = ADWIN()
        result = detector.update(1.0).update(2.0).update(3.0)
        self.assertIs(result, detector)
        self.assertEqual(detector.width, 3)

    def test_repr(self):
        """Test string representation."""
        detector = ADWIN(delta=0.01, clock=16)
        repr_str = repr(detector)
        self.assertIn("ADWIN", repr_str)
        self.assertIn("delta=0.01", repr_str)
        self.assertIn("clock=16", repr_str)
        self.assertIn("width=", repr_str)


class TestADWINBucketCompression(unittest.TestCase):
    """Test ADWIN bucket compression behavior."""

    def test_bucket_compression_occurs(self):
        """Test that bucket compression happens with many samples."""
        detector = ADWIN(max_buckets=5)
        # Add many samples to trigger compression
        for i in range(1000):
            detector.update(float(i % 10))
        # Window should be maintained efficiently
        self.assertGreater(detector.width, 0)
        self.assertLessEqual(detector.width, 1000)


class TestADWINAutoReset(unittest.TestCase):
    """Test ADWIN auto-reset behavior after drift detection."""

    def test_auto_reset_on_subsequent_update(self):
        """Test that state resets automatically after drift is detected."""
        detector = ADWIN(delta=0.002, grace_period=5)

        # Phase 1: Build up stable state with mean=0
        for _ in range(200):
            detector.update(0.0)
        width_before_shift = detector.width

        # Phase 2: Trigger drift with large mean shift
        drift_detected = False
        for _ in range(200):
            detector.update(10.0)
            if detector.drift_detected:
                drift_detected = True
                break

        self.assertTrue(drift_detected, "Drift should be detected")

        # Phase 3: Next update should auto-reset the detector
        # The width should be much smaller after reset
        detector.update(10.0)
        # After auto-reset, width should be small (just the new data)
        self.assertLess(detector.width, width_before_shift)

    def test_multiple_drift_detection(self):
        """Test that multiple drifts can be detected in sequence."""
        detector = ADWIN(delta=0.002, grace_period=5, clock=8)

        drifts_detected = []

        # Phase 1: Mean = 0
        for _ in range(100):
            detector.update(0.0)

        # Phase 2: Mean = 100 (first drift)
        for i in range(100):
            detector.update(100.0)
            if detector.drift_detected:
                drifts_detected.append(("phase2", i))
                # Auto-reset happens on next update

        # Phase 3: Mean = -100 (second drift)
        for i in range(100):
            detector.update(-100.0)
            if detector.drift_detected:
                drifts_detected.append(("phase3", i))

        # Phase 4: Mean = 500 (third drift)
        for i in range(100):
            detector.update(500.0)
            if detector.drift_detected:
                drifts_detected.append(("phase4", i))

        # Should detect at least 2 drifts (may detect more depending on sensitivity)
        self.assertGreaterEqual(
            len(drifts_detected),
            2,
            f"Should detect multiple drifts, got: {drifts_detected}",
        )

    def test_n_detections_increments_correctly(self):
        """Test that n_detections counts multiple drifts correctly."""
        detector = ADWIN(delta=0.01, grace_period=5)

        # Trigger multiple drifts
        for _phase, mean in enumerate([0, 100, -100, 200]):
            for _ in range(150):
                detector.update(float(mean))

        # Should have detected multiple drifts
        self.assertGreater(detector.n_detections, 1)


if __name__ == "__main__":
    unittest.main()
