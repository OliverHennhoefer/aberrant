"""Tests for Page-Hinkley drift detector."""

import unittest

from aberrant.drift.page_hinkley import PageHinkley


class TestPageHinkley(unittest.TestCase):
    """Test suite for Page-Hinkley drift detector."""

    def test_initialization_default_params(self):
        """Test PageHinkley initialization with default parameters."""
        detector = PageHinkley()
        self.assertEqual(detector.min_instances, 30)
        self.assertEqual(detector.delta, 0.005)
        self.assertEqual(detector.threshold, 50.0)
        self.assertEqual(detector.alpha, 0.9999)
        self.assertEqual(detector.mode, "both")
        self.assertFalse(detector.drift_detected)
        self.assertEqual(detector.n_detections, 0)

    def test_initialization_custom_params(self):
        """Test PageHinkley initialization with custom parameters."""
        detector = PageHinkley(
            min_instances=50,
            delta=0.01,
            threshold=100.0,
            alpha=0.999,
            mode="up",
        )
        self.assertEqual(detector.min_instances, 50)
        self.assertEqual(detector.delta, 0.01)
        self.assertEqual(detector.threshold, 100.0)
        self.assertEqual(detector.alpha, 0.999)
        self.assertEqual(detector.mode, "up")

    def test_invalid_min_instances(self):
        """Test that invalid min_instances raises ValueError."""
        with self.assertRaises(ValueError) as context:
            PageHinkley(min_instances=0)
        self.assertIn("min_instances must be positive", str(context.exception))

        with self.assertRaises(ValueError):
            PageHinkley(min_instances=-1)

    def test_invalid_delta(self):
        """Test that negative delta raises ValueError."""
        with self.assertRaises(ValueError) as context:
            PageHinkley(delta=-0.1)
        self.assertIn("delta must be non-negative", str(context.exception))

    def test_invalid_threshold(self):
        """Test that invalid threshold raises ValueError."""
        with self.assertRaises(ValueError) as context:
            PageHinkley(threshold=0)
        self.assertIn("threshold must be positive", str(context.exception))

        with self.assertRaises(ValueError):
            PageHinkley(threshold=-1)

    def test_invalid_alpha(self):
        """Test that invalid alpha raises ValueError."""
        with self.assertRaises(ValueError) as context:
            PageHinkley(alpha=0)
        self.assertIn("alpha must be in (0, 1]", str(context.exception))

        with self.assertRaises(ValueError):
            PageHinkley(alpha=1.1)

    def test_invalid_mode(self):
        """Test that invalid mode raises ValueError."""
        with self.assertRaises(ValueError) as context:
            PageHinkley(mode="invalid")
        self.assertIn("mode must be", str(context.exception))

    def test_update_returns_self(self):
        """Test that update returns self for method chaining."""
        detector = PageHinkley()
        result = detector.update(1.0)
        self.assertIs(result, detector)

    def test_non_finite_observation_is_rejected_without_mutating_state(self):
        detector = PageHinkley()
        detector.update(1.0)

        with self.assertRaisesRegex(ValueError, "observation must be finite"):
            detector.update(float("-inf"))

        self.assertEqual(detector._n, 1)
        self.assertEqual(detector.mean, 1.0)

    def test_mean_property(self):
        """Test the mean property tracks running mean."""
        detector = PageHinkley()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values:
            detector.update(v)
        self.assertAlmostEqual(detector.mean, 3.0, places=5)

    def test_no_drift_before_min_instances(self):
        """Test that no drift is detected before min_instances."""
        detector = PageHinkley(min_instances=100)
        # Add data with clear mean shift, but under min_instances
        for _ in range(50):
            detector.update(0.0)
        for _ in range(49):
            detector.update(100.0)
        self.assertFalse(detector.drift_detected)
        self.assertEqual(detector.n_detections, 0)

    def test_no_drift_on_stationary_data(self):
        """Test that no drift is detected on stationary data."""
        detector = PageHinkley(threshold=50.0)
        # Generate stationary data
        for _ in range(200):
            detector.update(0.0)
        self.assertEqual(detector.n_detections, 0)

    def test_drift_detected_mode_up(self):
        """Test drift detection in 'up' mode."""
        detector = PageHinkley(min_instances=20, threshold=30.0, delta=0.001, mode="up")
        # Stable phase
        for _ in range(50):
            detector.update(0.0)
        # Upward shift
        detected = False
        for _ in range(100):
            detector.update(5.0)
            if detector.drift_detected:
                detected = True
                break
        self.assertTrue(detected, "Drift should be detected on upward shift")

    def test_drift_detected_mode_down(self):
        """Test drift detection in 'down' mode."""
        detector = PageHinkley(
            min_instances=20, threshold=30.0, delta=0.001, mode="down"
        )
        # Stable phase at high value
        for _ in range(50):
            detector.update(10.0)
        # Downward shift
        detected = False
        for _ in range(100):
            detector.update(0.0)
            if detector.drift_detected:
                detected = True
                break
        self.assertTrue(detected, "Drift should be detected on downward shift")

    def test_drift_detected_mode_both(self):
        """Test drift detection in 'both' mode."""
        detector = PageHinkley(
            min_instances=20, threshold=30.0, delta=0.001, mode="both"
        )
        # Stable phase
        for _ in range(50):
            detector.update(5.0)
        # Shift (either direction should be detected)
        detected = False
        for _ in range(100):
            detector.update(15.0)
            if detector.drift_detected:
                detected = True
                break
        self.assertTrue(detected, "Drift should be detected in both mode")

    def test_reset(self):
        """Test reset clears all state."""
        detector = PageHinkley()
        for i in range(100):
            detector.update(float(i))
        detector.reset()
        self.assertEqual(detector._n, 0)
        self.assertEqual(detector.mean, 0.0)
        self.assertEqual(detector.n_detections, 0)
        self.assertFalse(detector.drift_detected)

    def test_method_chaining(self):
        """Test that methods can be chained."""
        detector = PageHinkley()
        result = detector.update(1.0).update(2.0).update(3.0)
        self.assertIs(result, detector)

    def test_repr(self):
        """Test string representation."""
        detector = PageHinkley(min_instances=50, threshold=100.0, mode="up")
        repr_str = repr(detector)
        self.assertIn("PageHinkley", repr_str)
        self.assertIn("min_instances=50", repr_str)
        self.assertIn("threshold=100.0", repr_str)
        self.assertIn("mode='up'", repr_str)


class TestPageHinkleyAutoReset(unittest.TestCase):
    """Test PageHinkley auto-reset behavior after drift detection."""

    def test_auto_reset_on_subsequent_update(self):
        """Test that state resets automatically after drift is detected."""
        detector = PageHinkley(min_instances=20, threshold=30.0, delta=0.001)

        # Phase 1: Build up stable state
        for _ in range(50):
            detector.update(0.0)
        n_before = detector._n

        # Phase 2: Trigger drift with large mean shift
        drift_detected = False
        for _ in range(100):
            detector.update(10.0)
            if detector.drift_detected:
                drift_detected = True
                break

        self.assertTrue(drift_detected, "Drift should be detected")

        # Phase 3: Next update should auto-reset the detector
        detector.update(10.0)
        # After auto-reset, _n should be small (just the new data)
        self.assertLess(detector._n, n_before)

    def test_multiple_drift_detection(self):
        """Test that multiple drifts can be detected in sequence."""
        detector = PageHinkley(min_instances=20, threshold=25.0, delta=0.001)

        drifts_detected = []

        # Phase 1: Mean = 0
        for _ in range(50):
            detector.update(0.0)

        # Phase 2: Mean = 10 (first drift)
        for i in range(100):
            detector.update(10.0)
            if detector.drift_detected:
                drifts_detected.append(("phase2", i))

        # Phase 3: Mean = -10 (second drift)
        for i in range(100):
            detector.update(-10.0)
            if detector.drift_detected:
                drifts_detected.append(("phase3", i))

        # Phase 4: Mean = 50 (third drift)
        for i in range(100):
            detector.update(50.0)
            if detector.drift_detected:
                drifts_detected.append(("phase4", i))

        # Should detect at least 2 drifts
        self.assertGreaterEqual(
            len(drifts_detected),
            2,
            f"Should detect multiple drifts, got: {drifts_detected}",
        )

    def test_n_detections_increments_correctly(self):
        """Test that n_detections counts multiple drifts correctly."""
        detector = PageHinkley(min_instances=20, threshold=25.0, delta=0.001)

        # Trigger multiple drifts
        for mean in [0, 20, -20, 40]:
            for _ in range(75):
                detector.update(float(mean))

        # Should have detected multiple drifts
        self.assertGreater(detector.n_detections, 1)


if __name__ == "__main__":
    unittest.main()
