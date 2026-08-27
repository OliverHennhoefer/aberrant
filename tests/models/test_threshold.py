"""Tests for ThresholdModel baseline anomaly detector."""

import unittest

from aberrant.model.threshold import ThresholdModel
from tests.utils import DataGenerator


class TestThresholdModel(unittest.TestCase):
    """Test suite for ThresholdModel."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.data_generator = DataGenerator(seed=42)

    def test_initialization_validation(self):
        """Test that at least one parameter must be provided."""
        # Should raise ValueError when both are None
        with self.assertRaises(ValueError) as context:
            ThresholdModel()

        self.assertIn("At least one", str(context.exception))

        # Should succeed with only ceiling
        model1 = ThresholdModel(ceiling=10.0)
        self.assertIsNotNone(model1)

        # Should succeed with only floor
        model2 = ThresholdModel(floor=0.0)
        self.assertIsNotNone(model2)

        # Should succeed with both
        model3 = ThresholdModel(ceiling=10.0, floor=0.0)
        self.assertIsNotNone(model3)

    def test_ceiling_only_scalar(self):
        """Test scalar ceiling threshold detection."""
        model = ThresholdModel(ceiling=10.0)

        # Values below ceiling should return 0.0
        self.assertEqual(model.score_one({"x": 5.0}), 0.0)
        self.assertEqual(model.score_one({"x": 10.0}), 0.0)  # Exact boundary
        self.assertEqual(model.score_one({"x": 9.99}), 0.0)

        # Values above ceiling should return 1.0
        self.assertEqual(model.score_one({"x": 10.01}), 1.0)
        self.assertEqual(model.score_one({"x": 100.0}), 1.0)

        # Multiple features: any violation triggers anomaly
        self.assertEqual(model.score_one({"x": 5.0, "y": 5.0}), 0.0)
        self.assertEqual(model.score_one({"x": 5.0, "y": 15.0}), 1.0)
        self.assertEqual(model.score_one({"x": 15.0, "y": 5.0}), 1.0)

    def test_floor_only_scalar(self):
        """Test scalar floor threshold detection."""
        model = ThresholdModel(floor=0.0)

        # Values above floor should return 0.0
        self.assertEqual(model.score_one({"x": 5.0}), 0.0)
        self.assertEqual(model.score_one({"x": 0.0}), 0.0)  # Exact boundary
        self.assertEqual(model.score_one({"x": 0.01}), 0.0)

        # Values below floor should return 1.0
        self.assertEqual(model.score_one({"x": -0.01}), 1.0)
        self.assertEqual(model.score_one({"x": -100.0}), 1.0)

        # Multiple features: any violation triggers anomaly
        self.assertEqual(model.score_one({"x": 5.0, "y": 5.0}), 0.0)
        self.assertEqual(model.score_one({"x": 5.0, "y": -1.0}), 1.0)
        self.assertEqual(model.score_one({"x": -1.0, "y": 5.0}), 1.0)

    def test_corridor_scalar(self):
        """Test corridor (both ceiling and floor) with scalar values."""
        model = ThresholdModel(ceiling=10.0, floor=0.0)

        # Values within corridor should return 0.0
        self.assertEqual(model.score_one({"x": 5.0}), 0.0)
        self.assertEqual(model.score_one({"x": 0.0}), 0.0)  # Lower boundary
        self.assertEqual(model.score_one({"x": 10.0}), 0.0)  # Upper boundary

        # Values outside corridor should return 1.0
        self.assertEqual(model.score_one({"x": -0.01}), 1.0)  # Below floor
        self.assertEqual(model.score_one({"x": 10.01}), 1.0)  # Above ceiling

        # Multiple features
        self.assertEqual(model.score_one({"x": 5.0, "y": 8.0}), 0.0)  # Both ok
        self.assertEqual(model.score_one({"x": -1.0, "y": 5.0}), 1.0)  # x too low
        self.assertEqual(model.score_one({"x": 5.0, "y": 11.0}), 1.0)  # y too high

    def test_ceiling_dict_per_feature(self):
        """Test per-feature ceiling thresholds using dict."""
        model = ThresholdModel(ceiling={"x": 10.0, "y": 20.0})

        # Each feature checked against its own threshold
        self.assertEqual(model.score_one({"x": 5.0, "y": 15.0}), 0.0)
        self.assertEqual(model.score_one({"x": 10.0, "y": 20.0}), 0.0)

        # x violation
        self.assertEqual(model.score_one({"x": 10.01, "y": 15.0}), 1.0)

        # y violation
        self.assertEqual(model.score_one({"x": 5.0, "y": 20.01}), 1.0)

        # Both violations
        self.assertEqual(model.score_one({"x": 11.0, "y": 21.0}), 1.0)

        # Feature not in dict is ignored
        self.assertEqual(model.score_one({"x": 5.0, "y": 15.0, "z": 1000.0}), 0.0)

    def test_floor_dict_per_feature(self):
        """Test per-feature floor thresholds using dict."""
        model = ThresholdModel(floor={"x": 0.0, "y": 10.0})

        # Each feature checked against its own threshold
        self.assertEqual(model.score_one({"x": 5.0, "y": 15.0}), 0.0)
        self.assertEqual(model.score_one({"x": 0.0, "y": 10.0}), 0.0)

        # x violation
        self.assertEqual(model.score_one({"x": -0.01, "y": 15.0}), 1.0)

        # y violation
        self.assertEqual(model.score_one({"x": 5.0, "y": 9.99}), 1.0)

        # Both violations
        self.assertEqual(model.score_one({"x": -1.0, "y": 5.0}), 1.0)

        # Feature not in dict is ignored
        self.assertEqual(model.score_one({"x": 5.0, "y": 15.0, "z": -1000.0}), 0.0)

    def test_corridor_dict_per_feature(self):
        """Test corridor with per-feature dict thresholds."""
        model = ThresholdModel(
            ceiling={"temp": 100.0, "pressure": 50.0},
            floor={"temp": 0.0, "pressure": 10.0},
        )

        # Within corridor
        self.assertEqual(model.score_one({"temp": 50.0, "pressure": 30.0}), 0.0)
        self.assertEqual(model.score_one({"temp": 0.0, "pressure": 10.0}), 0.0)
        self.assertEqual(model.score_one({"temp": 100.0, "pressure": 50.0}), 0.0)

        # Temperature violations
        self.assertEqual(model.score_one({"temp": -0.01, "pressure": 30.0}), 1.0)
        self.assertEqual(model.score_one({"temp": 100.01, "pressure": 30.0}), 1.0)

        # Pressure violations
        self.assertEqual(model.score_one({"temp": 50.0, "pressure": 9.99}), 1.0)
        self.assertEqual(model.score_one({"temp": 50.0, "pressure": 50.01}), 1.0)

    def test_mixed_scalar_and_dict(self):
        """Test mixing scalar and dict parameter types."""
        # Scalar ceiling, dict floor
        model1 = ThresholdModel(
            ceiling=100.0,
            floor={"x": 0.0, "y": 10.0},  # Applies to all
        )

        self.assertEqual(model1.score_one({"x": 50.0, "y": 50.0}), 0.0)
        self.assertEqual(model1.score_one({"x": 101.0, "y": 50.0}), 1.0)  # ceiling
        self.assertEqual(model1.score_one({"x": -1.0, "y": 50.0}), 1.0)  # x floor
        self.assertEqual(model1.score_one({"x": 50.0, "y": 9.0}), 1.0)  # y floor

        # Dict ceiling, scalar floor
        model2 = ThresholdModel(ceiling={"x": 10.0, "y": 20.0}, floor=0.0)

        self.assertEqual(model2.score_one({"x": 5.0, "y": 15.0}), 0.0)
        self.assertEqual(model2.score_one({"x": 11.0, "y": 15.0}), 1.0)  # x ceiling
        self.assertEqual(model2.score_one({"x": 5.0, "y": 21.0}), 1.0)  # y ceiling
        self.assertEqual(model2.score_one({"x": -1.0, "y": 15.0}), 1.0)  # floor

    def test_exact_boundary_values(self):
        """Test values exactly at threshold boundaries."""
        model = ThresholdModel(ceiling=10.0, floor=0.0)

        # Exact boundaries should NOT trigger anomaly (non-strict inequalities)
        self.assertEqual(model.score_one({"x": 0.0}), 0.0)
        self.assertEqual(model.score_one({"x": 10.0}), 0.0)

        # Just beyond boundaries should trigger
        self.assertEqual(model.score_one({"x": -1e-10}), 1.0)
        self.assertEqual(model.score_one({"x": 10.0 + 1e-10}), 1.0)

    def test_no_violation_returns_zero(self):
        """Test that no violations return 0.0."""
        model = ThresholdModel(ceiling=100.0, floor=-100.0)

        test_data = self.data_generator.generate_streaming_data(n=100)

        for point in test_data:
            # Scale values to be within range
            scaled_point = {k: v * 10 for k, v in point.items()}  # Scale to ~[-30, 30]
            score = model.score_one(scaled_point)
            self.assertEqual(score, 0.0)

    def test_learn_one_does_nothing(self):
        """Test that learn_one doesn't change model state."""
        model = ThresholdModel(ceiling=10.0, floor=0.0)

        # Test point that should trigger anomaly
        test_point = {"x": 15.0}

        # Score before learning
        score_before = model.score_one(test_point)
        self.assertEqual(score_before, 1.0)

        # Learn many data points
        test_data = self.data_generator.generate_streaming_data(n=100)
        for point in test_data:
            model.learn_one(point)

        # Score should be unchanged
        score_after = model.score_one(test_point)
        self.assertEqual(score_after, 1.0)

        # Representation should be unchanged
        initial_repr = repr(ThresholdModel(ceiling=10.0, floor=0.0))
        self.assertEqual(repr(model), initial_repr)

    def test_repr_string(self):
        """Test string representation with various configurations."""
        # Ceiling only
        model1 = ThresholdModel(ceiling=10.0)
        self.assertEqual(repr(model1), "ThresholdModel(ceiling=10.0, floor=None)")

        # Floor only
        model2 = ThresholdModel(floor=0.0)
        self.assertEqual(repr(model2), "ThresholdModel(ceiling=None, floor=0.0)")

        # Both
        model3 = ThresholdModel(ceiling=10.0, floor=0.0)
        self.assertEqual(repr(model3), "ThresholdModel(ceiling=10.0, floor=0.0)")

        # Dict parameters
        model4 = ThresholdModel(ceiling={"x": 10.0})
        self.assertEqual(
            repr(model4), "ThresholdModel(ceiling={'x': 10.0}, floor=None)"
        )

    def test_consistency_across_features(self):
        """Test that all features are checked consistently."""
        model = ThresholdModel(ceiling=10.0)

        # Single feature
        self.assertEqual(model.score_one({"a": 5.0}), 0.0)
        self.assertEqual(model.score_one({"a": 15.0}), 1.0)

        # Multiple features - first violates
        self.assertEqual(model.score_one({"a": 15.0, "b": 5.0, "c": 5.0}), 1.0)

        # Multiple features - middle violates
        self.assertEqual(model.score_one({"a": 5.0, "b": 15.0, "c": 5.0}), 1.0)

        # Multiple features - last violates
        self.assertEqual(model.score_one({"a": 5.0, "b": 5.0, "c": 15.0}), 1.0)

        # Multiple features - multiple violate (returns on first)
        self.assertEqual(model.score_one({"a": 15.0, "b": 15.0}), 1.0)

    def test_edge_cases(self):
        """Test edge cases for ThresholdModel."""
        model = ThresholdModel(ceiling=10.0, floor=0.0)

        # Extreme values
        extreme_cases = [
            ({"x": 1e10}, 1.0),  # Very large
            ({"x": -1e10}, 1.0),  # Very negative
            ({"x": 1e-10}, 0.0),  # Very small positive
            ({"x": 5.0}, 0.0),  # Normal
        ]

        for case, expected in extreme_cases:
            with self.subTest(case=case):
                score = model.score_one(case)
                self.assertEqual(score, expected)

    def test_empty_dict_handling(self):
        """Test behavior with empty feature dict."""
        model = ThresholdModel(ceiling=10.0)

        # Empty dict should return 0.0 (no features to violate)
        score = model.score_one({})
        self.assertEqual(score, 0.0)

    def test_multiple_instances_independent(self):
        """Test that multiple ThresholdModel instances are independent."""
        model1 = ThresholdModel(ceiling=10.0)
        model2 = ThresholdModel(ceiling=20.0)

        test_point = {"x": 15.0}

        # Different thresholds should give different results
        score1 = model1.score_one(test_point)
        score2 = model2.score_one(test_point)

        self.assertEqual(score1, 1.0)  # Violates ceiling=10
        self.assertEqual(score2, 0.0)  # Does not violate ceiling=20

    def test_threshold_mappings_are_owned_by_the_model(self):
        ceiling = {"x": 10.0}
        floor = {"x": 0.0}
        model = ThresholdModel(ceiling=ceiling, floor=floor)

        ceiling["x"] = 100.0
        floor["x"] = -100.0

        self.assertEqual(model.score_one({"x": 11.0}), 1.0)
        self.assertEqual(model.score_one({"x": -1.0}), 1.0)

    def test_streaming_consistency(self):
        """Test that model behaves consistently in streaming scenario."""
        model = ThresholdModel(ceiling=5.0, floor=0.0)

        test_data = [
            ({"x": 3.0}, 0.0),
            ({"x": 6.0}, 1.0),
            ({"x": -1.0}, 1.0),
            ({"x": 0.0}, 0.0),
            ({"x": 5.0}, 0.0),
            ({"x": 5.01}, 1.0),
        ]

        for i, (point, expected) in enumerate(test_data):
            with self.subTest(i=i):
                model.learn_one(point)  # Learning does nothing but test it
                score = model.score_one(point)
                self.assertEqual(score, expected)

    def test_per_feature_threshold_independence(self):
        """Test that per-feature thresholds are independent."""
        model = ThresholdModel(
            ceiling={"temp": 100.0, "pressure": 50.0},
            floor={"temp": 0.0, "pressure": 10.0},
        )

        # Features with no specified threshold should be ignored
        self.assertEqual(model.score_one({"humidity": 1000.0}), 0.0)
        self.assertEqual(model.score_one({"humidity": -1000.0}), 0.0)

        # Mix of specified and unspecified features
        self.assertEqual(
            model.score_one({"temp": 50.0, "pressure": 30.0, "humidity": 1000.0}), 0.0
        )
        self.assertEqual(
            model.score_one({"temp": 101.0, "pressure": 30.0, "humidity": 1000.0}), 1.0
        )

    def test_large_feature_set(self):
        """Test with many features."""
        # Create thresholds for many features
        n_features = 50
        ceiling_dict = {f"feature_{i}": 10.0 for i in range(n_features)}
        floor_dict = {f"feature_{i}": 0.0 for i in range(n_features)}

        model = ThresholdModel(ceiling=ceiling_dict, floor=floor_dict)

        # All within bounds
        point_ok = {f"feature_{i}": 5.0 for i in range(n_features)}
        self.assertEqual(model.score_one(point_ok), 0.0)

        # One feature violates
        point_bad = {f"feature_{i}": 5.0 for i in range(n_features)}
        point_bad["feature_25"] = 11.0
        self.assertEqual(model.score_one(point_bad), 1.0)


if __name__ == "__main__":
    unittest.main()
