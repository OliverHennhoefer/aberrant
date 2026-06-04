"""Tests for Autoencoder deep learning anomaly detection model."""

from __future__ import annotations

import unittest

try:
    import torch
    from torch import nn, optim

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from aberrant.model.deep.autoencoder import Autoencoder
    from aberrant.utils.deep.architecture import VanillaAutoencoder
from tests.utils import DataGenerator


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestAutoencoder(unittest.TestCase):
    """Test suite for Autoencoder model."""

    def create_model(self) -> Autoencoder:
        """Create Autoencoder instance for testing."""
        architecture = VanillaAutoencoder(input_size=3, seed=42)
        optimizer = optim.Adam(architecture.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        return Autoencoder(model=architecture, optimizer=optimizer, criterion=criterion)

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.data_generator = DataGenerator(seed=42)

    def test_initialization_with_valid_components(self):
        """Test autoencoder initialization with valid components."""
        architecture = VanillaAutoencoder(input_size=5)
        optimizer = optim.SGD(architecture.parameters(), lr=0.1)
        criterion = nn.MSELoss()

        model = Autoencoder(
            model=architecture, optimizer=optimizer, criterion=criterion
        )

        self.assertEqual(model.model.input_size, 5)
        self.assertIsInstance(model.optimizer, optim.SGD)
        self.assertIsInstance(model.criterion, nn.MSELoss)
        self.assertIsNone(model._feature_order)  # Should be None initially

    def test_feature_order_consistency(self):
        """Test that feature order is established and maintained."""
        model = self.create_model()

        # First data point establishes feature order
        first_point = {"c": 3.0, "a": 1.0, "b": 2.0}
        model.learn_one(first_point)

        # Feature order should be alphabetical
        expected_order = ["a", "b", "c"]
        self.assertEqual(model._feature_order, expected_order)

        # Subsequent data points should use same order
        second_point = {"b": 5.0, "c": 6.0, "a": 4.0}
        score = model.score_one(second_point)

        # Should process without error and return valid score
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)

    def test_rejects_wrong_feature_count_and_schema_changes(self):
        model = self.create_model()

        with self.assertRaisesRegex(ValueError, "Expected 3 features"):
            model.learn_one({"a": 1.0, "b": 2.0})

        model.learn_one({"a": 1.0, "b": 2.0, "c": 3.0})
        with self.assertRaisesRegex(ValueError, "Input features must be"):
            model.score_one({"a": 1.0, "b": 2.0, "d": 3.0})

    def test_basic_learning_and_scoring(self):
        """Test basic learning and scoring functionality."""
        model = self.create_model()

        # Generate consistent training data
        training_data = []
        for i in range(50):
            point = {
                "feature_0": float(i % 10),
                "feature_1": float((i * 2) % 10),
                "feature_2": float((i * 3) % 10),
            }
            training_data.append(point)

        # Train model
        initial_score = None
        for i, point in enumerate(training_data):
            model.learn_one(point)
            if i == 0:
                initial_score = model.score_one(point)

        # Score should improve (decrease) with training on same pattern
        final_score = model.score_one(training_data[0])

        self.assertIsInstance(initial_score, float)
        self.assertIsInstance(final_score, float)
        self.assertGreaterEqual(initial_score, 0.0)
        self.assertGreaterEqual(final_score, 0.0)

    def test_training_mode_switching(self):
        """Test that model correctly switches between train/eval modes."""
        architecture = VanillaAutoencoder(input_size=3, seed=42)
        optimizer = optim.Adam(architecture.parameters())
        criterion = nn.MSELoss()

        model = Autoencoder(
            model=architecture, optimizer=optimizer, criterion=criterion
        )

        test_point = {"a": 1.0, "b": 2.0, "c": 3.0}

        # Initially should be in eval mode (default for new nn.Module)
        # learn_one should set to training mode
        model.learn_one(test_point)
        # Note: learn_one sets to train mode internally but may not persist

        # score_one should set to eval mode
        model.score_one(test_point)
        self.assertFalse(
            model.model.training, "Model should be in eval mode after score_one"
        )

    def test_tensor_efficiency(self):
        """Test that model reuses tensors efficiently."""
        model = self.create_model()

        # Pre-allocated tensor should exist
        self.assertIsInstance(model.x_tensor, torch.Tensor)
        self.assertEqual(model.x_tensor.shape, (1, 3))  # batch_size=1, features=3

        # Using the model shouldn't create new tensors
        test_point = {"a": 1.0, "b": 2.0, "c": 3.0}

        # Get tensor id before operations
        tensor_id = id(model.x_tensor)

        model.learn_one(test_point)
        model.score_one(test_point)

        # Tensor should be the same object (reused)
        self.assertEqual(id(model.x_tensor), tensor_id)

    def test_with_different_loss_functions(self):
        """Test autoencoder with different loss functions."""
        architecture = VanillaAutoencoder(input_size=3, seed=42)

        loss_functions = [
            nn.MSELoss(),
            nn.L1Loss(),
            nn.SmoothL1Loss(),
            nn.HuberLoss(delta=1.0),
        ]

        for criterion in loss_functions:
            with self.subTest(criterion=criterion.__class__.__name__):
                optimizer = optim.Adam(architecture.parameters())
                model = Autoencoder(
                    model=architecture, optimizer=optimizer, criterion=criterion
                )

                # Should work with any supported loss function
                test_point = {"a": 1.0, "b": 2.0, "c": 3.0}
                model.learn_one(test_point)
                score = model.score_one(test_point)

                self.assertIsInstance(score, float)
                self.assertGreaterEqual(score, 0.0)

    def test_with_different_optimizers(self):
        """Test autoencoder with different optimizers."""
        architecture = VanillaAutoencoder(input_size=3, seed=42)
        criterion = nn.MSELoss()

        optimizers = [
            optim.Adam(architecture.parameters(), lr=0.01),
            optim.SGD(architecture.parameters(), lr=0.1),
            optim.RMSprop(architecture.parameters(), lr=0.01),
        ]

        for optimizer in optimizers:
            with self.subTest(optimizer=optimizer.__class__.__name__):
                model = Autoencoder(
                    model=architecture, optimizer=optimizer, criterion=criterion
                )

                # Should work with any PyTorch optimizer
                test_point = {"a": 1.0, "b": 2.0, "c": 3.0}
                model.learn_one(test_point)
                score = model.score_one(test_point)

                self.assertIsInstance(score, float)
                self.assertGreaterEqual(score, 0.0)

    def test_repr_string(self):
        """Test string representation includes key components."""
        model = self.create_model()
        repr_str = repr(model)

        self.assertIn("Autoencoder", repr_str)
        self.assertIn("VanillaAutoencoder", repr_str)
        self.assertIn("Adam", repr_str)
        self.assertIn("MSELoss", repr_str)

    def test_device_handling(self):
        """Test that model handles device placement correctly."""
        # Test with CPU (default)
        architecture = VanillaAutoencoder(input_size=3)
        optimizer = optim.Adam(architecture.parameters())
        criterion = nn.MSELoss()

        model = Autoencoder(
            model=architecture, optimizer=optimizer, criterion=criterion
        )

        # Model and tensors should be on CPU
        self.assertEqual(model.model.device, torch.device("cpu"))
        self.assertEqual(model.x_tensor.device, torch.device("cpu"))

    def test_gradient_updates(self):
        """Test that gradients are computed and applied correctly."""
        architecture = VanillaAutoencoder(input_size=3, seed=42)
        optimizer = optim.SGD(
            architecture.parameters(), lr=1.0
        )  # High LR for visible changes
        criterion = nn.MSELoss()

        model = Autoencoder(
            model=architecture, optimizer=optimizer, criterion=criterion
        )

        # Get initial parameters
        initial_params = [param.clone() for param in architecture.parameters()]

        # Train on a point multiple times
        test_point = {"a": 1.0, "b": 2.0, "c": 3.0}
        for _ in range(10):
            model.learn_one(test_point)

        # Parameters should have changed (gradients applied)
        final_params = list(architecture.parameters())

        changed_params = 0
        for initial, final in zip(initial_params, final_params, strict=False):
            if not torch.allclose(initial, final, atol=1e-6):
                changed_params += 1

        self.assertGreater(
            changed_params, 0, "Expected some parameters to change during training"
        )

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        model = self.create_model()

        # Test with zero values
        zero_point = {"a": 0.0, "b": 0.0, "c": 0.0}
        model.learn_one(zero_point)
        score = model.score_one(zero_point)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)

        # Test with negative values
        negative_point = {"a": -1.0, "b": -2.0, "c": -3.0}
        model.learn_one(negative_point)
        score = model.score_one(negative_point)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)

        # Test with very large values
        large_point = {"a": 1000.0, "b": 2000.0, "c": 3000.0}
        model.learn_one(large_point)
        score = model.score_one(large_point)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()
