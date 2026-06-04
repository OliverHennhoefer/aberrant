"""Integration tests for univariate statistical models."""

import unittest

from sklearn.metrics import average_precision_score

from aberrant.model.stat.uni import MovingAverage, MovingMedian, MovingVariance
from aberrant.stream.dataset import Dataset, load
from tests.integration._settings import MAX_TEST_STANDARD, WARMUP_SAMPLES


class TestUnivariateStatisticalModels(unittest.TestCase):
    """
    Tests univariate statistical models on the SHUTTLE dataset.
    Each test extracts the first feature to create a univariate stream.
    """

    def setUp(self):
        """Set up test configuration."""
        self.WARMUP_SAMPLES = WARMUP_SAMPLES
        self.MAX_TEST_SAMPLES = MAX_TEST_STANDARD
        self.dataset = load(Dataset.SHUTTLE)

    def run_univariate_test(self, model, expected_range):
        """Helper to run a test for any univariate statistical model."""
        labels, scores = [], []
        warmup_count = 0
        test_count = 0

        # The dataset stream needs to be re-initialized for each test
        dataset_stream = load(Dataset.SHUTTLE)

        for _i, (features, label) in enumerate(dataset_stream.stream()):
            # Prepare univariate data by extracting the first feature
            first_feature_key = list(features.keys())[0]
            x_uni = {first_feature_key: features[first_feature_key]}

            if warmup_count < self.WARMUP_SAMPLES:
                if label == 0:
                    model.learn_one(x_uni)
                    warmup_count += 1
                continue

            if test_count >= self.MAX_TEST_SAMPLES:
                break

            score = model.score_one(x_uni)
            model.learn_one(x_uni)
            labels.append(label)
            scores.append(score)
            test_count += 1

        self.assertGreater(len(scores), 0, "No scores were generated.")
        pr_auc = average_precision_score(labels, scores)

        lower, upper = expected_range
        self.assertGreaterEqual(
            pr_auc,
            lower,
            f"PR-AUC {pr_auc:.3f} is below expected range [{lower}, {upper}] for {model.__class__.__name__}",
        )
        self.assertLessEqual(
            pr_auc,
            upper,
            f"PR-AUC {pr_auc:.3f} is above expected range [{lower}, {upper}] for {model.__class__.__name__}",
        )

    def test_moving_average(self):
        model = MovingAverage(window_size=100)
        self.run_univariate_test(model, expected_range=(0.95, 1.0))

    def test_moving_variance(self):
        model = MovingVariance(window_size=100)
        self.run_univariate_test(model, expected_range=(0.9, 0.99))

    def test_moving_median(self):
        model = MovingMedian(window_size=100)
        self.run_univariate_test(model, expected_range=(0.05, 0.15))


if __name__ == "__main__":
    unittest.main()
