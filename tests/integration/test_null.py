"""Integration test for the NullModel."""

import unittest

from sklearn.metrics import average_precision_score

from aberrant.model.null import NullModel
from aberrant.stream.dataset import Dataset, load
from tests.integration._settings import MAX_TEST_STANDARD, WARMUP_SAMPLES


class TestNullModel(unittest.TestCase):
    """Test NullModel with the SHUTTLE dataset."""

    def test_shuttle_dataset_pr_auc(self):
        """
        Tests the NullModel, which should have a PR-AUC equal to the dataset's anomaly rate.
        """
        # Test configuration
        DATASET = Dataset.SHUTTLE

        # Create model
        model = NullModel()

        # Load dataset
        dataset_stream = load(DATASET)

        labels, scores = [], []
        warmup_count = 0
        test_count = 0

        # Process dataset stream
        for _i, (features, label) in enumerate(dataset_stream.stream()):
            if warmup_count < WARMUP_SAMPLES:
                if label == 0:
                    model.learn_one(features)
                    warmup_count += 1
                continue

            if test_count >= MAX_TEST_STANDARD:
                break

            score = model.score_one(features)
            model.learn_one(features)
            labels.append(label)
            scores.append(score)
            test_count += 1

        # Calculate and assert PR-AUC
        self.assertGreater(len(scores), 0, "No test samples were processed.")
        pr_auc = average_precision_score(labels, scores)

        # For a NullModel (always scoring 0), PR-AUC is equivalent to the anomaly rate in the test set.
        expected_pr_auc = sum(labels) / len(labels)

        self.assertAlmostEqual(
            pr_auc,
            expected_pr_auc,
            places=6,
            msg=f"PR-AUC {pr_auc:.10f} should be exactly {expected_pr_auc:.10f}",
        )


if __name__ == "__main__":
    unittest.main()
