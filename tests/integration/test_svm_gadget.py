"""Integration test for the graph-gated one-class SVM model."""

import unittest

from sklearn.metrics import average_precision_score

from aberrant.model.svm.gadget import GraphGatedOneClassSVM
from aberrant.stream.dataset import Dataset, load
from tests.integration._settings import MAX_TEST_STANDARD, WARMUP_SAMPLES


class TestGadgetSVM(unittest.TestCase):
    """Test GraphGatedOneClassSVM with the SHUTTLE dataset."""

    def test_shuttle_dataset_pr_auc(self):
        """
        Tests the graph-gated SVM model on SHUTTLE and snapshots PR-AUC.
        """
        # Test configuration
        DATASET = Dataset.SHUTTLE

        # Create model
        model = GraphGatedOneClassSVM(nu=0.1, learning_rate=0.01, lambda_reg=0.01)

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
        lower_bound, upper_bound = 0.05, 0.15
        self.assertGreaterEqual(
            pr_auc,
            lower_bound,
            f"PR-AUC {pr_auc:.3f} is below expected range [{lower_bound}, {upper_bound}]",
        )
        self.assertLessEqual(
            pr_auc,
            upper_bound,
            f"PR-AUC {pr_auc:.3f} is above expected range [{lower_bound}, {upper_bound}]",
        )


if __name__ == "__main__":
    unittest.main()
