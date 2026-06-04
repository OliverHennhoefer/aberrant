"""Integration test for the StreamRandomHistogramForest model."""

import unittest

from sklearn.metrics import average_precision_score

from aberrant.model.iforest.rand_hist import StreamRandomHistogramForest
from aberrant.stream.dataset import Dataset, load
from tests.integration._settings import MAX_TEST_SHORT, WARMUP_SAMPLES


class TestStreamRandomHistogramForest(unittest.TestCase):
    """Test StreamRandomHistogramForest with the SHUTTLE dataset."""

    def test_shuttle_dataset_pr_auc(self):
        """
        Tests the StreamRandomHistogramForest model on the SHUTTLE dataset and snapshots the PR-AUC score.
        """
        # Test configuration
        SEED = 42
        DATASET = Dataset.SHUTTLE

        # Create model
        model = StreamRandomHistogramForest(
            n_estimators=25, max_depth=10, window_size=256, seed=SEED
        )

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

            if test_count >= MAX_TEST_SHORT:
                break

            score = model.score_one(features)
            model.learn_one(features)
            labels.append(label)
            scores.append(score)
            test_count += 1

        # Calculate and assert PR-AUC
        self.assertGreater(len(scores), 0, "No test samples were processed.")
        pr_auc = average_precision_score(labels, scores)
        lower_bound, upper_bound = 0.90, 1.00
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
