"""Integration test for the LODA model."""

import unittest

from sklearn.metrics import average_precision_score

from aberrant.model.sketch import StreamingLODA as LODA
from aberrant.stream.dataset import Dataset, load
from aberrant.transform.preprocessing import StandardScaler
from tests.integration._settings import MAX_TEST_SHORT, WARMUP_SAMPLES


class TestLODA(unittest.TestCase):
    """Test LODA on the SHUTTLE dataset."""

    def test_shuttle_dataset_pr_auc(self) -> None:
        """Smoke-test LODA quality on SHUTTLE with bounded PR-AUC range."""
        dataset_stream = load(Dataset.SHUTTLE)
        model = StandardScaler() | LODA(
            n_projections=64,
            n_bins=24,
            sparsity=0.3,
            warm_up_samples=256,
            decay=1.0,
            time_key=None,
            pseudocount=0.5,
            seed=42,
        )

        labels: list[int] = []
        scores: list[float] = []
        warmup_count = 0
        test_count = 0

        for features, label in dataset_stream.stream():
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

        self.assertGreater(len(scores), 0, "No test samples were processed.")
        pr_auc = average_precision_score(labels, scores)

        lower_bound, upper_bound = 0.55, 0.99
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
