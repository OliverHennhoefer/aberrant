"""Integration test for the RSHash model."""

import unittest

from sklearn.metrics import average_precision_score

from aberrant.model.sketch import StreamingRSHash as RSHash
from aberrant.stream.dataset import Dataset, load
from tests.integration._settings import MAX_TEST_SHORT, WARMUP_SAMPLES


class TestRSHash(unittest.TestCase):
    """Test RSHash on the SHUTTLE dataset."""

    def test_shuttle_dataset_pr_auc(self) -> None:
        """Smoke-test RSHash quality on SHUTTLE with bounded PR-AUC range."""
        dataset_stream = load(Dataset.SHUTTLE)
        model = RSHash(
            components_num=24,
            hash_num=4,
            bins=512,
            subspace_size=3,
            bin_width=1.0,
            decay=0.01,
            warm_up_samples=128,
            time_key="t",
            seed=42,
        )

        labels: list[int] = []
        scores: list[float] = []
        warmup_count = 0
        test_count = 0

        for i, (features, label) in enumerate(dataset_stream.stream()):
            sample = dict(features)
            sample["t"] = float(i)

            if warmup_count < WARMUP_SAMPLES:
                if label == 0:
                    model.learn_one(sample)
                    warmup_count += 1
                continue

            if test_count >= MAX_TEST_SHORT:
                break

            score = model.score_one(sample)
            model.learn_one(sample)
            labels.append(label)
            scores.append(score)
            test_count += 1

        self.assertGreater(len(scores), 0, "No test samples were processed.")
        pr_auc = average_precision_score(labels, scores)

        lower_bound, upper_bound = 0.55, 0.98
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
