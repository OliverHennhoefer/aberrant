"""Integration test for the MIDAS graph model on a synthetic edge stream."""

import unittest

import numpy as np

from aberrant.model.graph import MIDAS
from tests.integration._quality import assert_detects

WARMUP_SAMPLES = 1_000
MAX_TEST_SHORT = 500
# Measured on main at ROC-AUC 0.989 and PR-AUC 0.870; the floors leave a margin.
ROC_FLOOR = 0.95
PR_FLOOR = 0.80


class TestMIDAS(unittest.TestCase):
    """Test MIDAS quality on a synthetic dynamic graph stream."""

    @staticmethod
    def _sample_edge(
        rng: np.random.Generator,
        t: int,
        anomaly_rate: float,
    ) -> tuple[dict[str, float], int]:
        bucket = t // 32
        if rng.random() < anomaly_rate:
            if rng.random() < 0.5:
                src = int(rng.integers(1_000, 1_100))
                dst = int(rng.integers(1_100, 1_200))
            else:
                src = int(rng.integers(200, 260))
                dst = int(rng.integers(260, 320))
            return {"src": float(src), "dst": float(dst), "t": float(bucket)}, 1

        # Nominal behavior: repeated community motifs.
        community = int(rng.integers(0, 4))
        start = community * 25
        src_local = int(rng.integers(0, 25))
        dst_local = (src_local + 1) % 25 if rng.random() < 0.9 else (src_local + 2) % 25
        src = start + src_local
        dst = start + dst_local
        return {"src": float(src), "dst": float(dst), "t": float(bucket)}, 0

    def test_synthetic_edge_stream_pr_auc(self) -> None:
        """Smoke-test MIDAS quality on a synthetic stream against quality floors."""
        rng = np.random.default_rng(42)
        model = MIDAS(
            source_key="src",
            destination_key="dst",
            time_key="t",
            n_hashes=4,
            sketch_width=1024,
            warm_up=128,
            use_relational=True,
            normalize_score=False,
            seed=42,
        )

        labels: list[int] = []
        scores: list[float] = []
        warmup_count = 0
        test_count = 0
        t = 0

        while warmup_count < WARMUP_SAMPLES:
            sample, label = self._sample_edge(rng, t=t, anomaly_rate=0.0)
            if label == 0:
                model.learn_one(sample)
                warmup_count += 1
            t += 1

        while test_count < MAX_TEST_SHORT:
            sample, label = self._sample_edge(rng, t=t, anomaly_rate=0.12)
            score = model.score_one(sample)
            model.learn_one(sample)
            labels.append(label)
            scores.append(score)
            test_count += 1
            t += 1

        assert_detects(labels, scores, roc_auc_floor=ROC_FLOOR, pr_auc_floor=PR_FLOOR)


if __name__ == "__main__":
    unittest.main()
