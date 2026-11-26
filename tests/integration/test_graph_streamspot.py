"""Integration test for the StreamSpot graph model on a synthetic edge stream."""

import unittest

import numpy as np

from aberrant.model.graph import StreamSpot
from tests.integration._quality import assert_detects

WARMUP_SAMPLES = 1_000
MAX_TEST_SHORT = 500
# Measured on main at ROC-AUC 0.953 and PR-AUC 0.591; the floors leave a margin.
ROC_FLOOR = 0.90
PR_FLOOR = 0.52


class TestStreamSpot(unittest.TestCase):
    """Test StreamSpot quality on a synthetic dynamic graph stream."""

    @staticmethod
    def _sample_edge(
        rng: np.random.Generator,
        t: int,
        anomaly_rate: float,
    ) -> tuple[dict[str, float], int]:
        graph_id = int(rng.integers(0, 40))
        bucket = t // 8

        if rng.random() < anomaly_rate:
            src = int(rng.integers(1_000, 1_250))
            dst = int(rng.integers(2_000, 2_250))
            etype = 1
            label = 1
        else:
            # Stable motif per graph under nominal behavior.
            src = graph_id * 10
            dst = graph_id * 10 + 1
            etype = 0
            label = 0

        sample = {
            "graph": float(graph_id),
            "src": float(src),
            "dst": float(dst),
            "etype": float(etype),
            "t": float(bucket),
        }
        return sample, label

    def test_synthetic_edge_stream_pr_auc(self) -> None:
        """Smoke-test StreamSpot quality on a synthetic stream against quality floors."""
        rng = np.random.default_rng(42)
        model = StreamSpot(
            graph_key="graph",
            source_key="src",
            destination_key="dst",
            edge_type_key="etype",
            time_key="t",
            sketch_dim=256,
            shingle_size=2,
            n_clusters=8,
            max_graphs=256,
            warm_up=16,
            seed=42,
        )

        labels: list[int] = []
        scores: list[float] = []
        warmup_count = 0
        test_count = 0
        t = 0

        while warmup_count < WARMUP_SAMPLES:
            sample, _label = self._sample_edge(rng, t=t, anomaly_rate=0.0)
            model.learn_one(sample)
            warmup_count += 1
            t += 1

        while test_count < MAX_TEST_SHORT:
            sample, label = self._sample_edge(rng, t=t, anomaly_rate=0.15)
            score = model.score_one(sample)
            model.learn_one(sample)
            labels.append(label)
            scores.append(score)
            test_count += 1
            t += 1

        assert_detects(labels, scores, roc_auc_floor=ROC_FLOOR, pr_auc_floor=PR_FLOOR)


if __name__ == "__main__":
    unittest.main()
