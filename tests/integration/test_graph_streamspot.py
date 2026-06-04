"""Integration test for the StreamSpot graph model on a synthetic edge stream."""

import unittest

import numpy as np
from sklearn.metrics import average_precision_score

from aberrant.model.graph import SignedGraphSketchDetector as StreamSpot
from tests.integration._settings import MAX_TEST_SHORT, WARMUP_SAMPLES


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
        """Smoke-test StreamSpot quality on a synthetic stream."""
        rng = np.random.default_rng(42)
        model = StreamSpot(
            graph_key="graph",
            source_key="src",
            destination_key="dst",
            edge_type_key="etype",
            time_key="t",
            sketch_dim=256,
            shingle_size=2,
            num_clusters=8,
            max_graphs=256,
            warm_up_graphs=16,
            normalize_score=False,
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

        self.assertGreater(len(scores), 0, "No test samples were processed.")
        self.assertGreater(sum(labels), 0, "Test stream did not generate anomalies.")
        pr_auc = average_precision_score(labels, scores)

        lower_bound, upper_bound = 0.55, 1.00
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
