"""Integration test for the AnoEdge-L graph model."""

import unittest

from aberrant.model.graph import AnoEdgeL


class TestAnoEdgeL(unittest.TestCase):
    """Test AnoEdge-L on its dense-edge anomaly target."""

    def test_dense_microcluster_scores_above_sparse_edge(self) -> None:
        model = AnoEdgeL(
            count_min_rows=128,
            count_min_cols=128,
            num_hashes=4,
            num_dense_submatrices=1,
            time_decay_factor=1.0,
            seed=42,
        )

        for _ in range(12):
            for src in range(5):
                for dst in range(100, 105):
                    model.learn_one({"src": float(src), "dst": float(dst), "t": 1.0})

        dense_score = model.score_one({"src": 2.0, "dst": 102.0, "t": 1.0})
        sparse_score = model.score_one({"src": 999.0, "dst": 998.0, "t": 1.0})

        self.assertGreater(dense_score, sparse_score)


if __name__ == "__main__":
    unittest.main()
