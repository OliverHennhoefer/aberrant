"""Graph-edge detectors on a synthetic stream with injected edge bursts.

Normal traffic stays inside four communities; anomalous edges link nodes that
never talk or cross communities. MIDAS, ISCONNA and AnoEdge-L score single
edges; StreamSpot scores whole graphs built from typed edges. Every detector
scores an edge before it learns it, and learns every edge; labels only feed
the metrics. Run with ``uv run --extra eval python examples/graph.py``.
"""

import numpy as np

from aberrant.base import ModelProtocol
from aberrant.datasets import Sample
from aberrant.evaluate import PrequentialEvaluator
from aberrant.model.graph import ISCONNA, MIDAS, AnoEdgeL, StreamSpot


def _edges() -> list[Sample]:
    rng = np.random.default_rng(42)
    events = []
    for index in range(10_000):
        anomalous = index > 1_000 and rng.random() < 0.05
        if anomalous:
            source, target = int(rng.integers(1_000, 1_100)), int(rng.integers(0, 100))
        else:
            start = 25 * int(rng.integers(0, 4))
            local = int(rng.integers(0, 25))
            source, target = start + local, start + (local + 1) % 25
        edge = {"src": float(source), "dst": float(target), "t": float(index // 32)}
        events.append((edge, int(anomalous)))
    return events


def _graphs() -> list[Sample]:
    rng = np.random.default_rng(42)
    events = []
    for index in range(10_000):
        graph = int(rng.integers(0, 40))
        anomalous = index > 1_000 and rng.random() < 0.05
        if anomalous:
            source, target, kind = int(rng.integers(1_000, 1_250)), 2_000, 1
        else:
            source, target, kind = graph * 10, graph * 10 + 1, 0
        edge = {
            "graph": float(graph),
            "src": float(source),
            "dst": float(target),
            "etype": float(kind),
            "t": float(index // 8),
        }
        events.append((edge, int(anomalous)))
    return events


def _report(name: str, detector: ModelProtocol, events: list[Sample]) -> None:
    evaluator = PrequentialEvaluator(
        detector, warm_up=500, metrics=True, metric_window_size=None
    )
    result = evaluator.evaluate(events)
    print(
        f"{name:<10} average precision {result.average_precision:.3f}  "
        f"ROC AUC {result.roc_auc:.3f}"
    )


def main() -> None:
    edges = _edges()
    _report("MIDAS", MIDAS(seed=1), edges)
    _report("ISCONNA", ISCONNA(seed=1), edges)
    _report("AnoEdgeL", AnoEdgeL(seed=1), edges)
    _report(
        "StreamSpot",
        StreamSpot(edge_type_key="etype", max_graphs=256, warm_up=16, seed=1),
        _graphs(),
    )


if __name__ == "__main__":
    main()
