"""Statistical detectors: multivariate tails on Shuttle, univariate spikes.

The empirical-tail detectors and the moving Mahalanobis distance score the
first 20,000 Shuttle events. The moving z-score and its robust median/MAD
variant score a synthetic sensor reading with injected spikes. Every detector
scores an event before it learns it, and learns every event; labels only feed
the metrics. Run with ``uv run --extra eval python examples/stat.py``.
"""

from itertools import islice

import numpy as np

from aberrant.base import ModelProtocol
from aberrant.datasets import Dataset, Sample, load
from aberrant.evaluate import PrequentialEvaluator
from aberrant.model.stat import (
    MovingMAD,
    MovingMahalanobisDistance,
    MovingZScore,
    OnlineCOPOD,
    OnlineECOD,
)


def _sensor() -> list[Sample]:
    rng = np.random.default_rng(7)
    events = []
    for index in range(5_000):
        value = 20.0 + np.sin(index / 50) + rng.normal(0.0, 0.1)
        spike = index > 300 and rng.random() < 0.01
        events.append(({"temperature": value + 3.0 * spike}, int(spike)))
    return events


def _report(name: str, detector: ModelProtocol, events: list[Sample]) -> None:
    evaluator = PrequentialEvaluator(
        detector, warmup=300, metrics=True, metric_window_size=None
    )
    result = evaluator.evaluate(events)
    print(
        f"{name:<26} average precision {result.average_precision:.3f}  "
        f"ROC AUC {result.roc_auc:.3f}"
    )


def main() -> None:
    shuttle = list(islice(load(Dataset.SHUTTLE), 20_000))
    _report("OnlineECOD", OnlineECOD(), shuttle)
    _report("OnlineCOPOD", OnlineCOPOD(), shuttle)
    _report("MovingMahalanobisDistance", MovingMahalanobisDistance(), shuttle)

    sensor = _sensor()
    _report("MovingZScore", MovingZScore(window_size=200), sensor)
    _report("MovingMAD", MovingMAD(window_size=200), sensor)


if __name__ == "__main__":
    main()
