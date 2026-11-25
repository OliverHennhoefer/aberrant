"""Distance- and density-based detectors on the Shuttle benchmark.

Distances need comparable feature scales, so every detector follows a
``StandardScaler``. Each scores an event before it learns it, and learns every
event; labels only feed the metrics. Run with
``uv run --extra eval python examples/distance.py``.
"""

from itertools import islice

from aberrant.datasets import Dataset, load
from aberrant.evaluate import PrequentialEvaluator
from aberrant.model.distance import KNN, LocalOutlierFactor, SDOStream
from aberrant.transform import StandardScaler


def main() -> None:
    events = list(islice(load(Dataset.SHUTTLE), 20_000))
    detectors = {
        "KNN": StandardScaler() | KNN(k=10, window_size=500),
        "LocalOutlierFactor": StandardScaler() | LocalOutlierFactor(),
        "SDOStream": StandardScaler() | SDOStream(seed=1),
    }
    for name, detector in detectors.items():
        evaluator = PrequentialEvaluator(
            detector, warmup=500, metrics=True, metric_window_size=None
        )
        result = evaluator.evaluate(events)
        print(
            f"{name:<19} average precision {result.average_precision:.3f}  "
            f"ROC AUC {result.roc_auc:.3f}"
        )


if __name__ == "__main__":
    main()
