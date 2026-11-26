"""Sketch- and projection-based detectors on the Shuttle benchmark.

Each detector scores an event before it learns it, and learns every event;
labels only feed the metrics. Run with
``uv run --extra eval python examples/sketch.py``.
"""

from itertools import islice

from aberrant.datasets import Dataset, load
from aberrant.evaluate import PrequentialEvaluator
from aberrant.model.sketch import LODA, MStream, RSHash


def main() -> None:
    events = list(islice(load(Dataset.SHUTTLE), 20_000))
    detectors = {
        "LODA": LODA(seed=1),
        "RSHash": RSHash(seed=1),
        "MStream": MStream(seed=1),
    }
    for name, detector in detectors.items():
        evaluator = PrequentialEvaluator(
            detector, warm_up=500, metrics=True, metric_window_size=None
        )
        result = evaluator.evaluate(events)
        print(
            f"{name:<8} average precision {result.average_precision:.3f}  "
            f"ROC AUC {result.roc_auc:.3f}"
        )


if __name__ == "__main__":
    main()
