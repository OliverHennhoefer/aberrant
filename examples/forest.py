"""Isolation and half-space forests on the Shuttle benchmark.

Every detector scores each of the first 20,000 events before it learns it, and
learns every event, normal or not. Labels only feed the metrics. Run with
``uv run --extra eval python examples/forest.py``.

``RandomCutForest``, ``RHF`` and ``MondrianIsolationForest`` follow the same
pattern; they are left out here because they take a minute or more on this
stream.
"""

from itertools import islice

from aberrant.base import ModelProtocol
from aberrant.datasets import Dataset, load
from aberrant.evaluate import PrequentialEvaluator
from aberrant.model.forest import (
    ASDIsolationForest,
    HalfSpaceTrees,
    OnlineIsolationForest,
    XStream,
)
from aberrant.transform import MinMaxScaler


def main() -> None:
    events = list(islice(load(Dataset.SHUTTLE), 20_000))
    detectors: dict[str, ModelProtocol] = {
        # Half-space trees split the unit cube, so they need scaled features.
        "HalfSpaceTrees": MinMaxScaler() | HalfSpaceTrees(seed=1),
        "OnlineIsolationForest": OnlineIsolationForest(seed=1),
        "ASDIsolationForest": ASDIsolationForest(seed=1),
        "XStream": XStream(seed=1),
    }
    for name, detector in detectors.items():
        evaluator = PrequentialEvaluator(
            detector, warm_up=500, metrics=True, metric_window_size=None
        )
        result = evaluator.evaluate(events)
        print(
            f"{name:<22} average precision {result.average_precision:.3f}  "
            f"ROC AUC {result.roc_auc:.3f}"
        )


if __name__ == "__main__":
    main()
