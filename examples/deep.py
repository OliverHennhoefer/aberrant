"""Reconstruction-based detectors on the Shuttle benchmark.

``OnlineAutoencoderEnsemble`` (KitNET) needs only NumPy. ``Autoencoder`` wraps
a PyTorch network you supply and needs the ``dl`` extra. Both score an event
before they learn it, and learn every event; labels only feed the metrics. Run
with ``uv run --extra eval --extra dl python examples/deep.py``.
"""

from itertools import islice

from torch import nn, optim

from aberrant.datasets import Dataset, load
from aberrant.evaluate import PrequentialEvaluator
from aberrant.model.deep import (
    Autoencoder,
    OnlineAutoencoderEnsemble,
    VanillaAutoencoder,
)
from aberrant.transform import MinMaxScaler


def main() -> None:
    events = list(islice(load(Dataset.SHUTTLE), 20_000))
    network = VanillaAutoencoder(input_size=9, seed=1)
    detectors = {
        # KitNET learns its feature map, then its autoencoders, before scoring.
        "OnlineAutoencoderEnsemble": MinMaxScaler()
        | OnlineAutoencoderEnsemble(feature_map_grace=500, ad_grace=2_000, seed=1),
        "Autoencoder": MinMaxScaler()
        | Autoencoder(
            model=network,
            optimizer=optim.Adam(network.parameters(), lr=0.005),
            criterion=nn.MSELoss(),
        ),
    }
    for name, detector in detectors.items():
        evaluator = PrequentialEvaluator(
            detector, warm_up=500, metrics=True, metric_window_size=None
        )
        result = evaluator.evaluate(events)
        print(
            f"{name:<26} average precision {result.average_precision:.3f}  "
            f"ROC AUC {result.roc_auc:.3f}"
        )


if __name__ == "__main__":
    main()
