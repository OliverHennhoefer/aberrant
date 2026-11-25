"""Wall-clock smoke tests: each default model scores and learns a short stream.

The bounds sit about ten times above typical runs, so they catch an accidental
quadratic loop or a lost vectorisation, not small slowdowns. Detection quality
and exact scores are covered by each model's own tests.
"""

import time
from collections.abc import Callable
from typing import Any

import pytest

from aberrant.model.distance import LocalOutlierFactor
from aberrant.model.forest import (
    RHF,
    ASDIsolationForest,
    HalfSpaceTrees,
    OnlineIsolationForest,
    RandomCutForest,
    XStream,
)
from aberrant.model.forest.mondrian import MondrianIsolationForest
from aberrant.model.graph import ISCONNA
from aberrant.model.sketch import LODA, MStream, RSHash
from aberrant.model.stat import MovingCovariance
from aberrant.model.svm import AdaptiveOneClassSVM
from aberrant.model.timeseries import (
    DAMP,
    MultivariateRollingMatrixProfile,
    RollingMatrixProfile,
)
from tests import score_sequences


def _stream(
    *, seed: int, n_events: int = 2000, n_features: int = 8, kind: str = "gaussian"
) -> Callable[[], list]:
    return lambda: score_sequences.make_stream(
        seed=seed, n_events=n_events, n_features=n_features, kind=kind
    )


def _series(*, seed: int, n_events: int, n_channels: int) -> Callable[[], list]:
    return lambda: score_sequences.make_series(
        seed=seed, n_events=n_events, n_channels=n_channels
    )


# (model factory, event factory, bound in seconds); typical runs in comments.
CASES = {
    "lof": (LocalOutlierFactor, _stream(seed=1), 5.0),  # ~0.4 s
    "moving-covariance": (
        lambda: MovingCovariance(window_size=1000),
        _stream(seed=13, n_features=2),
        5.0,
    ),  # ~0.25 s
    "half-space-trees": (
        lambda: HalfSpaceTrees(seed=1),
        _stream(seed=1, kind="unit"),
        1.0,
    ),  # ~0.05 s
    "online-isolation-forest": (
        lambda: OnlineIsolationForest(n_trees=25, window_size=256, seed=1),
        _stream(seed=1),
        3.0,
    ),  # ~0.3 s
    "asd-isolation-forest": (
        lambda: ASDIsolationForest(seed=7),
        _stream(seed=7),
        4.0,
    ),  # ~0.4 s
    "xstream": (lambda: XStream(seed=1), _stream(seed=1), 5.0),  # ~0.5 s
    "random-histogram-forest": (
        lambda: RHF(seed=1),
        _stream(seed=1, n_events=1000),
        10.0,
    ),  # ~1 s
    "random-cut-forest": (
        lambda: RandomCutForest(seed=1),
        _stream(seed=1, n_events=600),
        20.0,
    ),  # ~2 s
    "mondrian-isolation-forest": (
        lambda: MondrianIsolationForest(seed=1),
        _stream(seed=1, n_events=600),
        20.0,
    ),  # ~2 s
    "loda": (lambda: LODA(seed=1), _stream(seed=1), 1.0),  # ~0.05 s
    "rshash": (lambda: RSHash(seed=8), _stream(seed=8), 2.0),  # ~0.15 s
    "mstream": (
        lambda: MStream(seed=17),
        lambda: score_sequences.make_multi_aspect_stream(
            seed=17, n_events=2000, n_numeric=8, n_categorical=0
        ),
        4.0,
    ),  # ~0.3 s
    "isconna": (
        lambda: ISCONNA(seed=35),
        lambda: score_sequences.make_edge_stream(seed=35, n_events=2000, n_nodes=50),
        2.0,
    ),  # ~0.13 s
    "adaptive-svm": (
        lambda: AdaptiveOneClassSVM(seed=10),
        _stream(seed=10),
        4.0,
    ),  # ~0.4 s
    "x-lag-damp": (
        lambda: DAMP(subsequence_length=32),
        _series(seed=39, n_events=2000, n_channels=1),
        4.0,
    ),  # ~0.3 s
    "rolling-matrix-profile": (
        lambda: RollingMatrixProfile(subsequence_length=32),
        _series(seed=37, n_events=1500, n_channels=1),
        6.0,
    ),  # ~0.4 s
    "multivariate-rolling-matrix-profile": (
        lambda: MultivariateRollingMatrixProfile(subsequence_length=32),
        _series(seed=38, n_events=1000, n_channels=2),
        8.0,
    ),  # ~0.65 s
}


@pytest.mark.parametrize(
    ("make_model", "make_events", "bound"), CASES.values(), ids=CASES
)
def test_default_model_scores_and_learns_a_short_stream_quickly(
    make_model: Callable[[], Any], make_events: Callable[[], list], bound: float
) -> None:
    model = make_model()
    events = make_events()

    start = time.perf_counter()
    scores = score_sequences.score_then_learn(model, events)
    elapsed = time.perf_counter() - start

    assert len(scores) == len(events)
    assert elapsed < bound, f"took {elapsed:.2f} s, bound {bound} s"
