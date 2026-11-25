"""Detection quality of every tabular detector on SHUTTLE.

Each case learns 1,000 normal rows and then scores 5,000 rows in stream
order, 387 of them anomalous. Each floor sits 0.02 (ROC-AUC) and 0.05
(PR-AUC) below the lowest value that ``main`` reached over the seeds 42 and 0
to 3, so a harmless change that moves a detector's random draws still passes
and a real regression does not. There are no upper bounds: a better detector
keeps passing. This lane runs seed 42; the nightly benchmark runs the others.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from aberrant.model.deep.kitnet import OnlineAutoencoderEnsemble
from aberrant.model.distance import CellNeighborhoodDetector, SDOStream
from aberrant.model.distance.knn import KNN
from aberrant.model.forest import (
    RHF,
    ASDIsolationForest,
    HalfSpaceTrees,
    MondrianIsolationForest,
    OnlineIsolationForest,
    RandomCutForest,
    XStream,
)
from aberrant.model.null import NullModel
from aberrant.model.random import RandomModel
from aberrant.model.sketch import LODA, RSHash
from aberrant.model.stat import (
    MovingAverage,
    MovingCovariance,
    MovingMahalanobisDistance,
    MovingVariance,
    OnlineCOPOD,
    OnlineECOD,
)
from aberrant.model.svm import AdaptiveOneClassSVM
from aberrant.stream import Dataset, get_dataset_info
from aberrant.transform import MinMaxScaler, StandardScaler
from tests.integration._quality import (
    Case,
    assert_detects,
    first_features,
    prequential,
    with_arrival_time,
)

WARM_UP = 1_000
N_TEST = 5_000
SEED = 42


def _knn(seed: int, backend: str = "numpy") -> object:
    return StandardScaler() | KNN(k=55, window_size=250, backend=backend)


def _autoencoder(seed: int) -> object:
    from torch import nn, optim  # noqa: PLC0415

    from aberrant.model.deep import VanillaAutoencoder  # noqa: PLC0415
    from aberrant.model.deep.autoencoder import Autoencoder  # noqa: PLC0415

    architecture = VanillaAutoencoder(
        input_size=get_dataset_info(Dataset.SHUTTLE).n_features, seed=seed
    )
    return Autoencoder(
        model=architecture,
        optimizer=optim.Adam(architecture.parameters(), lr=0.001),
        criterion=nn.MSELoss(),
    )


CASES = [
    Case(
        "asd-isolation-forest",
        lambda s: ASDIsolationForest(n_trees=50, max_samples=256, seed=s),
        0.97,
        0.87,
    ),
    Case(
        "half-space-trees",
        lambda s: (
            MinMaxScaler()
            | HalfSpaceTrees(n_trees=25, height=15, window_size=250, seed=s)
        ),
        0.94,
        0.71,
    ),
    Case(
        "mondrian-isolation-forest",
        lambda s: MondrianIsolationForest(
            n_trees=50, subspace_size=128, lifetime=1.0, seed=s
        ),
        0.93,
        0.55,
    ),
    Case(
        "online-isolation-forest",
        lambda s: OnlineIsolationForest(
            n_trees=100,
            max_leaf_samples=32,
            tree_type="adaptive",
            subsample=1.0,
            window_size=1024,
            branching_factor=2,
            metric="axisparallel",
            seed=s,
        ),
        0.95,
        0.6,
    ),
    Case(
        "random-histogram-forest",
        lambda s: RHF(n_trees=25, max_depth=10, window_size=256, seed=s),
        0.96,
        0.77,
    ),
    Case(
        "random-cut-forest",
        lambda s: RandomCutForest(
            n_trees=12,
            sample_size=256,
            shingle_size=1,
            warm_up=256,
            seed=s,
        ),
        0.86,
        0.26,
    ),
    Case(
        "xstream",
        lambda s: XStream(
            k=32,
            n_chains=32,
            depth=10,
            sketch_width=256,
            n_hashes=3,
            window_size=128,
            warm_up=128,
            density=0.25,
            seed=s,
        ),
        0.95,
        0.6,
    ),
    Case(
        "cell-neighborhood",
        lambda s: (
            StandardScaler()
            | CellNeighborhoodDetector(
                k=30,
                radius=1.5,
                window_size=2048,
                slide_size=128,
                warm_up=2,
            )
        ),
        0.68,
        0.25,
    ),
    Case(
        "sdostream",
        lambda s: SDOStream(
            max_observers=128,
            time_scale=256.0,
            inactive_quantile=0.3,
            n_neighbors=8,
            distance="euclidean",
            warm_up=12,
            seed=s,
        ),
        0.97,
        0.75,
    ),
    Case("knn", _knn, 0.97, 0.82),
    Case(
        "knn-faiss",
        lambda s: _knn(s, backend="faiss"),
        0.97,
        0.82,
        requires=("faiss",),
    ),
    Case(
        "loda",
        lambda s: (
            StandardScaler()
            | LODA(
                n_projections=64,
                n_bins=24,
                sparsity=0.3,
                warm_up=256,
                decay=1.0,
                time_key=None,
                pseudocount=0.5,
                seed=s,
            )
        ),
        0.92,
        0.52,
    ),
    Case(
        "rshash",
        lambda s: RSHash(
            n_members=24,
            n_hashes=4,
            sketch_width=512,
            subspace_size=3,
            bin_width=1.0,
            decay=0.01,
            warm_up=128,
            time_key="t",
            seed=s,
        ),
        0.95,
        0.7,
        adapt=with_arrival_time,
    ),
    Case(
        "online-autoencoder-ensemble",
        lambda s: (
            StandardScaler()
            | OnlineAutoencoderEnsemble(
                max_ae_size=4,
                feature_map_grace=256,
                ad_grace=512,
                learning_rate=0.03,
                hidden_ratio=0.75,
                adaptive_after_warmup=False,
                seed=s,
            )
        ),
        0.95,
        0.79,
    ),
    Case("autoencoder-torch", _autoencoder, 0.84, 0.28, requires=("torch",)),
    Case(
        "adaptive-kernel-svm",
        lambda s: AdaptiveOneClassSVM(nu=0.1, initial_gamma=1.0, seed=s),
        0.77,
        0.29,
    ),
    Case("ecod-sketch", lambda s: OnlineECOD(), 0.95, 0.72),
    Case("ecod-window", lambda s: OnlineECOD(memory="window"), 0.96, 0.82),
    Case(
        "ecod-hybrid-pareto",
        lambda s: OnlineECOD(memory="hybrid", tail="pareto"),
        0.96,
        0.72,
    ),
    Case("ecod-decay", lambda s: OnlineECOD(half_life=2_000.0), 0.96, 0.78),
    Case("copod-pyod", lambda s: OnlineCOPOD(aggregation="pyod"), 0.97, 0.87),
    Case(
        "moving-covariance",
        lambda s: MovingCovariance(window_size=100),
        0.74,
        0.17,
        adapt=first_features(2),
    ),
    Case(
        "moving-mahalanobis",
        lambda s: MovingMahalanobisDistance(window_size=100),
        0.94,
        0.57,
    ),
    Case(
        "moving-average",
        lambda s: MovingAverage(window_size=100),
        0.96,
        0.91,
        adapt=first_features(1),
    ),
    Case(
        "moving-variance",
        lambda s: MovingVariance(window_size=100),
        0.95,
        0.9,
        adapt=first_features(1),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_detector_keeps_its_quality_on_shuttle(case: Case) -> None:
    for module in case.requires:
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"{case.name} needs {module}")
    result = prequential(
        case, Dataset.SHUTTLE, warm_up=WARM_UP, n_test=N_TEST, seed=SEED
    )
    assert_detects(
        result.labels,
        result.scores,
        roc_auc_floor=case.roc_auc_floor,
        pr_auc_floor=case.pr_auc_floor,
    )


def test_harness_gives_a_constant_score_no_credit() -> None:
    null = Case("null", lambda s: NullModel(), 0.51, 0.0)
    result = prequential(
        null, Dataset.SHUTTLE, warm_up=WARM_UP, n_test=N_TEST, seed=SEED
    )
    assert result.roc_auc == 0.5
    assert result.pr_auc == pytest.approx(result.labels.mean())
    with pytest.raises(AssertionError, match="every score is the same"):
        assert_detects(
            result.labels, result.scores, roc_auc_floor=0.51, pr_auc_floor=0.0
        )


def test_harness_gives_a_random_score_chance_level() -> None:
    random = Case("random", lambda s: RandomModel(seed=s), 0.51, 0.0)
    result = prequential(
        random, Dataset.SHUTTLE, warm_up=WARM_UP, n_test=N_TEST, seed=SEED
    )
    assert abs(result.roc_auc - 0.5) < 0.05
    assert abs(result.pr_auc - float(np.mean(result.labels))) < 0.03
    with pytest.raises(AssertionError, match="ROC-AUC .* fell below"):
        assert_detects(
            result.labels, result.scores, roc_auc_floor=0.6, pr_auc_floor=0.0
        )
