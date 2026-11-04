"""Seeded score sequences recorded from reference implementations.

Each JSON file under ``tests/data/score_sequences`` holds, per case, the model
parameters, the seeded stream that was fed to it and the scores it produced
when scoring and then learning every event, or every batch for cases that name
a ``batch_size``. The files were recorded with
aberrant 1.4.0, before the detectors were optimised for speed, so tests can
check that faster implementations still produce the same score sequences.

To re-record a file with the importable aberrant, run for example::

    python tests/score_sequences.py streaming_loda
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

import aberrant
from aberrant.model.distance import LocalOutlierFactor
from aberrant.model.iforest import (
    ASDIsolationForest,
    HalfSpaceTrees,
    MondrianIsolationForest,
    OnlineIsolationForest,
    RandomCutForest,
    StreamRandomHistogramForest,
    XStream,
)
from aberrant.model.sketch import StreamingLODA
from aberrant.model.svm import IncrementalOneClassSVMAdaptiveKernel

FIXTURE_DIR = Path(__file__).parent / "data" / "score_sequences"


def make_stream(
    seed: int,
    n_events: int,
    n_features: int,
    *,
    kind: str = "gaussian",
    time_key: str | None = None,
    outlier_scale: float = 8.0,
) -> list[dict[str, float]]:
    """Build a seeded stream of plain-float feature dictionaries.

    ``gaussian`` draws standard normal features and scales every 17th event by
    ``outlier_scale`` to create outliers. ``unit`` draws two clusters inside ``[0, 1]`` and
    replaces every 23rd event with a uniform draw. ``grid`` rounds standard
    normal features to multiples of one half, so that events repeat.
    """
    rng = np.random.default_rng(seed)
    if kind == "gaussian":
        data = rng.normal(size=(n_events, n_features))
        data[::17] *= outlier_scale
    elif kind == "unit":
        centers = rng.uniform(0.2, 0.8, size=(2, n_features))
        labels = rng.integers(0, 2, size=n_events)
        noise = rng.normal(0.0, 0.05, size=(n_events, n_features))
        data = np.clip(centers[labels] + noise, 0.0, 1.0)
        data[::23] = rng.uniform(0.0, 1.0, size=data[::23].shape)
    elif kind == "grid":
        data = np.round(rng.normal(size=(n_events, n_features)) * 2.0) / 2.0
    else:
        raise ValueError(f"Unknown stream kind: {kind}")

    names = [f"x{index}" for index in range(n_features)]
    events = [dict(zip(names, row, strict=True)) for row in data.tolist()]
    if time_key is not None:
        for index, event in enumerate(events):
            event[time_key] = float(index)
    return events


def score_then_learn(model: Any, events: list[dict[str, float]]) -> list[float]:
    """Score each event before learning it and return the scores."""
    scores = []
    for event in events:
        scores.append(float(model.score_one(event)))
        model.learn_one(event)
    return scores


def score_then_learn_batches(
    model: Any, events: list[dict[str, float]], batch_size: int
) -> list[float]:
    """Score each batch of event rows with ``score_batch`` before learning it."""
    data = np.array([list(event.values()) for event in events], dtype=np.float32)
    scores: list[float] = []
    for start in range(0, len(data), batch_size):
        batch = data[start : start + batch_size]
        scores.extend(float(score) for score in model.score_batch(batch))
        model.learn_batch(batch)
    return scores


def replay(model: Any, case: dict[str, Any]) -> list[float]:
    """Feed a case's stream to ``model`` and return the scores."""
    events = make_stream(**case["stream"])
    if "batch_size" in case:
        return score_then_learn_batches(model, events, case["batch_size"])
    return score_then_learn(model, events)


def load(name: str) -> list[dict[str, Any]]:
    """Return the recorded cases of one fixture file."""
    with (FIXTURE_DIR / f"{name}.json").open(encoding="utf-8") as handle:
        return list(json.load(handle)["cases"])


# Fixture name -> (model factory, cases). Cases name the model parameters and
# the ``make_stream`` arguments; recording adds the scores.
CASES: dict[str, tuple[Callable[..., Any], list[dict[str, Any]]]] = {
    "asd_isolation_forest": (
        ASDIsolationForest,
        [
            {
                "name": "defaults",
                "params": {"seed": 111},
                "stream": {"seed": 112, "n_events": 700, "n_features": 4},
            },
            {
                "name": "small_window_fast_retrain",
                "params": {
                    "n_estimators": 20,
                    "max_samples": 32,
                    "window_size": 48,
                    "retrain_interval": 16,
                    "seed": 113,
                },
                "stream": {"seed": 114, "n_events": 400, "n_features": 3},
            },
            {
                "name": "unit_clusters",
                "params": {"n_estimators": 30, "max_samples": 64, "seed": 115},
                "stream": {
                    "seed": 116,
                    "n_events": 400,
                    "n_features": 5,
                    "kind": "unit",
                },
            },
            {
                # Repeated values leave nodes whose features are all constant.
                "name": "repeated_points",
                "params": {
                    "n_estimators": 25,
                    "max_samples": 40,
                    "window_size": 60,
                    "retrain_interval": 20,
                    "seed": 117,
                },
                "stream": {
                    "seed": 118,
                    "n_events": 300,
                    "n_features": 2,
                    "kind": "grid",
                },
            },
            {
                "name": "two_point_samples",
                "params": {
                    "n_estimators": 10,
                    "max_samples": 2,
                    "window_size": 5,
                    "retrain_interval": 3,
                    "seed": 119,
                },
                "stream": {"seed": 120, "n_events": 150, "n_features": 1},
            },
            {
                "name": "many_features",
                "params": {
                    "n_estimators": 15,
                    "max_samples": 50,
                    "window_size": 80,
                    "retrain_interval": 30,
                    "seed": 121,
                },
                "stream": {"seed": 122, "n_events": 300, "n_features": 12},
            },
        ],
    ),
    "half_space_trees": (
        HalfSpaceTrees,
        [
            {
                "name": "defaults_short_window",
                "params": {"window_size": 50, "seed": 7},
                "stream": {
                    "seed": 21,
                    "n_events": 600,
                    "n_features": 8,
                    "kind": "unit",
                },
            },
            {
                "name": "small_trees",
                "params": {"n_trees": 3, "height": 3, "window_size": 7, "seed": 8},
                "stream": {
                    "seed": 22,
                    "n_events": 300,
                    "n_features": 2,
                    "kind": "unit",
                },
            },
        ],
    ),
    # The streams are continuous, so no two distances tie and the neighbor
    # order does not depend on the platform's sorting algorithm.
    "incremental_one_class_svm_adaptive_kernel": (
        IncrementalOneClassSVMAdaptiveKernel,
        [
            {
                "name": "defaults",
                "params": {"seed": 91},
                "stream": {"seed": 92, "n_events": 400, "n_features": 4},
            },
            {
                "name": "small_budget_fast_adaptation",
                "params": {
                    "sv_budget": 12,
                    "buffer_size": 30,
                    "adaptation_rate": 0.5,
                    "seed": 93,
                },
                "stream": {"seed": 94, "n_events": 400, "n_features": 3},
            },
            {
                "name": "unit_clusters",
                "params": {
                    "nu": 0.5,
                    "initial_gamma": 0.1,
                    "gamma_bounds": [0.01, 10.0],
                    "seed": 95,
                },
                "stream": {
                    "seed": 96,
                    "n_events": 300,
                    "n_features": 5,
                    "kind": "unit",
                },
            },
            {
                "name": "repeated_points",
                "params": {"sv_budget": 40, "seed": 97},
                "stream": {
                    "seed": 98,
                    "n_events": 300,
                    "n_features": 2,
                    "kind": "grid",
                },
            },
            {
                "name": "loose_tolerance",
                "params": {"nu": 0.8, "tolerance": 0.05, "seed": 99},
                "stream": {"seed": 100, "n_events": 300, "n_features": 6},
            },
            {
                # Only the first event ever becomes a support vector.
                "name": "single_support_vector",
                "params": {"nu": 1.0, "tolerance": 0.1, "seed": 101},
                "stream": {"seed": 102, "n_events": 200, "n_features": 6},
            },
            {
                "name": "full_budget",
                "params": {"nu": 1.0, "tolerance": 0.01, "seed": 103},
                "stream": {"seed": 104, "n_events": 500, "n_features": 6},
            },
        ],
    ),
    "local_outlier_factor": (
        LocalOutlierFactor,
        [
            {
                "name": "euclidean_k10",
                "params": {"k": 10, "window_size": 150},
                "stream": {"seed": 41, "n_events": 400, "n_features": 4},
            },
            {
                "name": "manhattan_k3",
                "params": {"k": 3, "window_size": 40, "distance": "manhattan"},
                "stream": {"seed": 42, "n_events": 300, "n_features": 3},
            },
            {
                "name": "window_one_above_k",
                "params": {"k": 2, "window_size": 3},
                "stream": {"seed": 43, "n_events": 100, "n_features": 2},
            },
        ],
    ),
    # The windows stay below ``max_leaf_samples * (2 * branching_factor)**2``, so
    # the tree depth limit, a ratio of logarithms, never sits on an integer
    # where the last bit of a platform's ``log`` could change a split decision.
    "online_isolation_forest": (
        OnlineIsolationForest,
        [
            {
                "name": "adaptive_binary",
                "params": {
                    "num_trees": 10,
                    "max_leaf_samples": 8,
                    "window_size": 100,
                    "seed": 3,
                },
                "stream": {"seed": 31, "n_events": 400, "n_features": 4},
            },
            {
                "name": "fixed_ternary_subsampled",
                "params": {
                    "num_trees": 8,
                    "max_leaf_samples": 4,
                    "tree_type": "fixed",
                    "subsample": 0.7,
                    "window_size": 60,
                    "branching_factor": 3,
                    "seed": 4,
                },
                "stream": {"seed": 32, "n_events": 300, "n_features": 3},
            },
            {
                "name": "adaptive_batches",
                "params": {
                    "num_trees": 6,
                    "max_leaf_samples": 4,
                    "window_size": 40,
                    "seed": 5,
                },
                "stream": {"seed": 33, "n_events": 300, "n_features": 5},
                "batch_size": 7,
            },
        ],
    ),
    "mondrian_isolation_forest": (
        MondrianIsolationForest,
        [
            {
                "name": "small_forest",
                "params": {
                    "n_estimators": 10,
                    "subspace_size": 3,
                    "seed": 81,
                },
                "stream": {"seed": 82, "n_events": 300, "n_features": 4},
            },
            {
                # Rebuilds every tree from the latest 32 points every 32
                # events after the first 64.
                "name": "windowed",
                "params": {
                    "n_estimators": 8,
                    "subspace_size": 2,
                    "lambda_": 2.0,
                    "window_size": 32,
                    "seed": 83,
                },
                "stream": {"seed": 84, "n_events": 300, "n_features": 3},
            },
            {
                # A long lifetime keeps splitting, so the trees grow deep.
                "name": "long_lifetime",
                "params": {"n_estimators": 6, "lambda_": 50.0, "seed": 85},
                "stream": {"seed": 86, "n_events": 250, "n_features": 2},
            },
            {
                "name": "unit_clusters",
                "params": {"n_estimators": 12, "seed": 87},
                "stream": {
                    "seed": 88,
                    "n_events": 300,
                    "n_features": 5,
                    "kind": "unit",
                },
            },
            {
                # Repeated points fall inside existing blocks and only add
                # to their counts.
                "name": "repeated_points",
                "params": {"n_estimators": 8, "lambda_": 4.0, "seed": 89},
                "stream": {
                    "seed": 90,
                    "n_events": 300,
                    "n_features": 2,
                    "kind": "grid",
                },
            },
        ],
    ),
    "random_cut_forest": (
        RandomCutForest,
        [
            {
                "name": "small_forest",
                "params": {
                    "n_trees": 10,
                    "sample_size": 64,
                    "warmup_samples": 32,
                    "seed": 71,
                },
                "stream": {"seed": 72, "n_events": 300, "n_features": 4},
            },
            {
                "name": "shingled_normalized",
                "params": {
                    "n_trees": 6,
                    "sample_size": 32,
                    "shingle_size": 3,
                    "warmup_samples": 8,
                    "normalize_score": True,
                    "score_scale": 4.0,
                    "seed": 73,
                },
                "stream": {"seed": 74, "n_events": 250, "n_features": 2},
            },
            {
                "name": "unit_clusters",
                "params": {
                    "n_trees": 8,
                    "sample_size": 48,
                    "warmup_samples": 16,
                    "seed": 75,
                },
                "stream": {
                    "seed": 76,
                    "n_events": 300,
                    "n_features": 5,
                    "kind": "unit",
                },
            },
            {
                # Repeated points share leaves, and removing one of them
                # leaves its leaf in place.
                "name": "repeated_points",
                "params": {
                    "n_trees": 8,
                    "sample_size": 40,
                    "warmup_samples": 10,
                    "seed": 77,
                },
                "stream": {
                    "seed": 78,
                    "n_events": 300,
                    "n_features": 2,
                    "kind": "grid",
                },
            },
            {
                # Two stored points, so every removal replaces the root.
                "name": "two_point_sample",
                "params": {
                    "n_trees": 5,
                    "sample_size": 2,
                    "warmup_samples": 1,
                    "seed": 79,
                },
                "stream": {"seed": 80, "n_events": 150, "n_features": 1},
            },
        ],
    ),
    "stream_random_histogram_forest": (
        StreamRandomHistogramForest,
        [
            {
                "name": "small_forest",
                "params": {
                    "n_estimators": 6,
                    "max_depth": 5,
                    "window_size": 32,
                    "seed": 61,
                },
                "stream": {"seed": 62, "n_events": 250, "n_features": 4},
            },
            {
                "name": "deep_single_feature",
                "params": {
                    "n_estimators": 3,
                    "max_depth": 8,
                    "window_size": 20,
                    "seed": 63,
                },
                "stream": {"seed": 64, "n_events": 150, "n_features": 1},
            },
            {
                "name": "unit_clusters",
                "params": {
                    "n_estimators": 4,
                    "max_depth": 4,
                    "window_size": 50,
                    "seed": 65,
                },
                "stream": {
                    "seed": 66,
                    "n_events": 300,
                    "n_features": 6,
                    "kind": "unit",
                },
            },
            {
                # Outliers near 1e80 overflow the fourth moments, so some
                # kurtosis weights are NaN.
                "name": "overflowing_moments",
                "params": {
                    "n_estimators": 4,
                    "max_depth": 5,
                    "window_size": 24,
                    "seed": 67,
                },
                "stream": {
                    "seed": 68,
                    "n_events": 200,
                    "n_features": 3,
                    "outlier_scale": 1e80,
                },
            },
        ],
    ),
    "streaming_loda": (
        StreamingLODA,
        [
            {
                "name": "defaults_short_warm_up",
                "params": {"warm_up_samples": 64, "seed": 3},
                "stream": {"seed": 11, "n_events": 400, "n_features": 8},
            },
            {
                "name": "decay_time_key_ten_bins",
                "params": {
                    "n_projections": 40,
                    "n_bins": 10,
                    "sparsity": 0.4,
                    "warm_up_samples": 32,
                    "decay": 0.99,
                    "time_key": "t",
                    "pseudocount": 0.5,
                    "seed": 5,
                },
                "stream": {
                    "seed": 12,
                    "n_events": 300,
                    "n_features": 5,
                    "time_key": "t",
                },
            },
        ],
    ),
    "xstream": (
        XStream,
        [
            {
                "name": "small_chains",
                "params": {
                    "k": 20,
                    "n_chains": 10,
                    "depth": 6,
                    "cms_width": 64,
                    "cms_num_hashes": 3,
                    "window_size": 32,
                    "init_sample_size": 32,
                    "seed": 6,
                },
                "stream": {"seed": 51, "n_events": 300, "n_features": 5},
            },
            {
                # Four projected dimensions over twelve levels, so chains
                # revisit dimensions and halve their intervals again.
                "name": "revisited_dimensions",
                "params": {
                    "k": 4,
                    "n_chains": 8,
                    "depth": 12,
                    "cms_width": 128,
                    "cms_num_hashes": 2,
                    "window_size": 25,
                    "init_sample_size": 20,
                    "density": 1.0,
                    "seed": 7,
                },
                "stream": {"seed": 52, "n_events": 300, "n_features": 3},
            },
            {
                "name": "defaults_short_windows",
                "params": {"window_size": 64, "init_sample_size": 64, "seed": 8},
                "stream": {"seed": 53, "n_events": 300, "n_features": 8},
            },
            {
                # One initial point leaves every chain scale at 1, so the
                # 1e30 outliers land in bins far beyond the int64 range.
                "name": "huge_bins",
                "params": {
                    "k": 8,
                    "n_chains": 6,
                    "depth": 10,
                    "cms_width": 32,
                    "cms_num_hashes": 2,
                    "window_size": 16,
                    "init_sample_size": 1,
                    "density": 0.5,
                    "seed": 9,
                },
                "stream": {
                    "seed": 54,
                    "n_events": 200,
                    "n_features": 4,
                    "outlier_scale": 1e30,
                },
            },
        ],
    ),
}


def record(name: str) -> None:
    """Re-record one fixture file with the importable aberrant."""
    factory, cases = CASES[name]
    recorded = []
    for case in cases:
        scores = replay(factory(**case["params"]), case)
        recorded.append({**case, "scores": scores})
    payload = {"recorded_with": f"aberrant {aberrant.__version__}", "cases": recorded}
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    with (FIXTURE_DIR / f"{name}.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
        handle.write("\n")


if __name__ == "__main__":
    for fixture in sys.argv[1:]:
        record(fixture)
