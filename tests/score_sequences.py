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
from aberrant.model.iforest import HalfSpaceTrees, OnlineIsolationForest
from aberrant.model.sketch import StreamingLODA

FIXTURE_DIR = Path(__file__).parent / "data" / "score_sequences"


def make_stream(
    seed: int,
    n_events: int,
    n_features: int,
    *,
    kind: str = "gaussian",
    time_key: str | None = None,
) -> list[dict[str, float]]:
    """Build a seeded stream of plain-float feature dictionaries.

    ``gaussian`` draws standard normal features and scales every 17th event by
    eight to create outliers. ``unit`` draws two clusters inside ``[0, 1]`` and
    replaces every 23rd event with a uniform draw.
    """
    rng = np.random.default_rng(seed)
    if kind == "gaussian":
        data = rng.normal(size=(n_events, n_features))
        data[::17] *= 8.0
    elif kind == "unit":
        centers = rng.uniform(0.2, 0.8, size=(2, n_features))
        labels = rng.integers(0, 2, size=n_events)
        noise = rng.normal(0.0, 0.05, size=(n_events, n_features))
        data = np.clip(centers[labels] + noise, 0.0, 1.0)
        data[::23] = rng.uniform(0.0, 1.0, size=data[::23].shape)
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
