"""Compare the throughput of aberrant's hot detectors with river and PySAD.

This script is not part of CI. Run it from a scratch environment that has the
peer libraries installed next to aberrant, for example::

    python -m venv /tmp/peers
    /tmp/peers/bin/pip install -e . river pysad
    /tmp/peers/bin/python benchmarks/compare_peers.py

Every case processes the same stream one event at a time: it scores the event
and then learns it, with warm-up events included in the timing. Each library
receives its native input (``dict[str, float]`` for aberrant and river, a
float64 row for PySAD), prepared before the clock starts. PySAD runs through
``fit_score_partial``, its own score-then-learn call, which projects each event
once. Throughput is the median over ``--runs`` fresh models with the same seed,
and ROC-AUC comes from the first run so speed never hides a quality regression.

Peer pairs use matched parameters. LODA uses 100 projections and 10 bins on
every side, and aberrant's default LODA configuration is reported separately.
SHUTTLE is min-max scaled with its global feature ranges before streaming,
because both Half-Space Trees implementations expect inputs in ``[0, 1]``. The
synthetic stream is generated inside ``[0, 1]`` for the same reason.

LocalOutlierFactor uses 10 neighbors on both sides. river's incremental LOF
keeps every point it learns and slows down with each one, while aberrant keeps
its default window of 1,000 points, so both stream only the first 2,000 events.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import platform
import pstats
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import metadata
from typing import Any

import numpy as np
from scipy.stats import rankdata

import aberrant
from aberrant.model.distance import LocalOutlierFactor
from aberrant.model.iforest import HalfSpaceTrees, OnlineIsolationForest
from aberrant.model.sketch import StreamingLODA
from aberrant.stream.dataset import Dataset, load

try:
    from river import anomaly as river_anomaly
except ImportError:
    river_anomaly = None

try:
    from pysad.models import LODA as PySADLODA
except ImportError:
    PySADLODA = None

MODEL_SEED = 42
LOF_EVENTS = 2_000


@dataclass(frozen=True)
class Stream:
    """One labelled stream in every input format the cases need."""

    name: str
    dicts: list[dict[str, float]]
    rows: list[np.ndarray]
    labels: np.ndarray


@dataclass(frozen=True)
class Case:
    """One detector configuration from one library."""

    group: str
    library: str
    parameters: str
    factory: Callable[[], Any]
    input_kind: str = "dict"  # "dict" for score_one/learn_one, "row" for PySAD
    max_events: int | None = None  # stream only this many events when set


def synthetic_stream(n_events: int, n_features: int = 8, seed: int = 7) -> Stream:
    """Three Gaussian clusters inside ``[0, 1]`` with 2% uniform anomalies."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0.25, 0.75, size=(3, n_features))
    cluster = rng.integers(0, 3, size=n_events)
    data = centers[cluster] + rng.normal(0.0, 0.05, size=(n_events, n_features))
    labels = (rng.random(n_events) < 0.02).astype(np.int64)
    anomalies = int(labels.sum())
    data[labels == 1] = rng.uniform(0.0, 1.0, size=(anomalies, n_features))
    return _stream("synthetic", np.clip(data, 0.0, 1.0), labels)


def shuttle_stream(n_events: int) -> Stream:
    """The first ``n_events`` SHUTTLE events, min-max scaled to ``[0, 1]``."""
    with np.load(load(Dataset.SHUTTLE).file_path) as archive:
        data = archive["X"].astype(np.float64)
        labels = archive["y"].astype(np.int64)
    low = data.min(axis=0)
    span = np.where(data.max(axis=0) > low, data.max(axis=0) - low, 1.0)
    scaled = (data - low) / span
    return _stream("SHUTTLE", scaled[:n_events], labels[:n_events])


def _stream(name: str, data: np.ndarray, labels: np.ndarray) -> Stream:
    names = [f"f{index}" for index in range(data.shape[1])]
    dicts = [dict(zip(names, row.tolist(), strict=True)) for row in data]
    rows = [np.ascontiguousarray(row) for row in data]
    return Stream(name=name, dicts=dicts, rows=rows, labels=labels)


def build_cases() -> list[Case]:
    """Return the matched peer pairs plus aberrant-only configurations."""
    cases = [
        Case(
            "Half-Space Trees",
            "aberrant",
            "10 trees, height 8, window 250",
            lambda: HalfSpaceTrees(
                n_trees=10, height=8, window_size=250, seed=MODEL_SEED
            ),
        ),
        Case(
            "LODA",
            "aberrant",
            "100 projections, 10 bins",
            lambda: StreamingLODA(n_projections=100, n_bins=10, seed=MODEL_SEED),
        ),
        Case(
            "LODA",
            "aberrant",
            "defaults (100 projections, 32 bins)",
            lambda: StreamingLODA(seed=MODEL_SEED),
        ),
        Case(
            "OnlineIsolationForest",
            "aberrant",
            "defaults (100 trees, window 2048)",
            lambda: OnlineIsolationForest(seed=MODEL_SEED),
        ),
        Case(
            "LocalOutlierFactor",
            "aberrant",
            f"10 neighbors, first {LOF_EVENTS:,} events",
            lambda: LocalOutlierFactor(k=10),
            max_events=LOF_EVENTS,
        ),
    ]
    if river_anomaly is None:
        print("river is not installed; skipping river cases", file=sys.stderr)
    else:
        cases += [
            Case(
                "Half-Space Trees",
                "river",
                "10 trees, height 8, window 250",
                lambda: river_anomaly.HalfSpaceTrees(
                    n_trees=10, height=8, window_size=250, seed=MODEL_SEED
                ),
            ),
            Case(
                "LODA",
                "river",
                "100 projections, 10 bins",
                lambda: river_anomaly.LODA(
                    n_bins=10, n_random_cuts=100, seed=MODEL_SEED
                ),
            ),
            Case(
                "LocalOutlierFactor",
                "river",
                f"10 neighbors, first {LOF_EVENTS:,} events",
                lambda: river_anomaly.LocalOutlierFactor(n_neighbors=10),
                max_events=LOF_EVENTS,
            ),
        ]
    if PySADLODA is None:
        print("PySAD is not installed; skipping PySAD cases", file=sys.stderr)
    else:
        cases.append(
            Case(
                "LODA",
                "PySAD",
                "100 projections, 10 bins",
                lambda: PySADLODA(
                    num_bins=10, num_random_cuts=100, random_state=MODEL_SEED
                ),
                input_kind="row",
            )
        )
    order = {"aberrant": 0, "river": 1, "PySAD": 2}
    return sorted(cases, key=lambda case: (case.group, order[case.library]))


def run_once(case: Case, stream: Stream) -> tuple[float, np.ndarray]:
    """Score then learn every event with a fresh model; return events/s, scores."""
    if case.max_events is not None:
        stream = Stream(
            name=stream.name,
            dicts=stream.dicts[: case.max_events],
            rows=stream.rows[: case.max_events],
            labels=stream.labels[: case.max_events],
        )
    model = case.factory()
    scores = [0.0] * len(stream.labels)
    if case.input_kind == "row":
        step = model.fit_score_partial
        start = time.perf_counter()
        for index, row in enumerate(stream.rows):
            scores[index] = step(row)
        elapsed = time.perf_counter() - start
    else:
        score_one = model.score_one
        learn_one = model.learn_one
        start = time.perf_counter()
        for index, event in enumerate(stream.dicts):
            scores[index] = score_one(event)
            learn_one(event)
        elapsed = time.perf_counter() - start
    return len(scores) / elapsed, np.asarray(scores, dtype=np.float64)


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney ROC-AUC with average ranks for tied scores."""
    positives = labels == 1
    n_positive = int(positives.sum())
    n_negative = len(labels) - n_positive
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ranks = rankdata(np.nan_to_num(scores, nan=0.0, posinf=1e300, neginf=-1e300))
    rank_sum = float(ranks[positives].sum())
    return (rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)


def measure(
    cases: Sequence[Case], streams: Sequence[Stream], runs: int
) -> list[dict[str, Any]]:
    """Time every case on every stream and return one result row each."""
    results = []
    for stream in streams:
        for case in cases:
            throughputs = []
            first_scores: np.ndarray | None = None
            for _ in range(runs):
                throughput, scores = run_once(case, stream)
                throughputs.append(throughput)
                if first_scores is None:
                    first_scores = scores
            assert first_scores is not None
            row = {
                "stream": stream.name,
                "events": len(first_scores),
                "detector": case.group,
                "library": case.library,
                "parameters": case.parameters,
                "events_per_second": statistics.median(throughputs),
                "runs": throughputs,
                "roc_auc": roc_auc(stream.labels[: len(first_scores)], first_scores),
            }
            results.append(row)
            print(
                f"{stream.name:>9} | {case.library:>8} | {case.group} "
                f"({case.parameters}): {row['events_per_second']:,.0f} ev/s, "
                f"ROC-AUC {row['roc_auc']:.4f}",
                file=sys.stderr,
            )
    return results


def environment() -> dict[str, str]:
    """Record the interpreter and library versions behind a measurement."""
    versions = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "machine": platform.machine(),
        "aberrant": aberrant.__version__,
        "aberrant_path": str(aberrant.__file__),
        "numpy": np.__version__,
    }
    for distribution in ("scipy", "river", "pysad"):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = "not installed"
    return versions


def markdown_table(results: Sequence[dict[str, Any]], runs: int) -> str:
    """Render results as a Markdown table, fastest peer ratio included."""
    lines = [
        f"| Stream | Detector | Library | Parameters | ev/s (median of {runs}) "
        "| vs. fastest peer | ROC-AUC |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in results:
        peers = [
            other["events_per_second"]
            for other in results
            if other["stream"] == row["stream"]
            and other["detector"] == row["detector"]
            and other["parameters"] == row["parameters"]
            and other["library"] != "aberrant"
        ]
        ratio = (
            f"{row['events_per_second'] / max(peers):.2f}x"
            if row["library"] == "aberrant" and peers
            else ""
        )
        lines.append(
            f"| {row['stream']} | {row['detector']} | {row['library']} "
            f"| {row['parameters']} | {row['events_per_second']:,.0f} "
            f"| {ratio} | {row['roc_auc']:.4f} |"
        )
    return "\n".join(lines)


def profile(case: Case, stream: Stream, limit: int) -> None:
    """Print the hottest functions of one run, sorted by own time."""
    profiler = cProfile.Profile()
    profiler.enable()
    run_once(case, stream)
    profiler.disable()
    pstats.Stats(profiler).strip_dirs().sort_stats("tottime").print_stats(limit)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--events", type=int, default=20_000)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--streams",
        default="synthetic,shuttle",
        help="comma-separated subset of: synthetic, shuttle",
    )
    parser.add_argument(
        "--detectors",
        default="",
        help="comma-separated detector names to keep, e.g. 'LODA,Half-Space Trees'",
    )
    parser.add_argument(
        "--libraries", default="", help="comma-separated libraries to keep"
    )
    parser.add_argument("--json", help="also write the results to this JSON file")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="print a cProfile summary of one run per selected case instead",
    )
    parser.add_argument("--profile-limit", type=int, default=20)
    args = parser.parse_args(argv)

    cases = build_cases()
    if args.detectors:
        keep = {name.strip() for name in args.detectors.split(",")}
        cases = [case for case in cases if case.group in keep]
    if args.libraries:
        keep = {name.strip() for name in args.libraries.split(",")}
        cases = [case for case in cases if case.library in keep]

    loaders = {"synthetic": synthetic_stream, "shuttle": shuttle_stream}
    streams = [loaders[name.strip()](args.events) for name in args.streams.split(",")]

    if args.profile:
        for case in cases:
            print(f"### {case.library} {case.group} ({case.parameters})")
            profile(case, streams[0], args.profile_limit)
        return

    versions = environment()
    results = measure(cases, streams, args.runs)
    print(", ".join(f"{key} {value}" for key, value in versions.items()))
    print()
    print(markdown_table(results, args.runs))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"environment": versions, "results": results}, handle, indent=2)


if __name__ == "__main__":
    main()
