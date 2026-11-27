"""Measure every aberrant detector's speed, latency, state size and quality.

This is the internal benchmark: run it before and after a change and compare
the two JSON files. It is not part of CI. From the repository root::

    uv run python benchmarks/detectors.py --json before.json
    # change something
    uv run python benchmarks/detectors.py --json after.json --baseline before.json

Cases come from the model catalog, so a new detector is measured without
editing this file. The ``tabular`` suite runs every tabular detector at its
defaults (with ``seed=42`` where it takes one) on all features, except for the
few overrides in ``TABULAR_PARAMS``. The ``series``
suite runs the univariate, bivariate and matrix-profile detectors on the first
one or two features. Graph detectors have no labelled stream yet and are left
out. A few slow detectors stream fewer events; ``EVENT_CAPS`` lists them.

Every case scores and then learns each event of a fresh model, warm-up
included. For each case the report holds:

- ``events_per_second``: the median over ``--runs`` fresh models.
- ``spread``: the fastest run over the slowest minus 1, a rough noise level.
- ``p50_us``, ``p99_us`` and ``max_us``: per-event score-plus-learn latency
  over all runs. Periodic refits show up in ``p99_us`` and ``max_us``.
- ``state_bytes``: the size of the pickled model after the stream.
- ``ready_at``: events learned before the model reported itself ready, or
  null when it never did. A model that is not ready scores 0.
- ``roc_auc`` and ``pr_auc``: over every event, labels from the stream.
- ``fingerprint``: a hash of the exact score sequence. A change that should
  only make a detector faster or smaller must leave it unchanged.
  ``deterministic`` is false when runs with the same seed disagree.

``--baseline`` prints the change against an earlier JSON file, and with
``--require-same-scores`` the script exits with status 1 when any score
sequence changed. ``--profile`` prints a cProfile summary per case instead.

Streams: ``synthetic`` comes from ``aberrant.datasets.synthetic`` (8
correlated features, every injected anomaly kind, no download); any
``aberrant.datasets.Dataset`` name, such as ``shuttle`` or ``http``, replays
that dataset. Every stream is min-max scaled to ``[0, 1]`` with its global
ranges before the clock starts, because Half-Space Trees expects that range.

``--peers`` adds matched configurations of river and PySAD, when installed,
in a ``peers`` suite. They use matched parameters: Half-Space Trees 10 trees
of height 8 over windows of 250; LODA 100 projections and 10 bins; LOF 10
neighbors over the first 2,000 events (river's LOF keeps every point); iForestASD
100 trees, 256 samples and a window of 2,048 over the first 5,000 events
(PySAD scores through scikit-learn one event at a time); Random Cut Forest 40
trees over 256 points over the first 3,000 events, with PySAD's
``time_decay=1.0`` favoring recent points the most; RS-Hash 24 components and
4 hash tables; xStream 100 projections, 100 chains of depth 15 and windows of
256. PySAD needs 0.7 or later, where ``fit_score_partial`` scores before it
learns.
"""

from __future__ import annotations

import argparse
import contextlib
import cProfile
import hashlib
import importlib
import inspect
import json
import pickle
import platform
import pstats
import statistics
import subprocess
import sys
import time
import tracemalloc
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

import aberrant
from aberrant.catalog import MODEL_CATALOG, create_model, get_model_spec
from aberrant.catalog import model_capabilities as capabilities_of
from aberrant.datasets import Dataset, load, synthetic

SEED = 42
# Detectors that need minutes for the default stream stream only this many events.
EVENT_CAPS = {
    "asd_isolation_forest": 5_000,
    "cell_neighborhood_detector": 5_000,
    "mondrian_isolation_forest": 3_000,
    "random_cut_forest": 3_000,
    "rhf": 5_000,
}
# Detectors that need parameters, or that only wrap or threshold other models.
SKIPPED = {"autoencoder", "null", "score_ensemble", "threshold"}
SERIES_PARAMS: dict[str, dict[str, object]] = {
    "damp": {"subsequence_length": 25},
    "multivariate_rolling_matrix_profile": {"subsequence_length": 25},
    "rolling_matrix_profile": {"subsequence_length": 25},
    "residual_detector": {
        "target": "f0",
        "model": {"id": "seasonal_naive", "params": {"season_length": 100}},
    },
}
# Defaults that would make a detector meaningless on these streams. Streams are
# scaled to [0, 1], where the default radius of 1.5 (sized for standardized
# features) holds every point's neighbors; and the Kitsune defaults of 5,000 plus
# 50,000 warm-up events would leave the ensemble unready for the whole stream.
TABULAR_PARAMS: dict[str, dict[str, object]] = {
    "cell_neighborhood_detector": {"radius": 0.15},
    "online_autoencoder_ensemble": {
        "feature_map_warm_up": 500,
        "detector_warm_up": 2_000,
    },
}
SERIES_FEATURES = {"bivariate": 2, "univariate": 1}
TIME_SERIES_IDS = set(SERIES_PARAMS)


@dataclass(frozen=True)
class Stream:
    """One labelled stream, as feature dicts and as float64 rows."""

    name: str
    dicts: list[dict[str, float]]
    rows: np.ndarray
    labels: np.ndarray

    def head(self, n_events: int, n_features: int | None = None) -> Stream:
        """The first ``n_events`` events, keeping the first ``n_features``."""
        rows = self.rows[:n_events, :n_features]
        names = [f"f{index}" for index in range(rows.shape[1])]
        dicts = [dict(zip(names, row.tolist(), strict=True)) for row in rows]
        return Stream(self.name, dicts, rows, self.labels[:n_events])


@dataclass(frozen=True)
class Case:
    """One detector configuration."""

    suite: str
    name: str
    library: str
    parameters: str
    factory: Callable[[], Any]
    n_features: int | None = None  # None keeps every feature
    max_events: int | None = None
    input_kind: str = "dict"  # "row" for PySAD's fit_score_partial


def make_stream(name: str, n_events: int) -> Stream:
    """Load or generate a stream and min-max scale it to ``[0, 1]``."""
    if name == "synthetic":
        data, labels = synthetic(n_samples=n_events, n_features=8, seed=7).arrays()
    else:
        data, labels = load(Dataset(name)).arrays()
    data = np.asarray(data[:n_events], dtype=np.float64)
    labels = np.asarray(labels[:n_events], dtype=np.int64)
    low = data.min(axis=0)
    span = data.max(axis=0) - low
    scaled = (data - low) / np.where(span > 0, span, 1.0)
    return Stream(name, [], scaled, labels).head(n_events)


def _takes_seed(model_id: str) -> bool:
    module, _, name = get_model_spec(model_id).import_path.rpartition(".")
    cls = getattr(importlib.import_module(module), name)
    return "seed" in inspect.signature(cls).parameters


def catalog_cases(suites: set[str]) -> list[Case]:
    """One case per catalog detector, at its defaults."""
    cases = []
    for model_id in MODEL_CATALOG:
        if model_id in SKIPPED:
            continue
        event_kind = capabilities_of(model_id).event_kind.value
        if event_kind == "tabular" and model_id not in TIME_SERIES_IDS:
            suite, n_features = "tabular", None
        elif event_kind in SERIES_FEATURES or model_id in TIME_SERIES_IDS:
            suite = "series"
            n_features = SERIES_FEATURES.get(event_kind, 1)
            if model_id == "multivariate_rolling_matrix_profile":
                n_features = 2
        else:
            continue  # graph and edge streams
        if suite not in suites:
            continue
        params = dict(SERIES_PARAMS.get(model_id, TABULAR_PARAMS.get(model_id, {})))
        if _takes_seed(model_id):
            params["seed"] = SEED
        cases.append(
            Case(
                suite=suite,
                name=model_id,
                library="aberrant",
                parameters=json.dumps(params, sort_keys=True) if params else "defaults",
                factory=lambda model_id=model_id, params=params: create_model(
                    model_id, params
                ),
                n_features=n_features,
                max_events=EVENT_CAPS.get(model_id),
            )
        )
    return sorted(cases, key=lambda case: case.suite != "tabular")


def peer_cases() -> list[Case]:
    """Matched aberrant, river and PySAD configurations."""

    def aberrant_case(
        name: str, model_id: str, max_events: int | None = None, **params: object
    ) -> Case:
        params.setdefault("seed", SEED)
        if model_id == "local_outlier_factor":
            params.pop("seed")
        return Case(
            "peers",
            name,
            "aberrant",
            json.dumps(params, sort_keys=True),
            lambda: create_model(model_id, params),
            max_events=max_events,
        )

    cases = [
        aberrant_case(
            "Half-Space Trees",
            "half_space_trees",
            n_trees=10,
            height=8,
            window_size=250,
        ),
        aberrant_case("LODA", "loda", n_projections=100, n_bins=10),
        aberrant_case("LOF", "local_outlier_factor", 2_000, k=10),
        aberrant_case("iForestASD", "asd_isolation_forest", 5_000, window_size=2048),
        aberrant_case(
            "Random Cut Forest", "random_cut_forest", 3_000, n_trees=40, sample_size=256
        ),
        aberrant_case("RS-Hash", "rshash", n_members=24, n_hashes=4),
        aberrant_case(
            "xStream", "xstream", k=100, n_chains=100, depth=15, window_size=256
        ),
    ]
    try:
        from river import anomaly  # noqa: PLC0415
    except ImportError:
        print("river is not installed; skipping river cases", file=sys.stderr)
    else:
        cases += [
            Case(
                "peers",
                "Half-Space Trees",
                "river",
                "matched",
                lambda: anomaly.HalfSpaceTrees(
                    n_trees=10, height=8, window_size=250, seed=SEED
                ),
            ),
            Case(
                "peers",
                "LODA",
                "river",
                "matched",
                lambda: anomaly.LODA(n_bins=10, n_random_cuts=100, seed=SEED),
            ),
            Case(
                "peers",
                "LOF",
                "river",
                "matched",
                lambda: anomaly.LocalOutlierFactor(n_neighbors=10),
                max_events=2_000,
            ),
        ]
    try:
        version = tuple(int(part) for part in metadata.version("pysad").split(".")[:2])
        from pysad import models  # noqa: PLC0415
    except (metadata.PackageNotFoundError, ImportError):
        version = (0, 0)
    if version < (0, 7):
        print(
            "PySAD 0.7 or later is not installed; skipping PySAD cases", file=sys.stderr
        )
    else:
        cases += [
            Case(
                "peers",
                "iForestASD",
                "PySAD",
                "matched",
                lambda: models.IForestASD(
                    window_size=2048, n_estimators=100, random_state=SEED
                ),
                max_events=5_000,
                input_kind="row",
            ),
            Case(
                "peers",
                "LODA",
                "PySAD",
                "matched",
                lambda: models.LODA(
                    num_bins=10, num_random_cuts=100, random_state=SEED
                ),
                input_kind="row",
            ),
            Case(
                "peers",
                "Random Cut Forest",
                "PySAD",
                "matched",
                lambda: models.RobustRandomCutForest(
                    num_trees=40, tree_size=256, time_decay=1.0, random_state=SEED
                ),
                max_events=3_000,
                input_kind="row",
            ),
            Case(
                "peers",
                "RS-Hash",
                "PySAD",
                "matched",
                lambda: models.RSHash(
                    np.zeros(8),
                    np.ones(8),
                    num_components=24,
                    num_hash_fns=4,
                    random_state=SEED,
                ),
                input_kind="row",
            ),
            Case(
                "peers",
                "xStream",
                "PySAD",
                "matched",
                lambda: models.xStream(
                    num_components=100,
                    n_chains=100,
                    depth=15,
                    window_size=256,
                    random_state=SEED,
                ),
                input_kind="row",
            ),
        ]
    return cases


def run_once(
    case: Case, stream: Stream
) -> tuple[Any, np.ndarray, np.ndarray, int | None]:
    """Score then learn every event with a fresh model.

    Returns the model, the scores, the nanoseconds each event took, and the
    number of events learned when the model first reported itself ready.
    """
    model = case.factory()
    n_events = len(stream.labels)
    scores = np.empty(n_events, dtype=np.float64)
    elapsed = np.empty(n_events, dtype=np.int64)
    clock = time.perf_counter_ns
    # aberrant models expose a ``readiness`` property; peers have none.
    has_readiness = hasattr(model, "readiness")
    ready_at = None
    if case.input_kind == "row":
        step = model.fit_score_partial
        for index, row in enumerate(stream.rows):
            start = clock()
            scores[index] = step(row)
            elapsed[index] = clock() - start
    else:
        score_one = model.score_one
        learn_one = model.learn_one
        for index, event in enumerate(stream.dicts):
            start = clock()
            scores[index] = score_one(event)
            learn_one(event)
            elapsed[index] = clock() - start
            if ready_at is None and has_readiness and model.readiness.ready:
                ready_at = index + 1
    return model, scores, elapsed, ready_at


def _ranks(values: np.ndarray) -> np.ndarray:
    """Ranks from 1 with ties sharing their average rank."""
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    boundaries = np.flatnonzero(np.diff(ordered)) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [len(values)]))
    ranks = np.empty(len(values), dtype=np.float64)
    for start, stop in zip(starts, stops, strict=True):
        ranks[order[start:stop]] = (start + stop + 1) / 2
    return ranks


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney ROC-AUC; NaN when one class is missing."""
    positives = labels == 1
    n_positive = int(positives.sum())
    n_negative = len(labels) - n_positive
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ranks = _ranks(np.nan_to_num(scores, nan=0.0, posinf=1e300, neginf=-1e300))
    rank_sum = float(ranks[positives].sum())
    return (rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)


def pr_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Average precision, with tied scores counted as one threshold."""
    n_positive = int((labels == 1).sum())
    if n_positive == 0:
        return float("nan")
    clean = np.nan_to_num(scores, nan=0.0, posinf=1e300, neginf=-1e300)
    order = np.argsort(-clean, kind="mergesort")
    ordered, hits = clean[order], (labels[order] == 1).astype(np.float64)
    last = np.r_[np.flatnonzero(np.diff(ordered)), len(ordered) - 1]
    true_positives = np.cumsum(hits)[last]
    precision = true_positives / (last + 1)
    recall_gain = np.diff(np.r_[0.0, true_positives]) / n_positive
    return float(np.sum(precision * recall_gain))


def fingerprint(scores: np.ndarray) -> str:
    """A short hash of the exact score sequence, NaN payloads normalized."""
    canonical = np.where(np.isnan(scores), np.nan, scores).astype("<f8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()[:16]


def peak_kib(case: Case, stream: Stream) -> float:
    """Peak traced Python allocation of one run, in KiB."""
    tracemalloc.start()
    try:
        run_once(case, stream)
        return tracemalloc.get_traced_memory()[1] / 1024
    finally:
        tracemalloc.stop()


def measure(case: Case, stream: Stream, runs: int, memory: bool) -> dict[str, Any]:
    """Run one case ``runs`` times and summarize it."""
    stream = stream.head(case.max_events or len(stream.labels), case.n_features)
    throughputs, latencies, prints = [], [], []
    model, first_scores = None, None
    for _ in range(runs):
        model, scores, elapsed, ready_at = run_once(case, stream)
        throughputs.append(len(scores) / (elapsed.sum() / 1e9))
        latencies.append(elapsed)
        prints.append(fingerprint(scores))
        if first_scores is None:
            first_scores = scores
    assert first_scores is not None
    pooled = np.concatenate(latencies) / 1e3
    try:
        state_bytes: int | None = len(pickle.dumps(model))
    except Exception:  # noqa: BLE001 - some peer models do not pickle
        state_bytes = None
    labels = stream.labels
    row = {
        "suite": case.suite,
        "stream": stream.name,
        "case": case.name,
        "library": case.library,
        "parameters": case.parameters,
        "events": len(labels),
        "features": stream.rows.shape[1],
        "events_per_second": statistics.median(throughputs),
        "spread": max(throughputs) / min(throughputs) - 1,
        "p50_us": float(np.percentile(pooled, 50)),
        "p99_us": float(np.percentile(pooled, 99)),
        "max_us": float(pooled.max()),
        "ready_at": ready_at,
        "state_bytes": state_bytes,
        "roc_auc": roc_auc(labels, first_scores),
        "pr_auc": pr_auc(labels, first_scores),
        "fingerprint": prints[0],
        "deterministic": len(set(prints)) == 1,
    }
    if memory:
        row["peak_kib"] = peak_kib(case, stream)
    return row


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def environment() -> dict[str, Any]:
    """Record what a measurement depends on."""
    versions = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "aberrant": aberrant.__version__,
        "numpy": np.__version__,
        "commit": _git("rev-parse", "--short", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--", "aberrant")),
    }
    for distribution in ("river", "pysad"):
        with contextlib.suppress(metadata.PackageNotFoundError):
            versions[distribution] = metadata.version(distribution)
    return versions


def _key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["suite"],
        row["stream"],
        row["case"],
        row["library"],
        row["parameters"],
        row["events"],
    )


def report(results: Sequence[dict[str, Any]], runs: int) -> str:
    """Results as a Markdown table."""
    lines = [
        f"| Suite | Stream | Detector | Library | Events | ev/s (median of {runs}) "
        "| Spread | p99 us | State KiB | Ready at | ROC-AUC | PR-AUC | Scores |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in results:
        state = (
            "" if row["state_bytes"] is None else f"{row['state_bytes'] / 1024:,.0f}"
        )
        scores = row["fingerprint"] + ("" if row["deterministic"] else " (varies)")
        if row["library"] != "aberrant":
            ready = ""
        elif row["ready_at"] is None:
            ready = "never"
        else:
            ready = f"{row['ready_at']:,}"
        lines.append(
            f"| {row['suite']} | {row['stream']} | {row['case']} | {row['library']} "
            f"| {row['events']:,} | {row['events_per_second']:,.0f} | {row['spread']:.1%} "
            f"| {row['p99_us']:,.0f} | {state} | {ready} | {row['roc_auc']:.4f} | {row['pr_auc']:.4f} "
            f"| `{scores}` |"
        )
    return "\n".join(lines)


def compare(
    results: Sequence[dict[str, Any]], baseline: dict[str, Any]
) -> tuple[str, int]:
    """The change against a baseline as a Markdown table, and the changed score count."""
    before = {_key(row): row for row in baseline["results"]}
    lines = [
        "| Stream | Detector | Library | Speed | p99 | State | ROC-AUC | PR-AUC | Scores |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    changed = 0
    for row in results:
        old = before.get(_key(row))
        if old is None:
            lines.append(
                f"| {row['stream']} | {row['case']} | {row['library']} | new | | | | | |"
            )
            continue
        same = row["fingerprint"] == old["fingerprint"]
        changed += not same
        state = (
            f"{row['state_bytes'] / old['state_bytes'] - 1:+.1%}"
            if row["state_bytes"] and old["state_bytes"]
            else ""
        )
        lines.append(
            f"| {row['stream']} | {row['case']} | {row['library']} "
            f"| {row['events_per_second'] / old['events_per_second'] - 1:+.1%} "
            f"| {row['p99_us'] / old['p99_us'] - 1:+.1%} | {state} "
            f"| {row['roc_auc'] - old['roc_auc']:+.4f} | {row['pr_auc'] - old['pr_auc']:+.4f} "
            f"| {'same' if same else 'CHANGED'} |"
        )
    return "\n".join(lines), changed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suites", default="tabular,series", help="tabular, series")
    parser.add_argument(
        "--cases", default="", help="comma-separated catalog ids or names to keep"
    )
    parser.add_argument(
        "--streams",
        default="synthetic",
        help="synthetic or Dataset names, comma-separated",
    )
    parser.add_argument("--events", type=int, default=10_000)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--memory", action="store_true", help="also trace peak allocation (slow)"
    )
    parser.add_argument(
        "--peers", action="store_true", help="add river and PySAD comparisons"
    )
    parser.add_argument(
        "--json", help="write the results and the environment to this file"
    )
    parser.add_argument(
        "--baseline", help="compare with the JSON file of an earlier run"
    )
    parser.add_argument(
        "--require-same-scores",
        action="store_true",
        help="exit with status 1 when a score sequence differs from --baseline",
    )
    parser.add_argument(
        "--profile", action="store_true", help="print a cProfile summary per case"
    )
    parser.add_argument("--profile-limit", type=int, default=25)
    args = parser.parse_args(argv)

    suites = {suite.strip() for suite in args.suites.split(",") if suite.strip()}
    cases = catalog_cases(suites)
    if args.peers:
        cases += peer_cases()
    if args.cases:
        keep = {name.strip() for name in args.cases.split(",")}
        cases = [case for case in cases if case.name in keep]
    if not cases:
        parser.error("no case matches --suites and --cases")
    streams = [
        make_stream(name.strip(), args.events) for name in args.streams.split(",")
    ]

    if args.profile:
        for stream in streams:
            for case in cases:
                print(f"\n## {stream.name}: {case.name} ({case.library})")
                head = stream.head(
                    case.max_events or len(stream.labels), case.n_features
                )
                profiler = cProfile.Profile()
                profiler.enable()
                run_once(case, head)
                profiler.disable()
                pstats.Stats(profiler).strip_dirs().sort_stats("tottime").print_stats(
                    args.profile_limit
                )
        return 0

    results = []
    for stream in streams:
        for case in cases:
            row = measure(case, stream, args.runs, args.memory)
            results.append(row)
            print(
                f"{row['stream']:>9} | {row['library']:>8} | {row['case']}: "
                f"{row['events_per_second']:,.0f} ev/s, ROC-AUC {row['roc_auc']:.4f}",
                file=sys.stderr,
            )
    print(report(results, args.runs))
    if args.json:
        payload = {"environment": environment(), "runs": args.runs, "results": results}
        Path(args.json).write_text(
            json.dumps(payload, indent=1) + "\n", encoding="utf-8"
        )
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        table, changed = compare(results, baseline)
        print(
            f"\nAgainst {args.baseline} (commit {baseline['environment'].get('commit')}):\n"
        )
        print(table)
        if args.require_same_scores and changed:
            print(f"\n{changed} score sequence(s) changed.", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
