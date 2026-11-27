"""Measure every aberrant detector's speed, latency, state size and quality.

This is the internal benchmark. It is not part of CI. To judge a change, edit
the working tree and compare it with the last commit::

    uv run python benchmarks/detectors.py --cases loda --against HEAD \
        --pareto --require-same-scores --log .benchmarks/results.tsv \
        --note "what the change does"

``--against REF`` checks REF out into a temporary git worktree and imports it
through ``PYTHONPATH``, so both sides share this environment and this harness
and no second ``uv sync`` is needed. Each of ``--rounds`` rounds (7 by
default) starts a fresh worker process per side and runs every case on both
back to back, alternating which goes first, one fresh model per side and
round (more for a detector that streams in under ``--min-seconds``, 0.25).
Both halves of a pair see the same machine load, and fresh processes spread
what a process keeps for its life, such as its hash seed and memory layout,
over the rounds. The report holds the median of the per-round speed ratios
and a distribution-free interval of at least 95% for it (from a sign test;
with 7 rounds, the slowest to the fastest ratio). A case is "faster" or
"slower" only when the interval excludes no change, the median moves by at
least ``--min-gain`` (5%), and as many fresh rounds, which run whenever speed
or p99 moved, agree on their own. After a kept
commit, ``HEAD`` moves with it, so the next run compares against the new
head. ``--log`` appends one row per case to a tab-separated file;
``.benchmarks/`` is git-ignored. Do not run anything else meanwhile.

``--pareto`` judges every axis of every case: speed and p99 latency through
the interval above, p99 with its own ``--latency-tolerance`` (25%); pickled
state, peak and retained memory beyond 2% and 4 KiB (``SIZE_TOLERANCE`` and
``SIZE_FLOOR_BYTES``), from one traced run per side and case;
ROC-AUC and PR-AUC, which move only when the scores do; the event at which
the model becomes ready; and determinism across rounds. It exits 0 when some
axis got better and none got worse, 1 when any got worse, and 3 when nothing
moved beyond the noise. ``--accept-tradeoff REASON`` keeps a change that
made some axis worse and records the reason in the log; such a change needs
review. With ``--cases``, only those cases are judged, so a change to shared
code needs a final run without it.

Without ``--against``, the script measures the working tree once and can
compare with an earlier ``--json`` file through ``--baseline``.

Cases come from the model catalog, so a new detector is measured without
editing this file. The ``tabular`` suite runs every tabular detector at its
defaults (with ``seed=42`` where it takes one) on all features, except for the
few overrides in ``TABULAR_PARAMS``. The ``series``
suite runs the univariate, bivariate and matrix-profile detectors on the first
one or two features. Graph detectors have no labelled stream yet and are left
out. A few slow detectors stream fewer events; ``EVENT_CAPS`` lists them.

Every case scores and then learns each event of a fresh model, warm-up
included. For each case the report holds:

- ``events_per_second``: the median over ``--runs`` fresh models, or over
  as many as ``--min-seconds`` needs (``runs`` records how many).
- ``spread``: the fastest run over the slowest minus 1, a rough noise level.
- ``p50_us``, ``p99_us`` and ``max_us``: per-event score-plus-learn latency
  over all runs. Periodic refits show up in ``p99_us`` and ``max_us``.
- ``state_bytes``: the size of the pickled model after the stream. Pickles
  drop caches, so ``--memory`` also reports ``retained_kib``, everything the
  live model holds after the stream, and ``peak_kib``.
- ``window_filled``: whether the stream outgrew the model's ``window_size``.
  When it did not, eviction never ran and the fingerprint does not cover it.
- ``ready_at``: events learned before the model reported itself ready, or
  null when it never did. A model that is not ready scores 0.
- ``roc_auc`` and ``pr_auc``: over every event, labels from the stream.
- ``fingerprint``: a hash of the exact score sequence. A change that should
  only make a detector faster or smaller must leave it unchanged.
  ``deterministic`` is false when runs with the same seed disagree.

With ``--require-same-scores`` the script exits with status 1 when any score
sequence changed, and with ``--baseline`` also when a row has no counterpart.
``--profile`` prints a cProfile summary per case instead; cProfile inflates
code that makes many small Python calls, so confirm a hotspot's gain with
``--against``.

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
import gc
import hashlib
import importlib
import inspect
import json
import math
import os
import pickle
import platform
import pstats
import statistics
import subprocess
import sys
import tempfile
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
# A window of 2,000 lets the capped stream of 5,000 events exercise eviction.
# Every overridden detector also runs at its defaults as ``<id>@defaults``, so a
# change that helps one regime cannot hide a regression in the other.
TABULAR_PARAMS: dict[str, dict[str, object]] = {
    "cell_neighborhood_detector": {"radius": 0.15, "window_size": 2_000},
    "online_autoencoder_ensemble": {
        "feature_map_warm_up": 500,
        "detector_warm_up": 2_000,
    },
}
SERIES_FEATURES = {"bivariate": 2, "univariate": 1}
TIME_SERIES_IDS = set(SERIES_PARAMS)
LOG_COLUMNS = (
    "time",
    "base",
    "candidate",
    "stream",
    "case",
    "parameters",
    "base_ev_s",
    "candidate_ev_s",
    "ratio",
    "ratio_low",
    "ratio_high",
    "verdict",
    "scores",
    "p99_ratio",
    "state_ratio",
    "peak_ratio",
    "retained_ratio",
    "roc_auc_delta",
    "pr_auc_delta",
    "better",
    "worse",
    "tradeoff",
    "note",
)
# A size counts as changed under ``--pareto`` only beyond both floors. Pickled
# state repeats exactly, but the traced memory of a small model moves by a KiB
# or two between runs of the same code, which is several percent of it.
SIZE_TOLERANCE = 0.02
SIZE_FLOOR_BYTES = 4 * 1024
# Size axes of the Pareto check: name, row field, and whether ``--memory`` is
# needed for it.
SIZE_AXES = (
    ("state", "state_bytes", False),
    ("peak memory", "peak_kib", True),
    ("retained memory", "retained_kib", True),
)


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
        variants = [(model_id, SERIES_PARAMS.get(model_id, {}))]
        if model_id in TABULAR_PARAMS:
            variants = [
                (model_id, TABULAR_PARAMS[model_id]),
                (f"{model_id}@defaults", {}),
            ]
        for name, overrides in variants:
            params = dict(overrides)
            if _takes_seed(model_id):
                params["seed"] = SEED
            cases.append(
                Case(
                    suite=suite,
                    name=name,
                    library="aberrant",
                    parameters=json.dumps(params, sort_keys=True)
                    if params
                    else "defaults",
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


def memory_kib(case: Case, stream: Stream) -> tuple[float, float]:
    """Peak traced allocation of one run, and what the live model retains, in KiB.

    The retained size counts everything the model still holds after the
    stream, caches and buffers included, which pickling drops.
    """
    tracemalloc.start()
    try:
        model, scores, elapsed, _ = run_once(case, stream)
        gc.collect()
        current, peak = tracemalloc.get_traced_memory()
        retained = current - scores.nbytes - elapsed.nbytes
        del model
        return peak / 1024, retained / 1024
    finally:
        tracemalloc.stop()


# The most fresh models ``--min-seconds`` may add to one measurement.
MAX_RUNS = 100


def measure(
    case: Case, stream: Stream, runs: int, memory: bool, min_seconds: float = 0.0
) -> dict[str, Any]:
    """Run one case ``runs`` times, or until ``min_seconds`` are timed, and summarize it.

    A detector that streams 10,000 events in 20 ms is timed mostly by the noise
    of the machine; more fresh models give its median something to work with.
    """
    stream = stream.head(case.max_events or len(stream.labels), case.n_features)
    throughputs: list[float] = []
    latencies: list[np.ndarray] = []
    prints: list[str] = []
    model, first_scores = None, None
    timed_ns = 0
    while len(throughputs) < runs or (
        timed_ns < min_seconds * 1e9 and len(throughputs) < MAX_RUNS
    ):
        model, scores, elapsed, ready_at = run_once(case, stream)
        timed_ns += int(elapsed.sum())
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
        "runs": len(throughputs),
        "spread": max(throughputs) / min(throughputs) - 1
        if len(throughputs) > 1
        else None,
        "p50_us": float(np.percentile(pooled, 50)),
        "p99_us": float(np.percentile(pooled, 99)),
        "max_us": float(pooled.max()),
        "ready_at": ready_at,
        "window_filled": _window_filled(model, len(labels)),
        "state_bytes": state_bytes,
        "roc_auc": roc_auc(labels, first_scores),
        "pr_auc": pr_auc(labels, first_scores),
        "fingerprint": prints[0],
        "deterministic": len(set(prints)) == 1,
    }
    if memory:
        row["peak_kib"], row["retained_kib"] = memory_kib(case, stream)
    return row


def _window_filled(model: Any, n_events: int) -> bool | None:
    """Whether the stream outgrew the model's ``window_size``, if it has one."""
    window = getattr(model, "window_size", None)
    if not isinstance(window, int) or isinstance(window, bool):
        return None
    return n_events > window


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent.parent), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
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
        "aberrant_path": str(Path(aberrant.__file__).resolve().parent),
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


def _spread(row: dict[str, Any]) -> str:
    return "n/a" if row.get("spread") is None else f"{row['spread']:.1%}"


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
        if row.get("window_filled") is False:
            scores += " (window not filled)"
        if row["library"] != "aberrant":
            ready = ""
        elif row["ready_at"] is None:
            ready = "never"
        else:
            ready = f"{row['ready_at']:,}"
        lines.append(
            f"| {row['suite']} | {row['stream']} | {row['case']} | {row['library']} "
            f"| {row['events']:,} | {row['events_per_second']:,.0f} | {_spread(row)} "
            f"| {row['p99_us']:,.0f} | {state} | {ready} | {row['roc_auc']:.4f} | {row['pr_auc']:.4f} "
            f"| `{scores}` |"
        )
    return "\n".join(lines)


def compare(
    results: Sequence[dict[str, Any]], baseline: dict[str, Any]
) -> tuple[str, int]:
    """The change against a baseline as a Markdown table, and the problem count.

    A problem is a changed score sequence, a result with no baseline row, or a
    baseline row of a selected detector that this run did not produce (another
    stream, event count or parameter set).
    """
    before = {_key(row): row for row in baseline["results"]}
    seen = set()
    lines = [
        "| Stream | Detector | Library | Speed | Spread before / after | p99 "
        "| State | ROC-AUC | PR-AUC | Scores |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    problems = 0
    for row in results:
        old = before.get(_key(row))
        if old is None:
            problems += 1
            lines.append(
                f"| {row['stream']} | {row['case']} | {row['library']} "
                "| | | | | | | NO BASELINE |"
            )
            continue
        seen.add(_key(row))
        same = row["fingerprint"] == old["fingerprint"]
        problems += not same
        state = (
            f"{row['state_bytes'] / old['state_bytes'] - 1:+.1%}"
            if row["state_bytes"] and old["state_bytes"]
            else ""
        )
        lines.append(
            f"| {row['stream']} | {row['case']} | {row['library']} "
            f"| {row['events_per_second'] / old['events_per_second'] - 1:+.1%} "
            f"| {_spread(old)} / {_spread(row)} "
            f"| {row['p99_us'] / old['p99_us'] - 1:+.1%} | {state} "
            f"| {row['roc_auc'] - old['roc_auc']:+.4f} "
            f"| {row['pr_auc'] - old['pr_auc']:+.4f} "
            f"| {'same' if same else 'CHANGED'} |"
        )
    selected = {row["case"] for row in results}
    for key, old in before.items():
        if key not in seen and old["case"] in selected:
            problems += 1
            lines.append(
                f"| {old['stream']} | {old['case']} | {old['library']} "
                "| | | | | | | NOT RUN |"
            )
    return "\n".join(lines), problems


def _median_interval(ratios: np.ndarray) -> tuple[float, float]:
    """A distribution-free interval for the median, of at least 95% coverage.

    It runs from the k-th smallest to the k-th largest ratio, with the largest
    k for which a sign test puts at most 2.5% on either side; a bootstrap is
    too narrow with a handful of rounds. With fewer than 6 rounds there is no
    such k, and the interval is unbounded.
    """
    n = len(ratios)
    k, tail = 0, 0.0
    while k < n // 2:
        tail += math.comb(n, k) / 2**n
        if tail > 0.025:
            break
        k += 1
    if k == 0:
        return -math.inf, math.inf
    ordered = np.sort(ratios)
    return float(ordered[k - 1]), float(ordered[n - k])


def against(args: argparse.Namespace) -> int:
    """Alternate the working tree with ``args.against`` and compare them.

    The base is checked out into a temporary git worktree and imported through
    ``PYTHONPATH``, so both sides share this environment and this harness; only
    the ``aberrant`` package differs. Each round runs one fresh model per side,
    alternating which side goes first, and the speed and latency verdicts come
    from the per-round ratios. Each side is one long-lived worker process, and
    every round of a case runs back to back on both sides, so the two halves of
    a pair are seconds apart and see the same load. With ``--memory`` or
    ``--pareto``, one more run per side traces memory, which repeats closely
    enough to need no rounds.
    """
    if args.pareto and args.rounds < 6:
        print("--pareto needs --rounds 6 or more for its intervals", file=sys.stderr)
        return 2
    root = Path(__file__).resolve().parent.parent
    base_sha = _git("rev-parse", "--short", args.against)
    if base_sha == "unknown":
        print(f"cannot resolve {args.against!r}", file=sys.stderr)
        return 2
    candidate = _git("rev-parse", "--short", "HEAD") + (
        "+dirty" if _git("status", "--porcelain", "--", "aberrant") else ""
    )
    child = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--suites",
        args.suites,
        "--streams",
        args.streams,
        "--events",
        str(args.events),
        "--worker",
    ]
    if args.cases:
        child += ["--cases", args.cases]
    min_seconds = 0.25 if args.min_seconds is None else args.min_seconds
    rows = _alternate(root, child, args, min_seconds)
    if rows is None:
        return 2
    lines, log_rows, checks = _verdicts(rows, args, f"{args.against} ({base_sha})")
    for entry in log_rows:
        entry[1:3] = [base_sha, candidate]
    print(f"Candidate {candidate} against {args.against} ({base_sha}):\n")
    print("\n".join(lines))
    if args.log:
        _append_log(Path(args.log), log_rows)
    failed = False
    changed = sum(not check["same_scores"] for check in checks)
    if args.require_same_scores and changed:
        print(f"\n{changed} score sequence(s) changed.", file=sys.stderr)
        failed = True
    slower = sum("speed" in check["worse"] for check in checks)
    if args.forbid_slower and slower:
        print(f"\n{slower} case(s) got slower.", file=sys.stderr)
        failed = True
    if args.pareto:
        status = _pareto_gate(checks, args)
        if status:
            return status
    return 1 if failed else 0


def _pareto_gate(checks: list[dict[str, Any]], args: argparse.Namespace) -> int:
    """Print the Pareto verdict; 0 keeps the change, 1 or 3 rejects it."""
    worse = [check for check in checks if check["worse"]]
    better = [check for check in checks if check["better"]]
    print(
        f"\nPareto check over {len(checks)} case(s): {len(better)} with an axis "
        f"better, {len(worse)} with an axis worse.",
        file=sys.stderr,
    )
    if args.cases:
        print(
            "Only the cases in --cases ran. Before keeping a change outside one "
            "detector's own module, run the check without --cases.",
            file=sys.stderr,
        )
    if worse:
        for check in worse:
            print(
                f"  worse: {check['stream']} {check['case']}: "
                + ", ".join(check["worse"]),
                file=sys.stderr,
            )
        if args.accept_tradeoff:
            print(
                f"Trade-off accepted: {args.accept_tradeoff}\n"
                "List it in the pull request for review.",
                file=sys.stderr,
            )
            return 0
        print(
            "NOT A PARETO IMPROVEMENT. Revert, or keep it only as a reasoned "
            'trade-off with --accept-tradeoff "why it is a net benefit".',
            file=sys.stderr,
        )
        return 1
    if not better:
        print(
            "NO MEASURABLE IMPROVEMENT. Nothing got worse, but nothing got "
            "better beyond the noise either.",
            file=sys.stderr,
        )
        return 3
    print("PARETO IMPROVEMENT.", file=sys.stderr)
    return 0


class _Worker:
    """One ``--worker`` process importing aberrant from ``source``."""

    def __init__(self, child: list[str], source: Path, min_seconds: float) -> None:
        self.min_seconds = min_seconds
        self.process = subprocess.Popen(
            child,
            env={**os.environ, "PYTHONPATH": str(source)},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        hello = json.loads(self._read())
        self.jobs: int = hello["jobs"]
        imported = Path(hello["environment"]["aberrant_path"])
        self.imported_from_source = imported.is_relative_to(source.resolve())
        self.imported = imported

    def _read(self) -> str:
        assert self.process.stdout is not None
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("a benchmark worker stopped")
        return line

    def ask(self, job: int, *, traced: bool = False) -> dict[str, Any]:
        """Measure one job once, tracing memory when ``traced``."""
        assert self.process.stdin is not None
        request = {"job": job, "memory": traced, "min_seconds": self.min_seconds}
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        return dict(json.loads(self._read()))

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        self.process.wait(timeout=60)


def _alternate(
    root: Path, child: list[str], args: argparse.Namespace, min_seconds: float
) -> dict[str, list[list[dict[str, Any]]]] | None:
    """Run every job on a worktree of ``args.against`` and on ``root``, alternating.

    ``rows[side][round]`` lists the result of every job in report order. A job
    whose speed or p99 moved gets as many fresh rounds again, listed by job
    under ``base_extra`` and ``candidate_extra`` (empty for the others). With
    ``--memory`` or ``--pareto``, one traced run per side and job lands under
    ``base_memory`` and ``candidate_memory``.
    """
    with tempfile.TemporaryDirectory(prefix="aberrant-base-") as scratch:
        tree = Path(scratch) / "tree"
        subprocess.run(
            ["git", "-C", str(root), "worktree", "add", "--detach", "--quiet"]
            + [str(tree), args.against],
            check=True,
        )
        try:
            sources = {"base": tree, "candidate": root}
            for side, source in sources.items():
                probe = _Worker(child, source, min_seconds)
                probe.close()
                if not probe.imported_from_source:
                    print(
                        f"{side} imported aberrant from {probe.imported}",
                        file=sys.stderr,
                    )
                    return None
            return _paired_rounds(
                lambda: {
                    side: _Worker(child, source, min_seconds)
                    for side, source in sources.items()
                },
                probe.jobs,
                args,
            )
        finally:
            subprocess.run(
                ["git", "-C", str(root), "worktree", "remove", "--force", str(tree)],
                check=False,
            )


def _paired_rounds(
    start: Callable[[], dict[str, _Worker]], n_jobs: int, args: argparse.Namespace
) -> dict[str, list[list[dict[str, Any]]]]:
    """Each round starts a fresh pair of workers and runs every job on both.

    Fresh processes each round spread what one process carries for its whole
    life, such as its hash seed and memory layout, over the rounds; running a
    job on both sides back to back puts both halves of a pair under the same
    machine load.
    """
    sides = ("base", "candidate")
    traced = args.memory or args.pareto
    memory: dict[str, list[dict[str, Any]]] = {side: [] for side in sides}

    def rounds_of(jobs: list[int], label: str) -> dict[str, list[list[dict[str, Any]]]]:
        runs: dict[str, list[list[dict[str, Any]]]] = {
            side: [[] for _ in range(n_jobs)] for side in sides
        }
        for round_index in range(args.rounds):
            workers = start()
            try:
                for job in jobs:
                    flip = (round_index + job) % 2
                    for side in sides[::-1] if flip else sides:
                        runs[side][job].append(workers[side].ask(job))
                if traced and label == "rounds" and round_index == 0:
                    for job in jobs:
                        for side in sides:
                            memory[side].append(workers[side].ask(job, traced=True))
            finally:
                for worker in workers.values():
                    worker.close()
            print(f"{label}: {round_index + 1} of {args.rounds} done", file=sys.stderr)
        return runs

    first = rounds_of(list(range(n_jobs)), "rounds")
    moved = [
        job
        for job in range(n_jobs)
        if any(
            verdict[3]
            for verdict in _timing(
                first["base"][job], first["candidate"][job], args
            ).values()
        )
    ]
    extra = rounds_of(moved, f"confirming {len(moved)} case(s)") if moved else None
    rows: dict[str, list[list[dict[str, Any]]]] = {}
    for side in sides:
        rows[side] = [
            [first[side][job][round_index] for job in range(n_jobs)]
            for round_index in range(args.rounds)
        ]
        rows[f"{side}_extra"] = extra[side] if extra else [[] for _ in range(n_jobs)]
        rows[f"{side}_memory"] = [memory[side]] if traced else []
    return rows


def _timed(ratios: np.ndarray, min_gain: float) -> tuple[float, float, float, int]:
    """Median ratio, its 95% interval, and +1 up, -1 down or 0 within noise."""
    median = float(np.median(ratios))
    low, high = _median_interval(ratios)
    if low > 1.0 and median >= 1.0 + min_gain:
        return median, low, high, 1
    if high < 1.0 and median <= 1.0 - min_gain:
        return median, low, high, -1
    return median, low, high, 0


def _size_change(
    before: dict[str, Any], after: dict[str, Any], field: str
) -> tuple[float | None, int]:
    """The ratio of a size field, and +1 smaller, -1 larger or 0 within the floors."""
    old, new = before.get(field), after.get(field)
    if old is None or new is None or old <= 0:
        return None, 0
    ratio = float(new / old)
    scale = 1 if field == "state_bytes" else 1024  # the traced fields are in KiB
    if abs(new - old) * scale <= SIZE_FLOOR_BYTES or abs(ratio - 1) <= SIZE_TOLERANCE:
        return ratio, 0
    return ratio, 1 if ratio < 1 else -1


def _timing(
    before: list[dict[str, Any]],
    cand: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, tuple[float, float, float, int]]:
    """Speed and p99 verdicts from paired rounds; +1 is better for both."""
    pairs = list(zip(before, cand, strict=True))
    speed = np.array(
        [c["events_per_second"] / b["events_per_second"] for b, c in pairs]
    )
    p99 = np.array([c["p99_us"] / b["p99_us"] for b, c in pairs])
    median, low, high, direction = _timed(p99, args.latency_tolerance)
    return {
        "speed": _timed(speed, args.min_gain),
        "p99 latency": (median, low, high, -direction),
    }


def _pareto_axes(
    before: list[dict[str, Any]],
    cand: list[dict[str, Any]],
    extra: tuple[list[dict[str, Any]], list[dict[str, Any]]] | None,
    memory: tuple[dict[str, Any], dict[str, Any]] | None,
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Judge one case on every axis; ``better`` and ``worse`` name the axes.

    ``extra`` holds the confirmation rounds of a case whose first rounds moved
    a timing: the move counts only when the fresh rounds agree on its own.
    """
    better: list[str] = []
    worse: list[str] = []

    def judge(axis: str, direction: int) -> None:
        if direction > 0:
            better.append(axis)
        elif direction < 0:
            worse.append(axis)

    first = _timing(before, cand, args)
    again = _timing(*extra, args) if extra else None
    if extra:
        before, cand = before + extra[0], cand + extra[1]
    shown = _timing(before, cand, args)
    for axis, (_, _, _, direction) in first.items():
        judge(axis, direction if again and again[axis][3] == direction else 0)
    speed_median, low, high, _ = shown["speed"]
    p99_median = shown["p99 latency"][0]

    sizes: dict[str, float | None] = {}
    for axis, field, needs_memory in SIZE_AXES:
        pair = memory if needs_memory else (before[0], cand[0])
        ratio, direction = (None, 0) if pair is None else _size_change(*pair, field)
        sizes[axis] = ratio
        judge(axis, direction)

    base_prints = {row["fingerprint"] for row in before}
    cand_prints = {row["fingerprint"] for row in cand}
    same = len(base_prints | cand_prints) == 1
    deltas = {}
    for metric in ("roc_auc", "pr_auc"):
        delta = float(
            np.median([row[metric] for row in cand])
            - np.median([row[metric] for row in before])
        )
        deltas[metric] = 0.0 if same or np.isnan(delta) else delta
        judge(metric.replace("_", "-").upper(), int(np.sign(deltas[metric])))
    if len(cand_prints) > 1 and len(base_prints) == 1:
        worse.append("determinism")

    def ready(row: dict[str, Any]) -> float:
        return math.inf if row["ready_at"] is None else row["ready_at"]

    base_ready, cand_ready = ready(before[0]), ready(cand[0])
    if before[0]["library"] == "aberrant":
        judge("readiness", (cand_ready < base_ready) - (cand_ready > base_ready))
    return {
        "stream": before[0]["stream"],
        "case": before[0]["case"],
        "speed": (speed_median, low, high),
        "p99": p99_median,
        "sizes": sizes,
        "deltas": deltas,
        "ready_at": cand[0]["ready_at"],
        "same_scores": same,
        "better": better,
        "worse": worse,
    }


def _pct(ratio: float | None) -> str:
    return "n/a" if ratio is None else f"{ratio - 1:+.1%}"


def _verdicts(
    rows: dict[str, list[list[dict[str, Any]]]], args: argparse.Namespace, base: str
) -> tuple[list[str], list[list[str]], list[dict[str, Any]]]:
    """Per-case verdicts as table lines, log rows and the per-axis checks."""
    lines = [
        "| Stream | Detector | Base ev/s | Speed (median) | 95% interval | p99 "
        "| State | Peak | Retained | ROC-AUC | PR-AUC | Scores | Pareto |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
        "| ---: | ---: | --- | --- |",
    ]
    log_rows: list[list[str]] = []
    checks = []
    memory_rows = (
        list(zip(rows["base_memory"][0], rows["candidate_memory"][0], strict=True))
        if rows["base_memory"]
        else None
    )
    for index, base_row in enumerate(rows["base"][0]):
        before = [run[index] for run in rows["base"]]
        cand = [run[index] for run in rows["candidate"]]
        if _key(cand[0]) != _key(base_row):
            raise RuntimeError("base and candidate ran different cases")
        memory = memory_rows[index] if memory_rows else None
        if memory and not _key(memory[0]) == _key(memory[1]) == _key(base_row):
            raise RuntimeError("the memory pass ran different cases")
        extra = (rows["base_extra"][index], rows["candidate_extra"][index])
        check = _pareto_axes(
            before, cand, extra if extra[0] else None, memory, args=args
        )
        checks.append(check)
        median, low, high = check["speed"]
        verdict = (
            "faster"
            if "speed" in check["better"]
            else "slower"
            if "speed" in check["worse"]
            else "within noise"
        )
        if check["better"] and check["worse"]:
            pareto = "trade-off: " + ", ".join(
                [f"+{axis}" for axis in check["better"]]
                + [f"-{axis}" for axis in check["worse"]]
            )
        elif check["worse"]:
            pareto = "worse: " + ", ".join(check["worse"])
        elif check["better"]:
            pareto = "better: " + ", ".join(check["better"])
        else:
            pareto = "no change"
        sizes, deltas = check["sizes"], check["deltas"]
        base_speed = np.median([row["events_per_second"] for row in before])
        cand_speed = np.median([row["events_per_second"] for row in cand])
        lines.append(
            f"| {base_row['stream']} | {base_row['case']} | {base_speed:,.0f} "
            f"| {median - 1:+.1%} | {low - 1:+.1%} to {high - 1:+.1%} "
            f"| {_pct(check['p99'])} | {_pct(sizes['state'])} "
            f"| {_pct(sizes['peak memory'])} | {_pct(sizes['retained memory'])} "
            f"| {deltas['roc_auc']:+.4f} | {deltas['pr_auc']:+.4f} "
            f"| {'same' if check['same_scores'] else 'CHANGED'} | {pareto} |"
        )
        log_rows.append(
            [
                time.strftime("%Y-%m-%dT%H:%M:%S"),
                base,
                "",
                base_row["stream"],
                base_row["case"],
                base_row["parameters"],
                f"{base_speed:.1f}",
                f"{cand_speed:.1f}",
                f"{median:.4f}",
                f"{low:.4f}",
                f"{high:.4f}",
                verdict,
                "same" if check["same_scores"] else "changed",
                f"{check['p99']:.4f}",
                *(
                    "" if sizes[axis] is None else f"{sizes[axis]:.4f}"
                    for axis, _, _ in SIZE_AXES
                ),
                f"{deltas['roc_auc']:+.6f}",
                f"{deltas['pr_auc']:+.6f}",
                ",".join(check["better"]),
                ",".join(check["worse"]),
                args.accept_tradeoff if check["worse"] else "",
                args.note,
            ]
        )
    return lines, log_rows, checks


def _append_log(log: Path, entries: list[list[str]]) -> None:
    """Append tab-separated rows, writing the header to a new file."""
    header = "\t".join(LOG_COLUMNS)
    if log.exists():
        with log.open(encoding="utf-8") as handle:
            if handle.readline().rstrip("\n") != header:
                print(
                    f"\n{log} has other columns; not appending. Move it aside "
                    "to start a new log.",
                    file=sys.stderr,
                )
                return
    log.parent.mkdir(parents=True, exist_ok=True)
    new = not log.exists()
    with log.open("a", encoding="utf-8") as handle:
        if new:
            handle.write(header + "\n")
        for entry in entries:
            handle.write("\t".join(field.replace("\t", " ") for field in entry) + "\n")
    print(f"\nAppended {len(entries)} row(s) to {log}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface."""
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
        "--min-seconds",
        type=float,
        default=None,
        help="add fresh models until a case has been timed this long (at most "
        f"{MAX_RUNS} runs); 0 by default, 0.25 for each round of --against",
    )
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
    parser.add_argument(
        "--against",
        metavar="REF",
        help="alternate the working tree with this git ref (for example HEAD) "
        "and report the median speed ratio with a 95% interval",
    )
    parser.add_argument(
        "--rounds", type=int, default=7, help="alternating rounds for --against"
    )
    parser.add_argument(
        "--min-gain",
        type=float,
        default=0.05,
        help="smallest median speed change --against calls faster or slower",
    )
    parser.add_argument(
        "--latency-tolerance",
        type=float,
        default=0.25,
        help="smallest median p99 change --pareto calls better or worse; p99 "
        "moved up to 18%% between runs of the same code on a shared machine",
    )
    parser.add_argument(
        "--log",
        help="append one tab-separated row per case of --against to this file, "
        "for example .benchmarks/results.tsv",
    )
    parser.add_argument("--note", default="", help="the idea, for the --log row")
    parser.add_argument(
        "--forbid-slower",
        action="store_true",
        help="with --against, exit with status 1 when any case got slower",
    )
    parser.add_argument(
        "--pareto",
        action="store_true",
        help="with --against, judge every axis (speed, p99 latency, pickled "
        "state, peak and retained memory, ROC-AUC, PR-AUC, readiness, "
        "determinism) and exit 0 only when some axis got better and none got "
        "worse; 1 when one got worse, 3 when nothing changed beyond the noise. "
        "Implies --memory.",
    )
    parser.add_argument(
        "--accept-tradeoff",
        metavar="REASON",
        default="",
        help="with --pareto, keep a change that makes some axis worse, and "
        "record why it is a net benefit in the --log row",
    )
    parser.add_argument("--quiet", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def profile_cases(cases: Sequence[Case], streams: Sequence[Stream], limit: int) -> None:
    """Print a cProfile summary of one run per case and stream."""
    for stream in streams:
        for case in cases:
            print(f"\n## {stream.name}: {case.name} ({case.library})")
            head = stream.head(case.max_events or len(stream.labels), case.n_features)
            profiler = cProfile.Profile()
            profiler.enable()
            run_once(case, head)
            profiler.disable()
            pstats.Stats(profiler).strip_dirs().sort_stats("tottime").print_stats(limit)


def select(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> tuple[list[Case], list[Stream]]:
    """The cases and streams that ``args`` selects."""
    suites = {suite.strip() for suite in args.suites.split(",") if suite.strip()}
    cases = catalog_cases(suites)
    if args.peers:
        cases += peer_cases()
    if args.cases:
        keep = {name.strip() for name in args.cases.split(",")}
        # A detector id also selects its ``@defaults`` variant.
        cases = [
            case
            for case in cases
            if case.name in keep or case.name.partition("@")[0] in keep
        ]
    if not cases:
        parser.error("no case matches --suites and --cases")
    streams = [
        make_stream(name.strip(), args.events) for name in args.streams.split(",")
    ]
    return cases, streams


def worker(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Serve measurements for ``--against`` over stdin and stdout.

    The first line out holds the environment and the number of jobs, one per
    stream and case in report order. Each line in names a job and whether to
    trace memory; each line out is that job's result row.
    """
    out = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8")
    sys.stdout = sys.stderr  # keep anything a model prints off the protocol
    cases, streams = select(args, parser)
    jobs = [(stream, case) for stream in streams for case in cases]
    out.write(json.dumps({"environment": environment(), "jobs": len(jobs)}) + "\n")
    out.flush()
    for line in sys.stdin:
        request = json.loads(line)
        stream, case = jobs[request["job"]]
        row = measure(case, stream, 1, request["memory"], request["min_seconds"])
        out.write(json.dumps(row) + "\n")
        out.flush()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.against:
        return against(args)
    if args.worker:
        return worker(args, parser)
    cases, streams = select(args, parser)

    if args.profile:
        profile_cases(cases, streams, args.profile_limit)
        return 0

    results = []
    for stream in streams:
        for case in cases:
            row = measure(case, stream, args.runs, args.memory, args.min_seconds or 0)
            results.append(row)
            if args.quiet:
                continue
            print(
                f"{row['stream']:>9} | {row['library']:>8} | {row['case']}: "
                f"{row['events_per_second']:,.0f} ev/s, ROC-AUC {row['roc_auc']:.4f}",
                file=sys.stderr,
            )
    if not args.quiet:
        print(report(results, args.runs))
    if args.json:
        payload = {"environment": environment(), "runs": args.runs, "results": results}
        Path(args.json).write_text(
            json.dumps(payload, indent=1) + "\n", encoding="utf-8"
        )
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        table, problems = compare(results, baseline)
        print(
            f"\nAgainst {args.baseline} (commit {baseline['environment'].get('commit')}):\n"
        )
        print(table)
        if args.require_same_scores and problems:
            print(
                f"\n{problems} row(s) changed scores or have no counterpart.",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
