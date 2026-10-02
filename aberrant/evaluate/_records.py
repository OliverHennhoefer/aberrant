"""Immutable observations and summaries from prequential evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from aberrant.catalog import DetectorConfig


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    """One successfully processed event; indices start at zero.

    Excluded events have no score. Labels are evaluation data only. Timing
    measures the model calls, excluding validation, readiness and policy logic.
    """

    index: int
    label: int | None
    score: float | None
    learned: bool
    exclusion: Literal["warmup", "not_ready"] | None
    score_time_ns: int | None
    learn_time_ns: int | None


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """A detached summary of the evaluator's cumulative processing.

    Ranking metrics and ``metric_prevalence`` cover only ``metric_samples``:
    the latest labeled, scored events within ``metric_window_size``, or all
    such events when the window is None. ``prevalence`` covers all labeled,
    scored events. Metrics are None when disabled or mathematically undefined.
    Exact full-stream metrics retain O(n) score/label pairs; windowed metrics
    retain O(window_size). No event dictionaries or trace records are retained.
    """

    n_seen: int
    n_scored: int
    n_learned: int
    n_warmup: int
    n_not_ready: int
    n_labeled: int
    prevalence: float | None
    metric_samples: int
    metrics_enabled: bool
    metric_window_size: int | None
    metric_prevalence: float | None
    average_precision: float | None
    roc_auc: float | None
    score_time_ns: int | None
    learn_time_ns: int | None
    package_version: str
    config: DetectorConfig | None
    config_fingerprint: str | None
    warmup: int
    uses_readiness_callback: bool
    learn_policy_name: str
    higher_is_more_anomalous: bool
