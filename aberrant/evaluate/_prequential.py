"""Stateful score-before-learn evaluation for anomaly streams."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from time import perf_counter_ns
from typing import Literal

import numpy as np

from aberrant import __version__
from aberrant.base import ModelProtocol
from aberrant.base.protocols import FeatureMap
from aberrant.catalog import DetectorConfig, WarmupUnit
from aberrant.utils.validation import coerce_finite_number

from ._metrics import RankingMetrics
from ._records import EvaluationRecord, EvaluationResult

Readiness = Callable[[ModelProtocol], bool]
LearnFilter = Callable[[FeatureMap, float | None], bool]


def _nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _binary_label(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool | np.bool_):
        return int(value)
    numeric = coerce_finite_number(value, label="Evaluation label")
    if numeric not in (0.0, 1.0):
        raise ValueError("Evaluation label must be 0, 1 or None")
    return int(numeric)


class PrequentialEvaluator:
    """Evaluate anomaly events against prior state, then optionally learn them.

    Args:
        model: An existing structural model, or a catalog configuration from
            which a fresh model is built. Existing models are never reset.
        warmup: Number of successful learns excluded before scoring. For a
            configuration without transformers, None uses catalog event-based
            warm-up. Pipelines and other warm-up units require an explicit
            value or readiness callback. Existing model instances default to 0.
        readiness: Optional predicate evaluated against current model state
            before scoring, after warm-up. False excludes the candidate from
            scoring while still permitting learning. It must not mutate state.
        learn_filter: Optional decision receiving the event and its raw score
            (None during exclusion). Labels are never supplied to this callback
            or the model. Returning False skips learning, including in warm-up.
        learn_policy_name: Required descriptive name when using learn_filter.
        metrics: Enable optional scikit-learn ranking metrics from ``[eval]``.
            Disabled by default, keeping evaluation usable in the base install.
        metric_window_size: Retain at most this many labeled, scored events.
            None explicitly opts into O(n) retention for exact full-stream AP
            and ROC AUC. The window counts labeled scored events, not arrivals.
        higher_is_more_anomalous: Ranking orientation; None uses the catalog
            value, or True for an existing model. Unknown catalog orientation
            requires an explicit value. Trace scores remain in their raw scale.
        measure_time: Measure separate scoring and learning call durations.

    Calls to update, iter_evaluate and evaluate continue the same model and
    evaluator state. Create a new evaluator for each independent comparison.
    Stream order is preserved; seeds and dataset order are caller-controlled.
    Model exceptions propagate without advancing evaluator statistics. Model
    state rollback is governed by the supplied model's own contract.
    """

    def __init__(  # noqa: PLR0912
        self,
        model: ModelProtocol | DetectorConfig,
        *,
        warmup: int | None = None,
        readiness: Readiness | None = None,
        learn_filter: LearnFilter | None = None,
        learn_policy_name: str | None = None,
        metrics: bool = False,
        metric_window_size: int | None = 1000,
        higher_is_more_anomalous: bool | None = None,
        measure_time: bool = True,
    ) -> None:
        if warmup is not None:
            _nonnegative_int(warmup, "warmup")
        if metric_window_size is not None:
            _nonnegative_int(metric_window_size, "metric_window_size")
            if metric_window_size == 0:
                raise ValueError("metric_window_size must be positive or None")
        if type(metrics) is not bool or type(measure_time) is not bool:
            raise ValueError("metrics and measure_time must be booleans")
        if (
            higher_is_more_anomalous is not None
            and type(higher_is_more_anomalous) is not bool
        ):
            raise ValueError("higher_is_more_anomalous must be a boolean or None")
        if readiness is not None and not callable(readiness):
            raise ValueError("readiness must be callable or None")
        if learn_filter is not None and not callable(learn_filter):
            raise ValueError("learn_filter must be callable or None")
        if learn_filter is not None and (
            not isinstance(learn_policy_name, str) or not learn_policy_name.strip()
        ):
            raise ValueError("learn_filter requires a non-empty learn_policy_name")
        if learn_filter is None and learn_policy_name is not None:
            raise ValueError("learn_policy_name requires a learn_filter")

        self._config: DetectorConfig | None = None
        self._fingerprint: str | None = None
        if isinstance(model, DetectorConfig):
            self._config = model.normalized()
            self._fingerprint = self._config.fingerprint()
            capabilities = self._config.capabilities()
            if warmup is None and readiness is None:
                requirement = capabilities.warmup
                if (
                    self._config.transformers
                    or requirement.unit is not WarmupUnit.EVENTS
                    or requirement.minimum is None
                ):
                    raise ValueError(
                        "Configure warmup or readiness explicitly for this detector"
                    )
                warmup = requirement.minimum
            if higher_is_more_anomalous is None:
                higher_is_more_anomalous = capabilities.higher_is_more_anomalous
                if higher_is_more_anomalous is None:
                    raise ValueError(
                        "Configure score orientation explicitly for this detector"
                    )
        elif not isinstance(model, ModelProtocol):
            raise TypeError("model must satisfy ModelProtocol or be a DetectorConfig")

        self._ranking = RankingMetrics(enabled=metrics, window_size=metric_window_size)
        self._metrics_enabled = metrics
        self.model: ModelProtocol = (
            model.build() if isinstance(model, DetectorConfig) else model
        )
        self._warmup = 0 if warmup is None else warmup
        self._readiness = readiness
        self._learn_filter = learn_filter
        self._policy_name = learn_policy_name or "learn_all"
        self._window_size = metric_window_size
        self._higher = (
            True if higher_is_more_anomalous is None else higher_is_more_anomalous
        )
        self._measure_time = measure_time
        self._n_seen = self._n_scored = self._n_learned = 0
        self._n_warmup = self._n_not_ready = 0
        self._n_labeled = self._n_positive = 0
        self._score_ns = self._learn_ns = 0

    def update(self, x: FeatureMap, label: object = None) -> EvaluationRecord:
        """Process one event and return its immutable trace record.

        Validate the entire event and label before calling the model. No scores
        are fabricated for excluded events. Callback decisions must be booleans.
        Input dictionaries are copied so model/policy calls cannot edit the
        caller's sample. Concurrent calls on one evaluator are unsupported.
        """
        y = _binary_label(label)
        if not x or any(not isinstance(key, str) or not key for key in x):
            raise ValueError("Events require non-empty string feature names")
        event = {
            key: coerce_finite_number(value, label=f"Feature '{key}'")
            for key, value in x.items()
        }
        exclusion: Literal["warmup", "not_ready"] | None = None
        if self._n_learned < self._warmup:
            exclusion = "warmup"
        elif self._readiness is not None:
            ready = self._readiness(self.model)
            if type(ready) is not bool:
                raise ValueError("readiness must return a boolean")
            if not ready:
                exclusion = "not_ready"

        score = None
        score_ns = learn_ns = None
        if exclusion is None:
            score_event = dict(event)
            started = perf_counter_ns() if self._measure_time else 0
            raw_score = self.model.score_one(score_event)
            score_ns = perf_counter_ns() - started if self._measure_time else None
            score = coerce_finite_number(raw_score, label="Anomaly score")
        learn = (
            True
            if self._learn_filter is None
            else self._learn_filter(dict(event), score)
        )
        if type(learn) is not bool:
            raise ValueError("learn_filter must return a boolean")
        if learn:
            learn_event = dict(event)
            started = perf_counter_ns() if self._measure_time else 0
            self.model.learn_one(learn_event)
            learn_ns = perf_counter_ns() - started if self._measure_time else None

        record = EvaluationRecord(
            index=self._n_seen,
            label=y,
            score=score,
            learned=learn,
            exclusion=exclusion,
            score_time_ns=score_ns,
            learn_time_ns=learn_ns,
        )
        self._n_seen += 1
        self._n_learned += int(learn)
        self._n_warmup += int(exclusion == "warmup")
        self._n_not_ready += int(exclusion == "not_ready")
        self._score_ns += score_ns or 0
        self._learn_ns += learn_ns or 0
        if score is not None:
            self._n_scored += 1
            if y is not None:
                self._n_labeled += 1
                self._n_positive += y
            self._ranking.update(y, score if self._higher else -score)
        return record

    def iter_evaluate(
        self, stream: Iterable[tuple[FeatureMap, object]]
    ) -> Iterator[EvaluationRecord]:
        """Lazily process an ordered stream; write yielded records to any sink."""
        for x, label in stream:
            yield self.update(x, label)

    def evaluate(self, stream: Iterable[tuple[FeatureMap, object]]) -> EvaluationResult:
        """Consume the stream and return a cumulative result without storing traces."""
        for _record in self.iter_evaluate(stream):
            pass
        return self.result()

    def result(self) -> EvaluationResult:
        """Compute a detached snapshot; this never scores or learns a model."""
        ap, auc, metric_prevalence = self._ranking.snapshot()
        prevalence = self._n_positive / self._n_labeled if self._n_labeled else None
        return EvaluationResult(
            n_seen=self._n_seen,
            n_scored=self._n_scored,
            n_learned=self._n_learned,
            n_warmup=self._n_warmup,
            n_not_ready=self._n_not_ready,
            n_labeled=self._n_labeled,
            prevalence=prevalence,
            metric_samples=len(self._ranking.pairs),
            metrics_enabled=self._metrics_enabled,
            metric_window_size=self._window_size,
            metric_prevalence=metric_prevalence,
            average_precision=ap,
            roc_auc=auc,
            score_time_ns=self._score_ns if self._measure_time else None,
            learn_time_ns=self._learn_ns if self._measure_time else None,
            package_version=__version__,
            config=self._config,
            config_fingerprint=self._fingerprint,
            warmup=self._warmup,
            uses_readiness_callback=self._readiness is not None,
            learn_policy_name=self._policy_name,
            higher_is_more_anomalous=self._higher,
        )
