"""Bounded score retention and optional exact ranking metrics."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from importlib import import_module
from typing import cast

from aberrant.base.exceptions import MissingOptionalDependencyError

_Metric = Callable[[Sequence[int], Sequence[float]], float]


class RankingMetrics:
    """Keep only labeled scores and load the evaluation extra on demand."""

    def __init__(self, *, enabled: bool, window_size: int | None) -> None:
        self.pairs: deque[tuple[int, float]] = deque(maxlen=window_size)
        self._ap: _Metric | None = None
        self._auc: _Metric | None = None
        if enabled:
            try:
                metrics = import_module("sklearn.metrics")
            except ImportError as exc:
                raise MissingOptionalDependencyError(
                    "prequential_evaluator", "eval"
                ) from exc
            self._ap = cast(_Metric, metrics.average_precision_score)
            self._auc = cast(_Metric, metrics.roc_auc_score)

    def update(self, label: int | None, score: float) -> None:
        if self._ap is not None and label is not None:
            self.pairs.append((label, score))

    def snapshot(self) -> tuple[float | None, float | None, float | None]:
        if not self.pairs:
            return None, None, None
        labels, scores = zip(*self.pairs, strict=True)
        positives = sum(labels)
        prevalence = positives / len(labels)
        assert self._ap is not None and self._auc is not None
        # AP is undefined without a positive class; ROC AUC needs both classes.
        ap = float(self._ap(labels, scores)) if positives else None
        auc = float(self._auc(labels, scores)) if 0 < positives < len(labels) else None
        return ap, auc, prevalence
