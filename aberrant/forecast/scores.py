"""Nonconformity scores of a realised value against a forecast, and intervals."""

from __future__ import annotations

import math
from typing import Literal

from aberrant.base.forecast import Forecast
from aberrant.utils.validation import coerce_finite_number

NonconformityScore = Literal["absolute", "scaled", "cqr"]
SCORES: tuple[NonconformityScore, ...] = ("absolute", "scaled", "cqr")


def _check(score: str, alpha: float | None) -> float | None:
    if score not in SCORES:
        raise ValueError(f"score must be one of {', '.join(SCORES)}")
    if score != "cqr":
        return None
    if alpha is None:
        raise ValueError("The cqr score needs alpha")
    alpha = coerce_finite_number(alpha, label="alpha")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    return alpha


def _spread(forecast: Forecast, lead: int) -> float:
    spread = forecast.spread(lead)
    if spread is None:
        raise ValueError("The scaled score needs a forecast with a spread")
    return spread


def nonconformity(
    forecast: Forecast,
    lead: int,
    y: float,
    score: NonconformityScore = "absolute",
    *,
    alpha: float | None = None,
) -> float:
    """How far ``y`` falls from the forecast at ``lead``; larger is stranger.

    - ``absolute``: ``|y - mean|``, in target units.
    - ``scaled``: ``|y - mean| / spread``, comparable across leads and series.
    - ``cqr``: ``max(lo - y, y - hi)`` for the quantiles at ``alpha / 2`` and
      ``1 - alpha / 2``; negative inside that interval (conformalized quantile
      regression).

    Feed the score to a calibrator with the target event as its key. Each lead
    is its own score stream, since forecast errors grow with the lead.
    """
    alpha = _check(score, alpha)
    value = coerce_finite_number(y, label="y")
    if score == "cqr":
        assert alpha is not None
        lo = forecast.quantile(lead, alpha / 2)
        hi = forecast.quantile(lead, 1 - alpha / 2)
        return max(lo - value, value - hi)
    residual = abs(value - forecast.point(lead))
    if score == "scaled":
        return residual / _spread(forecast, lead)
    return residual


def interval(
    forecast: Forecast,
    lead: int,
    threshold: float,
    score: NonconformityScore = "absolute",
    *,
    alpha: float | None = None,
) -> tuple[float, float]:
    """The values of ``y`` whose ``score`` stays at or below ``threshold``.

    An infinite threshold, as a calibrator issues before its first score,
    gives ``(-inf, inf)``. A ``cqr`` threshold below ``-(hi - lo) / 2`` leaves
    no such value; the interval then collapses to the quantiles' midpoint.
    """
    alpha = _check(score, alpha)
    if math.isnan(threshold):
        raise ValueError("threshold must not be NaN")
    if threshold == math.inf:
        return -math.inf, math.inf
    if score == "cqr":
        assert alpha is not None
        lo = forecast.quantile(lead, alpha / 2) - threshold
        hi = forecast.quantile(lead, 1 - alpha / 2) + threshold
        if lo > hi:
            lo = hi = (lo + hi) / 2
        return lo, hi
    width = max(threshold, 0.0)
    if score == "scaled":
        width *= _spread(forecast, lead)
    mean = forecast.point(lead)
    return mean - width, mean + width
