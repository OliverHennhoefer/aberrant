"""Forecast-before-learn evaluation of an online forecaster, lead by lead."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from time import perf_counter_ns

from aberrant import __version__
from aberrant._utils.validation import coerce_event_number
from aberrant.base import ForecasterProtocol
from aberrant.base.forecast import Forecast


@dataclass(frozen=True, slots=True)
class LeadScores:
    """Accuracy at one lead over every scored target.

    ``bias`` is the mean of ``actual - forecast``. ``mase`` divides the MAE by
    the mean absolute seasonal difference of the observed series, so below 1
    beats the in-sample seasonal naive forecast. ``spread_ratio`` divides the
    MAE by the mean spread issued for the scored targets: near 1 when the
    spread estimates the MAE. Each is None when undefined.
    """

    lead: int
    n_scored: int
    mae: float | None
    rmse: float | None
    bias: float | None
    mase: float | None
    spread_ratio: float | None


@dataclass(frozen=True, slots=True)
class ForecastRecord:
    """One processed value; indices start at zero.

    ``forecast`` was issued before ``value`` was learned, so its lead 1
    targets ``value``. ``errors[k - 1]`` is ``value`` minus the lead ``k``
    point of the forecast issued ``k - 1`` events earlier, or None when the
    value is missing or no forecast at that lead targets it.
    """

    index: int
    value: float | None
    forecast: Forecast | None
    errors: tuple[float | None, ...]
    forecast_time_ns: int | None
    learn_time_ns: int | None


@dataclass(frozen=True, slots=True)
class ForecastResult:
    """A detached summary of the evaluator's cumulative processing."""

    n_seen: int
    n_observed: int
    n_forecasts: int
    horizon: int
    warmup: int
    season_length: int
    leads: tuple[LeadScores, ...]
    forecast_time_ns: int | None
    learn_time_ns: int | None
    package_version: str

    def lead(self, lead: int) -> LeadScores:
        """The scores at ``lead``, from 1 to ``horizon``."""
        if isinstance(lead, bool) or not isinstance(lead, int):
            raise KeyError(f"No scores for lead {lead!r}")
        if not 1 <= lead <= self.horizon:
            raise KeyError(f"No scores for lead {lead!r}")
        return self.leads[lead - 1]


@dataclass(slots=True)
class _LeadTotals:
    n: int = 0
    absolute: float = 0.0
    squared: float = 0.0
    signed: float = 0.0
    n_spread: int = 0
    spread: float = 0.0

    def add(self, error: float, spread: float | None) -> None:
        self.n += 1
        self.absolute += abs(error)
        self.squared += error * error
        self.signed += error
        if spread is not None:
            self.n_spread += 1
            self.spread += spread

    def scores(self, lead: int, naive: float | None) -> LeadScores:
        if not self.n:
            return LeadScores(lead, 0, None, None, None, None, None)
        mae = self.absolute / self.n
        spread_ratio = None
        if self.n_spread == self.n and self.spread > 0:
            spread_ratio = mae / (self.spread / self.n)
        return LeadScores(
            lead=lead,
            n_scored=self.n,
            mae=mae,
            rmse=math.sqrt(self.squared / self.n),
            bias=self.signed / self.n,
            mase=mae / naive if naive else None,
            spread_ratio=spread_ratio,
        )


@dataclass(slots=True)
class _Seasonal:
    """Mean absolute difference of observed values ``season_length`` apart."""

    season_length: int
    recent: deque[float | None] = field(init=False)
    n: int = 0
    total: float = 0.0

    def __post_init__(self) -> None:
        self.recent = deque(maxlen=self.season_length)

    def added(self, value: float | None) -> tuple[int, float]:
        past = self.recent[0] if len(self.recent) == self.season_length else None
        if value is None or past is None:
            return self.n, self.total
        return self.n + 1, self.total + abs(value - past)

    @property
    def mean(self) -> float | None:
        return self.total / self.n if self.n else None


def _value(value: object) -> float | None:
    if value is None:
        return None
    return coerce_event_number(value, label="Forecast evaluation value")


class ForecastEvaluator:
    """Forecast each value before learning it, and score every lead.

    Each update asks the forecaster for a forecast, then learns the value.
    A forecast issued before the ``n``-th value targets it at lead 1 and the
    values after it at leads up to the horizon, so every observed value is
    scored once at each lead whose forecast exists. A missing value is
    learned as missing and scored at no lead.

    Args:
        forecaster: The forecaster, used as it is and never reset.
        warmup: Number of values learned before forecasts are scored.
            Forecasts the forecaster declines (``None``) are never scored.
        season_length: Lag of the naive forecast that scales ``mase``; 1
            compares with the last value.
        measure_time: Measure separate forecasting and learning durations.

    Pending forecasts take O(horizon) memory. Forecaster exceptions propagate
    without advancing the evaluator; the forecaster's state after a failed
    ``learn_one`` follows its own contract.
    """

    def __init__(
        self,
        forecaster: ForecasterProtocol,
        *,
        warmup: int = 0,
        season_length: int = 1,
        measure_time: bool = True,
    ) -> None:
        if not isinstance(forecaster, ForecasterProtocol):
            raise TypeError("forecaster must satisfy ForecasterProtocol")
        horizon = forecaster.horizon
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("forecaster.horizon must be a positive integer")
        if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
            raise ValueError("warmup must be a non-negative integer")
        if (
            isinstance(season_length, bool)
            or not isinstance(season_length, int)
            or season_length < 1
        ):
            raise ValueError("season_length must be a positive integer")
        if type(measure_time) is not bool:
            raise ValueError("measure_time must be a boolean")
        self.forecaster = forecaster
        self._horizon = horizon
        self._warmup = warmup
        self._measure_time = measure_time
        self._pending: deque[tuple[int, Forecast | None]] = deque(maxlen=horizon)
        self._totals = [_LeadTotals() for _ in range(horizon)]
        self._seasonal = _Seasonal(season_length)
        self._n_seen = self._n_observed = self._n_forecasts = 0
        self._forecast_ns = self._learn_ns = 0

    def _scored(self, forecast: Forecast | None) -> Forecast | None:
        if forecast is None or self._n_seen < self._warmup:
            return None
        if not isinstance(forecast, Forecast):
            raise TypeError("forecast_one must return a Forecast or None")
        return forecast

    def _errors(
        self, value: float | None, current: Forecast | None
    ) -> list[tuple[float, float | None] | None]:
        errors: list[tuple[float, float | None] | None] = [None] * self._horizon
        if value is None:
            return errors
        for issued, forecast in [*self._pending, (self._n_seen, current)]:
            lead = self._n_seen - issued + 1
            if (
                forecast is not None
                and lead <= self._horizon
                and lead in forecast.leads
            ):
                errors[lead - 1] = (value - forecast.point(lead), forecast.spread(lead))
        return errors

    def update(self, value: float | None) -> ForecastRecord:
        """Forecast, learn ``value``, and return the value's trace record."""
        y = _value(value)
        started = perf_counter_ns() if self._measure_time else 0
        forecast = self._scored(self.forecaster.forecast_one())
        forecast_ns = perf_counter_ns() - started if self._measure_time else None
        errors = self._errors(y, forecast)
        seasonal = self._seasonal.added(y)
        started = perf_counter_ns() if self._measure_time else 0
        self.forecaster.learn_one(y)
        learn_ns = perf_counter_ns() - started if self._measure_time else None
        self._pending.append((self._n_seen, forecast))
        for totals, error in zip(self._totals, errors, strict=True):
            if error is not None:
                totals.add(*error)
        self._seasonal.n, self._seasonal.total = seasonal
        self._seasonal.recent.append(y)
        record = ForecastRecord(
            index=self._n_seen,
            value=y,
            forecast=forecast,
            errors=tuple(None if error is None else error[0] for error in errors),
            forecast_time_ns=forecast_ns,
            learn_time_ns=learn_ns,
        )
        self._n_seen += 1
        self._n_observed += y is not None
        self._n_forecasts += forecast is not None
        self._forecast_ns += forecast_ns or 0
        self._learn_ns += learn_ns or 0
        return record

    def iter_evaluate(self, stream: Iterable[float | None]) -> Iterator[ForecastRecord]:
        """Lazily process an ordered stream; write yielded records to any sink."""
        for value in stream:
            yield self.update(value)

    def evaluate(self, stream: Iterable[float | None]) -> ForecastResult:
        """Consume the stream and return a cumulative result without traces."""
        for _record in self.iter_evaluate(stream):
            pass
        return self.result()

    def result(self) -> ForecastResult:
        """A detached snapshot; this never forecasts or learns."""
        naive = self._seasonal.mean
        return ForecastResult(
            n_seen=self._n_seen,
            n_observed=self._n_observed,
            n_forecasts=self._n_forecasts,
            horizon=self._horizon,
            warmup=self._warmup,
            season_length=self._seasonal.season_length,
            leads=tuple(
                totals.scores(lead, naive)
                for lead, totals in enumerate(self._totals, start=1)
            ),
            forecast_time_ns=self._forecast_ns if self._measure_time else None,
            learn_time_ns=self._learn_ns if self._measure_time else None,
            package_version=__version__,
        )
