"""Forecasts combined online from several forecasters by expert aggregation."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence

import numpy as np

from aberrant._utils.validation import coerce_event_number, coerce_finite_number
from aberrant.base.forecast import Forecast
from aberrant.base.protocols import ForecasterProtocol
from aberrant.base.readiness import Readiness
from aberrant.forecast._arithmetic import LeadErrors
from aberrant.forecast.base import BaseForecaster, check_spread_parameters


class EnsembleForecaster(BaseForecaster):
    """Combine forecasters, weighting each lead by how well each has done.

    Every member learns every value. For each lead the ensemble forecasts a
    weighted mean of the members' forecasts, and learns the weights online
    by ML-Poly (Gaillard, Stoltz and van Erven, 2014) on the absolute error:
    a member gains weight while it would have beaten the ensemble, in
    proportion to how much, scaled by its own record. This needs no learning
    rate, adapts to the scale of the series, and does at most a little worse
    than the best convex combination of the members. ``forgetting`` fades
    old regrets, so the weights follow a member that becomes the best one.

    Args:
        forecasters: Two or more distinct forecasters with the same horizon.
            The ensemble owns them: it teaches them every value.
        forgetting: Weight factor in (0, 1] applied to past regrets at every
            update. 1 never forgets; 0.99 remembers roughly the last 100.
        residual_alpha: Weight of each new absolute error in the faded mean
            absolute error kept for every lead, in (0, 1]. Until
            ``1 / residual_alpha`` errors are seen at a lead, the plain mean
            is used.
        min_scale: Positive floor for the reported spread, in target units.

    Notes:
        The ensemble is ready once every member is ready and its own forecast
        at the longest lead has been scored once. Its spread at lead ``h`` is
        the faded mean absolute error of its own lead ``h`` forecasts. A
        member that issues no forecast at an origin is left out there, and
        the others share its weight.

        ``learn_one(None)`` passes the missing value to every member and
        updates no weight; each forecast then uses the weights and reports
        the spread of the longer lead it now is.

        Each event costs what the members cost plus O(horizon * members). An
        OverflowError from a member propagates after the members before it
        learned the value, so reset the ensemble after one.

    Examples:
        ```python
        from aberrant.forecast import EnsembleForecaster, HoltWinters, SeasonalNaive

        model = EnsembleForecaster(
            [SeasonalNaive(season_length=4, horizon=2),
             HoltWinters(season_length=4, horizon=2)]
        )
        for t in range(40):
            model.learn_one(10.0 + [0.0, 2.0, 0.0, -2.0][t % 4])
        model.weights  # one row per lead, one weight per member
        ```

    References:
        Gaillard, Stoltz and van Erven, A Second-order Bound with Excess
        Losses, COLT 2014 (ML-Poly). Cesa-Bianchi and Lugosi, Prediction,
        Learning, and Games, 2006, chapter 2.
    """

    def __init__(
        self,
        forecasters: Sequence[ForecasterProtocol],
        *,
        forgetting: float = 0.99,
        residual_alpha: float = 0.05,
        min_scale: float = 1e-6,
    ) -> None:
        if isinstance(forecasters, str | bytes) or not isinstance(
            forecasters, Sequence
        ):
            raise ValueError("forecasters must be a sequence of forecasters")
        members = tuple(forecasters)
        if len(members) < 2:
            raise ValueError("forecasters needs at least two forecasters")
        if not all(isinstance(member, ForecasterProtocol) for member in members):
            raise ValueError("every member must be a forecaster")
        if len({id(member) for member in members}) != len(members):
            raise ValueError("every member must be a distinct forecaster")
        horizons = {member.horizon for member in members}
        if len(horizons) != 1:
            raise ValueError("every member must have the same horizon")
        super().__init__(horizon=horizons.pop())
        self.forecasters = members
        self.forgetting = coerce_finite_number(forgetting, label="forgetting")
        if not 0 < self.forgetting <= 1:
            raise ValueError("forgetting must be in (0, 1]")
        self.residual_alpha, self.min_scale = check_spread_parameters(
            residual_alpha, min_scale
        )
        self._reset_state()

    def _reset_state(self) -> None:
        """Forget the regrets and the errors."""
        shape = (self.horizon, len(self.forecasters))
        self.n_samples_seen = 0
        self._regret = np.zeros(shape)
        self._squares = np.zeros(shape)
        # The members' forecasts at each recent origin, NaN where one gave none.
        self._members: deque[np.ndarray] = deque(maxlen=self.horizon)
        self._stale = 0
        self._errors = LeadErrors(
            self.horizon, self.residual_alpha, self.min_scale, seeded=False
        )

    @property
    def readiness(self) -> Readiness:
        """Ready once every member is ready and the longest lead is scored."""
        assert self._errors.counts is not None
        required = [member.readiness.required for member in self.forecasters]
        known = [count for count in required if count is not None]
        total = max(known) + self.horizon if len(known) == len(required) else None
        return Readiness(self._errors.counts[-1] > 0, self.n_samples_seen, total)

    @property
    def weights(self) -> tuple[tuple[float, ...], ...]:
        """Each member's weight at every lead, uniform before any regret."""
        return tuple(
            tuple(row.tolist()) for row in self._weights(self._regret, self._squares)
        )

    def _weights(self, regret: np.ndarray, squares: np.ndarray) -> np.ndarray:
        """ML-Poly weights: positive regret over its sum of squares."""
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = np.where(squares > 0, np.maximum(regret, 0.0) / squares, 0.0)
        totals = raw.sum(axis=1, keepdims=True)
        uniform = np.full_like(raw, 1 / raw.shape[1])
        return np.where(totals > 0, raw / np.where(totals > 0, totals, 1.0), uniform)

    def learn_one(self, y: float | None) -> None:
        """Learn the next value of the series, or ``None`` for a missing one."""
        value = None if y is None else coerce_event_number(y, label="y")
        errors = self._errors
        regret, squares = self._regret, self._squares
        mae = list(errors.mae)
        if value is not None and errors.issued:
            leads = [
                lead
                for lead in range(1, min(self.horizon, len(errors.issued)) + 1)
                if errors.observed[-lead]
            ]
            if leads:
                rows = np.array(leads) - 1
                combined = np.array([errors.issued[-lead][lead - 1] for lead in leads])
                members = np.array(
                    [self._members[-lead][:, lead - 1] for lead in leads]
                )
                gradient = np.sign(combined - value)
                gradient[gradient == 0] = 1.0
                # Linearized regret of the ensemble against each member.
                instant = gradient[:, None] * (combined[:, None] - members)
                instant[np.isnan(instant)] = 0.0
                regret, squares = regret.copy(), squares.copy()
                regret[rows] = self.forgetting * regret[rows] + instant
                squares[rows] = self.forgetting * squares[rows] + instant * instant
            mae = errors.updated(value)
        for member in self.forecasters:
            member.learn_one(value)
        forecasts = [member.forecast_one() for member in self.forecasters]
        stale = 0 if value is not None else self._stale + 1
        issued = self._issued(forecasts, regret, squares, stale)
        if issued is not None and issued[2]:
            stale = self._stale + 1
        if not (np.isfinite(regret).all() and np.isfinite(squares).all()):
            raise OverflowError("Forecast arithmetic exceeds the float range")
        self._regret, self._squares = regret, squares
        self._stale = stale
        if issued is not None:
            means, members, _ = issued
            self._members.append(members)
            errors.publish(mae, means, stale)
        self.n_samples_seen += 1

    def _issued(
        self,
        forecasts: list[Forecast | None],
        regret: np.ndarray,
        squares: np.ndarray,
        stale: int,
    ) -> tuple[tuple[float, ...], np.ndarray, bool] | None:
        """The combined forecast, the members' forecasts and whether all failed.

        When no member forecasts, the last forecast advances as across a gap.
        """
        if not all(member.readiness.ready for member in self.forecasters):
            return None
        if all(forecast is None for forecast in forecasts):
            if not self._errors.issued:
                return None
            last = self._errors.issued[-1]
            members = np.roll(self._members[-1], -1, axis=1)
            return (*last[1:], last[-1]), members, True
        members = np.array(
            [
                forecast.mean if forecast is not None else [math.nan] * self.horizon
                for forecast in forecasts
            ],
            dtype=float,
        )
        available = ~np.isnan(members[:, 0])
        rows = np.minimum(stale + np.arange(self.horizon), self.horizon - 1)
        weights = self._weights(regret, squares)[rows] * available
        totals = weights.sum(axis=1, keepdims=True)
        weights = np.where(
            totals > 0,
            weights / np.where(totals > 0, totals, 1.0),
            available / available.sum(),
        )
        with np.errstate(over="ignore", invalid="ignore"):
            means = (weights * np.where(available[:, None], members, 0.0).T).sum(axis=1)
        if not np.isfinite(means).all():
            raise OverflowError("Forecast arithmetic exceeds the float range")
        return tuple(means.tolist()), members, False

    def forecast_one(self) -> Forecast | None:
        """Forecast leads ``1..horizon``, or ``None`` before readiness."""
        if not self.readiness.ready:
            return None
        return Forecast(
            origin=self.n_samples_seen,
            leads=tuple(range(1, self.horizon + 1)),
            mean=self._errors.issued[-1],
            scale=self._errors.scales,
        )
