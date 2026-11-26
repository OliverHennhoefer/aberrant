"""Echo state networks: a fixed random reservoir with an online linear readout."""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from aberrant._utils.validation import (
    coerce_event_number,
    coerce_finite_number,
    coerce_integer_count,
)
from aberrant.base.forecast import Forecast
from aberrant.base.readiness import Readiness
from aberrant.forecast._arithmetic import LeadErrors, finite, fsum
from aberrant.forecast._regression import seeded_rng
from aberrant.forecast.autoregressive import _differencing
from aberrant.forecast.base import BaseForecaster, check_spread_parameters
from aberrant.forecast.linear import rls_update


def _unit_interval(value: object, label: str, *, closed: bool) -> float:
    number = coerce_finite_number(value, label=label)
    if not (0 < number <= 1 if closed else 0 < number < 1):
        bound = "]" if closed else ")"
        raise ValueError(f"{label} must be in (0, 1{bound}")
    return number


def _positive(value: object, label: str) -> float:
    number = coerce_finite_number(value, label=label)
    if number <= 0:
        raise ValueError(f"{label} must be positive")
    return number


class EchoStateForecaster(BaseForecaster):
    """Forecast with a random recurrent reservoir and a readout fit online.

    Each standardized, optionally differenced value ``u_t`` drives a fixed
    random reservoir of ``n_reservoir`` tanh units::

        x_t = (1 - a) x_{t-1} + a tanh(W x_{t-1} + w u_t + b)

    with leak rate ``a`` and ``W`` sparse with ``connectivity`` and scaled
    to ``spectral_radius``, so the state is a fading nonlinear memory of
    the series (Jaeger, 2001). Only the linear readout from ``[x_t, u_t,
    1]`` to the next value learns, by recursive least squares with
    ``forgetting``, so every event costs the same and learning is exact.
    Leads beyond 1 feed the forecasts back into the reservoir and undo the
    differencing.

    Args:
        horizon: Largest lead to forecast.
        n_reservoir: Number of reservoir units, at least 1.
        spectral_radius: Positive spectral radius of ``W``; below 1 the
            reservoir forgets its start, larger values remember longer.
        leak_rate: Leak rate ``a`` in (0, 1]; smaller values follow slower
            dynamics.
        input_scale: Positive scale of the uniform input weights ``w`` and
            biases ``b``; larger values make the reservoir more nonlinear.
        connectivity: Share in (0, 1] of nonzero recurrent weights.
        season_length: Observations per cycle ``m``; 1 for none.
        difference: Non-seasonal differences ``d``, 0, 1 or 2.
        seasonal_difference: Seasonal differences ``D``, 0 or 1. Needs
            ``season_length`` of at least 2.
        forgetting: Weight factor in (0, 1] applied to past events at every
            readout update.
        delta: Positive initial variance of the readout weights, in
            standardized units; smaller values regularize more.
        warm_up: Differenced values whose mean and standard deviation fix
            the standardization, at least 1.
        residual_alpha: Weight of each new absolute error in the faded mean
            absolute error kept for every lead, in (0, 1]. Until
            ``1 / residual_alpha`` errors are seen at a lead, the plain mean
            is used.
        min_scale: Positive floor for the reported spread, in target units.
        seed: Seed of the reservoir, or ``None``.

    Notes:
        The spread of lead ``h`` is the faded mean absolute error of the lead
        ``h`` forecasts. The forecaster is ready once the longest lead has
        been scored once: after ``d + D m`` values to difference, ``warm_up``
        differenced values and ``horizon`` more values.

        ``learn_one(None)`` learns that a value is missing: once forecasting,
        the lead-1 forecast drives the reservoir in its place and the readout
        learns nothing; each forecast reports the spread of the longer lead
        it now is. Before that, the last observed value takes its place.
        Missing values before the first observed one do not start the series.

        State is O(n_reservoir^2), and each event costs
        O(n_reservoir^2 horizon). Unrepresentable arithmetic raises
        OverflowError without changing state.

    Examples:
        ```python
        import math

        from aberrant.forecast import EchoStateForecaster

        model = EchoStateForecaster(seed=0)
        for t in range(1000):
            model.learn_one(math.sin(t / 3))
        model.forecast_one().mean  # close to (sin(1000 / 3),)
        ```

    References:
        Jaeger, The "echo state" approach to analysing and training
        recurrent neural networks, GMD Report 148, 2001. Lukosevicius, A
        Practical Guide to Applying Echo State Networks, Neural Networks:
        Tricks of the Trade, 2012.
    """

    def __init__(
        self,
        *,
        horizon: int = 1,
        n_reservoir: int = 100,
        spectral_radius: float = 0.9,
        leak_rate: float = 1.0,
        input_scale: float = 0.2,
        connectivity: float = 0.1,
        season_length: int = 1,
        difference: int = 0,
        seasonal_difference: int = 0,
        forgetting: float = 0.999,
        delta: float = 1.0,
        warm_up: int = 32,
        residual_alpha: float = 0.05,
        min_scale: float = 1e-6,
        seed: int | None = None,
    ) -> None:
        super().__init__(horizon=horizon)
        self.n_reservoir = coerce_integer_count(n_reservoir, label="n_reservoir")
        self.spectral_radius = _positive(spectral_radius, "spectral_radius")
        self.leak_rate = _unit_interval(leak_rate, "leak_rate", closed=True)
        self.input_scale = _positive(input_scale, "input_scale")
        self.connectivity = _unit_interval(connectivity, "connectivity", closed=True)
        self.season_length = coerce_integer_count(season_length, label="season_length")
        self.difference = coerce_integer_count(
            difference, label="difference", minimum=0
        )
        if self.difference > 2:
            raise ValueError("difference must be 0, 1 or 2")
        self.seasonal_difference = coerce_integer_count(
            seasonal_difference, label="seasonal_difference", minimum=0
        )
        if self.seasonal_difference > 1:
            raise ValueError("seasonal_difference must be 0 or 1")
        if self.seasonal_difference and self.season_length < 2:
            raise ValueError("seasonal_difference needs season_length >= 2")
        self.forgetting = _unit_interval(forgetting, "forgetting", closed=True)
        self.delta = _positive(delta, "delta")
        self.warm_up = coerce_integer_count(warm_up, label="warm_up")
        self.residual_alpha, self.min_scale = check_spread_parameters(
            residual_alpha, min_scale
        )
        self.seed = seed
        rng = seeded_rng(seed)
        n = self.n_reservoir
        recurrent = rng.standard_normal((n, n)) * (
            rng.random((n, n)) < self.connectivity
        )
        radius = float(np.abs(np.linalg.eigvals(recurrent)).max())
        if radius > 0:
            recurrent *= self.spectral_radius / radius
        self._recurrent = recurrent
        self._inputs = rng.uniform(-self.input_scale, self.input_scale, n)
        self._biases = rng.uniform(-self.input_scale, self.input_scale, n)
        self._differencing = _differencing(
            self.difference, self.seasonal_difference, self.season_length
        )
        self._reset_state()

    @property
    def _undone(self) -> int:
        return self.difference + self.seasonal_difference * self.season_length

    def _reset_state(self) -> None:
        """Forget the scaling, the reservoir state, the readout and the errors."""
        size = self.n_reservoir + 2
        self.n_samples_seen = 0
        # Events since the first observed value.
        self._learned = 0
        self._values: deque[float] = deque(maxlen=self._undone or None)
        self._buffer: list[tuple[float, bool]] = []
        self._scaling: tuple[float, float] | None = None
        self._state = np.zeros(self.n_reservoir)
        self._design: np.ndarray | None = None
        self._weights = np.zeros(size)
        self._covariance = np.eye(size) * self.delta
        self._stale = 0
        self._last = 0.0
        self._errors = LeadErrors(
            self.horizon, self.residual_alpha, self.min_scale, seeded=False
        )

    @property
    def readiness(self) -> Readiness:
        """Ready once a forecast at the longest lead has been scored."""
        assert self._errors.counts is not None
        required = self._undone + self.warm_up + self.horizon
        return Readiness(self._errors.counts[-1] > 0, self._learned, required)

    def _step(self, state: np.ndarray, u: float) -> np.ndarray:
        drive = np.tanh(self._recurrent @ state + self._inputs * u + self._biases)
        stepped: np.ndarray = (1 - self.leak_rate) * state + self.leak_rate * drive
        return stepped

    def _standard(self, value: float, scaling: tuple[float, float]) -> float:
        center, spread = scaling
        return finite((value - center) / spread)

    def learn_one(self, y: float | None) -> None:
        """Learn the next value of the series, or ``None`` for a missing one."""
        value = None if y is None else coerce_event_number(y, label="y")
        if value is None and self._learned == 0:
            self.n_samples_seen += 1
            return
        observed = value is not None
        if value is None:
            issued = self._errors.issued
            value = issued[-1][0] if issued else self._last
        undone = self._undone
        differenced = None
        if undone == 0 or len(self._values) == undone:
            differenced = fsum(
                (
                    value,
                    *(
                        weight * self._values[-lag]
                        for lag, weight in self._differencing
                    ),
                )
            )
        scaling, buffer, pending = self._queued(differenced, observed)
        state, design, weights, covariance = self._driven(scaling, pending)
        mae = (
            self._errors.updated(value)
            if observed and self._errors.issued
            else list(self._errors.mae)
        )
        means = None
        if pending:
            assert scaling is not None and design is not None
            means = self._means(weights, state, design, [*self._values, value], scaling)
        if undone:
            self._values.append(value)
        self._scaling, self._buffer = scaling, buffer
        self._state, self._design = state, design
        self._weights, self._covariance = weights, covariance
        stale = 0 if observed else self._stale + 1
        if means is not None:
            self._errors.publish(mae, means, stale)
        self._stale = stale
        if observed:
            self._last = value
        self._learned += 1
        self.n_samples_seen += 1

    def _queued(
        self, differenced: float | None, observed: bool
    ) -> tuple[
        tuple[float, float] | None, list[tuple[float, bool]], list[tuple[float, bool]]
    ]:
        """Scaling, warm-up buffer and the values that now drive the reservoir."""
        scaling, buffer = self._scaling, self._buffer
        if differenced is None:
            return scaling, buffer, []
        if scaling is not None:
            return scaling, buffer, [(differenced, observed)]
        buffer = [*buffer, (differenced, observed)]
        if len(buffer) < self.warm_up:
            return None, buffer, []
        return _scaling(np.array([v for v, _ in buffer])), [], buffer

    def _driven(
        self, scaling: tuple[float, float] | None, pending: list[tuple[float, bool]]
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
        """Reservoir state, readout design and readout after ``pending``."""
        state, design = self._state, self._design
        weights, covariance = self._weights, self._covariance
        for target, learns in pending:
            assert scaling is not None
            u = self._standard(target, scaling)
            if design is not None and learns:
                _, weights, covariance = rls_update(
                    weights, covariance, design, u, self.forgetting
                )
            state = self._step(state, u)
            design = np.concatenate((state, (u, 1.0)))
        return state, design, weights, covariance

    def _means(
        self,
        weights: np.ndarray,
        state: np.ndarray,
        design: np.ndarray,
        values: list[float],
        scaling: tuple[float, float],
    ) -> tuple[float, ...]:
        """Recursive forecasts of leads ``1..horizon`` from the new state."""
        center, spread = scaling
        means = []
        for lead in range(self.horizon):
            with np.errstate(over="ignore", invalid="ignore"):
                u = finite(float(weights @ design))
                step = finite(center + spread * u)
            mean = fsum(
                (step, *(-weight * values[-lag] for lag, weight in self._differencing))
            )
            means.append(mean)
            values.append(mean)
            if lead + 1 < self.horizon:
                state = self._step(state, u)
                design = np.concatenate((state, (u, 1.0)))
        return tuple(means)

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


def _scaling(values: np.ndarray) -> tuple[float, float]:
    with np.errstate(over="ignore", invalid="ignore"):
        center, spread = float(values.mean()), float(values.std())
    if not (math.isfinite(center) and math.isfinite(spread)):
        raise OverflowError("Forecast arithmetic exceeds the float range")
    return center, spread if spread > 0 else 1.0
