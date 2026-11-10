"""Online linear regression by recursive least squares."""

from __future__ import annotations

import math

import numpy as np

from aberrant.base.forecast import Forecast
from aberrant.base.protocols import FeatureMap
from aberrant.forecast._arithmetic import blend
from aberrant.forecast.base import BaseRegressor, check_spread_parameters
from aberrant.utils.validation import (
    FeatureSchema,
    PreparedFeatures,
    coerce_finite_number,
)


class RLSRegressor(BaseRegressor):
    """Predict a target linearly from covariates, by recursive least squares.

    Each event updates the weights exactly as a least-squares refit with
    geometrically faded event weights would, at O(d^2) cost for ``d``
    coefficients. The covariate names are fixed by the first learned event.

    Args:
        forgetting: Weight factor in (0, 1] applied to past events at every
            update. 1 weighs all events equally; 0.999 remembers roughly the
            last 1,000.
        delta: Positive initial variance of the coefficients. Larger values
            mean a weaker prior towards zero.
        fit_intercept: Whether to learn a constant term.
        residual_alpha: Weight of each new absolute error in the faded mean
            absolute error that gives the spread, in (0, 1].
        min_scale: Positive floor for the reported spread, in target units.

    Notes:
        Once ``d`` events are learned, each further event's error is measured
        against the prediction made before learning it, and the faded mean
        absolute error of those errors is the reported spread. The model is
        ready after the first such error, at ``d + 1`` events. With
        ``forgetting < 1`` and covariates that stop varying, the coefficient
        covariance can grow without bound; keep inputs informative or use 1.
        Unrepresentable arithmetic raises OverflowError without changing state.

    Examples:
        ```python
        from aberrant.forecast import RLSRegressor

        model = RLSRegressor()
        for a, b in [(0.0, 1.0), (1.0, 0.0), (1.0, 1.0), (2.0, 1.0)]:
            x = {"a": a, "b": b}
            prediction = model.predict_one(x)  # None until ready
            model.learn_one(x, 2.0 * a - b + 1.0)
        model.predict_one({"a": 3.0, "b": 2.0}).mean  # close to (5.0,)
        ```

    References:
        Haykin, Adaptive Filter Theory, 5th ed., chapter 10 (recursive
        least-squares adaptive filters).
    """

    def __init__(
        self,
        *,
        forgetting: float = 0.999,
        delta: float = 1000.0,
        fit_intercept: bool = True,
        residual_alpha: float = 0.05,
        min_scale: float = 1e-6,
    ) -> None:
        self.forgetting = coerce_finite_number(forgetting, label="forgetting")
        if not 0 < self.forgetting <= 1:
            raise ValueError("forgetting must be in (0, 1]")
        self.delta = coerce_finite_number(delta, label="delta")
        if self.delta <= 0:
            raise ValueError("delta must be positive")
        if not isinstance(fit_intercept, bool):
            raise ValueError("fit_intercept must be a boolean")
        self.fit_intercept = fit_intercept
        self.residual_alpha, self.min_scale = check_spread_parameters(
            residual_alpha, min_scale
        )
        self.reset()

    @property
    def is_ready(self) -> bool:
        """Whether a spread has been measured, after ``d + 1`` events."""
        return self._mae is not None

    @property
    def feature_names(self) -> tuple[str, ...] | None:
        """Covariate names in coefficient order, once learned."""
        return self._schema.names

    @property
    def coefficients(self) -> dict[str, float]:
        """Learned weight per covariate, and ``"intercept"`` when fitted."""
        names = self._schema.names or ()
        weights = dict(zip(names, self._weights.tolist(), strict=False))
        if self.fit_intercept and self._weights.size:
            weights["intercept"] = float(self._weights[-1])
        return weights

    def reset(self) -> None:
        """Forget coefficients, covariate names and the spread."""
        self._schema = FeatureSchema()
        self.n_samples_seen = 0
        self._weights = np.zeros(0)
        self._covariance = np.zeros((0, 0))
        self._mae: float | None = None

    def _design(self, x: FeatureMap) -> tuple[np.ndarray, PreparedFeatures]:
        prepared = self._schema.preview(x)
        values = prepared.values
        if self.fit_intercept:
            values = np.append(values, 1.0)
        return values, prepared

    def predict_one(self, x: FeatureMap) -> Forecast | None:
        """Predict the target of the event with covariates ``x`` at lead 0."""
        z, _ = self._design(x)
        if self._mae is None:
            return None
        with np.errstate(over="ignore", invalid="ignore"):
            mean = float(self._weights @ z)
        if not math.isfinite(mean):
            raise OverflowError("Regression arithmetic exceeds the float range")
        return Forecast(
            origin=self.n_samples_seen,
            leads=(0,),
            mean=(mean,),
            scale=(max(self.min_scale, self._mae),),
        )

    def learn_one(self, x: FeatureMap, y: float) -> None:
        """Learn the target ``y`` of the event with covariates ``x``."""
        target = coerce_finite_number(y, label="y")
        z, prepared = self._design(x)
        d = z.size
        if self._weights.size == 0:
            weights = np.zeros(d)
            covariance = np.eye(d) * self.delta
        else:
            weights, covariance = self._weights, self._covariance
        with np.errstate(over="ignore", invalid="ignore"):
            error = target - float(weights @ z)
            pz = covariance @ z
            denominator = self.forgetting + float(z @ pz)
            gain = pz / denominator
            new_weights = weights + gain * error
            # Joseph form (I - k z')P(I - k z')' + lambda k k', in two rank-one
            # steps: it stays positive where P - k (Pz)' cancels to zero.
            reduced = covariance - np.outer(gain, pz)
            new_covariance = (
                reduced
                - np.outer(reduced @ z, gain)
                + self.forgetting * np.outer(gain, gain)
            ) / self.forgetting
            new_covariance = (new_covariance + new_covariance.T) / 2
        if not (
            math.isfinite(error)
            and math.isfinite(denominator)
            and np.isfinite(new_weights).all()
            and np.isfinite(new_covariance).all()
        ):
            raise OverflowError("Regression arithmetic exceeds the float range")
        mae = self._mae
        if self.n_samples_seen >= d:
            mae = (
                abs(error)
                if mae is None
                else blend(mae, abs(error), self.residual_alpha)
            )
        self._schema.commit(prepared)
        self._weights, self._covariance, self._mae = new_weights, new_covariance, mae
        self.n_samples_seen += 1

    def __repr__(self) -> str:
        return (
            f"RLSRegressor(forgetting={self.forgetting:g}, delta={self.delta:g}, "
            f"fit_intercept={self.fit_intercept})"
        )
