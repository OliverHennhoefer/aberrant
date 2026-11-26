"""Online regressors beyond least squares: kernel, robust linear and neighbours."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from aberrant._utils.validation import coerce_finite_number, coerce_integer_count
from aberrant.forecast._regression import ScaledRegressor, seeded_rng
from aberrant.forecast.linear import rls_update


def _positive(value: object, label: str) -> float:
    number = coerce_finite_number(value, label=label)
    if number <= 0:
        raise ValueError(f"{label} must be positive")
    return number


def _forgetting(value: object) -> float:
    number = coerce_finite_number(value, label="forgetting")
    if not 0 < number <= 1:
        raise ValueError("forgetting must be in (0, 1]")
    return number


class RandomFourierRegressor(ScaledRegressor):
    """Kernel ridge regression online, through random Fourier features.

    Each standardized covariate vector ``z`` maps to ``n_components`` random
    cosine features ``sqrt(2 / D) cos(W z + b)`` with Gaussian ``W`` of
    standard deviation ``1 / length_scale``, whose inner products approximate
    a Gaussian kernel (Rahimi and Recht, 2007). Recursive least squares with
    ``forgetting`` fits a linear model of ``z``, those features and an
    intercept, so the regression is smooth and nonlinear and each event costs
    the same.

    Args:
        n_components: Number of random features ``D``, at least 1.
        length_scale: Positive kernel length scale, in standard deviations of
            the covariates. Smaller values fit sharper bends.
        forgetting: Weight factor in (0, 1] applied to past events at every
            update.
        delta: Positive initial variance of the weights, in standardized
            target units; smaller values regularize more.
        warm_up: Events whose means and standard deviations fix the scaling
            of the covariates and the target, at least 1.
        residual_alpha: Weight of each new absolute error in the faded mean
            absolute error that gives the spread, in (0, 1].
        min_scale: Positive floor for the reported spread, in target units.
        seed: Seed of the random features, or ``None``.

    Notes:
        Ready after ``warm_up + 1`` events. State is O((d + D)^2) for ``d``
        covariates and each event costs O((d + D)^2). Unrepresentable
        arithmetic raises OverflowError without changing state.

    Examples:
        ```python
        import math

        from aberrant.forecast import RandomFourierRegressor

        model = RandomFourierRegressor(seed=0)
        for i in range(500):
            a = (i % 100) / 25 - 2
            model.learn_one({"a": a}, math.sin(2 * a))
        model.predict_one({"a": 0.5}).mean  # close to (sin(1.0),)
        ```

    References:
        Rahimi and Recht, Random Features for Large-Scale Kernel Machines,
        NeurIPS 2007.
    """

    def __init__(
        self,
        *,
        n_components: int = 64,
        length_scale: float = 1.0,
        forgetting: float = 0.999,
        delta: float = 100.0,
        warm_up: int = 32,
        residual_alpha: float = 0.05,
        min_scale: float = 1e-6,
        seed: int | None = None,
    ) -> None:
        self.n_components = coerce_integer_count(n_components, label="n_components")
        self.length_scale = _positive(length_scale, "length_scale")
        self.forgetting = _forgetting(forgetting)
        self.delta = _positive(delta, "delta")
        self._init_scaling(warm_up, residual_alpha, min_scale)
        self.seed = seed
        seeded_rng(seed)
        self._reset_state()

    def _reset_model(self) -> None:
        self._frequencies = np.zeros((self.n_components, 0))
        self._phases = np.zeros(self.n_components)
        self._weights = np.zeros(0)
        self._covariance = np.zeros((0, 0))

    def _start(self, features: int) -> None:
        rng = seeded_rng(self.seed)
        self._frequencies = rng.normal(
            0.0, 1.0 / self.length_scale, size=(self.n_components, features)
        )
        self._phases = rng.uniform(0.0, 2 * math.pi, size=self.n_components)
        size = features + self.n_components + 1
        self._weights = np.zeros(size)
        self._covariance = np.eye(size) * self.delta

    def _design(self, z: np.ndarray) -> np.ndarray:
        scale = math.sqrt(2 / self.n_components)
        waves = scale * np.cos(self._frequencies @ z + self._phases)
        return np.concatenate((z, waves, (1.0,)))

    def _predict(self, z: np.ndarray) -> float:
        return float(self._weights @ self._design(z))

    def _learned(self, z: np.ndarray, target: float) -> dict[str, Any]:
        _, weights, covariance = rls_update(
            self._weights, self._covariance, self._design(z), target, self.forgetting
        )
        return {"_weights": weights, "_covariance": covariance}


class PassiveAggressiveRegressor(ScaledRegressor):
    """Online linear regression by passive-aggressive updates (PA-II).

    Each event moves the weights just far enough towards fitting it, by the
    smallest change that removes its epsilon-insensitive loss, damped by
    ``C`` (Crammer et al., 2006). Errors within ``epsilon`` leave the
    weights unchanged. Covariates and target are standardized, so
    ``epsilon`` is in standard deviations of the target.

    Args:
        C: Positive aggressiveness; smaller values change the weights less
            per event and average out noise.
        epsilon: Error, in target standard deviations, that needs no update,
            at least 0.
        warm_up: Events whose means and standard deviations fix the scaling
            of the covariates and the target, at least 1.
        residual_alpha: Weight of each new absolute error in the faded mean
            absolute error that gives the spread, in (0, 1].
        min_scale: Positive floor for the reported spread, in target units.

    Notes:
        Ready after ``warm_up + 1`` events. State is O(d) and each event
        costs O(d). Unrepresentable arithmetic raises OverflowError without
        changing state.

    References:
        Crammer, Dekel, Keshet, Shalev-Shwartz and Singer, Online
        Passive-Aggressive Algorithms, JMLR 7, 2006.
    """

    def __init__(
        self,
        *,
        C: float = 0.01,  # noqa: N803
        epsilon: float = 0.0,
        warm_up: int = 32,
        residual_alpha: float = 0.05,
        min_scale: float = 1e-6,
    ) -> None:
        self.C = _positive(C, "C")
        self.epsilon = coerce_finite_number(epsilon, label="epsilon")
        if self.epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        self._init_scaling(warm_up, residual_alpha, min_scale)
        self._reset_state()

    def _reset_model(self) -> None:
        self._weights = np.zeros(0)

    def _start(self, features: int) -> None:
        self._weights = np.zeros(features + 1)

    def _predict(self, z: np.ndarray) -> float:
        return float(self._weights[:-1] @ z + self._weights[-1])

    def _learned(self, z: np.ndarray, target: float) -> dict[str, Any]:
        design = np.append(z, 1.0)
        error = target - float(self._weights @ design)
        loss = max(0.0, abs(error) - self.epsilon)
        step = loss / (float(design @ design) + 1 / (2 * self.C))
        return {"_weights": self._weights + math.copysign(step, error) * design}


class SGDRegressor(ScaledRegressor):
    """Online linear regression by AdaGrad on the Huber loss.

    The Huber loss is squared for errors up to ``huber`` target standard
    deviations and linear beyond, so an anomaly in the target moves the
    weights no more than a large ordinary error. AdaGrad scales each weight's
    step by its own gradient history, so the step needs little tuning.

    Args:
        learning_rate: Positive base step of AdaGrad.
        huber: Positive error, in target standard deviations, beyond which
            the loss is linear.
        l2: L2 penalty on the weights, not the intercept, at least 0.
        warm_up: Events whose means and standard deviations fix the scaling
            of the covariates and the target, at least 1.
        residual_alpha: Weight of each new absolute error in the faded mean
            absolute error that gives the spread, in (0, 1].
        min_scale: Positive floor for the reported spread, in target units.

    Notes:
        Ready after ``warm_up + 1`` events. State is O(d) and each event
        costs O(d). Unrepresentable arithmetic raises OverflowError without
        changing state.

    References:
        Duchi, Hazan and Singer, Adaptive Subgradient Methods for Online
        Learning and Stochastic Optimization, JMLR 12, 2011. Huber, Robust
        Estimation of a Location Parameter, 1964.
    """

    def __init__(
        self,
        *,
        learning_rate: float = 0.1,
        huber: float = 1.35,
        l2: float = 0.0,
        warm_up: int = 32,
        residual_alpha: float = 0.05,
        min_scale: float = 1e-6,
    ) -> None:
        self.learning_rate = _positive(learning_rate, "learning_rate")
        self.huber = _positive(huber, "huber")
        self.l2 = coerce_finite_number(l2, label="l2")
        if self.l2 < 0:
            raise ValueError("l2 must be non-negative")
        self._init_scaling(warm_up, residual_alpha, min_scale)
        self._reset_state()

    def _reset_model(self) -> None:
        self._weights = np.zeros(0)
        self._squares = np.zeros(0)

    def _start(self, features: int) -> None:
        self._weights = np.zeros(features + 1)
        self._squares = np.zeros(features + 1)

    def _predict(self, z: np.ndarray) -> float:
        return float(self._weights[:-1] @ z + self._weights[-1])

    def _learned(self, z: np.ndarray, target: float) -> dict[str, Any]:
        design = np.append(z, 1.0)
        error = float(self._weights @ design) - target
        slope = max(-self.huber, min(self.huber, error))
        gradient = slope * design
        gradient[:-1] += self.l2 * self._weights[:-1]
        squares = self._squares + gradient * gradient
        step = self.learning_rate / np.sqrt(np.where(squares > 0, squares, 1.0))
        return {"_weights": self._weights - step * gradient, "_squares": squares}


class KNNRegressor(ScaledRegressor):
    """Predict the mean target of the nearest recent events.

    The ``window_size`` most recent events are kept, and a prediction is the
    mean target of the ``k`` whose standardized covariates lie nearest, by
    Euclidean distance, or weighted by inverse distance with
    ``weighted=True``. As a forecaster's regressor this is analog
    forecasting: the future of the most similar recent situations.

    Args:
        k: Number of neighbours, at least 1 and at most ``window_size``.
        window_size: Number of recent events searched, at least 1.
        weighted: Whether nearer neighbours count more.
        warm_up: Events whose means and standard deviations fix the scaling
            of the covariates and the target, at least 1.
        residual_alpha: Weight of each new absolute error in the faded mean
            absolute error that gives the spread, in (0, 1].
        min_scale: Positive floor for the reported spread, in target units.

    Notes:
        Ready after ``warm_up + 1`` events. State is O(window_size * d) and
        each prediction costs O(window_size * d).
    """

    def __init__(
        self,
        *,
        k: int = 5,
        window_size: int = 500,
        weighted: bool = False,
        warm_up: int = 32,
        residual_alpha: float = 0.05,
        min_scale: float = 1e-6,
    ) -> None:
        self.k = coerce_integer_count(k, label="k")
        self.window_size = coerce_integer_count(window_size, label="window_size")
        if self.k > self.window_size:
            raise ValueError("k must be at most window_size")
        if not isinstance(weighted, bool):
            raise ValueError("weighted must be a boolean")
        self.weighted = weighted
        self._init_scaling(warm_up, residual_alpha, min_scale)
        self._reset_state()

    def _reset_model(self) -> None:
        self._rows = np.zeros((0, 0))
        self._targets = np.zeros(0)
        self._next = 0

    def _start(self, features: int) -> None:
        self._rows = np.zeros((0, features))

    def _predict(self, z: np.ndarray) -> float:
        targets = self._targets
        if not targets.size:
            return 0.0
        distances = np.sqrt(((self._rows - z) ** 2).sum(axis=1))
        k = min(self.k, targets.size)
        nearest = np.argpartition(distances, k - 1)[:k]
        if not self.weighted:
            return float(targets[nearest].mean())
        weights = 1 / np.maximum(distances[nearest], 1e-12)
        return float(weights @ targets[nearest] / weights.sum())

    def _learned(self, z: np.ndarray, target: float) -> dict[str, Any]:
        if self._targets.size < self.window_size:
            return {
                "_rows": np.vstack((self._rows, z)),
                "_targets": np.append(self._targets, target),
            }
        rows, targets = self._rows.copy(), self._targets.copy()
        rows[self._next], targets[self._next] = z, target
        return {
            "_rows": rows,
            "_targets": targets,
            "_next": (self._next + 1) % self.window_size,
        }
