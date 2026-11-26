"""Compare the accuracy and the spreads of aberrant's forecasters.

This script is not part of CI. Run it from the repository root::

    uv run python benchmarks/compare_forecasters.py

Every forecaster learns the same synthetic series one value at a time and
forecasts leads ``1..horizon`` after each one. For each series and forecaster
the table reports the mean absolute error at lead 1 and at the longest lead
over the events after ``--burn-in``, and how that error compares with the
reported spread: the forecasters report a faded mean absolute error, so a
ratio near 1 means the spread is calibrated. ``KalmanForecaster`` reports a
predictive standard deviation instead, whose ratio is ``sqrt(2 / pi)``, about
0.80, when it is calibrated on Gaussian errors. Microseconds per event cover
learning and forecasting.

The series cover a stationary autoregression, a seasonal autoregression, a
random walk, a noisy trend with a fixed season, and a series that wanders in
both its level and its season (``(1 - B)(1 - B^m) y = e``), and a series
that switches from the autoregression to a noisy fixed season halfway. All innovations are
standard normal, so the best possible lead-1 error is ``sqrt(2 / pi)``, about
0.80, wherever the model class contains the process.
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Callable

import numpy as np

from aberrant.base.protocols import ForecasterProtocol
from aberrant.forecast import (
    AutoRegressive,
    EnsembleForecaster,
    HoltWinters,
    KalmanForecaster,
    SeasonalNaive,
)

SEASON = 12


def _recursive(n: int, seed: int, step: Callable[[list[float], float], float]):
    rng = np.random.default_rng(seed)
    values = [0.0] * (SEASON + 1)
    for e in rng.standard_normal(n):
        values.append(step(values, float(e)))
    return values[SEASON + 1 :]


def series(n: int, seed: int) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend = 10 + 0.01 * t + 3 * np.sin(2 * np.pi * t / SEASON)
    return {
        "ar2": _recursive(n, seed, lambda y, e: 0.6 * y[-1] - 0.3 * y[-2] + e),
        "seasonal_ar": _recursive(
            n, seed, lambda y, e: 0.5 * y[-1] + 0.8 * y[-12] - 0.4 * y[-13] + e
        ),
        "random_walk": [float(v) for v in np.cumsum(rng.standard_normal(n))],
        "trend_season": [float(v) for v in trend + rng.standard_normal(n)],
        "wandering_season": _recursive(
            n, seed, lambda y, e: y[-1] + y[-12] - y[-13] + e
        ),
        "regime_switch": [
            *_recursive(n // 2, seed, lambda y, e: 0.6 * y[-1] - 0.3 * y[-2] + e),
            *(
                float(v)
                for v in 3 * np.sin(2 * np.pi * t[n // 2 :] / SEASON)
                + rng.standard_normal(n - n // 2)
            ),
        ],
    }


def forecasters(horizon: int) -> dict[str, Callable[[], ForecasterProtocol]]:
    return {
        "SeasonalNaive(m=12)": lambda: SeasonalNaive(
            season_length=SEASON, horizon=horizon
        ),
        "HoltWinters(m=12)": lambda: HoltWinters(season_length=SEASON, horizon=horizon),
        "Kalman(level)": lambda: KalmanForecaster(horizon=horizon),
        "Kalman(trend)": lambda: KalmanForecaster(horizon=horizon, slope_ratio=0.01),
        "Kalman(season)": lambda: KalmanForecaster(
            horizon=horizon, season_length=SEASON
        ),
        "Kalman(season, h=2)": lambda: KalmanForecaster(
            horizon=horizon, season_length=SEASON, harmonics=2
        ),
        "AR(p=2)": lambda: AutoRegressive(horizon=horizon, order=2),
        "AR(p=2, P=1, m=12)": lambda: AutoRegressive(
            horizon=horizon, order=2, season_length=SEASON, seasonal_order=1
        ),
        "AR(p=2, P=1, d=1, D=1)": lambda: AutoRegressive(
            horizon=horizon,
            order=2,
            season_length=SEASON,
            seasonal_order=1,
            difference=1,
            seasonal_difference=1,
        ),
        "Ensemble(HW, Kalman, 2 AR)": lambda: EnsembleForecaster(
            [
                HoltWinters(season_length=SEASON, horizon=horizon),
                KalmanForecaster(horizon=horizon, season_length=SEASON),
                AutoRegressive(
                    horizon=horizon, order=2, season_length=SEASON, seasonal_order=1
                ),
                AutoRegressive(
                    horizon=horizon,
                    order=2,
                    season_length=SEASON,
                    seasonal_order=1,
                    difference=1,
                    seasonal_difference=1,
                ),
            ]
        ),
    }


def evaluate(
    model: ForecasterProtocol, values: list[float], horizon: int, burn_in: int
) -> tuple[float, float, float, float, float]:
    errors: dict[int, list[float]] = {1: [], horizon: []}
    spreads: dict[int, list[float]] = {1: [], horizon: []}
    pending: list[tuple[int, object]] = []
    start = time.perf_counter()
    for index, y in enumerate(values):
        model.learn_one(y)
        forecast = model.forecast_one()
        if forecast is not None and index >= burn_in:
            pending.append((index, forecast))
    elapsed = time.perf_counter() - start
    for origin, forecast in pending:
        for lead, lead_errors in errors.items():
            target = origin + lead
            if target < len(values):
                lead_errors.append(abs(values[target] - forecast.point(lead)))
                spreads[lead].append(forecast.scale[lead - 1])
    first, last = (float(np.mean(errors[lead])) for lead in (1, horizon))
    ratios = (
        float(np.mean(errors[lead]) / np.mean(spreads[lead])) for lead in (1, horizon)
    )
    return first, last, *ratios, 1e6 * elapsed / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--events", type=int, default=4000)
    parser.add_argument("--burn-in", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    h = args.horizon
    print(
        f"{'series':<17} {'forecaster':<23} {'MAE@1':>7} {f'MAE@{h}':>7} "
        f"{'ratio@1':>8} {f'ratio@{h}':>8} {'us/event':>9}"
    )
    for name, values in series(args.events, args.seed).items():
        for label, factory in forecasters(h).items():
            row = evaluate(factory(), values, h, args.burn_in)
            if not all(math.isfinite(v) for v in row):
                raise RuntimeError(f"{label} on {name} gave {row}")
            mae1, mae_h, ratio1, ratio_h, micros = row
            print(
                f"{name:<17} {label:<23} {mae1:7.3f} {mae_h:7.3f} "
                f"{ratio1:8.2f} {ratio_h:8.2f} {micros:9.1f}"
            )
        print()


if __name__ == "__main__":
    main()
