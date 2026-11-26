"""Compare the accuracy, the spreads and the cost of aberrant's forecasters.

This script is not part of CI. Run it from the repository root::

    uv run python benchmarks/compare_forecasters.py
    uv run python benchmarks/compare_forecasters.py --seeds 3 --markdown

Every forecaster learns the same synthetic series one value at a time and
forecasts leads ``1..horizon`` after each one. For each series and forecaster
the table reports the mean absolute error at lead 1 and at the longest lead
over the events after ``--burn-in``, and how that error compares with the
reported spread: the forecasters report a faded mean absolute error, so a
ratio near 1 means the spread is calibrated. ``KalmanForecaster`` reports a
predictive standard deviation instead, whose ratio is ``sqrt(2 / pi)``, about
0.80, when it is calibrated on Gaussian errors. Microseconds per event cover
learning and forecasting. With ``--seeds`` above 1 every number is the mean
over that many series of each kind.

The series cover a stationary autoregression, a seasonal autoregression, a
random walk, a noisy trend with a fixed season, a series that wanders in
both its level and its season (``(1 - B)(1 - B^m) y = e``), and a series
that switches from the autoregression to a noisy fixed season halfway. A
self-exciting threshold autoregression (SETAR), whose dynamics differ below
and above 0, and a noisy logistic map need a nonlinear model. Except for the
logistic map, all innovations are standard normal, so the best possible
lead-1 error is ``sqrt(2 / pi)``, about 0.80, wherever the model class
contains the process.

The summary ranks the forecasters by their lead-1 error relative to the best
forecaster on each series, as a geometric mean over all series: 1.00 means
best everywhere. It also counts the series a forecaster is best or within 2% of
the best on.
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Callable

import numpy as np

from aberrant.base.protocols import ForecasterProtocol, RegressorProtocol
from aberrant.forecast import (
    AdaptiveRandomForestRegressor,
    AutoRegressive,
    EchoStateForecaster,
    EnsembleForecaster,
    HoeffdingTreeRegressor,
    HoltWinters,
    KalmanForecaster,
    KNNRegressor,
    LagForecaster,
    PassiveAggressiveRegressor,
    RandomFourierRegressor,
    RLSRegressor,
    SeasonalNaive,
    SGDRegressor,
)

SEASON = 12
Results = dict[str, dict[str, tuple[float, ...]]]


def _recursive(n: int, seed: int, step: Callable[[list[float], float], float]):
    rng = np.random.default_rng(seed)
    values = [0.0] * (SEASON + 1)
    for e in rng.standard_normal(n):
        values.append(step(values, float(e)))
    return values[SEASON + 1 :]


def _logistic(n: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    values = [0.3]
    for e in rng.standard_normal(n - 1):
        x = 3.7 * values[-1] * (1 - values[-1]) + 0.005 * float(e)
        values.append(min(max(x, 0.001), 0.999))
    return values


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
        "setar": _recursive(
            n,
            seed,
            lambda y, e: (0.8 * y[-1] + 1.0 if y[-1] < 0 else -0.6 * y[-1] - 0.5) + e,
        ),
        "regime_switch": [
            *_recursive(n // 2, seed, lambda y, e: 0.6 * y[-1] - 0.3 * y[-2] + e),
            *(
                float(v)
                for v in 3 * np.sin(2 * np.pi * t[n // 2 :] / SEASON)
                + rng.standard_normal(n - n // 2)
            ),
        ],
        "logistic": _logistic(n, seed),
    }


def forecasters(horizon: int) -> dict[str, Callable[[], ForecasterProtocol]]:
    h = horizon
    m = SEASON

    def lagged(
        regressor: Callable[[], RegressorProtocol],
    ) -> Callable[[], ForecasterProtocol]:
        return lambda: LagForecaster(regressor(), horizon=h, order=3)

    def classic() -> list[ForecasterProtocol]:
        return [
            HoltWinters(season_length=m, horizon=h),
            KalmanForecaster(horizon=h, season_length=m),
            AutoRegressive(horizon=h, order=2, season_length=m, seasonal_order=1),
            AutoRegressive(
                horizon=h,
                order=2,
                season_length=m,
                seasonal_order=1,
                difference=1,
                seasonal_difference=1,
            ),
        ]

    return {
        "SeasonalNaive": lambda: SeasonalNaive(season_length=m, horizon=h),
        "HoltWinters": lambda: HoltWinters(season_length=m, horizon=h),
        "Kalman(level, fixed)": lambda: KalmanForecaster(horizon=h, learn_ratios=False),
        "Kalman(level)": lambda: KalmanForecaster(horizon=h),
        "Kalman(trend)": lambda: KalmanForecaster(horizon=h, slope_ratio=0.01),
        "Kalman(season, fixed)": lambda: KalmanForecaster(
            horizon=h, season_length=m, learn_ratios=False
        ),
        "Kalman(season)": lambda: KalmanForecaster(horizon=h, season_length=m),
        "Kalman(trend, season)": lambda: KalmanForecaster(
            horizon=h, slope_ratio=0.01, season_length=m
        ),
        "AR(p=2)": lambda: AutoRegressive(horizon=h, order=2),
        "AR(p=2, P=1)": lambda: AutoRegressive(
            horizon=h, order=2, season_length=m, seasonal_order=1
        ),
        "AR(p=2, P=1, d=1, D=1)": lambda: AutoRegressive(
            horizon=h,
            order=2,
            season_length=m,
            seasonal_order=1,
            difference=1,
            seasonal_difference=1,
        ),
        "Lag(RLS)": lagged(RLSRegressor),
        "Lag(PassiveAggressive)": lagged(PassiveAggressiveRegressor),
        "Lag(SGD Huber)": lagged(SGDRegressor),
        "Lag(RandomFourier)": lagged(lambda: RandomFourierRegressor(seed=0)),
        "Lag(KNN)": lagged(lambda: KNNRegressor(k=10, weighted=True)),
        "Lag(HoeffdingTree)": lagged(HoeffdingTreeRegressor),
        "Lag(AdaptiveForest)": lagged(lambda: AdaptiveRandomForestRegressor(seed=0)),
        "EchoState": lambda: EchoStateForecaster(horizon=h, seed=0),
        "EchoState(D=1)": lambda: EchoStateForecaster(
            horizon=h, season_length=m, seasonal_difference=1, seed=0
        ),
        "Ensemble(classic)": lambda: EnsembleForecaster(classic()),
        "Ensemble(mixed)": lambda: EnsembleForecaster(
            [
                *classic(),
                LagForecaster(HoeffdingTreeRegressor(), horizon=h, order=3),
                EchoStateForecaster(horizon=h, seed=0),
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


def summary(results: Results) -> list[tuple[str, float, int, float]]:
    """Relative lead-1 error, series near the best, and cost, best first.

    The relative error is the geometric mean over the series of each
    forecaster's lead-1 error over the best one's, so no series with tiny
    errors outweighs the others.
    """
    labels = next(iter(results.values())).keys()
    rows = []
    for label in labels:
        logs, near, micros = [], 0, []
        for by_label in results.values():
            best = min(row[0] for row in by_label.values())
            ratio = by_label[label][0] / best
            logs.append(math.log(ratio))
            near += ratio <= 1.02
            micros.append(by_label[label][4])
        relative = math.exp(float(np.mean(logs)))
        rows.append((label, relative, near, float(np.mean(micros))))
    return sorted(rows, key=lambda row: row[1])


def run(events: int, burn_in: int, horizon: int, seeds: range) -> Results:
    """Mean scores of every forecaster on every series over ``seeds``."""
    factories = forecasters(horizon)
    totals: dict[str, dict[str, list[tuple[float, ...]]]] = {}
    for seed in seeds:
        for name, values in series(events, seed).items():
            for label, factory in factories.items():
                row = evaluate(factory(), values, horizon, burn_in)
                if not all(math.isfinite(v) for v in row):
                    raise RuntimeError(f"{label} on {name} gave {row}")
                totals.setdefault(name, {}).setdefault(label, []).append(row)
    return {
        name: {
            label: tuple(float(v) for v in np.mean(rows, axis=0))
            for label, rows in by_label.items()
        }
        for name, by_label in totals.items()
    }


def _print_table(results: Results, h: int) -> None:
    print(
        f"{'series':<17} {'forecaster':<23} {'MAE@1':>7} {f'MAE@{h}':>7} "
        f"{'ratio@1':>8} {f'ratio@{h}':>8} {'us/event':>9}"
    )
    for name, by_label in results.items():
        for label, (mae1, mae_h, ratio1, ratio_h, micros) in by_label.items():
            print(
                f"{name:<17} {label:<23} {mae1:7.3f} {mae_h:7.3f} "
                f"{ratio1:8.2f} {ratio_h:8.2f} {micros:9.1f}"
            )
        print()
    print(
        f"{'forecaster':<23} {'relative MAE@1':>15} {'near best':>10} {'us/event':>9}"
    )
    for label, relative, near, micros in summary(results):
        print(f"{label:<23} {relative:15.3f} {near:10d} {micros:9.1f}")


def _print_markdown(results: Results, h: int) -> None:
    count = len(results)
    print(f"| Forecaster | Relative MAE@1 | Near best (of {count}) | µs per event |")
    print("| --- | ---: | ---: | ---: |")
    for label, relative, near, micros in summary(results):
        print(f"| `{label}` | {relative:.3f} | {near} | {micros:.0f} |")
    print()
    print(f"| Series | Best MAE@1 | Best forecaster | Best MAE@{h} |")
    print("| --- | ---: | --- | ---: |")
    for name, by_label in results.items():
        label, row = min(by_label.items(), key=lambda item: item[1][0])
        best_h = min(other[1] for other in by_label.values())
        print(f"| {name} | {row[0]:.3f} | `{label}` | {best_h:.3f} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--events", type=int, default=4000)
    parser.add_argument("--burn-in", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    seeds = range(args.seed, args.seed + args.seeds)
    results = run(args.events, args.burn_in, args.horizon, seeds)
    if args.markdown:
        _print_markdown(results, args.horizon)
    else:
        _print_table(results, args.horizon)


if __name__ == "__main__":
    main()
