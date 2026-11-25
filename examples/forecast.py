"""Explain the highest-scoring event of a seasonal stream by its forecast.

``ResidualDetector`` scores each value by its scaled error against a forecast
made before the value arrived. Here a Holt-Winters forecaster tracks a daily
cycle of request counts, and one value at a peak phase drops to a level that
is ordinary at other phases. Run with ``uv run python examples/forecast.py``.
"""

import numpy as np

from aberrant.forecast import HoltWinters, ResidualDetector

SEASON = 24
ANOMALY = 10 * SEASON + 6


def main() -> None:
    rng = np.random.default_rng(42)
    # The spread floor matches the simulated observation noise.
    detector = ResidualDetector(
        HoltWinters(season_length=SEASON, min_scale=0.5), target="requests"
    )
    peak: tuple[float, int, float, float, float] | None = None
    for index in range(12 * SEASON):
        value = 100 + 30 * np.sin(2 * np.pi * index / SEASON) + rng.normal(0, 0.5)
        if index == ANOMALY:
            value -= 30
        event = {"requests": float(value)}
        score, forecast, residual = detector.explain_one(event)
        if forecast is not None and residual is not None:
            candidate = (score, index, event["requests"], forecast, residual)
            peak = candidate if peak is None or score > peak[0] else peak
        detector.learn_one(event)

    assert peak is not None
    score, index, value, forecast, residual = peak
    print(f"Injected anomaly at event {ANOMALY}; highest score at event {index}")
    print(f"Observed {value:.1f} against a forecast of {forecast:.1f}")
    print(f"Residual {residual:.1f}, scaled error {score:.1f}")


if __name__ == "__main__":
    main()
