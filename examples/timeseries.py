"""Matrix-profile detectors find an injected discord in a periodic series.

Each detector scores a value before it learns it. A subsequence detector
scores the subsequence that ends at the current value, so the discord shows
once its last value arrives. The first cycles have little history to match,
so the example ranks scores from the fifth cycle on. Run with
``uv run python examples/timeseries.py``.
"""

import numpy as np

from aberrant.model.timeseries import (
    DAMP,
    MultivariateRollingMatrixProfile,
    RollingMatrixProfile,
)

LENGTH = 48
DISCORD = 20 * LENGTH
BURN_IN = 4 * LENGTH


def _series() -> np.ndarray:
    rng = np.random.default_rng(42)
    period = np.sin(2.0 * np.pi * np.arange(LENGTH) / LENGTH)
    series = np.tile(period, 40) + rng.normal(0.0, 0.01, 40 * LENGTH)
    series[DISCORD : DISCORD + LENGTH] = np.sign(
        np.sin(6.0 * np.pi * np.arange(LENGTH) / LENGTH)
    ) + np.linspace(-1.0, 1.0, LENGTH)
    return series


def main() -> None:
    series = _series()
    univariate = [{"value": float(value)} for value in series]
    # The second channel is a shifted copy, so both channels carry the discord.
    bivariate = [{"value": float(v), "shifted": float(v) + 1.0} for v in series]
    detectors = {
        "DAMP": (DAMP(subsequence_length=LENGTH, x_lag=16 * LENGTH), univariate),
        "RollingMatrixProfile": (
            RollingMatrixProfile(subsequence_length=LENGTH, window_size=16 * LENGTH),
            univariate,
        ),
        "MultivariateRollingMatrixProfile": (
            MultivariateRollingMatrixProfile(
                subsequence_length=LENGTH, window_size=16 * LENGTH
            ),
            bivariate,
        ),
    }
    print(f"Injected discord: values {DISCORD} to {DISCORD + LENGTH - 1}")
    for name, (detector, events) in detectors.items():
        scores = []
        for event in events:
            scores.append(detector.score_one(event))
            detector.learn_one(event)
        end = BURN_IN + int(np.argmax(scores[BURN_IN:]))
        print(f"{name:<33} highest score: values {end - LENGTH + 1} to {end}")


if __name__ == "__main__":
    main()
