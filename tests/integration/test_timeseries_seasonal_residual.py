"""A declarative seasonal detector flags a value normal at other phases."""

import numpy as np

from aberrant.catalog import DetectorConfig


def test_fixed_cadence_seasonal_drop_is_explained_and_recovery_continues():
    period = 24
    anomaly_index = 10 * period + 6
    rng = np.random.default_rng(42)
    values = 100 + 30 * np.sin(2 * np.pi * np.arange(20 * period) / period)
    values += rng.normal(0.0, 0.5, len(values))
    values[anomaly_index] -= 30.0
    model = DetectorConfig.from_mapping(
        {
            "model": {
                "id": "seasonal_residual_detector",
                "params": {
                    "season_length": period,
                    "key": "requests",
                    "min_scale": 0.5,
                },
            }
        }
    ).build()
    scores = []
    for index, value in enumerate(values):
        event = {"requests": float(value)}
        scores.append(model.score_one(event))
        if index == anomaly_index:
            score, forecast, residual = model.explain_one(event)
            assert score == scores[-1]
            assert forecast is not None and residual is not None
            assert residual < -25.0
            assert np.isclose(value - forecast, residual)
        model.learn_one(event)
    scores = np.asarray(scores)
    assert np.all(np.isfinite(scores))
    assert np.all(scores[: 2 * period] == 0.0)
    assert int(np.argmax(scores)) == anomaly_index
    normal_upper = np.quantile(scores[2 * period : anomaly_index], 0.99)
    assert scores[anomaly_index] > 10 * normal_upper
    assert np.max(scores[-period:]) < scores[anomaly_index] / 10
