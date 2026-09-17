"""A changing channel relationship that independent scalar profiles miss."""

import numpy as np

from aberrant.catalog import DetectorConfig, build_detector
from aberrant.model.timeseries import (
    MultivariateRollingMatrixProfile,
    RollingMatrixProfile,
)


def test_phase_change_with_familiar_individual_channels():
    rng = np.random.default_rng(42)
    m = 32
    period = np.sin(2 * np.pi * np.arange(m) / m)
    series = np.repeat(np.tile(period, 40)[:, None], 2, axis=1)
    series += rng.normal(0, 0.002, series.shape)
    change = 20 * m
    series[change : change + m, 1] *= -1
    params = {"subsequence_length": m, "window_size": 8 * m, "exclusion_zone": m - 1}
    detector = build_detector(
        DetectorConfig.from_mapping(
            {
                "transformers": [
                    {"id": "feature_schema_guard", "params": {"features": ["a", "b"]}}
                ],
                "model": {
                    "id": "multivariate_rolling_matrix_profile",
                    "params": params,
                },
            }
        )
    )
    direct = MultivariateRollingMatrixProfile(**params)
    independent = [RollingMatrixProfile(**params), RollingMatrixProfile(**params)]
    scores = []
    individual_scores = []
    for index, values in enumerate(series):
        event = {"a": float(values[0]), "b": float(values[1])}
        score, start, channels = direct.explain_one(event)
        assert detector.score_one(event) == score
        if direct.is_ready:
            assert (
                max(0, index - params["window_size"] + 1) <= start <= index - 2 * m + 1
            )
            assert score == max(channels.values())
        else:
            assert (score, start, channels) == (0.0, None, {})
        scores.append(score)
        individual_scores.append(
            [
                model.score_one({"value": float(value)})
                for model, value in zip(independent, values, strict=True)
            ]
        )
        detector.learn_one(event)
        direct.learn_one(event)
        for model, value in zip(independent, values, strict=True):
            model.learn_one({"value": float(value)})

    scores = np.asarray(scores)
    assert np.all(np.isfinite(scores)) and np.all(scores >= 0)
    # The whole flipped period is individually familiar in both channels, but
    # no earlier interval contains that combination of phases.
    novel = change + m - 1
    assert scores[novel] > 7.5
    assert max(individual_scores[novel]) < 0.1
    assert scores[novel] > 30 * np.quantile(scores[4 * m : change], 0.99)
    assert np.max(scores[-2 * m :]) < 0.1
    assert direct.is_ready and direct.n_history == params["window_size"]
