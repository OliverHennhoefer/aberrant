"""Continuous discord scoring through a declaratively constructed pipeline."""

import numpy as np

from aberrant.catalog import DetectorConfig, build_detector


def test_periodic_stream_with_a_novel_subsequence():
    rng = np.random.default_rng(42)
    m = 24
    period = np.sin(2.0 * np.pi * np.arange(m) / m)
    series = np.tile(period, 40) + rng.normal(0.0, 0.01, 40 * m)
    discord_start = 20 * m
    series[discord_start : discord_start + m] = np.sign(
        np.sin(6.0 * np.pi * np.arange(m) / m)
    ) + np.linspace(-1.0, 1.0, m)
    detector = build_detector(
        DetectorConfig.from_mapping(
            {
                "transformers": [
                    {"id": "feature_schema_guard", "params": {"features": ["value"]}}
                ],
                "model": {
                    "id": "rolling_matrix_profile",
                    "params": {"subsequence_length": m, "window_size": 16 * m},
                },
            }
        )
    )
    scores = []
    for value in series:
        event = {"value": float(value)}
        scores.append(detector.score_one(event))
        detector.learn_one(event)
    scores = np.asarray(scores)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0.0)
    peak = int(np.argmax(scores[4 * m :])) + 4 * m
    assert discord_start <= peak < discord_start + 2 * m - 1
    assert scores[peak] > 10 * np.quantile(scores[4 * m : discord_start], 0.99)
    # After the discord leaves retained history, ordinary motifs still score.
    assert np.max(scores[-2 * m :]) < scores[peak] / 10
