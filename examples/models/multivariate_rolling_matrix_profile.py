"""Detect a changed channel relationship and explain its nearest common match."""

import numpy as np

from aberrant.model.timeseries import (
    MultivariateRollingMatrixProfile,
    RollingMatrixProfile,
)

rng = np.random.default_rng(42)
m = 32
period = np.sin(2.0 * np.pi * np.arange(m) / m)
series = np.repeat(np.tile(period, 40)[:, None], 2, axis=1)
series += rng.normal(0.0, 0.002, series.shape)
change_start = 20 * m
series[change_start : change_start + m, 1] *= -1

params = {"subsequence_length": m, "window_size": 8 * m, "exclusion_zone": m - 1}
model = MultivariateRollingMatrixProfile(**params)
individual = {name: RollingMatrixProfile(**params) for name in ("sensor_a", "sensor_b")}

for index, values in enumerate(series):
    event = {"sensor_a": float(values[0]), "sensor_b": float(values[1])}
    # explain_one scores the candidate without learning, just like score_one.
    score, match_start, channel_distances = model.explain_one(event)
    if index == change_start + m - 1:
        scalar_scores = {
            name: detector.score_one({name: event[name]})
            for name, detector in individual.items()
        }
        print(f"Changed relationship: [{change_start}, {change_start + m})")
        print(f"Joint score: {score:.3f}; common reference starts at {match_start}")
        print(f"Distances at that reference: {channel_distances}")
        print(f"Independent scalar scores: {scalar_scores}")
    model.learn_one(event)
    for name, detector in individual.items():
        detector.learn_one({name: event[name]})

print(f"Retained observations: {model.n_history} of {model.n_samples_seen} learned")
