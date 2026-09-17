"""Score each event and locate its nearest preceding subsequence in bounded history."""

import numpy as np

from aberrant.model.timeseries import RollingMatrixProfile

rng = np.random.default_rng(42)
m = 24
period = np.sin(2.0 * np.pi * np.arange(m) / m)
series = np.tile(period, 40) + rng.normal(0.0, 0.01, 40 * m)
discord_start = 20 * m
series[discord_start : discord_start + m] = np.sign(
    np.sin(6.0 * np.pi * np.arange(m) / m)
) + np.linspace(-1.0, 1.0, m)

model = RollingMatrixProfile(subsequence_length=m, window_size=16 * m)
peak_score = 0.0
peak_start: int | None = None
peak_match: int | None = None

for index, value in enumerate(series):
    event = {"value": float(value)}
    # match_one returns the same distance as score_one, plus its reference index.
    score, match_start = model.match_one(event)
    if index >= 4 * m and score > peak_score:
        peak_score = score
        peak_start = index - m + 1
        peak_match = match_start
    model.learn_one(event)

print(f"Injected discord: [{discord_start}, {discord_start + m})")
print(f"Highest-scoring subsequence starts at {peak_start}; score={peak_score:.3f}")
print(f"Its nearest preceding match starts at {peak_match}")
print(f"Retained samples: {model.n_history} of {model.n_samples_seen} learned")
