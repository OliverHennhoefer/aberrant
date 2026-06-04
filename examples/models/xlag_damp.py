import numpy as np

from aberrant.model.timeseries import XLagDAMP

rng = np.random.default_rng(42)
subsequence_length = 48
period = np.sin(2.0 * np.pi * np.arange(subsequence_length) / subsequence_length)
series = np.tile(period, 40).astype(np.float64)
series += rng.normal(0.0, 0.01, series.size)

discord_start = 20 * subsequence_length
discord = np.sign(
    np.sin(6.0 * np.pi * np.arange(subsequence_length) / subsequence_length)
) + np.linspace(-1.0, 1.0, subsequence_length)
series[discord_start : discord_start + subsequence_length] = discord

model = XLagDAMP(
    subsequence_length=subsequence_length,
    x_lag=16 * subsequence_length,
    start_index=4 * subsequence_length,
)

scores: list[float] = []
for value in series:
    sample = {"value": float(value)}
    scores.append(model.score_one(sample))
    model.learn_one(sample)

discord_end = int(np.argmax(scores))
detected_start = discord_end - subsequence_length + 1

print(
    f"Expected discord region: [{discord_start}, "
    f"{discord_start + subsequence_length})"
)
print(
    f"Detected discord subsequence: [{detected_start}, "
    f"{detected_start + subsequence_length})"
)
print(f"Top left-discord score: {scores[discord_end]:.3f}")
