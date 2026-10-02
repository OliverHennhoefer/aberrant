"""Select and explain the highest-scoring event in a known seasonal stream."""

import numpy as np

from aberrant.model.timeseries import SeasonalResidualDetector

rng = np.random.default_rng(42)
season_length = 24
noise_scale = 0.5
# Set the residual-scale floor in requests to the simulated observation noise.
detector = SeasonalResidualDetector(
    season_length=season_length, key="requests", min_scale=noise_scale
)
normal_scores = []
peak: tuple[float, int, float, float, float] | None = None
anomaly_index = 10 * season_length + 6
for index in range(12 * season_length):
    value = 100 + 30 * np.sin(2 * np.pi * index / season_length)
    value += rng.normal(0.0, noise_scale)
    if index == anomaly_index:
        # A value near 100 is ordinary at other phases, but this phase expects 130.
        value -= 30
    event = {"requests": float(value)}
    score, forecast, residual = detector.explain_one(event)
    # Forecasts become available after two learned cycles. Rank every ready event.
    if forecast is not None and residual is not None:
        if peak is None or score > peak[0]:
            peak = (score, index, event["requests"], forecast, residual)
        # The known clean prefix supplies a comparison, independently of ranking.
        if index < anomaly_index:
            normal_scores.append(score)
    # Every observation advances one phase. Anomaly learning also adapts the model.
    detector.learn_one(event)

assert peak is not None  # This stream contains more than two complete cycles.
score, peak_index, value, forecast, residual = peak
print(f"Injected anomaly index: {anomaly_index}")
print(f"Highest-scoring event index: {peak_index}")
print(f"Observation: {value:.2f}; prior forecast: {forecast:.2f}")
print(f"Signed residual: {residual:.2f}; scaled error: {score:.2f}")
print(f"Earlier normal 99th-percentile score: {np.quantile(normal_scores, 0.99):.2f}")
