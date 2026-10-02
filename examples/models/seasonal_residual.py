"""Explain an unexpectedly low observation in a known seasonal stream."""

import numpy as np

from aberrant.model.timeseries import SeasonalResidualDetector

rng = np.random.default_rng(42)
season_length = 24
detector = SeasonalResidualDetector(
    season_length=season_length, key="requests", min_scale=0.5
)
normal_scores = []
anomaly_index = 10 * season_length + 6
for index in range(12 * season_length):
    value = 100 + 30 * np.sin(2 * np.pi * index / season_length)
    value += rng.normal(0.0, 0.5)
    if index == anomaly_index:
        # A value near 100 is ordinary at other phases, but this phase expects 130.
        value -= 30
    event = {"requests": float(value)}
    score, forecast, residual = detector.explain_one(event)
    if index == anomaly_index:
        print(f"Observation: {value:.2f}; prior forecast: {forecast:.2f}")
        print(f"Signed residual: {residual:.2f}; scaled error: {score:.2f}")
        print(
            f"Earlier normal 99th-percentile score: {np.quantile(normal_scores, 0.99):.2f}"
        )
    elif detector.is_ready and index < anomaly_index:
        normal_scores.append(score)
    # Every observation advances one phase. Anomaly learning also adapts the model.
    detector.learn_one(event)
