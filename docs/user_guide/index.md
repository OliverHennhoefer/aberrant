# User Guide

This guide explains ABERRANT design conventions and practical streaming usage.

## Streaming contract

- Input is a `dict[str, float]`.
- Call `learn_one(x)` to update state.
- Call `score_one(x)` to obtain anomaly score.
- Keep a fixed key set per model instance.

## Score semantics by model family

- `aberrant.model.ThresholdModel`: binary score (`0.0` normal, `1.0` anomaly).
- `aberrant.model.iforest.*`: bounded score in `[0, 1]` (higher means more anomalous).
- `aberrant.model.distance.LocalOutlierFactor`: unbounded positive LOF-like score (`~1` normal, higher outlierness).
- `aberrant.model.timeseries.XLagDAMP`: z-normalized left-discord distance;
  highest peaks are discord candidates.
- `aberrant.model.svm.*`: margin-style continuous score (scale is model-specific).
- `aberrant.model.stat.*`: delta/statistic scores, usually unbounded.

## Warmup behavior

Most models return conservative values (`0.0` or near-baseline) before enough history is observed.
Always set warmup rules in your application logic.

## Recommended workflow

1. Pick one detector family based on latency and interpretability needs.
2. Add preprocessing (`MinMaxScaler` or `StandardScaler`) if feature scales differ.
3. Define warmup and thresholding policy explicitly.
4. Track drift and recalibrate thresholds over time.
