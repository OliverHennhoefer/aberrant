# Evaluation API

Stateful anomaly evaluation preserves score-before-learn ordering and produces
immutable records and results. See the [evaluation guide](../user_guide/evaluation.md)
for warm-up, ranking metrics, retention, learning policies, and reproducibility.
Forecasters are evaluated the same way, forecast before learn, with scores
for every lead.

::: aberrant.evaluate.PrequentialEvaluator

::: aberrant.evaluate.EvaluationRecord

::: aberrant.evaluate.EvaluationResult

::: aberrant.evaluate.ForecastEvaluator

::: aberrant.evaluate.ForecastRecord

::: aberrant.evaluate.ForecastResult

::: aberrant.evaluate.LeadScores

::: aberrant.evaluate.vus

::: aberrant.evaluate.VolumeUnderSurface
