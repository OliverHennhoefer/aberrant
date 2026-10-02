"""Prequential anomaly evaluation with bounded, optional ranking metrics."""

from ._prequential import PrequentialEvaluator
from ._records import EvaluationRecord, EvaluationResult

__all__ = ["EvaluationRecord", "EvaluationResult", "PrequentialEvaluator"]
