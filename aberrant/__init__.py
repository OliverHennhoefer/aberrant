"""ABERRANT: Online Anomaly Detection library for streaming data.

A Python library implementing the online learning paradigm for anomaly
detection. Models process data one point at a time and update their state
incrementally. Memory strategies vary from fixed-size summaries to bounded
windows and periodic internal rebuilds.

Modules:
    base: Core abstract classes (BaseModel, BaseTransformer, Pipeline)
    drift: Concept drift detection algorithms (ADWIN, KSWIN, PageHinkley)
    model: Anomaly detection models
    stream: Streaming data utilities
    transform: Data transformers (scalers, projections)
"""

__version__ = "0.5.0"
