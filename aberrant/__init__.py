"""aberrant: online anomaly detection for streaming data.

Every component learns one event at a time and keeps bounded state.

Packages:
    base: base classes, protocols, ``Pipeline`` and ``Readiness``
    calibration: turn scores into alarm thresholds (``ScoreCalibrator``)
    catalog: build components from declarative configurations
    compose: branched graph pipelines
    datasets: labelled benchmark datasets and synthetic streams
    drift: concept-drift detectors (ADWIN, CUSUM, KSWIN, PageHinkley)
    evaluate: prequential evaluation and metrics
    exceptions: the errors aberrant raises
    forecast: forecasters and forecast-residual detection
    model: anomaly detection models, grouped by family
    transform: imputation, encoding, scaling, sequences and projections
"""

__version__ = "2.0.0"
