"""Compare seeded detectors on identical ordered, unlabeled learning events.

Run with ``python -m examples.prequential_evaluation`` after installing
``aberrant[eval]``. Every evaluator builds a fresh detector from configuration.
"""

import numpy as np

from aberrant.catalog import ComponentConfig, DetectorConfig
from aberrant.evaluate import PrequentialEvaluator


def main() -> None:
    rng = np.random.default_rng(42)
    events: list[tuple[dict[str, float], int]] = []
    for index in range(600):
        anomaly = index >= 64 and index % 25 == 0
        # A gradual shift exercises adaptation without changing arrival order.
        values = rng.normal(loc=index / 600, size=2)
        if anomaly:
            values += 5.0
        events.append(({"x": float(values[0]), "y": float(values[1])}, int(anomaly)))

    for seed in (7, 19):
        config = DetectorConfig(
            model=ComponentConfig(
                "online_isolation_forest",
                {"num_trees": 10, "window_size": 128, "seed": seed},
            ),
            transformers=(ComponentConfig("standard_scaler"),),
        )
        evaluator = PrequentialEvaluator(
            config, warmup=64, metrics=True, metric_window_size=None
        )
        result = evaluator.evaluate(events)
        print(
            f"seed={seed} scored={result.n_scored} AP={result.average_precision:.3f} ROC_AUC={result.roc_auc:.3f} prevalence={result.prevalence:.3f}"
        )
        assert result.score_time_ns is not None and result.learn_time_ns is not None
        print(
            f"score_us={result.score_time_ns / result.n_scored / 1000:.1f} learn_us={result.learn_time_ns / result.n_learned / 1000:.1f} config={result.config_fingerprint}"
        )


if __name__ == "__main__":
    main()
