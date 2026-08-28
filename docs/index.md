<p align="center">
  <img src="img/aberrant.png" alt="ABERRANT" width="760">
</p>

<p align="center">
  <strong>Online anomaly detection, one event at a time.</strong>
</p>

ABERRANT is a typed Python library for unsupervised anomaly detection on data
streams. A detector evaluates one event with `score_one(event)` and updates
itself with `learn_one(event)`, so applications can score and adapt without a
separate batch-retraining interface.

Most detectors accept a numeric feature mapping (`dict[str, float]`). Graph,
time-aware, and scalar time-series detectors define narrower event contracts in
the [model guide](user_guide/models.md).

!!! note "Development status"

    ABERRANT is pre-1.0 and under active development. The documented public
    imports are intentional, but APIs can still change between releases.

## Start here

<div class="grid cards" markdown>

-   **Install ABERRANT**

    Choose the base package or only the optional dependencies your detector
    requires.

    [Installation](installation.md)

-   **Run a complete example**

    Build a detector, warm it up, and score a synthetic stream without a
    download or optional dependency.

    [Quickstart](quickstart.md)

-   **Choose a detector**

    Compare inputs, state policy, warm-up, score range, and method provenance.

    [Model guide](user_guide/models.md)

-   **Use the exact API**

    Inspect generated signatures, parameters, return values, and public
    methods.

    [API reference](api/index.md)

</div>

## Complete minimal example

The following program uses only ABERRANT's base dependencies. It deliberately
scores each evaluated event before learning it.

```python
import numpy as np

from aberrant.model.iforest import OnlineIsolationForest
from aberrant.transform.preprocessing import StandardScaler

rng = np.random.default_rng(42)
values = np.vstack(
    [
        rng.normal(size=(300, 2)),
        rng.normal(loc=5.0, size=(12, 2)),
    ]
)

detector = StandardScaler() | OnlineIsolationForest(
    num_trees=25,
    window_size=256,
    seed=42,
)

ranked_scores: list[tuple[int, float]] = []
for index, row in enumerate(values):
    event = {"x": float(row[0]), "y": float(row[1])}

    if index >= 64:
        ranked_scores.append((index, detector.score_one(event)))

    detector.learn_one(event)

for index, score in sorted(
    ranked_scores,
    key=lambda item: item[1],
    reverse=True,
)[:5]:
    print(f"event={index}, anomaly_score={score:.3f}")
```

`score_one` does not learn the candidate or call transformer learning methods;
some models may still advance ancillary state such as a model-local random
generator or bounded lookup cache. During `Pipeline.learn_one`, every
transformer learns first and passes its **post-update** transform to the next
stage. The
[pipeline lifecycle](user_guide/pipelines.md) explains why that distinction
matters.

## What is included

- Isolation forests, distance and density methods, bounded sketches, dynamic
  graph detectors, online statistics, experimental SVMs, scalar time-series
  discord detection, and reconstruction models.
- Incremental scaling and projection transformers that compose with `|`.
- ADWIN, KSWIN, and Page-Hinkley drift detectors for scalar monitoring signals.
- A registry-backed, checksum-validated benchmark dataset cache.
- Runtime-checkable component protocols and `py.typed` metadata for downstream
  type checking.

Higher values normally indicate greater anomaly evidence, but score range,
warm-up, and calibration are detector-specific. Consult the
[score contracts](user_guide/models.md#score-contracts) before comparing or
thresholding outputs.
