# User Guide

The user guide documents ABERRANT's behavioral contracts: what an event must
contain, when state changes, what a score means, how long a detector needs to
warm up, and which state-growth policy it uses. Constructor signatures and
individual method parameters live in the [API reference](../api/index.md).

## The online lifecycle

For an event `x`, a model exposes two separate operations:

- `score_one(x) -> float` evaluates `x` without learning it.
- `learn_one(x) -> None` updates learned state and produces no score.

Keeping these operations separate lets the caller choose evaluation semantics.
It does not impose total functional purity: a concrete scorer may advance
ancillary state such as a model-local random generator or bounded query cache.
For test-then-train processing, call `score_one` first and `learn_one` second.
Calling them in the opposite order asks a different question: how unusual is
the event after the model has already incorporated it?

Pipelines preserve the same read/write separation, with one deliberate detail:
during `Pipeline.learn_one`, every transformer learns first and sends its
post-update transform downstream. See [Pipelines](pipelines.md).

## Input boundaries

| Component | Expected input |
| --- | --- |
| General multivariate detector | A non-empty `dict[str, float]` with finite numeric values and, for schema-owning models, the same feature-key set after the first successful `learn_one` |
| Univariate moving statistic and `XLagDAMP` | Exactly one consistently named numeric feature |
| Bivariate covariance/correlation statistic | Exactly two consistently named numeric features |
| Edge-stream detector | Named source and destination fields, usually integer-like, plus the configured timestamp field when `time_key` is not `None` |
| Drift detector | One finite scalar passed to `update(value)` |

Schema-owning models canonicalize feature order; callers need a stable **key
set**, not a stable dictionary insertion order. The scalers instead maintain
independent state per observed key, while `IncrementalPCA` and
`RandomProjection` preserve explicit `keys=` order or the first learned event's
order. The [transformer guide](transformers.md) documents those differences.

## Score contracts

An anomaly score is a detector-specific ranking signal. It is not generally a
probability, a p-value, or a portable quantity across models.

- Higher scores indicate greater anomaly evidence for detector defaults.
- Some models offer optional normalization to `[0, 1)` or `[0, 1]`; that
  transform does not make the result a calibrated probability.
- `RandomCutForest` returns raw, unbounded CoDisp by default.
- LOF-like scores use a baseline near `1`, while several detectors return `0`
  during explicit warm-up.
- Moving-statistic models return a statistic change. With `abs_diff=False`, the
  sign records direction and the result is no longer a simple higher-is-more-
  anomalous magnitude.
- `NullModel` and `RandomModel` are baselines, not meaningful anomaly scorers.

Use the per-class table in [Models](models.md#score-contracts) before setting a
threshold or comparing results.

## State and warm-up are part of the model

“Online” means a model accepts incremental updates; it does not imply bounded
memory. ABERRANT includes several state policies:

- fixed sliding windows and explicit unlearning;
- periodic replacement from a recent window;
- fixed-size sketches or observer sets;
- exponentially faded summaries;
- growing trees or parameter state without automatic forgetting.

Likewise, there is no package-wide warm-up value. Some classes expose an exact
warm-up parameter, some become useful only after a complete reference window,
and some can return an uncalibrated score immediately. Make the evaluation and
alerting warm-up explicit in application code.

## Guide map

<div class="grid cards" markdown>

-   **[Concepts and terminology](concepts.md)**

    Precise definitions for event, score, alert, anomaly, drift, event time,
    schema, warm-up, bounded state, and prequential evaluation.

-   **[Models](models.md)**

    Selection criteria, input contracts, score scales, state policies, and
    provenance.

-   **[Transformers](transformers.md)** and **[Pipelines](pipelines.md)**

    Incremental preprocessing, projection, structural protocols, and update
    order.

-   **[Streaming datasets](streaming.md)**

    Registered artifacts, local NPZ streams, batching, validation, and cache
    behavior.

-   **[Drift detection](drift.md)** and **[Evaluation](evaluation.md)**

    Scalar change monitoring and leakage-resistant stream evaluation.

-   **[Best practices](best_practices.md)**

    Production-oriented schema, time, reproducibility, concurrency, threshold,
    and persistence guidance.

</div>
