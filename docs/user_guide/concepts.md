# Concepts and Terminology

This page fixes the vocabulary used throughout ABERRANT. The distinctions are
operational: they describe what the library does and prevent similar terms
from silently acquiring different meanings across model families.

## Event, sample, and observation

**Event**
:   One item presented to an anomaly model or transformer, normally as a
    `dict[str, float]`. The guide prefers *event* when discussing application
    streams.

**Sample**
:   A single data point. Model descriptions often use *sample* when matching
    statistical or paper terminology. In the online API, one sample is one
    event.

**Observation**
:   The finite scalar passed to a drift detector's `update` method. It can be an
    anomaly score, prediction loss, residual, feature value, or another scalar
    monitoring signal.

## Anomaly, outlier, score, and alert

**Anomaly** or **outlier**
:   An event judged unusual relative to a reference distribution, neighborhood,
    learned representation, rule, or temporal pattern. ABERRANT uses these
    words descriptively; an unsupervised score alone does not establish that an
    event is erroneous, harmful, or semantically abnormal.

**Anomaly score**
:   A model-specific numeric ranking signal returned by `score_one`. Higher
    values normally represent stronger anomaly evidence. Numeric scale and
    baseline are not standardized across model families.

**Prediction**
:   A discrete decision produced by methods such as `predict_one`, where a
    class exposes one, or by an external threshold policy. A score is not itself
    a binary prediction.

**Alert**
:   An application action triggered from a score or prediction. Alert policy
    includes thresholds, suppression, grouping, rate limits, and business cost;
    it is intentionally outside most ABERRANT detectors.

## Learning and evaluation

**Online or incremental learning**
:   Updating state from one event at a time. This describes the update interface,
    not the amount of retained state.

**Prequential evaluation**
:   Also called test-then-train evaluation. Event *t* is first evaluated using
    the model learned through *t - 1*, then used for learning. This preserves
    stream chronology and avoids training on the event whose score is being
    measured. The terminology follows Gama, Sebastião, and Rodrigues,
    [*Issues in Evaluation of Stream Learning Algorithms*](https://doi.org/10.1145/1557019.1557060).

**Warm-up**
:   An initial period in which state is being established and scores are omitted
    from evaluation or alerting. A model may enforce warm-up by returning
    `0.0`, but application-defined warm-up can be longer. Zero is therefore not
    always evidence of normality.

**Candidate-inclusive score**
:   A score whose mathematical definition includes a provisional insertion of
    the query event. ABERRANT implementations that need this behavior preview
    the insertion without committing it to learned reference state.

## Schema and time

**Feature schema**
:   The feature-name set and the deterministic order used to vectorize it. Most
    schema-owning models establish a sorted order on the first successful
    `learn_one` and reject missing or additional features afterward. A
    dictionary's insertion order does not need to match that canonical order.

**Arrival order**
:   The sequence in which the process receives events. When a time-aware model
    has `time_key=None`, ABERRANT uses a one-based arrival index as its implicit
    clock.

**Event time**
:   A timestamp carried by the event under the configured `time_key`. ABERRANT's
    shared clock accepts non-decreasing values, so equal timestamps are allowed
    and a smaller later value is rejected. Models that use integer time buckets
    require integer-like timestamps.

**Preview and commit**
:   Validation boundaries first preview schema and time without mutation. A
    successful `learn_one` commits them; `score_one` does not. A failed update
    therefore cannot partially establish the shared schema or advance its
    clock.

## State and change

**Bounded state**
:   State whose configured maximum does not grow with stream length. A sliding
    window, fixed-size sketch, or fixed observer set can be bounded. A growing
    online tree is incremental but not necessarily bounded.

**Forgetting**
:   Reducing the influence of older events through eviction, replacement,
    exponential fading, or explicit unlearning. The exact mechanism changes the
    model's effective reference distribution.

**Concept drift**
:   A change over time in the data-generating relationship or distribution.
    In supervised learning this is often formalized as a change in
    `P_t(X, Y)`; without labels, monitoring can only expose changes in observed
    features, scores, residuals, or other proxies. See Gama et al.,
    [*A Survey on Concept Drift Adaptation*](https://doi.org/10.1145/2523813).

**Drift detection**
:   A statistical signal that a monitored scalar has changed. It does not
    diagnose the cause and does not automatically decide whether to reset a
    model, recalibrate a threshold, retrain, or investigate upstream data.

!!! warning "An anomaly is not a drift, and a drift is not an anomaly"

    An anomaly is local evidence about one event. Drift is evidence of a change
    across observations. A burst of anomalies can cause a drift signal, but the
    two outputs answer different questions and require separate response
    policies.
