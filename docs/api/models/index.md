# Models API

Model documentation is split by event structure and method family.

<div class="grid cards" markdown>

-   **[Core utility models](core.md)**

    Static/adaptive thresholds and null/random baselines.

-   **[Isolation forests](iforest.md)**

    Incremental, window-replaced, half-space, Mondrian, random-cut, histogram,
    and xStream variants.

-   **[Distance and neighborhood](distance.md)**

    KNN, LOF, observer-distance, and point-scoring cell adaptations.

-   **[Sketch detectors](sketch.md)**

    MStream, streaming LODA, and streaming RS-Hash adaptations.

-   **[Graph streams](graph.md)**

    AnoEdge-L, ISCONNA, MIDAS, and signed graph sketches.

-   **[Time series](timeseries.md)**

    Pure-online X-Lag Amnesic DAMP.

-   **[Statistical models](stat.md)**

    Candidate-induced changes in moving univariate/bivariate statistics and
    squared Mahalanobis distance.

-   **[SVM models](svm.md)**

    Experimental budgeted and graph-gated one-class heuristics.

-   **[Reconstruction models](deep.md)**

    NumPy online autoencoder ensemble and optional user-supplied PyTorch
    autoencoder.

</div>

All model classes expose `learn_one` and `score_one`, but their event
schemas, readiness conditions, state bounds, and score scales differ. Start
with the [model guide](../../user_guide/models.md) when selecting a class.
