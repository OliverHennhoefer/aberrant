# Models API

Model documentation is split by event structure and method family.

<div class="grid cards" markdown>

-   **[Core utility models](core.md)**

    A fixed-bound threshold and null/random baselines.

-   **[Forests](forest.md)**

    Incremental, window-replaced, half-space, Mondrian, random-cut, histogram,
    and xStream variants.

-   **[Distance and neighborhood](distance.md)**

    KNN, LOF, SDOStream and a cell-neighborhood detector.

-   **[Sketch detectors](sketch.md)**

    MStream, LODA and RS-Hash.

-   **[Graph streams](graph.md)**

    AnoEdge-L, ISCONNA, MIDAS and StreamSpot.

-   **[Time series](timeseries.md)**

    Exact rolling matrix-profile scores, univariate and multivariate, and
    DAMP.

-   **[Statistical models](stat.md)**

    Moving z-score and MAD, moving covariance and correlation, squared
    Mahalanobis distance, and online ECOD and COPOD.

-   **[SVM models](svm.md)**

    A budgeted one-class SVM with an adaptive kernel width.

-   **[Reconstruction models](deep.md)**

    NumPy online autoencoder ensemble and optional user-supplied PyTorch
    autoencoder.

</div>

All model classes expose `learn_one` and `score_one`, but their event
schemas, readiness conditions, state bounds, and score scales differ. Start
with the [model guide](../../user_guide/models.md) when selecting a class.
