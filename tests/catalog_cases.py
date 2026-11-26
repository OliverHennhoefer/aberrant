"""Parameters and events that build and feed every declarative catalog entry."""

from collections.abc import Callable

from aberrant.catalog import MODEL_CATALOG, TRANSFORMER_CATALOG

MODEL_PARAMS: dict[str, dict[str, object]] = {
    "threshold": {"ceiling": 1.0},
    "residual_detector": {
        "target": "a",
        "model": {"id": "seasonal_naive", "params": {"season_length": 4}},
    },
    "multivariate_rolling_matrix_profile": {"subsequence_length": 4},
    "rolling_matrix_profile": {"subsequence_length": 4},
    "damp": {"subsequence_length": 4},
    "score_ensemble": {
        "models": [
            {"id": "loda", "params": {"warm_up": 10, "seed": 1}},
            {"id": "half_space_trees", "params": {"window_size": 10, "seed": 2}},
        ],
        "window_size": 20,
        "warm_up": 5,
    },
}
# Small warm-ups, so a short stream reaches every model's readiness.
QUICK_PARAMS: dict[str, dict[str, object]] = {
    "online_autoencoder_ensemble": {"feature_map_grace": 5, "ad_grace": 7},
    "knn": {"k": 3},
    "local_outlier_factor": {"k": 3},
    "cell_neighborhood_detector": {"k": 3, "slide_size": 4, "warm_up": 2},
    "sdostream": {"max_observers": 8, "n_neighbors": 3},
    "anoedge_l": {"warm_up": 5},
    "isconna": {"warm_up": 5},
    "midas": {"warm_up": 5},
    "streamspot": {"warm_up": 1},
    "asd_isolation_forest": {"max_samples": 16, "n_trees": 5},
    "half_space_trees": {"window_size": 10, "n_trees": 5},
    "mondrian_isolation_forest": {"n_trees": 5},
    "online_isolation_forest": {"n_trees": 5, "max_leaf_samples": 8},
    "random_cut_forest": {"n_trees": 5, "sample_size": 16},
    "rhf": {"n_trees": 5, "window_size": 16},
    "xstream": {"window_size": 16, "warm_up": 16, "n_chains": 4},
    "mstream": {"warm_up": 2},
    "loda": {"warm_up": 10},
    "rshash": {"warm_up": 10, "n_members": 4},
    "online_copod": {"warm_up": 10},
    "online_ecod": {"warm_up": 10},
}
TRANSFORMER_PARAMS: dict[str, dict[str, object]] = {
    "feature_schema_guard": {"features": ["a", "b"]},
    "select": {"features": ["a"]},
    "rename": {"mapping": {"a": "c"}},
    "incremental_pca": {"n_components": 1, "warm_up": 5},
    "random_projection": {"n_components": 1},
    "deseasonalize": {"season_length": 4},
}
_EDGE_MODELS = {"anoedge_l", "isconna", "midas", "streamspot"}
_MULTIVARIATE_STATISTICS = {
    "moving_correlation_coefficient",
    "moving_covariance",
    "moving_mahalanobis_distance",
}
_UNIVARIATE_MODELS = {
    "rolling_matrix_profile",
    "damp",
}


def event_factory(component_id: str) -> Callable[[int], dict[str, object]]:
    if component_id in _EDGE_MODELS:

        def edge(index: int) -> dict[str, object]:
            event: dict[str, object] = {
                "src": index % 5,
                "dst": index * 7 % 5,
                "t": index,
            }
            if component_id == "streamspot":
                event["graph"] = 1
            return event

        return edge
    univariate = component_id in _UNIVARIATE_MODELS or (
        component_id.startswith("moving_")
        and component_id not in _MULTIVARIATE_STATISTICS
    )
    if univariate:
        return lambda index: {"a": index % 7 + 0.1}
    return lambda index: {"a": index % 7 + 0.1, "b": index % 3 + 0.2}


def model_params(component_id: str) -> dict[str, object]:
    """Return small, valid parameters for one catalog model."""
    if component_id.startswith("moving_"):
        return {"window_size": 8}
    return dict(MODEL_PARAMS.get(component_id, {}))


def quick_params(component_id: str) -> dict[str, object]:
    """Return parameters whose warm-up a stream of about a hundred events ends."""
    return {**model_params(component_id), **QUICK_PARAMS.get(component_id, {})}


MODEL_IDS = [
    component_id
    for component_id, spec in MODEL_CATALOG.items()
    if spec.available and spec.declarative
]
TRANSFORMER_IDS = [
    component_id
    for component_id, spec in TRANSFORMER_CATALOG.items()
    if spec.available and spec.declarative
]
