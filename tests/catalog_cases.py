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
}
TRANSFORMER_PARAMS: dict[str, dict[str, object]] = {
    "feature_schema_guard": {"features": ["a", "b"]},
    "select": {"features": ["a"]},
    "rename": {"mapping": {"a": "c"}},
    "incremental_pca": {"n_components": 1, "warm_up": 5},
    "random_projection": {"n_components": 1},
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


MODEL_IDS = [
    component_id
    for component_id, spec in MODEL_CATALOG.items()
    if spec.available and spec.declarative and component_id != "knn"
]
TRANSFORMER_IDS = [
    component_id
    for component_id, spec in TRANSFORMER_CATALOG.items()
    if spec.available and spec.declarative
]
