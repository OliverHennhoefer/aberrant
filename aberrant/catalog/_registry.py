"""Built-in component catalog and lookup functions."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import TypeVar, cast

from aberrant.base.readiness import ReadinessUnit
from aberrant.exceptions import UnknownComponentError

from ._specs import (
    ComponentDependency,
    ComponentKind,
    ComponentSpec,
    EventKind,
    FeatureCount,
    FeatureSchemaKind,
    ModelCapabilities,
    ModelSpec,
    ScoreKind,
    StateKind,
    TransformerSpec,
)

Parameters = Mapping[str, object]
FeatureCountValue = FeatureCount | Callable[[Parameters], FeatureCount]
FeatureSchemaValue = FeatureSchemaKind | Callable[[Parameters], FeatureSchemaKind]
ScoreKindValue = ScoreKind | Callable[[Parameters], ScoreKind]
StateKindValue = StateKind | Callable[[Parameters], StateKind]
EventKindValue = EventKind | Callable[[Parameters], EventKind]

ANY_FEATURES = FeatureCount()
ONE_FEATURE = FeatureCount(1, 1)
TWO_FEATURES = FeatureCount(2, 2)
CATALOG_FORMAT_VERSION = 2
"""Format version of `catalog_manifest()`."""

_Spec = TypeVar("_Spec", bound=ComponentSpec)


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise TypeError("expected an integer-compatible JSON value")
    return int(value)


def _resolve_value(value: object, params: Parameters) -> object:
    return value(params) if callable(value) else value


def _capabilities(
    *,
    event_kind: EventKindValue,
    feature_count: FeatureCountValue = ANY_FEATURES,
    feature_schema: FeatureSchemaValue = FeatureSchemaKind.FIXED,
    score_kind: ScoreKindValue,
    state: StateKindValue,
    warmup_unit: ReadinessUnit = ReadinessUnit.EVENTS,
    requires_unit_interval: bool = False,
) -> Callable[[Parameters], ModelCapabilities]:
    """Declare what a model accepts and keeps; its warm-up comes from an instance."""

    def resolve(params: Parameters) -> ModelCapabilities:
        resolved_feature_count = _resolve_value(feature_count, params)
        resolved_feature_schema = _resolve_value(feature_schema, params)
        resolved_score_kind = _resolve_value(score_kind, params)
        resolved_state = _resolve_value(state, params)
        resolved_event_kind = _resolve_value(event_kind, params)
        assert isinstance(resolved_event_kind, EventKind)
        assert isinstance(resolved_feature_count, FeatureCount)
        assert isinstance(resolved_feature_schema, FeatureSchemaKind)
        assert isinstance(resolved_score_kind, ScoreKind)
        assert isinstance(resolved_state, StateKind)
        return ModelCapabilities(
            event_kind=resolved_event_kind,
            feature_count=resolved_feature_count,
            feature_schema=resolved_feature_schema,
            score_kind=resolved_score_kind,
            warmup=None,
            warmup_unit=warmup_unit,
            state=resolved_state,
            requires_unit_interval=requires_unit_interval,
        )

    return resolve


def _mondrian_state(params: Parameters) -> StateKind:
    return (
        StateKind.BOUNDED
        if params.get("window_size") is not None
        else StateKind.GROWING
    )


def _time_aware_features(params: Parameters) -> FeatureCount:
    return FeatureCount(2 if params.get("time_key") is not None else 1)


def _edge_features(params: Parameters) -> FeatureCount:
    count = 2 + int(params.get("time_key") is not None)
    return FeatureCount(count)


def _graph_edge_features(params: Parameters) -> FeatureCount:
    count = 3
    count += int(params.get("edge_type_key") is not None)
    count += int(params.get("time_key") is not None)
    return FeatureCount(count)


# Forecasters that predict the target from the event's other features at lead 0
# instead of forecasting the series ahead.
_REGRESSORS = frozenset(
    {
        "rls_regressor",
        "random_fourier_regressor",
        "passive_aggressive_regressor",
        "sgd_regressor",
        "knn_regressor",
        "hoeffding_tree_regressor",
        "adaptive_random_forest_regressor",
    }
)


def _wraps_regressor(params: Parameters) -> bool:
    """Whether the residual detector wraps a regressor rather than a forecaster.

    The model is a nested ``{"id", "params"}`` configuration, whose parameters
    are checked here, or an instance created directly from Python objects.
    """
    model = params.get("model")
    if isinstance(model, Mapping):
        component_id = model.get("id")
        if not isinstance(component_id, str):
            raise TypeError("model needs a forecaster id")
        nested = model.get("params") or {}
        if not isinstance(nested, Mapping):
            raise TypeError("model params must be a mapping")
        get_forecaster_spec(component_id).resolved_parameters(nested)
        return component_id in _REGRESSORS
    return hasattr(model, "predict_one")


def _residual_event_kind(params: Parameters) -> EventKind:
    return EventKind.TABULAR if _wraps_regressor(params) else EventKind.UNIVARIATE


def _residual_features(params: Parameters) -> FeatureCount:
    return FeatureCount(2) if _wraps_regressor(params) else ONE_FEATURE


def _xstream_state(params: Parameters) -> StateKind:
    return (
        StateKind.GROWING
        if params.get("max_feature_cache_size") is None
        else StateKind.BOUNDED
    )


def _moving_capabilities(
    feature_count: FeatureCount,
) -> Callable[[Parameters], ModelCapabilities]:
    return _capabilities(
        event_kind=(
            EventKind.UNIVARIATE
            if feature_count == ONE_FEATURE
            else EventKind.BIVARIATE
        ),
        feature_count=feature_count,
        score_kind=ScoreKind.NON_NEGATIVE,
        state=StateKind.BOUNDED,
    )


def _model(
    component_id: str,
    class_name: str,
    module: str,
    family: str,
    capabilities: Callable[[Parameters], ModelCapabilities],
    *,
    optional_extra: str | None = None,
    dependency_module: str | None = None,
    declarative: bool = True,
    dependencies: tuple[ComponentDependency, ...] = (),
) -> ModelSpec:
    return ModelSpec(
        id=component_id,
        display_name=class_name,
        kind=ComponentKind.MODEL,
        family=family,
        import_path=f"{module}.{class_name}",
        optional_extra=optional_extra,
        dependency_module=dependency_module,
        declarative=declarative,
        dependencies=dependencies,
        _capability_resolver=capabilities,
    )


_MODEL_SPECS = (
    _model(
        "null",
        "NullModel",
        "aberrant.model",
        "baseline",
        _capabilities(
            event_kind=EventKind.TABULAR,
            feature_schema=FeatureSchemaKind.UNRESTRICTED,
            score_kind=ScoreKind.ZERO,
            state=StateKind.STATELESS,
        ),
    ),
    _model(
        "random",
        "RandomModel",
        "aberrant.model",
        "baseline",
        _capabilities(
            event_kind=EventKind.TABULAR,
            feature_schema=FeatureSchemaKind.UNRESTRICTED,
            score_kind=ScoreKind.ZERO_TO_ONE,
            state=StateKind.BOUNDED,
        ),
    ),
    _model(
        "threshold",
        "ThresholdModel",
        "aberrant.model",
        "score_policy",
        _capabilities(
            event_kind=EventKind.TABULAR,
            feature_schema=FeatureSchemaKind.UNRESTRICTED,
            score_kind=ScoreKind.BINARY,
            state=StateKind.STATELESS,
        ),
    ),
    _model(
        "online_autoencoder_ensemble",
        "OnlineAutoencoderEnsemble",
        "aberrant.model.deep",
        "reconstruction",
        _capabilities(
            event_kind=EventKind.TABULAR,
            score_kind=ScoreKind.NON_NEGATIVE,
            state=StateKind.BOUNDED,
        ),
    ),
    _model(
        "autoencoder",
        "Autoencoder",
        "aberrant.model.deep",
        "reconstruction",
        _capabilities(
            event_kind=EventKind.CUSTOM,
            score_kind=ScoreKind.NON_NEGATIVE,
            state=StateKind.EXTERNAL,
        ),
        optional_extra="dl",
        dependency_module="torch",
        declarative=False,
    ),
    _model(
        "knn",
        "KNN",
        "aberrant.model.distance",
        "distance",
        _capabilities(
            event_kind=EventKind.TABULAR,
            score_kind=ScoreKind.NON_NEGATIVE,
            state=StateKind.BOUNDED,
        ),
    ),
    _model(
        "local_outlier_factor",
        "LocalOutlierFactor",
        "aberrant.model.distance",
        "distance",
        _capabilities(
            event_kind=EventKind.TABULAR,
            score_kind=ScoreKind.NON_NEGATIVE,
            state=StateKind.BOUNDED,
        ),
    ),
    _model(
        "cell_neighborhood_detector",
        "CellNeighborhoodDetector",
        "aberrant.model.distance",
        "distance",
        _capabilities(
            event_kind=EventKind.TABULAR,
            feature_count=_time_aware_features,
            score_kind=ScoreKind.ZERO_TO_ONE,
            state=StateKind.BOUNDED,
        ),
    ),
    _model(
        "sdostream",
        "SDOStream",
        "aberrant.model.distance",
        "distance",
        _capabilities(
            event_kind=EventKind.TABULAR,
            feature_count=_time_aware_features,
            score_kind=ScoreKind.NON_NEGATIVE,
            state=StateKind.BOUNDED,
            warmup_unit=ReadinessUnit.OBSERVERS,
        ),
    ),
    _model(
        "anoedge_l",
        "AnoEdgeL",
        "aberrant.model.graph",
        "graph",
        _capabilities(
            event_kind=EventKind.EDGE,
            feature_count=_edge_features,
            score_kind=ScoreKind.NON_NEGATIVE,
            state=StateKind.BOUNDED,
        ),
    ),
    _model(
        "isconna",
        "ISCONNA",
        "aberrant.model.graph",
        "graph",
        _capabilities(
            event_kind=EventKind.EDGE,
            feature_count=_edge_features,
            score_kind=ScoreKind.NON_NEGATIVE,
            state=StateKind.BOUNDED,
        ),
    ),
    _model(
        "midas",
        "MIDAS",
        "aberrant.model.graph",
        "graph",
        _capabilities(
            event_kind=EventKind.EDGE,
            feature_count=_edge_features,
            score_kind=ScoreKind.NON_NEGATIVE,
            state=StateKind.BOUNDED,
        ),
    ),
    _model(
        "streamspot",
        "StreamSpot",
        "aberrant.model.graph",
        "graph",
        _capabilities(
            event_kind=EventKind.GRAPH_EDGE,
            feature_count=_graph_edge_features,
            score_kind=ScoreKind.NON_NEGATIVE,
            state=StateKind.BOUNDED,
            warmup_unit=ReadinessUnit.ACTIVE_GRAPHS,
        ),
    ),
    _model(
        "asd_isolation_forest",
        "ASDIsolationForest",
        "aberrant.model.forest",
        "isolation_forest",
        _capabilities(
            event_kind=EventKind.TABULAR,
            score_kind=ScoreKind.ZERO_TO_ONE,
            state=StateKind.BOUNDED,
        ),
    ),
    _model(
        "half_space_trees",
        "HalfSpaceTrees",
        "aberrant.model.forest",
        "isolation_forest",
        _capabilities(
            event_kind=EventKind.TABULAR,
            score_kind=ScoreKind.ZERO_TO_ONE,
            state=StateKind.BOUNDED,
            requires_unit_interval=True,
        ),
    ),
    _model(
        "mondrian_isolation_forest",
        "MondrianIsolationForest",
        "aberrant.model.forest",
        "isolation_forest",
        _capabilities(
            event_kind=EventKind.TABULAR,
            score_kind=ScoreKind.ZERO_TO_ONE,
            state=_mondrian_state,
        ),
    ),
    _model(
        "online_isolation_forest",
        "OnlineIsolationForest",
        "aberrant.model.forest",
        "isolation_forest",
        _capabilities(
            event_kind=EventKind.TABULAR,
            score_kind=ScoreKind.ZERO_TO_ONE,
            state=StateKind.BOUNDED,
        ),
    ),
    _model(
        "random_cut_forest",
        "RandomCutForest",
        "aberrant.model.forest",
        "isolation_forest",
        _capabilities(
            event_kind=EventKind.TABULAR,
            score_kind=ScoreKind.NON_NEGATIVE,
            state=StateKind.BOUNDED,
        ),
    ),
    _model(
        "rhf",
        "RHF",
        "aberrant.model.forest",
        "isolation_forest",
        _capabilities(
            event_kind=EventKind.TABULAR,
            score_kind=ScoreKind.NON_NEGATIVE,
            state=StateKind.BOUNDED,
        ),
    ),
    _model(
        "xstream",
        "XStream",
        "aberrant.model.forest",
        "isolation_forest",
        _capabilities(
            event_kind=EventKind.TABULAR,
            feature_schema=FeatureSchemaKind.EVOLVING,
            score_kind=ScoreKind.ZERO_TO_ONE,
            state=_xstream_state,
        ),
    ),
    _model(
        "mstream",
        "MStream",
        "aberrant.model.sketch",
        "sketch",
        _capabilities(
            event_kind=EventKind.TABULAR,
            feature_count=_time_aware_features,
            score_kind=ScoreKind.NON_NEGATIVE,
            state=StateKind.BOUNDED,
            warmup_unit=ReadinessUnit.BUCKETS,
        ),
    ),
    _model(
        "loda",
        "LODA",
        "aberrant.model.sketch",
        "sketch",
        _capabilities(
            event_kind=EventKind.TABULAR,
            feature_count=_time_aware_features,
            score_kind=ScoreKind.NON_NEGATIVE,
            state=StateKind.BOUNDED,
        ),
    ),
    _model(
        "rshash",
        "RSHash",
        "aberrant.model.sketch",
        "sketch",
        _capabilities(
            event_kind=EventKind.TABULAR,
            feature_count=_time_aware_features,
            score_kind=ScoreKind.ZERO_TO_ONE,
            state=StateKind.BOUNDED,
        ),
    ),
    *(
        _model(
            component_id,
            class_name,
            "aberrant.model.stat",
            "statistical",
            _moving_capabilities(ONE_FEATURE),
        )
        for component_id, class_name in (
            ("moving_zscore", "MovingZScore"),
            ("moving_mad", "MovingMAD"),
        )
    ),
    _model(
        "moving_correlation_coefficient",
        "MovingCorrelationCoefficient",
        "aberrant.model.stat",
        "statistical",
        _moving_capabilities(TWO_FEATURES),
    ),
    _model(
        "moving_covariance",
        "MovingCovariance",
        "aberrant.model.stat",
        "statistical",
        _moving_capabilities(TWO_FEATURES),
    ),
    _model(
        "moving_mahalanobis_distance",
        "MovingMahalanobisDistance",
        "aberrant.model.stat",
        "statistical",
        _capabilities(
            event_kind=EventKind.TABULAR,
            score_kind=ScoreKind.NON_NEGATIVE,
            state=StateKind.BOUNDED,
        ),
    ),
    _model(
        "online_copod",
        "OnlineCOPOD",
        "aberrant.model.stat",
        "statistical",
        _capabilities(
            event_kind=EventKind.TABULAR,
            feature_count=_time_aware_features,
            score_kind=ScoreKind.NON_NEGATIVE,
            state=StateKind.BOUNDED,
        ),
    ),
    _model(
        "online_ecod",
        "OnlineECOD",
        "aberrant.model.stat",
        "statistical",
        _capabilities(
            event_kind=EventKind.TABULAR,
            feature_count=_time_aware_features,
            score_kind=ScoreKind.NON_NEGATIVE,
            state=StateKind.BOUNDED,
        ),
    ),
    _model(
        "adaptive_one_class_svm",
        "AdaptiveOneClassSVM",
        "aberrant.model.svm",
        "svm",
        _capabilities(
            event_kind=EventKind.TABULAR,
            score_kind=ScoreKind.SIGNED,
            state=StateKind.BOUNDED,
        ),
    ),
    _model(
        "multivariate_rolling_matrix_profile",
        "MultivariateRollingMatrixProfile",
        "aberrant.model.timeseries",
        "time_series",
        _capabilities(
            event_kind=EventKind.TABULAR,
            score_kind=ScoreKind.NON_NEGATIVE,
            state=StateKind.BOUNDED,
        ),
    ),
    _model(
        "residual_detector",
        "ResidualDetector",
        "aberrant.forecast",
        "forecast",
        _capabilities(
            event_kind=_residual_event_kind,
            feature_count=_residual_features,
            score_kind=ScoreKind.NON_NEGATIVE,
            state=StateKind.BOUNDED,
        ),
        dependencies=(ComponentDependency("model", ComponentKind.FORECASTER),),
    ),
    _model(
        "rolling_matrix_profile",
        "RollingMatrixProfile",
        "aberrant.model.timeseries",
        "time_series",
        _capabilities(
            event_kind=EventKind.UNIVARIATE,
            feature_count=ONE_FEATURE,
            score_kind=ScoreKind.NON_NEGATIVE,
            state=StateKind.BOUNDED,
        ),
    ),
    _model(
        "damp",
        "DAMP",
        "aberrant.model.timeseries",
        "time_series",
        _capabilities(
            event_kind=EventKind.UNIVARIATE,
            feature_count=ONE_FEATURE,
            score_kind=ScoreKind.NON_NEGATIVE,
            state=StateKind.BOUNDED,
        ),
    ),
)


def _preserve_features(
    _params: Parameters,
    input_feature_count: int | None,
) -> int | None:
    return input_feature_count


def _schema_features(
    params: Parameters,
    input_feature_count: int | None,
) -> int | None:
    features = params.get("features")
    if features is None:
        return input_feature_count
    return len(cast(Sequence[str], features))


def _projected_features(
    params: Parameters,
    input_feature_count: int | None,
) -> int:
    output = _as_int(params["n_components"])
    if input_feature_count is not None and output > input_feature_count:
        raise ValueError(
            f"n_components ({output}) exceeds input feature count "
            f"({input_feature_count})"
        )
    return output


def _projected_names(
    params: Parameters, _inputs: tuple[str, ...] | None
) -> tuple[str, ...]:
    return tuple(
        f"component_{index}" for index in range(_as_int(params["n_components"]))
    )


def _selected_names(
    params: Parameters, inputs: tuple[str, ...] | None
) -> tuple[str, ...]:
    features = cast(Sequence[str], params["features"])
    if not features:
        raise ValueError("features must be a non-empty sequence")
    if any(not isinstance(name, str) or not name for name in features):
        raise ValueError("features must contain non-empty names")
    if len(set(features)) != len(features):
        raise ValueError("features cannot contain duplicates")
    if inputs is not None and not set(features).issubset(inputs):
        raise ValueError("Selected features must exist in the input")
    return tuple(features)


def _selected_count(params: Parameters, _inputs: int | None) -> int:
    return len(_selected_names(params, None))


def _renamed_names(
    params: Parameters, inputs: tuple[str, ...] | None
) -> tuple[str, ...] | None:
    mapping = cast(Mapping[str, str], params["mapping"])
    if inputs is None:
        return None
    if not set(mapping).issubset(inputs):
        raise ValueError("Renamed features must exist in the input")
    names = tuple(mapping.get(name, name) for name in inputs)
    if len(names) != len(set(names)):
        raise ValueError("Renaming produces duplicate features")
    return names


def _lagged_names(
    params: Parameters, inputs: tuple[str, ...] | None
) -> tuple[str, ...] | None:
    if inputs is None:
        return None
    lags = sorted(cast(Sequence[int], params["lags"]))
    apply_to = params["apply_to"]
    lagged = inputs if apply_to is None else cast(Sequence[str], apply_to)
    if not set(lagged).issubset(inputs):
        raise ValueError("Lagged features must exist in the input")
    names = (*inputs, *(f"{name}_lag{lag}" for name in lagged for lag in lags))
    if len(names) != len(set(names)):
        raise ValueError("Lagging produces duplicate features")
    return names


def _lagged_count(params: Parameters, inputs: int | None) -> int | None:
    lags = len(cast(Sequence[int], params["lags"]))
    apply_to = params["apply_to"]
    if apply_to is None:
        return None if inputs is None else inputs * (1 + lags)
    return (
        None if inputs is None else inputs + len(cast(Sequence[str], apply_to)) * lags
    )


def _imputed_names(
    params: Parameters, inputs: tuple[str, ...] | None
) -> tuple[str, ...] | None:
    features = params.get("features")
    names = inputs if features is None else tuple(cast(Sequence[str], features))
    if names is None:
        return None
    if inputs is not None and features is not None and set(inputs) != set(names):
        raise ValueError("Imputed features must match the input")
    if not params["indicate"]:
        return names
    indicated = (*names, *(f"{name}_missing" for name in names))
    if len(indicated) != len(set(indicated)):
        raise ValueError("Missing indicators produce duplicate features")
    return indicated


def _imputed_count(params: Parameters, inputs: int | None) -> int | None:
    count = _schema_features(params, inputs)
    if count is None:
        return None
    return 2 * count if params["indicate"] else count


def _transformer(
    *,
    component_id: str,
    class_name: str,
    module: str,
    family: str,
    resolver: Callable[[Parameters, int | None], int | None],
    names_resolver: Callable[
        [Parameters, tuple[str, ...] | None], tuple[str, ...] | None
    ]
    | None = None,
) -> TransformerSpec:
    return TransformerSpec(
        id=component_id,
        display_name=class_name,
        kind=ComponentKind.TRANSFORMER,
        family=family,
        import_path=f"{module}.{class_name}",
        _feature_count_resolver=resolver,
        _feature_names_resolver=names_resolver,
    )


_TRANSFORMER_SPECS = (
    _transformer(
        component_id="select",
        class_name="Select",
        module="aberrant.transform",
        family="composition",
        resolver=_selected_count,
        names_resolver=_selected_names,
    ),
    _transformer(
        component_id="rename",
        class_name="Rename",
        module="aberrant.transform",
        family="composition",
        resolver=_preserve_features,
        names_resolver=_renamed_names,
    ),
    _transformer(
        component_id="passthrough",
        class_name="Passthrough",
        module="aberrant.transform",
        family="composition",
        resolver=_preserve_features,
    ),
    _transformer(
        component_id="feature_schema_guard",
        class_name="FeatureSchemaGuard",
        module="aberrant.transform",
        family="validation",
        resolver=_schema_features,
    ),
    _transformer(
        component_id="imputer",
        class_name="Imputer",
        module="aberrant.transform",
        family="validation",
        resolver=_imputed_count,
        names_resolver=_imputed_names,
    ),
    _transformer(
        component_id="min_max_scaler",
        class_name="MinMaxScaler",
        module="aberrant.transform",
        family="preprocessing",
        resolver=_preserve_features,
    ),
    _transformer(
        component_id="standard_scaler",
        class_name="StandardScaler",
        module="aberrant.transform",
        family="preprocessing",
        resolver=_preserve_features,
    ),
    _transformer(
        component_id="rolling_robust_scaler",
        class_name="RollingRobustScaler",
        module="aberrant.transform",
        family="preprocessing",
        resolver=_preserve_features,
    ),
    _transformer(
        component_id="lags",
        class_name="Lags",
        module="aberrant.transform",
        family="sequence",
        resolver=_lagged_count,
        names_resolver=_lagged_names,
    ),
    _transformer(
        component_id="difference",
        class_name="Difference",
        module="aberrant.transform",
        family="sequence",
        resolver=_preserve_features,
    ),
    _transformer(
        component_id="exponential_smoother",
        class_name="ExponentialSmoother",
        module="aberrant.transform",
        family="sequence",
        resolver=_preserve_features,
    ),
    _transformer(
        component_id="despike",
        class_name="Despike",
        module="aberrant.transform",
        family="sequence",
        resolver=_preserve_features,
    ),
    _transformer(
        component_id="rolling_statistic",
        class_name="RollingStatistic",
        module="aberrant.transform",
        family="sequence",
        resolver=_preserve_features,
    ),
    _transformer(
        component_id="deseasonalize",
        class_name="Deseasonalize",
        module="aberrant.transform",
        family="sequence",
        resolver=_preserve_features,
    ),
    _transformer(
        component_id="incremental_pca",
        class_name="IncrementalPCA",
        module="aberrant.transform",
        family="projection",
        resolver=_projected_features,
        names_resolver=_projected_names,
    ),
    _transformer(
        component_id="random_projection",
        class_name="RandomProjection",
        module="aberrant.transform",
        family="projection",
        resolver=_projected_features,
        names_resolver=_projected_names,
    ),
)

_FORECASTER_SPECS = (
    ComponentSpec(
        id="seasonal_naive",
        display_name="SeasonalNaive",
        kind=ComponentKind.FORECASTER,
        family="forecast",
        import_path="aberrant.forecast.SeasonalNaive",
    ),
    ComponentSpec(
        id="holt_winters",
        display_name="HoltWinters",
        kind=ComponentKind.FORECASTER,
        family="forecast",
        import_path="aberrant.forecast.HoltWinters",
    ),
    ComponentSpec(
        id="kalman_forecaster",
        display_name="KalmanForecaster",
        kind=ComponentKind.FORECASTER,
        family="forecast",
        import_path="aberrant.forecast.KalmanForecaster",
    ),
    ComponentSpec(
        id="autoregressive",
        display_name="AutoRegressive",
        kind=ComponentKind.FORECASTER,
        family="forecast",
        import_path="aberrant.forecast.AutoRegressive",
    ),
    ComponentSpec(
        id="echo_state_forecaster",
        display_name="EchoStateForecaster",
        kind=ComponentKind.FORECASTER,
        family="forecast",
        import_path="aberrant.forecast.EchoStateForecaster",
    ),
    ComponentSpec(
        id="lag_forecaster",
        display_name="LagForecaster",
        kind=ComponentKind.FORECASTER,
        family="forecast",
        import_path="aberrant.forecast.LagForecaster",
        dependencies=(ComponentDependency("regressor", ComponentKind.FORECASTER),),
    ),
    ComponentSpec(
        id="ensemble_forecaster",
        display_name="EnsembleForecaster",
        kind=ComponentKind.FORECASTER,
        family="forecast",
        import_path="aberrant.forecast.EnsembleForecaster",
        dependencies=(
            ComponentDependency("forecasters", ComponentKind.FORECASTER, many=True),
        ),
    ),
    ComponentSpec(
        id="rls_regressor",
        display_name="RLSRegressor",
        kind=ComponentKind.FORECASTER,
        family="forecast",
        import_path="aberrant.forecast.RLSRegressor",
    ),
    ComponentSpec(
        id="random_fourier_regressor",
        display_name="RandomFourierRegressor",
        kind=ComponentKind.FORECASTER,
        family="forecast",
        import_path="aberrant.forecast.RandomFourierRegressor",
    ),
    ComponentSpec(
        id="passive_aggressive_regressor",
        display_name="PassiveAggressiveRegressor",
        kind=ComponentKind.FORECASTER,
        family="forecast",
        import_path="aberrant.forecast.PassiveAggressiveRegressor",
    ),
    ComponentSpec(
        id="sgd_regressor",
        display_name="SGDRegressor",
        kind=ComponentKind.FORECASTER,
        family="forecast",
        import_path="aberrant.forecast.SGDRegressor",
    ),
    ComponentSpec(
        id="knn_regressor",
        display_name="KNNRegressor",
        kind=ComponentKind.FORECASTER,
        family="forecast",
        import_path="aberrant.forecast.KNNRegressor",
    ),
    ComponentSpec(
        id="hoeffding_tree_regressor",
        display_name="HoeffdingTreeRegressor",
        kind=ComponentKind.FORECASTER,
        family="forecast",
        import_path="aberrant.forecast.HoeffdingTreeRegressor",
    ),
    ComponentSpec(
        id="adaptive_random_forest_regressor",
        display_name="AdaptiveRandomForestRegressor",
        kind=ComponentKind.FORECASTER,
        family="forecast",
        import_path="aberrant.forecast.AdaptiveRandomForestRegressor",
    ),
)


def _index(specs: tuple[_Spec, ...]) -> Mapping[str, _Spec]:
    indexed = {spec.id: spec for spec in specs}
    if len(indexed) != len(specs):
        raise RuntimeError("Catalog component identifiers must be unique")
    return MappingProxyType(indexed)


MODEL_CATALOG: Mapping[str, ModelSpec] = _index(_MODEL_SPECS)
"""Every model by catalog ID, read-only."""
TRANSFORMER_CATALOG: Mapping[str, TransformerSpec] = _index(_TRANSFORMER_SPECS)
"""Every transformer by catalog ID, read-only."""
FORECASTER_CATALOG: Mapping[str, ComponentSpec] = _index(_FORECASTER_SPECS)
"""Every forecaster and regressor by catalog ID, read-only."""


def get_model_spec(component_id: str) -> ModelSpec:
    """Return one model entry or raise ``UnknownComponentError``."""
    try:
        return MODEL_CATALOG[component_id]
    except KeyError as exc:
        raise UnknownComponentError(ComponentKind.MODEL.value, component_id) from exc


def get_transformer_spec(component_id: str) -> TransformerSpec:
    """Return one transformer entry or raise ``UnknownComponentError``."""
    try:
        return TRANSFORMER_CATALOG[component_id]
    except KeyError as exc:
        raise UnknownComponentError(
            ComponentKind.TRANSFORMER.value,
            component_id,
        ) from exc


def get_forecaster_spec(component_id: str) -> ComponentSpec:
    """Return one forecaster or regressor entry or raise ``UnknownComponentError``."""
    try:
        return FORECASTER_CATALOG[component_id]
    except KeyError as exc:
        raise UnknownComponentError(
            ComponentKind.FORECASTER.value,
            component_id,
        ) from exc
