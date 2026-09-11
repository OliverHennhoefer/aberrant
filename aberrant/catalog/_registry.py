"""Built-in component catalog and lookup functions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

from aberrant import __version__
from aberrant.base.exceptions import UnknownComponentError

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
    WarmupRequirement,
    WarmupUnit,
)

Parameters = Mapping[str, object]
FeatureCountValue = FeatureCount | Callable[[Parameters], FeatureCount]
FeatureSchemaValue = FeatureSchemaKind | Callable[[Parameters], FeatureSchemaKind]
ScoreKindValue = ScoreKind | Callable[[Parameters], ScoreKind]
StateKindValue = StateKind | Callable[[Parameters], StateKind]
OrientationValue = bool | None | Callable[[Parameters], bool | None]

ANY_FEATURES = FeatureCount()
ONE_FEATURE = FeatureCount(1, 1)
TWO_FEATURES = FeatureCount(2, 2)
CATALOG_FORMAT_VERSION = 1


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise TypeError("expected an integer-compatible JSON value")
    return int(value)


def _events(count: int | None) -> WarmupRequirement:
    return WarmupRequirement(count, WarmupUnit.EVENTS)


def _fixed_warmup(count: int) -> Callable[[Parameters], WarmupRequirement]:
    return lambda _params: _events(count)


def _parameter_warmup(name: str) -> Callable[[Parameters], WarmupRequirement]:
    def resolve(params: Parameters) -> WarmupRequirement:
        value = params.get(name)
        return _events(None if value is None else _as_int(value))

    return resolve


def _resolve_value(value: object, params: Parameters) -> object:
    return value(params) if callable(value) else value


def _capabilities(
    *,
    event_kind: EventKind,
    feature_count: FeatureCountValue = ANY_FEATURES,
    feature_schema: FeatureSchemaValue = FeatureSchemaKind.FIXED,
    score_kind: ScoreKindValue,
    warmup: Callable[[Parameters], WarmupRequirement],
    state: StateKindValue,
    resettable: bool,
    higher_is_more_anomalous: OrientationValue = True,
    requires_unit_interval: bool = False,
) -> Callable[[Parameters], ModelCapabilities]:
    def resolve(params: Parameters) -> ModelCapabilities:
        resolved_feature_count = _resolve_value(feature_count, params)
        resolved_feature_schema = _resolve_value(feature_schema, params)
        resolved_score_kind = _resolve_value(score_kind, params)
        resolved_state = _resolve_value(state, params)
        resolved_orientation = _resolve_value(higher_is_more_anomalous, params)
        assert isinstance(resolved_feature_count, FeatureCount)
        assert isinstance(resolved_feature_schema, FeatureSchemaKind)
        assert isinstance(resolved_score_kind, ScoreKind)
        assert isinstance(resolved_state, StateKind)
        assert isinstance(resolved_orientation, bool) or resolved_orientation is None
        return ModelCapabilities(
            event_kind=event_kind,
            feature_count=resolved_feature_count,
            feature_schema=resolved_feature_schema,
            score_kind=resolved_score_kind,
            higher_is_more_anomalous=resolved_orientation,
            warmup=warmup(params),
            state=resolved_state,
            resettable=resettable,
            requires_unit_interval=requires_unit_interval,
        )

    return resolve


def _normalized_score(params: Parameters) -> ScoreKind:
    return (
        ScoreKind.ZERO_TO_ONE
        if bool(params.get("normalize_score", False))
        else ScoreKind.NON_NEGATIVE
    )


def _difference_score(params: Parameters) -> ScoreKind:
    return (
        ScoreKind.NON_NEGATIVE
        if bool(params.get("abs_diff", True))
        else ScoreKind.SIGNED
    )


def _difference_orientation(params: Parameters) -> bool | None:
    return True if bool(params.get("abs_diff", True)) else None


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


def _quantile_warmup(params: Parameters) -> WarmupRequirement:
    window_size = _as_int(params.get("window_size", 1000))
    return _events(min(window_size, max(10, int(window_size * 0.1))))


def _kitnet_warmup(params: Parameters) -> WarmupRequirement:
    feature_map = _as_int(params.get("feature_map_grace", 5000))
    detector = _as_int(params.get("ad_grace", 50000))
    return _events(feature_map + detector)


def _uses_faiss(params: Parameters) -> bool:
    engine = params.get("similarity_engine")
    if isinstance(engine, Mapping):
        return engine.get("id") == "faiss"
    return engine is not None and engine.__class__.__name__ == (
        "FaissSimilaritySearchEngine"
    )


def _knn_feature_schema(params: Parameters) -> FeatureSchemaKind:
    return (
        FeatureSchemaKind.FIXED
        if _uses_faiss(params)
        else FeatureSchemaKind.ENGINE_DEFINED
    )


def _knn_score(params: Parameters) -> ScoreKind:
    return ScoreKind.NON_NEGATIVE if _uses_faiss(params) else ScoreKind.ENGINE_DEFINED


def _knn_state(params: Parameters) -> StateKind:
    return StateKind.BOUNDED if _uses_faiss(params) else StateKind.EXTERNAL


def _knn_orientation(params: Parameters) -> bool | None:
    return True if _uses_faiss(params) else None


def _knn_warmup(params: Parameters) -> WarmupRequirement:
    k = _as_int(params.get("k", 1))
    engine = params.get("similarity_engine")
    if isinstance(engine, Mapping):
        engine_params = engine.get("params", {})
        if isinstance(engine_params, Mapping) and "warm_up" in engine_params:
            warm_up = _as_int(engine_params["warm_up"])
            window_size = _as_int(engine_params.get("window_size", warm_up))
            if not k <= warm_up <= window_size:
                raise ValueError("FAISS KNN requires k <= warm_up <= window_size")
            return _events(warm_up)
    engine_warm_up = getattr(engine, "warm_up", None)
    if engine_warm_up is not None:
        minimum = _as_int(engine_warm_up)
        engine_window_size = getattr(engine, "window_size", None)
        if k > minimum or (
            engine_window_size is not None and minimum > _as_int(engine_window_size)
        ):
            raise ValueError("KNN requires k <= warm_up <= window_size")
        return _events(minimum)
    return WarmupRequirement(None, WarmupUnit.ENGINE_DEFINED)


def _cell_warmup(params: Parameters) -> WarmupRequirement:
    slides = _as_int(params.get("warm_up_slides", 1))
    slide_size = _as_int(params.get("slide_size", 500))
    neighbors = _as_int(params.get("k", 50)) + 1
    return _events(max(slides * slide_size, neighbors))


def _sdo_warmup(params: Parameters) -> WarmupRequirement:
    k = _as_int(params.get("k", 256))
    configured = params.get("warm_up_observers")
    minimum = max(_as_int(params.get("x_neighbors", 6)), 2)
    if configured is not None:
        minimum = max(minimum, _as_int(configured))
    return WarmupRequirement(min(k, minimum), WarmupUnit.OBSERVERS)


def _active_graph_warmup(params: Parameters) -> WarmupRequirement:
    return WarmupRequirement(
        _as_int(params.get("warm_up_graphs", 32)),
        WarmupUnit.ACTIVE_GRAPHS,
    )


def _asd_warmup(params: Parameters) -> WarmupRequirement:
    window_size = params.get("window_size")
    return _events(
        _as_int(params.get("max_samples", 256) if window_size is None else window_size)
    )


def _random_cut_warmup(params: Parameters) -> WarmupRequirement:
    configured = params.get("warmup_samples")
    warmup = _as_int(
        params.get("sample_size", 256) if configured is None else configured
    )
    return _events(warmup + _as_int(params.get("shingle_size", 1)) - 1)


def _xstream_warmup(params: Parameters) -> WarmupRequirement:
    return _events(
        max(
            _as_int(params.get("init_sample_size", 256)),
            _as_int(params.get("window_size", 256)),
        )
    )


def _xstream_state(params: Parameters) -> StateKind:
    return (
        StateKind.GROWING
        if params.get("max_feature_cache_size") is None
        else StateKind.BOUNDED
    )


def _mstream_warmup(params: Parameters) -> WarmupRequirement:
    return WarmupRequirement(
        _as_int(params.get("warm_up_buckets", 0)),
        WarmupUnit.BUCKETS,
    )


def _damp_warmup(params: Parameters) -> WarmupRequirement:
    subsequence_length = params.get("subsequence_length")
    if subsequence_length is None:
        return _events(None)
    start_index = params.get("start_index")
    resolved = (
        4 * _as_int(subsequence_length) if start_index is None else _as_int(start_index)
    )
    return _events(resolved + _as_int(subsequence_length) - 2)


def _moving_capabilities(
    feature_count: FeatureCount,
    *,
    warmup: int,
) -> Callable[[Parameters], ModelCapabilities]:
    return _capabilities(
        event_kind=(
            EventKind.UNIVARIATE
            if feature_count == ONE_FEATURE
            else EventKind.BIVARIATE
        ),
        feature_count=feature_count,
        score_kind=_difference_score,
        warmup=_fixed_warmup(warmup),
        state=StateKind.BOUNDED,
        resettable=False,
        higher_is_more_anomalous=_difference_orientation,
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
            warmup=_fixed_warmup(0),
            state=StateKind.STATELESS,
            resettable=False,
            higher_is_more_anomalous=None,
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
            warmup=_fixed_warmup(0),
            state=StateKind.BOUNDED,
            resettable=False,
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
            warmup=_fixed_warmup(0),
            state=StateKind.STATELESS,
            resettable=False,
        ),
    ),
    _model(
        "quantile_threshold",
        "QuantileThreshold",
        "aberrant.model",
        "score_policy",
        _capabilities(
            event_kind=EventKind.SCORE,
            feature_count=ONE_FEATURE,
            score_kind=ScoreKind.ZERO_TO_ONE,
            warmup=_quantile_warmup,
            state=StateKind.BOUNDED,
            resettable=True,
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
            warmup=_kitnet_warmup,
            state=StateKind.BOUNDED,
            resettable=True,
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
            warmup=_fixed_warmup(0),
            state=StateKind.EXTERNAL,
            resettable=False,
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
            feature_schema=_knn_feature_schema,
            score_kind=_knn_score,
            warmup=_knn_warmup,
            state=_knn_state,
            resettable=False,
            higher_is_more_anomalous=_knn_orientation,
        ),
        dependencies=(
            ComponentDependency(
                "similarity_engine",
                ComponentKind.SIMILARITY_ENGINE,
            ),
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
            warmup=lambda params: _events(_as_int(params.get("k", 10)) + 1),
            state=StateKind.BOUNDED,
            resettable=False,
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
            warmup=_cell_warmup,
            state=StateKind.BOUNDED,
            resettable=True,
        ),
    ),
    _model(
        "sdo_stream",
        "SDOStream",
        "aberrant.model.distance",
        "distance",
        _capabilities(
            event_kind=EventKind.TABULAR,
            feature_count=_time_aware_features,
            score_kind=ScoreKind.NON_NEGATIVE,
            warmup=_sdo_warmup,
            state=StateKind.BOUNDED,
            resettable=True,
        ),
    ),
    _model(
        "stationary_region_neighbor_detector",
        "StationaryRegionNeighborDetector",
        "aberrant.model.distance",
        "distance",
        _capabilities(
            event_kind=EventKind.TABULAR,
            feature_count=_time_aware_features,
            score_kind=ScoreKind.ZERO_TO_ONE,
            warmup=_cell_warmup,
            state=StateKind.BOUNDED,
            resettable=True,
        ),
    ),
    _model(
        "ano_edge_l",
        "AnoEdgeL",
        "aberrant.model.graph",
        "graph",
        _capabilities(
            event_kind=EventKind.EDGE,
            feature_count=_edge_features,
            score_kind=_normalized_score,
            warmup=_parameter_warmup("warm_up_samples"),
            state=StateKind.BOUNDED,
            resettable=True,
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
            score_kind=_normalized_score,
            warmup=_parameter_warmup("warm_up_samples"),
            state=StateKind.BOUNDED,
            resettable=True,
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
            score_kind=_normalized_score,
            warmup=_parameter_warmup("warm_up_samples"),
            state=StateKind.BOUNDED,
            resettable=True,
        ),
    ),
    _model(
        "signed_graph_sketch_detector",
        "SignedGraphSketchDetector",
        "aberrant.model.graph",
        "graph",
        _capabilities(
            event_kind=EventKind.GRAPH_EDGE,
            feature_count=_graph_edge_features,
            score_kind=_normalized_score,
            warmup=_active_graph_warmup,
            state=StateKind.BOUNDED,
            resettable=True,
        ),
    ),
    _model(
        "asd_isolation_forest",
        "ASDIsolationForest",
        "aberrant.model.iforest",
        "isolation_forest",
        _capabilities(
            event_kind=EventKind.TABULAR,
            score_kind=ScoreKind.ZERO_TO_ONE,
            warmup=_asd_warmup,
            state=StateKind.BOUNDED,
            resettable=False,
        ),
    ),
    _model(
        "half_space_trees",
        "HalfSpaceTrees",
        "aberrant.model.iforest",
        "isolation_forest",
        _capabilities(
            event_kind=EventKind.TABULAR,
            score_kind=ScoreKind.ZERO_TO_ONE,
            warmup=_parameter_warmup("window_size"),
            state=StateKind.BOUNDED,
            resettable=False,
            requires_unit_interval=True,
        ),
    ),
    _model(
        "mondrian_isolation_forest",
        "MondrianIsolationForest",
        "aberrant.model.iforest",
        "isolation_forest",
        _capabilities(
            event_kind=EventKind.TABULAR,
            score_kind=ScoreKind.ZERO_TO_ONE,
            warmup=_fixed_warmup(2),
            state=StateKind.GROWING,
            resettable=False,
        ),
    ),
    _model(
        "online_isolation_forest",
        "OnlineIsolationForest",
        "aberrant.model.iforest",
        "isolation_forest",
        _capabilities(
            event_kind=EventKind.TABULAR,
            score_kind=ScoreKind.ZERO_TO_ONE,
            warmup=_fixed_warmup(1),
            state=StateKind.BOUNDED,
            resettable=False,
        ),
    ),
    _model(
        "random_cut_forest",
        "RandomCutForest",
        "aberrant.model.iforest",
        "isolation_forest",
        _capabilities(
            event_kind=EventKind.TABULAR,
            score_kind=_normalized_score,
            warmup=_random_cut_warmup,
            state=StateKind.BOUNDED,
            resettable=True,
        ),
    ),
    _model(
        "stream_random_histogram_forest",
        "StreamRandomHistogramForest",
        "aberrant.model.iforest",
        "isolation_forest",
        _capabilities(
            event_kind=EventKind.TABULAR,
            score_kind=ScoreKind.NON_NEGATIVE,
            warmup=_parameter_warmup("window_size"),
            state=StateKind.BOUNDED,
            resettable=False,
        ),
    ),
    _model(
        "x_stream",
        "XStream",
        "aberrant.model.iforest",
        "isolation_forest",
        _capabilities(
            event_kind=EventKind.TABULAR,
            feature_schema=FeatureSchemaKind.EVOLVING,
            score_kind=ScoreKind.ZERO_TO_ONE,
            warmup=_xstream_warmup,
            state=_xstream_state,
            resettable=True,
        ),
    ),
    _model(
        "m_stream",
        "MStream",
        "aberrant.model.sketch",
        "sketch",
        _capabilities(
            event_kind=EventKind.TABULAR,
            feature_count=_time_aware_features,
            score_kind=ScoreKind.NON_NEGATIVE,
            warmup=_mstream_warmup,
            state=StateKind.BOUNDED,
            resettable=True,
        ),
    ),
    _model(
        "streaming_loda",
        "StreamingLODA",
        "aberrant.model.sketch",
        "sketch",
        _capabilities(
            event_kind=EventKind.TABULAR,
            feature_count=_time_aware_features,
            score_kind=ScoreKind.NON_NEGATIVE,
            warmup=_parameter_warmup("warm_up_samples"),
            state=StateKind.BOUNDED,
            resettable=True,
        ),
    ),
    _model(
        "streaming_rs_hash",
        "StreamingRSHash",
        "aberrant.model.sketch",
        "sketch",
        _capabilities(
            event_kind=EventKind.TABULAR,
            feature_count=_time_aware_features,
            score_kind=ScoreKind.ZERO_TO_ONE,
            warmup=_parameter_warmup("warm_up_samples"),
            state=StateKind.BOUNDED,
            resettable=True,
        ),
    ),
    *(
        _model(
            component_id,
            class_name,
            "aberrant.model.stat",
            "statistical",
            _moving_capabilities(ONE_FEATURE, warmup=1),
        )
        for component_id, class_name in (
            ("moving_average", "MovingAverage"),
            ("moving_average_absolute_deviation", "MovingAverageAbsoluteDeviation"),
            ("moving_geometric_average", "MovingGeometricAverage"),
            ("moving_harmonic_average", "MovingHarmonicAverage"),
            ("moving_interquartile_range", "MovingInterquartileRange"),
            ("moving_kurtosis", "MovingKurtosis"),
            ("moving_median", "MovingMedian"),
            ("moving_quantile", "MovingQuantile"),
            ("moving_skewness", "MovingSkewness"),
            ("moving_variance", "MovingVariance"),
        )
    ),
    _model(
        "moving_correlation_coefficient",
        "MovingCorrelationCoefficient",
        "aberrant.model.stat",
        "statistical",
        _moving_capabilities(TWO_FEATURES, warmup=2),
    ),
    _model(
        "moving_covariance",
        "MovingCovariance",
        "aberrant.model.stat",
        "statistical",
        _moving_capabilities(TWO_FEATURES, warmup=2),
    ),
    _model(
        "moving_mahalanobis_distance",
        "MovingMahalanobisDistance",
        "aberrant.model.stat",
        "statistical",
        _capabilities(
            event_kind=EventKind.TABULAR,
            score_kind=ScoreKind.NON_NEGATIVE,
            warmup=_fixed_warmup(3),
            state=StateKind.BOUNDED,
            resettable=False,
        ),
    ),
    _model(
        "graph_gated_one_class_svm",
        "GraphGatedOneClassSVM",
        "aberrant.model.svm",
        "svm",
        _capabilities(
            event_kind=EventKind.TABULAR,
            score_kind=ScoreKind.NON_NEGATIVE,
            warmup=_fixed_warmup(1),
            state=StateKind.BOUNDED,
            resettable=False,
        ),
    ),
    _model(
        "incremental_one_class_svm_adaptive_kernel",
        "IncrementalOneClassSVMAdaptiveKernel",
        "aberrant.model.svm",
        "svm",
        _capabilities(
            event_kind=EventKind.TABULAR,
            score_kind=ScoreKind.SIGNED,
            warmup=_fixed_warmup(1),
            state=StateKind.BOUNDED,
            resettable=False,
        ),
    ),
    _model(
        "x_lag_damp",
        "XLagDAMP",
        "aberrant.model.timeseries",
        "time_series",
        _capabilities(
            event_kind=EventKind.UNIVARIATE,
            feature_count=ONE_FEATURE,
            score_kind=ScoreKind.NON_NEGATIVE,
            warmup=_damp_warmup,
            state=StateKind.BOUNDED,
            resettable=True,
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
    if not isinstance(features, (list, tuple)):
        raise TypeError("features must be an array")
    return len(features)


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


def _transformer(
    component_id: str,
    class_name: str,
    module: str,
    family: str,
    resolver: Callable[[Parameters, int | None], int | None],
) -> TransformerSpec:
    return TransformerSpec(
        id=component_id,
        display_name=class_name,
        kind=ComponentKind.TRANSFORMER,
        family=family,
        import_path=f"{module}.{class_name}",
        _feature_count_resolver=resolver,
    )


_TRANSFORMER_SPECS = (
    _transformer(
        "feature_schema_guard",
        "FeatureSchemaGuard",
        "aberrant.transform.preprocessing",
        "validation",
        _schema_features,
    ),
    _transformer(
        "min_max_scaler",
        "MinMaxScaler",
        "aberrant.transform.preprocessing",
        "preprocessing",
        _preserve_features,
    ),
    _transformer(
        "standard_scaler",
        "StandardScaler",
        "aberrant.transform.preprocessing",
        "preprocessing",
        _preserve_features,
    ),
    _transformer(
        "incremental_pca",
        "IncrementalPCA",
        "aberrant.transform.projection",
        "projection",
        _projected_features,
    ),
    _transformer(
        "random_projection",
        "RandomProjection",
        "aberrant.transform.projection",
        "projection",
        _projected_features,
    ),
)

_SIMILARITY_SPECS = (
    ComponentSpec(
        id="faiss",
        display_name="FaissSimilaritySearchEngine",
        kind=ComponentKind.SIMILARITY_ENGINE,
        family="nearest_neighbor",
        import_path="aberrant.similarity.FaissSimilaritySearchEngine",
        optional_extra="faiss",
        dependency_module="faiss",
    ),
)


def _index(specs: tuple[ComponentSpec, ...]) -> Mapping[str, ComponentSpec]:
    indexed = {spec.id: spec for spec in specs}
    if len(indexed) != len(specs):
        raise RuntimeError("Catalog component identifiers must be unique")
    return MappingProxyType(indexed)


MODEL_CATALOG: Mapping[str, ModelSpec] = MappingProxyType(
    {spec.id: spec for spec in _MODEL_SPECS}
)
TRANSFORMER_CATALOG: Mapping[str, TransformerSpec] = MappingProxyType(
    {spec.id: spec for spec in _TRANSFORMER_SPECS}
)
SIMILARITY_ENGINE_CATALOG: Mapping[str, ComponentSpec] = _index(_SIMILARITY_SPECS)

if len(MODEL_CATALOG) != len(_MODEL_SPECS):
    raise RuntimeError("Catalog model identifiers must be unique")
if len(TRANSFORMER_CATALOG) != len(_TRANSFORMER_SPECS):
    raise RuntimeError("Catalog transformer identifiers must be unique")


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


def get_similarity_engine_spec(component_id: str) -> ComponentSpec:
    """Return one similarity-engine entry or raise ``UnknownComponentError``."""
    try:
        return SIMILARITY_ENGINE_CATALOG[component_id]
    except KeyError as exc:
        raise UnknownComponentError(
            ComponentKind.SIMILARITY_ENGINE.value,
            component_id,
        ) from exc


def catalog_manifest() -> dict[str, object]:
    """Describe every built-in component without importing optional runtimes."""
    return {
        "version": CATALOG_FORMAT_VERSION,
        "aberrant_version": __version__,
        "models": [spec.as_dict() for spec in MODEL_CATALOG.values()],
        "transformers": [spec.as_dict() for spec in TRANSFORMER_CATALOG.values()],
        "similarity_engines": [
            spec.as_dict() for spec in SIMILARITY_ENGINE_CATALOG.values()
        ],
    }
