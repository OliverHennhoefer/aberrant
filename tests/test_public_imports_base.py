"""Public API import contract tests for base install."""

import importlib
import pkgutil
import sys

import pytest

import aberrant.base as base_api
import aberrant.model as model_api
import aberrant.model.deep as deep_api
import aberrant.model.distance as distance_api
import aberrant.model.forest as forest_api
import aberrant.model.graph as graph_api
import aberrant.model.sketch as sketch_api
import aberrant.model.svm as svm_api
import aberrant.model.timeseries as timeseries_api
from aberrant import __version__
from aberrant.base import (
    BaseModel,
    BaseTransformer,
    ModelProtocol,
    Pipeline,
    TransformerProtocol,
)
from aberrant.calibration import CALIBRATORS, ScoreCalibrator
from aberrant.catalog import MODEL_CATALOG, DetectorConfig, build_detector
from aberrant.datasets import Dataset, DatasetStream, load
from aberrant.drift import ADWIN, CUSUM, KSWIN, PageHinkley
from aberrant.evaluate import (
    EvaluationRecord,
    EvaluationResult,
    PrequentialEvaluator,
    VolumeUnderSurface,
    vus,
)
from aberrant.exceptions import ConfigurationError
from aberrant.forecast import (
    AutoRegressive,
    EnsembleForecaster,
    HoltWinters,
    KalmanForecaster,
    LagForecaster,
    ResidualDetector,
    RLSRegressor,
    SeasonalNaive,
    interval,
    nonconformity,
)
from aberrant.model import NullModel, RandomModel, ThresholdModel
from aberrant.model.distance import (
    KNN,
    CellNeighborhoodDetector,
    LocalOutlierFactor,
    SDOStream,
)
from aberrant.model.forest import (
    RHF,
    ASDIsolationForest,
    HalfSpaceTrees,
    MondrianIsolationForest,
    OnlineIsolationForest,
    RandomCutForest,
    XStream,
)
from aberrant.model.graph import ISCONNA, MIDAS, AnoEdgeL, StreamSpot
from aberrant.model.sketch import LODA, MStream, RSHash
from aberrant.model.stat import MovingCovariance, MovingZScore
from aberrant.model.svm import AdaptiveOneClassSVM
from aberrant.model.timeseries import (
    DAMP,
    MultivariateRollingMatrixProfile,
    RollingMatrixProfile,
)
from aberrant.transform import (
    FeatureSchemaGuard,
    IncrementalPCA,
    MinMaxScaler,
    RandomProjection,
    RollingRobustScaler,
    StandardScaler,
)


def test_public_imports_base_smoke() -> None:
    assert isinstance(__version__, str)
    assert BaseModel is not None
    assert BaseTransformer is not None
    assert ConfigurationError is not None
    assert Pipeline is not None
    assert ModelProtocol is not None
    assert TransformerProtocol is not None
    assert ADWIN is not None
    assert KSWIN is not None
    assert PageHinkley is not None
    assert CUSUM is not None
    assert NullModel is not None
    assert RandomModel is not None
    assert ThresholdModel is not None
    assert KNN is not None
    assert LocalOutlierFactor is not None
    assert CellNeighborhoodDetector is not None
    assert SDOStream is not None
    assert AnoEdgeL is not None
    assert ISCONNA is not None
    assert MIDAS is not None
    assert StreamSpot is not None
    assert ASDIsolationForest is not None
    assert HalfSpaceTrees is not None
    assert MondrianIsolationForest is not None
    assert OnlineIsolationForest is not None
    assert RandomCutForest is not None
    assert RHF is not None
    assert XStream is not None
    assert LODA is not None
    assert MStream is not None
    assert RSHash is not None
    assert AdaptiveOneClassSVM is not None
    assert (
        DAMP is not None
        and RollingMatrixProfile is not None
        and MultivariateRollingMatrixProfile is not None
    )
    assert MovingZScore is not None
    assert MovingCovariance is not None
    assert MinMaxScaler is not None
    assert StandardScaler is not None
    assert IncrementalPCA is not None
    assert RandomProjection is not None
    assert FeatureSchemaGuard is not None
    assert DetectorConfig is not None
    assert MODEL_CATALOG
    assert build_detector is not None
    assert Dataset is not None
    assert load is not None
    assert DatasetStream is not None


def test_calibration_public_imports_base_smoke() -> None:
    assert ScoreCalibrator is not None
    assert len(CALIBRATORS) == 17


def test_forecast_public_imports_base_smoke() -> None:
    assert base_api.ForecasterProtocol is not None
    assert base_api.RegressorProtocol is not None
    for component in (
        AutoRegressive,
        EnsembleForecaster,
        HoltWinters,
        KalmanForecaster,
        LagForecaster,
        ResidualDetector,
        RLSRegressor,
        SeasonalNaive,
    ):
        assert component is not None
    assert interval is not None and nonconformity is not None


def test_rolling_robust_scaler_public_import() -> None:
    assert RollingRobustScaler is not None


def test_evaluation_public_imports_base_smoke() -> None:
    assert EvaluationRecord is not None
    assert EvaluationResult is not None
    assert PrequentialEvaluator is not None
    assert VolumeUnderSurface is not None
    assert callable(vus)


def test_removed_compatibility_exports_stay_removed() -> None:
    removed = (
        (deep_api, "KitNET"),
        (distance_api, "NETS"),
        (distance_api, "STARE"),
        (model_api, "QuantileThreshold"),
        (distance_api, "StationaryRegionNeighborDetector"),
        (graph_api, "SignedGraphSketchDetector"),
        (forest_api, "MondrianForest"),
        (forest_api, "StreamRandomHistogramForest"),
        (sketch_api, "StreamingLODA"),
        (sketch_api, "StreamingRSHash"),
        (svm_api, "GADGETSVM"),
        (svm_api, "GraphGatedOneClassSVM"),
        (svm_api, "IncrementalOneClassSVMAdaptiveKernel"),
        (timeseries_api, "XLagDAMP"),
    )
    for module, name in removed:
        assert not hasattr(module, name)

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("aberrant.model.stat.uni")


def test_optional_exports_are_excluded_from_wildcard_contracts() -> None:
    assert deep_api.__all__ == ["OnlineAutoencoderEnsemble"]


_LAZY_TORCH_EXPORTS = [
    (deep_api, "Architecture", "aberrant.model.deep._architecture"),
    (deep_api, "Autoencoder", "aberrant.model.deep.autoencoder"),
    (deep_api, "VanillaAutoencoder", "aberrant.model.deep._autoencoders"),
    (deep_api, "VanillaLSTMAutoencoder", "aberrant.model.deep._autoencoders"),
]


@pytest.mark.parametrize(("module", "name", "source"), _LAZY_TORCH_EXPORTS)
def test_torch_exports_explain_the_missing_extra(
    monkeypatch: pytest.MonkeyPatch, module, name, source
) -> None:
    monkeypatch.delattr(module, name, raising=False)
    monkeypatch.delitem(sys.modules, source, raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)

    with pytest.raises(ImportError, match="optional 'dl' dependencies"):
        getattr(module, name)


@pytest.mark.parametrize(("module", "name", "source"), _LAZY_TORCH_EXPORTS)
def test_torch_exports_load_on_first_access(
    monkeypatch: pytest.MonkeyPatch, module, name, source
) -> None:
    pytest.importorskip("torch")
    monkeypatch.delattr(module, name, raising=False)

    exported = getattr(module, name)

    assert exported is getattr(importlib.import_module(source), name)
    assert module.__dict__[name] is exported


@pytest.mark.parametrize("module", [deep_api])
def test_unknown_attributes_are_still_attribute_errors(module) -> None:
    with pytest.raises(AttributeError, match="has no attribute 'Missing'"):
        _ = module.Missing


def _public_packages() -> list[str]:
    import aberrant  # noqa: PLC0415

    return [
        info.name
        for info in pkgutil.walk_packages(aberrant.__path__, "aberrant.")
        if info.ispkg and not any(part.startswith("_") for part in info.name.split("."))
    ]


def test_every_export_has_one_public_home() -> None:
    homes: dict[int, list[str]] = {}
    for package in _public_packages():
        module = importlib.import_module(package)
        for name in module.__all__:
            value = getattr(module, name)
            if isinstance(value, int | str):
                continue
            homes.setdefault(id(value), []).append(f"{package}.{name}")
    assert [paths for paths in homes.values() if len(paths) > 1] == []


@pytest.mark.parametrize(
    "module",
    [
        "aberrant.model.iforest",
        "aberrant.base.exceptions",
        "aberrant.similarity",
        "aberrant.base.similarity",
        "aberrant.base.architecture",
        "aberrant._utils.similar",
        "aberrant._utils.deep",
        "aberrant.transform.preprocessing",
        "aberrant.transform.projection",
        "aberrant.stream",
        "aberrant.utils",
    ],
)
def test_moved_modules_are_gone(module: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)
