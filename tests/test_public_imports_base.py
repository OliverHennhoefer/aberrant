"""Public API import contract tests for base install."""

import importlib
import sys

import pytest

import aberrant.base as base_api
import aberrant.model.deep as deep_api
import aberrant.model.distance as distance_api
import aberrant.model.graph as graph_api
import aberrant.model.iforest as iforest_api
import aberrant.model.sketch as sketch_api
import aberrant.model.svm as svm_api
import aberrant.stream.dataset as dataset_api
from aberrant import __version__
from aberrant.base import (
    BaseModel,
    BaseTransformer,
    ConfigurationError,
    ModelProtocol,
    Pipeline,
    TransformerProtocol,
)
from aberrant.calibration import CALIBRATORS, ScoreCalibrator
from aberrant.catalog import MODEL_CATALOG, DetectorConfig, build_detector
from aberrant.drift import ADWIN, KSWIN, PageHinkley
from aberrant.evaluate import EvaluationRecord, EvaluationResult, PrequentialEvaluator
from aberrant.forecast import (
    Forecast,
    HoltWinters,
    ResidualDetector,
    RLSRegressor,
    SeasonalNaive,
    interval,
    nonconformity,
)
from aberrant.model import NullModel, QuantileThreshold, RandomModel, ThresholdModel
from aberrant.model.distance import (
    KNN,
    CellNeighborhoodDetector,
    LocalOutlierFactor,
    SDOStream,
)
from aberrant.model.graph import (
    ISCONNA,
    MIDAS,
    AnoEdgeL,
    SignedGraphSketchDetector,
)
from aberrant.model.iforest import (
    ASDIsolationForest,
    HalfSpaceTrees,
    MondrianIsolationForest,
    OnlineIsolationForest,
    RandomCutForest,
    StreamRandomHistogramForest,
    XStream,
)
from aberrant.model.sketch import MStream, StreamingLODA, StreamingRSHash
from aberrant.model.stat import MovingAverage, MovingCovariance
from aberrant.model.svm import IncrementalOneClassSVMAdaptiveKernel
from aberrant.model.timeseries import (
    MultivariateRollingMatrixProfile,
    RollingMatrixProfile,
    XLagDAMP,
)
from aberrant.stream import Dataset, load
from aberrant.stream.dataset import BatchStreamer, NpzStreamer
from aberrant.transform.preprocessing import (
    FeatureSchemaGuard,
    MinMaxScaler,
    RollingRobustScaler,
    StandardScaler,
)
from aberrant.transform.projection import IncrementalPCA, RandomProjection


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
    assert NullModel is not None
    assert RandomModel is not None
    assert ThresholdModel is not None
    assert QuantileThreshold is not None
    assert KNN is not None
    assert LocalOutlierFactor is not None
    assert CellNeighborhoodDetector is not None
    assert SDOStream is not None
    assert AnoEdgeL is not None
    assert ISCONNA is not None
    assert MIDAS is not None
    assert SignedGraphSketchDetector is not None
    assert ASDIsolationForest is not None
    assert HalfSpaceTrees is not None
    assert MondrianIsolationForest is not None
    assert OnlineIsolationForest is not None
    assert RandomCutForest is not None
    assert StreamRandomHistogramForest is not None
    assert XStream is not None
    assert StreamingLODA is not None
    assert MStream is not None
    assert StreamingRSHash is not None
    assert IncrementalOneClassSVMAdaptiveKernel is not None
    assert (
        XLagDAMP is not None
        and RollingMatrixProfile is not None
        and MultivariateRollingMatrixProfile is not None
    )
    assert MovingAverage is not None
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
    assert BatchStreamer is not None
    assert NpzStreamer is not None


def test_calibration_public_imports_base_smoke() -> None:
    assert ScoreCalibrator is not None
    assert len(CALIBRATORS) == 15


def test_forecast_public_imports_base_smoke() -> None:
    assert Forecast is base_api.Forecast
    assert base_api.ForecasterProtocol is not None
    assert base_api.RegressorProtocol is not None
    for component in (HoltWinters, ResidualDetector, RLSRegressor, SeasonalNaive):
        assert component is not None
    assert interval is not None and nonconformity is not None


def test_rolling_robust_scaler_public_import() -> None:
    assert RollingRobustScaler is not None


def test_evaluation_public_imports_base_smoke() -> None:
    assert EvaluationRecord is not None
    assert EvaluationResult is not None
    assert PrequentialEvaluator is not None


def test_removed_compatibility_exports_stay_removed() -> None:
    removed = (
        (deep_api, "KitNET"),
        (distance_api, "NETS"),
        (distance_api, "STARE"),
        (distance_api, "StationaryRegionNeighborDetector"),
        (graph_api, "StreamSpot"),
        (iforest_api, "MondrianForest"),
        (sketch_api, "LODA"),
        (sketch_api, "RSHash"),
        (svm_api, "GADGETSVM"),
        (svm_api, "GraphGatedOneClassSVM"),
        (dataset_api, "DatasetStreamer"),
    )
    for module, name in removed:
        assert not hasattr(module, name)

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("aberrant.model.stat.uni")


def test_optional_exports_are_excluded_from_wildcard_contracts() -> None:
    assert "Architecture" not in base_api.__all__
    assert "Autoencoder" not in deep_api.__all__


_LAZY_TORCH_EXPORTS = [
    (base_api, "Architecture", "aberrant.base.architecture"),
    (deep_api, "Autoencoder", "aberrant.model.deep.autoencoder"),
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


@pytest.mark.parametrize("module", [base_api, deep_api])
def test_unknown_attributes_are_still_attribute_errors(module) -> None:
    with pytest.raises(AttributeError, match="has no attribute 'Missing'"):
        _ = module.Missing
