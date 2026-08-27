"""Public API import contract tests for base install."""

import importlib

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
    ModelProtocol,
    Pipeline,
    TransformerProtocol,
)
from aberrant.drift import ADWIN, KSWIN, PageHinkley
from aberrant.model import NullModel, QuantileThreshold, RandomModel, ThresholdModel
from aberrant.model.distance import (
    KNN,
    CellNeighborhoodDetector,
    LocalOutlierFactor,
    SDOStream,
    StationaryRegionNeighborDetector,
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
from aberrant.model.svm import (
    GraphGatedOneClassSVM,
    IncrementalOneClassSVMAdaptiveKernel,
)
from aberrant.model.timeseries import XLagDAMP
from aberrant.stream import Dataset, load
from aberrant.stream.dataset import BatchStreamer, NpzStreamer
from aberrant.transform.preprocessing import MinMaxScaler, StandardScaler
from aberrant.transform.projection import IncrementalPCA, RandomProjection


def test_public_imports_base_smoke() -> None:
    assert isinstance(__version__, str)
    assert BaseModel is not None
    assert BaseTransformer is not None
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
    assert StationaryRegionNeighborDetector is not None
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
    assert GraphGatedOneClassSVM is not None
    assert XLagDAMP is not None
    assert MovingAverage is not None
    assert MovingCovariance is not None
    assert MinMaxScaler is not None
    assert StandardScaler is not None
    assert IncrementalPCA is not None
    assert RandomProjection is not None
    assert Dataset is not None
    assert load is not None
    assert BatchStreamer is not None
    assert NpzStreamer is not None


def test_removed_compatibility_exports_stay_removed() -> None:
    removed = (
        (deep_api, "KitNET"),
        (distance_api, "NETS"),
        (distance_api, "STARE"),
        (graph_api, "StreamSpot"),
        (iforest_api, "MondrianForest"),
        (sketch_api, "LODA"),
        (sketch_api, "RSHash"),
        (svm_api, "GADGETSVM"),
        (dataset_api, "DatasetStreamer"),
    )
    for module, name in removed:
        assert not hasattr(module, name)

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("aberrant.model.stat.uni")


def test_optional_exports_are_excluded_from_wildcard_contracts() -> None:
    assert "Architecture" not in base_api.__all__
    assert "Autoencoder" not in deep_api.__all__
