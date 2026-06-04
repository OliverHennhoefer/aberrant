"""Public API import contract tests for base install."""

from aberrant import __version__
from aberrant.base import BaseModel, BaseTransformer, Pipeline
from aberrant.drift import ADWIN, KSWIN, PageHinkley
from aberrant.model import NullModel, QuantileThreshold, RandomModel, ThresholdModel
from aberrant.model.distance import (
    KNN,
    NETS,
    STARE,
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
    StreamSpot,
)
from aberrant.model.iforest import (
    ASDIsolationForest,
    HalfSpaceTrees,
    MondrianForest,
    MondrianIsolationForest,
    OnlineIsolationForest,
    RandomCutForest,
    StreamRandomHistogramForest,
    XStream,
)
from aberrant.model.sketch import LODA, MStream, RSHash, StreamingLODA, StreamingRSHash
from aberrant.model.stat import MovingAverage, MovingCovariance
from aberrant.model.svm import (
    GADGETSVM,
    GraphGatedOneClassSVM,
    IncrementalOneClassSVMAdaptiveKernel,
)
from aberrant.stream import Dataset, load
from aberrant.stream.dataset import BatchStreamer, DatasetStreamer
from aberrant.transform.preprocessing import MinMaxScaler, StandardScaler
from aberrant.transform.projection import IncrementalPCA, RandomProjection


def test_public_imports_base_smoke() -> None:
    assert isinstance(__version__, str)
    assert BaseModel is not None
    assert BaseTransformer is not None
    assert Pipeline is not None
    assert ADWIN is not None
    assert KSWIN is not None
    assert PageHinkley is not None
    assert NullModel is not None
    assert RandomModel is not None
    assert ThresholdModel is not None
    assert QuantileThreshold is not None
    assert KNN is not None
    assert LocalOutlierFactor is not None
    assert NETS is CellNeighborhoodDetector
    assert SDOStream is not None
    assert STARE is StationaryRegionNeighborDetector
    assert AnoEdgeL is not None
    assert ISCONNA is not None
    assert MIDAS is not None
    assert StreamSpot is SignedGraphSketchDetector
    assert ASDIsolationForest is not None
    assert HalfSpaceTrees is not None
    assert MondrianForest is MondrianIsolationForest
    assert OnlineIsolationForest is not None
    assert RandomCutForest is not None
    assert StreamRandomHistogramForest is not None
    assert XStream is not None
    assert LODA is StreamingLODA
    assert MStream is not None
    assert RSHash is StreamingRSHash
    assert IncrementalOneClassSVMAdaptiveKernel is not None
    assert GADGETSVM is not None
    assert GADGETSVM is GraphGatedOneClassSVM
    assert MovingAverage is not None
    assert MovingCovariance is not None
    assert MinMaxScaler is not None
    assert StandardScaler is not None
    assert IncrementalPCA is not None
    assert RandomProjection is not None
    assert Dataset is not None
    assert load is not None
    assert BatchStreamer is not None
    assert DatasetStreamer is not None
