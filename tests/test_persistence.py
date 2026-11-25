"""Pickles restore only in the aberrant version that saved them."""

import copy
import pickle

import pytest

import aberrant.base._parameters as parameters
from aberrant import __version__
from aberrant.calibration import ScoreCalibrator
from aberrant.compose import GraphNode, GraphPipeline
from aberrant.drift import ADWIN
from aberrant.exceptions import IncompatibleStateError
from aberrant.model.sketch import RSHash
from aberrant.transform import StandardScaler

_EVENTS = [{"x": float(index % 7), "y": float(index % 3)} for index in range(40)]


def _trained() -> RSHash:
    model = RSHash(seed=1)
    for event in _EVENTS:
        model.learn_one(event)
    model.score_one({"x": 9.0, "y": 9.0})
    return model


def test_a_pickle_carries_its_version_and_drops_caches() -> None:
    state = _trained().__getstate__()

    assert state[parameters.VERSION_KEY] == __version__
    assert state["_scored"] is None


@pytest.mark.parametrize(
    "component",
    [
        _trained(),
        StandardScaler() | _trained(),
        GraphPipeline([GraphNode("scaled", StandardScaler())], _trained(), "scaled"),
        ADWIN(),
        ScoreCalibrator("aci", alpha=0.1),
    ],
    ids=["model", "pipeline", "graph", "drift", "calibrator"],
)
def test_another_version_cannot_restore_a_pickle(
    component: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved = pickle.dumps(component)
    monkeypatch.setattr(parameters, "__version__", "2.0.1")

    with pytest.raises(
        IncompatibleStateError, match=rf"saved by aberrant {__version__}"
    ):
        pickle.loads(saved)


def test_a_pickle_from_before_the_version_stamp_is_rejected() -> None:
    model = RSHash(seed=1)
    unstamped = model.__getstate__()
    del unstamped[parameters.VERSION_KEY]

    with pytest.raises(IncompatibleStateError, match="a release before 2.0") as error:
        RSHash.__new__(RSHash).__setstate__(unstamped)
    assert (error.value.component, error.value.saved) == ("RSHash", None)


def test_copies_within_one_version_keep_working() -> None:
    model = _trained()
    probe = {"x": 2.0, "y": 1.0}

    assert copy.deepcopy(model).score_one(probe) == model.score_one(probe)
    assert pickle.loads(pickle.dumps(model)).score_one(probe) == model.score_one(probe)
