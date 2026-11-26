"""ScoreEnsemble: members' scores on one scale, then combined."""

import math
from contextlib import contextmanager

import pytest

from aberrant.base import Readiness
from aberrant.base.model import BaseModel
from aberrant.base.pipeline import Pipeline
from aberrant.catalog import build_detector
from aberrant.exceptions import ConfigurationError
from aberrant.model import NullModel, ScoreEnsemble
from aberrant.model.forest import HalfSpaceTrees
from aberrant.model.sketch import LODA
from aberrant.transform import Imputer


class Echo(BaseModel):
    """Scores an event by one of its features once it learned ``wait`` events."""

    def __init__(self, *, feature: str = "a", wait: int = 0) -> None:
        self.feature = feature
        self.wait = wait
        self.learned = 0
        self.scored = 0

    @property
    def readiness(self) -> Readiness:
        return Readiness.after(self.learned, self.wait)

    def learn_one(self, x: dict[str, float]) -> None:
        self.learned += 1

    def score_one(self, x: dict[str, float]) -> float:
        self.scored += 1
        return x[self.feature]


class Refusing(Echo):
    def learn_one(self, x: dict[str, float]) -> None:
        if x[self.feature] == 99.0:
            raise RuntimeError("refused")
        super().learn_one(x)


class Restoring(Echo):
    @contextmanager
    def learning_transaction(self, x):
        learned = self.learned
        try:
            yield
        except BaseException:
            self.learned = learned
            raise


def learned(ensemble: ScoreEnsemble, values) -> ScoreEnsemble:
    for a, b in values:
        ensemble.learn_one({"a": a, "b": b})
    return ensemble


def test_rank_is_the_share_of_learned_scores_below_ties_counting_half() -> None:
    ensemble = learned(
        ScoreEnsemble([Echo(feature="a"), Echo(feature="b")], window_size=4, warm_up=4),
        [(1.0, 10.0), (2.0, 20.0), (3.0, 30.0), (4.0, 40.0)],
    )
    assert ensemble.score_one({"a": 0.0, "b": 50.0}) == pytest.approx((0 + 1) / 2)
    assert ensemble.score_one({"a": 2.0, "b": 20.0}) == pytest.approx(0.375)


def test_zscore_uses_each_members_own_mean_and_spread() -> None:
    ensemble = learned(
        ScoreEnsemble(
            [Echo(feature="a"), Echo(feature="b")],
            normalize="zscore",
            window_size=2,
            warm_up=2,
        ),
        [(1.0, 5.0), (3.0, 5.0)],
    )
    assert ensemble.score_one({"a": 5.0, "b": 9.0}) == pytest.approx((3.0 + 0.0) / 2)


@pytest.mark.parametrize(
    ("combine", "expected"), [("mean", 0.5), ("max", 1.0), ("median", 0.5)]
)
def test_combines_the_normalized_scores(combine, expected) -> None:
    members = [Echo(feature="a"), Echo(feature="b"), Echo(feature="c")]
    ensemble = ScoreEnsemble(members, combine=combine, window_size=2, warm_up=2)
    for value in (1.0, 2.0):
        ensemble.learn_one({"a": value, "b": value, "c": value})
    assert ensemble.score_one({"a": 0.0, "b": 1.5, "c": 3.0}) == expected


def test_the_window_forgets_the_oldest_score() -> None:
    ensemble = learned(
        ScoreEnsemble([Echo(feature="a"), Echo(feature="b")], window_size=2, warm_up=1),
        [(1.0, 1.0), (5.0, 5.0), (6.0, 6.0)],
    )
    assert ensemble.score_one({"a": 2.0, "b": 2.0}) == 0.0
    assert ensemble.score_one({"a": 5.5, "b": 5.5}) == 0.5


def test_waits_for_the_slowest_member_and_then_warm_up() -> None:
    ensemble = ScoreEnsemble([Echo(wait=0), Echo(wait=3)], window_size=5, warm_up=2)
    assert ensemble.readiness == Readiness(False, 0, 5)
    for step in range(5):
        assert ensemble.score_one({"a": float(step)}) == 0.0
        ensemble.learn_one({"a": float(step)})
    assert ensemble.readiness == Readiness(True, 5, 5)
    assert ensemble.score_one({"a": 10.0}) == 1.0


def test_readiness_has_no_count_when_a_member_has_none() -> None:
    class Unknown(Echo):
        @property
        def readiness(self) -> Readiness:
            return Readiness(self.learned >= 1, self.learned, None)

    ensemble = ScoreEnsemble([Echo(), Unknown()], window_size=2, warm_up=1)
    assert ensemble.readiness.required is None


def test_learn_reuses_the_scores_of_the_same_event() -> None:
    members = [Echo(), Echo()]
    ensemble = ScoreEnsemble(members, window_size=2, warm_up=1)
    ensemble.score_one({"a": 1.0})
    ensemble.learn_one({"a": 1.0})
    assert [member.scored for member in members] == [1, 1]
    ensemble.score_one({"a": 2.0})
    ensemble.learn_one({"a": 3.0})
    assert [member.scored for member in members] == [3, 3]


def test_a_failing_member_restores_the_members_that_can() -> None:
    first, second = Restoring(), Refusing()
    ensemble = learned(
        ScoreEnsemble([first, second], window_size=4, warm_up=1), [(1.0, 1.0)]
    )
    with pytest.raises(RuntimeError):
        ensemble.learn_one({"a": 99.0})
    assert (first.learned, second.learned) == (1, 1)
    assert [list(history.recent) for history in ensemble._histories] == [[1.0]] * 2
    assert ensemble.readiness == Readiness(True, 1, 1)


def test_a_failed_downstream_update_restores_the_histories() -> None:
    plain, restoring = Echo(), Restoring()
    ensemble = learned(
        ScoreEnsemble([plain, restoring], window_size=4, warm_up=1), [(1.0, 1.0)]
    )
    with pytest.raises(RuntimeError), ensemble.learning_transaction({"a": 2.0}):
        ensemble.learn_one({"a": 2.0})
        raise RuntimeError("a later step refused")
    assert (plain.learned, restoring.learned) == (2, 1)
    assert [list(history.recent) for history in ensemble._histories] == [[1.0]] * 2
    assert ensemble.readiness == Readiness(True, 1, 1)


def test_runs_tabular_detectors_behind_a_transformer() -> None:
    ensemble = ScoreEnsemble(
        [LODA(warm_up=10, seed=1), HalfSpaceTrees(window_size=10, seed=2)],
        window_size=20,
        warm_up=5,
    )
    detector = Pipeline(Imputer(), ensemble)
    scores = []
    for t in range(80):
        event = {"x": math.sin(t / 5) / 2 + 0.5, "y": math.cos(t / 5) / 2 + 0.5}
        scores.append(detector.score_one(event))
        detector.learn_one(event)
    assert detector.readiness.ready
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert any(score > 0.0 for score in scores)


def test_the_catalog_builds_nested_members() -> None:
    detector = build_detector(
        {
            "model": {
                "id": "score_ensemble",
                "params": {
                    "models": [
                        {"id": "loda", "params": {"seed": 1}},
                        {"id": "half_space_trees", "params": {"seed": 2}},
                    ],
                    "combine": "max",
                },
            }
        }
    )
    assert isinstance(detector, ScoreEnsemble)
    assert detector.combine == "max"


def test_the_catalog_rejects_members_of_different_event_kinds() -> None:
    with pytest.raises(ConfigurationError, match="same kind of event"):
        build_detector(
            {
                "model": {
                    "id": "score_ensemble",
                    "params": {"models": [{"id": "loda"}, {"id": "midas"}]},
                }
            }
        )


@pytest.mark.parametrize(
    ("models", "params", "error"),
    [
        ("ab", {}, TypeError),
        ([NullModel()], {}, ValueError),
        ([NullModel(), object()], {}, TypeError),
        ("same", {}, ValueError),
        (None, {"normalize": "minmax"}, ValueError),
        (None, {"combine": "sum"}, ValueError),
        (None, {"window_size": 0}, ValueError),
        (None, {"window_size": 5, "warm_up": 6}, ValueError),
    ],
)
def test_rejects_invalid_configuration(models, params, error) -> None:
    if models is None:
        models = [NullModel(), NullModel()]
    elif models == "same":
        shared = NullModel()
        models = [shared, shared]
    with pytest.raises(error):
        ScoreEnsemble(models, **params)
