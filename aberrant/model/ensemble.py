"""Combine several detectors' scores on one scale."""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right, insort
from collections import deque
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from typing import Literal, cast

from aberrant._utils.validation import coerce_integer_count
from aberrant.base.model import BaseModel
from aberrant.base.protocols import ModelProtocol, TransactionalLearnerProtocol
from aberrant.base.readiness import Readiness, ReadinessUnit

Normalization = Literal["rank", "zscore"]
Combination = Literal["mean", "max", "median"]
_NORMALIZATIONS = ("rank", "zscore")
_COMBINATIONS = ("mean", "max", "median")


class _History:
    """A member's last learned scores, in arrival and in sorted order."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.recent: deque[float] = deque()
        self.ordered: list[float] = []

    def add(self, score: float) -> None:
        if len(self.recent) == self.size:
            leaving = self.recent.popleft()
            del self.ordered[bisect_left(self.ordered, leaving)]
        self.recent.append(score)
        insort(self.ordered, score)

    def copy(self) -> _History:
        copied = _History(self.size)
        copied.recent = deque(self.recent)
        copied.ordered = list(self.ordered)
        return copied

    def rank(self, score: float) -> float:
        below = bisect_left(self.ordered, score)
        equal = bisect_right(self.ordered, score) - below
        return (below + equal / 2) / len(self.ordered)

    def zscore(self, score: float) -> float:
        size = len(self.ordered)
        mean = math.fsum(self.ordered) / size
        spread = math.sqrt(
            math.fsum((item - mean) ** 2 for item in self.ordered) / size
        )
        return (score - mean) / spread if spread > 0 else 0.0


class ScoreEnsemble(BaseModel):
    """Combine the scores of several detectors on one scale.

    Every member learns every event. Each member's score is put on a common
    scale against that member's own last ``window_size`` learned scores:
    ``"rank"`` gives the share of them below the score, ties counting half,
    from 0 to 1, and ``"zscore"`` its distance from their mean in standard
    deviations (0 when they are all equal). The normalized scores are then
    combined by ``mean``, ``max`` or ``median``.

    A member's scores enter its history once the member is ready, and the
    ensemble scores 0.0 until every member is ready and every history holds
    ``warm_up`` scores. ``learn_one`` scores each member before it learns; it
    reuses the scores of a ``score_one`` call for the same event.

    Each event costs the members' own work plus O(M W) for M members and
    window W. When a later member fails to learn, the members that support
    ``learning_transaction`` restore their state; others keep what they
    learned.

    Args:
        models: Two or more detectors, used as they are.
        normalize: ``"rank"`` or ``"zscore"``.
        combine: ``"mean"``, ``"max"`` or ``"median"``.
        window_size: Learned scores each member's normalization keeps.
        warm_up: Learned scores each member needs before the ensemble scores,
            at most ``window_size``.
    """

    def __init__(
        self,
        models: Sequence[ModelProtocol],
        *,
        normalize: Normalization = "rank",
        combine: Combination = "mean",
        window_size: int = 1000,
        warm_up: int = 50,
    ) -> None:
        if isinstance(models, str | bytes) or not isinstance(models, Sequence):
            raise TypeError("models must be a sequence of detectors")
        members = list(models)
        if len(members) < 2:
            raise ValueError("models must hold at least two detectors")
        if not all(isinstance(member, ModelProtocol) for member in members):
            raise TypeError("every member must satisfy ModelProtocol")
        if len({id(member) for member in members}) != len(members):
            raise ValueError("models cannot hold the same detector twice")
        if normalize not in _NORMALIZATIONS:
            raise ValueError(f"normalize must be one of {', '.join(_NORMALIZATIONS)}")
        if combine not in _COMBINATIONS:
            raise ValueError(f"combine must be one of {', '.join(_COMBINATIONS)}")
        self.models = members
        self.normalize = normalize
        self.combine = combine
        self.window_size = coerce_integer_count(window_size, label="window_size")
        self.warm_up = coerce_integer_count(warm_up, label="warm_up")
        if self.warm_up > self.window_size:
            raise ValueError("warm_up must be at most window_size")
        self._histories = [_History(self.window_size) for _ in members]
        self._scored: tuple[dict[str, float], list[float | None]] | None = None
        self._seen = 0

    @property
    def readiness(self) -> Readiness:
        """Ready once every member is ready and every history holds
        ``warm_up`` scores: the slowest member's events plus ``warm_up``."""
        states = [member.readiness for member in self.models]
        counted = all(
            state.unit is ReadinessUnit.EVENTS and state.required is not None
            for state in states
        )
        required = (
            max(cast(int, state.required) for state in states) + self.warm_up
            if counted
            else None
        )
        ready = (
            all(state.ready for state in states)
            and min(len(history.recent) for history in self._histories) >= self.warm_up
        )
        return Readiness(ready, self._seen, required)

    def _member_scores(self, x: dict[str, float]) -> list[float | None]:
        if self._scored is not None and self._scored[0] == x:
            return self._scored[1]
        return [
            float(member.score_one(x)) if member.readiness.ready else None
            for member in self.models
        ]

    def score_one(self, x: dict[str, float]) -> float:
        """The combined normalized score, or 0.0 before readiness."""
        scores = self._member_scores(x)
        self._scored = (dict(x), scores)
        if not self.readiness.ready:
            return 0.0
        normalized = []
        for history, score in zip(self._histories, scores, strict=True):
            assert score is not None
            normalized.append(
                history.rank(score)
                if self.normalize == "rank"
                else history.zscore(score)
            )
        return _combined(normalized, self.combine)

    def learn_one(self, x: dict[str, float]) -> None:
        """Record each ready member's score of the event, then learn it."""
        scores = self._member_scores(x)
        self._scored = None
        with ExitStack() as transaction:
            for member in self.models:
                if isinstance(member, TransactionalLearnerProtocol):
                    transaction.enter_context(member.learning_transaction(x))
                member.learn_one(x)
        for history, score in zip(self._histories, scores, strict=True):
            if score is not None:
                history.add(score)
        self._seen += 1

    @contextmanager
    def learning_transaction(self, x: dict[str, float]) -> Iterator[None]:
        """Restore every member that supports it, and the score histories."""
        histories = [history.copy() for history in self._histories]
        seen = self._seen
        with ExitStack() as members:
            for member in self.models:
                if isinstance(member, TransactionalLearnerProtocol):
                    members.enter_context(member.learning_transaction(x))
            try:
                yield
            except BaseException:
                self._histories, self._seen = histories, seen
                self._scored = None
                raise


def _combined(scores: list[float], combine: str) -> float:
    if combine == "mean":
        return math.fsum(scores) / len(scores)
    if combine == "max":
        return max(scores)
    ordered = sorted(scores)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2
