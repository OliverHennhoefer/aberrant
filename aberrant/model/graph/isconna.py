"""ISCONNA detector for online anomaly detection in dynamic edge streams."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from aberrant.base.model import BaseModel
from aberrant.utils.validation import EdgeEventBoundary

# Width and gap occurrence counters stop growing at the signed int64 maximum.
_COUNTER_LIMIT = int(np.iinfo(np.int64).max) - 1


@dataclass
class _CMSGroup:
    """The frequency, width, and gap sketches used by the author implementation."""

    busy_current: np.ndarray
    busy_previous: np.ndarray
    width_time: np.ndarray
    gap_time: np.ndarray
    frequency_current: np.ndarray
    frequency_accumulated: np.ndarray
    width_current: np.ndarray
    width_accumulated: np.ndarray
    gap_current: np.ndarray
    gap_accumulated: np.ndarray

    @classmethod
    def create(cls, rows: int, cols: int) -> _CMSGroup:
        shape = (rows, cols)
        return cls(
            busy_current=np.zeros(shape, dtype=np.bool_),
            busy_previous=np.zeros(shape, dtype=np.bool_),
            width_time=np.ones(shape, dtype=np.int64),
            gap_time=np.ones(shape, dtype=np.int64),
            frequency_current=np.zeros(shape, dtype=np.float64),
            frequency_accumulated=np.zeros(shape, dtype=np.float64),
            width_current=np.zeros(shape, dtype=np.float64),
            width_accumulated=np.zeros(shape, dtype=np.float64),
            gap_current=np.zeros(shape, dtype=np.float64),
            gap_accumulated=np.zeros(shape, dtype=np.float64),
        )

    def cells(self, indices: np.ndarray) -> list[_Cell]:
        """Copy the queried cell of every row as plain Python numbers."""
        return [_Cell(self, row, column) for row, column in enumerate(indices.tolist())]

    def store(self, indices: np.ndarray, cells: list[_Cell]) -> None:
        """Write cells back to the queried cell of every row."""
        for row, column in enumerate(indices.tolist()):
            cells[row].write(self, row, column)


class _Cell:
    """
    One sketch cell as plain Python numbers.

    A query touches one cell per sketch row, so its transitions run on plain
    numbers, which round exactly like the arrays they are read from.
    """

    __slots__ = (
        "busy_current",
        "busy_previous",
        "frequency_accumulated",
        "frequency_current",
        "gap_accumulated",
        "gap_current",
        "gap_time",
        "width_accumulated",
        "width_current",
        "width_time",
    )

    def __init__(self, group: _CMSGroup, row: int, column: int) -> None:
        self.busy_current: bool = group.busy_current.item(row, column)
        self.busy_previous: bool = group.busy_previous.item(row, column)
        self.width_time: int = group.width_time.item(row, column)
        self.gap_time: int = group.gap_time.item(row, column)
        self.frequency_current: float = group.frequency_current.item(row, column)
        self.frequency_accumulated: float = group.frequency_accumulated.item(
            row, column
        )
        self.width_current: float = group.width_current.item(row, column)
        self.width_accumulated: float = group.width_accumulated.item(row, column)
        self.gap_current: float = group.gap_current.item(row, column)
        self.gap_accumulated: float = group.gap_accumulated.item(row, column)

    def write(self, group: _CMSGroup, row: int, column: int) -> None:
        """Write the cell's values to one cell of a sketch."""
        group.busy_current[row, column] = self.busy_current
        group.busy_previous[row, column] = self.busy_previous
        group.width_time[row, column] = self.width_time
        group.gap_time[row, column] = self.gap_time
        group.frequency_current[row, column] = self.frequency_current
        group.frequency_accumulated[row, column] = self.frequency_accumulated
        group.width_current[row, column] = self.width_current
        group.width_accumulated[row, column] = self.width_accumulated
        group.gap_current[row, column] = self.gap_current
        group.gap_accumulated[row, column] = self.gap_accumulated

    def roll_over(self, decay: float) -> None:
        """Move the cell to a new time bucket, as ``ISCONNA._reset_group`` does."""
        self.frequency_current *= decay
        if not self.busy_current:
            if self.busy_previous:
                self.gap_accumulated += self.gap_current
                self.gap_current *= decay
                self.gap_time = min(self.gap_time, _COUNTER_LIMIT) + 1
            self.gap_current += 1.0
        self.busy_previous = self.busy_current
        self.busy_current = False

    def observe(self, decay: float) -> None:
        """Count one occurrence of the cell's pattern in the current bucket."""
        self.frequency_current += 1.0
        self.frequency_accumulated += 1.0
        if not self.busy_current:
            if not self.busy_previous:
                self.width_accumulated += self.width_current
                self.width_current *= decay
                self.width_time = min(self.width_time, _COUNTER_LIMIT) + 1
            self.width_current += 1.0
        self.busy_current = True


class ISCONNA(BaseModel):
    """
    ISCONNA frequency-and-pattern detector for dynamic graph edge streams.

    This implementation follows the authors' ``ACore``, ``EdgeOnlyCore``, and
    ``EdgeNodeCore`` implementations. Each sketch tracks three signals:

    - frequency within the current timestamp against accumulated frequency,
    - width of consecutive timestamps in which an edge or node is present,
    - gap length of consecutive timestamps in which it is absent.

    Their G-test scores are combined as
    ``frequency**alpha * width**beta * gap**gamma``. With
    ``include_endpoints=True``, the maximum edge/source/destination score is
    used for each component, matching ``EdgeNodeCore``.

    ``score_one`` previews the candidate-inclusive author update without
    mutating state. Calling ``score_one(x)`` followed by ``learn_one(x)``
    therefore produces the same score as the authors' combined update-and-score
    call while preserving the library's score-before-learn convention.

    Notes:
    - Source and destination identifiers must be integer-like numbers.
    - Scores are continuous and non-negative.
    - With ``normalize_score=True``, scores are squashed to ``[0, 1)``.
    - State is bounded by a fixed sketch size.
    - Width and gap occurrence counters saturate at the signed int64 maximum.

    Args:
        source_key: Input field containing the integer-like source identifier.
        destination_key: Input field containing the integer-like destination
            identifier.
        time_key: Input field containing a non-decreasing integer-like time
            bucket. ``None`` assigns a new one-based bucket to every learned
            arrival.
        count_min_rows: Number of independently hashed rows in each sketch.
        count_min_cols: Number of counters per sketch row.
        time_decay_factor: Factor in ``(0, 1]`` applied to current pattern
            counts during bucket transitions.
        alpha: Non-negative exponent of the frequency G-test component.
        beta: Non-negative exponent of the consecutive-width G-test component.
        gamma: Non-negative exponent of the absence-gap G-test component.
        include_endpoints: Combine edge, source, and destination components by
            component-wise maxima. ``False`` scores only edge patterns.
        warm_up_samples: Number of learned edges before scoring begins.
        normalize_score: Apply ``score / (1 + score)`` to the non-negative raw
            combined statistic.
        seed: Seed for model-local sketch hash generation.

    References:
        Liu, R., Bhatia, S., & Hooi, B. (2021). Isconna: Streaming Anomaly
        Detection with Frequency and Patterns.
        https://arxiv.org/abs/2104.01632
        Original implementation: https://github.com/liurui39660/Isconna
    """

    def __init__(
        self,
        source_key: str = "src",
        destination_key: str = "dst",
        time_key: str | None = "t",
        count_min_rows: int = 2,
        count_min_cols: int = 3000,
        time_decay_factor: float = 0.7,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.5,
        include_endpoints: bool = True,
        warm_up_samples: int = 0,
        normalize_score: bool = False,
        seed: int | None = None,
    ) -> None:
        if count_min_rows <= 0:
            raise ValueError("count_min_rows must be positive")
        if count_min_cols <= 0:
            raise ValueError("count_min_cols must be positive")
        if not (0.0 < time_decay_factor <= 1.0):
            raise ValueError("time_decay_factor must be in (0, 1]")
        if alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        if beta < 0.0:
            raise ValueError("beta must be non-negative")
        if gamma < 0.0:
            raise ValueError("gamma must be non-negative")
        if warm_up_samples < 0:
            raise ValueError("warm_up_samples must be non-negative")

        self.source_key = source_key
        self.destination_key = destination_key
        self.time_key = time_key
        self.count_min_rows = count_min_rows
        self.count_min_cols = count_min_cols
        self.time_decay_factor = time_decay_factor
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.include_endpoints = include_endpoints
        self.warm_up_samples = warm_up_samples
        self.normalize_score = normalize_score
        self.seed = seed

        self._reset_state()

    def _reset_state(self) -> None:
        self._boundary = EdgeEventBoundary(
            source_key=self.source_key,
            destination_key=self.destination_key,
            time_key=self.time_key,
        )
        self._rng = np.random.default_rng(self.seed)
        max_uint32 = np.iinfo(np.uint32).max
        self._hash_multipliers = self._rng.integers(
            1,
            max_uint32,
            size=self.count_min_rows,
            dtype=np.uint64,
        )
        self._hash_offsets = self._rng.integers(
            0,
            max_uint32,
            size=self.count_min_rows,
            dtype=np.uint64,
        )
        self._hash_parameters = list(
            zip(
                self._hash_multipliers.tolist(),
                self._hash_offsets.tolist(),
                strict=True,
            )
        )

        self._edge = _CMSGroup.create(self.count_min_rows, self.count_min_cols)
        self._source = (
            _CMSGroup.create(self.count_min_rows, self.count_min_cols)
            if self.include_endpoints
            else None
        )
        self._destination = (
            _CMSGroup.create(self.count_min_rows, self.count_min_cols)
            if self.include_endpoints
            else None
        )

        self._current_bucket: int | None = None
        self._first_bucket: int | None = None
        self._samples_seen = 0

    def reset(self) -> None:
        """Reset learned state while keeping hyperparameters."""
        self._reset_state()

    @property
    def n_samples_seen(self) -> int:
        """Number of observed samples processed via learn_one."""
        return self._samples_seen

    def _indices(self, a: int, b: int) -> np.ndarray:
        # The upstream implementation uses ((a + 347*b) * p + q) % cols.
        combined = int(a) + 347 * int(b)
        return np.asarray(
            [
                (combined * multiplier + offset) % self.count_min_cols
                for multiplier, offset in self._hash_parameters
            ],
            dtype=np.intp,
        )

    @staticmethod
    def _g_test(current: float, accumulated: float, time: int) -> float:
        """Return the G-test statistic used by the original implementation."""
        if current == 0.0 or accumulated == 0.0 or time <= 1:
            return 0.0
        if time > 1e150:
            # Work in log space so a large elapsed-time ratio cannot overflow.
            log_ratio = math.log(current) + math.log(time - 1) - math.log(accumulated)
            return 2.0 * current * abs(log_ratio)
        return float(
            2.0 * current * abs(np.log(current * float(time - 1) / accumulated))
        )

    def _reset_group(self, group: _CMSGroup) -> None:
        group.frequency_current *= self.time_decay_factor

        absent = ~group.busy_current
        gap_continues = absent & group.busy_previous
        group.gap_accumulated[gap_continues] += group.gap_current[gap_continues]
        group.gap_current[gap_continues] *= self.time_decay_factor
        group.gap_time[gap_continues] = (
            np.minimum(group.gap_time[gap_continues], _COUNTER_LIMIT) + 1
        )
        group.gap_current[absent] += 1.0

        group.busy_previous[:] = group.busy_current
        group.busy_current.fill(False)

    def _rollover_for_learning(self, bucket: int) -> None:
        if self._current_bucket is None:
            self._current_bucket = bucket
            self._first_bucket = bucket
            return
        if bucket == self._current_bucket:
            return

        self._reset_group(self._edge)
        if self._source is not None and self._destination is not None:
            self._reset_group(self._source)
            self._reset_group(self._destination)
        self._current_bucket = bucket

    def _time_index(self, bucket: int) -> int:
        if self._first_bucket is None:
            return 1
        return bucket - self._first_bucket + 1

    def _preview_group(
        self,
        group: _CMSGroup,
        indices: np.ndarray,
        *,
        rollover: bool,
        time_index: int,
    ) -> tuple[float, float, float]:
        decay = self.time_decay_factor
        cells = group.cells(indices)
        for cell in cells:
            if rollover:
                cell.roll_over(decay)
            cell.observe(decay)
        return self._group_scores(cells, time_index)

    def _group_scores(
        self, cells: list[_Cell], time_index: int
    ) -> tuple[float, float, float]:
        # min with a key returns the first minimal cell, as np.argmin does.
        width_cell = min(cells, key=lambda cell: cell.width_time)
        gap_cell = min(cells, key=lambda cell: cell.gap_time)
        return (
            self._g_test(
                min(cell.frequency_current for cell in cells),
                min(cell.frequency_accumulated for cell in cells),
                time_index,
            ),
            self._g_test(
                width_cell.width_current,
                width_cell.width_accumulated,
                width_cell.width_time,
            ),
            self._g_test(
                gap_cell.gap_current,
                gap_cell.gap_accumulated,
                gap_cell.gap_time,
            ),
        )

    def _update_group(self, group: _CMSGroup, indices: np.ndarray) -> None:
        cells = group.cells(indices)
        for cell in cells:
            cell.observe(self.time_decay_factor)
        group.store(indices, cells)

    def _component_scores(
        self, bucket: int, src: int, dst: int
    ) -> tuple[float, float, float]:
        rollover = self._current_bucket is not None and bucket > self._current_bucket
        time_index = self._time_index(bucket)
        scores = [
            self._preview_group(
                self._edge,
                self._indices(src, dst),
                rollover=rollover,
                time_index=time_index,
            )
        ]
        if self._source is not None and self._destination is not None:
            scores.append(
                self._preview_group(
                    self._source,
                    self._indices(src, 0),
                    rollover=rollover,
                    time_index=time_index,
                )
            )
            scores.append(
                self._preview_group(
                    self._destination,
                    self._indices(dst, 0),
                    rollover=rollover,
                    time_index=time_index,
                )
            )

        # Each component is the maximum over the edge and its endpoints.
        frequency, width, gap = (
            max(component) for component in zip(*scores, strict=True)
        )
        return frequency, width, gap

    def learn_one(self, x: dict[str, float]) -> None:
        """Update detector state with one sample."""
        event = self._boundary.preview(x)
        bucket, src, dst = event.bucket, event.source, event.destination
        self._rollover_for_learning(bucket)

        self._update_group(self._edge, self._indices(src, dst))
        if self._source is not None and self._destination is not None:
            self._update_group(self._source, self._indices(src, 0))
            self._update_group(self._destination, self._indices(dst, 0))

        self._samples_seen += 1
        self._boundary.commit(event)

    def score_one(self, x: dict[str, float]) -> float:
        """Compute the candidate-inclusive anomaly score without mutating state."""
        event = self._boundary.preview(x)
        bucket, src, dst = event.bucket, event.source, event.destination
        if self._samples_seen < self.warm_up_samples:
            return 0.0

        frequency, width, gap = self._component_scores(bucket, src, dst)
        score = frequency**self.alpha * width**self.beta * gap**self.gamma
        if not np.isfinite(score):
            score = 0.0
        score = float(max(score, 0.0))
        if self.normalize_score:
            return score / (1.0 + score)
        return score

    def __repr__(self) -> str:
        return (
            "ISCONNA("
            f"source_key={self.source_key!r}, destination_key={self.destination_key!r}, "
            f"time_key={self.time_key!r}, count_min_rows={self.count_min_rows}, "
            f"count_min_cols={self.count_min_cols}, "
            f"time_decay_factor={self.time_decay_factor}, alpha={self.alpha}, "
            f"beta={self.beta}, gamma={self.gamma}, "
            f"include_endpoints={self.include_endpoints}, "
            f"warm_up_samples={self.warm_up_samples}, "
            f"normalize_score={self.normalize_score}, seed={self.seed}, "
            f"samples_seen={self._samples_seen})"
        )
