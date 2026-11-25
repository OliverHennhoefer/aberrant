"""MIDAS detector for online anomaly detection in dynamic edge streams."""

from __future__ import annotations

import hashlib
import math

import numpy as np

from aberrant._utils.validation import EdgeEventBoundary, coerce_integer_count
from aberrant.base.model import BaseModel


class _CountMinSketch:
    """Fixed-size count-min sketch with deterministic keyed hashing."""

    def __init__(
        self,
        rows: int,
        cols: int,
        rng: np.random.Generator | None = None,
        row_keys: tuple[bytes, ...] | None = None,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.table = np.zeros((rows, cols), dtype=np.float64)
        if row_keys is not None:
            self._row_keys = row_keys
        elif rng is not None:
            salts = rng.integers(
                low=0,
                high=np.iinfo(np.uint64).max,
                size=rows,
                dtype=np.uint64,
            )
            self._row_keys = tuple(
                int(salt).to_bytes(16, byteorder="little", signed=False)
                for salt in salts
            )
        else:
            raise ValueError("rng or row_keys is required")
        self._row_index = np.arange(rows, dtype=np.intp)

    def empty_copy(self) -> _CountMinSketch:
        """Return an empty sketch with the same hash functions."""
        return _CountMinSketch(self.rows, self.cols, row_keys=self._row_keys)

    def indices(self, payload: bytes) -> np.ndarray:
        """Return the payload bucket in every row."""
        indices = np.empty(self.rows, dtype=np.intp)
        for row, row_key in enumerate(self._row_keys):
            digest = hashlib.blake2b(payload, digest_size=8, key=row_key).digest()
            indices[row] = int.from_bytes(digest, byteorder="little", signed=False) % (
                self.cols
            )
        return indices

    def update_indices(self, indices: np.ndarray, value: float = 1.0) -> None:
        """Add value at precomputed row indices."""
        self.table[self._row_index, indices] += value

    def query_indices(self, indices: np.ndarray) -> float:
        """Return the count-min estimate at precomputed row indices."""
        return float(np.min(self.table[self._row_index, indices]))

    def clear(self) -> None:
        """Set all bins to zero."""
        self.table.fill(0.0)

    def decay(self, factor: float) -> None:
        """Multiply all bins by factor."""
        self.table *= factor


class MIDAS(BaseModel):
    """
    MIDAS or MIDAS-R detector for dynamic edge streams.

    ``use_relational=False`` follows the authors' ``NormalCore``: the current
    edge sketch is cleared at each new timestamp and the candidate-inclusive
    edge count is scored against its cumulative count.

    ``use_relational=True`` follows ``RelationalCore`` (MIDAS-R): current edge,
    source, and destination sketches are decayed at each new timestamp, and the
    final score is the maximum of their three published chi-square scores.

    ``score_one`` previews rollover and candidate insertion without mutating
    state. Calling ``score_one(x)`` followed by ``learn_one(x)`` therefore
    matches the authors' combined update-and-score operation under this
    library's score-before-learn convention.

    Notes:
    - Source and destination identifiers must be integer-like numbers.
    - Scores are continuous and non-negative.
    - State is bounded by fixed-size sketches.
    - Raw statistics larger than float64 can represent saturate at its maximum.

    Args:
        source_key: Input field containing the integer-like source identifier.
        destination_key: Input field containing the integer-like destination
            identifier.
        time_key: Input field containing a non-decreasing integer-like time
            bucket. ``None`` assigns a new one-based bucket to every learned
            arrival.
        n_hashes: Number of independently hashed rows in each count-min
            sketch.
        sketch_width: Number of counters per sketch row.
        time_decay_factor: Factor in ``(0, 1]`` applied to current edge and
            endpoint sketches at bucket changes in relational mode. NormalCore
            clears its current edge sketch instead.
        warm_up: Number of learned edges before scoring begins.
        use_relational: Use MIDAS-R endpoint sketches and decayed current
            counts. ``False`` selects the edge-only NormalCore.
        seed: Seed for model-local sketch hash generation.

    References:
        Bhatia, S., Hooi, B., Yoon, M., Shin, K., & Faloutsos, C. (2020).
        MIDAS: Microcluster-Based Detector of Anomalies in Edge Streams.
        https://ojs.aaai.org/index.php/AAAI/article/view/5724
        Original implementation: https://github.com/Stream-AD/MIDAS
    """

    def __init__(
        self,
        *,
        source_key: str = "src",
        destination_key: str = "dst",
        time_key: str | None = "t",
        n_hashes: int = 2,
        sketch_width: int = 1024,
        time_decay_factor: float = 0.5,
        warm_up: int = 0,
        use_relational: bool = True,
        seed: int | None = None,
    ) -> None:
        n_hashes = coerce_integer_count(n_hashes, label="n_hashes", minimum=1)
        sketch_width = coerce_integer_count(
            sketch_width, label="sketch_width", minimum=1
        )
        warm_up = coerce_integer_count(warm_up, label="warm_up", minimum=0)
        if not (0.0 < time_decay_factor <= 1.0):
            raise ValueError("time_decay_factor must be in (0, 1]")

        self.source_key = source_key
        self.destination_key = destination_key
        self.time_key = time_key
        self.n_hashes = n_hashes
        self.sketch_width = sketch_width
        self.time_decay_factor = time_decay_factor
        self.warm_up = warm_up
        self.use_relational = use_relational
        self.seed = seed

        self._reset_state()

    def _new_sketch_pair(self) -> tuple[_CountMinSketch, _CountMinSketch]:
        current = _CountMinSketch(
            rows=self.n_hashes,
            cols=self.sketch_width,
            rng=self._rng,
        )
        return current, current.empty_copy()

    def _reset_state(self) -> None:
        self._boundary = EdgeEventBoundary(
            source_key=self.source_key,
            destination_key=self.destination_key,
            time_key=self.time_key,
        )
        self._rng = np.random.default_rng(self.seed)
        self._edge_current, self._edge_total = self._new_sketch_pair()

        self._source_current: _CountMinSketch | None = None
        self._source_total: _CountMinSketch | None = None
        self._destination_current: _CountMinSketch | None = None
        self._destination_total: _CountMinSketch | None = None
        if self.use_relational:
            self._source_current, self._source_total = self._new_sketch_pair()
            self._destination_current, self._destination_total = self._new_sketch_pair()

        self._current_bucket: int | None = None
        self._first_bucket: int | None = None
        self._samples_seen = 0

    @property
    def n_samples_seen(self) -> int:
        """Number of observed samples processed via learn_one."""
        return self._samples_seen

    @staticmethod
    def _integer_bytes(value: int) -> bytes:
        return int(value).to_bytes((value.bit_length() + 8) // 8, "little", signed=True)

    @staticmethod
    def _source_payload(src: int) -> bytes:
        if not -(1 << 63) <= src < (1 << 63):
            return b"S" + MIDAS._integer_bytes(src)
        return b"s" + int(src).to_bytes(8, byteorder="little", signed=True)

    @staticmethod
    def _destination_payload(dst: int) -> bytes:
        if not -(1 << 63) <= dst < (1 << 63):
            return b"D" + MIDAS._integer_bytes(dst)
        return b"d" + int(dst).to_bytes(8, byteorder="little", signed=True)

    @staticmethod
    def _edge_payload(src: int, dst: int) -> bytes:
        if not (-(1 << 63) <= src < (1 << 63) and -(1 << 63) <= dst < (1 << 63)):
            source = MIDAS._integer_bytes(src)
            destination = MIDAS._integer_bytes(dst)
            return b"E" + str(len(source)).encode("ascii") + b":" + source + destination
        return (
            b"e"
            + int(src).to_bytes(8, byteorder="little", signed=True)
            + int(dst).to_bytes(8, byteorder="little", signed=True)
        )

    def _time_index(self, bucket: int) -> int:
        if self._first_bucket is None:
            return 1
        return bucket - self._first_bucket + 1

    def _rollover_for_learning(self, bucket: int) -> None:
        if self._current_bucket is None:
            self._current_bucket = bucket
            self._first_bucket = bucket
            return
        if bucket == self._current_bucket:
            return

        if self.use_relational:
            self._edge_current.decay(self.time_decay_factor)
            if (
                self._source_current is not None
                and self._destination_current is not None
            ):
                self._source_current.decay(self.time_decay_factor)
                self._destination_current.decay(self.time_decay_factor)
        else:
            self._edge_current.clear()
        self._current_bucket = bucket

    @staticmethod
    def _compute_score(current: float, total: float, time_index: int) -> float:
        """Return the score from ``NormalCore`` and ``RelationalCore``."""
        if total == 0.0 or time_index - 1 == 0:
            return 0.0
        if time_index > 1e150 or current > 1e150 / time_index:
            # The published expression squares a time-scaled residual. That
            # intermediate overflows long before the statistic itself does.
            residual = abs(current - total * (1 / time_index))
            if residual == 0.0:
                return 0.0
            log_score = (
                2.0 * math.log(residual)
                + math.log(time_index)
                + math.log(time_index / (time_index - 1))
                - math.log(total)
            )
            return math.exp(min(log_score, math.log(np.finfo(np.float64).max)))
        return float(
            ((current - total / float(time_index)) * float(time_index)) ** 2
            / (total * float(time_index - 1))
        )

    def _candidate_score(
        self,
        current: _CountMinSketch,
        total: _CountMinSketch,
        payload: bytes,
        *,
        rollover: bool,
        time_index: int,
    ) -> float:
        indices = current.indices(payload)
        current_count = current.query_indices(indices)
        if rollover:
            current_count = (
                current_count * self.time_decay_factor if self.use_relational else 0.0
            )
        current_count += 1.0
        total_count = total.query_indices(indices) + 1.0
        return self._compute_score(current_count, total_count, time_index)

    def _update_pair(
        self,
        current: _CountMinSketch,
        total: _CountMinSketch,
        payload: bytes,
    ) -> None:
        indices = current.indices(payload)
        current.update_indices(indices)
        total.update_indices(indices)

    def learn_one(self, x: dict[str, float]) -> None:
        """Update detector state with one edge."""
        event = self._boundary.preview(x)
        bucket, src, dst = event.bucket, event.source, event.destination
        self._rollover_for_learning(bucket)
        self._update_pair(
            self._edge_current,
            self._edge_total,
            self._edge_payload(src, dst),
        )

        if (
            self._source_current is not None
            and self._source_total is not None
            and self._destination_current is not None
            and self._destination_total is not None
        ):
            self._update_pair(
                self._source_current,
                self._source_total,
                self._source_payload(src),
            )
            self._update_pair(
                self._destination_current,
                self._destination_total,
                self._destination_payload(dst),
            )

        self._samples_seen += 1
        self._boundary.commit(event)

    def score_one(self, x: dict[str, float]) -> float:
        """Preview the candidate-inclusive MIDAS or MIDAS-R score."""
        event = self._boundary.preview(x)
        bucket, src, dst = event.bucket, event.source, event.destination
        if self._samples_seen < self.warm_up:
            return 0.0

        rollover = self._current_bucket is not None and bucket > self._current_bucket
        time_index = self._time_index(bucket)
        scores = [
            self._candidate_score(
                self._edge_current,
                self._edge_total,
                self._edge_payload(src, dst),
                rollover=rollover,
                time_index=time_index,
            )
        ]
        if (
            self._source_current is not None
            and self._source_total is not None
            and self._destination_current is not None
            and self._destination_total is not None
        ):
            scores.append(
                self._candidate_score(
                    self._source_current,
                    self._source_total,
                    self._source_payload(src),
                    rollover=rollover,
                    time_index=time_index,
                )
            )
            scores.append(
                self._candidate_score(
                    self._destination_current,
                    self._destination_total,
                    self._destination_payload(dst),
                    rollover=rollover,
                    time_index=time_index,
                )
            )

        score = max(scores)
        return score
