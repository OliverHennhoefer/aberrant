"""ISCONNA detector for online anomaly detection in dynamic edge streams."""

from __future__ import annotations

import hashlib

import numpy as np

from aberrant.base.model import BaseModel


class _CountMinSketch:
    """Fixed-size count-min sketch with deterministic keyed hashing."""

    def __init__(
        self,
        rows: int,
        cols: int,
        rng: np.random.Generator,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.table = np.zeros((rows, cols), dtype=np.float64)
        salts = rng.integers(
            low=0,
            high=np.iinfo(np.uint64).max,
            size=rows,
            dtype=np.uint64,
        )
        self._row_keys = tuple(
            int(salt).to_bytes(16, byteorder="little", signed=False) for salt in salts
        )
        self._row_index = np.arange(rows, dtype=np.intp)

    def _indices(self, payload: bytes) -> np.ndarray:
        indices = np.empty(self.rows, dtype=np.intp)
        for row, row_key in enumerate(self._row_keys):
            digest = hashlib.blake2b(payload, digest_size=8, key=row_key).digest()
            indices[row] = int.from_bytes(digest, byteorder="little", signed=False) % (
                self.cols
            )
        return indices

    def update(self, payload: bytes, value: float = 1.0) -> None:
        """Add value to the hashed bucket of each row."""
        indices = self._indices(payload)
        self.table[self._row_index, indices] += value

    def query(self, payload: bytes) -> float:
        """Return count-min estimate for payload."""
        indices = self._indices(payload)
        return float(np.min(self.table[self._row_index, indices]))

    def decay(self, factor: float) -> None:
        """Multiply all bins by the same factor."""
        self.table *= factor


class ISCONNA(BaseModel):
    """
    ISCONNA-style conditional detector for dynamic graph edge streams.

    The detector consumes one edge at a time with source, destination, and
    optional timestamp keys. It keeps decayed "current" sketches and cumulative
    "total" sketches for edge and endpoint frequencies, and scores each edge
    by contrasting current conditional surprise against historical conditional
    surprise. This is a frequency-only approximation: it does not implement
    ISCONNA's consecutive-presence/absence pattern component or its three-score
    aggregation.

    Notes:
    - Scores are continuous and non-negative.
    - With ``normalize_score=True``, scores are squashed to ``[0, 1)``.
    - State is bounded by a fixed sketch size.

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
        count_min_rows: int = 8,
        count_min_cols: int = 1024,
        time_decay_factor: float = 0.5,
        warm_up_samples: int = 64,
        normalize_score: bool = False,
        eps: float = 1e-9,
        seed: int | None = None,
    ) -> None:
        if not isinstance(source_key, str) or not source_key:
            raise ValueError("source_key must be a non-empty string")
        if not isinstance(destination_key, str) or not destination_key:
            raise ValueError("destination_key must be a non-empty string")
        if source_key == destination_key:
            raise ValueError("source_key and destination_key must be different")
        if time_key is not None and (not isinstance(time_key, str) or not time_key):
            raise ValueError("time_key must be a non-empty string or None")
        if time_key is not None and time_key in (source_key, destination_key):
            raise ValueError(
                "time_key must be different from source_key and destination_key"
            )
        if count_min_rows <= 0:
            raise ValueError("count_min_rows must be positive")
        if count_min_cols <= 0:
            raise ValueError("count_min_cols must be positive")
        if not (0.0 < time_decay_factor <= 1.0):
            raise ValueError("time_decay_factor must be in (0, 1]")
        if warm_up_samples <= 0:
            raise ValueError("warm_up_samples must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.source_key = source_key
        self.destination_key = destination_key
        self.time_key = time_key
        self.count_min_rows = count_min_rows
        self.count_min_cols = count_min_cols
        self.time_decay_factor = time_decay_factor
        self.warm_up_samples = warm_up_samples
        self.normalize_score = normalize_score
        self.eps = eps
        self.seed = seed

        self._reset_state()

    def _reset_state(self) -> None:
        self._rng = np.random.default_rng(self.seed)

        self._edge_current = _CountMinSketch(
            rows=self.count_min_rows,
            cols=self.count_min_cols,
            rng=self._rng,
        )
        self._edge_total = _CountMinSketch(
            rows=self.count_min_rows,
            cols=self.count_min_cols,
            rng=self._rng,
        )
        self._source_current = _CountMinSketch(
            rows=self.count_min_rows,
            cols=self.count_min_cols,
            rng=self._rng,
        )
        self._source_total = _CountMinSketch(
            rows=self.count_min_rows,
            cols=self.count_min_cols,
            rng=self._rng,
        )
        self._destination_current = _CountMinSketch(
            rows=self.count_min_rows,
            cols=self.count_min_cols,
            rng=self._rng,
        )
        self._destination_total = _CountMinSketch(
            rows=self.count_min_rows,
            cols=self.count_min_cols,
            rng=self._rng,
        )

        self._current_bucket: int | None = None
        self._arrival_index = 0
        self._samples_seen = 0
        self._current_edge_mass = 0.0
        self._total_edge_mass = 0.0

    def reset(self) -> None:
        """Reset learned state while keeping hyperparameters."""
        self._reset_state()

    @property
    def n_samples_seen(self) -> int:
        """Number of observed samples processed via learn_one."""
        return self._samples_seen

    def _coerce_numeric(self, value: float, key: str) -> float:
        if not isinstance(value, int | float | np.number):
            raise ValueError(f"Feature '{key}' must be numeric")
        as_float = float(value)
        if not np.isfinite(as_float):
            raise ValueError(f"Feature '{key}' must be finite")
        return as_float

    def _coerce_bucket(self, value: float) -> int:
        as_float = self._coerce_numeric(value, self.time_key or "t")
        as_int = int(round(as_float))
        if not np.isclose(as_float, float(as_int), rtol=0.0, atol=1e-9):
            raise ValueError("Timestamp must be integer-like")
        return as_int

    def _prepare_sample(self, x: dict[str, float]) -> tuple[int, float, float]:
        if not x:
            raise ValueError("Input dictionary cannot be empty")

        if self.source_key not in x:
            raise ValueError(f"Missing source_key '{self.source_key}' in input sample")
        if self.destination_key not in x:
            raise ValueError(
                f"Missing destination_key '{self.destination_key}' in input sample"
            )

        src = self._coerce_numeric(x[self.source_key], self.source_key)
        dst = self._coerce_numeric(x[self.destination_key], self.destination_key)

        if self.time_key is None:
            bucket = self._arrival_index + 1
        else:
            if self.time_key not in x:
                raise ValueError(f"Missing time_key '{self.time_key}' in input sample")
            bucket = self._coerce_bucket(x[self.time_key])

        return bucket, src, dst

    def _rollover_if_needed(self, bucket: int) -> None:
        if self._current_bucket is None:
            self._current_bucket = bucket
            return

        if bucket < self._current_bucket:
            raise ValueError(
                f"Non-monotonic timestamp: received {bucket}, current {self._current_bucket}"
            )
        if bucket == self._current_bucket:
            return

        delta = bucket - self._current_bucket
        decay = self.time_decay_factor**delta

        self._edge_current.decay(decay)
        self._source_current.decay(decay)
        self._destination_current.decay(decay)
        self._current_edge_mass *= decay
        self._current_bucket = bucket

    def _source_payload(self, src: float) -> bytes:
        return b"s" + np.float64(src).tobytes()

    def _destination_payload(self, dst: float) -> bytes:
        return b"d" + np.float64(dst).tobytes()

    def _edge_payload(self, src: float, dst: float) -> bytes:
        return b"e" + np.float64(src).tobytes() + np.float64(dst).tobytes()

    def _raw_score(self, src: float, dst: float) -> float:
        edge_payload = self._edge_payload(src, dst)
        source_payload = self._source_payload(src)
        destination_payload = self._destination_payload(dst)

        current_edge = self._edge_current.query(edge_payload) + 1.0
        total_edge = self._edge_total.query(edge_payload) + 1.0
        current_source = self._source_current.query(source_payload) + 1.0
        current_destination = self._destination_current.query(destination_payload) + 1.0
        total_source = self._source_total.query(source_payload) + 1.0
        total_destination = self._destination_total.query(destination_payload) + 1.0

        current_mass = max(self._current_edge_mass + 1.0, 1.0)
        total_mass = max(self._total_edge_mass + 1.0, 1.0)

        expected_current = (current_source * current_destination) / current_mass
        expected_total = (total_source * total_destination) / total_mass

        conditional_rarity = expected_total / (total_edge + self.eps)
        novelty = 1.0 / (total_edge + self.eps)
        burst = (current_edge + self.eps) / (expected_current + self.eps)

        raw = (conditional_rarity + novelty) * burst
        if not np.isfinite(raw):
            return 0.0
        return float(max(raw, 0.0))

    def learn_one(self, x: dict[str, float]) -> None:
        """Update detector state with one sample."""
        bucket, src, dst = self._prepare_sample(x)
        self._rollover_if_needed(bucket)

        edge_payload = self._edge_payload(src, dst)
        source_payload = self._source_payload(src)
        destination_payload = self._destination_payload(dst)

        self._edge_current.update(edge_payload, 1.0)
        self._edge_total.update(edge_payload, 1.0)
        self._source_current.update(source_payload, 1.0)
        self._source_total.update(source_payload, 1.0)
        self._destination_current.update(destination_payload, 1.0)
        self._destination_total.update(destination_payload, 1.0)

        self._samples_seen += 1
        self._current_edge_mass += 1.0
        self._total_edge_mass += 1.0
        if self.time_key is None:
            self._arrival_index += 1

    def score_one(self, x: dict[str, float]) -> float:
        """Compute anomaly score for one sample."""
        bucket, src, dst = self._prepare_sample(x)
        self._rollover_if_needed(bucket)

        if self._samples_seen < self.warm_up_samples:
            return 0.0

        score = self._raw_score(src, dst)
        if self.normalize_score:
            return float(score / (1.0 + score))
        return score

    def __repr__(self) -> str:
        return (
            "ISCONNA("
            f"source_key={self.source_key!r}, destination_key={self.destination_key!r}, "
            f"time_key={self.time_key!r}, count_min_rows={self.count_min_rows}, "
            f"count_min_cols={self.count_min_cols}, "
            f"time_decay_factor={self.time_decay_factor}, "
            f"warm_up_samples={self.warm_up_samples}, "
            f"normalize_score={self.normalize_score}, eps={self.eps}, "
            f"seed={self.seed}, samples_seen={self._samples_seen})"
        )
