"""MIDAS detector for online anomaly detection in dynamic edge streams."""

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

    def clear(self) -> None:
        """Set all bins to zero."""
        self.table.fill(0.0)


class MIDAS(BaseModel):
    """
    MIDAS-style detector for anomaly detection in dynamic edge streams.

    MIDAS scores each edge by contrasting its candidate-inclusive count in the
    current time bucket with a historical expectation derived from cumulative
    counts. Optionally, a relational score from source and destination
    frequencies can be fused via ``max``. The edge score follows the MIDAS
    chi-square form; ``use_relational=True`` is a custom endpoint-independence
    extension and is not the published MIDAS-R relational sketch.

    Notes:
    - Scores are continuous and non-negative.
    - With ``normalize_score=True``, scores are squashed to ``[0, 1)``.
    - State is bounded by fixed-size sketches.

    References:
        Bhatia, S., Hooi, B., Yoon, M., Shin, K., & Faloutsos, C. (2020).
        MIDAS: Microcluster-Based Detector of Anomalies in Edge Streams.
        https://ojs.aaai.org/index.php/AAAI/article/view/5724
        Original implementation: https://github.com/Stream-AD/MIDAS
    """

    def __init__(
        self,
        source_key: str = "src",
        destination_key: str = "dst",
        time_key: str | None = "t",
        count_min_rows: int = 4,
        count_min_cols: int = 2048,
        warm_up_samples: int = 128,
        use_relational: bool = True,
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
        if warm_up_samples <= 0:
            raise ValueError("warm_up_samples must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.source_key = source_key
        self.destination_key = destination_key
        self.time_key = time_key
        self.count_min_rows = count_min_rows
        self.count_min_cols = count_min_cols
        self.warm_up_samples = warm_up_samples
        self.use_relational = use_relational
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

        self._source_current: _CountMinSketch | None = None
        self._source_total: _CountMinSketch | None = None
        self._destination_current: _CountMinSketch | None = None
        self._destination_total: _CountMinSketch | None = None
        if self.use_relational:
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
        self._bucket_index = 0
        self._samples_seen = 0
        self._arrival_index = 0

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
            self._bucket_index = 1
            return

        if bucket < self._current_bucket:
            raise ValueError(
                f"Non-monotonic timestamp: received {bucket}, current {self._current_bucket}"
            )
        if bucket == self._current_bucket:
            return

        delta = bucket - self._current_bucket
        self._edge_current.clear()
        if self._source_current is not None and self._destination_current is not None:
            self._source_current.clear()
            self._destination_current.clear()

        self._current_bucket = bucket
        self._bucket_index += delta

    def _source_payload(self, src: float) -> bytes:
        return b"s" + np.float64(src).tobytes()

    def _destination_payload(self, dst: float) -> bytes:
        return b"d" + np.float64(dst).tobytes()

    def _edge_payload(self, src: float, dst: float) -> bytes:
        return b"e" + np.float64(src).tobytes() + np.float64(dst).tobytes()

    def _edge_score(self, src: float, dst: float) -> tuple[float, float]:
        edge_payload = self._edge_payload(src, dst)
        current_count = self._edge_current.query(edge_payload) + 1.0
        total_count = self._edge_total.query(edge_payload) + 1.0

        expected = total_count / max(float(self._bucket_index), 1.0)
        score = ((current_count - expected) ** 2) / (expected + self.eps)
        if not np.isfinite(score):
            return 0.0, current_count
        return float(max(score, 0.0)), current_count

    def _relational_score(self, src: float, dst: float, current_count: float) -> float:
        if (
            self._source_total is None
            or self._destination_total is None
            or not self.use_relational
        ):
            return 0.0

        source_payload = self._source_payload(src)
        destination_payload = self._destination_payload(dst)
        source_total = self._source_total.query(source_payload) + 1.0
        destination_total = self._destination_total.query(destination_payload) + 1.0

        total_mass = max(float(self._samples_seen + 1), 1.0)
        expected_relational = (source_total * destination_total) / (
            total_mass + self.eps
        )
        score = ((current_count - expected_relational) ** 2) / (
            expected_relational + self.eps
        )
        if not np.isfinite(score):
            return 0.0
        return float(max(score, 0.0))

    def learn_one(self, x: dict[str, float]) -> None:
        """Update detector state with one sample."""
        bucket, src, dst = self._prepare_sample(x)
        self._rollover_if_needed(bucket)

        edge_payload = self._edge_payload(src, dst)
        self._edge_current.update(edge_payload, 1.0)
        self._edge_total.update(edge_payload, 1.0)

        if (
            self._source_current is not None
            and self._source_total is not None
            and self._destination_current is not None
            and self._destination_total is not None
        ):
            source_payload = self._source_payload(src)
            destination_payload = self._destination_payload(dst)
            self._source_current.update(source_payload, 1.0)
            self._source_total.update(source_payload, 1.0)
            self._destination_current.update(destination_payload, 1.0)
            self._destination_total.update(destination_payload, 1.0)

        self._samples_seen += 1
        if self.time_key is None:
            self._arrival_index += 1

    def score_one(self, x: dict[str, float]) -> float:
        """Compute anomaly score for one sample."""
        bucket, src, dst = self._prepare_sample(x)
        self._rollover_if_needed(bucket)

        if self._samples_seen < self.warm_up_samples or self._bucket_index < 2:
            return 0.0

        edge_score, current_count = self._edge_score(src, dst)
        raw_score = edge_score

        if self.use_relational:
            relational_score = self._relational_score(src, dst, current_count)
            raw_score = max(raw_score, relational_score)

        if self.normalize_score:
            return float(raw_score / (1.0 + raw_score))
        return raw_score

    def __repr__(self) -> str:
        return (
            "MIDAS("
            f"source_key={self.source_key!r}, destination_key={self.destination_key!r}, "
            f"time_key={self.time_key!r}, count_min_rows={self.count_min_rows}, "
            f"count_min_cols={self.count_min_cols}, warm_up_samples={self.warm_up_samples}, "
            f"use_relational={self.use_relational}, normalize_score={self.normalize_score}, "
            f"eps={self.eps}, seed={self.seed}, samples_seen={self._samples_seen})"
        )
