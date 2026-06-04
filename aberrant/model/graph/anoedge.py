"""AnoEdge-L detector for online anomaly detection in dynamic edge streams."""

from __future__ import annotations

import hashlib

import numpy as np

from aberrant.base.model import BaseModel


class AnoEdgeL(BaseModel):
    """
    AnoEdge-L style detector for dynamic graph edge streams.

    The detector maps source and destination node identifiers into multiple
    fixed-size sketch planes and evaluates each candidate edge using local
    neighborhood density around its hashed cell. The final score is the median
    of per-plane rarity-density scores. It is a local-density approximation and
    does not implement the dense-submatrix search used by the paper's AnoEdge-L.

    Notes:
    - Scores are continuous and non-negative.
    - With ``normalize_score=True``, scores are squashed to ``[0, 1)``.
    - State is bounded by ``num_hashes * count_min_rows * count_min_cols``.

    References:
        Bhatia, S., Wadhwa, M., Kawaguchi, K., Shah, N., Yu, P. S., &
        Hooi, B. (2023). Sketch-Based Anomaly Detection in Streaming Graphs.
        Original implementation: https://github.com/Stream-AD/AnoGraph
    """

    @staticmethod
    def _validate_required_name(value: str, label: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} must be a non-empty string")
        return value

    @staticmethod
    def _validate_optional_name(value: str | None, label: str) -> str | None:
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"{label} must be a non-empty string or None")
        return value

    @staticmethod
    def _validate_hyperparameters(
        *,
        count_min_rows: int,
        count_min_cols: int,
        num_hashes: int,
        local_radius: int,
        time_decay_factor: float,
        warm_up_samples: int,
        normalize_score: bool,
        predict_threshold: float,
        eps: float,
    ) -> None:
        if count_min_rows <= 0:
            raise ValueError("count_min_rows must be positive")
        if count_min_cols <= 0:
            raise ValueError("count_min_cols must be positive")
        if num_hashes <= 0:
            raise ValueError("num_hashes must be positive")
        if local_radius < 0:
            raise ValueError("local_radius must be non-negative")
        if not (0.0 < time_decay_factor <= 1.0):
            raise ValueError("time_decay_factor must be in (0, 1]")
        if warm_up_samples <= 0:
            raise ValueError("warm_up_samples must be positive")
        if normalize_score:
            if not (0.0 <= predict_threshold <= 1.0):
                raise ValueError(
                    "predict_threshold must be in [0, 1] when normalize_score=True"
                )
        elif predict_threshold < 0.0:
            raise ValueError(
                "predict_threshold must be non-negative when normalize_score=False"
            )
        if eps <= 0.0:
            raise ValueError("eps must be positive")

    def __init__(
        self,
        source_key: str = "src",
        destination_key: str = "dst",
        time_key: str | None = "t",
        count_min_rows: int = 256,
        count_min_cols: int = 256,
        num_hashes: int = 4,
        local_radius: int = 2,
        time_decay_factor: float = 1.0,
        warm_up_samples: int = 128,
        normalize_score: bool = False,
        predict_threshold: float = 0.5,
        eps: float = 1e-9,
        seed: int | None = None,
    ) -> None:
        self.source_key = self._validate_required_name(source_key, "source_key")
        self.destination_key = self._validate_required_name(
            destination_key, "destination_key"
        )
        self.time_key = self._validate_optional_name(time_key, "time_key")

        if self.source_key == self.destination_key:
            raise ValueError("source_key and destination_key must be different")
        if self.time_key is not None and self.time_key in (
            self.source_key,
            self.destination_key,
        ):
            raise ValueError(
                "time_key must be different from source_key and destination_key"
            )
        self._validate_hyperparameters(
            count_min_rows=count_min_rows,
            count_min_cols=count_min_cols,
            num_hashes=num_hashes,
            local_radius=local_radius,
            time_decay_factor=time_decay_factor,
            warm_up_samples=warm_up_samples,
            normalize_score=normalize_score,
            predict_threshold=predict_threshold,
            eps=eps,
        )

        self.count_min_rows = count_min_rows
        self.count_min_cols = count_min_cols
        self.num_hashes = num_hashes
        self.local_radius = local_radius
        self.time_decay_factor = time_decay_factor
        self.warm_up_samples = warm_up_samples
        self.normalize_score = normalize_score
        self.predict_threshold = predict_threshold
        self.eps = eps
        self.seed = seed

        self._reset_state()

    def _reset_state(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        max_uint64 = int(np.iinfo(np.uint64).max)
        row_salts = self._rng.integers(
            low=0,
            high=max_uint64,
            size=self.num_hashes,
            dtype=np.uint64,
        )
        col_salts = self._rng.integers(
            low=0,
            high=max_uint64,
            size=self.num_hashes,
            dtype=np.uint64,
        )
        self._row_keys = tuple(
            int(salt).to_bytes(16, byteorder="little", signed=False)
            for salt in row_salts
        )
        self._col_keys = tuple(
            int(salt).to_bytes(16, byteorder="little", signed=False)
            for salt in col_salts
        )

        self._sketch = np.zeros(
            (self.num_hashes, self.count_min_rows, self.count_min_cols),
            dtype=np.float64,
        )
        self._row_mass = np.zeros((self.num_hashes, self.count_min_rows), dtype=np.float64)
        self._col_mass = np.zeros((self.num_hashes, self.count_min_cols), dtype=np.float64)
        self._total_mass = np.zeros(self.num_hashes, dtype=np.float64)

        self._current_bucket: int | None = None
        self._arrival_index = 0
        self._samples_seen = 0

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
        self._sketch *= decay
        self._row_mass *= decay
        self._col_mass *= decay
        self._total_mass *= decay
        self._current_bucket = bucket

    def _source_payload(self, src: float) -> bytes:
        return b"s" + np.float64(src).tobytes()

    def _destination_payload(self, dst: float) -> bytes:
        return b"d" + np.float64(dst).tobytes()

    def _hash_index(self, payload: bytes, key: bytes, size: int) -> int:
        digest = hashlib.blake2b(payload, digest_size=8, key=key).digest()
        return int.from_bytes(digest, byteorder="little", signed=False) % size

    def _hashed_cells(self, src: float, dst: float) -> list[tuple[int, int, int]]:
        src_payload = self._source_payload(src)
        dst_payload = self._destination_payload(dst)
        cells: list[tuple[int, int, int]] = []

        for h, (row_key, col_key) in enumerate(
            zip(self._row_keys, self._col_keys, strict=True)
        ):
            row = self._hash_index(src_payload, row_key, self.count_min_rows)
            col = self._hash_index(dst_payload, col_key, self.count_min_cols)
            cells.append((h, row, col))
        return cells

    def _local_density(self, h: int, row: int, col: int) -> float:
        row_start = max(0, row - self.local_radius)
        row_stop = min(self.count_min_rows, row + self.local_radius + 1)
        col_start = max(0, col - self.local_radius)
        col_stop = min(self.count_min_cols, col + self.local_radius + 1)

        area = float((row_stop - row_start) * (col_stop - col_start))
        if area <= 0.0:
            return 0.0

        window_sum = float(
            np.sum(
                self._sketch[h, row_start:row_stop, col_start:col_stop],
                dtype=np.float64,
            )
        )
        return (window_sum + 1.0) / area

    def _score_hashed_cell(self, h: int, row: int, col: int) -> float:
        candidate_cell = self._sketch[h, row, col] + 1.0
        if candidate_cell <= 0.0:
            return 0.0

        local_density = self._local_density(h, row, col)
        total_mass = self._total_mass[h]
        global_density = (total_mass + 1.0) / float(
            self.count_min_rows * self.count_min_cols
        )

        row_mass = self._row_mass[h, row] + 1.0
        col_mass = self._col_mass[h, col] + 1.0
        marginal_expectation = (row_mass * col_mass) / (total_mass + 1.0)

        density_ratio = local_density / (global_density + self.eps)
        pair_surprise = marginal_expectation / (candidate_cell + self.eps)
        node_novelty = 1.0 / np.sqrt(row_mass * col_mass)

        boosted_surprise = float(
            np.log1p(max(density_ratio, 0.0)) * np.log1p(max(pair_surprise, 0.0))
        )
        score = boosted_surprise + node_novelty
        if not np.isfinite(score):
            return 0.0
        return float(max(score, 0.0))

    def learn_one(self, x: dict[str, float]) -> None:
        """Update detector state with one sample."""
        bucket, src, dst = self._prepare_sample(x)
        self._rollover_if_needed(bucket)

        for h, row, col in self._hashed_cells(src, dst):
            self._sketch[h, row, col] += 1.0
            self._row_mass[h, row] += 1.0
            self._col_mass[h, col] += 1.0
            self._total_mass[h] += 1.0

        self._samples_seen += 1
        if self.time_key is None:
            self._arrival_index += 1

    def score_one(self, x: dict[str, float]) -> float:
        """Compute anomaly score for one sample."""
        bucket, src, dst = self._prepare_sample(x)
        self._rollover_if_needed(bucket)

        if self._samples_seen < self.warm_up_samples:
            return 0.0

        scores = [
            self._score_hashed_cell(h, row, col)
            for h, row, col in self._hashed_cells(src, dst)
        ]
        if not scores:
            return 0.0

        raw_score = float(np.median(np.asarray(scores, dtype=np.float64)))
        if not np.isfinite(raw_score):
            raw_score = 0.0
        raw_score = max(raw_score, 0.0)

        if self.normalize_score:
            return float(raw_score / (1.0 + raw_score))
        return raw_score

    def predict_one(self, x: dict[str, float]) -> int:
        """Return binary anomaly prediction using ``predict_threshold``."""
        return int(self.score_one(x) >= self.predict_threshold)

    def __repr__(self) -> str:
        return (
            "AnoEdgeL("
            f"source_key={self.source_key!r}, destination_key={self.destination_key!r}, "
            f"time_key={self.time_key!r}, count_min_rows={self.count_min_rows}, "
            f"count_min_cols={self.count_min_cols}, num_hashes={self.num_hashes}, "
            f"local_radius={self.local_radius}, time_decay_factor={self.time_decay_factor}, "
            f"warm_up_samples={self.warm_up_samples}, "
            f"normalize_score={self.normalize_score}, "
            f"predict_threshold={self.predict_threshold}, eps={self.eps}, "
            f"seed={self.seed}, samples_seen={self._samples_seen})"
        )
