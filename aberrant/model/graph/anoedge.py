"""AnoEdge-L detector for online anomaly detection in dynamic edge streams."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from aberrant.base.model import BaseModel
from aberrant.utils.validation import EdgeEventBoundary

_MatrixValue = Callable[[int, int], float]


class _DenseSubmatrix:
    """Local dense-submatrix state ported from the authors' ``Submatrix``."""

    def __init__(self, row: int, col: int, value: float = 0.0) -> None:
        self.total = float(value)
        self.row_sums = {row: float(value)}
        self.col_sums = {col: float(value)}

    def copy(self) -> _DenseSubmatrix:
        """Return an independent copy for non-mutating score previews."""
        result = object.__new__(_DenseSubmatrix)
        result.total = self.total
        result.row_sums = self.row_sums.copy()
        result.col_sums = self.col_sums.copy()
        return result

    @property
    def density(self) -> float:
        """Return sum divided by the geometric mean of selected dimensions."""
        return self.total / math.sqrt(len(self.row_sums) * len(self.col_sums))

    def decay(self, factor: float) -> None:
        """Decay the submatrix statistics in place."""
        self.total *= factor
        for row in self.row_sums:
            self.row_sums[row] *= factor
        for col in self.col_sums:
            self.col_sums[col] *= factor

    def _add_row(self, row: int, value: float, matrix: _MatrixValue) -> None:
        self.row_sums[row] = value
        for col in self.col_sums:
            self.col_sums[col] += matrix(row, col)

    def _add_col(self, col: int, value: float, matrix: _MatrixValue) -> None:
        self.col_sums[col] = value
        for row in self.row_sums:
            self.row_sums[row] += matrix(row, col)

    def _del_row(self, row: int, matrix: _MatrixValue) -> None:
        del self.row_sums[row]
        for col in self.col_sums:
            self.col_sums[col] -= matrix(row, col)

    def _del_col(self, col: int, matrix: _MatrixValue) -> None:
        del self.col_sums[col]
        for row in self.row_sums:
            self.row_sums[row] -= matrix(row, col)

    def check_and_add(self, row: int, col: int, matrix: _MatrixValue) -> bool:
        """Add candidate dimensions only when they increase density."""
        row_present = row in self.row_sums
        col_present = col in self.col_sums

        if row_present and col_present:
            self.total += 1.0
            self.row_sums[row] += 1.0
            self.col_sums[col] += 1.0
            return False

        row_sum = (
            0.0
            if row_present
            else sum(matrix(row, selected_col) for selected_col in self.col_sums)
        )
        col_sum = (
            0.0
            if col_present
            else sum(matrix(selected_row, col) for selected_row in self.row_sums)
        )
        n_rows = len(self.row_sums) + int(not row_present)
        n_cols = len(self.col_sums) + int(not col_present)
        candidate_total = self.total + row_sum + col_sum
        if not row_present and not col_present:
            candidate_total += matrix(row, col)

        candidate_density = candidate_total / math.sqrt(n_rows * n_cols)
        if self.density >= candidate_density:
            return False

        if not row_present and not col_present:
            self._add_row(row, row_sum, matrix)
            self._add_col(col, col_sum + matrix(row, col), matrix)
        elif not row_present:
            self._add_row(row, row_sum, matrix)
        elif not col_present:
            self._add_col(col, col_sum, matrix)
        self.total = candidate_total
        return True

    def check_and_delete(self, matrix: _MatrixValue) -> bool:
        """Delete the weakest row or column when doing so increases density."""
        min_row: tuple[int, float] | None = None
        if len(self.row_sums) > 1:
            min_row = min(self.row_sums.items(), key=lambda item: (item[1], item[0]))

        min_col: tuple[int, float] | None = None
        if len(self.col_sums) > 1:
            min_col = min(self.col_sums.items(), key=lambda item: (item[1], item[0]))

        row_density = 0.0
        if min_row is not None:
            row_density = (self.total - min_row[1]) / math.sqrt(
                (len(self.row_sums) - 1) * len(self.col_sums)
            )

        col_density = 0.0
        if min_col is not None:
            col_density = (self.total - min_col[1]) / math.sqrt(
                len(self.row_sums) * (len(self.col_sums) - 1)
            )

        if self.density < row_density and col_density < row_density:
            if min_row is None:
                return False
            self._del_row(min_row[0], matrix)
            self.total -= min_row[1]
            return True
        if self.density < col_density and row_density < col_density:
            if min_col is None:
                return False
            self._del_col(min_col[0], matrix)
            self.total -= min_col[1]
            return True
        return False

    def likelihood(self, row: int, col: int, matrix: _MatrixValue) -> float:
        """Return the local edge likelihood used by AnoEdge-L."""
        score = sum(matrix(selected_row, col) for selected_row in self.row_sums)
        score += sum(matrix(row, selected_col) for selected_col in self.col_sums)
        count = len(self.row_sums) + len(self.col_sums)

        if row in self.row_sums and col in self.col_sums:
            score -= matrix(row, col)
            count -= 1
        return score / float(count)


class AnoEdgeL(BaseModel):
    """
    AnoEdge-L local dense-submatrix detector for dynamic graph edge streams.

    Source and destination identifiers are hashed into higher-order count-min
    sketch matrices. For every sketch plane, one or more local dense
    submatrices are maintained with the authors' greedy add/delete updates. The
    edge score is the minimum across planes of the summed local-submatrix
    likelihoods.

    The authors' implementation inserts an edge before updating submatrices and
    scoring it. ``score_one`` reproduces that candidate-inclusive operation on
    copies of the small submatrix states, without mutating the sketch or
    advancing time. ``learn_one`` applies the same operation to learned state.

    Notes:
    - Source and destination identifiers must be integer-like numbers.
    - Scores are continuous and non-negative; denser anomalous edges score
      higher.
    - With ``normalize_score=True``, scores are squashed to ``[0, 1)``.
    - State is bounded by the configured higher-order sketch dimensions.

    Args:
        source_key: Input field containing the integer-like source identifier.
        destination_key: Input field containing the integer-like destination
            identifier.
        time_key: Input field containing a non-decreasing integer-like time
            bucket. ``None`` assigns a new one-based bucket to every learned
            arrival.
        count_min_rows: Row dimension of every higher-order sketch plane.
        count_min_cols: Column dimension of every higher-order sketch plane.
        num_hashes: Number of independently hashed sketch planes.
        num_dense_submatrices: Local dense submatrices maintained per plane. It
            cannot exceed either sketch dimension.
        time_decay_factor: Factor in ``(0, 1]`` applied to sketches and local
            submatrices when the time bucket advances.
        warm_up_samples: Number of learned edges before scoring begins.
        normalize_score: Apply ``score / (1 + score)`` to the non-negative raw
            score.
        predict_threshold: Boundary used by ``predict_one`` on the selected
            raw or normalized score scale.
        seed: Seed for model-local sketch hash generation.

    References:
        Bhatia, S., Wadhwa, M., Kawaguchi, K., Shah, N., Yu, P. S., &
        Hooi, B. (2023). Sketch-Based Anomaly Detection in Streaming Graphs.
        https://doi.org/10.1145/3580305.3599273
        Original implementation: https://github.com/Stream-AD/AnoGraph
    """

    @staticmethod
    def _validate_parameters(
        *,
        count_min_rows: int,
        count_min_cols: int,
        num_hashes: int,
        num_dense_submatrices: int,
        time_decay_factor: float,
        warm_up_samples: int,
        normalize_score: bool,
        predict_threshold: float,
    ) -> None:
        if count_min_rows <= 0:
            raise ValueError("count_min_rows must be positive")
        if count_min_cols <= 0:
            raise ValueError("count_min_cols must be positive")
        if num_hashes <= 0:
            raise ValueError("num_hashes must be positive")
        if num_dense_submatrices <= 0:
            raise ValueError("num_dense_submatrices must be positive")
        if num_dense_submatrices > min(count_min_rows, count_min_cols):
            raise ValueError(
                "num_dense_submatrices cannot exceed either sketch dimension"
            )
        if not (0.0 < time_decay_factor <= 1.0):
            raise ValueError("time_decay_factor must be in (0, 1]")
        if warm_up_samples < 0:
            raise ValueError("warm_up_samples must be non-negative")
        if normalize_score and not (0.0 <= predict_threshold <= 1.0):
            raise ValueError(
                "predict_threshold must be in [0, 1] when normalize_score=True"
            )
        if not normalize_score and predict_threshold < 0.0:
            raise ValueError(
                "predict_threshold must be non-negative when normalize_score=False"
            )

    def __init__(
        self,
        source_key: str = "src",
        destination_key: str = "dst",
        time_key: str | None = "t",
        count_min_rows: int = 256,
        count_min_cols: int = 256,
        num_hashes: int = 4,
        num_dense_submatrices: int = 1,
        time_decay_factor: float = 1.0,
        warm_up_samples: int = 0,
        normalize_score: bool = False,
        predict_threshold: float = 0.5,
        seed: int | None = None,
    ) -> None:
        self._validate_parameters(
            count_min_rows=count_min_rows,
            count_min_cols=count_min_cols,
            num_hashes=num_hashes,
            num_dense_submatrices=num_dense_submatrices,
            time_decay_factor=time_decay_factor,
            warm_up_samples=warm_up_samples,
            normalize_score=normalize_score,
            predict_threshold=predict_threshold,
        )

        self.source_key = source_key
        self.destination_key = destination_key
        self.time_key = time_key
        self.count_min_rows = count_min_rows
        self.count_min_cols = count_min_cols
        self.num_hashes = num_hashes
        self.num_dense_submatrices = num_dense_submatrices
        self.time_decay_factor = time_decay_factor
        self.warm_up_samples = warm_up_samples
        self.normalize_score = normalize_score
        self.predict_threshold = predict_threshold
        self.seed = seed

        self._reset_state()

    def _reset_state(self) -> None:
        self._boundary = EdgeEventBoundary(
            source_key=self.source_key,
            destination_key=self.destination_key,
            time_key=self.time_key,
        )
        self._rng = np.random.default_rng(self.seed)
        hash_modulus = max(self.count_min_rows, self.count_min_cols)
        self._hash_a = self._rng.integers(
            1,
            max(hash_modulus, 2),
            size=self.num_hashes,
            dtype=np.int64,
        )
        self._hash_b = self._rng.integers(
            0,
            hash_modulus,
            size=self.num_hashes,
            dtype=np.int64,
        )
        self._sketch = np.zeros(
            (self.num_hashes, self.count_min_rows, self.count_min_cols),
            dtype=np.float64,
        )
        self._dense_submatrices = [
            [
                _DenseSubmatrix(index, index)
                for index in range(self.num_dense_submatrices)
            ]
            for _ in range(self.num_hashes)
        ]
        self._current_bucket: int | None = None
        self._samples_seen = 0

    def reset(self) -> None:
        """Reset learned state while keeping hyperparameters."""
        self._reset_state()

    @property
    def n_samples_seen(self) -> int:
        """Number of observed samples processed via learn_one."""
        return self._samples_seen

    def _hash(self, value: int, plane: int, size: int) -> int:
        return (value * int(self._hash_a[plane]) + int(self._hash_b[plane])) % size

    def _hashed_cells(self, src: int, dst: int) -> list[tuple[int, int, int]]:
        return [
            (
                plane,
                self._hash(src, plane, self.count_min_rows),
                self._hash(dst, plane, self.count_min_cols),
            )
            for plane in range(self.num_hashes)
        ]

    def _rollover_for_learning(self, bucket: int) -> None:
        if self._current_bucket is None:
            self._current_bucket = bucket
            return
        if bucket == self._current_bucket:
            return

        self._sketch *= self.time_decay_factor
        for plane_submatrices in self._dense_submatrices:
            for submatrix in plane_submatrices:
                submatrix.decay(self.time_decay_factor)
        self._current_bucket = bucket

    def _preview_plane(
        self,
        plane: int,
        row: int,
        col: int,
        *,
        rollover: bool,
    ) -> float:
        factor = self.time_decay_factor if rollover else 1.0
        sketch = self._sketch[plane]

        def matrix_value(query_row: int, query_col: int) -> float:
            candidate = 1.0 if query_row == row and query_col == col else 0.0
            return float(sketch[query_row, query_col]) * factor + candidate

        score = 0.0
        for learned in self._dense_submatrices[plane]:
            submatrix = learned.copy()
            if rollover:
                submatrix.decay(factor)
            if submatrix.check_and_add(row, col, matrix_value):
                while submatrix.check_and_delete(matrix_value):
                    pass
            score += submatrix.likelihood(row, col, matrix_value)
        return score

    def _update_plane(self, plane: int, row: int, col: int) -> float:
        sketch = self._sketch[plane]
        sketch[row, col] += 1.0

        def matrix_value(query_row: int, query_col: int) -> float:
            return float(sketch[query_row, query_col])

        score = 0.0
        for submatrix in self._dense_submatrices[plane]:
            if submatrix.check_and_add(row, col, matrix_value):
                while submatrix.check_and_delete(matrix_value):
                    pass
            score += submatrix.likelihood(row, col, matrix_value)
        return score

    def learn_one(self, x: dict[str, float]) -> None:
        """Insert an edge and update the local dense-submatrix states."""
        event = self._boundary.preview(x)
        bucket, src, dst = event.bucket, event.source, event.destination
        self._rollover_for_learning(bucket)
        for plane, row, col in self._hashed_cells(src, dst):
            self._update_plane(plane, row, col)

        self._samples_seen += 1
        self._boundary.commit(event)

    def score_one(self, x: dict[str, float]) -> float:
        """Preview the candidate-inclusive AnoEdge-L score without mutation."""
        event = self._boundary.preview(x)
        bucket, src, dst = event.bucket, event.source, event.destination
        if self._samples_seen < self.warm_up_samples:
            return 0.0

        rollover = self._current_bucket is not None and bucket > self._current_bucket
        score = min(
            self._preview_plane(plane, row, col, rollover=rollover)
            for plane, row, col in self._hashed_cells(src, dst)
        )
        score = float(max(score, 0.0))
        if self.normalize_score:
            return score / (1.0 + score)
        return score

    def predict_one(self, x: dict[str, float]) -> int:
        """Return binary anomaly prediction using ``predict_threshold``."""
        return int(self.score_one(x) >= self.predict_threshold)

    def __repr__(self) -> str:
        return (
            "AnoEdgeL("
            f"source_key={self.source_key!r}, destination_key={self.destination_key!r}, "
            f"time_key={self.time_key!r}, count_min_rows={self.count_min_rows}, "
            f"count_min_cols={self.count_min_cols}, num_hashes={self.num_hashes}, "
            f"num_dense_submatrices={self.num_dense_submatrices}, "
            f"time_decay_factor={self.time_decay_factor}, "
            f"warm_up_samples={self.warm_up_samples}, "
            f"normalize_score={self.normalize_score}, "
            f"predict_threshold={self.predict_threshold}, seed={self.seed}, "
            f"samples_seen={self._samples_seen})"
        )
