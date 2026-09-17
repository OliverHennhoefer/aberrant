"""Shared rolling matrix-profile lifecycle and exact distance kernels."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from itertools import islice
from typing import Generic, TypeVar, cast

import numpy as np
from scipy.signal import fftconvolve

from aberrant.base.model import BaseModel
from aberrant.utils.validation import FeatureSchema, PreparedFeatures

_Sample = TypeVar("_Sample", float, np.ndarray)
_Statistics = TypeVar(
    "_Statistics", tuple[int, float, float], list[tuple[int, float, float]]
)


def _resolve_parameters(
    subsequence_length: int,
    window_size: int | None,
    normalize: bool,
    exclusion_zone: int | None,
) -> tuple[int, int]:
    """Validate shared configuration and resolve capacity and exclusion defaults."""
    parameters = {
        "subsequence_length": subsequence_length,
        "window_size": window_size,
        "exclusion_zone": exclusion_zone,
    }
    for name, value in parameters.items():
        if value is None and name != "subsequence_length":
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    if subsequence_length < 2:
        raise ValueError("subsequence_length must be at least 2")
    if not isinstance(normalize, bool):
        raise ValueError("normalize must be a boolean")
    resolved_zone = (
        (subsequence_length + 3) // 4 if exclusion_zone is None else exclusion_zone
    )
    resolved_window = 16 * subsequence_length if window_size is None else window_size
    if resolved_zone < 0:
        raise ValueError("exclusion_zone must be nonnegative")
    if resolved_window < subsequence_length + resolved_zone + 1:
        raise ValueError(
            "window_size must be at least subsequence_length + exclusion_zone + 1"
        )
    return resolved_window, resolved_zone


class _BaseRollingMatrixProfile(BaseModel, Generic[_Sample, _Statistics]):
    """Keep observations, window statistics, and absolute indices aligned."""

    _expected_size: int | None = None

    def __init__(
        self,
        subsequence_length: int,
        window_size: int | None = None,
        *,
        normalize: bool = True,
        exclusion_zone: int | None = None,
    ) -> None:
        resolved_window, resolved_zone = _resolve_parameters(
            subsequence_length, window_size, normalize, exclusion_zone
        )
        self.subsequence_length = subsequence_length
        self.window_size = resolved_window
        self.normalize = normalize
        self.exclusion_zone = resolved_zone
        self.reset()

    def reset(self) -> None:
        """Clear history, schema, and absolute indices; preserve configuration."""
        self._schema = FeatureSchema(expected_size=self._expected_size)
        self._history: deque[_Sample] = deque(maxlen=self.window_size)
        self._statistics: deque[_Statistics] = deque(
            maxlen=self.window_size - self.subsequence_length + 1
        )
        self._samples_seen = 0

    @property
    def n_samples_seen(self) -> int:
        """Number of successfully learned events since initialization or reset."""
        return self._samples_seen

    @property
    def n_history(self) -> int:
        """Number of retained observations, at most ``window_size``."""
        return len(self._history)

    @property
    def is_ready(self) -> bool:
        """Whether the next valid event has an eligible reference subsequence."""
        return self._samples_seen >= self.subsequence_length + self.exclusion_zone

    def _learn(
        self,
        prepared: PreparedFeatures,
        value: _Sample,
        summarize: Callable[[np.ndarray], _Statistics],
    ) -> None:
        statistics = None
        if len(self._history) >= self.subsequence_length - 1:
            # Iterate from the newest end so learning remains O(m), not O(W).
            tail: list[_Sample] = list(
                islice(reversed(self._history), self.subsequence_length - 1)
            )
            query = np.asarray([*reversed(tail), value], dtype=np.float64)
            statistics = summarize(query)

        self._schema.commit(prepared)
        self._history.append(value)
        if statistics is not None:
            self._statistics.append(statistics)
        self._samples_seen += 1

    def _query(
        self, value: _Sample
    ) -> tuple[np.ndarray, np.ndarray, list[_Statistics], int]:
        """Prepare an eligible reference series and query without advancing state."""
        history = np.asarray(self._history, dtype=np.float64)
        egress = int(len(history) == self.window_size)
        combined = np.concatenate(
            (history[egress:], np.asarray([value], dtype=np.float64))
        )
        count = len(combined) - self.subsequence_length - self.exclusion_zone
        statistics = list(islice(self._statistics, egress, egress + count))
        window_start = max(0, self._samples_seen - self.window_size + 1)
        return (
            combined[: count + self.subsequence_length - 1],
            combined[-self.subsequence_length :],
            statistics,
            window_start,
        )


def _window_statistics(values: np.ndarray) -> tuple[int, float, float]:
    """Return a power-of-two exponent, mean offset, and population deviation.

    Statistics use scaled differences from the first value. Scaling by a power
    of two preserves small differences at large offsets and also keeps very
    small or large finite inputs away from squared-norm underflow/overflow.
    """
    exponent = math.frexp(float(np.max(np.abs(values))))[1]
    scaled = np.ldexp(values, -exponent)
    shifted = scaled - scaled[0]
    mean = float(np.mean(shifted))
    std = float(np.linalg.norm(shifted - mean)) / math.sqrt(values.size)
    return exponent, mean, std


def _multivariate_window_statistics(
    values: np.ndarray,
) -> list[tuple[int, float, float]]:
    return [
        _window_statistics(values[:, channel]) for channel in range(values.shape[1])
    ]


def _normalized_window(
    values: np.ndarray, statistics: tuple[int, float, float]
) -> np.ndarray:
    exponent, mean, std = statistics
    scaled = np.ldexp(values, -exponent)
    return cast(np.ndarray, ((scaled - scaled[0]) - mean) / std)


def _distance_bounds(
    series: np.ndarray,
    query: np.ndarray,
    statistics: np.ndarray,
    query_statistics: tuple[int, float, float],
    *,
    normalize: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute MASS squared distances and conservative roundoff allowances.

    These estimates only screen candidates. Every potential minimum is measured
    directly, including candidates whose FFT estimate is ill-conditioned.
    Return squared lower/upper bounds and their distance scale exponent.
    Multiply their square roots by 2**exponent to recover distance units.
    """
    m = query.size
    if normalize and query_statistics[2] == 0.0:
        squared = np.where(statistics[:, 2] == 0.0, 0.0, float(m))
        return squared, squared.copy(), 0

    exponent = math.frexp(float(np.max(np.abs(series))))[1]
    if not normalize:
        exponent = max(exponent, query_statistics[0])
    scaled = np.ldexp(series, -exponent)
    centered = scaled - scaled[0]
    shifts = statistics[:, 0].astype(np.int64) - exponent
    means = centered[: statistics.shape[0]] + np.ldexp(statistics[:, 1], shifts)
    deviations = np.ldexp(statistics[:, 2], shifts)

    if normalize:
        transformed_query = _normalized_window(query, query_statistics)
    else:
        transformed_query = np.ldexp(query, -exponent) - scaled[0]
    products = fftconvolve(centered, transformed_query[::-1], mode="valid")

    # FFT forward/inverse error grows with transform depth and operand norms.
    # The margin also covers moment arithmetic and the subsequent subtraction.
    depth = max(1, (series.size + m - 2).bit_length())
    rounding = 64.0 * np.finfo(np.float64).eps * (m + depth)
    # Bound the operand norms by sqrt(length) * max(abs(x)); squaring very
    # small values in an ordinary norm would underflow and erase this margin.
    product_error = (
        rounding
        * math.sqrt(series.size * m)
        * float(np.max(np.abs(centered)))
        * float(np.max(np.abs(transformed_query)))
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if normalize:
            covariance = products - means * float(np.sum(transformed_query))
            squared = 2.0 * m - 2.0 * covariance / deviations
            error = 2.0 * product_error / deviations + rounding * m
            constant = statistics[:, 2] == 0.0
            squared[constant] = float(m)
            error[constant] = 0.0
        else:
            energy = m * (deviations * deviations + means * means)
            query_energy = float(np.dot(transformed_query, transformed_query))
            squared = energy + query_energy - 2.0 * products
            error = np.maximum(
                2.0 * product_error + rounding * (energy + query_energy),
                np.finfo(np.float64).tiny,
            )

        unreliable = ~np.isfinite(squared) | ~np.isfinite(error) | (squared < -error)
        lower = np.maximum(0.0, squared - error)
        upper = np.maximum(0.0, squared + error)
    lower[unreliable] = 0.0
    upper[unreliable] = np.inf
    return lower, upper, 0 if normalize else exponent


def _nearest_match(
    series: np.ndarray,
    query: np.ndarray,
    statistics: list[tuple[int, float, float]],
    *,
    normalize: bool,
) -> tuple[float, int]:
    """Find the exact nearest eligible window, resolving ties chronologically."""
    m = query.size
    query_statistics = _window_statistics(query)
    stats = np.asarray(statistics, dtype=np.float64)
    constants = stats[:, 2] == 0.0
    if normalize and query_statistics[2] == 0.0:
        matches = np.flatnonzero(constants)
        return (0.0, int(matches[0])) if matches.size else (math.sqrt(m), 0)

    lower, upper, _ = _distance_bounds(
        series, query, stats, query_statistics, normalize=normalize
    )
    candidates = np.flatnonzero(lower <= float(np.min(upper)))
    normalized_query = (
        _normalized_window(query, query_statistics) if normalize else query
    )
    best_distance = math.inf
    best_index = 0
    for candidate in candidates:
        index = int(candidate)
        reference = series[index : index + m]
        if normalize:
            distance = (
                math.sqrt(m)
                if constants[index]
                else float(
                    np.linalg.norm(
                        _normalized_window(reference, statistics[index])
                        - normalized_query
                    )
                )
            )
        else:
            # math.dist uses a scaled norm, including for subnormal differences.
            distance = math.dist(reference, query)
        if distance < best_distance:
            best_distance, best_index = distance, index
            if distance == 0.0:
                break
    if not math.isfinite(best_distance):
        raise OverflowError("Nearest Euclidean distance exceeds the float64 range")
    return best_distance, best_index


def _rescale_bounds(
    lower: np.ndarray, upper: np.ndarray, exponent: int
) -> tuple[np.ndarray, np.ndarray]:
    """Put squared channel bounds into common distance units, rounding outward."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        lower = np.maximum(0.0, np.nextafter(np.sqrt(lower), -np.inf))
        upper = np.nextafter(np.sqrt(upper), np.inf)
        lower = np.maximum(0.0, np.nextafter(np.ldexp(lower, exponent), -np.inf))
        upper = np.nextafter(np.ldexp(upper, exponent), np.inf)
    unreliable = ~np.isfinite(lower) | ~np.isfinite(upper)
    lower[unreliable] = 0.0
    upper[unreliable] = np.inf
    return lower, upper


def _nearest_multivariate_match(
    series: np.ndarray,
    query: np.ndarray,
    statistics: list[list[tuple[int, float, float]]],
    *,
    normalize: bool,
) -> tuple[float, int, list[float]]:
    """Minimize the largest channel distance at a common historical start."""
    m = query.shape[0]
    stats = np.asarray(statistics, dtype=np.float64)
    query_statistics = _multivariate_window_statistics(query)
    lower = np.zeros(len(statistics))
    upper = np.zeros(len(statistics))
    transformed_queries = []
    for channel, query_stats in enumerate(query_statistics):
        channel_lower, channel_upper, exponent = _distance_bounds(
            series[:, channel],
            query[:, channel],
            stats[:, channel, :],
            query_stats,
            normalize=normalize,
        )
        channel_lower, channel_upper = _rescale_bounds(
            channel_lower, channel_upper, exponent
        )
        np.maximum(lower, channel_lower, out=lower)
        np.maximum(upper, channel_upper, out=upper)
        transformed_queries.append(
            _normalized_window(query[:, channel], query_stats)
            if normalize and query_stats[2] != 0.0
            else query[:, channel]
        )

    # Screen joint bounds only: the common winner need not be any channel's
    # individual nearest match. All plausible joint minima need refinement.
    candidates = np.flatnonzero(lower <= float(np.min(upper)))
    best_distance = math.inf
    best_index = 0
    best_channels: list[float] = []
    for candidate in candidates:
        index = int(candidate)
        distances = []
        for channel, query_stats in enumerate(query_statistics):
            reference = series[index : index + m, channel]
            if normalize:
                reference_stats = statistics[index][channel]
                constant_reference = reference_stats[2] == 0.0
                constant_query = query_stats[2] == 0.0
                if constant_reference or constant_query:
                    distance = (
                        0.0 if constant_reference and constant_query else math.sqrt(m)
                    )
                else:
                    distance = float(
                        np.linalg.norm(
                            _normalized_window(reference, reference_stats)
                            - transformed_queries[channel]
                        )
                    )
            else:
                distance = math.dist(reference, query[:, channel])
            distances.append(distance)
        joint_distance = max(distances)
        if joint_distance < best_distance:
            best_distance, best_index, best_channels = joint_distance, index, distances
            if joint_distance == 0.0:
                break
    if not math.isfinite(best_distance):
        raise OverflowError("Nearest Euclidean distance exceeds the float64 range")
    return best_distance, best_index, best_channels
