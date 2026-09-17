"""Exact latest-subsequence matrix-profile scores over bounded history."""

from __future__ import annotations

import math
from collections import deque
from itertools import islice
from typing import cast

import numpy as np
from scipy.signal import fftconvolve

from aberrant.base.model import BaseModel
from aberrant.utils.validation import FeatureSchema


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
) -> tuple[np.ndarray, np.ndarray]:
    """Compute MASS squared distances and conservative roundoff allowances.

    These estimates only screen candidates. Every potential minimum is measured
    directly, including candidates whose FFT estimate is ill-conditioned.
    Distances in raw mode share a power-of-two scale, sufficient for screening.
    """
    m = query.size
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
    return lower, upper


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

    lower, upper = _distance_bounds(
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


class RollingMatrixProfile(BaseModel):
    """Exact, causal anomaly scores for the newest scalar subsequence.

    Each score is the minimum distance from the subsequence ending at the
    candidate event to an eligible preceding subsequence in the latest
    ``window_size`` values, including that candidate. Higher distances mean
    greater novelty. History and temporary memory are bounded by the window.

    Args:
        subsequence_length: Number of consecutive values per subsequence,
            at least two.
        window_size: Total retained sample capacity, defaulting to
            ``16 * subsequence_length``. Must be at least
            ``subsequence_length + exclusion_zone + 1``.
        normalize: Use population z-normalized Euclidean distance when true;
            otherwise use ordinary Euclidean distance, preserving level and
            amplitude differences.
        exclusion_zone: Exclude references whose start is within this many
            positions of the query start. Defaults to
            ``ceil(subsequence_length / 4)``. Set to
            ``subsequence_length - 1`` to require nonoverlapping matches.

    Notes:
        - Input contains exactly one consistently named finite numeric feature.
        - Call ``score_one`` or ``match_one`` before ``learn_one``. Scoring is
          read-only, and learning does not require a preceding score call.
        - Warm-up returns ``0.0`` and no match until
          ``subsequence_length + exclusion_zone`` samples have been learned.
          Thereafter, every valid event has an eligible reference.
        - Normalized distance between two constant windows is zero, even at
          different levels; between a constant and a nonconstant it is
          ``sqrt(subsequence_length)``. Use raw mode for level anomalies.
        - A fresh FFT distance profile screens matches on each scored event;
          direct distances resolve potential minima. Typical scoring work is
          ``O(window_size * log(window_size))`` with a numerical fallback of
          at most ``O(window_size * subsequence_length)``. Learning takes
          ``O(subsequence_length)`` work.
        - This model exposes the newest left-profile entry, not a historical
          matrix profile or a global motif/discord ranking. Raw distances
          exceeding the floating-point range raise ``OverflowError``.

    References:
        Matrix Profile I: All Pairs Similarity Joins for Time Series.
        https://doi.org/10.1109/ICDM.2016.0179

        STUMPY exclusion and constant-subsequence conventions:
        https://stumpy.readthedocs.io/en/latest/api.html
    """

    def __init__(
        self,
        subsequence_length: int,
        window_size: int | None = None,
        *,
        normalize: bool = True,
        exclusion_zone: int | None = None,
    ) -> None:
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
        resolved_window = (
            16 * subsequence_length if window_size is None else window_size
        )
        if resolved_zone < 0:
            raise ValueError("exclusion_zone must be nonnegative")
        if resolved_window < subsequence_length + resolved_zone + 1:
            raise ValueError(
                "window_size must be at least subsequence_length + exclusion_zone + 1"
            )
        self.subsequence_length = subsequence_length
        self.window_size = resolved_window
        self.normalize = normalize
        self.exclusion_zone = resolved_zone
        self.reset()

    def reset(self) -> None:
        """Clear learned history, feature schema, and indices; keep configuration."""
        self._schema = FeatureSchema(expected_size=1)
        self._history: deque[float] = deque(maxlen=self.window_size)
        self._statistics: deque[tuple[int, float, float]] = deque(
            maxlen=self.window_size - self.subsequence_length + 1
        )
        self._samples_seen = 0

    @property
    def n_samples_seen(self) -> int:
        """Number of successfully learned events since initialization or reset."""
        return self._samples_seen

    @property
    def n_history(self) -> int:
        """Number of learned samples retained, at most ``window_size``."""
        return len(self._history)

    @property
    def is_ready(self) -> bool:
        """Whether the next valid event has an eligible reference subsequence."""
        return self._samples_seen >= self.subsequence_length + self.exclusion_zone

    def learn_one(self, x: dict[str, float]) -> None:
        """Append one event and its window statistics without computing a score."""
        prepared = self._schema.preview(x)
        value = float(prepared.values[0])
        statistics = None
        if len(self._history) >= self.subsequence_length - 1:
            # Iterate from the newest end so learning remains O(m), not O(W).
            tail = list(islice(reversed(self._history), self.subsequence_length - 1))
            query = np.asarray([*reversed(tail), value], dtype=np.float64)
            statistics = _window_statistics(query)

        self._schema.commit(prepared)
        self._history.append(value)
        if statistics is not None:
            self._statistics.append(statistics)
        self._samples_seen += 1

    def score_one(self, x: dict[str, float]) -> float:
        """Return the newest subsequence's nearest distance without learning."""
        return self.match_one(x)[0]

    def match_one(self, x: dict[str, float]) -> tuple[float, int | None]:
        """Return ``(distance, reference_start)`` without changing learned state.

        Indices count successfully learned events from zero since the last
        reset. Before readiness the result is ``(0.0, None)``. Equal distances
        choose the earliest retained start. Returned indices describe this
        query; later learning may evict the corresponding samples.
        """
        prepared = self._schema.preview(x)
        if not self.is_ready:
            return 0.0, None
        history = np.asarray(self._history, dtype=np.float64)
        egress = int(history.size == self.window_size)
        combined = np.append(history[egress:], prepared.values[0])
        count = combined.size - self.subsequence_length - self.exclusion_zone
        statistics = list(islice(self._statistics, egress, egress + count))
        distance, local_start = _nearest_match(
            combined[: count + self.subsequence_length - 1],
            combined[-self.subsequence_length :],
            statistics,
            normalize=self.normalize,
        )
        window_start = max(0, self._samples_seen - self.window_size + 1)
        return distance, window_start + local_start
