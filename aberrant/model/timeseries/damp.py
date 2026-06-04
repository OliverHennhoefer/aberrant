"""Pure-online X-Lag Amnesic DAMP time-series discord detector."""

from __future__ import annotations

import math
from collections import deque
from typing import cast

import numpy as np

from aberrant.base.model import BaseModel


def _next_power_of_two(value: int) -> int:
    """Return the smallest power of two greater than or equal to ``value``."""
    return 1 << (value - 1).bit_length()


def _mass_distance_profile(
    series: np.ndarray,
    query: np.ndarray,
    *,
    eps: float,
) -> np.ndarray:
    """
    Compute the MASS_V2 z-normalized Euclidean distance profile.

    This is a NumPy translation of ``MASS_V2`` from the authors' official DAMP
    MATLAB implementation. Population standard deviations match MATLAB's
    ``std(..., 1)`` calls in that source.
    """
    m = int(query.size)
    n = int(series.size)
    if m < 2:
        raise ValueError("query must contain at least two values")
    if n < m:
        raise ValueError("series must be at least as long as query")

    query_mean = float(np.mean(query))
    centered_query = query - query_mean
    query_std = float(np.std(centered_query))
    if query_std <= eps:
        raise ValueError("DAMP does not support constant query subsequences")

    # Centering is algebraically neutral for z-normalized distance and avoids
    # variance cancellation when a sensor has a large numeric offset.
    centered_series = series - float(np.mean(series))
    cumulative = np.concatenate(
        ([0.0], np.cumsum(centered_series, dtype=np.float64))
    )
    cumulative_sq = np.concatenate(
        ([0.0], np.cumsum(centered_series * centered_series, dtype=np.float64))
    )
    window_sums = cumulative[m:] - cumulative[:-m]
    window_sq_sums = cumulative_sq[m:] - cumulative_sq[:-m]
    window_means = window_sums / float(m)
    window_variances = np.maximum(
        (window_sq_sums / float(m)) - (window_means * window_means),
        0.0,
    )
    window_stds = np.sqrt(window_variances)
    if np.any(window_stds <= eps):
        raise ValueError("DAMP does not support constant candidate subsequences")

    reversed_query = centered_query[::-1]
    padded_query = np.pad(reversed_query, (0, n - m))
    dot_products = np.fft.ifft(
        np.fft.fft(centered_series) * np.fft.fft(padded_query)
    ).real[m - 1 : n]

    normalized_dot = dot_products / (window_stds * query_std)
    squared_distances = 2.0 * (float(m) - normalized_dot)
    return cast(np.ndarray, np.sqrt(np.maximum(squared_distances, 0.0)))


class XLagDAMP(BaseModel):
    """
    Pure-online X-Lag Amnesic Discord Aware Matrix Profile detector.

    Each event supplies one scalar time-series value. Once enough history is
    available, the model scores the subsequence ending at the current event by
    its z-normalized Euclidean distance to its nearest preceding subsequence.
    Higher scores indicate stronger left-discord candidates.

    This implementation follows the authors' ``DAMP_X_Lag_Amnesic.m`` source:

    - backward processing only (``lookahead=0``), the authors' pure-online mode,
    - MASS_V2 distance profiles with population standard deviations,
    - iterative doubling starting at ``2^nextpow2(8 * subsequence_length)``,
    - best-so-far early abandoning,
    - exact search over at most the most recent ``x_lag`` values.

    X-Lag amnesia bounds both memory and worst-case search history. As in the
    reference algorithm, early-abandoned scores are approximate: they are
    bounded by the exact left-discord score and the current best-so-far score.
    The highest peaks are the meaningful discord candidates.

    Args:
        subsequence_length: Number of consecutive values in each subsequence.
        x_lag: Maximum number of values searched backward. Defaults to the
            authors' ``16 * subsequence_length``.
        start_index: One-based subsequence start index at which processing
            begins. Defaults to the authors' recommendation of at least four
            cycles, ``4 * subsequence_length``.
        eps: Numerical threshold used to reject constant subsequences.

    Notes:
        - Input must contain exactly one consistently named numeric feature.
        - Scores are ``0.0`` before ``start_index`` is reached.
        - Constant subsequences are rejected, matching the reference
          implementation's input restriction.
        - State is bounded by ``x_lag + subsequence_length - 1`` learned values.

    References:
        Lu, Y., Wu, R., Mueen, A., Zuluaga, M. A., & Keogh, E. (2022).
        Matrix Profile XXIV: Scaling Time Series Anomaly Detection to Trillions
        of Datapoints and Ultra-fast Arriving Data Streams.
        https://doi.org/10.1145/3534678.3539271

        Original authors' implementation and documentation:
        https://sites.google.com/view/discord-aware-matrix-profile/documentation
    """

    def __init__(
        self,
        subsequence_length: int,
        x_lag: int | None = None,
        start_index: int | None = None,
        eps: float = 1e-12,
    ) -> None:
        if subsequence_length < 2:
            raise ValueError("subsequence_length must be at least 2")

        resolved_x_lag = (
            16 * subsequence_length if x_lag is None else x_lag
        )
        if resolved_x_lag < subsequence_length:
            raise ValueError("x_lag must be at least subsequence_length")

        resolved_start_index = (
            4 * subsequence_length if start_index is None else start_index
        )
        if resolved_start_index < subsequence_length:
            raise ValueError("start_index must be at least subsequence_length")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.subsequence_length = subsequence_length
        self.x_lag = resolved_x_lag
        self.start_index = resolved_start_index
        self.eps = eps

        self._initial_search_length = _next_power_of_two(
            8 * self.subsequence_length
        )
        self._history_capacity = self.x_lag + self.subsequence_length - 1
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset learned state while preserving hyperparameters."""
        self._feature_name: str | None = None
        self._history: deque[float] = deque(maxlen=self._history_capacity)
        self._samples_seen = 0
        self._subsequences_processed = 0
        self._best_so_far = float("-inf")
        self._last_score = 0.0

    def reset(self) -> None:
        """Reset learned state while preserving hyperparameters."""
        self._reset_state()

    @property
    def n_samples_seen(self) -> int:
        """Number of values processed through ``learn_one``."""
        return self._samples_seen

    @property
    def n_history(self) -> int:
        """Number of learned values retained in bounded history."""
        return len(self._history)

    @property
    def n_subsequences_processed(self) -> int:
        """Number of subsequences processed after warmup."""
        return self._subsequences_processed

    @property
    def best_score(self) -> float:
        """Highest exact left-discord score observed so far."""
        if not math.isfinite(self._best_so_far):
            return 0.0
        return self._best_so_far

    @property
    def last_score(self) -> float:
        """Score computed during the most recent ``learn_one`` call."""
        return self._last_score

    @property
    def is_ready(self) -> bool:
        """Whether the next event starts a processable subsequence."""
        query_start = self._samples_seen - self.subsequence_length + 2
        return query_start >= self.start_index

    def _extract_value(self, x: dict[str, float], *, mutate_schema: bool) -> float:
        if not x:
            raise ValueError("Input dictionary cannot be empty")
        if len(x) != 1:
            raise ValueError("XLagDAMP requires exactly one feature")

        feature_name, raw_value = next(iter(x.items()))
        if not isinstance(feature_name, str):
            raise ValueError("Feature key must be a string")
        if not isinstance(raw_value, int | float | np.number):
            raise ValueError(f"Feature '{feature_name}' is not numeric")
        value = float(raw_value)
        if not np.isfinite(value):
            raise ValueError(f"Feature '{feature_name}' must be finite")

        if self._feature_name is None:
            if mutate_schema:
                self._feature_name = feature_name
        elif feature_name != self._feature_name:
            raise ValueError(
                "Inconsistent feature key. "
                f"Expected '{self._feature_name}', received '{feature_name}'."
            )
        return value

    def _minimum_mass_distance(
        self,
        series: np.ndarray,
        query: np.ndarray,
    ) -> float:
        profile = _mass_distance_profile(series, query, eps=self.eps)
        return float(np.min(profile))

    def _backward_process(
        self,
        combined: np.ndarray,
        query_start: int,
        query: np.ndarray,
    ) -> tuple[float, float]:
        """Run the authors' X-Lag backward-processing loop."""
        approximate_distance = float("inf")
        search_length = self._initial_search_length
        first_iteration = True
        expansion_num = 0
        best_so_far = self._best_so_far

        while approximate_distance >= best_so_far:
            far_start = (
                query_start
                - search_length
                + 1
                + expansion_num * self.subsequence_length
            )
            reached_x_lag = far_start <= query_start - self.x_lag
            reached_beginning = far_start < 0
            search_exceeds_x_lag = self.x_lag < search_length

            if reached_x_lag or reached_beginning or search_exceeds_x_lag:
                segment_start = max(0, query_start - self.x_lag)
                segment = combined[segment_start : query_start + 1]
                approximate_distance = self._minimum_mass_distance(segment, query)
                best_so_far = max(best_so_far, approximate_distance)
                break

            if first_iteration:
                first_iteration = False
                segment = combined[
                    query_start - search_length + 1 : query_start + 1
                ]
            else:
                segment_start = (
                    query_start
                    - search_length
                    + 1
                    + expansion_num * self.subsequence_length
                )
                segment_end = (
                    query_start
                    + 1
                    - search_length // 2
                    + expansion_num * self.subsequence_length
                )
                segment = combined[segment_start:segment_end]

            approximate_distance = self._minimum_mass_distance(segment, query)
            if approximate_distance < best_so_far:
                break

            search_length *= 2
            expansion_num += 1

        return approximate_distance, best_so_far

    def _candidate_result(self, value: float) -> tuple[float, float] | None:
        if not self.is_ready:
            return None

        combined = np.fromiter(
            (*self._history, value),
            dtype=np.float64,
            count=len(self._history) + 1,
        )
        query_start = combined.size - self.subsequence_length
        query = combined[query_start:]
        return self._backward_process(combined, query_start, query)

    def learn_one(self, x: dict[str, float]) -> None:
        """Process one value and update the online DAMP state."""
        value = self._extract_value(x, mutate_schema=True)
        result = self._candidate_result(value)
        if result is not None:
            self._last_score, self._best_so_far = result
            self._subsequences_processed += 1
        else:
            self._last_score = 0.0

        self._history.append(value)
        self._samples_seen += 1

    def score_one(self, x: dict[str, float]) -> float:
        """Score the subsequence ending at this value without mutating state."""
        value = self._extract_value(x, mutate_schema=False)
        result = self._candidate_result(value)
        if result is None:
            return 0.0
        score, _best_so_far = result
        return score

    def __repr__(self) -> str:
        return (
            "XLagDAMP("
            f"subsequence_length={self.subsequence_length}, x_lag={self.x_lag}, "
            f"start_index={self.start_index}, samples_seen={self._samples_seen}, "
            f"subsequences_processed={self._subsequences_processed}, "
            f"best_score={self.best_score})"
        )
