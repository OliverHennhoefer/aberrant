"""Exact latest-subsequence matrix-profile scores over bounded history."""

from __future__ import annotations

from aberrant.model.timeseries._matrix_profile import (
    _BaseRollingMatrixProfile,
    _nearest_match,
    _window_statistics,
)


class RollingMatrixProfile(_BaseRollingMatrixProfile[float, tuple[int, float, float]]):
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

    _expected_size = 1

    def learn_one(self, x: dict[str, float]) -> None:
        """Append one event and its window statistics without computing a score."""
        prepared = self._schema.preview(x)
        self._learn(prepared, float(prepared.values[0]), _window_statistics)

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
        series, query, statistics, window_start = self._query(float(prepared.values[0]))
        distance, local_start = _nearest_match(
            series, query, statistics, normalize=self.normalize
        )
        return distance, window_start + local_start
