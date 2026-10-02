"""Shared order statistics for finite, sorted observations."""

import math
from collections.abc import Sequence
from sys import float_info


def linear_quantile(sorted_values: Sequence[float], probability: float) -> float:
    """Interpolate a nonempty finite sorted sample at ``(n - 1) * q``.

    Callers supply a probability in [0, 1]. Interpolating from the nearer
    endpoint preserves small spreads; opposite extreme endpoints use a
    weighted sum when their difference overflows.
    """
    position = (len(sorted_values) - 1) * probability
    lower_index = int(position)
    fraction = position - lower_index
    lower = float(sorted_values[lower_index])
    if fraction == 0.0:
        return lower
    upper = float(sorted_values[lower_index + 1])
    if lower == upper:
        return lower

    difference = upper - lower
    if not math.isfinite(difference):
        return (1.0 - fraction) * lower + fraction * upper

    # Lift subnormal gaps before multiplying so half-ULP ties survive until
    # the final rounding back to the original scale.
    subnormal = difference < float_info.min or (
        abs(lower) < float_info.min and abs(upper) < float_info.min
    )
    if subnormal:
        lower = math.ldexp(lower, 52)
        upper = math.ldexp(upper, 52)
        difference = upper - lower
    interpolated = (
        lower + fraction * difference
        if fraction <= 0.5
        else upper - (1.0 - fraction) * difference
    )
    return math.ldexp(interpolated, -52) if subnormal else interpolated
