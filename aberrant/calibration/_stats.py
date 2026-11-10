"""Student-t helpers for the Bayesian calibrators.

These follow the reference implementation step by step (Lanczos log-gamma,
the exact finite t series, a corrected normal for large degrees of freedom),
so the calibrators reproduce its thresholds.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

_LANCZOS = (
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
)


@dataclass(frozen=True)
class StudentT:
    """A Student-t distribution with ``nu`` degrees of freedom."""

    nu: float
    mu: float
    sigma: float


def lgamma(x: float) -> float:
    """Logarithm of the gamma function for x > 0 (Lanczos, g = 7)."""
    if x < 0.5:
        return math.log(math.pi / math.sin(math.pi * x)) - lgamma(1 - x)
    x -= 1
    a = _LANCZOS[0]
    t = x + 7.5
    for i in range(1, 9):
        a += _LANCZOS[i] / (x + i)
    return 0.5 * math.log(2 * math.pi) + (x + 0.5) * math.log(t) - t + math.log(a)


def _betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c = 1.0
    d = 1 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1 / d
    h = d
    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-12:
            break
    return h


def betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    bt = math.exp(
        lgamma(a + b) - lgamma(a) - lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    )
    if x < (a + 1) / (a + b + 2):
        return bt * _betacf(a, b, x) / a
    return 1 - bt * _betacf(b, a, 1 - x) / b


def norm_cdf(z: float) -> float:
    """Standard normal CDF (Abramowitz & Stegun 7.1.26)."""
    x = abs(z) / math.sqrt(2)
    t = 1 / (1 + 0.3275911 * x)
    y = t * (
        0.254829592
        + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))
    )
    erfc = y * math.exp(-x * x)
    return 1 - erfc / 2 if z >= 0 else erfc / 2


def _t_log_norm(nu: float) -> float:
    return lgamma((nu + 1) / 2) - lgamma(nu / 2) - 0.5 * math.log(nu * math.pi)


def t_log_pdf(x: float, d: StudentT) -> float:
    """Log density of a Student-t distribution."""
    z = (x - d.mu) / d.sigma
    return (
        _t_log_norm(d.nu)
        - math.log(d.sigma)
        - (d.nu + 1) / 2 * math.log1p(z * z / d.nu)
    )


def t_cdf_std(z: float, nu: float) -> float:
    """Standard Student-t CDF.

    Integer ``nu <= 60`` uses the exact finite series (A&S 26.7.3-4), larger
    ``nu`` a corrected normal (A&S 26.7.8), other ``nu`` the incomplete beta.
    """
    if nu > 60:
        return norm_cdf(z * (1 - 1 / (4 * nu)) / math.sqrt(1 + z * z / (2 * nu)))
    if float(nu).is_integer():
        degrees = int(nu)
        th = math.atan(abs(z) / math.sqrt(nu))
        c2 = math.cos(th) ** 2
        s = math.sin(th)
        if degrees % 2 == 1:
            term = math.cos(th)
            total = term if degrees > 1 else 0.0
            for k in range(3, degrees - 1, 2):
                term *= c2 * (k - 1) / k
                total += term
            area = 2 / math.pi * (th + s * total)
        else:
            term, total = 1.0, 1.0
            for k in range(2, degrees - 1, 2):
                term *= c2 * (k - 1) / k
                total += term
            area = s * total
        return 0.5 + area / 2 if z >= 0 else 0.5 - area / 2
    x = nu / (nu + z * z)
    tail = 0.5 * betai(nu / 2, 0.5, x)
    return 1 - tail if z >= 0 else tail


def t_mixture_quantile(
    components: list[StudentT],
    weights: list[float],
    p: float,
    guess: float | None = None,
    tol: float = 1e-7,
) -> float:
    """Quantile of a finite mixture of Student-t distributions.

    Safeguarded Newton on the mixture CDF inside a maintained bracket, so it
    always converges. ``weights`` must sum to 1.
    """
    if p <= 0:
        return -math.inf
    if p >= 1:
        return math.inf
    log_norm = [0.0] * len(components)
    mean = scale = weight_sum = 0.0
    for i, c in enumerate(components):
        if not weights[i] > 0:
            continue
        log_norm[i] = _t_log_norm(c.nu) - math.log(c.sigma)
        mean += weights[i] * c.mu
        scale = max(scale, c.sigma)
        weight_sum += weights[i]
    if not weight_sum > 0 or not scale > 0:
        return math.nan
    mean /= weight_sum

    def evaluate(x: float) -> tuple[float, float]:
        cdf = density = 0.0
        for i, c in enumerate(components):
            if not weights[i] > 0:
                continue
            z = (x - c.mu) / c.sigma
            cdf += weights[i] * t_cdf_std(z, c.nu)
            density += weights[i] * math.exp(
                log_norm[i] - (c.nu + 1) / 2 * math.log1p(z * z / c.nu)
            )
        return cdf / weight_sum, density / weight_sum

    return _bracketed_newton(evaluate, p, mean, scale, guess, tol)


def _bracketed_newton(
    evaluate: Callable[[float], tuple[float, float]],
    p: float,
    mean: float,
    scale: float,
    guess: float | None,
    tol: float = 1e-7,
) -> float:
    """Newton from the guess or the mean, bisecting inside a bracket a < x <= b.

    While one side of the bracket is still open, the step doubles outwards.
    """
    low, high = -math.inf, math.inf
    x = guess if guess is not None and math.isfinite(guess) else mean
    step = scale
    for _ in range(100):
        cdf, density = evaluate(x)
        if abs(cdf - p) < tol:
            break
        if cdf < p:
            low = x
        else:
            high = x
        if high - low < 1e-10 * scale:
            break
        nx = x - (cdf - p) / density if density > 0 else math.nan
        if not low < nx < high:
            if math.isfinite(low) and math.isfinite(high):
                nx = 0.5 * (low + high)
            else:
                nx = x + step if cdf < p else x - step
                step *= 2
        x = nx
    return x
