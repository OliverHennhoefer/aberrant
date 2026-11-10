"""Calibrators built on a probabilistic model of the score stream."""

from __future__ import annotations

import math
from collections.abc import Hashable
from dataclasses import dataclass

from aberrant.calibration._conformal import SortedScores
from aberrant.calibration._stats import StudentT, t_log_pdf, t_mixture_quantile
from aberrant.calibration._window import ScoreHistory


@dataclass(frozen=True)
class _Suff:
    """Weighted Gaussian statistics: total weight, mean, squared deviations."""

    n: float = 0.0
    mean: float = 0.0
    ss: float = 0.0

    def add(self, x: float, delta: float = 1.0) -> _Suff:
        """Add ``x`` with unit weight after discounting old weight by ``delta``."""
        n0 = delta * self.n
        n = n0 + 1
        d = x - self.mean
        return _Suff(n, self.mean + d / n, delta * self.ss + n0 * d * d / n)


@dataclass(frozen=True)
class _Prior:
    """Normal-Inverse-Gamma prior NIG(m, k, a, b)."""

    m: float
    k: float
    a: float
    b: float


def _predictive(prior: _Prior, st: _Suff) -> StudentT:
    k = prior.k + st.n
    m = (prior.k * prior.m + st.n * st.mean) / k
    a = prior.a + st.n / 2
    b = prior.b + 0.5 * st.ss + prior.k * st.n * (st.mean - prior.m) ** 2 / (2 * k)
    return StudentT(nu=2 * a, mu=m, sigma=math.sqrt(b * (k + 1) / (a * k)))


def _empirical_prior(h: ScoreHistory, k0: float = 0.5, a0: float = 1.0) -> _Prior:
    """A weak prior centred on the recent scores (empirical Bayes)."""
    xs = h.recent(h.window)
    n = 0
    mean = m2 = 0.0
    for x in xs:
        n += 1
        d = x - mean
        mean += d / n
        m2 += d * (x - mean)
    if n > 1:
        v = m2 / (n - 1)
    elif n == 1:
        v = max(xs[0] ** 2, 1e-12)
    else:
        v = 1.0
    return _Prior(m=mean, k=k0, a=a0, b=a0 * max(v, 1e-12))


class DiscountedBayes:
    """Gaussian score with unknown mean and variance, evidence discounted per step.

    The threshold is the Student-t posterior predictive quantile. Purely
    model-based: no coverage guarantee when the score is not Gaussian.
    """

    def __init__(self, history: ScoreHistory, alpha: float, delta: float) -> None:
        self.history = history
        self.alpha = alpha
        self.delta = delta
        self.st = _Suff()

    def predict(self, key: Hashable) -> float:
        if self.history.count < 2:
            return math.inf
        e = _empirical_prior(self.history)
        prior = _Prior(m=e.m, k=1e-6, a=0.5, b=1e-6 * e.b)
        return t_mixture_quantile([_predictive(prior, self.st)], [1.0], 1 - self.alpha)

    def update(self, key: Hashable, score: float) -> None:
        self.st = self.st.add(score, self.delta)


@dataclass
class _Run:
    length: int
    w: float
    st: _Suff
    prior: _Prior
    pred: StudentT


class BOCPD:
    """Bayesian online changepoint detection with a NIG model per regime.

    Keeps the posterior over the current run length (constant hazard
    ``1 / lambda``) and each run length's statistics, pruned to ``max_runs``.
    """

    def __init__(
        self,
        history: ScoreHistory,
        lam: float,
        max_runs: int = 32,
        min_weight: float = 1e-4,
    ) -> None:
        self.history = history
        self.hazard = 1 / max(lam, 1)
        self.max_runs = max_runs
        self.min_weight = min_weight
        self.runs: list[_Run] = []
        self._last_q: float | None = None

    def quantile(self, p: float) -> float:
        if not self.runs:
            return math.inf
        q = t_mixture_quantile(
            [r.pred for r in self.runs], [r.w for r in self.runs], p, self._last_q
        )
        if math.isfinite(q):
            self._last_q = q
        return q

    def survival(self, max_age: int) -> list[float]:
        """``P(r_t > a)``: the score ``a`` steps back is in the current regime."""
        out = [0.0] * max_age
        for r in self.runs:
            if r.length > 0:
                out[min(r.length, max_age) - 1] += r.w
        for a in range(max_age - 2, -1, -1):
            out[a] += out[a + 1]
        return out

    def observe(self, x: float) -> None:
        hazard = self.hazard
        if not self.runs:
            self.runs = [self._fresh(x, 1.0)]
            return
        runs = self.runs
        prior = _empirical_prior(self.history)
        log_cp = math.log(hazard) + t_log_pdf(x, _predictive(prior, _Suff()))
        top = log_cp
        # A unit hazard (lambda = 1) forces a changepoint every step.
        log_stay = math.log1p(-hazard) if hazard < 1 else -math.inf
        for r in runs:
            r.w = math.log(r.w) + log_stay + t_log_pdf(x, r.pred)
            top = max(top, r.w)
        total = 0.0
        for r in runs:
            r.w = math.exp(r.w - top)
            total += r.w
            r.length += 1
            r.st = r.st.add(x)
            r.pred = _predictive(r.prior, r.st)
        cp = math.exp(log_cp - top)
        total += cp
        runs.insert(0, self._fresh(x, cp, prior))
        kept: list[_Run] = []
        kept_weight = 0.0
        for r in runs:
            r.w /= total
            if r.w >= self.min_weight:
                kept.append(r)
                kept_weight += r.w
        while len(kept) > self.max_runs:
            smallest = min(range(len(kept)), key=lambda i: kept[i].w)
            kept_weight -= kept[smallest].w
            del kept[smallest]
        for r in kept:
            r.w /= kept_weight
        self.runs = kept

    def _fresh(self, x: float, w: float, prior: _Prior | None = None) -> _Run:
        prior = prior or _empirical_prior(self.history)
        st = _Suff().add(x)
        return _Run(length=1, w=w, st=st, prior=prior, pred=_predictive(prior, st))


class BOCPDQuantile:
    """The BOCPD predictive quantile as threshold (no coverage guarantee)."""

    def __init__(self, history: ScoreHistory, alpha: float, lam: float) -> None:
        self.alpha = alpha
        self.model = BOCPD(history, lam)

    def predict(self, key: Hashable) -> float:
        return self.model.quantile(1 - self.alpha)

    def update(self, key: Hashable, score: float) -> None:
        self.model.observe(score)


class BOCPDACI:
    """BOCPD predictive quantile at a level ACI adapts, so coverage holds."""

    def __init__(
        self, history: ScoreHistory, alpha: float, lam: float, gamma: float
    ) -> None:
        self.history = history
        self.alpha = alpha
        self.gamma = gamma
        self.alpha_t = alpha
        self.model = BOCPD(history, lam)
        self.pending: dict[Hashable, float] = {}

    def predict(self, key: Hashable) -> float:
        q = self.history.cap(self.model.quantile(1 - self.alpha_t))
        self.pending[key] = q
        return q

    def update(self, key: Hashable, score: float) -> None:
        q = self.pending.pop(key, None)
        if q is not None:
            self.alpha_t += self.gamma * (self.alpha - (1 if score > q else 0))
        self.model.observe(score)


def _powers(r: float, n: int) -> list[float]:
    out = [0.0] * n
    v = 1.0
    for i in range(n):
        out[i] = v
        v *= r
    return out


class ChangepointCP:
    """Weighted split conformal, weighting each score by its regime probability.

    A past score's weight is the BOCPD posterior probability that it belongs
    to the current regime, floored at ``floor * rho^age``. The weights depend
    on the data, so NexCP's coverage-gap bound does not apply verbatim.
    """

    def __init__(
        self,
        history: ScoreHistory,
        alpha: float,
        lam: float,
        max_len: int = 500,
        floor: float = 0.02,
        rho: float = 0.99,
    ) -> None:
        self.alpha = alpha
        self.floor = floor
        self.model = BOCPD(history, lam)
        self.scores = SortedScores(max_len)
        self.decay = _powers(rho, max_len + 1)

    def _weight(self, survival: list[float], age: int) -> float:
        return max(survival[age], self.floor * self.decay[age])

    def predict(self, key: Hashable) -> float:
        items = self.scores.items
        if not items:
            return math.inf
        survival = self.model.survival(len(items))
        newest = self.scores.newest
        total = 1.0
        for _, index in items:
            total += self._weight(survival, newest - index)
        cumulative = 0.0
        for score, index in items:
            cumulative += self._weight(survival, newest - index)
            if cumulative / total >= 1 - self.alpha:
                return score
        return math.inf

    def update(self, key: Hashable, score: float) -> None:
        self.model.observe(score)
        self.scores.push(score)


class BayesianBelief:
    """Online conformal prediction with a regularized (Bayesian) belief.

    The belief mixes a uniform prior on the scores' range with the discounted
    empirical score distribution; the threshold is its quantile. It ignores
    past coverage errors and never returns an infinite threshold after the
    first score.
    """

    def __init__(self, alpha: float, beta: float) -> None:
        self.alpha = alpha
        self.beta = beta
        self.lam = math.sqrt(1 - beta) / (beta + math.sqrt(1 - beta))
        # Beyond this age a score weighs < 1e-6 of the newest; its mass goes to the prior.
        max_len = min(5000, math.ceil(math.log(1e-6) / math.log(max(beta, 1e-9))) + 1)
        self.scores = SortedScores(max_len)
        self.decay = _powers(beta, max_len + 1)
        self.low = math.inf
        self.high = -math.inf

    def _uniform(self, x: float) -> float:
        span = self.high - self.low
        if span > 0:
            return min(1.0, max(0.0, (x - self.low) / span))
        return 1.0 if x >= self.low else 0.0

    def predict(self, key: Hashable) -> float:
        items = self.scores.items
        if not items:
            return math.inf
        newest = self.scores.newest
        c = (1 - self.lam) * (1 - self.beta)
        empirical = sum(c * self.decay[newest - index] for _, index in items)
        p0 = max(0.0, 1 - empirical)
        low, span = self.low, self.high - self.low
        target = 1 - self.alpha
        cumulative = 0.0
        for score, index in items:
            if cumulative + p0 * self._uniform(score) >= target:
                if span > 0 and p0 > 0:
                    return low + span * min(1.0, (target - cumulative) / p0)
                return score
            cumulative += c * self.decay[newest - index]
            if cumulative + p0 * self._uniform(score) >= target:
                return score
        if span > 0 and p0 > 0:
            return low + span * min(1.0, max(0.0, (target - cumulative) / p0))
        return self.high

    def update(self, key: Hashable, score: float) -> None:
        self.low = min(self.low, score)
        self.high = max(self.high, score)
        self.scores.push(score)
