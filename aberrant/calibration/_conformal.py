"""Online conformal calibrators: rolling split, ACI and its successors."""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence

from aberrant.calibration._window import ScoreHistory, pinball

DEFAULT_GAMMAS = (0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128)


def initial_threshold(history: ScoreHistory, alpha: float) -> float | None:
    """The window's conformal quantile, as threshold trackers start from it."""
    if history.count == 0:
        return None
    q0 = history.conformal_quantile(1 - alpha)
    return q0 if math.isfinite(q0) else history.max_abs_score


class RollingSplit:
    """Rolling split conformal: the empirical quantile of the last W scores."""

    def __init__(self, history: ScoreHistory, alpha: float) -> None:
        self.history = history
        self.alpha = alpha

    def predict(self, key: Hashable) -> float:
        return self.history.conformal_quantile(1 - self.alpha)

    def update(self, key: Hashable, score: float) -> None:
        return None


class ACI:
    """Adaptive Conformal Inference: ``alpha_t += gamma (alpha - err_t)``."""

    def __init__(self, history: ScoreHistory, alpha: float, gamma: float) -> None:
        self.history = history
        self.alpha = alpha
        self.gamma = gamma
        self.alpha_t = alpha
        self.pending: dict[Hashable, float] = {}

    def predict(self, key: Hashable) -> float:
        q = self.history.cap(self.history.conformal_quantile(1 - self.alpha_t))
        self.pending[key] = q
        return q

    def update(self, key: Hashable, score: float) -> None:
        q = self.pending.pop(key, None)
        if q is None:
            return
        err = 1 if score > q else 0
        self.alpha_t += self.gamma * (self.alpha - err)


class DtACI:
    """Dynamically-tuned ACI: ACI experts over a grid of step sizes.

    The experts are reweighted with exponential weights on the pinball loss
    of ``beta_t``, using the deterministic weighted-average combination.
    """

    def __init__(
        self,
        history: ScoreHistory,
        alpha: float,
        gammas: Sequence[float],
        interval: float,
    ) -> None:
        k = len(gammas)
        self.history = history
        self.alpha = alpha
        self.gammas = tuple(gammas)
        self.alphas = [alpha] * k
        self.weights = [1 / k] * k
        self.pending: dict[Hashable, list[float]] = {}
        self.sigma = 1 / (2 * interval)
        self.eta = math.sqrt(3 / interval) * math.sqrt(
            (math.log(k * interval) + 2) / ((1 - alpha) ** 2 * alpha**3)
        )

    def predict(self, key: Hashable) -> float:
        h = self.history
        self.pending[key] = [h.cap(h.conformal_quantile(1 - a)) for a in self.alphas]
        a = sum(w * x for w, x in zip(self.weights, self.alphas, strict=True))
        return h.cap(h.conformal_quantile(1 - a))

    def update(self, key: Hashable, score: float) -> None:
        qs = self.pending.pop(key, None)
        if qs is None:
            return
        # beta_t = sup{beta : score <= Q_{1-beta}} on the current window.
        ordered = self.history.window_sorted
        below = 0
        while below < len(ordered) and ordered[below] < score:
            below += 1
        beta = 1 - below / (len(ordered) + 1)
        k = len(self.alphas)
        total = 0.0
        for i in range(k):
            loss = pinball(beta, self.alphas[i], self.alpha)
            self.weights[i] *= math.exp(-self.eta * loss)
            total += self.weights[i]
        for i in range(k):
            self.weights[i] = (
                (1 - self.sigma) * (self.weights[i] / total) + self.sigma / k
                if total > 0 and math.isfinite(total)
                else 1 / k
            )
            err = 1 if score > qs[i] else 0
            self.alphas[i] += self.gammas[i] * (self.alpha - err)


class AgACI:
    """Aggregated ACI: ACI experts combined by exponentially weighted aggregation."""

    def __init__(
        self, history: ScoreHistory, alpha: float, gammas: Sequence[float]
    ) -> None:
        self.history = history
        self.alpha = alpha
        self.gammas = tuple(gammas)
        self.alphas = [alpha] * len(gammas)
        self.log_weights = [0.0] * len(gammas)
        self.pending: dict[Hashable, list[float]] = {}
        self.t = 0

    def _expert(self, a: float) -> float:
        q = self.history.conformal_quantile(1 - a)
        cap = self.history.max_abs_score
        if math.isfinite(q):
            return q
        return cap if q > 0 else -cap

    def predict(self, key: Hashable) -> float:
        qs = [self._expert(a) for a in self.alphas]
        self.pending[key] = qs
        top = max(self.log_weights)
        num = den = 0.0
        for log_weight, q in zip(self.log_weights, qs, strict=True):
            w = math.exp(log_weight - top)
            num += w * q
            den += w
        return math.inf if self.history.count == 0 else num / den

    def update(self, key: Hashable, score: float) -> None:
        qs = self.pending.pop(key, None)
        if qs is None:
            return
        self.t += 1
        scale = max(self.history.recent_range(200), 1e-12)
        eta = math.sqrt(8 * math.log(len(qs)) / self.t) / scale
        for i, q in enumerate(qs):
            self.log_weights[i] -= eta * pinball(score, q, 1 - self.alpha)
            err = 1 if score > q else 0
            self.alphas[i] += self.gammas[i] * (self.alpha - err)


class QuantileTracker:
    """Quantile tracking, the P controller of conformal PID control.

    Online gradient descent on the pinball loss, ``q += eta (err - alpha)``,
    with the scale-free step ``eta = lr * max |score|`` over the recent window.
    """

    def __init__(
        self, history: ScoreHistory, alpha: float, lr: float, scale_window: int
    ) -> None:
        self.history = history
        self.alpha = alpha
        self.lr = lr
        self.scale_window = scale_window
        self.q: float | None = None
        self.pending: dict[Hashable, float] = {}
        self.err_sum = 0.0
        self.t = 0

    def _base(self) -> float:
        if self.q is None:
            if self.history.count == 0:
                return math.inf
            q0 = self.history.conformal_quantile(1 - self.alpha)
            self.q = q0 if math.isfinite(q0) else self.history.max_abs_score
        return self.q

    def predict(self, key: Hashable) -> float:
        q = self.history.cap(self._base())
        self.pending[key] = q
        return q

    def update(self, key: Hashable, score: float) -> None:
        q = self.pending.pop(key, None)
        if q is None:
            return
        err = 1 if score > q else 0
        self.t += 1
        self.err_sum += err - self.alpha
        if self.q is None:
            return
        scale = self.history.recent_range(self.scale_window) or abs(score)
        self.q += self.lr * scale * (err - self.alpha)


class ConformalPID(QuantileTracker):
    """Conformal PID control without a scorecaster.

    The quantile tracker plus the saturated integrator of past coverage errors.
    """

    def __init__(
        self,
        history: ScoreHistory,
        alpha: float,
        lr: float,
        scale_window: int,
        ki: float,
        horizon_guess: float,
    ) -> None:
        super().__init__(history, alpha, lr, scale_window)
        self.ki = ki
        log_horizon = math.log(horizon_guess)
        self.csat = 2 / math.pi * (math.ceil(log_horizon * 0.01) - 1 / log_horizon)

    def predict(self, key: Hashable) -> float:
        q = self._base()
        integral = 0.0
        if math.isfinite(q) and self.ki > 0 and self.t > 0:
            scale = self.history.recent_range(self.scale_window)
            arg = self.err_sum * math.log(self.t + 1) / (self.csat * (self.t + 1))
            limit = math.pi / 2 * 0.999
            arg = max(-limit, min(limit, arg))
            integral = self.ki * scale * math.tan(arg)
        out = self.history.cap(q + integral)
        self.pending[key] = out
        return out


class NexCP:
    """Non-exchangeable split conformal with geometric weights ``rho^age``.

    The test point carries unit mass at ``+inf``.
    """

    def __init__(self, alpha: float, rho: float, max_len: int) -> None:
        self.alpha = alpha
        self.rho = rho
        self.scores = SortedScores(max_len)

    def predict(self, key: Hashable) -> float:
        items = self.scores.items
        if not items:
            return math.inf
        newest = self.scores.newest
        total = 1.0
        for _, index in items:
            total += self.rho ** (newest - index)
        cumulative = 0.0
        for score, index in items:
            cumulative += self.rho ** (newest - index)
            if cumulative / total >= 1 - self.alpha:
                return score
        return math.inf

    def update(self, key: Hashable, score: float) -> None:
        self.scores.push(score)


class ECI:
    """Error-quantified conformal inference.

    The quantile tracker's update plus a term ``(s - q) f'(s - q)`` with a
    sigmoid ``f`` of slope ``c``, in units of the recent score range. With
    ``cutoff > 0`` the term applies only when ``|s - q|`` exceeds that share
    of the range. Coverage holds only up to a bias that grows with the term.
    """

    def __init__(
        self,
        history: ScoreHistory,
        alpha: float,
        lr: float,
        c: float,
        cutoff: float,
        scale_window: int = 100,
    ) -> None:
        self.history = history
        self.alpha = alpha
        self.lr = lr
        self.c = c
        self.cutoff = cutoff
        self.scale_window = scale_window
        self.q: float | None = None
        self.pending: dict[Hashable, float] = {}

    def predict(self, key: Hashable) -> float:
        if self.q is None:
            self.q = initial_threshold(self.history, self.alpha)
        if self.q is None:
            return math.inf
        q = self.history.cap(self.q)
        self.pending[key] = q
        return q

    def update(self, key: Hashable, score: float) -> None:
        q = self.pending.pop(key, None)
        if q is None or self.q is None:
            return
        spread = self.history.spread(self.scale_window) or abs(score) or 1.0
        err = 1 if score > q else 0
        u = (score - q) / spread
        eq = 0.0
        if abs(u) > self.cutoff:
            f = 1 / (1 + math.exp(-self.c * u))
            eq = u * self.c * f * (1 - f)
        self.q += self.lr * spread * (err - self.alpha + eq)


class SFOGD:
    """Scale-free online gradient descent on the pinball loss.

    ``q -= D g / sqrt(sum g^2)`` with ``D`` the range of the recent scores,
    so the step settles without a tuned learning rate.
    """

    def __init__(
        self, history: ScoreHistory, alpha: float, scale_window: int = 100
    ) -> None:
        self.history = history
        self.alpha = alpha
        self.scale_window = scale_window
        self.q: float | None = None
        self.g2 = 0.0
        self.pending: dict[Hashable, float] = {}

    def predict(self, key: Hashable) -> float:
        if self.q is None:
            self.q = initial_threshold(self.history, self.alpha)
        if self.q is None:
            return math.inf
        q = self.history.cap(self.q)
        self.pending[key] = q
        return q

    def update(self, key: Hashable, score: float) -> None:
        q = self.pending.pop(key, None)
        if q is None or self.q is None:
            return
        g = (1 if score > q else 0) - self.alpha
        self.g2 += g * g
        spread = self.history.spread(self.scale_window) or abs(score) or 1.0
        self.q += spread / math.sqrt(self.g2) * g


class _Expert:
    __slots__ = ("start", "lifetime", "q", "g2", "z", "wz", "age")

    def __init__(self, start: int, lifetime: int, q: float) -> None:
        self.start = start
        self.lifetime = lifetime
        self.q = q
        self.g2 = 0.0
        self.z = 0.0
        self.wz = 0.0
        self.age = 0

    def weight(self) -> float:
        return 0.0 if self.age == 0 else self.z / self.age * (1 + self.wz)


class SAOCP:
    """Strongly adaptive online conformal prediction.

    A new SF-OGD expert starts every step, living ``g * 2^u`` steps where
    ``u`` counts the trailing zeros of its start time. Live experts are
    combined by coin betting, which gives low regret on every interval.
    """

    def __init__(
        self, history: ScoreHistory, alpha: float, g: int, scale_window: int = 100
    ) -> None:
        self.history = history
        self.alpha = alpha
        self.g = g
        self.scale_window = scale_window
        self.experts: list[_Expert] = []
        self.t = 1

    def _meta(self) -> float:
        if not self.experts:
            initial = initial_threshold(self.history, self.alpha)
            return 0.0 if initial is None else initial
        zp = zw = num = num_prior = 0.0
        for e in self.experts:
            prior = 1 / (e.start * e.start * (1 + math.floor(math.log2(e.start))))
            p = prior * max(0.0, e.weight())
            zp += prior
            zw += p
            num += p * e.q
            num_prior += prior * e.q
        return num / zw if zw > 0 else num_prior / zp

    def predict(self, key: Hashable) -> float:
        if self.history.count == 0:
            return math.inf
        return self._meta()

    def update(self, key: Hashable, score: float) -> None:
        spread = self.history.spread(self.scale_window) or abs(score) or 1.0
        tau = 1 - self.alpha
        start = self._meta()
        self.experts = [e for e in self.experts if e.age <= e.lifetime]
        u, t = 0, self.t
        while t % 2 == 0:
            t //= 2
            u += 1
        self.experts.append(_Expert(self.t, self.g * 2**u, start))
        meta_loss = pinball(score, self._meta(), tau)
        norm = math.sqrt(3) * spread * max(self.alpha, tau)
        for e in self.experts:
            w = e.weight()
            gm = min(
                1.0,
                max(
                    -1.0 if w > 0 else 0.0,
                    (meta_loss - pinball(score, e.q, tau)) / norm,
                ),
            )
            e.z += gm
            e.wz += gm * w
            e.age += 1
            grad = -tau if score > e.q else 1 - tau
            e.g2 += grad * grad
            e.q -= spread / math.sqrt(e.g2) * grad
        self.t += 1


class SortedScores:
    """Scores in ascending order with their arrival index, the newest ``max_len``."""

    def __init__(self, max_len: int) -> None:
        self.max_len = max_len
        self.items: list[tuple[float, int]] = []
        self.n = 0

    @property
    def newest(self) -> int:
        return self.n - 1

    def push(self, score: float) -> None:
        items = self.items
        low, high = 0, len(items)
        while low < high:
            mid = (low + high) >> 1
            if items[mid][0] < score:
                low = mid + 1
            else:
                high = mid
        items.insert(low, (score, self.n))
        self.n += 1
        if len(items) > self.max_len:
            oldest = self.n - 1 - self.max_len
            for i, (_, index) in enumerate(items):
                if index == oldest:
                    del items[i]
                    break
