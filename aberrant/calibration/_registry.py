"""The calibration methods, their parameters and how to build them."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from aberrant.calibration._bayesian import (
    BOCPDACI,
    BayesianBelief,
    BOCPDQuantile,
    ChangepointCP,
    DiscountedBayes,
)
from aberrant.calibration._conformal import (
    ACI,
    DEFAULT_GAMMAS,
    ECI,
    SAOCP,
    SFOGD,
    AgACI,
    ConformalPID,
    DtACI,
    NexCP,
    QuantileTracker,
    RollingSplit,
)
from aberrant.calibration._window import ScoreHistory
from aberrant.utils.validation import coerce_finite_number


class OnlineCalibrator(Protocol):
    """A threshold on one score stream, learned with delayed feedback.

    ``predict(key)`` issues a threshold when a prediction is made, and
    ``update(key, score)`` arrives once that prediction's score is known.
    A score above the threshold is a miss at the target level ``1 - alpha``.
    """

    def predict(self, key: object) -> float: ...

    def update(self, key: object, score: float) -> None: ...


@dataclass(frozen=True)
class CalibratorParameter:
    """One parameter of a calibration method.

    ``minimum`` and ``maximum`` bound the values the method accepts;
    ``low`` and ``high`` are the range worth offering in an editor.
    """

    key: str
    label: str
    default: float
    low: float
    high: float
    step: float
    minimum: float
    maximum: float = math.inf
    exclusive_minimum: bool = False
    exclusive_maximum: bool = False
    integer: bool = False
    help: str = ""

    def coerce(self, value: object) -> float:
        """Validate a value for this parameter."""
        if isinstance(value, bool):
            raise ValueError(f"{self.key} must be numeric")
        number = coerce_finite_number(value, label=self.key)
        if self.integer and not number.is_integer():
            raise ValueError(f"{self.key} must be an integer")
        too_low = (
            number <= self.minimum if self.exclusive_minimum else number < self.minimum
        )
        too_high = (
            number >= self.maximum if self.exclusive_maximum else number > self.maximum
        )
        if too_low or too_high:
            low = "(" if self.exclusive_minimum else "["
            high = ")" if self.exclusive_maximum else "]"
            raise ValueError(
                f"{self.key} must be in {low}{self.minimum:g}, {self.maximum:g}{high}"
            )
        return number


Coverage = Literal["exchangeable", "long_run", "none"]
Factory = Callable[[ScoreHistory, float, Mapping[str, float]], OnlineCalibrator]


@dataclass(frozen=True)
class CalibratorSpec:
    """A calibration method: what it is, its parameters and its guarantee.

    ``coverage`` says what the threshold promises about the score stream:

    - ``"exchangeable"``: a new score exceeds the threshold with probability
      at most ``alpha`` when it is exchangeable with the calibration scores.
    - ``"long_run"``: the fraction of scores above the threshold tends to
      ``alpha`` for any score sequence. This bounds every exceedance, normal
      or anomalous, not the false-positive rate on normal scores alone.
    - ``"none"``: no coverage guarantee; calibrated only in practice.
    """

    id: str
    name: str
    reference: str
    summary: str
    family: Literal["conformal", "bayesian"]
    coverage: Coverage
    parameters: tuple[CalibratorParameter, ...]
    factory: Factory

    def defaults(self) -> dict[str, float]:
        """Every parameter at its default."""
        return {p.key: p.default for p in self.parameters}

    def resolve(self, params: Mapping[str, object]) -> dict[str, float]:
        """Defaults overridden by ``params``, each value validated."""
        known = {p.key: p for p in self.parameters}
        unknown = sorted(set(params) - set(known))
        if unknown:
            raise ValueError(f"{self.id} has no parameter {', '.join(unknown)}")
        resolved = self.defaults()
        for key, value in params.items():
            resolved[key] = known[key].coerce(value)
        return resolved


def _gamma(default: float = 0.01) -> CalibratorParameter:
    return CalibratorParameter(
        "gamma", "Step size", default, 0.001, 0.2, 0.001, 0.0, exclusive_minimum=True
    )


def _lr(help: str = "") -> CalibratorParameter:
    return CalibratorParameter(
        "lr", "Learning rate", 0.05, 0.001, 0.5, 0.001, 0.0,
        exclusive_minimum=True, help=help,
    )  # fmt: skip


def _run_length() -> CalibratorParameter:
    return CalibratorParameter(
        "lambda", "Run length", 250, 10, 5000, 10, 1.0,
        help="Expected steps between changepoints (hazard 1/lambda).",
    )  # fmt: skip


def _discount(key: str, label: str, default: float, help: str) -> CalibratorParameter:
    return CalibratorParameter(
        key, label, default, 0.8, 0.9999, 0.0001, 0.0, 1.0,
        exclusive_minimum=True, exclusive_maximum=True, help=help,
    )  # fmt: skip


_RANGE = "Step as a fraction of the recent score range."

CALIBRATORS: dict[str, CalibratorSpec] = {
    spec.id: spec
    for spec in (
        CalibratorSpec(
            "split", "Rolling split CP", "Vovk et al. 2005; rolling calibration window",
            "Empirical (1-alpha)(1+1/n) quantile of the last W scores. Valid under "
            "exchangeability, slow to react to shifts.",
            "conformal", "exchangeable", (),
            lambda h, a, p: RollingSplit(h, a),
        ),
        CalibratorSpec(
            "aci", "ACI", "Gibbs & Candes, NeurIPS 2021",
            "Adapts the working miscoverage level alpha_t += gamma (alpha - err_t). "
            "Long-run coverage holds for any sequence.",
            "conformal", "long_run", (_gamma(),),
            lambda h, a, p: ACI(h, a, p["gamma"]),
        ),
        CalibratorSpec(
            "dtaci", "DtACI", "Gibbs & Candes, JMLR 2024",
            "Runs ACI over a grid of step sizes and reweights the experts online, so "
            "the step size tunes itself.",
            "conformal", "long_run",
            (
                CalibratorParameter(
                    "interval", "Interval", 100, 10, 1000, 10, 1.0,
                    help="Local-regret interval that sets the learning rates.",
                ),
            ),
            lambda h, a, p: DtACI(h, a, DEFAULT_GAMMAS, p["interval"]),
        ),
        CalibratorSpec(
            "agaci", "AgACI", "Zaffran et al., ICML 2022",
            "Aggregates ACI experts with online expert aggregation on the pinball "
            "loss.",
            "conformal", "long_run", (),
            lambda h, a, p: AgACI(h, a, DEFAULT_GAMMAS),
        ),
        CalibratorSpec(
            "qt", "Quantile tracker", "Angelopoulos, Candes & Tibshirani, NeurIPS 2023",
            "Online gradient descent on the pinball loss of the threshold itself: "
            "q += eta (err - alpha).",
            "conformal", "long_run", (_lr(_RANGE),),
            lambda h, a, p: QuantileTracker(h, a, p["lr"], 100),
        ),
        CalibratorSpec(
            "pid", "Conformal PID", "Angelopoulos, Candes & Tibshirani, NeurIPS 2023",
            "Quantile tracker plus a saturated integrator of past coverage errors.",
            "conformal", "long_run",
            (
                _lr(),
                CalibratorParameter(
                    "ki", "Integrator gain", 0.5, 0, 5, 0.05, 0.0,
                    help="In units of the recent score range.",
                ),
            ),
            lambda h, a, p: ConformalPID(h, a, p["lr"], 100, p["ki"], 1000),
        ),
        CalibratorSpec(
            "nexcp", "NexCP", "Barber, Candes, Ramdas & Tibshirani, Ann. Stat. 2023",
            "Weighted split conformal with geometric weights rho^age, robust to drift.",
            "conformal", "long_run",
            (_discount("rho", "Decay", 0.99, "Weight of a score one step older."),),
            lambda h, a, p: NexCP(a, p["rho"], 500),
        ),
        CalibratorSpec(
            "eci", "ECI", "Wu, Hu, Bao, Xia & Zou, ICLR 2025",
            "Quantile tracker plus an error-quantification term that also weighs how "
            "far the score fell from the threshold. Trades some coverage for width.",
            "conformal", "none",
            (
                _lr(_RANGE),
                CalibratorParameter(
                    "c", "Sigmoid slope", 1, 0.05, 50, 0.05, 0.0,
                    exclusive_minimum=True,
                    help="Slope of the smoothed miss indicator, per unit of the "
                    "recent score range.",
                ),
                CalibratorParameter(
                    "h", "Cutoff", 0, 0, 1, 0.01, 0.0,
                    help="Apply the error term only beyond this share of the range.",
                ),
            ),
            lambda h, a, p: ECI(h, a, p["lr"], p["c"], p["h"]),
        ),
        CalibratorSpec(
            "sfogd", "SF-OGD", "Orabona & Pal 2016; Bhatnagar et al., ICML 2023",
            "Scale-free gradient descent on the pinball loss, with no learning rate "
            "to tune.",
            "conformal", "long_run", (),
            lambda h, a, p: SFOGD(h, a),
        ),
        CalibratorSpec(
            "saocp", "SAOCP", "Bhatnagar, Wang, Xiong & Bai, ICML 2023",
            "SF-OGD experts on geometric covering intervals, combined by coin "
            "betting: low regret on every time interval.",
            "conformal", "long_run",
            (
                CalibratorParameter(
                    "g", "Lifetime", 8, 1, 64, 1, 1.0, integer=True,
                    help="An expert started at t lives g * 2^(trailing zeros of t) "
                    "steps.",
                ),
            ),
            lambda h, a, p: SAOCP(h, a, int(p["g"])),
        ),
        CalibratorSpec(
            "belief", "Bayesian belief CP",
            "Zhang, Bombara & Yang, The Benefit of Being Bayesian in Online "
            "Conformal Prediction, 2024",
            "Threshold = quantile of a belief: a prior mixed with the discounted "
            "empirical score distribution.",
            "bayesian", "none",
            (_discount("beta", "Discount", 0.99, "Per-step discount of past scores."),),
            lambda h, a, p: BayesianBelief(a, p["beta"]),
        ),
        CalibratorSpec(
            "bocpd_aci", "BOCPD + ACI", "Adams & MacKay 2007; Gibbs & Candes 2021",
            "Bayesian changepoint predictive, with its level set by ACI so long-run "
            "coverage holds.",
            "bayesian", "long_run", (_run_length(), _gamma()),
            lambda h, a, p: BOCPDACI(h, a, p["lambda"], p["gamma"]),
        ),
        CalibratorSpec(
            "cpcp", "Changepoint CP", "Barber et al. 2023 weights from Adams & MacKay 2007",
            "Weighted split conformal; a past score's weight is the posterior "
            "probability it is in the current regime.",
            "bayesian", "none", (_run_length(),),
            lambda h, a, p: ChangepointCP(h, a, p["lambda"]),
        ),
        CalibratorSpec(
            "bocpd", "BOCPD", "Adams & MacKay 2007",
            "Bayesian online changepoint detection with a Normal-Inverse-Gamma model "
            "per regime; threshold = predictive quantile.",
            "bayesian", "none", (_run_length(),),
            lambda h, a, p: BOCPDQuantile(h, a, p["lambda"]),
        ),
        CalibratorSpec(
            "nig", "Discounted Bayes", "West & Harrison 1997, discount variance learning",
            "Gaussian score with unknown mean and variance, evidence discounted per "
            "step; Student-t predictive quantile.",
            "bayesian", "none",
            (
                _discount(
                    "delta", "Discount", 0.98,
                    "Keeps about 1/(1 - delta) effective observations.",
                ),
            ),
            lambda h, a, p: DiscountedBayes(h, a, p["delta"]),
        ),
    )
}  # fmt: skip
