"""The calibration methods, their parameters and how to build them."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Protocol, cast

from aberrant.base.descriptors import (
    ParameterDescriptor,
    ValueDescriptor,
    parameters_schema,
    resolve_parameters,
)
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
from aberrant.calibration._extreme import SPOT
from aberrant.calibration._window import ScoreHistory


class OnlineCalibrator(Protocol):
    """A threshold on one score stream, learned with delayed feedback.

    ``predict(key)`` issues a threshold when a prediction is made, and
    ``update(key, score)`` arrives once that prediction's score is known.
    A score above the threshold is a miss at the target level ``1 - alpha``.
    """

    def predict(self, key: object) -> float: ...

    def update(self, key: object, score: float) -> None: ...


def _parameter(
    name: str,
    title: str,
    default: float,
    *,
    minimum: float,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
    exclusive_maximum: bool = False,
    integer: bool = False,
    description: str = "",
) -> ParameterDescriptor:
    value = ValueDescriptor.number(
        integer=integer,
        minimum=minimum,
        maximum=maximum,
        exclusive_minimum=exclusive_minimum,
        exclusive_maximum=exclusive_maximum,
    )
    return ParameterDescriptor(
        name, value, int(default) if integer else float(default), title, description
    )


Coverage = Literal["exchangeable", "long_run", "none"]
"""What a method's threshold guarantees: finite-sample coverage under
exchangeable scores, long-run average coverage, or no guarantee."""
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

    A method that ``extrapolates`` may issue a threshold above every score
    seen; the others are clipped to the largest absolute score.
    """

    id: str
    name: str
    reference: str
    summary: str
    family: Literal["conformal", "bayesian", "extreme_value"]
    coverage: Coverage
    parameters: tuple[ParameterDescriptor, ...]
    factory: Factory
    extrapolates: bool = False

    def defaults(self) -> dict[str, float]:
        """Every parameter at its default."""
        return self.resolve({})

    def resolve(self, params: Mapping[str, object]) -> dict[str, float]:
        """Defaults overridden by ``params``, each value validated.

        Raises:
            ConfigurationError: A parameter is unknown or out of range.
        """
        return cast(
            "dict[str, float]",
            resolve_parameters(self.parameters, params, label=self.id),
        )

    def schema(self) -> dict[str, object]:
        """The JSON Schema of the parameters, as the catalog gives components."""
        return parameters_schema(self.parameters)


def _gamma() -> ParameterDescriptor:
    return _parameter("gamma", "Step size", 0.01, minimum=0.0, exclusive_minimum=True)


def _lr(description: str = "") -> ParameterDescriptor:
    return _parameter(
        "lr", "Learning rate", 0.05, minimum=0.0, exclusive_minimum=True,
        description=description,
    )  # fmt: skip


def _run_length() -> ParameterDescriptor:
    return _parameter(
        "lambda", "Run length", 250, minimum=1.0,
        description="Expected steps between changepoints (hazard 1/lambda).",
    )  # fmt: skip


def _discount(
    name: str, title: str, default: float, description: str
) -> ParameterDescriptor:
    return _parameter(
        name, title, default, minimum=0.0, maximum=1.0,
        exclusive_minimum=True, exclusive_maximum=True, description=description,
    )  # fmt: skip


_RANGE = "Step as a fraction of the recent score range."


def _level() -> ParameterDescriptor:
    return _parameter(
        "level", "Initial threshold", 0.98, minimum=0.0, maximum=1.0,
        exclusive_minimum=True, exclusive_maximum=True,
        description="Quantile of the window above which scores count as "
        "peaks; lowered when that leaves fewer than ten.",
    )  # fmt: skip


CALIBRATORS: Mapping[str, CalibratorSpec] = MappingProxyType({
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
                _parameter(
                    "interval", "Interval", 100, minimum=1.0,
                    description="Local-regret interval that sets the learning rates.",
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
                _parameter(
                    "ki", "Integrator gain", 0.5, minimum=0.0,
                    description="In units of the recent score range.",
                ),
            ),
            lambda h, a, p: ConformalPID(history=h, alpha=a, lr=p["lr"], scale_window=100, ki=p["ki"], horizon_guess=1000),
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
                _parameter(
                    "c", "Sigmoid slope", 1, minimum=0.0, exclusive_minimum=True,
                    description="Slope of the smoothed miss indicator, per unit "
                    "of the recent score range.",
                ),
                _parameter(
                    "h", "Cutoff", 0, minimum=0.0,
                    description="Apply the error term only beyond this share of "
                    "the range.",
                ),
            ),
            lambda h, a, p: ECI(history=h, alpha=a, lr=p["lr"], c=p["c"], cutoff=p["h"]),
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
                _parameter(
                    "g", "Lifetime", 8, minimum=1, integer=True,
                    description="An expert started at t lives "
                    "g * 2^(trailing zeros of t) steps.",
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
            lambda h, a, p: ChangepointCP(history=h, alpha=a, lam=p["lambda"]),
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
        CalibratorSpec(
            "spot", "SPOT", "Siffer, Fouque, Termier & Largouet, KDD 2017",
            "Generalized Pareto tail fitted to the peaks of the last W scores; "
            "extrapolates thresholds for alpha below 1/W.",
            "extreme_value", "none", (_level(),),
            lambda h, a, p: SPOT(h, a, p["level"]),
            extrapolates=True,
        ),
        CalibratorSpec(
            "dspot", "DSPOT", "Siffer, Fouque, Termier & Largouet, KDD 2017",
            "SPOT on the deviation from the mean of the last d scores, so a "
            "drifting score level does not count as extreme.",
            "extreme_value", "none",
            (
                _level(),
                _parameter(
                    "depth", "Depth", 10, minimum=1, integer=True,
                    description="Scores in the moving mean.",
                ),
            ),
            lambda h, a, p: SPOT(h, a, p["level"], int(p["depth"])),
            extrapolates=True,
        ),
    )
})  # fmt: skip
"""Every calibration method by ID, read-only."""
