# Score Calibration

A detector's score says how unusual an event is, but not where to draw the
alarm line. `aberrant.calibration` learns that line online: a threshold that
`1 - alpha` of future scores should stay below, which keeps adapting while
the score distribution shifts.

```python
from aberrant.calibration import ScoreCalibrator

calibrator = ScoreCalibrator("aci", alpha=0.01, window_size=500)
for t, score in enumerate(scores):
    alarm = score > calibrator.predict(t)
    calibrator.update(t, score)
```

## Delayed feedback

`predict(key)` issues a threshold for one prediction and `update(key, score)`
reports that prediction's score whenever it is known. For an anomaly score
both happen at the same step. For a forecast at lead `k`, the threshold is
issued at `t - k` and its score arrives with the reading at `t`. Keys are any
hashable value; a method that adapts to its own misses compares each score
with the threshold it issued for that key.

Before the first score `predict` returns `+inf`, so nothing alarms. Later
thresholds are clipped to the largest absolute score seen, except those of
the extreme-value methods, which may extrapolate beyond it. Scores may be
negative, as with conformalized quantile regression.

## Methods

`CALIBRATORS` lists each method with its parameters, its family and its
`coverage`, the promise its threshold makes about the score stream:

- `exchangeable`: a new score exceeds the threshold with probability at most
  `alpha` when it is exchangeable with the calibration scores.
- `long_run`: the fraction of scores above the threshold tends to `alpha`
  for any score sequence.
- `none`: no guarantee; the method is calibrated only in practice.

| Method | Id | Coverage | Idea |
| --- | --- | --- | --- |
| Rolling split CP | `split` | Exchangeable | Conformal quantile of the last W scores |
| ACI | `aci` | Long run | Adapts the working level by its misses |
| DtACI | `dtaci` | Long run | ACI over a grid of step sizes, reweighted online |
| AgACI | `agaci` | Long run | ACI experts aggregated on the pinball loss |
| Quantile tracker | `qt` | Long run | Gradient descent on the threshold's pinball loss |
| Conformal PID | `pid` | Long run | Quantile tracker plus an error integrator |
| NexCP | `nexcp` | Long run | Split conformal with geometric age weights |
| ECI | `eci` | None | Quantile tracker plus an error-size term |
| SF-OGD | `sfogd` | Long run | Scale-free gradient descent, no learning rate |
| SAOCP | `saocp` | Long run | SF-OGD experts on covering intervals, coin betting |
| Bayesian belief CP | `belief` | None | Quantile of a prior mixed with discounted scores |
| BOCPD + ACI | `bocpd_aci` | Long run | Changepoint predictive at an ACI-adapted level |
| Changepoint CP | `cpcp` | None | Split conformal weighted by regime probability |
| BOCPD | `bocpd` | None | Changepoint predictive quantile |
| Discounted Bayes | `nig` | None | Student-t predictive with discounted evidence |
| SPOT | `spot` | None | Generalized Pareto tail of the window's peaks |
| DSPOT | `dspot` | None | SPOT on the deviation from a moving mean |

`CalibratorSpec.schema()` gives each method's parameters as JSON Schema,
with the bounds the method accepts.

## Very small alarm rates

A window of `W` scores cannot place a conformal quantile above its largest
score, so for `alpha` below `1 / W` the window methods fall back to the
largest absolute score ever seen. That threshold never comes down again.

SPOT and DSPOT (Siffer et al., KDD 2017) fit a generalized Pareto
distribution to the window's peaks, the scores above its `level` quantile,
and read the threshold off that tail. They extrapolate beyond the window, and
the threshold follows the scores when they shrink. DSPOT calibrates the
deviation from the mean of the last `depth` scores, for a score level that
drifts.

On exponential scores with `alpha = 1e-4` and `W = 2000`, the mean SPOT
threshold is within 2% of the true quantile, and after the scores halve it
falls to the new quantile, while split, ACI and NexCP stay at the old record.
The threshold is an estimate from a few dozen peaks, so its noise lets 2 to 3
times `alpha` of the scores through. Where `alpha * W` is 1 or more, the rate
matches `alpha`. A larger window means more peaks and a steadier threshold.

## From thresholds to alarms

A calibrator is the middle of three stages. The detector scores how unusual
each observation is, the calibrator keeps a boundary on those scores, and a
detection rule decides which exceedances justify an alarm.

Long-run coverage at `alpha = 0.1` means the share of steps where the score
stays at or below the boundary, `mean(s_t <= q_t)` over `t = 1..T`, tends to
0.9. About 10% of all scores exceed the boundary over the long run. It is
not one exceedance in every ten steps, nor a 10% exceedance probability at
each step.

That frequency counts normal and anomalous scores together. On a stream
that is entirely normal every exceedance is a false positive, so the
false-positive rate is `alpha`. On a mixed stream, controlling the total
says nothing about the fraction of normal scores above the boundary. A
false-positive guarantee needs the boundary tied to normal data: an
`exchangeable` method calibrated on a reference window known to be normal,
with the detector frozen afterwards so the scores stay exchangeable. A
smaller `alpha` changes how often scores exceed the boundary but does not
create that tie by itself.

The detection rule needs its own assessment. A single extreme exceedance, a
run of exceedances, their size, or how hard they push the boundary to adapt
are all reasonable rules, and none inherits `alpha`. Three consecutive
exceedances do not alarm with probability `alpha ** 3`, because coverage
does not make exceedances independent. Detection also differs by unit: the
alarm budget bounds pointwise recall, but an incident alarm raised at the
first exceedance can stay active after an adaptive boundary has caught up
with a fault.

## Provenance

The methods are ports of the online conformal and Bayesian methods compared
in the Timestral Uncertainty Arena. The tests replay fixtures generated from
its TypeScript and require the same thresholds.
