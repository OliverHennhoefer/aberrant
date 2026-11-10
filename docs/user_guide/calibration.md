# Score Calibration

A detector's score says how unusual an event is, but not where to draw the
alarm line. `aberrant.calibration` learns that line online: a threshold that
`1 - alpha` of future scores should stay below, which keeps adapting while
the score distribution shifts.

```python
from aberrant.calibration import ScoreCalibrator

calibrator = ScoreCalibrator("aci", alpha=0.01, window=500)
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
thresholds are clipped to the largest absolute score seen. Scores may be
negative, as with conformalized quantile regression.

## Methods

`CALIBRATORS` lists each method with its parameters, its family and whether
long-run coverage is guaranteed for any score sequence.

| Method | Id | Guarantee | Idea |
| --- | --- | --- | --- |
| Rolling split CP | `split` | Under exchangeability | Conformal quantile of the last W scores |
| ACI | `aci` | Yes | Adapts the working level by its misses |
| DtACI | `dtaci` | Yes | ACI over a grid of step sizes, reweighted online |
| AgACI | `agaci` | Yes | ACI experts aggregated on the pinball loss |
| Quantile tracker | `qt` | Yes | Gradient descent on the threshold's pinball loss |
| Conformal PID | `pid` | Yes | Quantile tracker plus an error integrator |
| NexCP | `nexcp` | Yes | Split conformal with geometric age weights |
| ECI | `eci` | Biased | Quantile tracker plus an error-size term |
| SF-OGD | `sfogd` | Yes | Scale-free gradient descent, no learning rate |
| SAOCP | `saocp` | Yes | SF-OGD experts on covering intervals, coin betting |
| Bayesian belief CP | `belief` | No | Quantile of a prior mixed with discounted scores |
| BOCPD + ACI | `bocpd_aci` | Yes | Changepoint predictive at an ACI-adapted level |
| Changepoint CP | `cpcp` | No | Split conformal weighted by regime probability |
| BOCPD | `bocpd` | No | Changepoint predictive quantile |
| Discounted Bayes | `nig` | No | Student-t predictive with discounted evidence |

Each parameter's `minimum` and `maximum` bound what the method accepts, and
`low` and `high` give the range worth offering in an editor.

## Exact p-values

These methods return thresholds. A conformal p-value with a finite-sample
guarantee needs scores that stop changing after calibration, as with a split
calibrated on frozen detectors; the adaptive methods trade that exactness for
tracking a moving distribution.

## Provenance

The methods are ports of the online conformal and Bayesian methods compared
in the Timestral Uncertainty Arena. The tests replay fixtures generated from
its TypeScript and require the same thresholds.
