# Audit of the Year 2 pipeline

The Year 2 evaluation was rebuilt in `modeling/experiments/y3_benchmark.py`. On the original data it reproduces the Year 2 Seq2Seq-GRU results:

| | Year 2 report (MAE / RMSE / R²) | Reproduced, seed 0 |
|---|---|---|
| Starlink DL | 8.55 / 14.95 / 0.406 | 8.36 / 14.84 / 0.403 |
| OneWeb DL | 6.12 / 23.73 / 0.217 | 6.62 / 24.14 / 0.211 |

Sources: `modeling/results_y3/explore/{starlink_dl,oneweb_dl}/`. The Starlink run used the data before the A3 fix.

Three reviewers then read the script and the data pipeline independently, each from a different angle (leakage, metrics, training). Two more reviewers tried to refute every finding. The findings that survived are listed below.

| # | Issue | Evidence | Fix |
|---|---|---|---|
| A1 | The horizon was labeled in seconds, but the steps are 200 ms. "t+15s" is 3 s ahead. | The median row spacing is 0.200 s on all six links. | Horizons are now reported as h × 0.2 s. |
| A2 | Year 2 built the 95% intervals from quantiles of the **test** residuals, so PICP = 0.95 by construction. | `evaluation/metrics.py::prediction_interval` | Split-conformal and CQR intervals are calibrated on the validation sessions and scored on the test sessions. |
| A3 | The Starlink parser interpolates missing server pings toward the next real ping (`parse_starlink.py`, `INTERP_COLS`, `interpolate(method="time")`). 11.9% of Starlink DL and 1.2% of UL test RTT targets are interpolated. In every test window whose last input row is interpolated (11.9% of DL windows, 1.2% of UL), that value lies strictly between the previous row and the first target, against 47% (DL) and 34% (UL) for real pings. | `results_y3/audit/interp_check.json` (`experiments/y3_interp_check.py`). With the fix, the persistence RTT RMSE at the first step rises from 56.7 to 62.2 ms (`results_y3/audit/{legacy,causal}/starlink_dl/`). | `DataConfig(causal_ping=True)` for every method in the study. The inputs use the last real ping and the targets stay the measured series. The 5G and OneWeb RTT have no interpolated values. |
| A4 | OneWeb UL jitter and loss are zero in every row. Year 2 reported R² = 1 for them and included that in the aggregate. | Data check: 0 non-zero values in 65 files. | R² is NaN for constant targets. Per-target results are reported. |
| A5 | The Year 2 "relative RMSE" compared the model with a lag-1 shift of the flattened multi-target, multi-horizon series, not with a persistence forecast. | `evaluation/metrics.py::relative_rmse` | Skill is now computed against ŷ(t+h) = y(t) for each target and step. |
| A6 | Each Year 2 result comes from a single training run. The report and the stored run differ for OneWeb UL (MAE 22.81 vs 23.93). | `results/oneweb_llt_ul/comparison.csv` | 3 seeds per neural model, plus ensembles. |

Other findings, kept as limitations:
- **Persistence on Starlink DL RTT.** The RTT repeats with a one-second period under the 1 Hz iperf load, so persistence is a weak baseline there. A seasonal repeat baseline (`snaive`) was added.
- **iperf3 timestamps.** Reports are stamped at the start of each 1 s interval and snapped to ping rows. An iperf input at row t can therefore describe traffic up to about 1 s later. This alignment is shared by all methods and by Year 2. The clean fix is in the parsers: stamp each report at the end of its interval.
- **Calibration set reuse.** The validation sessions are used both for early stopping and for conformal calibration. Exact coverage is therefore not guaranteed. The coverage measured on the test sessions is reported instead.
