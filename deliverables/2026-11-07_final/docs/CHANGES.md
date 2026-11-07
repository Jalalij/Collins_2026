# Code changes for the final report

This file lists every change made to the modeling code for the final report of the project and why. "Previous" means the code and results of the earlier Modeling Dynamics report. The previous behavior is still the default everywhere: each new behavior is turned on with a flag or by picking a new method name, so the earlier runs in `modeling/results/` can still be reproduced.

The changes came in two rounds. Round 1 (2026-11-01 to 11-05) built the benchmark and the final forecaster. Round 2 (2026-11-06 to 11-07) added the analyses that explain the results, the comparison models that bound how specific the gain is, the leak experiment, and the scheduler emulation.

## Round 2: analyses and comparison models (2026-11-06/07)

### New methods in `modeling/experiments/final_benchmark.py`

| Method | What it is | Why |
|---|---|---|
| `fr_dist_w4` | The final forecaster with pinball weight 4 instead of 1 in the loss. | Sensitivity of the main result to the weight. The weight is compared on the calibration sessions, never on test. |
| `fr_gauss` | The final forecaster with Gaussian heads (mean and SD per step and target, Gaussian negative log-likelihood). Its quantiles are those of the normal distribution. | Shows how much of the gain needs free (asymmetric) quantiles. |
| `lqr_fr` | A linear map from the ridge design (last input row + 30-row target history) to the outputs of one final-model head, with the same ordered quantiles and the same loss. | The linear twin of the final model. It shows how much of the gain the distributional output gives without a network. |

New options: `--qweight`, `--hidden_size`, `--train_stride`, `--save_models`, `--legacy_data` with `--eval_causal` (leak experiment: train on the previous interpolated inputs, test on both input versions, with an assertion that the targets are identical). `SplitData` now also keeps the time stamp, the session id and the handover flag of each window.

### New code in `modeling/models/seq2seq_dist.py`
`GaussHead`, `ThroughputSeq2SeqGauss` and `gauss_nll` for the Gaussian comparison model. The output layout is the same 6 channels as the final model, so evaluation and calibration code is shared.

### New analysis scripts (`modeling/experiments/`, outputs in `modeling/results_final/analysis/<name>/`)

| Script | Question it answers | Main outputs |
|---|---|---|
| `fr_dataset_stats.py` | How large is the data set? | `dataset_stats.json` |
| `fr_volatility.py` | What is predictable: the next change, the level, or the spread? | `volatility/acf_summary.csv`, `window_corr.csv`, `deciles.csv`, figure `fr_volatility.tex` |
| `fr_condcov.py` | Does each interval cover 95% in calm and in volatile windows? Where does the interval gain come from? | `condcov/summary.csv`, `pairs.csv`, figures `fr_condcov.tex`, `fr_condwidth.tex`, table `fr_condcov_tab.tex` |
| `fr_bootstrap.py` | Are the gains larger than session-to-session chance? | `bootstrap/bootstrap_summary.csv`, `main_vs_others.csv`, `is_decomposition.csv`, figure `fr_bootstrap.tex` |
| `fr_scheduler.py` | What do the forecasts buy a deadline-aware duplication scheduler? | `scheduler/headline.csv`, `link_<link>.json`, figures `fr_scheduler*.tex` |
| `fr_attention.py` | What does the model look at and what does it use? | `attention/attn_summary.csv`, figures `fr_attention*.tex` |
| `fr_leak.py` | How much does the interpolation leak flatter a trained model? | `leak/leak_summary.json`, figure `fr_leak.tex` |
| `fr_trace.py` | Intuition figure: one minute with both calibrated intervals | `trace/trace_starlink_dl.json`, figure `fr_trace.tex` |
| `fr_figures.py` | Template-style figures: horizon curves with zoom insets, link x horizon heatmaps | figures `fr_hzoom.tex`, `fr_heat.tex`, `fr_heat_col.tex` |
| `fr_paper_numbers.py` | Every number of the paper that comes from these analyses, with its source file | `paper_twc/figs/data/numbers_story.tex`, `tab_main_boot.tex`, `tab_extent.tex`, `results_final/analysis/paper_numbers.json` |

Every threshold, decile edge, conformal offset and knob in these scripts is set on the calibration (validation) sessions. Test sessions are only scored. Confidence intervals resample whole test sessions, because the windows of one session overlap.

### Run scripts
`modeling/results_final/main/jobs_v2.txt` and `scheduler_v2.sh` (the 42 round-2 training jobs, at most 6 at a time and 2 on 5G), `run_leak.sh` (the leak experiment, 6 jobs). `reports/final_report/build_figs.py` compiles the analysis figures of the report as standalone PDFs (the report has more pgfplots figures than TeX's memory holds in one run). `paper_twc/make_paper_tables.py` copies the tables, figure data and figures into the paper.

### Known issue
`SplitData.ts` in `final_benchmark.py` is 1000 times too small, because pandas 3 parses the time stamps at microsecond resolution. No metric uses it. `fr_bootstrap.py` converts it correctly (`unix_seconds()`).

## Round 1: benchmark and final forecaster (2026-11-01 to 11-05)

### New files

| File | What it does |
|---|---|
| `modeling/models/seq2seq_dist.py` | **Final forecaster (DS2S).** It keeps the previous encoder and attention and anchors the decoder on the last observation. Each target head outputs a mean and the quantiles 0.025, 0.25, 0.5, 0.75 and 0.975 of the change from the last observation. The quantiles are built from softplus gaps around the median, so they cannot cross. The loss (`dist_loss`) is the squared error of the mean plus the average pinball loss of the quantiles. |
| `modeling/experiments/final_benchmark.py` | One script that trains and scores every method on the previous session split (seed 42, 70/15/15). It builds windows on the GPU from contiguous arrays, which makes 5G epochs about 10x faster, with values identical to `ThroughputFileDataset`. Methods: `persistence`, `snaive`, `ridge`, `ridge_fr`, `prev_seq2seq`, `fr_resid`, `fr_full`, `fr_dist` (plus the round-2 methods above). Metrics per target and horizon step: MAE, RMSE, R², skill against persistence, split-conformal and CQR intervals (PICP, interval score), and the previous test-quantile intervals for comparison. |
| `modeling/experiments/final_aggregate.py` | Averages over seeds, builds 3-seed ensembles (mean and every quantile averaged), computes relative changes, and writes the pgfplots `.dat` files. |
| `modeling/experiments/final_extras.py` | Spike-warning AUROC, Starlink metrics on measured pings only, the trace data, and the CQR offsets of each ensemble for deployment. |
| `modeling/experiments/final_tables.py` | Writes the LaTeX tables and number macros of the report and paper from `results_final/summary/`. |
| `modeling/experiments/final_audit_numbers.py` | Writes the macros of the audit section (reproduction of the previous results, effect of the ping fix, parameter counts). |
| `modeling/results_final/main/run_job.sh`, `scheduler.sh`, `jobs.txt` | The 78 round-1 jobs (6 links x [4 baselines + 4 neural variants x 3 seeds]). |

### Changed files

`modeling/dataset.py` (all off by default, `DataConfig` flags):
- `causal_ping`: the Starlink parser fills missing `ping_server_rtt_ms` rows by time interpolation toward the **next** real ping. `causal_ping_fix()` finds these rows (a real ping has at most 3 decimals, an interpolated value has more), keeps the measured series as `<col>__orig` for the targets, replaces the input with the last real ping, and stores an `__interp` mask. See `DATA_AUDIT.md`, A3.
- `clean`: radio values outside physical ranges are set to missing and forward filled, for example the SINR logger sentinel -32768 in some 5G files.
- `engineered`: causal per-session context features for each target: rows since the last change, last-1 s mean and max, last-30 s and last-60 s mean, last-60 s SD, session-to-date mean, plus `row_dt`.
- `clip_z`: after z-scoring, the non-target inputs are clipped to ±clip_z (5 in this study).

`modeling/evaluation/metrics.py`: `r2()` returns NaN for a constant target instead of 1.0 (the previous aggregate averaged R² = 1 for the all-zero OneWeb UL jitter and loss, A4).

`modeling/models/seq2seq.py` (already on the branch before this work, evaluated here as the ablation S2S-A): the decoder starts from the last observed values and predicts the change (residual form). Teacher forcing is decided per sample, and gradients are not detached through fed-back predictions.

`final_benchmark.py`, fix found while running: on Apple GPUs (MPS) the previous decoder kept about 2 GB per inference batch in the allocator cache, and the 5G validation pass ran out of memory. `predict()` now synchronizes and empties the MPS cache after each batch.

### Not changed
`run.py`, `train.py`, `inference.py` and the other model files behave as in the previous study. The previous Seq2Seq is `models/seq2seq_prev.py`, trained through `final_benchmark.py --methods prev_seq2seq`.
