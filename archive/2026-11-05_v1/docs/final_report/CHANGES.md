# final code changes

This file lists every change made to the modeling code in final and why. The initial behavior is still the default everywhere. Each new behavior has to be turned on with a flag or by picking a new model, so the initial runs in `modeling/results/` can still be reproduced.

## New files

| File | What it does |
|---|---|
| `modeling/models/seq2seq_dist.py` | **final forecaster.** It keeps the initial encoder, attention and residual decoder. Each target head outputs a mean and the quantiles 0.025, 0.25, 0.5, 0.75 and 0.975 of the change from the last observation. The quantiles are built from softplus gaps around the median, so they cannot cross. The loss (`dist_loss`) is squared error on the mean plus the average pinball loss of the quantiles. |
| `modeling/experiments/final_benchmark.py` | One script that trains and scores every method on the initial session split (seed 42, 70/15/15). It builds windows on the GPU from contiguous arrays, which makes 5G epochs about 10x faster, and the values are identical to `ThroughputFileDataset`. Methods: `persistence`, `snaive`, `ridge`, `ridge_fr`, `prev_seq2seq`, `fr_resid`, `fr_full`, `fr_dist`. Metrics are reported per target and per horizon step: MAE, RMSE, R², skill against persistence, split-conformal and CQR intervals (PICP, interval score), and the initial test-quantile intervals for comparison. |
| `modeling/experiments/final_aggregate.py` | Averages over seeds, builds 3-seed ensembles (mean and every quantile averaged), computes relative changes, and writes the pgfplots `.dat` files. |
| `modeling/experiments/final_extras.py` | Spike-warning AUROC, Starlink metrics on measured pings only, the trace data for the report figure, and the CQR offsets of each ensemble for deployment. |
| `modeling/experiments/final_tables.py` | Writes every LaTeX table and number macro of the report and paper from `results_final/summary/`. |
| `modeling/experiments/final_audit_numbers.py` | Writes the macros of the audit section (reproduction of the earlier results, effect of the ping fix, parameter counts). |
| `modeling/results_final/main/run_job.sh`, `scheduler.sh`, `jobs.txt` | The 78 jobs of the study (6 links × [4 baselines + 4 neural variants × 3 seeds]). The scheduler runs at most 6 jobs at a time and at most 2 5G jobs. |

## Changed files

### `modeling/dataset.py`
All of the following are off by default (`DataConfig` flags):
- `causal_ping` (new): the Starlink parser fills missing `ping_server_rtt_ms` rows by time interpolation toward the **next** real ping. `causal_ping_fix()` finds these rows: a real ping has at most 3 decimals and an interpolated value has more. It keeps the measured series as `<col>__orig`, which `target_source()` uses as the target. It replaces the input with the last real ping, and it stores an `__interp` mask. See `DATA_AUDIT.md`, A3.
- `clean`: radio values outside physical ranges are set to missing and forward filled. For example, SINR −32768 is a logger sentinel in some 5G files.
- `engineered`: causal per-session context features for each target. These are rows since the last change, last-1 s mean and max, last-30 s and last-60 s mean, last-60 s std, session-to-date mean, plus `row_dt`.
- `clip_z`: after z-scoring, the non-target inputs are clipped to ±clip_z (final uses 5).
- `ThroughputFileDataset` and `build_dataloaders` read targets through `target_source()`, so the measured series is used when `causal_ping` is on.

### `modeling/evaluation/metrics.py`
- `r2()` returns NaN for a constant target instead of 1.0. The initial aggregate averaged R² = 1 for the all-zero OneWeb UL jitter and loss (A4).

### `modeling/models/seq2seq.py` (already on the branch before the final report, now evaluated)
- The decoder starts from the last observed target values instead of zeros and predicts the change (residual form).
- Teacher forcing is decided per sample, and gradients are not detached through fed-back predictions.

### `modeling/experiments/final_benchmark.py`: fixes found while running the study
- On Apple GPUs (MPS), the initial decoder kept about 2 GB per inference batch in the allocator cache. On the 5G validation set (about 166k windows) the process was killed for running out of memory. `predict()` now empties the MPS cache after each batch.

## Not changed
- `run.py`, `train.py`, `inference.py` and the other model files behave as in the initial study.
- The initial Seq2Seq is `models/seq2seq_prev.py`. The study trains it through `final_benchmark.py --methods prev_seq2seq`.
