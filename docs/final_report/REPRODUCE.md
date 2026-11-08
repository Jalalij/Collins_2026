# Reproducing the final report and the paper

Hardware used: Apple M3 Max (48 GB) with PyTorch on the MPS backend. Round 1 (the main study) takes about 7 hours with 6 parallel jobs, round 2 (comparison models and leak experiment) about 6 hours. The 5G jobs are the slowest. All analyses run on the CPU from the saved predictions.

**Prediction arrays only.** The trained models are in the repository but `results_final/main/*/preds/` (about 6.5 GB) is not. To rebuild it without retraining (about 10 minutes), after unpacking `data_release/` into `modeling/data/`:
```bash
cd modeling && python3 experiments/regen_preds.py
```
It loads `models/<method>_s<seed>.pt`, recomputes the linear baselines, and checks every file against the saved metrics CSVs (`results_final/main/regen_preds_check.csv`).

```bash
cd modeling
pip install -r requirements.txt   # plus scikit-learn

# 1. Main study: 78 jobs (6 links x [4 baselines + 4 neural variants x 3 seeds])
sh results_final/main/scheduler.sh            # or: cat results_final/main/jobs.txt | xargs -P 6 -L 1 results_final/main/run_job.sh

# 2. Round 2: Gaussian heads, pinball weight 4, linear quantile model (42 jobs) and the leak experiment (6 jobs)
sh results_final/main/scheduler_v2.sh         # runs jobs_v2.txt and results_final/main/run_leak.sh

# 3. Summaries, ensembles, figure data
python3 experiments/final_aggregate.py         # results_final/summary/*.csv + reports/final_report/data/h_*.dat
python3 experiments/final_extras.py            # spike AUROC, real-ping metrics, trace data, CQR offsets

# 4. Audit evidence
python3 experiments/final_benchmark.py --provider starlink --direction dl --methods persistence --legacy_data --out_dir results_final/audit/legacy
python3 experiments/final_benchmark.py --provider starlink --direction dl --methods persistence snaive ridge ridge_fr --out_dir results_final/audit/causal
python3 experiments/final_interp_check.py

# 5. Analyses (each writes results_final/analysis/<name>/ and its figures to reports/final_report/)
python3 experiments/fr_dataset_stats.py
LSN_DEVICE=cpu python3 experiments/fr_volatility.py
LSN_DEVICE=cpu python3 experiments/fr_condcov.py
LSN_DEVICE=cpu python3 experiments/fr_bootstrap.py
LSN_DEVICE=cpu python3 experiments/fr_scheduler.py
LSN_DEVICE=cpu python3 experiments/fr_attention.py
python3 experiments/fr_leak.py
python3 experiments/fr_trace.py
python3 experiments/fr_figures.py

# 6. Tables and number macros (report and paper)
python3 experiments/final_audit_numbers.py
python3 experiments/final_tables.py
python3 experiments/fr_paper_numbers.py        # every number of the paper, with its source file

# 7. Documents (Tectonic, or pdfLaTeX + BibTeX)
cd ../reports/final_report && python3 build_figs.py && tectonic -X compile main.tex
cd ../../paper_twc && python3 make_paper_tables.py && tectonic -X compile main.tex && python3 check_style.py
```

A single run, for example the final model on Starlink DL with seed 0:
```bash
python3 experiments/final_benchmark.py --provider starlink --direction dl --methods fr_dist --seeds 0 \
    --epochs 60 --patience 15 --save_models --out_dir results_final/main
```
On 5G add `--train_stride 4`. `--legacy_data` keeps the parser-interpolated Starlink pings of the previous pipeline, and `--eval_causal` then also scores the model on causal test inputs.

## Where each number comes from
`modeling/results_final/analysis/paper_numbers.json` maps every macro of the paper (`paper_twc/figs/data/numbers_story.tex`) to its value and source file. The older macros (`numbers_audit.tex`, `numbers_results.tex`, `numbers_analysis.tex`) are written by `final_audit_numbers.py` and `final_tables.py` from `results_final/summary/`.

## Using a trained final model
`results_final/main/<link>/models/fr_dist_s<seed>.pt` holds the state dict together with the feature list, the feature and target normalization statistics, the clipping settings and the data variant. `results_final/summary/cqr_offsets_<link>.json` gives the conformal offsets `c[h][k]` of the 3-seed ensemble. The 95% interval is `[q0.025 - c, q0.975 + c]` of the seed-averaged quantiles.
