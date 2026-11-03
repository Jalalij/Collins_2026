# Reproducing the Year 3 results

Hardware used: Apple M3 Max (48 GB) with PyTorch 2.13 on the MPS backend. The full study takes about 7 hours with 6 parallel jobs. The 5G jobs are the slowest.

```bash
cd modeling
pip install -r requirements.txt   # plus scikit-learn for the spike AUROC

# 1. The study: 78 jobs (6 links x [4 baselines + 4 neural variants x 3 seeds])
sh results_y3/main/scheduler.sh            # or: cat results_y3/main/jobs.txt | xargs -P 6 -L 1 results_y3/main/run_job.sh

# 2. Summaries, ensembles, figure data
python3 experiments/y3_aggregate.py         # results_y3/summary/{runs,summary,relative_change,horizon_*}.csv + reports/year3/data/h_*.dat
python3 experiments/y3_extras.py            # spike AUROC, real-ping metrics, trace data, CQR offsets

# 3. Audit evidence
python3 experiments/y3_benchmark.py --provider starlink --direction dl --methods persistence --legacy_data --out_dir results_y3/audit/legacy
python3 experiments/y3_benchmark.py --provider starlink --direction dl --methods persistence snaive ridge ridge_y3 --out_dir results_y3/audit/causal
python3 experiments/y3_interp_check.py      # results_y3/audit/interp_check.json

# 4. Tables and number macros for the report and paper
python3 experiments/y3_audit_numbers.py
python3 experiments/y3_tables.py

# 5. Documents (Tectonic or pdfLaTeX + BibTeX)
cd ../reports/year3 && tectonic -X compile main.tex
cd ../../paper_twc && tectonic -X compile main.tex
```

A single run, for example the Year 3 model on Starlink DL with seed 0:
```bash
python3 experiments/y3_benchmark.py --provider starlink --direction dl --methods y3_dist --seeds 0 \
    --epochs 60 --patience 15 --save_models --out_dir results_y3/main
```
On 5G add `--train_stride 4`. `--legacy_data` keeps the parser-interpolated Starlink pings, which is the exact Year 2 input.

## Using a trained Year 3 model
`results_y3/main/<link>/models/y3_dist_s<seed>.pt` holds the state dict together with the feature list, the feature and target normalization statistics, the clipping settings and the data variant. `results_y3/summary/cqr_offsets_<link>.json` gives the conformal offsets `c[h][k]` of the 3-seed ensemble. The 95% interval is `[q0.025 - c, q0.975 + c]` of the seed-averaged quantiles.
