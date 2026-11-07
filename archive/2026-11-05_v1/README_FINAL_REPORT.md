# Collins final report (the initial study): link-state forecasting

| What | Where |
|---|---|
| final report (PDF + LaTeX, TikZ/pgfplots figures) | `reports/final_report/Collins_Final_Report_LinkState_Forecasting.pdf`, `reports/final_report/` |
| IEEE TWC paper draft (PDF + LaTeX) | `paper_twc/main.pdf`, `paper_twc/` |
| Supplementary docs | `docs/final_report/CHANGES.md`, `DATA_AUDIT.md`, `REPRODUCE.md` |
| final forecaster | `modeling/models/seq2seq_dist.py` |
| Benchmark, analyses, table/figure generators | `modeling/experiments/` |
| Per-run results, settings, trained models (`*.pt`, Git LFS) | `modeling/results_final/main/<link>/` |
| Sensitivity sweeps | `modeling/results_final/sweeps/` |
| Summaries (CSV) and deployment CQR offsets | `modeling/results_final/summary/` |
| Measurement data (Git LFS archives) | `data_release/` |
| Earlier report sources (baseline) | `reports/year2_sources/` |

Headline (test sessions, 6 links, 3 seeds, against the initial Seq2Seq retrained on the same corrected data): 95% interval score of the RTT forecasts 8-39% better, RTT MAE 2-18% lower, RTT RMSE about unchanged. See the report for the full results and the audit of the initial pipeline.

Test-set prediction arrays (`results_final/main/*/preds/`, about 3 GB) are not in the repository. `docs/final_report/REPRODUCE.md` regenerates them.
