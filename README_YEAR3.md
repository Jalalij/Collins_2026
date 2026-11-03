# Collins Year 3: link-state forecasting

| What | Where |
|---|---|
| Year 3 report (PDF + LaTeX, TikZ/pgfplots figures) | `reports/year3/Collins_Year3_LinkState_Forecasting_Report.pdf`, `reports/year3/` |
| IEEE TWC paper draft (PDF + LaTeX) | `paper_twc/main.pdf`, `paper_twc/` |
| Supplementary docs | `docs/year3/CHANGES.md`, `DATA_AUDIT.md`, `REPRODUCE.md` |
| Year 3 forecaster | `modeling/models/seq2seq_dist.py` |
| Benchmark, analyses, table/figure generators | `modeling/experiments/` |
| Per-run results, settings, trained models (`*.pt`, Git LFS) | `modeling/results_y3/main/<link>/` |
| Sensitivity sweeps | `modeling/results_y3/sweeps/` |
| Summaries (CSV) and deployment CQR offsets | `modeling/results_y3/summary/` |
| Measurement data (Git LFS archives) | `data_release/` |
| Year 2 report sources (baseline) | `reports/year2_sources/` |

Headline (test sessions, 6 links, 3 seeds, against the Year 2 Seq2Seq retrained on the same corrected data): 95% interval score of the RTT forecasts 8-39% better, RTT MAE 2-18% lower, RTT RMSE about unchanged. See the report for the full results and the audit of the Year 2 pipeline.

Test-set prediction arrays (`results_y3/main/*/preds/`, about 3 GB) are not in the repository. `docs/year3/REPRODUCE.md` regenerates them.
