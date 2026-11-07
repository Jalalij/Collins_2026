# Collins project, final report: link-state forecasting for LEO and 5G multipath

| What | Where |
|---|---|
| Dated snapshot of the current deliverables (paper, report, docs, number audit) | `deliverables/2026-11-07_final/` |
| Previous version of the paper, report, and slides (kept unchanged) | `archive/2026-11-05_v1/` |
| IEEE TWC paper, live LaTeX source (IEEEtran, TikZ/pgfplots) | `paper_twc/` (`main.pdf`) |
| Final report, live LaTeX source | `reports/final_report/` (`main.pdf`) |
| Supplementary docs | `docs/final_report/CHANGES.md`, `DATA_AUDIT.md`, `REPRODUCE.md` |
| Final forecaster (DS2S) and the Gaussian comparison model | `modeling/models/seq2seq_dist.py` |
| Benchmark, analyses, table/figure/number generators | `modeling/experiments/` (`final_*.py`, `fr_*.py`) |
| Per-run results, settings, trained models (`*.pt`, Git LFS) | `modeling/results_final/main/<link>/` |
| Leak experiment (models trained on the previous interpolated inputs) | `modeling/results_final/leak/` |
| Analyses (predictability, conditional coverage, bootstrap, scheduler, attention) | `modeling/results_final/analysis/` |
| Sensitivity sweeps | `modeling/results_final/sweeps/` |
| Summaries (CSV) and deployment CQR offsets | `modeling/results_final/summary/` |
| Measurement data (Git LFS archives) | `data_release/` |

Headline (test sessions, 6 links, three-seed ensembles, against the previous Seq2Seq retrained on the same corrected data): the 95% interval score of the RTT forecasts improves by 9.4% to 41.1%, significantly on every link with a session-level bootstrap. The RTT MAE improves on five links, and the RMSE changes by at most 5%. Past RTT changes hardly predict the next change, but the spread of the RTT depends on a link state that the recent input reveals. A calibrated distributional forecaster exploits this, and in an emulated deadline-aware duplication scheduler it needs 55% less duplication on Starlink DL with no more late packets.

Test-set prediction arrays (`results_final/**/preds/`, several GB) are not in the repository. `docs/final_report/REPRODUCE.md` regenerates them.
