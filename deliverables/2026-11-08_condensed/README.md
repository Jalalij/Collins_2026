# Deliverables 2026-11-08, condensed final report

| File | What it is |
|---|---|
| `Final_report_condensed_2026-11-08.pdf` | Condensed edition of the final report (32 pages, down from 72), for review |
| `Final_report_condensed_source_2026-11-08.zip` | LaTeX source (`main_condensed.tex`, IEEEtran references, TikZ/pgfplots figures), Overleaf-ready |

The live source is `reports/final_report/main_condensed.tex` with its sections in `reports/final_report/sections_condensed/`.
It reuses the data, figures, and generated number macros of the full report, so every number still comes from the scripts.
The full 72-page report and the paper in `deliverables/2026-11-07_final/` are unchanged.

## What the condensed edition changes
- Keeps every main result table and the key figures, and shortens the explanations.
- Puts "What Changed Since the Previous Report" right after the executive summary.
- Merges the introduction, system model, and data into one section, and the method steps, training, and metrics into one section.
- Merges the statistical validity results into the results section and the sensitivity, handover, and attention analyses into one robustness section.
- New Section 11, "Open Questions, Open Problems, and How to Close the Gaps": a table of nine gaps with evidence, fix, completion criterion, and effort, followed by a three-stage plan.
- Leaves out the per-horizon appendix (values in `modeling/results_final/summary/horizon_mean.csv`), the detailed system diagram, and secondary figures (per-session bars, time of day, scatter, example traces on every link, raw-quantile reliability, jitter and loss per horizon).

Build: `cd reports/final_report && tectonic -X compile main_condensed.tex` (or `latexmk -pdf main_condensed.tex`).
