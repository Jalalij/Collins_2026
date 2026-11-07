#!/bin/sh
# Rebuild every summary, analysis, table, figure and number from the saved runs, then the paper
# and the report. No training. Run from anywhere: sh modeling/results_final/refresh_all.sh
set -e
cd "$(dirname "$0")/.."
export LSN_DEVICE=cpu OMP_NUM_THREADS=4
echo "== summaries";  python3 experiments/final_aggregate.py > results_final/summary/aggregate_stdout.txt
python3 experiments/final_extras.py > results_final/summary/extras_stdout.txt
python3 experiments/final_audit_numbers.py > /dev/null
python3 experiments/final_tables.py > /dev/null
echo "== analyses"
python3 experiments/fr_condcov.py > results_final/analysis/condcov/rerun.log 2>&1
python3 experiments/fr_bootstrap.py > results_final/analysis/bootstrap/rerun.log 2>&1
python3 experiments/fr_scheduler.py > results_final/analysis/scheduler/rerun.log 2>&1
python3 experiments/fr_leak.py > /dev/null
python3 experiments/fr_trace.py > /dev/null
python3 experiments/fr_figures.py > /dev/null
python3 experiments/fr_paper_numbers.py > results_final/analysis/paper_numbers.log
echo "== paper"
cd ../paper_twc && python3 make_paper_tables.py > /dev/null && tectonic -X compile main.tex > /dev/null 2>&1 && python3 check_style.py | tail -3
echo "== report"
cd ../reports/final_report && python3 build_figs.py > /dev/null && tectonic -X compile main.tex > /dev/null 2>&1
echo "== done"
