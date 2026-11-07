#!/bin/sh
# usage: run_leak.sh <method> <seed>  (Starlink DL, trained on the parser-interpolated inputs,
# scored on both the interpolated and the causal test inputs)
m=$1; seed=$2
cd "$(dirname "$0")/../.."
python3 experiments/final_benchmark.py --provider starlink --direction dl --methods $m --seeds $seed \
  --epochs 60 --patience 15 --legacy_data --eval_causal --save_models --out_dir results_final/leak \
  > results_final/leak_logs_${m}_s${seed}.out 2>&1
