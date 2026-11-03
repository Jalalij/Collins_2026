#!/bin/sh
# usage: run_sweep.sh <config> <link> <method> <seed>
# configs: L15 L60 H30 hid64 hid256 qw025 qw4
cfg=$1; ds=$2; m=$3; seed=$4
prov=${ds%_*}; dir=${ds#*_}
extra=""; case $cfg in
  L15) extra="--input_len 15";; L60) extra="--input_len 60";; H30) extra="--output_len 30";;
  hid64) extra="--hidden_size 64";; hid256) extra="--hidden_size 256";;
  qw025) extra="--qweight 0.25";; qw4) extra="--qweight 4";;
esac
cd "$(dirname "$0")/../.."
python3 experiments/y3_benchmark.py --provider $prov --direction $dir --methods $m --seeds $seed \
  --epochs 60 --patience 15 $extra --out_dir results_y3/sweeps/$cfg \
  > results_y3/sweeps/log_${cfg}_${ds}_${m}_s${seed}.out 2>&1
