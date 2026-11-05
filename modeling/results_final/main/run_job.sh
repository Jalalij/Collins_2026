#!/bin/sh
# usage: run_job.sh <provider_direction> <method[,method]> <seed>
ds=$1; methods=$(echo $2 | tr ',' ' '); seed=$3
prov=${ds%_*}; dir=${ds#*_}
stride=1; [ "$prov" = "quectel" ] && stride=4
cd "$(dirname "$0")/../.."
python3 experiments/final_benchmark.py --provider $prov --direction $dir --methods $methods --seeds $seed \
  --epochs 60 --patience 15 --train_stride $stride --save_models --out_dir results_final/main \
  > results_final/main/logs_${ds}_$(echo $2 | tr ',' '-')_s${seed}.out 2>&1
