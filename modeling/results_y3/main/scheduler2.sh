#!/bin/sh
# Runs jobs.txt with at most 6 jobs in total and at most 2 5G jobs at a time
# (5G jobs need ~6-8 GB RAM each). Skips jobs whose result CSV exists or that are running.
cd "$(dirname "$0")"
while read ds methods seed; do
  m1=$(echo $methods | cut -d, -f1)
  [ -f "$ds/${m1}_s${seed}.csv" ] && continue
  prov=${ds%_*}; dir=${ds#*_}
  pgrep -f "provider $prov --direction $dir --methods $(echo $methods | tr ',' ' ') --seeds $seed " >/dev/null && continue
  while :; do
    tot=$(pgrep -f y3_benchmark.py | wc -l); q=$(pgrep -f "y3_benchmark.py --provider quectel" | wc -l)
    if [ $tot -lt 6 ] && { [ "$prov" != "quectel" ] || [ $q -lt 2 ]; }; then break; fi
    sleep 20
  done
  echo "$(date +%H:%M) start $ds $methods $seed"
  ./run_job.sh $ds $methods $seed &
  sleep 15
done < ${1:-jobs.txt}
wait
echo "$(date +%H:%M) all done"
