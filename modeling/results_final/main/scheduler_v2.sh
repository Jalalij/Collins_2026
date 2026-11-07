#!/bin/sh
# Runs jobs_v2.txt (main study additions) and the leak jobs with at most 6 jobs in total
# and at most 2 5G jobs at a time. Skips jobs whose result CSV exists or that are running.
cd "$(dirname "$0")"
wait_slot() {
  while :; do
    tot=$(pgrep -f final_benchmark.py | wc -l); q=$(pgrep -f "final_benchmark.py --provider quectel" | wc -l)
    if [ $tot -lt 6 ] && { [ "$1" != "quectel" ] || [ $q -lt 2 ]; }; then break; fi
    sleep 20
  done
}
for m in prev_seq2seq fr_dist; do for s in 0 1 2; do
  [ -f ../leak/starlink_dl/${m}_s${s}.csv ] && continue
  wait_slot starlink; echo "$(date +%H:%M) start leak $m $s"; ./run_leak.sh $m $s & sleep 15
done; done
while read ds methods seed; do
  [ -f "$ds/${methods}_s${seed}.csv" ] && continue
  prov=${ds%_*}
  wait_slot $prov
  echo "$(date +%H:%M) start $ds $methods $seed"
  ./run_job.sh $ds $methods $seed &
  sleep 15
done < jobs_v2.txt
wait
echo "$(date +%H:%M) all done"
