#!/bin/sh
# Like scheduler_v2.sh, but without head-of-line blocking: each pass over jobs_v2.txt starts every
# job that fits (at most 6 jobs in total and 2 on 5G). A job is skipped if its result CSV exists or
# it is already running, and it is started at most twice.
cd "$(dirname "$0")"
: > started_v3.txt
while :; do
  pending=0
  while read ds m s; do
    [ -f "$ds/${m}_s${s}.csv" ] && continue
    prov=${ds%_*}; dir=${ds#*_}
    if pgrep -f "provider $prov --direction $dir --methods $m --seeds $s " >/dev/null; then pending=1; continue; fi
    n=$(grep -c "^$ds $m $s\$" started_v3.txt)
    [ "$n" -ge 2 ] && continue
    pending=1
    tot=$(pgrep -f final_benchmark.py | wc -l); q=$(pgrep -f "final_benchmark.py --provider quectel" | wc -l)
    if [ "$tot" -lt 6 ] && { [ "$prov" != "quectel" ] || [ "$q" -lt 2 ]; }; then
      echo "$(date +%H:%M) start $ds $m $s"
      echo "$ds $m $s" >> started_v3.txt
      ./run_job.sh $ds $m $s &
      sleep 15
    fi
  done < jobs_v2.txt
  [ $pending -eq 0 ] && break
  sleep 30
done
wait
echo "$(date +%H:%M) all done"
