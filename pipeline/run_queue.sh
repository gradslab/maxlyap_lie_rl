#!/usr/bin/env bash
# usage: pipeline/run_queue.sh JOBS_FILE [LANES]
# Each line of JOBS_FILE is name<TAB>output<TAB>command, run from lie_rl/. A job is skipped when its output exists.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JOBS="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; LANES=${2:-4}
cd "$ROOT/lie_rl"; L=results/pipeline; mkdir -p $L/claims $L/logs
export PY=${PY:-python} PYD=${PYD:-python} OMP_NUM_THREADS=2 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2"
lane() {
  while IFS=$'\t' read -r name out cmd; do
    [[ -z "$name" || -f "$out" ]] && continue
    mkdir "$L/claims/$name" 2>/dev/null || continue
    echo "$(date '+%F %T') start $name" >> $L/progress
    mkdir -p "$(dirname "$out")"
    if bash -c "$cmd" < /dev/null > "$L/logs/$name.txt" 2>&1 && [[ -f "$out" ]]; then
      echo "$(date '+%F %T') done $name" >> $L/progress
    else
      echo "$(date '+%F %T') FAIL $name" >> $L/progress
    fi
  done < "$JOBS"
}
for k in $(seq 1 "$LANES"); do lane & sleep 1; done
wait
