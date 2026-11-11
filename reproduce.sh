#!/usr/bin/env bash
# usage: bash reproduce.sh [setup | tables | check | renders | all | stage NAME]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
PY="$ROOT/.venv/bin/python"; PYD="$ROOT/.venv-deploy/bin/python"
export JAX_ENABLE_X64=1

setup() {
  python3.12 -m venv .venv && "$PY" -m pip install -q -r requirements.txt
  python3.12 -m venv .venv-deploy && "$PYD" -m pip install -q -r requirements_deploy.txt
}

tables() {
  cd "$ROOT/paper"
  "$PY" final_levels.py > /dev/null
  "$PY" build2.py
  "$PY" aggregate_hb.py > out/numbers_neural_baseline.txt
  "$PY" add_hinge_columns.py
  "$PY" tab_policies.py
  "$PY" tab_fail.py
  "$PY" ablation2.py > out/numbers_ablation.txt
  "$PY" compute2.py > out/numbers_sec5.txt
  (cd "$ROOT/lie_rl" && "$PY" scripts/l4dc_curves.py ../paper/out/figs > /dev/null)
  ROWH=1.5 "$PY" slices.py out/figs/fig_main_text1.5.pdf 1.5 0.3 0.42 -0.42 1 > /dev/null
  OMP_NUM_THREADS=1 "$PY" so2_figdata.py > /dev/null
  "$PY" make_fig_so2.py out/trpo_SO2_ls-1.5_2.npz out/figs/fig_so2_proof.pdf
  "$PYD" make_fig1_v6.py > /dev/null
  (cd "$ROOT/deploy" && "$PYD" scripts/final_deploy.py ../paper/out > /dev/null)
  "$PY" text_numbers.py
  "$PY" compare.py
  "$PY" "$ROOT/web/table_data.py" out/tab_pert_all.tex
}

check() {
  cd "$ROOT/lie_rl"
  "$PY" scripts/stability_test.py
  for s in 0 1 2; do for a in ppo trpo sac td3; do
    id="SO2__${a}_SO2_ls-1.5__s${s}__base"; [[ -f results/final/cert/$id.pkl ]] || continue
    "$PY" scripts/rx_F_so2cert.py ${a}_SO2_ls-1.5 $s results/final/cert/$id.pkl results/final/eval/$id.json results/check/so2cert/$id.json | tail -1
  done; done
}

renders() {
  "$PYD" render/render_cubesat.py
  "$PYD" render/render_rov.py
}

all() {
  for f in pipeline/jobs/*.txt; do PY="$PY" PYD="$PYD" bash pipeline/run_queue.sh "$f" 4; done
  tables
}

case "${1:-tables}" in
  setup) setup ;;
  tables) tables ;;
  check) check ;;
  renders) renders ;;
  all) all ;;
  stage) PY="$PY" PYD="$PYD" bash pipeline/run_queue.sh "pipeline/jobs/$2.txt" "${3:-4}" ;;
  *) echo "usage: bash reproduce.sh [setup | tables | check | renders | all | stage NAME]"; exit 1 ;;
esac
