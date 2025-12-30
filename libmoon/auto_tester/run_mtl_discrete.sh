#!/usr/bin/env bash
set -euo pipefail

# Resolve directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

seed_num=1
epoch=2

for seed in $(seq 0 $(($seed_num)))
do
  for solver_name in mgdaub random epo pmgda agg_ls agg_tche agg_pbi agg_cosmos hvgrad moosvgd
  #for solver_name in agg_mtche agg_softmtche agg_softtche
  do
    python "$SCRIPT_DIR/run_mtl_discrete.py" --problem-name adult --solver-name $solver_name --use-plt False
     --epoch $epoch --seed-idx $seed
  done
done

sleep 100