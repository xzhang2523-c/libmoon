#!/usr/bin/env bash
set -euo pipefail

# Resolve directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

seed_num=1

#for seed in $(seq 1 $(($seed_num)))
#do
#  for dataset in mnist fashion fmnist adult
#  do
#    for solver in agg_ls epo pmgda
#    do
#      python "$SCRIPT_DIR/run_mtl_psl.py" --solver-name $solver --problem-name $dataset --seed-idx $seed --epoch 40
#    done
#  done
#done

sleep 10