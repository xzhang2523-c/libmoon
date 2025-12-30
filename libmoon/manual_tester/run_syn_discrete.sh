
#!/usr/bin/env bash
set -euo pipefail

# Resolve directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

problem=regression
seed_num=1
epoch=20
# mgdaub random epo pmgda
# preference-based: agg_ls agg_tche agg_pbi agg_cosmos, agg_softtche
# set-based pmtl hvgrad moosvgd
# zxy method: pmgda, uniform.
for seed in $(seq 0 $(($seed_num)))
do
  # for solver_name in mgdaub random epo pmgda agg_ls agg_tche agg_mtche agg_pbi agg_cosmos agg_softtche pmtl hvgrad moosvgd
  for solver_name in mgdaub random epo agg_ls agg_tche agg_pbi agg_cosmos pmtl gradhv moosvgd
  do
    python "$SCRIPT_DIR/run_syn_discrete.py" --solver-name $solver_name --draw-fig False --epoch $epoch --problem-name $problem --seed-idx $seed
  done
done


sleep 100