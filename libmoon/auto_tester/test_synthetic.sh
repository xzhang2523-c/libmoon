#!/usr/bin/env bash
set -euo pipefail

# Determine the directory where this script lives so relative python calls work
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

nepoch=20

# Online using libmoon/auto_tester/X
# pycharm using auto_tester/X

for agg in STche
do
  python "$SCRIPT_DIR/test_synthetic.py" --solver-name GradAgg --agg-name $agg --n-epoch $nepoch
done

for solver in UMOD
do
  python "$SCRIPT_DIR/test_synthetic.py" --solver-name $solver --agg-name STche --n-epoch $nepoch
done

for solver in PMGDA EPO MOOSVGD GradHV PMTL MGDAUB
do
  python "$SCRIPT_DIR/test_synthetic.py" --solver-name $solver --agg-name STche --n-epoch $nepoch
done

echo over