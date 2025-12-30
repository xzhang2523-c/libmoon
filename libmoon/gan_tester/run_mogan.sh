#!/usr/bin/env bash
set -euo pipefail

# Resolve directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for pref0 in 0.0 0.25 0.5 0.75 1.0
do
   python "$SCRIPT_DIR/test.py" --pref0 $pref0
done
sleep 100