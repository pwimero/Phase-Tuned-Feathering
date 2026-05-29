#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/optimization}"
POPSIZE="${POPSIZE:-10}"
MAXITER="${MAXITER:-30}"
N_ETA="${N_ETA:-48}"
SCENARIOS=("quiet_ground" "forward_focus")

SECONDS=0

for SCENARIO in "${SCENARIOS[@]}"; do
  echo ""
  echo "Running Genetic Algorithm for Scenario: $SCENARIO"
  python3 -u -m phase_tuned_feathering.pipeline_ga \
    --scenario "$SCENARIO" \
    --output-dir "$OUTPUT_DIR" \
    --popsize "$POPSIZE" \
    --maxiter "$MAXITER" \
    --n-eta "$N_ETA"
done

echo ""
echo "GA Optimization Pipeline completed in $SECONDS seconds."
