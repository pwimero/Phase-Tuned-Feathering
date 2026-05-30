#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/optimization}"
POPSIZE="${POPSIZE:-9}"
MAXITER="${MAXITER:-0}"
N_ETA="${N_ETA:-12}"
N_TARGETS="${N_TARGETS:-50}"
START_SEED="${START_SEED:-1}"
FREEDOM_LEVELS="${FREEDOM_LEVELS:-incidence,incidence_spacing,incidence_spacing_root_z,incidence_spacing_root_z_sweep,incidence_spacing_root_z_sweep_z_curve,full}"

SECONDS=0

END_SEED=$((START_SEED + N_TARGETS - 1))

echo ""
echo " GENERALIZED DIRECTIVITY CAMPAIGN"
echo " Targets: $N_TARGETS | Seeds: ${START_SEED}-${END_SEED}"
echo " Pop multiplier per active parameter: $POPSIZE | MaxIter: ${MAXITER:-0} | N_eta: $N_ETA"
echo " Freedom levels: $FREEDOM_LEVELS"

python3 -u -m phase_tuned_feathering.pipeline_ga \
  --campaign-dir "$OUTPUT_DIR" \
  --popsize "$POPSIZE" \
  --maxiter "$MAXITER" \
  --n-eta "$N_ETA" \
  --n-targets "$N_TARGETS" \
  --start-seed "$START_SEED" \
  --freedom-levels "$FREEDOM_LEVELS"

echo ""
echo "Campaign completed in $SECONDS seconds."
echo "Results: $OUTPUT_DIR/campaign_summary.csv"
echo "Plots:   $OUTPUT_DIR/campaign_*.svg"
