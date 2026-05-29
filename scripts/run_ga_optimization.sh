#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/optimization}"
POPSIZE="${POPSIZE:-100}"
MAXITER="${MAXITER:-100}"
N_ETA="${N_ETA:-60}"
N_TARGETS="${N_TARGETS:-10}"
START_SEED="${START_SEED:-1}"

SECONDS=0

END_SEED=$((START_SEED + N_TARGETS - 1))

echo ""
echo " GENERALIZED DIRECTIVITY CAMPAIGN"
echo " Targets: $N_TARGETS | Seeds: ${START_SEED}-${END_SEED}"
echo " Pop: $POPSIZE | MaxIter: $MAXITER | N_eta: $N_ETA"

for i in $(seq 1 "$N_TARGETS"); do
  SEED=$((START_SEED + i - 1))
  TARGET_DIR=$(printf "%s/target_%03d" "$OUTPUT_DIR" "$SEED")

  echo ""
  echo "--------------------------------------------------------"
  echo " Target $i/$N_TARGETS  (seed=$SEED)"
  echo "--------------------------------------------------------"

  python3 -u -m phase_tuned_feathering.pipeline_ga \
    --output-dir "$TARGET_DIR" \
    --popsize "$POPSIZE" \
    --maxiter "$MAXITER" \
    --n-eta "$N_ETA" \
    --seed "$SEED"
done

echo ""
echo "========================================================"
echo " All $N_TARGETS targets complete. Aggregating results..."
echo "========================================================"

python3 -u -m phase_tuned_feathering.aggregate_campaign \
  --campaign-dir "$OUTPUT_DIR"

echo ""
echo "Campaign completed in $SECONDS seconds."
echo "Results: $OUTPUT_DIR/campaign_summary.csv"
echo "Plots:   $OUTPUT_DIR/campaign_*.svg"
