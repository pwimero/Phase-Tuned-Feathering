#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/validation}"
N_ETA="${N_ETA:-48}"
SOURCE_CHORD_FRACTION="${SOURCE_CHORD_FRACTION:-1.0}"
SIM_CSV="${SIM_CSV:-}"
SIM_MODE="${SIM_MODE:-surrogate}"
NO_CALIBRATE_LEVEL="${NO_CALIBRATE_LEVEL:-0}"

CALIBRATION_ARGS=()
if [[ "$NO_CALIBRATE_LEVEL" == "1" ]]; then
  CALIBRATION_ARGS+=(--no-calibrate-level)
fi

SECONDS=0

if [[ -n "$SIM_CSV" ]]; then
  python3 -m phase_tuned_feathering.pipeline \
    --simulation-csv "$SIM_CSV" \
    --output-dir "$OUTPUT_DIR" \
    --n-eta "$N_ETA" \
    --source-chord-fraction "$SOURCE_CHORD_FRACTION" \
    "${CALIBRATION_ARGS[@]}"
else
  python3 -m phase_tuned_feathering.pipeline \
    --simulation-mode "$SIM_MODE" \
    --output-dir "$OUTPUT_DIR" \
    --n-eta "$N_ETA" \
    --source-chord-fraction "$SOURCE_CHORD_FRACTION" \
    "${CALIBRATION_ARGS[@]}"
fi

echo ""
echo "Pipeline completed in $SECONDS seconds."
