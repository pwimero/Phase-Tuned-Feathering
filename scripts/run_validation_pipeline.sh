#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/validation}"
N_ETA="${N_ETA:-24}"
SOURCE_CHORD_FRACTION="${SOURCE_CHORD_FRACTION:-1.0}"
SIM_CSV="${SIM_CSV:-}"

python3 -m unittest discover -s tests

if [[ -n "$SIM_CSV" ]]; then
  python3 -m phase_tuned_feathering.pipeline \
    --simulation-csv "$SIM_CSV" \
    --output-dir "$OUTPUT_DIR" \
    --n-eta "$N_ETA" \
    --source-chord-fraction "$SOURCE_CHORD_FRACTION"
else
  python3 -m phase_tuned_feathering.pipeline \
    --output-dir "$OUTPUT_DIR" \
    --n-eta "$N_ETA" \
    --source-chord-fraction "$SOURCE_CHORD_FRACTION"
fi
