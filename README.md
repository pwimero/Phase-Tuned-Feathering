# Phase Tuned Feathering

This repository now contains a dependency-light Python research package that
shares the feathered-wingtip geometry rules with the Fusion 360 CAD script and
implements the model-first acoustic workflow described in the manuscript plan.

Core entry points:

- `phase_tuned_feathering.default_geometry()`
- `phase_tuned_feathering.source_grid(...)`
- `phase_tuned_feathering.evaluate_spp(...)`
- `phase_tuned_feathering.band_spl(...)`
- `phase_tuned_feathering.directivity(...)`
- `phase_tuned_feathering.screen_aero(...)`
- `phase_tuned_feathering.optimize_stage(...)`
- `phase_tuned_feathering.compare_theory_to_simulation(...)`

The package uses SI units internally. The Fusion script converts those values
back to Fusion 360's centimeter-based API conventions.

## Validation pipeline

Run the complete smoke-tested comparison pipeline:

```bash
scripts/run_validation_pipeline.sh
```

By default this creates deterministic synthetic simulation data so the
comparison machinery can be exercised before external simulation results exist.
When real simulated performance data are available, provide them as:

```bash
SIM_CSV=/path/to/simulation.csv scripts/run_validation_pipeline.sh
```

Required simulation CSV columns:

- `frequency_hz`
- `observer_x`
- `observer_y`
- `observer_z`
- `spp`

Optional columns:

- `case_id`
- `split` (`calibration` or `validation`)
- `weight`

Pipeline outputs are written to `outputs/validation` by default:

- `geometry_metadata.json`
- `source_grid.csv`
- `synthetic_simulation.csv` when no `SIM_CSV` is provided
- `theory_vs_simulation.csv`
- `validation_summary.json`
