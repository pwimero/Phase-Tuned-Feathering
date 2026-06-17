# Phase Tuned Feathering


Core entry points:

- `phase_tuned_feathering.default_geometry()`
- `phase_tuned_feathering.source_grid(...)`
- `phase_tuned_feathering.evaluate_spp(...)`
- `phase_tuned_feathering.screen_aero(...)`
- `phase_tuned_feathering.simulate_surrogate_dataset(...)`
- `phase_tuned_feathering.compare_theory_to_simulation(...)`
- `phase_tuned_feathering.source_csd_matrix(...)`
- `phase_tuned_feathering.radiation_operator(...)`
- `phase_tuned_feathering.modal_radiation_decomposition(...)`
- `phase_tuned_feathering.mechanism_metrics(...)`
- `phase_tuned_feathering.radiating_covariance_gain(...)`
- `phase_tuned_feathering.controllability_jacobian(...)`

## Radiating-covariance validation

Run the Version 1 theory and benchmark validation suite:

```bash
scripts/run_radiating_covariance_validation.py
```

This writes exact algebraic checks, canonical array-factor validation,
diagonal no-phase validation, controlled mechanism examples, a partial-coherence
mechanism phase diagram, 50-target campaign radiating-covariance metrics, and
the `FWA-Bench-0` reproducibility package. Outputs are written to
`outputs/radiating_covariance`.

The package uses SI units internally. The Fusion script converts those values
back to Fusion 360's centimeter-based API conventions.

## Validation pipeline

Run the complete smoke-tested comparison pipeline:

```bash
scripts/run_validation_pipeline.sh
```

By default this creates low-cost surrogate aeroacoustic simulation data from
the shared feather geometry, then compares the theoretical model against that
surrogate. The surrogate uses local low-order aero states plus calibrated,
BPM/TNO-inspired self-noise mechanisms; it is not full CFD.

The comparison applies one scalar source-level calibration using rows marked
`split=calibration`, then reports the held-out `split=validation` error with
that fixed scale. To inspect raw uncalibrated error:

```bash
NO_CALIBRATE_LEVEL=1 scripts/run_validation_pipeline.sh
```

For a smoke test that uses theory plus deterministic perturbation instead of
the surrogate, run:

```bash
SIM_MODE=synthetic scripts/run_validation_pipeline.sh
```

When external simulated performance data are available, provide them as:

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
- `simulator_geometry_plan.svg`
- `simulator_geometry_side.svg`
- `simulator_geometry_front.svg`
- `simulator_geometry_isometric.svg`
- `surrogate_simulation.csv` by default
- `synthetic_simulation.csv` when `SIM_MODE=synthetic`
- `theory_vs_simulation.csv`
- `validation_summary.json`
- `validation_scatter.svg`
- `validation_spectrum_overlay.svg`
- `validation_error_by_frequency.svg`
- `validation_error_histogram.svg`
- `validation_directivity.svg`
- `validation_error_heatmap.svg`

The SVG files show the exact source-line geometry seen by the simulator and
the acoustic model. They are not Fusion solid-body renders; they show source
points, feather IDs, and loading-vector directions in paper coordinates.
The validation SVGs show theory-vs-simulation agreement, spectral overlays,
error distributions, error by frequency, and directivity comparison.
