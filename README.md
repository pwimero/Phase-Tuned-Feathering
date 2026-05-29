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

The package uses SI units internally. The Fusion script converts those values
back to Fusion 360's centimeter-based API conventions.
