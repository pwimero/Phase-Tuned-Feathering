"""Phase-tuned feathering research package.

The package is intentionally dependency-light so the mathematical model,
geometry parity tests, and low-order feasibility checks can run before any
heavy CFD or plotting stack is introduced.
"""

from .acoustics import (
    SpectralResult,
    evaluate_spp,
    evaluate_spp_reference,
    level1_compact_spp,
    level2_deterministic_pressure,
)
from .aero import AeroScreenResult, screen_aero
from .closures import ClosureParams, FlowConfig
from .geometry import (
    FeatherRoot,
    FeatherSection,
    SourceGrid,
    WingGeometryParams,
    default_geometry,
    feather_root_properties,
    feather_section,
    feather_sections,
    fusion_parameter_values,
    source_grid,
)
from .observers import ObserverGrid
from .io import geometry_metadata, write_geometry_metadata_json, write_source_grid_csv
from .simulation import (
    SurrogateAeroState,
    SurrogateNoiseConfig,
    local_aero_state,
    local_aero_states,
    simulate_surrogate_dataset,
    write_surrogate_simulation_csv,
)
from .validation import (
    ComparisonResult,
    ComparisonRow,
    SimulationDataset,
    SimulationRecord,
    compare_spectral_results,
    compare_theory_to_simulation,
    comparison_summary_text,
    generate_synthetic_simulation_dataset,
    load_simulation_csv,
    write_comparison_csv,
    write_simulation_csv,
    write_summary_json,
    write_synthetic_simulation_csv,
)
from .visualization import (
    directivity_comparison_svg,
    error_by_frequency_svg,
    error_histogram_svg,
    source_grid_svg,
    spectrum_overlay_svg,
    theory_vs_simulation_scatter_svg,
    write_validation_figures,
    write_simulator_geometry_renders,
    write_source_grid_svg,
)

__all__ = [
    "AeroScreenResult",
    "ClosureParams",
    "ComparisonResult",
    "ComparisonRow",
    "FeatherRoot",
    "FeatherSection",
    "FlowConfig",
    "ObserverGrid",
    "SimulationDataset",
    "SimulationRecord",
    "SourceGrid",
    "SpectralResult",
    "SurrogateAeroState",
    "SurrogateNoiseConfig",
    "WingGeometryParams",
    "compare_spectral_results",
    "compare_theory_to_simulation",
    "comparison_summary_text",
    "default_geometry",
    "directivity_comparison_svg",
    "error_by_frequency_svg",
    "error_histogram_svg",
    "evaluate_spp",
    "evaluate_spp_reference",
    "feather_root_properties",
    "feather_section",
    "feather_sections",
    "fusion_parameter_values",
    "generate_synthetic_simulation_dataset",
    "geometry_metadata",
    "level1_compact_spp",
    "level2_deterministic_pressure",
    "local_aero_state",
    "local_aero_states",
    "load_simulation_csv",
    "screen_aero",
    "simulate_surrogate_dataset",
    "source_grid",
    "source_grid_svg",
    "spectrum_overlay_svg",
    "theory_vs_simulation_scatter_svg",
    "write_comparison_csv",
    "write_geometry_metadata_json",
    "write_simulation_csv",
    "write_source_grid_csv",
    "write_source_grid_svg",
    "write_summary_json",
    "write_simulator_geometry_renders",
    "write_surrogate_simulation_csv",
    "write_synthetic_simulation_csv",
    "write_validation_figures",
]
