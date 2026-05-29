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
from .metrics import band_spl, directivity, sector_spl, total_acoustic_proxy
from .observers import ObserverGrid, Sector
from .optimization import OptimizationConfig, OptimizationResult, optimize_stage
from .io import geometry_metadata, write_geometry_metadata_json, write_source_grid_csv

__all__ = [
    "AeroScreenResult",
    "ClosureParams",
    "FeatherRoot",
    "FeatherSection",
    "FlowConfig",
    "ObserverGrid",
    "OptimizationConfig",
    "OptimizationResult",
    "Sector",
    "SourceGrid",
    "SpectralResult",
    "WingGeometryParams",
    "band_spl",
    "default_geometry",
    "directivity",
    "evaluate_spp",
    "evaluate_spp_reference",
    "feather_root_properties",
    "feather_section",
    "feather_sections",
    "fusion_parameter_values",
    "geometry_metadata",
    "level1_compact_spp",
    "level2_deterministic_pressure",
    "optimize_stage",
    "screen_aero",
    "sector_spl",
    "source_grid",
    "total_acoustic_proxy",
    "write_geometry_metadata_json",
    "write_source_grid_csv",
]
