"""Low-cost aeroacoustic surrogate simulation.

This module is intentionally cheaper than CFD. It builds the simulated
performance data from the shared feather geometry, a low-order local aero
state, and BPM/TNO-inspired airfoil self-noise mechanisms. The output is the
same CSV schema consumed by :mod:`phase_tuned_feathering.validation`.

The model is a surrogate, not high-fidelity aeroacoustic validation. Its role
is to provide a defensible laptop-scale comparison layer before experiments or
unsteady CFD are available.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from .closures import ClosureParams, FlowConfig, coherence_value
from .geometry import SourceGrid, WingGeometryParams, default_geometry, source_grid
from .observers import ObserverGrid
from .validation import SimulationDataset, SimulationRecord, write_simulation_csv


@dataclass(frozen=True)
class SurrogateNoiseConfig:
    """Calibratable constants for the low-cost aeroacoustic surrogate."""

    source_level_scale: float = 2.5e-10
    convection_ratio: float = 0.70
    turbulent_intensity: float = 0.03
    stall_angle_deg: float = 14.0
    separation_onset_deg: float = 8.0
    zero_lift_cd: float = 0.010
    lift_drag_factor: float = 0.020
    tbl_te_weight: float = 1.0
    separation_weight: float = 0.35
    bluntness_weight: float = 0.08
    tip_vortex_weight: float = 0.12
    tbl_peak_strouhal: float = 0.08
    separation_peak_strouhal: float = 0.025
    bluntness_peak_strouhal: float = 0.20
    tip_peak_strouhal: float = 0.12
    spectral_width: float = 1.15
    calibration_frequency_count: int = 1


@dataclass(frozen=True)
class SurrogateAeroState:
    source_index: int
    feather_id: int
    eta: float
    chord_m: float
    segment_length_m: float
    incidence_deg: float
    reynolds: float
    cl: float
    cd: float
    boundary_layer_thickness_m: float
    displacement_thickness_m: float
    trailing_edge_thickness_m: float
    separation_factor: float


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _safe_reynolds(chord_m: float, flow: FlowConfig) -> float:
    return max(
        flow.rho0 * flow.u_inf * chord_m / max(flow.dynamic_viscosity, 1.0e-12),
        1.0,
    )


def _thin_airfoil_state(
    incidence_deg: float,
    config: SurrogateNoiseConfig,
) -> tuple[float, float, float]:
    alpha = math.radians(incidence_deg)
    stall_alpha = math.radians(config.stall_angle_deg)
    stall_softener = 1.0 / (1.0 + (abs(alpha) / max(stall_alpha, 1.0e-12)) ** 8)
    cl = 2.0 * math.pi * alpha * stall_softener
    cd = config.zero_lift_cd + config.lift_drag_factor * cl * cl
    separation_factor = _clamp(
        (abs(incidence_deg) - config.separation_onset_deg)
        / max(config.stall_angle_deg - config.separation_onset_deg, 1.0e-12),
        0.0,
        1.5,
    )
    return cl, cd, separation_factor * separation_factor


def local_aero_state(
    grid: SourceGrid,
    source_index: int,
    flow: FlowConfig,
    config: SurrogateNoiseConfig | None = None,
) -> SurrogateAeroState:
    config = SurrogateNoiseConfig() if config is None else config
    chord = grid.chords[source_index]
    incidence_deg = grid.incidence_deg[source_index]
    reynolds = _safe_reynolds(chord, flow)
    cl, cd, separation_factor = _thin_airfoil_state(incidence_deg, config)
    re_term = reynolds ** 0.2
    boundary_layer_thickness = 0.37 * chord / re_term
    displacement_thickness = 0.046 * chord / re_term
    trailing_edge_thickness = min(0.005, 0.08 * chord)

    return SurrogateAeroState(
        source_index=source_index,
        feather_id=grid.feather_ids[source_index],
        eta=grid.etas[source_index],
        chord_m=chord,
        segment_length_m=grid.weights[source_index],
        incidence_deg=incidence_deg,
        reynolds=reynolds,
        cl=cl,
        cd=cd,
        boundary_layer_thickness_m=boundary_layer_thickness,
        displacement_thickness_m=displacement_thickness,
        trailing_edge_thickness_m=trailing_edge_thickness,
        separation_factor=separation_factor,
    )


def local_aero_states(
    grid: SourceGrid,
    flow: FlowConfig,
    config: SurrogateNoiseConfig | None = None,
) -> tuple[SurrogateAeroState, ...]:
    return tuple(
        local_aero_state(grid, index, flow, config)
        for index in range(grid.n)
    )


def _log_gaussian_shape(
    strouhal: float,
    peak: float,
    width: float,
) -> float:
    strouhal = max(strouhal, 1.0e-12)
    peak = max(peak, 1.0e-12)
    width = max(width, 1.0e-12)
    value = math.exp(-((math.log(strouhal / peak) / width) ** 2))
    high_frequency_rolloff = 1.0 / (1.0 + (strouhal / (8.0 * peak)) ** 2)
    return value * high_frequency_rolloff


def _directivity_factor(
    direction: tuple[float, float, float],
    loading_direction: tuple[float, float, float],
    mach: float,
) -> float:
    loading_projection = sum(direction[i] * loading_direction[i] for i in range(3))
    dipole = loading_projection * loading_projection
    convective = 1.0 / max((1.0 - mach * direction[0]) ** 4, 1.0e-12)
    lateral_edge_factor = 0.65 + 0.35 * (1.0 - abs(direction[1]))
    return max(dipole * convective * lateral_edge_factor, 0.0)


def _mechanism_psd(
    frequency_hz: float,
    state: SurrogateAeroState,
    flow: FlowConfig,
    config: SurrogateNoiseConfig,
) -> float:
    u_c = max(config.convection_ratio * flow.u_inf, 1.0e-12)
    delta_star = max(state.displacement_thickness_m, 1.0e-12)
    h_te = max(state.trailing_edge_thickness_m, 1.0e-12)
    st_delta = frequency_hz * delta_star / u_c
    st_chord = frequency_hz * state.chord_m / u_c
    st_h = frequency_hz * h_te / u_c

    tbl_shape = _log_gaussian_shape(
        st_delta,
        config.tbl_peak_strouhal,
        config.spectral_width,
    )
    sep_shape = _log_gaussian_shape(
        st_chord,
        config.separation_peak_strouhal,
        config.spectral_width * 1.25,
    )
    blunt_shape = _log_gaussian_shape(
        st_h,
        config.bluntness_peak_strouhal,
        config.spectral_width * 0.75,
    )
    tip_shape = _log_gaussian_shape(
        st_chord,
        config.tip_peak_strouhal,
        config.spectral_width,
    )

    lift_factor = 1.0 + 0.35 * abs(state.cl) + 0.20 * state.cl * state.cl
    re_factor = _clamp((state.reynolds / 1.0e6) ** 0.15, 0.45, 1.75)
    bl_factor = delta_star * state.segment_length_m
    base = (
        config.source_level_scale
        * flow.rho0**2
        * flow.u_inf**5
        * bl_factor
        * re_factor
    )

    tbl = config.tbl_te_weight * lift_factor * tbl_shape
    separated = (
        config.separation_weight
        * state.separation_factor
        * (1.0 + 0.5 * abs(state.cl))
        * sep_shape
    )
    bluntness_ratio = _clamp(h_te / delta_star, 0.0, 12.0)
    blunt = config.bluntness_weight * bluntness_ratio * bluntness_ratio * blunt_shape
    tip = config.tip_vortex_weight * (state.eta**8) * (1.0 + abs(state.cl)) * tip_shape

    turbulence_boost = 1.0 + 8.0 * config.turbulent_intensity
    return max(base * turbulence_boost * (tbl + separated + blunt + tip), 0.0)


def simulate_surrogate_dataset(
    params: WingGeometryParams | None = None,
    flow: FlowConfig | None = None,
    config: SurrogateNoiseConfig | None = None,
    observers: ObserverGrid | None = None,
    frequencies_hz: tuple[float, ...] = (250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0),
    n_eta: int = 48,
    source_chord_fraction: float = 1.0,
    case_id: str = "surrogate_sim",
) -> SimulationDataset:
    params = default_geometry() if params is None else params
    flow = FlowConfig() if flow is None else flow
    config = SurrogateNoiseConfig() if config is None else config
    observers = ObserverGrid.spherical(12, 24) if observers is None else observers

    grid = source_grid(
        params,
        n_eta=n_eta,
        source_chord_fraction=source_chord_fraction,
    )
    states = local_aero_states(grid, flow, config)
    mach = flow.mach
    radius_factor = 1.0 / ((4.0 * math.pi * max(flow.observer_radius, 1.0e-12)) ** 2)

    records: list[SimulationRecord] = []
    closures = ClosureParams()
    for frequency_index, frequency_hz in enumerate(frequencies_hz):
        split = (
            "calibration"
            if frequency_index < config.calibration_frequency_count
            else "validation"
        )
        wave_number = 2.0 * math.pi * frequency_hz / flow.c0
        
        psd_array = [_mechanism_psd(frequency_hz, state, flow, config) for state in states]
        
        gamma_matrix = [[0.0j] * len(states) for _ in range(len(states))]
        for m in range(len(states)):
            for n in range(len(states)):
                if m == n:
                    gamma_matrix[m][n] = 1.0 + 0.0j
                else:
                    gamma_matrix[m][n] = coherence_value(
                        grid.points[m],
                        grid.points[n],
                        grid.incidence_deg[m],
                        grid.incidence_deg[n],
                        frequency_hz,
                        flow,
                        closures,
                    )
        
        for direction, observer_weight in zip(observers.directions, observers.weights):
            complex_weights = []
            for source_index, state in enumerate(states):
                directivity = _directivity_factor(direction, grid.loading_directions[source_index], mach)
                amplitude = math.sqrt(psd_array[source_index] * directivity)
                phase = wave_number * sum(grid.points[source_index][i] * direction[i] for i in range(3))
                complex_weights.append(amplitude * complex(math.cos(phase), math.sin(phase)))

            spp = 0.0 + 0.0j
            for m in range(len(states)):
                w_m_conj = complex_weights[m].conjugate()
                row_gamma = gamma_matrix[m]
                for n in range(len(states)):
                    spp += w_m_conj * row_gamma[n] * complex_weights[n]

            records.append(
                SimulationRecord(
                    frequency_hz=frequency_hz,
                    direction=direction,
                    spp=max(spp.real * radius_factor, 0.0),
                    case_id=case_id,
                    split=split,
                    weight=observer_weight,
                )
            )

    return SimulationDataset(tuple(records))


def write_surrogate_simulation_csv(
    path: str | Path,
    params: WingGeometryParams | None = None,
    flow: FlowConfig | None = None,
    config: SurrogateNoiseConfig | None = None,
    observers: ObserverGrid | None = None,
    frequencies_hz: tuple[float, ...] = (250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0),
    n_eta: int = 48,
    source_chord_fraction: float = 1.0,
) -> Path:
    dataset = simulate_surrogate_dataset(
        params=params,
        flow=flow,
        config=config,
        observers=observers,
        frequencies_hz=frequencies_hz,
        n_eta=n_eta,
        source_chord_fraction=source_chord_fraction,
    )
    return write_simulation_csv(path, dataset)
