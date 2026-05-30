from dataclasses import dataclass, field, replace
from typing import Sequence
import math
import numpy as np
import time
import torch
from typing import Tuple, List

from .acoustics_torch import evaluate_spp_torch
from .closures import FlowConfig, ClosureParams
from .geometry import WingGeometryParams, default_geometry
from .observers import ObserverGrid


@dataclass(frozen=True)
class GAConfig:
    base_params: WingGeometryParams = field(default_factory=default_geometry)
    flow: FlowConfig = field(default_factory=FlowConfig)
    closures: ClosureParams = field(default_factory=ClosureParams)
    observers: ObserverGrid = field(default_factory=lambda: ObserverGrid.spherical(8, 10))
    frequencies_hz: tuple[float, ...] = (500.0, 1000.0, 2000.0)
    target_pattern_db: tuple[float, ...] = field(default_factory=tuple)
    n_eta: int = 16
    popsize: int = 9
    maxiter: int = 0
    mutation: tuple[float, float] = (0.5, 1.0)
    recombination: float = 0.7
    seed: int = 42
    patience: int = 20
    min_fitness_improvement_db2: float = 0.015
    freedom_level: str = "full"

    # Expanded where the recent campaign repeatedly saturated the bounds.
    incidence_bounds_deg: tuple[float, float] = (-20.0, 20.0)
    spacing_scale_bounds: tuple[float, float] = (0.65, 2.10)
    root_z_bounds: tuple[float, float] = (-0.45, 0.45)
    tip_sweep_bounds: tuple[float, float] = (-1.10, 0.80)
    tip_z_curve_bounds: tuple[float, float] = (-0.75, 1.00)
    min_tip_chord_scale_bounds: tuple[float, float] = (0.06, 0.55)
    intersection_penalty: float = 1.0e6
    intersection_n_eta: int = 8
    coarse_n_eta: int = 5
    coarse_observer_stride: int = 2
    coarse_frequency_stride: int = 2
    refinement_top_fraction: float = 0.18
    refinement_min_candidates: int = 6


FREEDOM_LEVEL_ORDER: tuple[str, ...] = (
    "incidence",
    "zero_coherence",
    "no_delay",
    "full",
)

FREEDOM_LEVEL_LABELS: dict[str, str] = {
    "incidence": "Incidence only",
    "zero_coherence": "Full geometry, zero coherence",
    "no_delay": "Full geometry, no incidence delay",
    "full": "Full model",
}


def effective_closures(
    freedom_level: str,
    closures: ClosureParams,
) -> ClosureParams:
    """Return closure parameters with ablation overrides applied.

    * ``zero_coherence`` forces the coherence model to ``"zero"`` so that
      every source pair is treated as uncorrelated.  This tests whether
      the array phase-tuning mechanism contributes at all.
    * ``no_delay`` sets ``incidence_delay_per_rad`` to zero, removing the
      incidence-dependent phase delay closure while keeping everything
      else (including coherence) intact.
    * ``incidence`` and ``full`` leave closures unchanged.
    """
    level = freedom_level.lower()
    if level == "zero_coherence":
        return replace(closures, coherence_model="zero")
    if level == "no_delay":
        return replace(closures, incidence_delay_per_rad=0.0)
    return closures


@dataclass(frozen=True)
class GAResult:
    best_params: WingGeometryParams
    best_score: float
    elapsed_seconds: float
    success: bool
    message: str


@dataclass(frozen=True)
class _FitnessContext:
    observers: ObserverGrid
    observer_dirs: torch.Tensor
    frequencies_hz: tuple[float, ...]
    frequency_tensor: torch.Tensor
    target_normalized: torch.Tensor
    flow: FlowConfig
    closures: ClosureParams
    scales: torch.Tensor


@dataclass(frozen=True)
class _GeometryTensorContext:
    total_wings: int
    basis: torch.Tensor
    root_chords: torch.Tensor
    tip_span: torch.Tensor
    feather_progress: torch.Tensor
    trap_weights: dict[int, torch.Tensor]
    source_chord_fraction: float
    min_feather_root_gap: float
    sweep_curve_exponent: float
    z_curve_exponent: float


@dataclass
class OptimizationRuntime:
    device: torch.device | None = None
    geometry_context: _GeometryTensorContext | None = None


def _incidence_basis(total_wings: int) -> tuple[tuple[float, ...], ...]:
    if total_wings <= 0:
        raise ValueError("total_wings must be positive.")
    rows: list[tuple[float, ...]] = []
    scale0 = math.sqrt(1.0 / total_wings)
    for wing_index in range(total_wings):
        row: list[float] = []
        for mode in range(total_wings):
            if mode == 0:
                row.append(scale0)
            else:
                row.append(
                    math.sqrt(2.0 / total_wings)
                    * math.cos(
                        math.pi
                        * (wing_index + 0.5)
                        * mode
                        / total_wings
                    )
                )
        rows.append(tuple(row))
    return tuple(rows)


def _decode_incidence_shape_coeffs(
    coeffs: Sequence[float],
    total_wings: int,
) -> tuple[float, ...]:
    if len(coeffs) != total_wings:
        raise ValueError(
            f"Expected {total_wings} incidence coefficients, got {len(coeffs)}."
        )
    basis = _incidence_basis(total_wings)
    values: list[float] = []
    for wing_index in range(total_wings):
        values.append(
            sum(basis[wing_index][mode] * float(coeffs[mode]) for mode in range(total_wings))
        )
    return tuple(values)


def _encode_incidence_shape_coeffs(
    incidence_deg: Sequence[float],
    total_wings: int,
) -> tuple[float, ...]:
    if len(incidence_deg) != total_wings:
        raise ValueError(
            f"Expected {total_wings} incidence values, got {len(incidence_deg)}."
        )
    basis = _incidence_basis(total_wings)
    coeffs: list[float] = []
    for mode in range(total_wings):
        coeffs.append(
            sum(float(incidence_deg[wing_index]) * basis[wing_index][mode] for wing_index in range(total_wings))
        )
    return tuple(coeffs)


def _incidence_coeff_bounds(config: GAConfig) -> tuple[tuple[float, float], ...]:
    total_wings = config.base_params.total_wings
    max_abs_incidence = max(
        abs(config.incidence_bounds_deg[0]),
        abs(config.incidence_bounds_deg[1]),
    )
    basis = _incidence_basis(total_wings)
    bounds: list[tuple[float, float]] = []
    for mode in range(total_wings):
        l1_norm = sum(abs(basis[wing_index][mode]) for wing_index in range(total_wings))
        coeff_limit = max_abs_incidence * l1_norm
        bounds.append((-coeff_limit, coeff_limit))
    return tuple(bounds)


def _unpack_genes(x: Sequence[float], base_params: WingGeometryParams) -> WingGeometryParams:
    """Unpack a 1D array of genes into a WingGeometryParams object."""
    total_wings = base_params.total_wings
    extra_gene_count = 11
    if len(x) != total_wings + extra_gene_count:
        raise ValueError(f"Expected {total_wings + extra_gene_count} genes, got {len(x)}")

    incidences = _decode_incidence_shape_coeffs(x[0:total_wings], total_wings)
    y_spacing = x[total_wings]
    wing_1_root_z = x[total_wings + 1]
    mid_root_z = x[total_wings + 2]
    wing_7_root_z = x[total_wings + 3]
    wing_1_tip_sweep = x[total_wings + 4]
    mid_tip_sweep = x[total_wings + 5]
    wing_7_tip_sweep = x[total_wings + 6]
    wing_1_tip_z_curve = x[total_wings + 7]
    mid_tip_z_curve = x[total_wings + 8]
    wing_7_tip_z_curve = x[total_wings + 9]
    min_tip_chord_scale = x[total_wings + 10]

    params = base_params.with_incidence_angles(incidences)
    params = replace(
        params,
        y_spacing_scale=y_spacing,
        wing_1_root_z_translation=wing_1_root_z,
        mid_wing_root_z_translation=mid_root_z,
        wing_7_root_z_translation=wing_7_root_z,
        wing_1_tip_sweep=wing_1_tip_sweep,
        mid_wing_tip_sweep=mid_tip_sweep,
        wing_7_tip_sweep=wing_7_tip_sweep,
        wing_1_tip_z_curve=wing_1_tip_z_curve,
        mid_wing_tip_z_curve=mid_tip_z_curve,
        wing_7_tip_z_curve=wing_7_tip_z_curve,
        min_tip_chord_scale=min_tip_chord_scale,
        min_feather_root_gap=0.0,
    )
    return params


def _get_bounds(config: GAConfig) -> list[tuple[float, float]]:
    total_wings = config.base_params.total_wings
    freedom_level = config.freedom_level.lower()
    if freedom_level not in FREEDOM_LEVEL_ORDER:
        valid = ", ".join(FREEDOM_LEVEL_ORDER)
        raise ValueError(f"Unknown freedom_level={config.freedom_level!r}. Valid values: {valid}")

    base = config.base_params
    base_mid_root_z = (
        base.mid_wing_root_z_translation
        if base.mid_wing_root_z_translation is not None
        else 0.5 * (base.wing_1_root_z_translation + base.wing_7_root_z_translation)
    )
    base_mid_tip_sweep = (
        base.mid_wing_tip_sweep
        if base.mid_wing_tip_sweep is not None
        else 0.5 * (base.wing_1_tip_sweep + base.wing_7_tip_sweep)
    )
    base_mid_tip_z_curve = (
        base.mid_wing_tip_z_curve
        if base.mid_wing_tip_z_curve is not None
        else 0.5 * (base.wing_1_tip_z_curve + base.wing_7_tip_z_curve)
    )

    def locked(value: float) -> tuple[float, float]:
        return (float(value), float(value))

    # "incidence" = only incidence varies; all geometry locked.
    # "zero_coherence", "no_delay", "full" = full geometry search
    #   (closure overrides are applied separately, not through bounds).
    allow_geometry = freedom_level != "incidence"

    bounds = []
    # Incidence shape coefficients in a full-rank orthonormal basis.
    bounds.extend(_incidence_coeff_bounds(config))
    bounds.append(config.spacing_scale_bounds if allow_geometry else locked(base.y_spacing_scale))
    bounds.append(config.root_z_bounds if allow_geometry else locked(base.wing_1_root_z_translation))
    bounds.append(config.root_z_bounds if allow_geometry else locked(base_mid_root_z))
    bounds.append(config.root_z_bounds if allow_geometry else locked(base.wing_7_root_z_translation))
    bounds.append(config.tip_sweep_bounds if allow_geometry else locked(base.wing_1_tip_sweep))
    bounds.append(config.tip_sweep_bounds if allow_geometry else locked(base_mid_tip_sweep))
    bounds.append(config.tip_sweep_bounds if allow_geometry else locked(base.wing_7_tip_sweep))
    bounds.append(config.tip_z_curve_bounds if allow_geometry else locked(base.wing_1_tip_z_curve))
    bounds.append(config.tip_z_curve_bounds if allow_geometry else locked(base_mid_tip_z_curve))
    bounds.append(config.tip_z_curve_bounds if allow_geometry else locked(base.wing_7_tip_z_curve))
    bounds.append(config.min_tip_chord_scale_bounds if allow_geometry else locked(base.min_tip_chord_scale))
    return bounds


def _active_parameter_count(bounds: Sequence[tuple[float, float]]) -> int:
    return sum(1 for low, high in bounds if float(high) > float(low))


def active_parameter_count(config: GAConfig) -> int:
    return _active_parameter_count(_get_bounds(config))


def effective_population_size(config: GAConfig) -> int:
    active_count = max(active_parameter_count(config), 1)
    return max(8, int(config.popsize) * active_count)


def _coarse_observer_indices(config: GAConfig) -> tuple[int, ...]:
    stride = max(config.coarse_observer_stride, 1)
    indices = tuple(range(0, len(config.observers.directions), stride))
    if not indices:
        return (0,)
    return indices


def _coarse_observers(config: GAConfig) -> ObserverGrid:
    indices = _coarse_observer_indices(config)
    return ObserverGrid(
        tuple(config.observers.directions[index] for index in indices),
        tuple(config.observers.weights[index] for index in indices),
    )


def _coarse_target_pattern(config: GAConfig) -> tuple[float, ...]:
    indices = _coarse_observer_indices(config)
    return tuple(config.target_pattern_db[index] for index in indices)


def _coarse_frequencies(config: GAConfig) -> tuple[float, ...]:
    stride = max(config.coarse_frequency_stride, 1)
    coarse = tuple(config.frequencies_hz[index] for index in range(0, len(config.frequencies_hz), stride))
    if not coarse:
        return config.frequencies_hz
    if coarse[-1] != config.frequencies_hz[-1]:
        coarse = coarse + (config.frequencies_hz[-1],)
    return coarse


def _quadratic_profile_tensor(
    progress: torch.Tensor,
    left: torch.Tensor,
    middle: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    t0 = 0.0
    t1 = 0.5
    t2 = 1.0
    l0 = ((progress - t1) * (progress - t2)) / ((t0 - t1) * (t0 - t2))
    l1 = ((progress - t0) * (progress - t2)) / ((t1 - t0) * (t1 - t2))
    l2 = ((progress - t0) * (progress - t1)) / ((t2 - t0) * (t2 - t1))
    return left * l0 + middle * l1 + right * l2


def _trap_weights_tensor(n_eta: int, device: torch.device) -> torch.Tensor:
    delta = 1.0 / (n_eta - 1)
    weights = torch.full((n_eta,), delta, dtype=torch.float32, device=device)
    weights[0] = 0.5 * delta
    weights[-1] = 0.5 * delta
    return weights


def _make_geometry_tensor_context(
    params: WingGeometryParams,
    device: torch.device,
) -> _GeometryTensorContext:
    total_wings = params.total_wings
    midpoint = (total_wings + 1) / 2.0
    edge_distance = max(midpoint - 1.0, 1.0)

    root_chords = []
    tip_span = []
    for feather_index in range(1, total_wings + 1):
        if total_wings == 1:
            chord = params.center_wing_chord
        elif feather_index <= midpoint:
            progress = (midpoint - feather_index) / max(midpoint - 1.0, 1.0)
            chord = params.center_wing_chord + (
                params.wing_1_chord - params.center_wing_chord
            ) * progress
        else:
            progress = (feather_index - midpoint) / max(total_wings - midpoint, 1.0)
            chord = params.center_wing_chord + (
                params.wing_7_chord - params.center_wing_chord
            ) * progress
        root_chords.append(chord)

        normalized_distance = abs(feather_index - midpoint) / edge_distance
        scale = params.mid_wing_span_scale - (
            params.mid_wing_span_scale - 1.0
        ) * normalized_distance
        tip_span.append(params.half_span * scale)

    basis = torch.tensor(_incidence_basis(total_wings), dtype=torch.float32, device=device)
    return _GeometryTensorContext(
        total_wings=total_wings,
        basis=basis,
        root_chords=torch.tensor(root_chords, dtype=torch.float32, device=device),
        tip_span=torch.tensor(tip_span, dtype=torch.float32, device=device),
        feather_progress=torch.linspace(0.0, 1.0, steps=total_wings, dtype=torch.float32, device=device),
        trap_weights={},
        source_chord_fraction=1.0,
        min_feather_root_gap=float(params.min_feather_root_gap),
        sweep_curve_exponent=float(params.sweep_curve_exponent),
        z_curve_exponent=float(params.z_curve_exponent),
    )


def _with_trap_weights(context: _GeometryTensorContext, n_eta: int, device: torch.device) -> _GeometryTensorContext:
    if n_eta in context.trap_weights:
        return context
    trap_weights = dict(context.trap_weights)
    trap_weights[n_eta] = _trap_weights_tensor(n_eta, device)
    return _GeometryTensorContext(
        total_wings=context.total_wings,
        basis=context.basis,
        root_chords=context.root_chords,
        tip_span=context.tip_span,
        feather_progress=context.feather_progress,
        trap_weights=trap_weights,
        source_chord_fraction=context.source_chord_fraction,
        min_feather_root_gap=context.min_feather_root_gap,
        sweep_curve_exponent=context.sweep_curve_exponent,
        z_curve_exponent=context.z_curve_exponent,
    )


def _decode_gene_fields_torch(
    genes_tensor: torch.Tensor,
    geometry_context: _GeometryTensorContext,
) -> dict[str, torch.Tensor]:
    total_wings = geometry_context.total_wings
    incidences = genes_tensor[:, :total_wings] @ geometry_context.basis.T
    spacing = genes_tensor[:, total_wings]
    root_z_left = genes_tensor[:, total_wings + 1]
    root_z_mid = genes_tensor[:, total_wings + 2]
    root_z_right = genes_tensor[:, total_wings + 3]
    tip_sweep_left = genes_tensor[:, total_wings + 4]
    tip_sweep_mid = genes_tensor[:, total_wings + 5]
    tip_sweep_right = genes_tensor[:, total_wings + 6]
    tip_z_left = genes_tensor[:, total_wings + 7]
    tip_z_mid = genes_tensor[:, total_wings + 8]
    tip_z_right = genes_tensor[:, total_wings + 9]
    min_tip_chord_scale = genes_tensor[:, total_wings + 10]

    progress = geometry_context.feather_progress.view(1, -1)
    tip_sweep = _quadratic_profile_tensor(
        progress,
        tip_sweep_left.view(-1, 1),
        tip_sweep_mid.view(-1, 1),
        tip_sweep_right.view(-1, 1),
    )
    tip_z = _quadratic_profile_tensor(
        progress,
        tip_z_left.view(-1, 1),
        tip_z_mid.view(-1, 1),
        tip_z_right.view(-1, 1),
    )
    root_z = _quadratic_profile_tensor(
        progress,
        root_z_left.view(-1, 1),
        root_z_mid.view(-1, 1),
        root_z_right.view(-1, 1),
    )

    root_chords = geometry_context.root_chords.view(1, -1)
    gap_nominal = root_chords * torch.clamp(spacing.view(-1, 1) - 1.0, min=0.0)
    gaps = torch.maximum(
        gap_nominal,
        torch.full_like(gap_nominal, geometry_context.min_feather_root_gap),
    )
    gaps[:, 0] = 0.0
    increments = root_chords + gaps
    increments[:, 0] = 0.0
    trailing_edge_y = torch.cumsum(increments, dim=1)

    return {
        "incidences": incidences,
        "tip_sweep": tip_sweep,
        "tip_z": tip_z,
        "root_z": root_z,
        "trailing_edge_y": trailing_edge_y,
        "min_tip_chord_scale": min_tip_chord_scale,
    }


def _paper_midline_points_torch(
    fields: dict[str, torch.Tensor],
    geometry_context: _GeometryTensorContext,
    eta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    root_chords = geometry_context.root_chords.view(1, -1, 1)
    sweep_curve = eta.pow(geometry_context.sweep_curve_exponent).view(1, 1, -1)
    z_curve = eta.pow(geometry_context.z_curve_exponent).view(1, 1, -1)

    rounding_progress = torch.clamp((eta - 0.8) / 0.2, min=0.0, max=1.0)
    base_scale = torch.sqrt(torch.clamp(1.0 - rounding_progress * rounding_progress, min=0.0))
    scale = torch.where(
        eta.view(1, -1) < 0.8,
        torch.ones((fields["min_tip_chord_scale"].shape[0], eta.numel()), dtype=torch.float32, device=eta.device),
        torch.maximum(
            base_scale.view(1, -1).expand(fields["min_tip_chord_scale"].shape[0], -1),
            fields["min_tip_chord_scale"].view(-1, 1),
        ),
    )

    current_chord = root_chords * scale.unsqueeze(1)
    trailing_edge_y = (
        fields["trailing_edge_y"].unsqueeze(-1)
        + fields["tip_sweep"].unsqueeze(-1) * sweep_curve
        - 0.5 * (root_chords - current_chord)
    )
    z = fields["root_z"].unsqueeze(-1) + fields["tip_z"].unsqueeze(-1) * z_curve
    span_station = (
        geometry_context.tip_span.view(1, -1, 1) * eta.view(1, 1, -1)
    ).expand(fields["incidences"].shape[0], -1, -1)

    incidence_rad = torch.deg2rad(fields["incidences"]).unsqueeze(-1)
    dy = current_chord * (geometry_context.source_chord_fraction - 0.25)
    pivot_y = trailing_edge_y - 0.75 * current_chord
    y_rot = pivot_y + dy * torch.cos(incidence_rad)
    z_rot = z - dy * torch.sin(incidence_rad)

    points = torch.stack((y_rot, span_station, z_rot), dim=-1)
    return points, current_chord, incidence_rad, z


def _source_grid_torch(
    fields: dict[str, torch.Tensor],
    geometry_context: _GeometryTensorContext,
    n_eta: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    geometry_context = _with_trap_weights(geometry_context, n_eta, fields["incidences"].device)
    eta = torch.linspace(0.0, 1.0, steps=n_eta, dtype=torch.float32, device=fields["incidences"].device)
    points, chords, incidence_rad, _z = _paper_midline_points_torch(fields, geometry_context, eta)

    h = 1.0e-5
    eta_prev = torch.clamp(eta - h, min=0.0)
    eta_next = torch.clamp(eta + h, max=1.0)
    points_prev, _, _, _ = _paper_midline_points_torch(fields, geometry_context, eta_prev)
    points_next, _, _, _ = _paper_midline_points_torch(fields, geometry_context, eta_next)
    delta = torch.clamp(eta_next - eta_prev, min=1.0e-12).view(1, 1, -1, 1)
    jacobian = torch.linalg.norm((points_next - points_prev) / delta, dim=-1)

    weights = geometry_context.trap_weights[n_eta].view(1, 1, -1) * jacobian
    loading_directions = torch.stack(
        (
            torch.sin(incidence_rad).expand_as(chords),
            torch.zeros_like(chords),
            torch.cos(incidence_rad).expand_as(chords),
        ),
        dim=-1,
    )
    incidence_deg = torch.rad2deg(incidence_rad).expand_as(chords)

    population = points.shape[0]
    total_sources = geometry_context.total_wings * n_eta
    return (
        points.reshape(population, total_sources, 3),
        chords.reshape(population, total_sources),
        incidence_deg.reshape(population, total_sources),
        loading_directions.reshape(population, total_sources, 3),
        weights.reshape(population, total_sources),
    )


def _segment_orientation(p: torch.Tensor, q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])


def _segment_on_segment(p: torch.Tensor, q: torch.Tensor, r: torch.Tensor, tolerance: float) -> torch.Tensor:
    return (
        (torch.minimum(p[..., 0], r[..., 0]) - tolerance <= q[..., 0])
        & (q[..., 0] <= torch.maximum(p[..., 0], r[..., 0]) + tolerance)
        & (torch.minimum(p[..., 1], r[..., 1]) - tolerance <= q[..., 1])
        & (q[..., 1] <= torch.maximum(p[..., 1], r[..., 1]) + tolerance)
    )


def _intersection_penalties_torch(
    fields: dict[str, torch.Tensor],
    geometry_context: _GeometryTensorContext,
    intersection_n_eta: int,
    intersection_penalty: float,
) -> torch.Tensor:
    eta = torch.linspace(0.0, 1.0, steps=intersection_n_eta, dtype=torch.float32, device=fields["incidences"].device)
    points, _, _, _ = _paper_midline_points_torch(fields, geometry_context, eta)
    projected = points[..., :2]
    tolerance = 1.0e-7

    counts = torch.zeros((projected.shape[0],), dtype=torch.float32, device=projected.device)
    for left_index in range(geometry_context.total_wings):
        a0 = projected[:, left_index, :-1, :]
        a1 = projected[:, left_index, 1:, :]
        for right_index in range(left_index + 1, geometry_context.total_wings):
            b0 = projected[:, right_index, :-1, :]
            b1 = projected[:, right_index, 1:, :]

            a0e = a0.unsqueeze(2)
            a1e = a1.unsqueeze(2)
            b0e = b0.unsqueeze(1)
            b1e = b1.unsqueeze(1)

            o1 = _segment_orientation(a0e, a1e, b0e)
            o2 = _segment_orientation(a0e, a1e, b1e)
            o3 = _segment_orientation(b0e, b1e, a0e)
            o4 = _segment_orientation(b0e, b1e, a1e)

            proper = (o1 * o2 < -tolerance) & (o3 * o4 < -tolerance)
            collinear = (
                ((torch.abs(o1) <= tolerance) & _segment_on_segment(a0e, b0e, a1e, tolerance))
                | ((torch.abs(o2) <= tolerance) & _segment_on_segment(a0e, b1e, a1e, tolerance))
                | ((torch.abs(o3) <= tolerance) & _segment_on_segment(b0e, a0e, b1e, tolerance))
                | ((torch.abs(o4) <= tolerance) & _segment_on_segment(b0e, a1e, b1e, tolerance))
            )
            counts += (proper | collinear).sum(dim=(1, 2), dtype=torch.float32)

    return counts * intersection_penalty


def _sample_distinct_parents(population_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    self_indices = torch.arange(population_size, device=device)

    a = torch.randint(0, population_size - 1, (population_size,), device=device)
    a = a + (a >= self_indices).to(torch.long)

    b = torch.randint(0, population_size, (population_size,), device=device)
    invalid_b = (b == self_indices) | (b == a)
    while torch.any(invalid_b):
        b[invalid_b] = torch.randint(0, population_size, (int(invalid_b.sum().item()),), device=device)
        invalid_b = (b == self_indices) | (b == a)

    c = torch.randint(0, population_size, (population_size,), device=device)
    invalid_c = (c == self_indices) | (c == a) | (c == b)
    while torch.any(invalid_c):
        c[invalid_c] = torch.randint(0, population_size, (int(invalid_c.sum().item()),), device=device)
        invalid_c = (c == self_indices) | (c == a) | (c == b)

    return a, b, c


def _make_fitness_context(
    observers: ObserverGrid,
    frequencies_hz: tuple[float, ...],
    target_pattern_db: tuple[float, ...],
    flow: FlowConfig,
    closures: ClosureParams,
    device: torch.device,
) -> _FitnessContext:
    observer_dirs = torch.as_tensor(observers.directions, dtype=torch.float32, device=device)
    frequency_tensor = torch.as_tensor(frequencies_hz, dtype=torch.float32, device=device)
    target_tensor = torch.as_tensor(target_pattern_db, dtype=torch.float32, device=device)
    target_normalized = target_tensor - torch.max(target_tensor)
    scales = torch.tensor(
        [closures.coherence_x, closures.coherence_y, closures.coherence_z],
        dtype=torch.float32,
        device=device,
    )
    return _FitnessContext(
        observers=observers,
        observer_dirs=observer_dirs,
        frequencies_hz=frequencies_hz,
        frequency_tensor=frequency_tensor,
        target_normalized=target_normalized,
        flow=flow,
        closures=closures,
        scales=scales,
    )


@torch.no_grad()
def differential_evolution_torch(
    objective_fn,
    bounds: List[Tuple[float, float]],
    popsize: int = 15,
    maxiter: int = 0,
    mutation: Tuple[float, float] = (0.5, 1.0),
    recombination: float = 0.7,
    seed: int = 42,
    device: torch.device = None,
    patience: int = 10,
    min_fitness_improvement_db2: float = 0.01,
    initial_genes: Sequence[float] | None = None,
):
    if device is None:
        device = torch.device("cpu")
        
    torch.manual_seed(seed)
    
    D = len(bounds)
    
    bounds_tensor = torch.tensor(bounds, dtype=torch.float32, device=device)
    low = bounds_tensor[:, 0]
    high = bounds_tensor[:, 1]
    active_indices = torch.nonzero(high > low, as_tuple=False).flatten()
    active_D = int(active_indices.numel())
    population_basis = max(active_D, 1)
    P = max(8, int(popsize) * population_basis)
    print(
        f"DE active parameters: {active_D}/{D}; "
        f"population: {P} ({popsize} per active parameter)"
    )
    
    # Initialize population
    pop = low.unsqueeze(0).expand(P, D).clone()
    if active_D:
        active_low = low[active_indices]
        active_high = high[active_indices]
        pop[:, active_indices] = active_low + torch.rand((P, active_D), device=device) * (
            active_high - active_low
        )
        if initial_genes is not None:
            init = torch.as_tensor(initial_genes, dtype=torch.float32, device=device)
            init = torch.max(torch.min(init, high), low)
            pop[0] = init
            warm_count = min(P - 1, max(4, P // 3))
            if warm_count > 0:
                span = torch.clamp(active_high - active_low, min=1.0e-6)
                jitter = 0.08 * span.unsqueeze(0) * torch.randn((warm_count, active_D), device=device)
                local = init[active_indices].unsqueeze(0) + jitter
                local = torch.max(torch.min(local, active_high), active_low)
                pop[1 : 1 + warm_count, active_indices] = local
    fitness = objective_fn(pop)
    
    best_idx = torch.argmin(fitness)
    best_fitness = fitness[best_idx].item()
    
    no_improve_count = 0
    gen = 0

    while True:
        gen += 1
        parent_a, parent_b, parent_c = _sample_distinct_parents(P, device)
        a = pop[parent_a]
        b = pop[parent_b]
        c = pop[parent_c]
        
        mut_factor = mutation[0] + torch.rand((P, 1), device=device) * (mutation[1] - mutation[0])
        mutant = pop.clone()
        if active_D:
            mutant[:, active_indices] = (
                a[:, active_indices]
                + mut_factor * (b[:, active_indices] - c[:, active_indices])
            )
        
        # Clamp mutant
        mutant = torch.max(torch.min(mutant, high), low)
        
        # Crossover
        cross_mask = torch.zeros((P, D), dtype=torch.bool, device=device)
        if active_D:
            active_cross_mask = torch.rand((P, active_D), device=device) < recombination
            force_cross = torch.randint(0, active_D, (P, 1), device=device)
            active_cross_mask.scatter_(1, force_cross, True)
            cross_mask[:, active_indices] = active_cross_mask
        
        trial = torch.where(cross_mask, mutant, pop)
        
        # Evaluate
        trial_fitness = objective_fn(trial)
        
        # Selection
        improved = trial_fitness < fitness
        pop = torch.where(improved.unsqueeze(1), trial, pop)
        fitness = torch.where(improved, trial_fitness, fitness)
        
        # Track best
        best_idx = torch.argmin(fitness)
        current_best = fitness[best_idx].item()
        
        if current_best < best_fitness - min_fitness_improvement_db2:
            best_fitness = current_best
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        if maxiter > 0:
            print(f"Generation {gen}/{maxiter} - Best Fitness: {best_fitness:.4f}")
        else:
            print(f"Generation {gen} - Best Fitness: {best_fitness:.4f}")
        
        if no_improve_count >= patience:
            print(f"Early stopping at generation {gen}: No improvement in {patience} generations.")
            break

        if maxiter > 0 and gen >= maxiter:
            print(f"Reached generation cap at {gen}.")
            break
        
    return pop[best_idx].cpu().numpy(), best_fitness


@torch.no_grad()
def _evaluate_population_fitness(
    results: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    context: _FitnessContext,
    device: torch.device,
) -> torch.Tensor:
    points, chords, incidence_deg, loading_directions, weights = results

    spp = evaluate_spp_torch(
        points,
        chords,
        incidence_deg,
        loading_directions,
        weights,
        observers=None,
        frequencies_hz=context.frequency_tensor,
        flow=context.flow,
        closures=context.closures,
        device=device,
        observer_dirs=context.observer_dirs,
        scales=context.scales,
    )

    if context.frequency_tensor.numel() == 1:
        spp_sum = spp
    else:
        df = context.frequency_tensor[1:] - context.frequency_tensor[:-1]
        df = df.unsqueeze(0).unsqueeze(-1)
        spp_sum = 0.5 * torch.sum(df * (spp[:, :-1, :] + spp[:, 1:, :]), dim=1)

    p_ref_sq = context.flow.p_ref ** 2
    spp_db = 10.0 * torch.log10(torch.clamp(spp_sum / p_ref_sq, min=1e-12))
    sim_peak, _ = torch.max(spp_db, dim=1, keepdim=True)
    sim_normalized = spp_db - sim_peak

    target_normalized = context.target_normalized.unsqueeze(0).expand(points.shape[0], -1)
    return torch.mean((sim_normalized - target_normalized) ** 2, dim=1)


@torch.no_grad()
def _objective_torch(
    genes_tensor: torch.Tensor,
    config: GAConfig,
    device: torch.device,
    geometry_context: _GeometryTensorContext,
    coarse_context: _FitnessContext,
    fine_context: _FitnessContext,
) -> torch.Tensor:
    fields = _decode_gene_fields_torch(genes_tensor, geometry_context)
    penalties = _intersection_penalties_torch(
        fields,
        geometry_context,
        config.intersection_n_eta,
        config.intersection_penalty,
    )

    coarse_results = _source_grid_torch(
        fields,
        geometry_context,
        config.coarse_n_eta,
    )
    coarse_fitness = _evaluate_population_fitness(
        coarse_results,
        coarse_context,
        device,
    ) + penalties

    population_size = genes_tensor.shape[0]
    refine_count = max(
        1,
        min(
            population_size,
            max(
                config.refinement_min_candidates,
                math.ceil(population_size * config.refinement_top_fraction),
            ),
        ),
    )
    refine_indices = torch.argsort(coarse_fitness)[:refine_count]
    fitness = coarse_fitness.clone()

    fine_fields = {key: value.index_select(0, refine_indices) for key, value in fields.items()}
    fine_results = _source_grid_torch(
        fine_fields,
        geometry_context,
        config.n_eta,
    )
    fine_fitness = _evaluate_population_fitness(
        fine_results,
        fine_context,
        device,
    ) + penalties.index_select(0, refine_indices)
    fitness[refine_indices] = fine_fitness
    return fitness


def run_optimization_torch(
    config: GAConfig,
    runtime: OptimizationRuntime | None = None,
    warm_start_genes: Sequence[float] | None = None,
) -> Tuple[WingGeometryParams, np.ndarray, float, bool]:
    runtime = OptimizationRuntime() if runtime is None else runtime
    device = runtime.device
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        runtime.device = device
    print(f"Using device: {device}")
    bounds = _get_bounds(config)
    geometry_context = runtime.geometry_context
    if geometry_context is None:
        geometry_context = _make_geometry_tensor_context(config.base_params, device)
        runtime.geometry_context = geometry_context
    coarse_context = _make_fitness_context(
        _coarse_observers(config),
        _coarse_frequencies(config),
        _coarse_target_pattern(config),
        config.flow,
        config.closures,
        device,
    )
    fine_context = _make_fitness_context(
        config.observers,
        config.frequencies_hz,
        config.target_pattern_db,
        config.flow,
        config.closures,
        device,
    )

    start_time = time.time()

    best_genes, best_fitness = differential_evolution_torch(
        objective_fn=lambda pop: _objective_torch(
            pop,
            config,
            device,
            geometry_context,
            coarse_context,
            fine_context,
        ),
        bounds=bounds,
        popsize=config.popsize,
        maxiter=config.maxiter,
        mutation=config.mutation,
        recombination=config.recombination,
        seed=config.seed,
        device=device,
        patience=config.patience,
        min_fitness_improvement_db2=config.min_fitness_improvement_db2,
        initial_genes=warm_start_genes,
    )
    
    elapsed = time.time() - start_time
    print(f"PyTorch GA Completed in {elapsed:.1f}s.")
    print(f"  Best Score (Normalized MSE): {best_fitness:.2f} dB^2")
    
    best_genes_float = [float(x) for x in best_genes]
    best_params = _unpack_genes(best_genes_float, config.base_params)
    
    return best_params, best_genes, float(best_fitness), True
