"""Radiating-covariance operators and controllability diagnostics.

This module keeps the theoretical objects used by the manuscript explicit:

* source covariance ``Cq``;
* observer radiation vector ``a``;
* integrated radiation operator ``W = sum_o w_o a_o a_o^H``;
* modal radiation ``sum_r lambda_r |a^H u_r|^2``;
* radiating off-diagonal covariance indices;
* spherical-harmonic directivity controllability diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from scipy import special

from .acoustics import evaluate_spp
from .closures import (
    ClosureParams,
    FlowConfig,
    source_cross_spectral_matrix,
    transfer_weights,
)
from .geometry import WingGeometryParams, default_geometry, source_grid
from .observers import ObserverGrid
from .validation import spectral_level_db


Array = np.ndarray


@dataclass(frozen=True)
class ModalRadiationResult:
    eigenvalues: Array
    modal_terms: Array
    direct_spp: float
    modal_spp: float
    relative_error: float


@dataclass(frozen=True)
class MechanismMetrics:
    raw_coherence_availability: float
    radiating_phase_steerability: float
    radiating_covariance_gain: float
    phase_ablation_rmse_db: float
    offdiag_bound_ratio_max: float

    @property
    def radiating_covariance_fraction(self) -> float:
        """Backward-compatible alias for the former name.

        The quantity is a radiation-weighted gain and is not bounded by one.
        """
        return self.radiating_covariance_gain


@dataclass(frozen=True)
class ControllabilityResult:
    variable_names: tuple[str, ...]
    coefficient_labels: tuple[str, ...]
    singular_values: tuple[float, ...]
    control_dimension_1e2: int
    control_dimension_1e3: int
    jacobian: Array


def radius_factor(flow: FlowConfig) -> float:
    radius = max(flow.observer_radius, 1.0e-12)
    return 1.0 / ((4.0 * math.pi * radius) ** 2)


def source_csd_matrix(
    grid,
    frequency_hz: float,
    flow: FlowConfig | None = None,
    closures: ClosureParams | None = None,
) -> Array:
    flow = FlowConfig() if flow is None else flow
    closures = ClosureParams() if closures is None else closures
    return np.asarray(
        source_cross_spectral_matrix(grid, frequency_hz, flow, closures),
        dtype=np.complex128,
    )


def transfer_vector(
    grid,
    observer_direction: tuple[float, float, float],
    frequency_hz: float,
    flow: FlowConfig | None = None,
    include_quadrature: bool = True,
) -> Array:
    flow = FlowConfig() if flow is None else flow
    return np.asarray(
        transfer_weights(
            grid,
            observer_direction,
            frequency_hz,
            flow,
            include_quadrature=include_quadrature,
        ),
        dtype=np.complex128,
    )


def spp_quadratic(weights: Array, cq: Array, flow: FlowConfig | None = None) -> float:
    flow = FlowConfig() if flow is None else flow
    value = np.vdot(weights, cq @ weights)
    return max(float(value.real) * radius_factor(flow), 0.0)


def modal_radiation_decomposition(
    weights: Array,
    cq: Array,
    flow: FlowConfig | None = None,
) -> ModalRadiationResult:
    flow = FlowConfig() if flow is None else flow
    hermitian_cq = 0.5 * (cq + cq.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian_cq)
    projections = eigenvectors.conj().T @ weights
    modal_terms = eigenvalues * np.abs(projections) ** 2
    direct_spp = spp_quadratic(weights, cq, flow)
    modal_spp = max(float(modal_terms.sum().real) * radius_factor(flow), 0.0)
    denominator = max(abs(direct_spp), 1.0e-300)
    return ModalRadiationResult(
        eigenvalues=eigenvalues,
        modal_terms=modal_terms * radius_factor(flow),
        direct_spp=direct_spp,
        modal_spp=modal_spp,
        relative_error=abs(direct_spp - modal_spp) / denominator,
    )


def diagonal_covariance(cq: Array) -> Array:
    return np.diag(np.diag(cq)).astype(np.complex128)


def rank_one_covariance(amplitudes: Array, phases: Array | None = None) -> Array:
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    if phases is None:
        phases = np.zeros_like(amplitudes)
    phases = np.asarray(phases, dtype=np.float64)
    q = amplitudes * np.exp(1j * phases)
    return np.outer(q, q.conj()).astype(np.complex128)


def radiation_operator(
    grid,
    observers: ObserverGrid,
    frequency_hz: float,
    flow: FlowConfig | None = None,
) -> Array:
    flow = FlowConfig() if flow is None else flow
    n_sources = grid.n
    operator = np.zeros((n_sources, n_sources), dtype=np.complex128)
    for direction, weight in zip(observers.directions, observers.weights):
        vector = transfer_vector(grid, direction, frequency_hz, flow)
        operator += float(weight) * np.outer(vector, vector.conj())
    return 0.5 * (operator + operator.conj().T)


def normalize_radiation_operator(operator: Array) -> Array:
    trace = float(np.trace(operator).real)
    if trace <= 1.0e-300:
        return operator.copy()
    return operator * (operator.shape[0] / trace)


def _sqrt_psd(matrix: Array, tolerance: float = 1.0e-13) -> Array:
    hermitian = 0.5 * (matrix + matrix.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    clipped = np.clip(eigenvalues, 0.0, None)
    clipped[eigenvalues < -tolerance * max(np.linalg.norm(hermitian, 2), 1.0)] = 0.0
    return (eigenvectors * np.sqrt(clipped)) @ eigenvectors.conj().T


def raw_coherence_availability(cq: Array) -> float:
    denominator = np.linalg.norm(cq, "fro")
    if denominator <= 1.0e-300:
        return 0.0
    return float(np.linalg.norm(cq - diagonal_covariance(cq), "fro") / denominator)


def radiating_phase_steerability(cq: Array, operator: Array) -> float:
    sqrt_w = _sqrt_psd(normalize_radiation_operator(operator))
    denominator_matrix = sqrt_w @ cq @ sqrt_w
    denominator = np.linalg.norm(denominator_matrix, "fro")
    if denominator <= 1.0e-300:
        return 0.0
    off = cq - diagonal_covariance(cq)
    numerator = np.linalg.norm(sqrt_w @ off @ sqrt_w, "fro")
    return float(numerator / denominator)


def radiating_covariance_gain(cq: Array, operator: Array) -> float:
    source_energy = float(np.trace(cq).real)
    if source_energy <= 1.0e-300:
        return 0.0
    normalized_operator = normalize_radiation_operator(operator)
    radiating_energy = float(np.trace(normalized_operator @ cq).real)
    return radiating_energy / source_energy


def radiating_covariance_fraction(cq: Array, operator: Array) -> float:
    """Backward-compatible alias for :func:`radiating_covariance_gain`."""
    return radiating_covariance_gain(cq, operator)


def phase_ablation_rmse_db(
    grid,
    observers: ObserverGrid,
    frequency_hz: float,
    flow: FlowConfig | None = None,
    closures: ClosureParams | None = None,
    peak_normalize: bool = True,
) -> float:
    flow = FlowConfig() if flow is None else flow
    closures = ClosureParams() if closures is None else closures
    cq = source_csd_matrix(grid, frequency_hz, flow, closures)
    diag = diagonal_covariance(cq)
    full_levels = []
    diagonal_levels = []
    for direction in observers.directions:
        vector = transfer_vector(grid, direction, frequency_hz, flow)
        full_levels.append(spectral_level_db(spp_quadratic(vector, cq, flow), flow))
        diagonal_levels.append(spectral_level_db(spp_quadratic(vector, diag, flow), flow))
    full = np.asarray(full_levels, dtype=np.float64)
    diagonal = np.asarray(diagonal_levels, dtype=np.float64)
    if peak_normalize:
        full = full - np.max(full)
        diagonal = diagonal - np.max(diagonal)
    weights = np.asarray(observers.weights, dtype=np.float64)
    weights = weights / max(float(weights.sum()), 1.0e-300)
    return float(np.sqrt(np.sum(weights * (full - diagonal) ** 2)))


def offdiag_interference_bound_ratio(
    cq: Array,
    weight_vectors: Iterable[Array],
) -> float:
    off = cq - diagonal_covariance(cq)
    off_norm = np.linalg.norm(off, "fro")
    max_ratio = 0.0
    for vector in weight_vectors:
        delta = abs(np.vdot(vector, off @ vector))
        bound = off_norm * float(np.vdot(vector, vector).real)
        if bound > 1.0e-300:
            max_ratio = max(max_ratio, float(delta / bound))
        elif delta > 1.0e-250:
            return math.inf
    return max_ratio


def mechanism_metrics(
    grid,
    observers: ObserverGrid,
    frequency_hz: float,
    flow: FlowConfig | None = None,
    closures: ClosureParams | None = None,
) -> MechanismMetrics:
    flow = FlowConfig() if flow is None else flow
    closures = ClosureParams() if closures is None else closures
    cq = source_csd_matrix(grid, frequency_hz, flow, closures)
    operator = radiation_operator(grid, observers, frequency_hz, flow)
    vectors = [
        transfer_vector(grid, direction, frequency_hz, flow)
        for direction in observers.directions
    ]
    return MechanismMetrics(
        raw_coherence_availability=raw_coherence_availability(cq),
        radiating_phase_steerability=radiating_phase_steerability(cq, operator),
        radiating_covariance_gain=radiating_covariance_gain(cq, operator),
        phase_ablation_rmse_db=phase_ablation_rmse_db(
            grid,
            observers,
            frequency_hz,
            flow,
            closures,
        ),
        offdiag_bound_ratio_max=offdiag_interference_bound_ratio(cq, vectors),
    )


def spherical_harmonic_labels(l_max: int) -> tuple[str, ...]:
    return tuple(
        f"l{ell}_m{m:+d}"
        for ell in range(l_max + 1)
        for m in range(-ell, ell + 1)
    )


def _spherical_angles(direction: tuple[float, float, float]) -> tuple[float, float]:
    x, y, z = direction
    theta = math.acos(max(min(z, 1.0), -1.0))
    phi = math.atan2(y, x)
    if phi < 0.0:
        phi += 2.0 * math.pi
    return theta, phi


def spherical_harmonic_matrix(observers: ObserverGrid, l_max: int) -> Array:
    rows = []
    for direction in observers.directions:
        theta, phi = _spherical_angles(direction)
        values = []
        for ell in range(l_max + 1):
            for m in range(-ell, ell + 1):
                if hasattr(special, "sph_harm_y"):
                    values.append(special.sph_harm_y(ell, m, theta, phi))
                else:
                    values.append(special.sph_harm(m, ell, phi, theta))
        rows.append(values)
    return np.asarray(rows, dtype=np.complex128)


def spherical_harmonic_coefficients(
    levels_db: Iterable[float],
    observers: ObserverGrid,
    l_max: int,
) -> Array:
    levels = np.asarray(tuple(levels_db), dtype=np.float64)
    harmonics = spherical_harmonic_matrix(observers, l_max)
    weights = np.asarray(observers.weights, dtype=np.float64)
    return np.sum((levels * weights)[:, None] * np.conjugate(harmonics), axis=0)


def real_coefficient_vector(coefficients: Array) -> Array:
    return np.concatenate([coefficients.real, coefficients.imag]).astype(np.float64)


def directivity_levels_db(
    params: WingGeometryParams,
    observers: ObserverGrid,
    frequency_hz: float,
    flow: FlowConfig | None = None,
    closures: ClosureParams | None = None,
    n_eta: int = 16,
    peak_normalize: bool = True,
) -> Array:
    flow = FlowConfig() if flow is None else flow
    closures = ClosureParams() if closures is None else closures
    result = evaluate_spp(
        source_grid(params, n_eta=n_eta),
        observers,
        [frequency_hz],
        flow,
        closures,
    )
    levels = np.asarray(
        [spectral_level_db(value, flow) for value in result.spp[0]],
        dtype=np.float64,
    )
    if peak_normalize:
        levels = levels - np.max(levels)
    return levels


def design_variable_names(
    params: WingGeometryParams | None = None,
    freedom_level: str = "full",
) -> tuple[str, ...]:
    params = default_geometry() if params is None else params
    incidence = tuple(f"incidence_{index}" for index in range(params.total_wings))
    if freedom_level == "incidence":
        return incidence
    return incidence + (
        "y_spacing_scale",
        "wing_1_root_z_translation",
        "mid_wing_root_z_translation",
        "wing_7_root_z_translation",
        "wing_1_tip_sweep",
        "mid_wing_tip_sweep",
        "wing_7_tip_sweep",
        "wing_1_tip_z_curve",
        "mid_wing_tip_z_curve",
        "wing_7_tip_z_curve",
        "min_tip_chord_scale",
    )


def design_variable_step(name: str) -> float:
    if name.startswith("incidence_"):
        return 0.25
    if name == "y_spacing_scale":
        return 0.01
    if name == "min_tip_chord_scale":
        return 0.005
    return 0.005


def _mid_value(value: float | None, left: float, right: float) -> float:
    return 0.5 * (left + right) if value is None else float(value)


def perturb_geometry(
    params: WingGeometryParams,
    variable_name: str,
    delta: float,
) -> WingGeometryParams:
    from dataclasses import replace

    if variable_name.startswith("incidence_"):
        index = int(variable_name.split("_", 1)[1])
        incidences = list(params.incidence_angles_deg())
        incidences[index] += delta
        return params.with_incidence_angles(incidences)
    if variable_name == "y_spacing_scale":
        return replace(params, y_spacing_scale=max(params.y_spacing_scale + delta, 0.0))
    if variable_name == "min_tip_chord_scale":
        return replace(
            params,
            min_tip_chord_scale=max(params.min_tip_chord_scale + delta, 1.0e-4),
        )
    if variable_name == "mid_wing_root_z_translation":
        value = _mid_value(
            params.mid_wing_root_z_translation,
            params.wing_1_root_z_translation,
            params.wing_7_root_z_translation,
        )
        return replace(params, mid_wing_root_z_translation=value + delta)
    if variable_name == "mid_wing_tip_sweep":
        value = _mid_value(
            params.mid_wing_tip_sweep,
            params.wing_1_tip_sweep,
            params.wing_7_tip_sweep,
        )
        return replace(params, mid_wing_tip_sweep=value + delta)
    if variable_name == "mid_wing_tip_z_curve":
        value = _mid_value(
            params.mid_wing_tip_z_curve,
            params.wing_1_tip_z_curve,
            params.wing_7_tip_z_curve,
        )
        return replace(params, mid_wing_tip_z_curve=value + delta)
    if not hasattr(params, variable_name):
        raise ValueError(f"Unknown design variable: {variable_name}")
    return replace(params, **{variable_name: getattr(params, variable_name) + delta})


def controllability_jacobian(
    params: WingGeometryParams,
    observers: ObserverGrid,
    frequency_hz: float,
    flow: FlowConfig | None = None,
    closures: ClosureParams | None = None,
    n_eta: int = 12,
    l_max: int = 3,
    freedom_level: str = "full",
) -> ControllabilityResult:
    flow = FlowConfig() if flow is None else flow
    closures = ClosureParams() if closures is None else closures
    variables = design_variable_names(params, freedom_level)
    columns = []
    for name in variables:
        step = design_variable_step(name)
        plus = perturb_geometry(params, name, step)
        minus = perturb_geometry(params, name, -step)
        plus_levels = directivity_levels_db(
            plus, observers, frequency_hz, flow, closures, n_eta=n_eta
        )
        minus_levels = directivity_levels_db(
            minus, observers, frequency_hz, flow, closures, n_eta=n_eta
        )
        plus_coeffs = real_coefficient_vector(
            spherical_harmonic_coefficients(plus_levels, observers, l_max)
        )
        minus_coeffs = real_coefficient_vector(
            spherical_harmonic_coefficients(minus_levels, observers, l_max)
        )
        columns.append((plus_coeffs - minus_coeffs) / (2.0 * step))
    jacobian = np.column_stack(columns) if columns else np.zeros((0, 0))
    singular_values = np.linalg.svd(jacobian, compute_uv=False) if columns else np.zeros(0)
    if singular_values.size == 0 or singular_values[0] <= 1.0e-300:
        d_1e2 = 0
        d_1e3 = 0
    else:
        d_1e2 = int(np.sum(singular_values / singular_values[0] > 1.0e-2))
        d_1e3 = int(np.sum(singular_values / singular_values[0] > 1.0e-3))
    labels = tuple(
        f"Re({label})" for label in spherical_harmonic_labels(l_max)
    ) + tuple(f"Im({label})" for label in spherical_harmonic_labels(l_max))
    return ControllabilityResult(
        variable_names=variables,
        coefficient_labels=labels,
        singular_values=tuple(float(value) for value in singular_values),
        control_dimension_1e2=d_1e2,
        control_dimension_1e3=d_1e3,
        jacobian=jacobian,
    )


def target_projection_score(
    target_levels_db: Iterable[float],
    observers: ObserverGrid,
    l_max: int,
    jacobian: Array,
    epsilon: float = 1.0e-2,
) -> float:
    target = np.asarray(tuple(target_levels_db), dtype=np.float64)
    target = target - np.max(target)
    target_vector = real_coefficient_vector(
        spherical_harmonic_coefficients(target, observers, l_max)
    )
    norm_sq = float(np.dot(target_vector, target_vector))
    if norm_sq <= 1.0e-300 or jacobian.size == 0:
        return 0.0
    u, s, _vh = np.linalg.svd(jacobian, full_matrices=False)
    if s.size == 0 or s[0] <= 1.0e-300:
        return 0.0
    rank = int(np.sum(s / s[0] > epsilon))
    if rank <= 0:
        return 0.0
    projected = u[:, :rank] @ (u[:, :rank].T @ target_vector)
    return float(np.dot(projected, projected) / norm_sq)
