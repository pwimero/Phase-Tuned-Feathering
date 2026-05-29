"""Flow settings and calibrated closure laws for the acoustic model."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

from .geometry import SourceGrid


@dataclass(frozen=True)
class FlowConfig:
    """Ambient flow and acoustic reference settings."""

    u_inf: float = 30.0
    rho0: float = 1.225
    c0: float = 343.0
    dynamic_viscosity: float = 1.81e-5
    p_ref: float = 20e-6
    observer_radius: float = 1.0

    @property
    def mach(self) -> float:
        if self.c0 <= 0.0:
            raise ValueError("c0 must be positive.")
        return self.u_inf / self.c0


@dataclass(frozen=True)
class ClosureParams:
    """Semi-empirical closure parameters.

    The transfer kernel and source autospectrum are calibrated surrogates. They
    should not be presented as absolute aeroacoustic laws without calibration.
    """

    cq: float = 1.0e-12
    beta: float = 0.7
    u_c: float | None = None
    coherence_x: float = 0.30
    coherence_y: float = 0.70
    coherence_z: float = 0.30
    incidence_amplitude_coeff: float = 0.0
    incidence_ref_deg: float = 0.0
    incidence_delay_per_rad: float = 0.0
    coherence_model: str = "exponential"

    def with_coherence_model(self, coherence_model: str) -> "ClosureParams":
        return replace(self, coherence_model=coherence_model)


def convection_velocity(flow: FlowConfig, closures: ClosureParams) -> float:
    velocity = closures.u_c if closures.u_c is not None else closures.beta * flow.u_inf
    if velocity <= 0.0:
        raise ValueError("Convection velocity must be positive.")
    return velocity


def angular_frequency(frequency_hz: float) -> float:
    if frequency_hz < 0.0:
        raise ValueError("frequency_hz must be non-negative.")
    return 2.0 * math.pi * frequency_hz


def frequency_shape(chi: float) -> float:
    if chi < 0.0:
        raise ValueError("chi must be non-negative.")
    if chi == 0.0:
        return 0.0
    return chi * chi / ((1.0 + chi * chi) ** (7.0 / 3.0))


def incidence_amplitude_factor(
    incidence_deg: float,
    closures: ClosureParams,
) -> float:
    alpha = math.radians(incidence_deg)
    alpha_ref = math.radians(closures.incidence_ref_deg)
    return 1.0 + closures.incidence_amplitude_coeff * (alpha - alpha_ref) ** 2


def incidence_delay_seconds(
    incidence_deg: float,
    closures: ClosureParams,
) -> float:
    alpha = math.radians(incidence_deg)
    alpha_ref = math.radians(closures.incidence_ref_deg)
    return closures.incidence_delay_per_rad * (alpha - alpha_ref)


def source_autospectra(
    grid: SourceGrid,
    frequency_hz: float,
    flow: FlowConfig,
    closures: ClosureParams,
) -> tuple[float, ...]:
    omega = angular_frequency(frequency_hz)
    u_c = convection_velocity(flow, closures)
    spectra: list[float] = []
    for chord, incidence_deg in zip(grid.chords, grid.incidence_deg):
        chi = omega * chord / u_c
        amplitude_factor = incidence_amplitude_factor(incidence_deg, closures)
        spectrum = (
            closures.cq
            * flow.rho0**2
            * flow.u_inf**5
            * chord**2
            * frequency_shape(chi)
            * amplitude_factor
        )
        spectra.append(max(spectrum, 0.0))
    return tuple(spectra)


def transfer_kernel(
    observer_direction: tuple[float, float, float],
    frequency_hz: float,
    flow: FlowConfig,
) -> float:
    """Calibrated Curle-like loading surrogate used in the first model."""

    omega = angular_frequency(frequency_hz)
    mach = flow.mach
    denominator = max((1.0 - mach * observer_direction[0]) ** 2, 1.0e-12)
    return (omega * omega / (flow.c0 * flow.c0)) / denominator


def transfer_weights(
    grid: SourceGrid,
    observer_direction: tuple[float, float, float],
    frequency_hz: float,
    flow: FlowConfig,
    include_quadrature: bool = True,
) -> tuple[complex, ...]:
    kernel = transfer_kernel(observer_direction, frequency_hz, flow)
    wave_number = angular_frequency(frequency_hz) / flow.c0
    weights: list[complex] = []
    for point, quadrature_weight, chord, loading_direction in zip(
        grid.points,
        grid.weights,
        grid.chords,
        grid.loading_directions,
    ):
        loading_projection = sum(
            loading_direction[index] * observer_direction[index] for index in range(3)
        )
        phase = wave_number * sum(point[index] * observer_direction[index] for index in range(3))
        magnitude = kernel * chord * loading_projection
        if include_quadrature:
            magnitude *= quadrature_weight
        weights.append(magnitude * complex(math.cos(phase), math.sin(phase)))
    return tuple(weights)


def coherence_value(
    point_m: tuple[float, float, float],
    point_n: tuple[float, float, float],
    incidence_m_deg: float,
    incidence_n_deg: float,
    frequency_hz: float,
    flow: FlowConfig,
    closures: ClosureParams,
) -> complex:
    model = closures.coherence_model.lower()
    if model == "zero":
        return 1.0 + 0.0j if point_m == point_n else 0.0 + 0.0j
    if model == "full":
        magnitude = 1.0
    elif model == "exponential":
        omega = angular_frequency(frequency_hz)
        u_c = convection_velocity(flow, closures)
        scales = (
            max(closures.coherence_x, 1.0e-12),
            max(closures.coherence_y, 1.0e-12),
            max(closures.coherence_z, 1.0e-12),
        )
        distance = sum(abs(point_m[i] - point_n[i]) / scales[i] for i in range(3))
        magnitude = math.exp(-omega * distance / u_c)
    else:
        raise ValueError(
            "coherence_model must be one of 'exponential', 'zero', or 'full'."
        )

    omega = angular_frequency(frequency_hz)
    delay_m = incidence_delay_seconds(incidence_m_deg, closures)
    delay_n = incidence_delay_seconds(incidence_n_deg, closures)
    phase = omega * (delay_m - delay_n)
    return magnitude * complex(math.cos(phase), math.sin(phase))


def source_cross_spectral_matrix(
    grid: SourceGrid,
    frequency_hz: float,
    flow: FlowConfig,
    closures: ClosureParams,
) -> tuple[tuple[complex, ...], ...]:
    autospectra = source_autospectra(grid, frequency_hz, flow, closures)
    rows: list[tuple[complex, ...]] = []
    for m in range(grid.n):
        row: list[complex] = []
        for n in range(grid.n):
            if m == n:
                gamma = 1.0 + 0.0j
            else:
                gamma = coherence_value(
                    grid.points[m],
                    grid.points[n],
                    grid.incidence_deg[m],
                    grid.incidence_deg[n],
                    frequency_hz,
                    flow,
                    closures,
                )
            row.append(math.sqrt(autospectra[m] * autospectra[n]) * gamma)
        rows.append(tuple(row))
    return tuple(rows)


def is_hermitian(
    matrix: tuple[tuple[complex, ...], ...],
    tolerance: float = 1.0e-9,
) -> bool:
    for i, row in enumerate(matrix):
        if len(row) != len(matrix):
            return False
        for j, value in enumerate(row):
            if abs(value - matrix[j][i].conjugate()) > tolerance:
                return False
    return True


def cholesky_psd_check(
    matrix: tuple[tuple[complex, ...], ...],
    tolerance: float = 1.0e-10,
) -> bool:
    """Return True when a Hermitian matrix is positive semidefinite.

    The implementation is a small dependency-free Cholesky check with tolerance
    for nearly zero pivots. It is intended for validation tests and modest
    source grids, not as a high-performance linear algebra backend.
    """

    n = len(matrix)
    if not is_hermitian(matrix, tolerance=max(tolerance * 10.0, 1.0e-9)):
        return False

    lower = [[0.0 + 0.0j for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            subtotal = sum(lower[i][k] * lower[j][k].conjugate() for k in range(j))
            value = matrix[i][j] - subtotal
            if i == j:
                real_value = value.real
                if real_value < -tolerance or abs(value.imag) > tolerance:
                    return False
                lower[i][j] = math.sqrt(max(real_value, 0.0)) + 0.0j
            else:
                pivot = lower[j][j]
                if abs(pivot) <= tolerance:
                    if abs(value) > tolerance:
                        return False
                    lower[i][j] = 0.0 + 0.0j
                else:
                    lower[i][j] = value / pivot
    return True
