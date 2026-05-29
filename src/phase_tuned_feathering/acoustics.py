"""Acoustic model hierarchy and PSD-safe pressure spectra."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .closures import (
    ClosureParams,
    FlowConfig,
    source_autospectra,
    source_cross_spectral_matrix,
    transfer_weights,
)
from .geometry import SourceGrid
from .observers import ObserverGrid


@dataclass(frozen=True)
class SpectralResult:
    frequencies_hz: tuple[float, ...]
    observers: ObserverGrid
    spp: tuple[tuple[float, ...], ...]

    def value(self, frequency_index: int, observer_index: int) -> float:
        return self.spp[frequency_index][observer_index]


def _radius_factor(flow: FlowConfig) -> float:
    radius = max(flow.observer_radius, 1.0e-12)
    return 1.0 / ((4.0 * math.pi * radius) ** 2)


def _quadratic_form(
    weights: tuple[complex, ...],
    matrix: tuple[tuple[complex, ...], ...],
) -> float:
    total = 0.0 + 0.0j
    for m, weight_m in enumerate(weights):
        for n, weight_n in enumerate(weights):
            total += weight_m.conjugate() * matrix[m][n] * weight_n
    return max(total.real, 0.0)


def evaluate_spp(
    grid: SourceGrid,
    observers: ObserverGrid,
    frequencies_hz: tuple[float, ...] | list[float],
    flow: FlowConfig | None = None,
    closures: ClosureParams | None = None,
) -> SpectralResult:
    """Evaluate Level 3 stochastic PSD using ``Spp = a^H Cq a``."""

    flow = FlowConfig() if flow is None else flow
    closures = ClosureParams() if closures is None else closures
    factor = _radius_factor(flow)
    frequency_tuple = tuple(float(frequency) for frequency in frequencies_hz)

    result_rows: list[tuple[float, ...]] = []
    for frequency_hz in frequency_tuple:
        cq = source_cross_spectral_matrix(grid, frequency_hz, flow, closures)
        observer_values: list[float] = []
        for direction in observers.directions:
            weights = transfer_weights(grid, direction, frequency_hz, flow)
            observer_values.append(factor * _quadratic_form(weights, cq))
        result_rows.append(tuple(observer_values))
    return SpectralResult(frequency_tuple, observers, tuple(result_rows))


def evaluate_spp_reference(
    grid: SourceGrid,
    observers: ObserverGrid,
    frequencies_hz: tuple[float, ...] | list[float],
    flow: FlowConfig | None = None,
    closures: ClosureParams | None = None,
) -> SpectralResult:
    """Slow explicit double-loop implementation for regression checks."""

    flow = FlowConfig() if flow is None else flow
    closures = ClosureParams() if closures is None else closures
    factor = _radius_factor(flow)
    frequency_tuple = tuple(float(frequency) for frequency in frequencies_hz)
    rows: list[tuple[float, ...]] = []
    for frequency_hz in frequency_tuple:
        cq = source_cross_spectral_matrix(grid, frequency_hz, flow, closures)
        observer_values: list[float] = []
        for direction in observers.directions:
            weights = transfer_weights(grid, direction, frequency_hz, flow)
            total = 0.0 + 0.0j
            for m in range(grid.n):
                for n in range(grid.n):
                    total += weights[m].conjugate() * cq[m][n] * weights[n]
            observer_values.append(factor * max(total.real, 0.0))
        rows.append(tuple(observer_values))
    return SpectralResult(frequency_tuple, observers, tuple(rows))


def level1_compact_spp(
    grid: SourceGrid,
    observers: ObserverGrid,
    frequency_hz: float,
    flow: FlowConfig | None = None,
    closures: ClosureParams | None = None,
) -> tuple[float, ...]:
    """Compact coherent feather array using one averaged source per feather."""

    flow = FlowConfig() if flow is None else flow
    closures = ClosureParams() if closures is None else closures
    compact_points: dict[int, list[int]] = {}
    for index, feather_id in enumerate(grid.feather_ids):
        compact_points.setdefault(feather_id, []).append(index)

    amplitudes = source_autospectra(grid, frequency_hz, flow, closures)
    values: list[float] = []
    factor = _radius_factor(flow)
    for direction in observers.directions:
        full_weights = transfer_weights(grid, direction, frequency_hz, flow)
        coherent_sum = 0.0 + 0.0j
        for indices in compact_points.values():
            averaged_weight = sum(full_weights[index] for index in indices) / len(indices)
            averaged_amplitude = math.sqrt(
                sum(amplitudes[index] for index in indices) / len(indices)
            )
            coherent_sum += averaged_weight * averaged_amplitude
        values.append(factor * abs(coherent_sum) ** 2)
    return tuple(values)


def level2_deterministic_pressure(
    grid: SourceGrid,
    observers: ObserverGrid,
    frequency_hz: float,
    flow: FlowConfig | None = None,
    closures: ClosureParams | None = None,
) -> tuple[complex, ...]:
    """Deterministic distributed-source pressure amplitude."""

    flow = FlowConfig() if flow is None else flow
    closures = ClosureParams() if closures is None else closures
    amplitudes = source_autospectra(grid, frequency_hz, flow, closures)
    radius = max(flow.observer_radius, 1.0e-12)
    factor = 1.0 / (4.0 * math.pi * radius)
    pressures: list[complex] = []
    for direction in observers.directions:
        weights = transfer_weights(grid, direction, frequency_hz, flow)
        pressure = sum(
            weights[index] * math.sqrt(amplitudes[index]) for index in range(grid.n)
        )
        pressures.append(factor * pressure)
    return tuple(pressures)
