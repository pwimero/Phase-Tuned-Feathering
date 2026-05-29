"""Acoustic metrics used by the manuscript and optimization loop."""

from __future__ import annotations

import math

from .acoustics import SpectralResult
from .closures import FlowConfig
from .observers import Sector


def _select_frequency_indices(
    frequencies_hz: tuple[float, ...],
    f_min: float | None,
    f_max: float | None,
) -> tuple[int, ...]:
    selected = tuple(
        index
        for index, frequency in enumerate(frequencies_hz)
        if (f_min is None or frequency >= f_min) and (f_max is None or frequency <= f_max)
    )
    if not selected:
        raise ValueError("No frequencies fall inside the requested band.")
    return selected


def _trapz(xs: tuple[float, ...], ys: tuple[float, ...]) -> float:
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length.")
    if len(xs) == 1:
        return ys[0]
    total = 0.0
    for left in range(len(xs) - 1):
        width = xs[left + 1] - xs[left]
        if width < 0.0:
            raise ValueError("xs must be sorted.")
        total += 0.5 * width * (ys[left] + ys[left + 1])
    return total


def band_spl(
    result: SpectralResult,
    flow: FlowConfig | None = None,
    f_min: float | None = None,
    f_max: float | None = None,
) -> tuple[float, ...]:
    flow = FlowConfig() if flow is None else flow
    selected = _select_frequency_indices(result.frequencies_hz, f_min, f_max)
    frequencies = tuple(result.frequencies_hz[index] for index in selected)
    values: list[float] = []
    for observer_index in range(len(result.observers.directions)):
        spectrum = tuple(result.spp[index][observer_index] for index in selected)
        pressure_variance = max(_trapz(frequencies, spectrum), 1.0e-300)
        values.append(10.0 * math.log10(pressure_variance / (flow.p_ref * flow.p_ref)))
    return tuple(values)


def directivity(result: SpectralResult) -> SpectralResult:
    rows: list[tuple[float, ...]] = []
    total_weight = sum(result.observers.weights)
    if total_weight <= 0.0:
        raise ValueError("Observer weights must sum to a positive value.")
    for row in result.spp:
        mean = sum(value * weight for value, weight in zip(row, result.observers.weights))
        mean /= total_weight
        if mean <= 0.0:
            rows.append(tuple(0.0 for _ in row))
        else:
            rows.append(tuple(value / mean for value in row))
    return SpectralResult(result.frequencies_hz, result.observers, tuple(rows))


def sector_spl(
    result: SpectralResult,
    sector: Sector,
    flow: FlowConfig | None = None,
    f_min: float | None = None,
    f_max: float | None = None,
) -> float:
    flow = FlowConfig() if flow is None else flow
    selected = _select_frequency_indices(result.frequencies_hz, f_min, f_max)
    frequencies = tuple(result.frequencies_hz[index] for index in selected)
    sector_weights = sector.observer_weights(result.observers)
    weight_sum = sum(sector_weights)
    if weight_sum <= 0.0:
        raise ValueError("Sector does not contain any observer weight.")

    band_values: list[float] = []
    for frequency_index in selected:
        angular_average = sum(
            result.spp[frequency_index][observer_index] * sector_weights[observer_index]
            for observer_index in range(len(sector_weights))
        )
        band_values.append(angular_average / weight_sum)
    pressure_variance = max(_trapz(frequencies, tuple(band_values)), 1.0e-300)
    return 10.0 * math.log10(pressure_variance / (flow.p_ref * flow.p_ref))


def total_acoustic_proxy(
    result: SpectralResult,
    f_min: float | None = None,
    f_max: float | None = None,
) -> float:
    selected = _select_frequency_indices(result.frequencies_hz, f_min, f_max)
    frequencies = tuple(result.frequencies_hz[index] for index in selected)
    total_weight = sum(result.observers.weights)
    if total_weight <= 0.0:
        raise ValueError("Observer weights must sum to a positive value.")
    angular_values: list[float] = []
    for frequency_index in selected:
        angular_values.append(
            sum(
                result.spp[frequency_index][observer_index]
                * result.observers.weights[observer_index]
                for observer_index in range(len(result.observers.weights))
            )
            / total_weight
        )
    return _trapz(frequencies, tuple(angular_values))
