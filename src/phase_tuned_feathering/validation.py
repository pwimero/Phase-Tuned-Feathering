"""Theory-versus-simulation validation utilities.

The validation layer compares the semi-analytical acoustic model against
external simulated performance data. It intentionally keeps a simple CSV schema
so results from AeroSandbox wrappers, panel-method scripts, CFD-lite studies,
or later high-fidelity CFD can be ingested without changing the model code.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable

from .acoustics import SpectralResult, evaluate_spp
from .closures import ClosureParams, FlowConfig
from .geometry import WingGeometryParams, default_geometry, source_grid
from .observers import ObserverGrid, unit_direction


REQUIRED_SIMULATION_COLUMNS = (
    "frequency_hz",
    "observer_x",
    "observer_y",
    "observer_z",
    "spp",
)


def _direction_key(
    direction: tuple[float, float, float],
    digits: int = 12,
) -> tuple[float, float, float]:
    unit = unit_direction(direction)
    return tuple(round(component, digits) for component in unit)  # type: ignore[return-value]


@dataclass(frozen=True)
class SimulationRecord:
    frequency_hz: float
    direction: tuple[float, float, float]
    spp: float
    case_id: str = "default"
    split: str = "validation"
    weight: float = 1.0


@dataclass(frozen=True)
class SimulationDataset:
    records: tuple[SimulationRecord, ...]

    def __post_init__(self) -> None:
        if not self.records:
            raise ValueError("SimulationDataset cannot be empty.")

    @property
    def frequencies_hz(self) -> tuple[float, ...]:
        return tuple(sorted({record.frequency_hz for record in self.records}))

    @property
    def directions(self) -> tuple[tuple[float, float, float], ...]:
        seen: set[tuple[float, float, float]] = set()
        directions: list[tuple[float, float, float]] = []
        for record in self.records:
            key = _direction_key(record.direction)
            if key not in seen:
                seen.add(key)
                directions.append(record.direction)
        return tuple(directions)

    def observer_grid(self) -> ObserverGrid:
        direction_weights: dict[tuple[float, float, float], list[float]] = {}
        for record in self.records:
            direction_weights.setdefault(record.direction, []).append(record.weight)
        directions = tuple(direction_weights)
        weights = tuple(
            sum(values) / len(values) for values in direction_weights.values()
        )
        return ObserverGrid(directions, weights)


@dataclass(frozen=True)
class ComparisonRow:
    case_id: str
    split: str
    frequency_hz: float
    direction: tuple[float, float, float]
    simulated_spp: float
    theory_spp: float
    simulated_level_db: float
    theory_level_db: float
    error_db: float
    relative_error: float
    weight: float


@dataclass(frozen=True)
class ComparisonResult:
    rows: tuple[ComparisonRow, ...]
    summary: dict[str, dict[str, float]]


def spectral_level_db(spp: float, flow: FlowConfig) -> float:
    return 10.0 * math.log10(max(spp, 1.0e-300) / (flow.p_ref * flow.p_ref))


def load_simulation_csv(path: str | Path) -> SimulationDataset:
    input_path = Path(path)
    with input_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{input_path} has no header row.")
        missing = [
            column for column in REQUIRED_SIMULATION_COLUMNS if column not in reader.fieldnames
        ]
        if missing:
            raise ValueError(
                f"{input_path} is missing required columns: {', '.join(missing)}"
            )
        records = []
        for row_number, row in enumerate(reader, start=2):
            try:
                direction = unit_direction(
                    (
                        float(row["observer_x"]),
                        float(row["observer_y"]),
                        float(row["observer_z"]),
                    )
                )
                spp = float(row["spp"])
                if spp < 0.0:
                    raise ValueError("spp must be non-negative")
                records.append(
                    SimulationRecord(
                        frequency_hz=float(row["frequency_hz"]),
                        direction=direction,
                        spp=spp,
                        case_id=row.get("case_id") or "default",
                        split=(row.get("split") or "validation").lower(),
                        weight=float(row.get("weight") or 1.0),
                    )
                )
            except Exception as exc:
                raise ValueError(f"Invalid simulation row {row_number}: {exc}") from exc
    return SimulationDataset(tuple(records))


def write_simulation_csv(
    path: str | Path,
    dataset: SimulationDataset,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case_id",
                "split",
                "frequency_hz",
                "observer_x",
                "observer_y",
                "observer_z",
                "spp",
                "weight",
            ]
        )
        for record in dataset.records:
            writer.writerow(
                [
                    record.case_id,
                    record.split,
                    record.frequency_hz,
                    record.direction[0],
                    record.direction[1],
                    record.direction[2],
                    record.spp,
                    record.weight,
                ]
            )
    return output_path


def _prediction_lookup(
    result: SpectralResult,
) -> dict[tuple[float, tuple[float, float, float]], float]:
    lookup: dict[tuple[float, tuple[float, float, float]], float] = {}
    for frequency_index, frequency_hz in enumerate(result.frequencies_hz):
        for direction_index, direction in enumerate(result.observers.directions):
            lookup[(frequency_hz, _direction_key(direction))] = (
                result.spp[frequency_index][direction_index]
            )
    return lookup


def compare_spectral_results(
    theory: SpectralResult,
    simulation: SimulationDataset,
    flow: FlowConfig | None = None,
) -> ComparisonResult:
    flow = FlowConfig() if flow is None else flow
    lookup = _prediction_lookup(theory)
    rows: list[ComparisonRow] = []
    for record in simulation.records:
        key = (record.frequency_hz, _direction_key(record.direction))
        if key not in lookup:
            raise ValueError(
                "Theory result does not contain simulation point "
                f"frequency={record.frequency_hz}, direction={record.direction}."
            )
        theory_spp = lookup[key]
        simulated_level = spectral_level_db(record.spp, flow)
        theory_level = spectral_level_db(theory_spp, flow)
        rows.append(
            ComparisonRow(
                case_id=record.case_id,
                split=record.split,
                frequency_hz=record.frequency_hz,
                direction=record.direction,
                simulated_spp=record.spp,
                theory_spp=theory_spp,
                simulated_level_db=simulated_level,
                theory_level_db=theory_level,
                error_db=theory_level - simulated_level,
                relative_error=(theory_spp - record.spp) / max(record.spp, 1.0e-300),
                weight=record.weight,
            )
        )
    return ComparisonResult(tuple(rows), summarize_comparison(rows))


def compare_theory_to_simulation(
    simulation: SimulationDataset,
    params: WingGeometryParams | None = None,
    flow: FlowConfig | None = None,
    closures: ClosureParams | None = None,
    n_eta: int = 32,
    source_chord_fraction: float = 1.0,
) -> ComparisonResult:
    params = default_geometry() if params is None else params
    flow = FlowConfig() if flow is None else flow
    closures = ClosureParams() if closures is None else closures
    grid = source_grid(
        params,
        n_eta=n_eta,
        source_chord_fraction=source_chord_fraction,
    )
    theory = evaluate_spp(
        grid,
        simulation.observer_grid(),
        simulation.frequencies_hz,
        flow,
        closures,
    )
    return compare_spectral_results(theory, simulation, flow)


def _weighted_mean(values: Iterable[float], weights: Iterable[float]) -> float:
    pairs = tuple(zip(values, weights))
    total_weight = sum(weight for _, weight in pairs)
    if total_weight <= 0.0:
        raise ValueError("Weights must sum to a positive value.")
    return sum(value * weight for value, weight in pairs) / total_weight


def _angle_deg(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> float:
    dot = max(min(sum(left[index] * right[index] for index in range(3)), 1.0), -1.0)
    return math.degrees(math.acos(dot))


def _lobe_angle_errors(rows: tuple[ComparisonRow, ...]) -> tuple[float, ...]:
    errors: list[float] = []
    split_frequency_pairs = sorted({(row.split, row.frequency_hz) for row in rows})
    for split, frequency_hz in split_frequency_pairs:
        subset = [
            row
            for row in rows
            if row.split == split and row.frequency_hz == frequency_hz
        ]
        if len(subset) < 2:
            continue
        theory_peak = max(subset, key=lambda row: row.theory_spp)
        simulated_peak = max(subset, key=lambda row: row.simulated_spp)
        errors.append(_angle_deg(theory_peak.direction, simulated_peak.direction))
    return tuple(errors)


def summarize_comparison(rows: tuple[ComparisonRow, ...] | list[ComparisonRow]) -> dict[str, dict[str, float]]:
    row_tuple = tuple(rows)
    if not row_tuple:
        raise ValueError("Cannot summarize an empty comparison.")

    splits = ("all",) + tuple(sorted({row.split for row in row_tuple}))
    summary: dict[str, dict[str, float]] = {}
    for split in splits:
        subset = row_tuple if split == "all" else tuple(row for row in row_tuple if row.split == split)
        if not subset:
            continue
        weights = tuple(row.weight for row in subset)
        error_db = tuple(row.error_db for row in subset)
        abs_error_db = tuple(abs(row.error_db) for row in subset)
        rel_error = tuple(row.relative_error for row in subset)
        theory_db = tuple(row.theory_level_db for row in subset)
        simulated_db = tuple(row.simulated_level_db for row in subset)
        bias = _weighted_mean(error_db, weights)
        mae = _weighted_mean(abs_error_db, weights)
        rmse = math.sqrt(_weighted_mean(tuple(value * value for value in error_db), weights))
        rel_rmse = math.sqrt(_weighted_mean(tuple(value * value for value in rel_error), weights))
        lobe_errors = _lobe_angle_errors(subset)
        split_summary = {
            "count": float(len(subset)),
            "bias_db": bias,
            "mae_db": mae,
            "rmse_db": rmse,
            "max_abs_error_db": max(abs_error_db),
            "relative_rmse": rel_rmse,
            "mean_theory_level_db": _weighted_mean(theory_db, weights),
            "mean_simulated_level_db": _weighted_mean(simulated_db, weights),
        }
        if lobe_errors:
            split_summary["mean_lobe_angle_error_deg"] = sum(lobe_errors) / len(lobe_errors)
            split_summary["max_lobe_angle_error_deg"] = max(lobe_errors)
        summary[split] = split_summary
    return summary


def write_comparison_csv(
    path: str | Path,
    result: ComparisonResult,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case_id",
                "split",
                "frequency_hz",
                "observer_x",
                "observer_y",
                "observer_z",
                "simulated_spp",
                "theory_spp",
                "simulated_level_db",
                "theory_level_db",
                "error_db",
                "relative_error",
                "weight",
            ]
        )
        for row in result.rows:
            writer.writerow(
                [
                    row.case_id,
                    row.split,
                    row.frequency_hz,
                    row.direction[0],
                    row.direction[1],
                    row.direction[2],
                    row.simulated_spp,
                    row.theory_spp,
                    row.simulated_level_db,
                    row.theory_level_db,
                    row.error_db,
                    row.relative_error,
                    row.weight,
                ]
            )
    return output_path


def write_summary_json(
    path: str | Path,
    result: ComparisonResult,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def generate_synthetic_simulation_dataset(
    params: WingGeometryParams | None = None,
    flow: FlowConfig | None = None,
    closures: ClosureParams | None = None,
    observers: ObserverGrid | None = None,
    frequencies_hz: tuple[float, ...] = (500.0, 1000.0, 2000.0),
    n_eta: int = 16,
    seed: int = 7,
    case_id: str = "synthetic_sim",
) -> SimulationDataset:
    """Generate deterministic comparison data for pipeline smoke tests.

    This is not validation evidence. It exists so the validation mechanics can
    be run before external simulation data are available.
    """

    params = default_geometry() if params is None else params
    flow = FlowConfig() if flow is None else flow
    closures = ClosureParams() if closures is None else closures
    observers = ObserverGrid.spherical(5, 8) if observers is None else observers
    theory = evaluate_spp(
        source_grid(params, n_eta=n_eta),
        observers,
        frequencies_hz,
        flow,
        closures,
    )
    rng = random.Random(seed)
    records: list[SimulationRecord] = []
    for frequency_index, frequency_hz in enumerate(theory.frequencies_hz):
        split = "calibration" if frequency_index == 0 else "validation"
        for direction_index, direction in enumerate(theory.observers.directions):
            directional_bias = 0.08 * direction[0] - 0.05 * direction[2]
            frequency_bias = 0.04 * frequency_index
            noise = rng.uniform(-0.03, 0.03)
            scale = max(0.2, 1.0 + directional_bias + frequency_bias + noise)
            records.append(
                SimulationRecord(
                    frequency_hz=frequency_hz,
                    direction=direction,
                    spp=theory.spp[frequency_index][direction_index] * scale,
                    case_id=case_id,
                    split=split,
                    weight=theory.observers.weights[direction_index],
                )
            )
    return SimulationDataset(tuple(records))


def write_synthetic_simulation_csv(
    path: str | Path,
    params: WingGeometryParams | None = None,
    flow: FlowConfig | None = None,
    closures: ClosureParams | None = None,
    observers: ObserverGrid | None = None,
    frequencies_hz: tuple[float, ...] = (500.0, 1000.0, 2000.0),
    n_eta: int = 16,
    seed: int = 7,
) -> Path:
    dataset = generate_synthetic_simulation_dataset(
        params=params,
        flow=flow,
        closures=closures,
        observers=observers,
        frequencies_hz=frequencies_hz,
        n_eta=n_eta,
        seed=seed,
    )
    return write_simulation_csv(path, dataset)


def comparison_summary_text(summary: dict[str, dict[str, float]]) -> str:
    lines = []
    for split, values in summary.items():
        lines.append(
            "{split}: count={count:.0f}, bias={bias:.3f} dB, "
            "MAE={mae:.3f} dB, RMSE={rmse:.3f} dB, max={max_err:.3f} dB".format(
                split=split,
                count=values["count"],
                bias=values["bias_db"],
                mae=values["mae_db"],
                rmse=values["rmse_db"],
                max_err=values["max_abs_error_db"],
            )
        )
    return "\n".join(lines)
