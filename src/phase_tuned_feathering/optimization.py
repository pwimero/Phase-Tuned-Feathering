"""Staged optimization utilities for the model-first research workflow."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import random

from .acoustics import evaluate_spp
from .aero import screen_aero
from .closures import ClosureParams, FlowConfig
from .geometry import WingGeometryParams, default_geometry, source_grid
from .metrics import sector_spl
from .observers import ObserverGrid, Sector


@dataclass(frozen=True)
class OptimizationConfig:
    base_params: WingGeometryParams = field(default_factory=default_geometry)
    flow: FlowConfig = field(default_factory=FlowConfig)
    closures: ClosureParams = field(default_factory=ClosureParams)
    observers: ObserverGrid = field(default_factory=lambda: ObserverGrid.spherical(5, 8))
    frequencies_hz: tuple[float, ...] = (500.0, 1000.0, 2000.0)
    target_sector: Sector = field(default_factory=lambda: Sector((0.0, 0.0, 1.0), 70.0))
    suppressed_sector: Sector = field(default_factory=lambda: Sector((0.0, 0.0, -1.0), 70.0))
    iterations: int = 50
    seed: int = 11
    n_eta: int = 8
    incidence_bounds_deg: tuple[float, float] = (-10.0, 10.0)
    spacing_scale_bounds: tuple[float, float] = (1.0, 1.45)
    root_z_bounds: tuple[float, float] = (-0.18, 0.18)
    tip_sweep_bounds: tuple[float, float] = (-0.70, 0.35)
    tip_z_curve_bounds: tuple[float, float] = (-0.35, 0.65)
    min_tip_chord_scale_bounds: tuple[float, float] = (0.10, 0.35)
    robust_samples: int = 5


@dataclass(frozen=True)
class OptimizationResult:
    stage: int
    best_params: WingGeometryParams
    best_score: float
    history: tuple[dict[str, float], ...]


def _uniform(rng: random.Random, bounds: tuple[float, float]) -> float:
    return rng.uniform(bounds[0], bounds[1])


def _candidate_for_stage(
    stage: int,
    config: OptimizationConfig,
    rng: random.Random,
) -> WingGeometryParams:
    if stage < 1 or stage > 5:
        raise ValueError("stage must be between 1 and 5.")

    params = config.base_params.with_incidence_angles(
        tuple(
            _uniform(rng, config.incidence_bounds_deg)
            for _ in range(config.base_params.total_wings)
        )
    )
    if stage >= 2:
        params = replace(
            params,
            y_spacing_scale=_uniform(rng, config.spacing_scale_bounds),
        )
    if stage >= 3:
        params = replace(
            params,
            wing_1_root_z_translation=_uniform(rng, config.root_z_bounds),
            wing_7_root_z_translation=_uniform(rng, config.root_z_bounds),
            wing_1_tip_sweep=_uniform(rng, config.tip_sweep_bounds),
            wing_7_tip_sweep=_uniform(rng, config.tip_sweep_bounds),
        )
    if stage >= 4:
        params = replace(
            params,
            wing_1_tip_z_curve=_uniform(rng, config.tip_z_curve_bounds),
            wing_7_tip_z_curve=_uniform(rng, config.tip_z_curve_bounds),
            min_tip_chord_scale=_uniform(rng, config.min_tip_chord_scale_bounds),
        )
    return params


def _regularization(params: WingGeometryParams) -> float:
    incidence = params.incidence_angles_deg()
    alpha_penalty = sum(
        (incidence[index + 1] - incidence[index]) ** 2
        for index in range(len(incidence) - 1)
    )
    return 0.01 * alpha_penalty


def _closure_samples(
    closures: ClosureParams,
    sample_count: int,
    rng: random.Random,
) -> tuple[ClosureParams, ...]:
    samples: list[ClosureParams] = []
    for _ in range(max(sample_count, 1)):
        samples.append(
            replace(
                closures,
                coherence_x=closures.coherence_x * rng.uniform(0.75, 1.25),
                coherence_y=closures.coherence_y * rng.uniform(0.75, 1.25),
                coherence_z=closures.coherence_z * rng.uniform(0.75, 1.25),
                incidence_delay_per_rad=closures.incidence_delay_per_rad
                * rng.uniform(0.5, 1.5),
            )
        )
    return tuple(samples)


def _score_once(
    params: WingGeometryParams,
    config: OptimizationConfig,
    closures: ClosureParams,
) -> float:
    grid = source_grid(params, n_eta=config.n_eta)
    result = evaluate_spp(
        grid,
        config.observers,
        config.frequencies_hz,
        config.flow,
        closures,
    )
    target = sector_spl(result, config.target_sector, config.flow)
    suppressed = sector_spl(result, config.suppressed_sector, config.flow)
    return target - suppressed


def _score_candidate(
    stage: int,
    params: WingGeometryParams,
    config: OptimizationConfig,
    rng: random.Random,
) -> tuple[float, dict[str, float]]:
    aero = screen_aero(params, config.flow)
    aero_penalty = 10.0 * len(aero.issues)
    regularization = _regularization(params)

    if stage >= 5:
        closure_samples = _closure_samples(config.closures, config.robust_samples, rng)
        scores = tuple(_score_once(params, config, sample) for sample in closure_samples)
        mean = sum(scores) / len(scores)
        variance = sum((score - mean) ** 2 for score in scores) / len(scores)
        robustness_penalty = variance ** 0.5
        raw_score = mean - robustness_penalty
    else:
        raw_score = _score_once(params, config, config.closures)
        robustness_penalty = 0.0

    score = raw_score - regularization - aero_penalty
    return score, {
        "score": score,
        "raw_acoustic_score": raw_score,
        "regularization": regularization,
        "aero_penalty": aero_penalty,
        "robustness_penalty": robustness_penalty,
    }


def optimize_stage(
    stage: int,
    config: OptimizationConfig | None = None,
) -> OptimizationResult:
    config = OptimizationConfig() if config is None else config
    if config.iterations <= 0:
        raise ValueError("iterations must be positive.")
    rng = random.Random(config.seed + stage * 1000)

    best_params = config.base_params
    best_score = float("-inf")
    history: list[dict[str, float]] = []

    for iteration in range(config.iterations):
        params = _candidate_for_stage(stage, config, rng)
        score, values = _score_candidate(stage, params, config, rng)
        values["iteration"] = float(iteration)
        history.append(values)
        if score > best_score:
            best_score = score
            best_params = params

    return OptimizationResult(
        stage=stage,
        best_params=best_params,
        best_score=best_score,
        history=tuple(history),
    )
