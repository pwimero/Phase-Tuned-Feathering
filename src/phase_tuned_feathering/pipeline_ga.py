"""Genetic Algorithm Optimization Pipeline.

This pipeline runs a Genetic Algorithm to find the optimal feather geometry
for an arbitrary acoustic directivity target. It then automatically feeds the
theoretically-optimized wing into the empirical Surrogate Simulator to prove
that the design successfully steers sound under real-world aerodynamic physics.

The target directivity is generated as a random mathematical shape on every
run, proving that feather geometry parameterization is a universal control
mechanism — not tuned to any specific scenario.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
import json

from .closures import ClosureParams, FlowConfig
from .acoustics import evaluate_spp
from .ga_optimization import (
    FREEDOM_LEVEL_LABELS,
    FREEDOM_LEVEL_ORDER,
    GAConfig,
    OptimizationRuntime,
    active_parameter_count,
    effective_closures,
    effective_population_size,
    run_optimization_torch,
)
from .aggregate_campaign import aggregate_campaign_dir
from .geometry import default_geometry, source_grid
from .io import write_geometry_metadata_json, write_source_grid_csv
from .observers import ObserverGrid
from .simulation import (
    SurrogateNoiseConfig,
    local_aero_states,
    write_surrogate_simulation_csv,
)
from .validation import (
    compare_theory_to_simulation,
    comparison_summary_text,
    load_simulation_csv,
    spectral_level_db,
    write_comparison_csv,
    write_summary_json,
)


def generate_arbitrary_target(
    observers: ObserverGrid,
    seed: int | None = None,
) -> tuple[tuple[float, ...], dict]:
    """Generate a random arbitrary directivity target shape.

    Creates a mathematical target pattern by combining random spherical
    harmonic-like lobes. The result is a tuple of relative dB values
    (peak normalized to 0 dB) across all observer directions.

    Returns the target pattern and a metadata dict describing how it was built.
    """
    rng = random.Random(seed)

    # Pick 1–3 random lobes to combine
    n_lobes = rng.randint(1, 3)
    lobes = []
    for _ in range(n_lobes):
        # Random unit-vector direction for this lobe
        theta = math.acos(2.0 * rng.random() - 1.0)
        phi = 2.0 * math.pi * rng.random()
        cx = math.sin(theta) * math.cos(phi)
        cy = math.sin(theta) * math.sin(phi)
        cz = math.cos(theta)
        # Random concentration (higher = tighter lobe)
        concentration = rng.uniform(1.0, 4.0)
        # Random amplitude weight
        amplitude = rng.uniform(0.3, 1.0)
        lobes.append({
            "center": (cx, cy, cz),
            "concentration": concentration,
            "amplitude": amplitude,
        })

    # Evaluate raw pattern at each observer
    raw = []
    for direction in observers.directions:
        value = 0.0
        for lobe in lobes:
            cx, cy, cz = lobe["center"]
            dot = direction[0] * cx + direction[1] * cy + direction[2] * cz
            # Clamp to [-1, 1]
            dot = max(-1.0, min(1.0, dot))
            value += lobe["amplitude"] * (0.5 * (1.0 + dot)) ** lobe["concentration"]
        raw.append(value)

    # Convert to dB-like scale and normalize peak to 0 dB
    max_raw = max(raw)
    if max_raw <= 0.0:
        max_raw = 1.0
    pattern_db = []
    for v in raw:
        ratio = max(v / max_raw, 1e-6)
        pattern_db.append(10.0 * math.log10(ratio))

    metadata = {
        "seed": seed,
        "n_lobes": n_lobes,
        "lobes": [
            {
                "center": list(l["center"]),
                "concentration": l["concentration"],
                "amplitude": l["amplitude"],
            }
            for l in lobes
        ],
    }

    return tuple(pattern_db), metadata


def _target_fn_from_pattern(
    observers: ObserverGrid,
    target_pattern_db: tuple[float, ...],
):
    """Build a callable that maps an observer direction to its target dB value.

    Uses nearest-neighbour lookup on the observer grid.
    """
    direction_to_db = {}
    for direction, db_val in zip(observers.directions, target_pattern_db):
        direction_to_db[direction] = db_val

    def _lookup(direction: tuple[float, float, float]) -> float:
        if direction in direction_to_db:
            return direction_to_db[direction]
        # Nearest-neighbour fallback
        best_dot = -2.0
        best_db = 0.0
        for obs_dir, db_val in direction_to_db.items():
            dot = sum(a * b for a, b in zip(direction, obs_dir))
            if dot > best_dot:
                best_dot = dot
                best_db = db_val
        return best_db

    return _lookup


def _shape_fit_metrics(rows, split: str = "validation") -> dict[str, float]:
    subset = [row for row in rows if row.split == split and row.target_level_db is not None]
    if not subset:
        subset = [row for row in rows if row.target_level_db is not None]
    if not subset:
        return {}

    frequencies = sorted({row.frequency_hz for row in subset})
    surrogate_rmses: list[float] = []
    theory_rmses: list[float] = []
    surrogate_maes: list[float] = []
    theory_maes: list[float] = []
    surrogate_peak_angle_errors: list[float] = []
    theory_peak_angle_errors: list[float] = []

    for frequency_hz in frequencies:
        freq_rows = [row for row in subset if row.frequency_hz == frequency_hz]
        if len(freq_rows) < 2:
            continue
        target_levels = [float(row.target_level_db) for row in freq_rows if row.target_level_db is not None]
        if not target_levels:
            continue
        target_peak = max(target_levels)
        sim_peak = max(row.simulated_level_db for row in freq_rows)
        theory_peak = max(row.theory_level_db for row in freq_rows)
        sim_errors = [
            (row.simulated_level_db - sim_peak) - (float(row.target_level_db) - target_peak)
            for row in freq_rows
        ]
        theory_errors = [
            (row.theory_level_db - theory_peak) - (float(row.target_level_db) - target_peak)
            for row in freq_rows
        ]
        surrogate_rmses.append(math.sqrt(sum(value * value for value in sim_errors) / len(sim_errors)))
        theory_rmses.append(math.sqrt(sum(value * value for value in theory_errors) / len(theory_errors)))
        surrogate_maes.append(sum(abs(value) for value in sim_errors) / len(sim_errors))
        theory_maes.append(sum(abs(value) for value in theory_errors) / len(theory_errors))

        target_peak_row = max(freq_rows, key=lambda row: float(row.target_level_db))
        sim_peak_row = max(freq_rows, key=lambda row: row.simulated_level_db)
        theory_peak_row = max(freq_rows, key=lambda row: row.theory_level_db)
        surrogate_peak_angle_errors.append(
            math.degrees(
                math.acos(
                    max(
                        min(
                            sum(
                                target_peak_row.direction[index] * sim_peak_row.direction[index]
                                for index in range(3)
                            ),
                            1.0,
                        ),
                        -1.0,
                    )
                )
            )
        )
        theory_peak_angle_errors.append(
            math.degrees(
                math.acos(
                    max(
                        min(
                            sum(
                                target_peak_row.direction[index] * theory_peak_row.direction[index]
                                for index in range(3)
                            ),
                            1.0,
                        ),
                        -1.0,
                    )
                )
            )
        )

    return {
        "surrogate_target_rmse_db": sum(surrogate_rmses) / len(surrogate_rmses),
        "surrogate_target_mae_db": sum(surrogate_maes) / len(surrogate_maes),
        "theory_target_rmse_db": sum(theory_rmses) / len(theory_rmses),
        "theory_target_mae_db": sum(theory_maes) / len(theory_maes),
        "surrogate_peak_angle_error_deg": sum(surrogate_peak_angle_errors) / len(surrogate_peak_angle_errors),
        "theory_peak_angle_error_deg": sum(theory_peak_angle_errors) / len(theory_peak_angle_errors),
    }


def _geometry_mutation_metrics(base_params, optimized_params) -> dict[str, float]:
    base_inc = base_params.incidence_angles_deg()
    opt_inc = optimized_params.incidence_angles_deg()
    inc_rms = math.sqrt(
        sum((opt_inc[index] - base_inc[index]) ** 2 for index in range(len(base_inc))) / len(base_inc)
    )
    root_z_rms = math.sqrt(
        (
            (optimized_params.wing_1_root_z_translation - base_params.wing_1_root_z_translation) ** 2
            + (optimized_params.wing_7_root_z_translation - base_params.wing_7_root_z_translation) ** 2
            + (
                (
                    (optimized_params.mid_wing_root_z_translation if optimized_params.mid_wing_root_z_translation is not None else 0.5 * (optimized_params.wing_1_root_z_translation + optimized_params.wing_7_root_z_translation))
                    - 0.5 * (base_params.wing_1_root_z_translation + base_params.wing_7_root_z_translation)
                ) ** 2
            )
        ) / 3.0
    )
    sweep_rms = math.sqrt(
        (
            (optimized_params.wing_1_tip_sweep - base_params.wing_1_tip_sweep) ** 2
            + (optimized_params.wing_7_tip_sweep - base_params.wing_7_tip_sweep) ** 2
            + (
                (
                    (optimized_params.mid_wing_tip_sweep if optimized_params.mid_wing_tip_sweep is not None else 0.5 * (optimized_params.wing_1_tip_sweep + optimized_params.wing_7_tip_sweep))
                    - 0.5 * (base_params.wing_1_tip_sweep + base_params.wing_7_tip_sweep)
                ) ** 2
            )
        ) / 3.0
    )
    tip_z_rms = math.sqrt(
        (
            (optimized_params.wing_1_tip_z_curve - base_params.wing_1_tip_z_curve) ** 2
            + (optimized_params.wing_7_tip_z_curve - base_params.wing_7_tip_z_curve) ** 2
            + (
                (
                    (optimized_params.mid_wing_tip_z_curve if optimized_params.mid_wing_tip_z_curve is not None else 0.5 * (optimized_params.wing_1_tip_z_curve + optimized_params.wing_7_tip_z_curve))
                    - 0.5 * (base_params.wing_1_tip_z_curve + base_params.wing_7_tip_z_curve)
                ) ** 2
            )
        ) / 3.0
    )
    return {
        "incidence_rms_change_deg": inc_rms,
        "root_z_rms_change_m": root_z_rms,
        "tip_sweep_rms_change_m": sweep_rms,
        "tip_z_rms_change_m": tip_z_rms,
        "spacing_scale_change": optimized_params.y_spacing_scale - base_params.y_spacing_scale,
        "tip_chord_scale_change": optimized_params.min_tip_chord_scale - base_params.min_tip_chord_scale,
        "mutation_magnitude_index": math.sqrt(
            inc_rms * inc_rms
            + (25.0 * root_z_rms) ** 2
            + (25.0 * sweep_rms) ** 2
            + (25.0 * tip_z_rms) ** 2
            + (4.0 * (optimized_params.y_spacing_scale - base_params.y_spacing_scale)) ** 2
            + (10.0 * (optimized_params.min_tip_chord_scale - base_params.min_tip_chord_scale)) ** 2
        ),
    }


def _theory_target_metrics(
    params,
    observers: ObserverGrid,
    frequencies_hz: tuple[float, ...],
    flow: FlowConfig,
    closures: ClosureParams,
    n_eta: int,
    target_fn,
) -> dict[str, float]:
    grid = source_grid(params, n_eta=n_eta)
    spectral = evaluate_spp(
        grid,
        observers=observers,
        frequencies_hz=frequencies_hz,
        flow=flow,
        closures=closures,
    )

    target_peak_errors: list[float] = []
    rmses: list[float] = []
    maes: list[float] = []
    target_levels_by_frequency: list[list[float]] = []
    theory_levels_by_frequency: list[list[float]] = []

    for frequency_index, _frequency_hz in enumerate(frequencies_hz):
        target_levels = [float(target_fn(direction)) for direction in observers.directions]
        theory_levels = [
            spectral_level_db(spectral.value(frequency_index, observer_index), flow)
            for observer_index in range(len(observers.directions))
        ]
        target_peak = max(target_levels)
        theory_peak = max(theory_levels)
        errors = [
            (theory_level - theory_peak) - (target_level - target_peak)
            for theory_level, target_level in zip(theory_levels, target_levels)
        ]
        rmses.append(math.sqrt(sum(value * value for value in errors) / len(errors)))
        maes.append(sum(abs(value) for value in errors) / len(errors))
        target_peak_index = max(range(len(target_levels)), key=lambda index: target_levels[index])
        theory_peak_index = max(range(len(theory_levels)), key=lambda index: theory_levels[index])
        dot = sum(
            observers.directions[target_peak_index][axis] * observers.directions[theory_peak_index][axis]
            for axis in range(3)
        )
        target_peak_errors.append(math.degrees(math.acos(max(min(dot, 1.0), -1.0))))
        target_levels_by_frequency.append(target_levels)
        theory_levels_by_frequency.append(theory_levels)

    return {
        "theory_target_rmse_db": sum(rmses) / len(rmses),
        "theory_target_mae_db": sum(maes) / len(maes),
        "theory_peak_angle_error_deg": sum(target_peak_errors) / len(target_peak_errors),
    }


def _aero_proxy_metrics(
    params,
    flow: FlowConfig,
    config: SurrogateNoiseConfig,
    n_eta: int,
) -> dict[str, float]:
    grid = source_grid(params, n_eta=n_eta)
    states = local_aero_states(grid, flow, config)
    dynamic_pressure = 0.5 * flow.rho0 * flow.u_inf * flow.u_inf

    lift_total = 0.0
    drag_total = 0.0
    weighted_separation = 0.0
    weighted_incidence = 0.0
    total_area_weight = 0.0
    feather_lift: dict[int, float] = {}

    for state in states:
        area_weight = state.chord_m * state.segment_length_m
        lift = dynamic_pressure * state.cl * area_weight
        drag = dynamic_pressure * state.cd * area_weight
        lift_total += lift
        drag_total += drag
        weighted_separation += state.separation_factor * area_weight
        weighted_incidence += abs(state.incidence_deg) * area_weight
        total_area_weight += area_weight
        feather_lift[state.feather_id] = feather_lift.get(state.feather_id, 0.0) + lift

    feather_lift_values = tuple(feather_lift[feather_id] for feather_id in sorted(feather_lift))
    mean_feather_lift = sum(feather_lift_values) / max(len(feather_lift_values), 1)
    spanwise_lift_std = math.sqrt(
        sum((value - mean_feather_lift) ** 2 for value in feather_lift_values)
        / max(len(feather_lift_values), 1)
    )
    spanwise_lift_cv = spanwise_lift_std / max(abs(mean_feather_lift), 1.0e-12)

    return {
        "lift_proxy": lift_total,
        "drag_proxy": drag_total,
        "lift_to_drag_proxy": lift_total / max(drag_total, 1.0e-12),
        "separation_burden": weighted_separation / max(total_area_weight, 1.0e-12),
        "mean_abs_incidence_deg": weighted_incidence / max(total_area_weight, 1.0e-12),
        "spanwise_lift_cv": spanwise_lift_cv,
        "lift_proxy_per_area": lift_total / max(total_area_weight, 1.0e-12),
        "drag_proxy_per_area": drag_total / max(total_area_weight, 1.0e-12),
        "reference_area_proxy": total_area_weight,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Genetic Algorithm to prove feather geometry can achieve "
            "any arbitrary directivity target."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/optimization"),
        help="Directory to save a single target run.",
    )
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        default=None,
        help="Directory to save a multi-target campaign.",
    )
    parser.add_argument(
        "--popsize",
        type=int,
        default=9,
        help="Multiplier for the GA population size per active parameter.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=0,
        help="Optional hard cap on generations. Use 0 for patience-based stopping only.",
    )
    parser.add_argument(
        "--n-eta",
        type=int,
        default=12,
        help="Integration points per feather.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Random seed for target pattern generation. "
            "If omitted, a random target is generated each run."
        ),
    )
    parser.add_argument(
        "--freedom-level",
        choices=FREEDOM_LEVEL_ORDER,
        default="full",
        help="Ablation condition: incidence (incidence-only), zero_coherence, no_delay, or full.",
    )
    parser.add_argument(
        "--freedom-levels",
        default="full",
        help="Comma-separated freedom levels for in-process campaigns.",
    )
    parser.add_argument(
        "--n-targets",
        type=int,
        default=1,
        help="Number of targets to run in campaign mode.",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=1,
        help="Starting seed for campaign mode.",
    )
    return parser


def _run_single_target(
    output_dir: Path,
    args: argparse.Namespace,
    runtime: OptimizationRuntime | None = None,
    warm_start_genes: list[float] | None = None,
) -> list[float]:
    output_dir.mkdir(parents=True, exist_ok=True)

    observers = ObserverGrid.spherical(8, 10)
    target_pattern_db, target_meta = generate_arbitrary_target(observers, seed=args.seed)
    target_fn = _target_fn_from_pattern(observers, target_pattern_db)

    print("\n========================================================")
    print(" GENERALIZED DIRECTIVITY SOLVER")
    print("========================================================")
    print(f" Target: {target_meta['n_lobes']} random lobe(s), seed={args.seed}")
    print(f" Freedom: {FREEDOM_LEVEL_LABELS[args.freedom_level]}")
    print(f" Observer grid: {len(observers.directions)} directions")
    print(f"========================================================\n")

    validation_frequencies_hz = (250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)
    validation_observers = ObserverGrid.spherical(7, 14)

    # Save the target definition
    with open(output_dir / "target_definition.json", "w") as f:
        json.dump(target_meta, f, indent=2)

    # 1. RUN GENETIC ALGORITHM
    flow = FlowConfig()
    closures = effective_closures(args.freedom_level, ClosureParams())

    config = GAConfig(
        target_pattern_db=target_pattern_db,
        observers=observers,
        popsize=args.popsize,
        maxiter=args.maxiter,
        n_eta=args.n_eta,
        flow=flow,
        closures=closures,
        freedom_level=args.freedom_level,
    )
    active_parameters = active_parameter_count(config)
    effective_population = effective_population_size(config)
    print(f" Active parameters: {active_parameters}")
    print(f" Effective population: {effective_population}")

    best_geometry, best_genes, best_score, success = run_optimization_torch(
        config,
        runtime=runtime,
        warm_start_genes=warm_start_genes,
    )
    optimized_params = best_geometry
    base_params = default_geometry()

    # Save metadata
    write_geometry_metadata_json(output_dir / "optimized_geometry.json", optimized_params)

    ga_stats = {
        "success": success,
        "score_mse_db2": best_score,
        "target_seed": args.seed,
        "target_lobes": target_meta["n_lobes"],
        "freedom_level": args.freedom_level,
        "freedom_label": FREEDOM_LEVEL_LABELS[args.freedom_level],
        "freedom_level_index": FREEDOM_LEVEL_ORDER.index(args.freedom_level) + 1,
        "incidence_parameterization": "orthonormal_cosine_shape_coefficients",
        "popsize": args.popsize,
        "active_parameter_count": active_parameters,
        "effective_population_size": effective_population,
        "maxiter": args.maxiter,
        "n_eta": args.n_eta,
        "ga_observer_count": len(observers.directions),
        "ga_frequencies_hz": list(config.frequencies_hz),
        "validation_observer_count": len(validation_observers.directions),
        "validation_frequencies_hz": list(validation_frequencies_hz),
        "coarse_n_eta": config.coarse_n_eta,
        "coarse_observer_stride": config.coarse_observer_stride,
        "coarse_frequency_stride": config.coarse_frequency_stride,
        "refinement_top_fraction": config.refinement_top_fraction,
        "refinement_min_candidates": config.refinement_min_candidates,
        "patience": config.patience,
        "min_fitness_improvement_db2": config.min_fitness_improvement_db2,
    }
    with open(output_dir / "ga_stats.json", "w") as f:
        json.dump(ga_stats, f, indent=2)


    # 2. VALIDATE IN SURROGATE SIMULATOR
    print("\nGA Search Complete. Evaluating optimal geometry in Surrogate Simulator...")

    source_grid_path = write_source_grid_csv(
        output_dir / "source_grid.csv",
        optimized_params,
        n_eta=args.n_eta,
    )

    simulation_path = output_dir / "surrogate_simulation.csv"
    write_surrogate_simulation_csv(
        simulation_path,
        params=optimized_params,
        flow=flow,
        config=SurrogateNoiseConfig(),
        n_eta=args.n_eta,
        frequencies_hz=validation_frequencies_hz,
        observers=validation_observers,
    )

    simulation = load_simulation_csv(simulation_path)
    comparison = compare_theory_to_simulation(
        simulation,
        params=optimized_params,
        flow=flow,
        closures=closures,
        n_eta=args.n_eta,
        target_fn=target_fn,
    )

    comparison_path = write_comparison_csv(
        output_dir / "theory_vs_simulation.csv",
        comparison,
    )
    summary_path = write_summary_json(
        output_dir / "validation_summary.json",
        comparison,
    )
    baseline_target_fit = _theory_target_metrics(
        base_params,
        observers=validation_observers,
        frequencies_hz=validation_frequencies_hz,
        flow=flow,
        closures=closures,
        n_eta=args.n_eta,
        target_fn=target_fn,
    )
    optimized_theory_target_fit = _theory_target_metrics(
        optimized_params,
        observers=validation_observers,
        frequencies_hz=validation_frequencies_hz,
        flow=flow,
        closures=closures,
        n_eta=args.n_eta,
        target_fn=target_fn,
    )
    validated_target_fit = _shape_fit_metrics(comparison.rows, split="validation")
    mutation_metrics = _geometry_mutation_metrics(base_params, optimized_params)
    surrogate_config = SurrogateNoiseConfig()
    baseline_aero = _aero_proxy_metrics(
        base_params,
        flow=flow,
        config=surrogate_config,
        n_eta=args.n_eta,
    )
    optimized_aero = _aero_proxy_metrics(
        optimized_params,
        flow=flow,
        config=surrogate_config,
        n_eta=args.n_eta,
    )
    target_fit_summary = {
        "baseline_theory": baseline_target_fit,
        "optimized_theory": optimized_theory_target_fit,
        "validated_surrogate": {
            key: value
            for key, value in validated_target_fit.items()
            if key.startswith("surrogate_")
        },
        "validated_theory": {
            key: value
            for key, value in validated_target_fit.items()
            if key.startswith("theory_")
        },
        "baseline_aero": baseline_aero,
        "optimized_aero": optimized_aero,
        "relative_aero": {
            "lift_retention": optimized_aero["lift_proxy"] / max(baseline_aero["lift_proxy"], 1.0e-12),
            "drag_ratio": optimized_aero["drag_proxy"] / max(baseline_aero["drag_proxy"], 1.0e-12),
            "lift_to_drag_retention": optimized_aero["lift_to_drag_proxy"] / max(baseline_aero["lift_to_drag_proxy"], 1.0e-12),
            "separation_ratio": optimized_aero["separation_burden"] / max(baseline_aero["separation_burden"], 1.0e-12),
            "separation_delta": optimized_aero["separation_burden"] - baseline_aero["separation_burden"],
            "spanwise_lift_cv_change": optimized_aero["spanwise_lift_cv"] - baseline_aero["spanwise_lift_cv"],
        },
        "improvement": {
            "theory_target_rmse_db": baseline_target_fit["theory_target_rmse_db"]
            - optimized_theory_target_fit["theory_target_rmse_db"],
            "theory_target_mae_db": baseline_target_fit["theory_target_mae_db"]
            - optimized_theory_target_fit["theory_target_mae_db"],
            "surrogate_target_rmse_db": baseline_target_fit["theory_target_rmse_db"]
            - validated_target_fit["surrogate_target_rmse_db"],
            "surrogate_target_mae_db": baseline_target_fit["theory_target_mae_db"]
            - validated_target_fit["surrogate_target_mae_db"],
            "theory_peak_angle_error_deg": baseline_target_fit["theory_peak_angle_error_deg"]
            - optimized_theory_target_fit["theory_peak_angle_error_deg"],
            "surrogate_peak_angle_error_deg": baseline_target_fit["theory_peak_angle_error_deg"]
            - validated_target_fit["surrogate_peak_angle_error_deg"],
            "rmse_improvement_per_mutation_index": (
                (baseline_target_fit["theory_target_rmse_db"] - validated_target_fit["surrogate_target_rmse_db"])
                / max(mutation_metrics["mutation_magnitude_index"], 1.0e-9)
            ),
            "rmse_improvement_per_drag_ratio": (
                (baseline_target_fit["theory_target_rmse_db"] - validated_target_fit["surrogate_target_rmse_db"])
                / max(
                    optimized_aero["drag_proxy"] / max(baseline_aero["drag_proxy"], 1.0e-12),
                    1.0e-9,
                )
            ),
        },
        "mutation": mutation_metrics,
    }
    with open(output_dir / "target_fit_summary.json", "w") as f:
        json.dump(target_fit_summary, f, indent=2)

    print("\nValidation comparison complete.")
    print(f"Optimal Geometry JSON: {output_dir / 'optimized_geometry.json'}")
    print(f"Simulation CSV: {simulation_path}")
    print(f"Target-fit summary: {output_dir / 'target_fit_summary.json'}")
    print("Per-run CSV/JSON artifacts saved; aggregate plots are generated by the campaign aggregator.")
    print(comparison_summary_text(comparison.summary))
    return [float(value) for value in best_genes]


def run_campaign(args: argparse.Namespace) -> None:
    if args.campaign_dir is None:
        raise ValueError("campaign_dir is required for campaign mode.")
    campaign_dir = args.campaign_dir
    campaign_dir.mkdir(parents=True, exist_ok=True)
    freedom_levels = [
        level.strip()
        for level in args.freedom_levels.split(",")
        if level.strip()
    ]
    runtime = OptimizationRuntime()
    end_seed = args.start_seed + args.n_targets - 1

    print("\n========================================================")
    print(" GENERALIZED DIRECTIVITY CAMPAIGN")
    print("========================================================")
    print(f" Targets: {args.n_targets} | Seeds: {args.start_seed}-{end_seed}")
    print(f" Pop multiplier per active parameter: {args.popsize} | MaxIter: {args.maxiter} | N_eta: {args.n_eta}")
    print(f" Freedom levels: {', '.join(freedom_levels)}")
    print("========================================================\n")

    for target_index in range(args.n_targets):
        seed = args.start_seed + target_index
        warm_start_genes: list[float] | None = None
        for freedom_level in freedom_levels:
            if len(freedom_levels) == 1 and freedom_level == "full":
                level_output_dir = campaign_dir
            else:
                level_output_dir = campaign_dir / freedom_level
            target_dir = level_output_dir / f"target_{seed:03d}"

            print("--------------------------------------------------------")
            print(f" Freedom {freedom_level} | Target {target_index + 1}/{args.n_targets}  (seed={seed})")
            print("--------------------------------------------------------")

            target_args = argparse.Namespace(**vars(args))
            target_args.output_dir = target_dir
            target_args.seed = seed
            target_args.freedom_level = freedom_level
            warm_start_genes = _run_single_target(
                target_dir,
                target_args,
                runtime=runtime,
                warm_start_genes=warm_start_genes,
            )
            print("")

    print("========================================================")
    print(f" All {args.n_targets} targets complete. Aggregating results...")
    print("========================================================")
    aggregate_campaign_dir(campaign_dir)


def run_pipeline(args: argparse.Namespace) -> None:
    if args.campaign_dir is not None:
        run_campaign(args)
        return
    _run_single_target(args.output_dir, args)


def main() -> None:
    parser = _build_parser()
    run_pipeline(parser.parse_args())


if __name__ == "__main__":
    main()
