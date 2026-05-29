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
from .ga_optimization import GAConfig, run_optimization_torch
from .io import write_geometry_metadata_json, write_source_grid_csv
from .observers import ObserverGrid
from .simulation import SurrogateNoiseConfig, write_surrogate_simulation_csv
from .validation import (
    compare_theory_to_simulation,
    comparison_summary_text,
    load_simulation_csv,
    write_comparison_csv,
    write_summary_json,
)
from .visualization import write_simulator_geometry_renders, write_validation_figures


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
        help="Directory to save the GA results and validation plots.",
    )
    parser.add_argument(
        "--popsize",
        type=int,
        default=10,
        help="Multiplier for the GA population size.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=30,
        help="Maximum generations for the GA.",
    )
    parser.add_argument(
        "--n-eta",
        type=int,
        default=48,
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
    return parser


def run_pipeline(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate an arbitrary target
    observers = ObserverGrid.spherical(10, 16)
    target_pattern_db, target_meta = generate_arbitrary_target(observers, seed=args.seed)
    target_fn = _target_fn_from_pattern(observers, target_pattern_db)

    print("\n========================================================")
    print(" GENERALIZED DIRECTIVITY SOLVER")
    print("========================================================")
    print(f" Target: {target_meta['n_lobes']} random lobe(s), seed={args.seed}")
    print(f" Observer grid: {len(observers.directions)} directions")
    print(f"========================================================\n")

    # Save the target definition
    with open(output_dir / "target_definition.json", "w") as f:
        json.dump(target_meta, f, indent=2)

    # 1. RUN GENETIC ALGORITHM
    flow = FlowConfig()
    closures = ClosureParams()

    config = GAConfig(
        target_pattern_db=target_pattern_db,
        popsize=args.popsize,
        maxiter=args.maxiter,
        flow=flow,
        closures=closures,
    )

    best_geometry, best_score, success = run_optimization_torch(config)
    optimized_params = best_geometry

    # Save metadata
    write_geometry_metadata_json(output_dir / "optimized_geometry.json", optimized_params)

    ga_stats = {
        "success": success,
        "score_mse_db2": best_score,
        "target_seed": args.seed,
        "target_lobes": target_meta["n_lobes"],
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

    render_paths = write_simulator_geometry_renders(
        output_dir,
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
        # Use high resolution validation metrics
        frequencies_hz=(250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0),
        observers=ObserverGrid.spherical(12, 24),
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
    figure_paths = write_validation_figures(output_dir, comparison)

    print("\nValidation comparison complete.")
    print(f"Optimal Geometry JSON: {output_dir / 'optimized_geometry.json'}")
    print(f"Simulation CSV: {simulation_path}")
    print("Geometry renders generated.")
    print("Validation figures generated.")
    print(comparison_summary_text(comparison.summary))


def main() -> None:
    parser = _build_parser()
    run_pipeline(parser.parse_args())


if __name__ == "__main__":
    main()
