"""Genetic Algorithm Optimization Pipeline.

This pipeline runs a Genetic Algorithm to find the optimal feather geometry
for a given acoustic directivity scenario. It then automatically feeds the 
theoretically-optimized wing into the empirical Surrogate Simulator to prove 
that the design successfully steers sound under real-world aerodynamic physics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json

from .closures import ClosureParams, FlowConfig
from .ga_optimization import GAConfig, run_optimization_torch
from .io import write_geometry_metadata_json, write_source_grid_csv
from .observers import ObserverGrid, Sector
from .simulation import SurrogateNoiseConfig, write_surrogate_simulation_csv
from .validation import (
    compare_theory_to_simulation,
    comparison_summary_text,
    load_simulation_csv,
    write_comparison_csv,
    write_summary_json,
)
from .visualization import write_simulator_geometry_renders, write_validation_figures


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Genetic Algorithm to design phase-tuned wings for target directivity."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/optimization"),
        help="Directory to save the GA results and validation plots.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="quiet_ground",
        choices=["quiet_ground", "forward_focus"],
        help="The directivity scenario to optimize for.",
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
    return parser


def run_pipeline(args: argparse.Namespace) -> None:
    output_dir = args.output_dir / args.scenario
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n========================================================")
    print(f" SCENARIO: {args.scenario}")
    print(f"========================================================\n")

    if args.scenario == "quiet_ground":
        # Target: UP (+z), Suppressed: DOWN (-z)
        target_sector = Sector(center=(0.0, 0.0, 1.0), half_angle_deg=70.0)
        suppressed_sector = Sector(center=(0.0, 0.0, -1.0), half_angle_deg=70.0)
    elif args.scenario == "forward_focus":
        # Target: FORWARD (+x), Suppressed: BACKWARD (-x)
        target_sector = Sector(center=(1.0, 0.0, 0.0), half_angle_deg=70.0)
        suppressed_sector = Sector(center=(-1.0, 0.0, 0.0), half_angle_deg=70.0)
    else:
        raise ValueError(f"Unknown scenario {args.scenario}")

    # 1. RUN GENETIC ALGORITHM
    flow = FlowConfig()
    closures = ClosureParams()

    config = GAConfig(
        target_sector=target_sector,
        suppressed_sector=suppressed_sector,
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
        "score_db": best_score,
    }
    with open(output_dir / "ga_stats.json", "w") as f:
        json.dump(ga_stats, f, indent=2)


    # 2. VALIDATE IN SURROGATE SIMULATOR
    print("\nGA Search Complete. Evaluating optimal geometry in Surrogate Simulator...")
    
    flow = FlowConfig()
    closures = ClosureParams()
    
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
