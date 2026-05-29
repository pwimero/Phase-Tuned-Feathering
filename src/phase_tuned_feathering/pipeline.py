"""Command-line validation pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from .closures import ClosureParams, FlowConfig
from .geometry import default_geometry
from .io import write_geometry_metadata_json, write_source_grid_csv
from .simulation import SurrogateNoiseConfig, write_surrogate_simulation_csv
from .validation import (
    compare_theory_to_simulation,
    comparison_summary_text,
    load_simulation_csv,
    write_comparison_csv,
    write_summary_json,
    write_synthetic_simulation_csv,
)
from .visualization import write_simulator_geometry_renders, write_validation_figures


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare phase-tuned feathering theory against simulated "
            "performance data."
        )
    )
    parser.add_argument(
        "--simulation-csv",
        type=Path,
        default=None,
        help=(
            "CSV with columns frequency_hz, observer_x, observer_y, "
            "observer_z, spp. Optional columns: case_id, split, weight. "
            "If omitted, a deterministic synthetic dataset is generated."
        ),
    )
    parser.add_argument(
        "--simulation-mode",
        choices=("surrogate", "synthetic"),
        default="surrogate",
        help=(
            "Generated simulation source when --simulation-csv is omitted. "
            "'surrogate' uses the low-cost aeroacoustic simulator; 'synthetic' "
            "uses theory plus perturbation for smoke tests only."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/validation"),
        help="Directory for generated metadata, comparison CSV, and summary JSON.",
    )
    parser.add_argument(
        "--n-eta",
        type=int,
        default=48,
        help="Source-line quadrature points per feather.",
    )
    parser.add_argument(
        "--source-chord-fraction",
        type=float,
        default=1.0,
        help="Source chord fraction: 1.0 is trailing edge, 0.25 is Fusion pivot.",
    )
    parser.add_argument(
        "--synthetic-seed",
        type=int,
        default=7,
        help="Seed used only when generating synthetic comparison data.",
    )
    parser.add_argument(
        "--no-calibrate-level",
        action="store_true",
        help=(
            "Disable scalar source-level calibration on rows marked "
            "split=calibration."
        ),
    )
    return parser


def run_pipeline(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    params = default_geometry()
    flow = FlowConfig()
    closures = ClosureParams()

    metadata_path = write_geometry_metadata_json(
        output_dir / "geometry_metadata.json",
        params,
    )
    source_grid_path = write_source_grid_csv(
        output_dir / "source_grid.csv",
        params,
        n_eta=args.n_eta,
        source_chord_fraction=args.source_chord_fraction,
    )
    render_paths = write_simulator_geometry_renders(
        output_dir,
        params,
        n_eta=args.n_eta,
        source_chord_fraction=args.source_chord_fraction,
    )

    simulation_mode = getattr(args, "simulation_mode", "surrogate")
    if args.simulation_csv is None and simulation_mode == "surrogate":
        simulation_path = output_dir / "surrogate_simulation.csv"
        write_surrogate_simulation_csv(
            simulation_path,
            params=params,
            flow=flow,
            config=SurrogateNoiseConfig(),
            n_eta=args.n_eta,
            source_chord_fraction=args.source_chord_fraction,
        )
    elif args.simulation_csv is None:
        simulation_path = output_dir / "synthetic_simulation.csv"
        write_synthetic_simulation_csv(
            simulation_path,
            params=params,
            flow=flow,
            closures=closures,
            n_eta=args.n_eta,
            seed=args.synthetic_seed,
        )
    else:
        simulation_path = args.simulation_csv

    simulation = load_simulation_csv(simulation_path)
    comparison = compare_theory_to_simulation(
        simulation,
        params=params,
        flow=flow,
        closures=closures,
        n_eta=args.n_eta,
        source_chord_fraction=args.source_chord_fraction,
        calibrate_level=not getattr(args, "no_calibrate_level", False),
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

    print("Validation comparison complete.")
    print(f"Simulation CSV: {simulation_path}")
    print("Geometry renders:")
    for render_path in render_paths:
        print(f"  {render_path}")
    print(f"Comparison CSV: {comparison_path}")
    print(f"Summary JSON: {summary_path}")
    print("Validation figures:")
    for figure_path in figure_paths:
        print(f"  {figure_path}")
    print(comparison_summary_text(comparison.summary))

    return {
        "metadata": metadata_path,
        "source_grid": source_grid_path,
        **{f"render_{index}": path for index, path in enumerate(render_paths)},
        "simulation": simulation_path,
        "comparison": comparison_path,
        "summary": summary_path,
        **{f"figure_{index}": path for index, path in enumerate(figure_paths)},
    }


def main() -> None:
    parser = _build_parser()
    run_pipeline(parser.parse_args())


if __name__ == "__main__":
    main()
