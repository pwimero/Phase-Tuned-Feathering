"""Campaign aggregation for the Generalized Directivity Solver.

Reads all target_xxx/ subdirectories produced by the campaign loop,
collects GA statistics and target definitions, and produces:
  - campaign_summary.csv   (one row per target)
  - campaign_mse_chart.svg (bar chart of MSE across all targets)
  - campaign_grid.svg      (small-multiple directivity polar plots)
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────
# Data collection
# ─────────────────────────────────────────────────────────────────────

def _collect_targets(campaign_dir: Path) -> list[dict]:
    """Walk target_xxx/ subdirectories and collect stats + target info."""
    results = []
    for target_dir in sorted(campaign_dir.iterdir()):
        if not target_dir.is_dir() or not target_dir.name.startswith("target_"):
            continue
        stats_path = target_dir / "ga_stats.json"
        target_path = target_dir / "target_definition.json"
        if not stats_path.exists():
            continue

        with open(stats_path) as f:
            stats = json.load(f)
        target_meta = {}
        if target_path.exists():
            with open(target_path) as f:
                target_meta = json.load(f)

        # Read the validation summary if it exists
        val_path = target_dir / "validation_summary.json"
        val_summary = {}
        if val_path.exists():
            with open(val_path) as f:
                val_summary = json.load(f)

        results.append({
            "folder": target_dir.name,
            "seed": target_meta.get("seed"),
            "n_lobes": target_meta.get("n_lobes"),
            "mse_db2": stats.get("score_mse_db2"),
            "success": stats.get("success", False),
            "rmse_db": val_summary.get("all", {}).get("rmse_db"),
            "mae_db": val_summary.get("all", {}).get("mae_db"),
            "bias_db": val_summary.get("all", {}).get("bias_db"),
        })
    return results


# ─────────────────────────────────────────────────────────────────────
# CSV output
# ─────────────────────────────────────────────────────────────────────

def _write_summary_csv(campaign_dir: Path, results: list[dict]) -> Path:
    path = campaign_dir / "campaign_summary.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["folder", "seed", "n_lobes", "mse_db2", "success",
                         "rmse_db", "mae_db", "bias_db"],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    return path


# ─────────────────────────────────────────────────────────────────────
# SVG: MSE bar chart
# ─────────────────────────────────────────────────────────────────────

def _mse_bar_chart_svg(results: list[dict], width: int = 1100, height: int = 520) -> str:
    """Horizontal bar chart of MSE per target, sorted best → worst."""
    sorted_results = sorted(results, key=lambda r: r["mse_db2"] or 999)
    n = len(sorted_results)
    if n == 0:
        return ""

    margin_left = 120
    margin_right = 40
    margin_top = 70
    margin_bottom = 60
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    bar_h = max(4, min(28, plot_h / n - 4))
    bar_gap = (plot_h - n * bar_h) / max(n, 1)

    mse_values = [r["mse_db2"] for r in sorted_results if r["mse_db2"] is not None]
    if not mse_values:
        return ""
    max_mse = max(mse_values) * 1.1
    mean_mse = sum(mse_values) / len(mse_values)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        f"<title>Campaign MSE Across Arbitrary Targets</title>",
        '<rect width="100%" height="100%" fill="#ffffff" />',
        f'<text x="28" y="36" font-family="Arial, sans-serif" font-size="22" '
        f'fill="#111">Shape-Matching Error Across {n} Arbitrary Targets</text>',
        f'<text x="28" y="56" font-family="Arial, sans-serif" font-size="13" '
        f'fill="#666">Lower MSE = better match. Mean MSE = {mean_mse:.1f} dB²</text>',
    ]

    # Mean line
    mean_x = margin_left + (mean_mse / max_mse) * plot_w
    svg.append(
        f'<line x1="{mean_x:.1f}" y1="{margin_top}" '
        f'x2="{mean_x:.1f}" y2="{height - margin_bottom}" '
        f'stroke="#e74c3c" stroke-width="2" stroke-dasharray="6,4" />'
    )
    svg.append(
        f'<text x="{mean_x + 6:.1f}" y="{margin_top + 14}" '
        f'font-family="Arial, sans-serif" font-size="11" fill="#e74c3c">'
        f'mean={mean_mse:.1f}</text>'
    )

    for i, r in enumerate(sorted_results):
        mse = r["mse_db2"]
        if mse is None:
            continue
        y = margin_top + i * (bar_h + bar_gap) + bar_gap / 2
        w = (mse / max_mse) * plot_w

        # Color: green if low MSE, orange if medium, red if high
        ratio = mse / max_mse
        if ratio < 0.4:
            color = "#2ecc71"
        elif ratio < 0.7:
            color = "#f39c12"
        else:
            color = "#e74c3c"

        svg.append(
            f'<rect x="{margin_left}" y="{y:.1f}" width="{w:.1f}" '
            f'height="{bar_h:.1f}" fill="{color}" rx="3" opacity="0.85">'
            f'<title>Seed {r["seed"]}: MSE={mse:.1f} dB², {r["n_lobes"]} lobes</title>'
            f'</rect>'
        )
        # Label on left
        svg.append(
            f'<text x="{margin_left - 8}" y="{y + bar_h / 2 + 4:.1f}" '
            f'font-family="Arial, sans-serif" font-size="11" text-anchor="end" '
            f'fill="#333">seed {r["seed"]}</text>'
        )
        # Value on right of bar
        svg.append(
            f'<text x="{margin_left + w + 6:.1f}" y="{y + bar_h / 2 + 4:.1f}" '
            f'font-family="Arial, sans-serif" font-size="11" fill="#555">'
            f'{mse:.1f}</text>'
        )

    # X axis label
    svg.append(
        f'<text x="{margin_left + plot_w / 2:.1f}" y="{height - 18}" '
        f'font-family="Arial, sans-serif" font-size="14" text-anchor="middle" '
        f'fill="#222">Normalized Mean Squared Error (dB²)</text>'
    )

    svg.append("</svg>")
    return "\n".join(svg)


# ─────────────────────────────────────────────────────────────────────
# SVG: Small-multiples directivity grid
# ─────────────────────────────────────────────────────────────────────

def _read_comparison_rows(target_dir: Path) -> list[dict]:
    """Read theory_vs_simulation.csv and return rows as dicts."""
    csv_path = target_dir / "theory_vs_simulation.csv"
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _mini_polar(
    cx: float, cy: float, radius: float,
    comparison_rows: list[dict],
    target_pattern_db: list[float] | None,
    observers_dirs: list[tuple[float, float, float]] | None,
    seed: int | None,
    mse: float | None,
) -> list[str]:
    """Generate a tiny polar directivity plot at (cx, cy)."""
    svg = []

    # Background circle
    svg.append(
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" '
        f'fill="#fafafa" stroke="#e0e0e0" stroke-width="0.8" />'
    )

    # Filter to a single frequency (pick 1000 Hz or first available)
    freq_rows = [r for r in comparison_rows
                 if r.get("split") == "validation"
                 and abs(float(r.get("frequency_hz", 0)) - 1000.0) < 1.0]
    if not freq_rows:
        freq_rows = [r for r in comparison_rows if r.get("split") == "validation"]
    if not freq_rows:
        svg.append(
            f'<text x="{cx:.1f}" y="{cy + 4:.1f}" font-family="Arial, sans-serif" '
            f'font-size="10" text-anchor="middle" fill="#999">no data</text>'
        )
        return svg

    # Sort by angle in x-z plane
    def angle_key(row):
        return math.atan2(float(row["observer_z"]), float(row["observer_x"]))

    freq_rows.sort(key=angle_key)

    # Get level bounds
    levels = [float(r["simulated_level_db"]) for r in freq_rows]
    theory_levels = [float(r["theory_level_db"]) for r in freq_rows]
    all_levels = levels + theory_levels
    min_level = min(all_levels)
    max_level = max(all_levels)
    span = max(max_level - min_level, 1.0)

    def polar_pt(row, level):
        angle = math.atan2(float(row["observer_z"]), float(row["observer_x"]))
        r_frac = 0.15 + 0.85 * (level - min_level) / span
        px = cx + radius * r_frac * math.cos(angle)
        py = cy - radius * r_frac * math.sin(angle)
        return px, py

    # Simulated polyline
    sim_pts = [polar_pt(r, float(r["simulated_level_db"])) for r in freq_rows]
    if len(sim_pts) > 1:
        coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in sim_pts + [sim_pts[0]])
        svg.append(
            f'<polyline points="{coords}" fill="none" '
            f'stroke="#d62728" stroke-width="1.5" stroke-linejoin="round" />'
        )

    # Theory polyline
    th_pts = [polar_pt(r, float(r["theory_level_db"])) for r in freq_rows]
    if len(th_pts) > 1:
        coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in th_pts + [th_pts[0]])
        svg.append(
            f'<polyline points="{coords}" fill="none" '
            f'stroke="#1f77b4" stroke-width="1.5" stroke-linejoin="round" />'
        )

    # Target polyline (if available)
    target_rows = [r for r in freq_rows if r.get("target_level_db")]
    if target_rows:
        target_levels = [float(r["target_level_db"]) for r in target_rows]
        # Offset target to align with simulated peak
        sim_peak = max(levels)
        target_peak = max(target_levels) if target_levels else sim_peak
        offset = sim_peak - target_peak

        tgt_pts = [polar_pt(r, float(r["target_level_db"]) + offset) for r in target_rows]
        if len(tgt_pts) > 1:
            coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in tgt_pts + [tgt_pts[0]])
            svg.append(
                f'<polyline points="{coords}" fill="none" '
                f'stroke="#2ca02c" stroke-width="1.5" stroke-dasharray="3,2" '
                f'stroke-linejoin="round" />'
            )

    # Label: seed
    svg.append(
        f'<text x="{cx:.1f}" y="{cy - radius - 6:.1f}" font-family="Arial, sans-serif" '
        f'font-size="11" text-anchor="middle" fill="#333">seed {seed}</text>'
    )
    # Label: MSE
    if mse is not None:
        svg.append(
            f'<text x="{cx:.1f}" y="{cy + radius + 14:.1f}" font-family="Arial, sans-serif" '
            f'font-size="10" text-anchor="middle" fill="#888">MSE {mse:.1f}</text>'
        )

    return svg


def _directivity_grid_svg(
    campaign_dir: Path,
    results: list[dict],
    width: int = 1200,
) -> str:
    """Generate a grid of small-multiple directivity polar plots."""
    n = len(results)
    if n == 0:
        return ""

    cols = min(5, n)
    rows = math.ceil(n / cols)
    cell_w = width // cols
    cell_h = cell_w  # square cells
    radius = cell_w * 0.32
    total_h = 90 + rows * cell_h + 60  # header + grid + legend

    mse_values = [r["mse_db2"] for r in results if r["mse_db2"] is not None]
    mean_mse = sum(mse_values) / len(mse_values) if mse_values else 0

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{total_h}" '
        f'viewBox="0 0 {width} {total_h}" role="img">',
        f"<title>Directivity Grid — {n} Arbitrary Targets</title>",
        '<rect width="100%" height="100%" fill="#ffffff" />',
        f'<text x="28" y="36" font-family="Arial, sans-serif" font-size="22" '
        f'fill="#111">Directivity Achieved vs Target — {n} Arbitrary Shapes</text>',
        f'<text x="28" y="56" font-family="Arial, sans-serif" font-size="13" '
        f'fill="#666">Each cell: 1 kHz polar plot. '
        f'Red=simulated, Blue=theory, Green dashed=target. '
        f'Mean MSE={mean_mse:.1f} dB²</text>',
    ]

    for i, r in enumerate(results):
        col = i % cols
        row_idx = i // cols
        cx = col * cell_w + cell_w // 2
        cy = 90 + row_idx * cell_h + cell_h // 2

        target_dir = campaign_dir / r["folder"]
        comparison_rows = _read_comparison_rows(target_dir)

        mini = _mini_polar(
            cx, cy, radius,
            comparison_rows=comparison_rows,
            target_pattern_db=None,
            observers_dirs=None,
            seed=r["seed"],
            mse=r["mse_db2"],
        )
        svg.extend(mini)

    # Legend at bottom
    legend_y = total_h - 36
    svg.append(
        f'<line x1="40" y1="{legend_y}" x2="70" y2="{legend_y}" '
        f'stroke="#d62728" stroke-width="2.5" />'
        f'<text x="78" y="{legend_y + 4}" font-family="Arial, sans-serif" '
        f'font-size="12" fill="#444">simulated</text>'
    )
    svg.append(
        f'<line x1="160" y1="{legend_y}" x2="190" y2="{legend_y}" '
        f'stroke="#1f77b4" stroke-width="2.5" />'
        f'<text x="198" y="{legend_y + 4}" font-family="Arial, sans-serif" '
        f'font-size="12" fill="#444">theory</text>'
    )
    svg.append(
        f'<line x1="260" y1="{legend_y}" x2="290" y2="{legend_y}" '
        f'stroke="#2ca02c" stroke-width="2.5" stroke-dasharray="5,3" />'
        f'<text x="298" y="{legend_y + 4}" font-family="Arial, sans-serif" '
        f'font-size="12" fill="#444">target shape</text>'
    )

    svg.append("</svg>")
    return "\n".join(svg)


# ─────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate results from a Generalized Directivity campaign."
    )
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        required=True,
        help="Root directory containing target_xxx/ subdirectories.",
    )
    args = parser.parse_args()

    campaign_dir = args.campaign_dir
    results = _collect_targets(campaign_dir)

    if not results:
        print("No target results found. Nothing to aggregate.")
        return

    print(f"Found {len(results)} target(s) in {campaign_dir}")

    # Write CSV
    csv_path = _write_summary_csv(campaign_dir, results)
    print(f"  Summary CSV: {csv_path}")

    # MSE bar chart
    bar_svg = _mse_bar_chart_svg(results)
    if bar_svg:
        bar_path = campaign_dir / "campaign_mse_chart.svg"
        bar_path.write_text(bar_svg, encoding="utf-8")
        print(f"  MSE chart:   {bar_path}")

    # Directivity grid
    grid_svg = _directivity_grid_svg(campaign_dir, results)
    if grid_svg:
        grid_path = campaign_dir / "campaign_directivity_grid.svg"
        grid_path.write_text(grid_svg, encoding="utf-8")
        print(f"  Grid plot:   {grid_path}")

    # Print summary stats
    mse_values = [r["mse_db2"] for r in results if r["mse_db2"] is not None]
    if mse_values:
        mean = sum(mse_values) / len(mse_values)
        best = min(mse_values)
        worst = max(mse_values)
        std = (sum((v - mean) ** 2 for v in mse_values) / len(mse_values)) ** 0.5
        print(f"\n  Campaign Statistics ({len(mse_values)} targets):")
        print(f"    Mean MSE:  {mean:.2f} dB²")
        print(f"    Std MSE:   {std:.2f} dB²")
        print(f"    Best MSE:  {best:.2f} dB²")
        print(f"    Worst MSE: {worst:.2f} dB²")


if __name__ == "__main__":
    main()
