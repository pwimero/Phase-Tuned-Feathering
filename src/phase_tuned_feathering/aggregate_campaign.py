"""Campaign aggregation for the Generalized Directivity Solver.

Reads all target_xxx/ subdirectories produced by the campaign loop,
collects GA statistics and target-fit summaries, and produces:
  - campaign_summary.csv
  - campaign_mse_chart.svg
  - campaign_target_fit_chart.svg
  - campaign_mutation_effectiveness.svg
  - campaign_aero_tradeoff.svg
  - campaign_aero_retention_chart.svg
  - campaign_directivity_grid.svg
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────
# Data collection
# ─────────────────────────────────────────────────────────────────────

def _collect_targets(campaign_dir: Path) -> list[dict]:
    """Walk target_xxx/ subdirectories and collect stats + target info."""
    results = []
    for stats_path in sorted(campaign_dir.rglob("ga_stats.json")):
        target_dir = stats_path.parent
        if not target_dir.name.startswith("target_"):
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

        fit_path = target_dir / "target_fit_summary.json"
        fit_summary = {}
        if fit_path.exists():
            with open(fit_path) as f:
                fit_summary = json.load(f)

        baseline_theory = fit_summary.get("baseline_theory", {})
        optimized_theory = fit_summary.get("optimized_theory", {})
        validated_surrogate = fit_summary.get("validated_surrogate", {})
        baseline_aero = fit_summary.get("baseline_aero", {})
        optimized_aero = fit_summary.get("optimized_aero", {})
        relative_aero = fit_summary.get("relative_aero", {})
        improvement = fit_summary.get("improvement", {})
        mutation = fit_summary.get("mutation", {})
        relative_folder = target_dir.relative_to(campaign_dir).as_posix()

        results.append({
            "folder": relative_folder,
            "target_path": str(target_dir),
            "seed": target_meta.get("seed"),
            "n_lobes": target_meta.get("n_lobes"),
            "freedom_level": stats.get("freedom_level", "full"),
            "freedom_label": stats.get("freedom_label", "Full geometry"),
            "freedom_level_index": stats.get("freedom_level_index", 6),
            "mse_db2": stats.get("score_mse_db2"),
            "success": stats.get("success", False),
            "rmse_db": val_summary.get("all", {}).get("rmse_db"),
            "mae_db": val_summary.get("all", {}).get("mae_db"),
            "bias_db": val_summary.get("all", {}).get("bias_db"),
            "baseline_theory_target_rmse_db": baseline_theory.get("theory_target_rmse_db"),
            "optimized_theory_target_rmse_db": optimized_theory.get("theory_target_rmse_db"),
            "validated_surrogate_target_rmse_db": validated_surrogate.get("surrogate_target_rmse_db"),
            "baseline_theory_target_mae_db": baseline_theory.get("theory_target_mae_db"),
            "optimized_theory_target_mae_db": optimized_theory.get("theory_target_mae_db"),
            "validated_surrogate_target_mae_db": validated_surrogate.get("surrogate_target_mae_db"),
            "baseline_peak_angle_error_deg": baseline_theory.get("theory_peak_angle_error_deg"),
            "optimized_theory_peak_angle_error_deg": optimized_theory.get("theory_peak_angle_error_deg"),
            "validated_surrogate_peak_angle_error_deg": validated_surrogate.get("surrogate_peak_angle_error_deg"),
            "baseline_lift_proxy": baseline_aero.get("lift_proxy"),
            "baseline_drag_proxy": baseline_aero.get("drag_proxy"),
            "baseline_lift_to_drag_proxy": baseline_aero.get("lift_to_drag_proxy"),
            "baseline_separation_burden": baseline_aero.get("separation_burden"),
            "optimized_lift_proxy": optimized_aero.get("lift_proxy"),
            "optimized_drag_proxy": optimized_aero.get("drag_proxy"),
            "optimized_lift_to_drag_proxy": optimized_aero.get("lift_to_drag_proxy"),
            "optimized_separation_burden": optimized_aero.get("separation_burden"),
            "optimized_mean_abs_incidence_deg": optimized_aero.get("mean_abs_incidence_deg"),
            "optimized_spanwise_lift_cv": optimized_aero.get("spanwise_lift_cv"),
            "lift_retention": relative_aero.get("lift_retention"),
            "drag_ratio": relative_aero.get("drag_ratio"),
            "lift_to_drag_retention": relative_aero.get("lift_to_drag_retention"),
            "separation_ratio": relative_aero.get("separation_ratio"),
            "separation_delta": relative_aero.get("separation_delta"),
            "spanwise_lift_cv_change": relative_aero.get("spanwise_lift_cv_change"),
            "theory_target_rmse_improvement_db": improvement.get("theory_target_rmse_db"),
            "surrogate_target_rmse_improvement_db": improvement.get("surrogate_target_rmse_db"),
            "theory_target_mae_improvement_db": improvement.get("theory_target_mae_db"),
            "surrogate_target_mae_improvement_db": improvement.get("surrogate_target_mae_db"),
            "mutation_magnitude_index": mutation.get("mutation_magnitude_index"),
            "incidence_rms_change_deg": mutation.get("incidence_rms_change_deg"),
            "root_z_rms_change_m": mutation.get("root_z_rms_change_m"),
            "tip_sweep_rms_change_m": mutation.get("tip_sweep_rms_change_m"),
            "tip_z_rms_change_m": mutation.get("tip_z_rms_change_m"),
            "spacing_scale_change": mutation.get("spacing_scale_change"),
            "tip_chord_scale_change": mutation.get("tip_chord_scale_change"),
            "rmse_improvement_per_mutation_index": improvement.get("rmse_improvement_per_mutation_index"),
            "rmse_improvement_per_drag_ratio": improvement.get("rmse_improvement_per_drag_ratio"),
        })
    return results


# ─────────────────────────────────────────────────────────────────────
# CSV output
# ─────────────────────────────────────────────────────────────────────

def _write_summary_csv(campaign_dir: Path, results: list[dict]) -> Path:
    path = campaign_dir / "campaign_summary.csv"
    fieldnames = [
        "folder", "target_path", "seed", "n_lobes", "freedom_level", "freedom_label",
        "freedom_level_index", "mse_db2", "success",
        "rmse_db", "mae_db", "bias_db",
        "baseline_theory_target_rmse_db", "optimized_theory_target_rmse_db",
        "validated_surrogate_target_rmse_db",
        "baseline_theory_target_mae_db", "optimized_theory_target_mae_db",
        "validated_surrogate_target_mae_db",
        "baseline_peak_angle_error_deg", "optimized_theory_peak_angle_error_deg",
        "validated_surrogate_peak_angle_error_deg",
        "baseline_lift_proxy", "baseline_drag_proxy", "baseline_lift_to_drag_proxy",
        "baseline_separation_burden", "optimized_lift_proxy", "optimized_drag_proxy",
        "optimized_lift_to_drag_proxy", "optimized_separation_burden",
        "optimized_mean_abs_incidence_deg", "optimized_spanwise_lift_cv",
        "lift_retention", "drag_ratio", "lift_to_drag_retention",
        "separation_ratio", "separation_delta", "spanwise_lift_cv_change",
        "theory_target_rmse_improvement_db", "surrogate_target_rmse_improvement_db",
        "theory_target_mae_improvement_db", "surrogate_target_mae_improvement_db",
        "mutation_magnitude_index", "incidence_rms_change_deg",
        "root_z_rms_change_m", "tip_sweep_rms_change_m", "tip_z_rms_change_m",
        "spacing_scale_change", "tip_chord_scale_change",
        "rmse_improvement_per_mutation_index", "rmse_improvement_per_drag_ratio",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    return path


def _mean(values: list[float]) -> float | None:
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _std(values: list[float]) -> float | None:
    values = [value for value in values if value is not None]
    if not values:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _freedom_summary(results: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in results:
        grouped.setdefault(row.get("freedom_level") or "full", []).append(row)

    summary: list[dict] = []
    for freedom_level, rows in grouped.items():
        first = rows[0]
        validated_rmse = [
            row["validated_surrogate_target_rmse_db"]
            for row in rows
            if row["validated_surrogate_target_rmse_db"] is not None
        ]
        improvement = [
            row["surrogate_target_rmse_improvement_db"]
            for row in rows
            if row["surrogate_target_rmse_improvement_db"] is not None
        ]
        mutation = [
            row["mutation_magnitude_index"]
            for row in rows
            if row["mutation_magnitude_index"] is not None
        ]
        drag_ratio = [
            row["drag_ratio"]
            for row in rows
            if row["drag_ratio"] is not None
        ]
        ld_retention = [
            row["lift_to_drag_retention"]
            for row in rows
            if row["lift_to_drag_retention"] is not None
        ]
        summary.append({
            "freedom_level": freedom_level,
            "freedom_label": first.get("freedom_label") or freedom_level,
            "freedom_level_index": first.get("freedom_level_index") or 999,
            "target_count": len(rows),
            "mean_validated_surrogate_target_rmse_db": _mean(validated_rmse),
            "std_validated_surrogate_target_rmse_db": _std(validated_rmse),
            "mean_surrogate_target_rmse_improvement_db": _mean(improvement),
            "std_surrogate_target_rmse_improvement_db": _std(improvement),
            "mean_mutation_magnitude_index": _mean(mutation),
            "mean_drag_ratio": _mean(drag_ratio),
            "mean_lift_to_drag_retention": _mean(ld_retention),
        })
    return sorted(summary, key=lambda row: row["freedom_level_index"])


def _write_freedom_summary_csv(campaign_dir: Path, summary: list[dict]) -> Path:
    path = campaign_dir / "campaign_freedom_summary.csv"
    fieldnames = [
        "freedom_level", "freedom_label", "freedom_level_index", "target_count",
        "mean_validated_surrogate_target_rmse_db",
        "std_validated_surrogate_target_rmse_db",
        "mean_surrogate_target_rmse_improvement_db",
        "std_surrogate_target_rmse_improvement_db",
        "mean_mutation_magnitude_index",
        "mean_drag_ratio",
        "mean_lift_to_drag_retention",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)
    return path


def _freedom_fit_chart_svg(summary: list[dict], width: int = 1512, height: int = 756) -> str:
    rows = [
        row for row in summary
        if row["mean_validated_surrogate_target_rmse_db"] is not None
    ]
    if len(rows) < 2:
        return ""

    margin_left = 340
    margin_right = 96
    margin_top = 116
    margin_bottom = 88
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    max_rmse = max(row["mean_validated_surrogate_target_rmse_db"] for row in rows) * 1.15
    row_gap = plot_h / max(len(rows), 1)
    bar_h = min(68.0, row_gap * 0.55)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        "<title>Mechanism Condition vs Target Fit</title>",
        '<rect width="100%" height="100%" fill="#ffffff" />',
        '<text x="32" y="40" font-family="Arial, sans-serif" font-size="26" fill="#111">Mechanism Condition vs Validated Target-Fit Error</text>',
        '<text x="32" y="66" font-family="Arial, sans-serif" font-size="16" fill="#666">Lower RMSE means better directivity matching. One bar per ablation condition, averaged across the same target seeds.</text>',
    ]
    for tick_index in range(6):
        value = max_rmse * tick_index / 5.0
        x = margin_left + (value / max(max_rmse, 1.0e-9)) * plot_w
        svg.append(f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{height - margin_bottom}" stroke="#eeeeee" stroke-width="1" />')
        svg.append(f'<text x="{x:.1f}" y="{height - margin_bottom + 22:.1f}" font-family="Arial, sans-serif" font-size="14" text-anchor="middle" fill="#666">{value:.1f}</text>')

    for index, row in enumerate(rows):
        value = row["mean_validated_surrogate_target_rmse_db"]
        y = margin_top + index * row_gap + (row_gap - bar_h) / 2
        w = (value / max(max_rmse, 1.0e-9)) * plot_w
        label = row["freedom_label"]
        svg.append(f'<rect x="{margin_left:.1f}" y="{y:.1f}" width="{w:.1f}" height="{bar_h:.1f}" fill="#1f77b4" rx="4" opacity="0.9" />')
        svg.append(f'<text x="{margin_left - 14:.1f}" y="{y + bar_h * 0.62:.1f}" font-family="Arial, sans-serif" font-size="17" text-anchor="end" fill="#333">{label}</text>')
        value_x = min(margin_left + w + 10.0, width - margin_right - 8.0)
        svg.append(f'<text x="{value_x:.1f}" y="{y + bar_h * 0.62:.1f}" font-family="Arial, sans-serif" font-size="15" fill="#333">{value:.2f} dB</text>')

    svg.append(f'<text x="{margin_left + plot_w / 2:.1f}" y="{height - 18}" font-family="Arial, sans-serif" font-size="18" text-anchor="middle" fill="#222">Mean validated target-shape RMSE (dB)</text>')
    svg.append("</svg>")
    return "\n".join(svg)


def _freedom_tradeoff_chart_svg(summary: list[dict], width: int = 1512, height: int = 756) -> str:
    rows = [
        row for row in summary
        if row["mean_surrogate_target_rmse_improvement_db"] is not None
        and row["mean_drag_ratio"] is not None
    ]
    if len(rows) < 2:
        return ""

    margin_left = 128
    margin_right = 360
    margin_top = 116
    margin_bottom = 108
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    min_x = min(row["mean_surrogate_target_rmse_improvement_db"] for row in rows)
    max_x = max(row["mean_surrogate_target_rmse_improvement_db"] for row in rows)
    min_y = min(row["mean_drag_ratio"] for row in rows)
    max_y = max(row["mean_drag_ratio"] for row in rows)
    x_pad = max(0.25, 0.15 * max(max_x - min_x, 1.0))
    y_pad = max(0.05, 0.15 * max(max_y - min_y, 0.1))
    min_x -= x_pad
    max_x += x_pad
    min_y = max(0.0, min(min_y - y_pad, 1.0))
    max_y = max(max_y + y_pad, 1.0)

    def x_map(value: float) -> float:
        return margin_left + (value - min_x) / max(max_x - min_x, 1.0e-9) * plot_w

    def y_map(value: float) -> float:
        return margin_top + (max_y - value) / max(max_y - min_y, 1.0e-9) * plot_h

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        "<title>Mechanism Condition Acoustic-Aero Tradeoff</title>",
        '<rect width="100%" height="100%" fill="#ffffff" />',
        '<text x="32" y="40" font-family="Arial, sans-serif" font-size="26" fill="#111">Mechanism Conditions: Acoustic Gain vs Drag Cost</text>',
        '<text x="32" y="66" font-family="Arial, sans-serif" font-size="16" fill="#666">Each point is one ablation condition averaged across target seeds. Right is better acoustic improvement; lower is lower drag cost.</text>',
    ]

    for tick_index in range(6):
        value = min_x + (max_x - min_x) * tick_index / 5.0
        x = x_map(value)
        svg.append(f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{height - margin_bottom}" stroke="#eeeeee" stroke-width="1" />')
        svg.append(f'<text x="{x:.1f}" y="{height - margin_bottom + 18}" font-family="Arial, sans-serif" font-size="14" text-anchor="middle" fill="#666">{value:.1f}</text>')
    for tick_index in range(6):
        value = min_y + (max_y - min_y) * tick_index / 5.0
        y = y_map(value)
        svg.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#f3f3f3" stroke-width="1" />')
        svg.append(f'<text x="{margin_left - 10}" y="{y + 4:.1f}" font-family="Arial, sans-serif" font-size="14" text-anchor="end" fill="#666">{value:.2f}</text>')

    svg.append(f'<line x1="{margin_left}" y1="{y_map(1.0):.1f}" x2="{width - margin_right}" y2="{y_map(1.0):.1f}" stroke="#999999" stroke-width="1.2" stroke-dasharray="4,4" />')
    palette = ["#1f77b4", "#2ca02c", "#e67e22", "#d62728", "#9467bd", "#8c564b"]
    duplicate_counts: dict[tuple[float, float], int] = {}
    legend_x = width - margin_right + 26
    legend_y = margin_top + 10
    svg.append(f'<rect x="{legend_x - 16}" y="{legend_y - 18}" width="{margin_right - 40}" height="{min(38 + 64 * len(rows), height - margin_top - margin_bottom - 8)}" fill="#fafafa" stroke="#dddddd" rx="8" />')
    svg.append(f'<text x="{legend_x:.1f}" y="{legend_y:.1f}" font-family="Arial, sans-serif" font-size="17" fill="#222">Condition key</text>')
    for index, row in enumerate(rows):
        x = x_map(row["mean_surrogate_target_rmse_improvement_db"])
        y = y_map(row["mean_drag_ratio"])
        key = (round(row["mean_surrogate_target_rmse_improvement_db"], 9), round(row["mean_drag_ratio"], 9))
        dup_index = duplicate_counts.get(key, 0)
        duplicate_counts[key] = dup_index + 1
        if dup_index:
            x += 10.0 * dup_index
            y -= 10.0 * dup_index
        color = palette[index % len(palette)]
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7.5" fill="{color}" stroke="#ffffff" stroke-width="1.5" opacity="0.95" />')
        row_y = legend_y + 28 + 56 * index
        svg.append(f'<circle cx="{legend_x + 8:.1f}" cy="{row_y - 6:.1f}" r="7.0" fill="{color}" stroke="#ffffff" stroke-width="1.2" />')
        svg.append(f'<text x="{legend_x + 24:.1f}" y="{row_y - 1:.1f}" font-family="Arial, sans-serif" font-size="15" fill="#222">{row["freedom_label"]}</text>')
        svg.append(f'<text x="{legend_x + 24:.1f}" y="{row_y + 18:.1f}" font-family="Arial, sans-serif" font-size="13" fill="#666">improvement {row["mean_surrogate_target_rmse_improvement_db"]:.2f} dB, drag ratio {row["mean_drag_ratio"]:.3f}</text>')

    svg.append(f'<text x="{margin_left + plot_w / 2:.1f}" y="{height - 16}" font-family="Arial, sans-serif" font-size="18" text-anchor="middle" fill="#222">Mean target RMSE improvement vs baseline (dB)</text>')
    svg.append(f'<text x="22" y="{margin_top + plot_h / 2:.1f}" font-family="Arial, sans-serif" font-size="18" fill="#222" transform="rotate(-90 22 {margin_top + plot_h / 2:.1f})">Mean drag ratio</text>')
    svg.append("</svg>")
    return "\n".join(svg)


# ─────────────────────────────────────────────────────────────────────
# SVG: MSE bar chart
# ─────────────────────────────────────────────────────────────────────

def _mse_bar_chart_svg(results: list[dict], width: int = 1485, height: int = 702) -> str:
    """Horizontal bar chart of MSE per target, sorted best → worst."""
    sorted_results = sorted(results, key=lambda r: r["mse_db2"] or 999)
    n = len(sorted_results)
    if n == 0:
        return ""

    margin_left = 112
    margin_right = 64
    margin_top = 102
    margin_bottom = 92
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
        f'<text x="32" y="40" font-family="Arial, sans-serif" font-size="26" '
        f'fill="#111">Shape-Matching Error Across {n} Arbitrary Targets</text>',
        f'<text x="32" y="66" font-family="Arial, sans-serif" font-size="16" '
        f'fill="#666">One bar per optimization run, sorted from best to worst. Lower MSE means better target-shape matching. Mean MSE = {mean_mse:.1f} dB²</text>',
    ]

    for tick_index in range(6):
        value = max_mse * tick_index / 5.0
        x = margin_left + (value / max(max_mse, 1.0e-9)) * plot_w
        svg.append(
            f'<line x1="{x:.1f}" y1="{margin_top}" '
            f'x2="{x:.1f}" y2="{height - margin_bottom}" '
            f'stroke="#efefef" stroke-width="1" />'
        )
        svg.append(
            f'<text x="{x:.1f}" y="{height - margin_bottom + 22}" '
            f'font-family="Arial, sans-serif" font-size="14" text-anchor="middle" '
            f'fill="#666">{value:.0f}</text>'
        )

    # Mean line
    mean_x = margin_left + (mean_mse / max_mse) * plot_w
    svg.append(
        f'<line x1="{mean_x:.1f}" y1="{margin_top}" '
        f'x2="{mean_x:.1f}" y2="{height - margin_bottom}" '
        f'stroke="#e74c3c" stroke-width="2" stroke-dasharray="6,4" />'
    )
    svg.append(
        f'<text x="{mean_x + 6:.1f}" y="{margin_top + 14}" '
        f'font-family="Arial, sans-serif" font-size="14" fill="#e74c3c">'
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
    svg.append(
        f'<text x="{margin_left + plot_w / 2:.1f}" y="{height - 18}" '
        f'font-family="Arial, sans-serif" font-size="18" text-anchor="middle" '
        f'fill="#222">Normalized Mean Squared Error (dB²)</text>'
    )

    svg.append("</svg>")
    return "\n".join(svg)


def _target_fit_chart_svg(results: list[dict], width: int = 1593, height: int = 756) -> str:
    sorted_results = sorted(
        [
            result for result in results
            if result["baseline_theory_target_rmse_db"] is not None
            and result["optimized_theory_target_rmse_db"] is not None
            and result["validated_surrogate_target_rmse_db"] is not None
        ],
        key=lambda result: result["validated_surrogate_target_rmse_db"],
    )
    if not sorted_results:
        return ""

    margin_left = 175
    margin_right = 54
    margin_top = 108
    margin_bottom = 94
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    n = len(sorted_results)
    group_h = plot_h / max(n, 1)
    bar_h = max(6.0, min(18.0, group_h / 4.5))
    max_rmse = max(
        max(
            result["baseline_theory_target_rmse_db"],
            result["optimized_theory_target_rmse_db"],
            result["validated_surrogate_target_rmse_db"],
        )
        for result in sorted_results
    ) * 1.1
    mean_surrogate = sum(
        result["validated_surrogate_target_rmse_db"] for result in sorted_results
    ) / len(sorted_results)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        "<title>Target-Fit Improvement Across Campaign</title>",
        '<rect width="100%" height="100%" fill="#ffffff" />',
        '<text x="32" y="40" font-family="Arial, sans-serif" font-size="26" fill="#111">Target-Fit Error Before and After Mutation</text>',
        f'<text x="32" y="66" font-family="Arial, sans-serif" font-size="16" fill="#666">Three bars per run: baseline theory, optimized theory, and validated surrogate. Rows are sorted by validated surrogate RMSE. Mean validated RMSE = {mean_surrogate:.2f} dB</text>',
    ]

    for tick_index in range(6):
        value = max_rmse * tick_index / 5.0
        x = margin_left + plot_w * tick_index / 5.0
        svg.append(
            f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{height - margin_bottom}" stroke="#eeeeee" stroke-width="1" />'
        )
        svg.append(
            f'<text x="{x:.1f}" y="{height - margin_bottom + 18}" font-family="Arial, sans-serif" font-size="14" text-anchor="middle" fill="#666">{value:.1f}</text>'
        )

    for index, result in enumerate(sorted_results):
        y0 = margin_top + index * group_h
        label_y = y0 + group_h / 2 + 4
        bars = [
            ("baseline_theory_target_rmse_db", "#9aa0a6"),
            ("optimized_theory_target_rmse_db", "#1f77b4"),
            ("validated_surrogate_target_rmse_db", "#d62728"),
        ]
        svg.append(
            f'<text x="{margin_left - 10}" y="{label_y:.1f}" font-family="Arial, sans-serif" font-size="14" text-anchor="end" fill="#333"></text>'
        )
        for bar_index, (key, color) in enumerate(bars):
            value = result[key]
            y = y0 + 4 + bar_index * (bar_h + 2)
            w = (value / max_rmse) * plot_w
            svg.append(
                f'<rect x="{margin_left}" y="{y:.1f}" width="{w:.1f}" height="{bar_h:.1f}" fill="{color}" rx="2" opacity="0.9" />'
            )
    legend_y = height - 28
    legend = [
        ("#9aa0a6", "baseline theory"),
        ("#1f77b4", "optimized theory"),
        ("#d62728", "validated surrogate"),
    ]
    legend_x = 40
    for color, label in legend:
        svg.append(f'<rect x="{legend_x}" y="{legend_y - 10}" width="18" height="10" fill="{color}" rx="2" />')
        svg.append(f'<text x="{legend_x + 26}" y="{legend_y}" font-family="Arial, sans-serif" font-size="16" fill="#444">{label}</text>')
        legend_x += 170
    svg.append(
        f'<text x="{margin_left + plot_w / 2:.1f}" y="{height - 8}" font-family="Arial, sans-serif" font-size="18" text-anchor="middle" fill="#222">Peak-normalized target-shape RMSE (dB)</text>'
    )
    svg.append("</svg>")
    return "\n".join(svg)


def _mutation_effectiveness_svg(results: list[dict], width: int = 1377, height: int = 756) -> str:
    points = [
        result for result in results
        if result["mutation_magnitude_index"] is not None
        and result["surrogate_target_rmse_improvement_db"] is not None
    ]
    if not points:
        return ""

    margin_left = 121
    margin_right = 54
    margin_top = 108
    margin_bottom = 94
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    max_x = max(result["mutation_magnitude_index"] for result in points) * 1.1
    min_y = min(result["surrogate_target_rmse_improvement_db"] for result in points)
    max_y = max(result["surrogate_target_rmse_improvement_db"] for result in points)
    if abs(max_y - min_y) < 1.0e-9:
        max_y += 1.0
        min_y -= 1.0
    pad_y = 0.1 * (max_y - min_y)
    min_y -= pad_y
    max_y += pad_y

    def x_map(value: float) -> float:
        return margin_left + (value / max(max_x, 1.0e-9)) * plot_w

    def y_map(value: float) -> float:
        return margin_top + (max_y - value) / max(max_y - min_y, 1.0e-9) * plot_h

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        "<title>Mutation Effectiveness</title>",
        '<rect width="100%" height="100%" fill="#ffffff" />',
        '<text x="28" y="36" font-family="Arial, sans-serif" font-size="29" fill="#111">How Much Mutation Bought How Much Target-Fit Improvement</text>',
        '<text x="28" y="56" font-family="Arial, sans-serif" font-size="17" fill="#666">Each point is one target. Higher is better. Left-shifted high points indicate efficient geometry mutation.</text>',
    ]

    zero_y = y_map(0.0)
    svg.append(f'<line x1="{margin_left}" y1="{zero_y:.1f}" x2="{width - margin_right}" y2="{zero_y:.1f}" stroke="#cccccc" stroke-width="1.2" stroke-dasharray="4,4" />')
    for tick_index in range(6):
        x_value = max_x * tick_index / 5.0
        x = x_map(x_value)
        svg.append(f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{height - margin_bottom}" stroke="#eeeeee" stroke-width="1" />')
        svg.append(f'<text x="{x:.1f}" y="{height - margin_bottom + 18}" font-family="Arial, sans-serif" font-size="14" text-anchor="middle" fill="#666">{x_value:.1f}</text>')
    for tick_index in range(6):
        y_value = min_y + (max_y - min_y) * tick_index / 5.0
        y = y_map(y_value)
        svg.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#f3f3f3" stroke-width="1" />')
        svg.append(f'<text x="{margin_left - 10}" y="{y + 4:.1f}" font-family="Arial, sans-serif" font-size="14" text-anchor="end" fill="#666">{y_value:.1f}</text>')

    for result in points:
        x = x_map(result["mutation_magnitude_index"])
        y = y_map(result["surrogate_target_rmse_improvement_db"])
        fill = "#2ca02c" if result["surrogate_target_rmse_improvement_db"] >= 0.0 else "#e74c3c"
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{fill}" opacity="0.9" />')
        svg.append(f'<text x="{x + 8:.1f}" y="{y - 8:.1f}" font-family="Arial, sans-serif" font-size="13" fill="#444"></text>')

    svg.append(f'<text x="{margin_left + plot_w / 2:.1f}" y="{height - 8}" font-family="Arial, sans-serif" font-size="18" text-anchor="middle" fill="#222">Mutation magnitude index</text>')
    svg.append(f'<text x="22" y="{margin_top + plot_h / 2:.1f}" font-family="Arial, sans-serif" font-size="18" fill="#222" transform="rotate(-90 22 {margin_top + plot_h / 2:.1f})">Validated surrogate RMSE improvement vs baseline (dB)</text>')
    svg.append("</svg>")
    return "\n".join(svg)


def _aero_tradeoff_svg(results: list[dict], width: int = 1404, height: int = 756) -> str:
    points = [
        result for result in results
        if result["validated_surrogate_target_rmse_db"] is not None
        and result["drag_ratio"] is not None
        and result["lift_to_drag_retention"] is not None
    ]
    if not points:
        return ""

    margin_left = 121
    margin_right = 67
    margin_top = 108
    margin_bottom = 94
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    min_x = min(result["validated_surrogate_target_rmse_db"] for result in points)
    max_x = max(result["validated_surrogate_target_rmse_db"] for result in points)
    min_y = min(result["drag_ratio"] for result in points)
    max_y = max(result["drag_ratio"] for result in points)
    x_pad = max(0.3, 0.1 * max(max_x - min_x, 1.0))
    y_pad = max(0.05, 0.12 * max(max_y - min_y, 0.1))
    min_x -= x_pad
    max_x += x_pad
    min_y = max(0.0, min_y - y_pad)
    max_y += y_pad
    min_ld = min(result["lift_to_drag_retention"] for result in points)
    max_ld = max(result["lift_to_drag_retention"] for result in points)

    def x_map(value: float) -> float:
        return margin_left + (value - min_x) / max(max_x - min_x, 1.0e-9) * plot_w

    def y_map(value: float) -> float:
        return margin_top + (max_y - value) / max(max_y - min_y, 1.0e-9) * plot_h

    def color_for_ld(value: float) -> str:
        if max_ld - min_ld < 1.0e-9:
            return "#1f77b4"
        ratio = (value - min_ld) / (max_ld - min_ld)
        red = int(231 - 120 * ratio)
        green = int(76 + 110 * ratio)
        blue = int(60 + 120 * ratio)
        return f"#{red:02x}{green:02x}{blue:02x}"

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        "<title>Aerodynamic Tradeoff vs Target Fit</title>",
        '<rect width="100%" height="100%" fill="#ffffff" />',
        '<text x="28" y="36" font-family="Arial, sans-serif" font-size="29" fill="#111">Acoustic Fit vs Aerodynamic Cost</text>',
        '<text x="28" y="56" font-family="Arial, sans-serif" font-size="17" fill="#666">Each point is one target. Left and low is better. Color shows lift-to-drag retention relative to the baseline geometry.</text>',
    ]

    for tick_index in range(6):
        x_value = min_x + (max_x - min_x) * tick_index / 5.0
        x = x_map(x_value)
        svg.append(f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{height - margin_bottom}" stroke="#eeeeee" stroke-width="1" />')
        svg.append(f'<text x="{x:.1f}" y="{height - margin_bottom + 18}" font-family="Arial, sans-serif" font-size="14" text-anchor="middle" fill="#666">{x_value:.1f}</text>')
    for tick_index in range(6):
        y_value = min_y + (max_y - min_y) * tick_index / 5.0
        y = y_map(y_value)
        svg.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#f3f3f3" stroke-width="1" />')
        svg.append(f'<text x="{margin_left - 10}" y="{y + 4:.1f}" font-family="Arial, sans-serif" font-size="14" text-anchor="end" fill="#666">{y_value:.2f}</text>')

    svg.append(f'<line x1="{margin_left}" y1="{y_map(1.0):.1f}" x2="{width - margin_right}" y2="{y_map(1.0):.1f}" stroke="#999999" stroke-width="1.2" stroke-dasharray="4,4" />')
    for result in points:
        x = x_map(result["validated_surrogate_target_rmse_db"])
        y = y_map(result["drag_ratio"])
        fill = color_for_ld(result["lift_to_drag_retention"])
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{fill}" opacity="0.92" />')
        svg.append(f'<text x="{x + 8:.1f}" y="{y - 8:.1f}" font-family="Arial, sans-serif" font-size="13" fill="#444"></text>')

    svg.append(f'<text x="{margin_left + plot_w / 2:.1f}" y="{height - 8}" font-family="Arial, sans-serif" font-size="18" text-anchor="middle" fill="#222">Validated surrogate target-shape RMSE (dB)</text>')
    svg.append(f'<text x="22" y="{margin_top + plot_h / 2:.1f}" font-family="Arial, sans-serif" font-size="18" fill="#222" transform="rotate(-90 22 {margin_top + plot_h / 2:.1f})">Profile-drag proxy ratio vs baseline</text>')
    svg.append(f'<text x="{width - 240}" y="{height - 42}" font-family="Arial, sans-serif" font-size="14" fill="#666">reference line: drag ratio = 1.0</text>')
    svg.append("</svg>")
    return "\n".join(svg)


def _aero_retention_chart_svg(results: list[dict], width: int = 1593, height: int = 837) -> str:
    sorted_results = sorted(
        [
            result for result in results
            if result["lift_retention"] is not None
            and result["drag_ratio"] is not None
            and result["lift_to_drag_retention"] is not None
        ],
        key=lambda result: result["validated_surrogate_target_rmse_db"]
        if result["validated_surrogate_target_rmse_db"] is not None else 1.0e9,
    )
    if not sorted_results:
        return ""

    margin_left = 175
    margin_right = 54
    margin_top = 113
    margin_bottom = 99
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    n = len(sorted_results)
    group_h = plot_h / max(n, 1)
    bar_h = max(6.0, min(18.0, group_h / 4.5))
    max_value = max(
        max(result["lift_retention"], result["drag_ratio"], result["lift_to_drag_retention"])
        for result in sorted_results
    ) * 1.1
    min_value = min(
        min(result["lift_retention"], result["drag_ratio"], result["lift_to_drag_retention"])
        for result in sorted_results
    )
    min_value = min(min_value * 1.1, 0.0)

    def x_map(value: float) -> float:
        return margin_left + (value - min_value) / max(max_value - min_value, 1.0e-9) * plot_w

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        "<title>Aerodynamic Retention Relative to Baseline</title>",
        '<rect width="100%" height="100%" fill="#ffffff" />',
        '<text x="32" y="40" font-family="Arial, sans-serif" font-size="26" fill="#111">Aerodynamic Proxy Retention Relative to Baseline Geometry</text>',
        '<text x="32" y="66" font-family="Arial, sans-serif" font-size="16" fill="#666">Per run: signed lift retention, drag ratio, and lift-to-drag retention. Rows are sorted by validated surrogate RMSE.</text>',
    ]

    zero_x = x_map(0.0)
    ref_x = x_map(1.0)
    svg.append(f'<line x1="{zero_x:.1f}" y1="{margin_top}" x2="{zero_x:.1f}" y2="{height - margin_bottom}" stroke="#d0d0d0" stroke-width="1" />')
    svg.append(f'<line x1="{ref_x:.1f}" y1="{margin_top}" x2="{ref_x:.1f}" y2="{height - margin_bottom}" stroke="#999999" stroke-width="1.2" stroke-dasharray="4,4" />')
    for tick_index in range(6):
        value = min_value + (max_value - min_value) * tick_index / 5.0
        x = x_map(value)
        svg.append(f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{height - margin_bottom}" stroke="#eeeeee" stroke-width="1" />')
        svg.append(f'<text x="{x:.1f}" y="{height - margin_bottom + 18}" font-family="Arial, sans-serif" font-size="14" text-anchor="middle" fill="#666">{value:.2f}</text>')

    for index, result in enumerate(sorted_results):
        y0 = margin_top + index * group_h
        label_y = y0 + group_h / 2 + 4
        bars = [
            ("lift_retention", "#2ca02c"),
            ("drag_ratio", "#e67e22"),
            ("lift_to_drag_retention", "#1f77b4"),
        ]
        svg.append(f'<text x="{margin_left - 10}" y="{label_y:.1f}" font-family="Arial, sans-serif" font-size="14" text-anchor="end" fill="#333"></text>')
        for bar_index, (key, color) in enumerate(bars):
            value = result[key]
            y = y0 + 4 + bar_index * (bar_h + 2)
            x0 = x_map(min(0.0, value))
            x1 = x_map(max(0.0, value))
            svg.append(f'<rect x="{x0:.1f}" y="{y:.1f}" width="{(x1 - x0):.1f}" height="{bar_h:.1f}" fill="{color}" rx="2" opacity="0.9" />')
    legend_y = height - 30
    legend = [
        ("#2ca02c", "lift retention"),
        ("#e67e22", "drag ratio"),
        ("#1f77b4", "lift-to-drag retention"),
    ]
    legend_x = 40
    for color, label in legend:
        svg.append(f'<rect x="{legend_x}" y="{legend_y - 10}" width="18" height="10" fill="{color}" rx="2" />')
        svg.append(f'<text x="{legend_x + 26}" y="{legend_y}" font-family="Arial, sans-serif" font-size="16" fill="#444">{label}</text>')
        legend_x += 190
    svg.append(f'<text x="{margin_left + plot_w / 2:.1f}" y="{height - 8}" font-family="Arial, sans-serif" font-size="18" text-anchor="middle" fill="#222">Relative aero-proxy value vs baseline</text>')
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
            f'font-size="13" text-anchor="middle" fill="#999">no data</text>'
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
        f'font-size="14" text-anchor="middle" fill="#333"></text>'
    )
    # Label: MSE
    if mse is not None:
        svg.append(
            f'<text x="{cx:.1f}" y="{cy + radius + 14:.1f}" font-family="Arial, sans-serif" '
            f'font-size="13" text-anchor="middle" fill="#888">MSE {mse:.1f}</text>'
        )

    return svg


def _directivity_grid_svg(
    campaign_dir: Path,
    results: list[dict],
    width: int = 1620,
) -> str:
    """Generate a grid of small-multiple directivity polar plots."""
    results = [r for r in results if str(r.get("freedom_level", "full")) == "full"]
    n = len(results)
    if n == 0:
        return ""

    cols = min(10, n)
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
        f'<text x="28" y="36" font-family="Arial, sans-serif" font-size="29" '
        f'fill="#111">Directivity Achieved vs Target — {n} Arbitrary Shapes</text>',
        f'<text x="28" y="56" font-family="Arial, sans-serif" font-size="17" '
        f'fill="#666">Each cell: 1 kHz polar plot. '
        f'Red=simulated, Blue=theory, Green dashed=target. '
        f'Mean MSE={mean_mse:.1f} dB²</text>',
    ]

    for i, r in enumerate(results):
        col = i % cols
        row_idx = i // cols
        cx = col * cell_w + cell_w // 2
        cy = 90 + row_idx * cell_h + cell_h // 2

        target_dir = Path(r.get("target_path") or (campaign_dir / r["folder"]))
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
        f'font-size="16" fill="#444">simulated</text>'
    )
    svg.append(
        f'<line x1="160" y1="{legend_y}" x2="190" y2="{legend_y}" '
        f'stroke="#1f77b4" stroke-width="2.5" />'
        f'<text x="198" y="{legend_y + 4}" font-family="Arial, sans-serif" '
        f'font-size="16" fill="#444">theory</text>'
    )
    svg.append(
        f'<line x1="260" y1="{legend_y}" x2="290" y2="{legend_y}" '
        f'stroke="#2ca02c" stroke-width="2.5" stroke-dasharray="5,3" />'
        f'<text x="298" y="{legend_y + 4}" font-family="Arial, sans-serif" '
        f'font-size="16" fill="#444">target shape</text>'
    )

    svg.append("</svg>")
    return "\n".join(svg)


# ─────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────

def aggregate_campaign_dir(campaign_dir: Path) -> None:
    results = _collect_targets(campaign_dir)

    if not results:
        print("No target results found. Nothing to aggregate.")
        return

    print(f"Found {len(results)} target(s) in {campaign_dir}")

    # Write CSV
    csv_path = _write_summary_csv(campaign_dir, results)
    print(f"  Summary CSV: {csv_path}")

    freedom_summary = _freedom_summary(results)
    if len(freedom_summary) > 1:
        freedom_csv_path = _write_freedom_summary_csv(campaign_dir, freedom_summary)
        print(f"  Freedom CSV: {freedom_csv_path}")

        freedom_fit_svg = _freedom_fit_chart_svg(freedom_summary)
        if freedom_fit_svg:
            freedom_fit_path = campaign_dir / "campaign_freedom_fit_chart.svg"
            freedom_fit_path.write_text(freedom_fit_svg, encoding="utf-8")
            print(f"  Freedom fit: {freedom_fit_path}")

        freedom_tradeoff_svg = _freedom_tradeoff_chart_svg(freedom_summary)
        if freedom_tradeoff_svg:
            freedom_tradeoff_path = campaign_dir / "campaign_freedom_tradeoff_chart.svg"
            freedom_tradeoff_path.write_text(freedom_tradeoff_svg, encoding="utf-8")
            print(f"  Freedom aero:{freedom_tradeoff_path}")

    # MSE bar chart
    bar_svg = _mse_bar_chart_svg(results)
    if bar_svg:
        bar_path = campaign_dir / "campaign_mse_chart.svg"
        bar_path.write_text(bar_svg, encoding="utf-8")
        print(f"  MSE chart:   {bar_path}")

    target_fit_svg = _target_fit_chart_svg(results)
    if target_fit_svg:
        target_fit_path = campaign_dir / "campaign_target_fit_chart.svg"
        target_fit_path.write_text(target_fit_svg, encoding="utf-8")
        print(f"  Target fit:  {target_fit_path}")

    mutation_svg = _mutation_effectiveness_svg(results)
    if mutation_svg:
        mutation_path = campaign_dir / "campaign_mutation_effectiveness.svg"
        mutation_path.write_text(mutation_svg, encoding="utf-8")
        print(f"  Mutation:    {mutation_path}")

    aero_tradeoff_svg = _aero_tradeoff_svg(results)
    if aero_tradeoff_svg:
        aero_tradeoff_path = campaign_dir / "campaign_aero_tradeoff.svg"
        aero_tradeoff_path.write_text(aero_tradeoff_svg, encoding="utf-8")
        print(f"  Aero trade:  {aero_tradeoff_path}")

    aero_retention_svg = _aero_retention_chart_svg(results)
    if aero_retention_svg:
        aero_retention_path = campaign_dir / "campaign_aero_retention_chart.svg"
        aero_retention_path.write_text(aero_retention_svg, encoding="utf-8")
        print(f"  Aero bars:   {aero_retention_path}")

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
    surrogate_target_rmses = [
        r["validated_surrogate_target_rmse_db"]
        for r in results
        if r["validated_surrogate_target_rmse_db"] is not None
    ]
    baseline_target_rmses = [
        r["baseline_theory_target_rmse_db"]
        for r in results
        if r["baseline_theory_target_rmse_db"] is not None
    ]
    mutation_efficiencies = [
        r["rmse_improvement_per_mutation_index"]
        for r in results
        if r["rmse_improvement_per_mutation_index"] is not None
    ]
    if surrogate_target_rmses and baseline_target_rmses:
        print(f"    Mean baseline target RMSE:  {sum(baseline_target_rmses) / len(baseline_target_rmses):.2f} dB")
        print(f"    Mean validated target RMSE: {sum(surrogate_target_rmses) / len(surrogate_target_rmses):.2f} dB")
    if mutation_efficiencies:
        print(f"    Mean RMSE gain per mutation index: {sum(mutation_efficiencies) / len(mutation_efficiencies):.3f}")
    lift_retentions = [r["lift_retention"] for r in results if r["lift_retention"] is not None]
    drag_ratios = [r["drag_ratio"] for r in results if r["drag_ratio"] is not None]
    ld_retentions = [r["lift_to_drag_retention"] for r in results if r["lift_to_drag_retention"] is not None]
    if lift_retentions and drag_ratios and ld_retentions:
        print(f"    Mean lift retention:          {sum(lift_retentions) / len(lift_retentions):.3f}")
        print(f"    Mean drag ratio:              {sum(drag_ratios) / len(drag_ratios):.3f}")
        print(f"    Mean lift-to-drag retention:  {sum(ld_retentions) / len(ld_retentions):.3f}")


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
    aggregate_campaign_dir(args.campaign_dir)


if __name__ == "__main__":
    main()
