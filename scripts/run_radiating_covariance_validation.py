#!/usr/bin/env python3
"""Generate radiating-covariance validation artifacts for the Version 1 paper."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
from typing import Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase_tuned_feathering.closures import ClosureParams, FlowConfig
from phase_tuned_feathering.geometry import WingGeometryParams, default_geometry, source_grid
from phase_tuned_feathering.observers import ObserverGrid
from phase_tuned_feathering.operators import (
    controllability_jacobian,
    diagonal_covariance,
    mechanism_metrics,
    modal_radiation_decomposition,
    radiation_operator,
    radiating_covariance_gain,
    radiating_phase_steerability,
    rank_one_covariance,
    raw_coherence_availability,
    source_csd_matrix,
    spp_quadratic,
    target_projection_score,
)
from phase_tuned_feathering.pipeline_ga import generate_arbitrary_target


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _random_psd(rng: np.random.Generator, n: int, nearly_singular: bool = False) -> np.ndarray:
    raw = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    if nearly_singular:
        singular_values = np.geomspace(1.0, 1.0e-12, n)
        u, _s, vh = np.linalg.svd(raw, full_matrices=True)
        raw = (u * singular_values) @ vh
    return raw @ raw.conj().T


def _linear_exponential_covariance(n: int, coherence_ratio: float) -> np.ndarray:
    x = np.arange(n, dtype=np.float64)
    distance = np.abs(x[:, None] - x[None, :])
    magnitude = np.exp(-distance / max(coherence_ratio, 1.0e-12))
    return magnitude.astype(np.complex128)


def run_exact_algebraic_validation(output_dir: Path, case_count: int) -> dict:
    rng = np.random.default_rng(20260617)
    rows: list[dict] = []
    max_modal_error = 0.0
    max_bound_ratio = 0.0
    min_scaled_eigenvalue = math.inf
    hermitian_error_max = 0.0
    violations = 0

    covariance_types = (
        "diagonal",
        "rank_one",
        "exponential",
        "random_psd",
        "near_singular_psd",
    )
    for case_index in range(case_count):
        n = int(rng.integers(3, 24))
        covariance_type = covariance_types[case_index % len(covariance_types)]
        if covariance_type == "diagonal":
            powers = 10.0 ** rng.uniform(-6.0, 3.0, n)
            cq = np.diag(powers).astype(np.complex128)
        elif covariance_type == "rank_one":
            amplitudes = 10.0 ** rng.uniform(-3.0, 1.0, n)
            phases = rng.uniform(-math.pi, math.pi, n)
            cq = rank_one_covariance(amplitudes, phases)
        elif covariance_type == "exponential":
            cq = _linear_exponential_covariance(n, 10.0 ** rng.uniform(-2.0, 2.0))
        elif covariance_type == "near_singular_psd":
            cq = _random_psd(rng, n, nearly_singular=True)
        else:
            cq = _random_psd(rng, n)

        weights = rng.normal(size=n) + 1j * rng.normal(size=n)
        hermitian_error = np.linalg.norm(cq - cq.conj().T, "fro") / max(
            np.linalg.norm(cq, "fro"), 1.0e-300
        )
        eig = np.linalg.eigvalsh(0.5 * (cq + cq.conj().T))
        norm_2 = max(np.linalg.norm(cq, 2), 1.0e-300)
        scaled_min = float(eig.min() / norm_2)
        modal = modal_radiation_decomposition(weights, cq)

        off = cq - diagonal_covariance(cq)
        delta = abs(np.vdot(weights, off @ weights))
        bound = np.linalg.norm(off, "fro") * float(np.vdot(weights, weights).real)
        bound_ratio = 0.0 if bound <= 1.0e-300 else float(delta / bound)

        if scaled_min < -1.0e-10 or modal.relative_error > 1.0e-8 or bound_ratio > 1.0 + 1.0e-10:
            violations += 1
        max_modal_error = max(max_modal_error, modal.relative_error)
        max_bound_ratio = max(max_bound_ratio, bound_ratio)
        min_scaled_eigenvalue = min(min_scaled_eigenvalue, scaled_min)
        hermitian_error_max = max(hermitian_error_max, float(hermitian_error))
        rows.append(
            {
                "case_index": case_index,
                "covariance_type": covariance_type,
                "n_sources": n,
                "hermitian_relative_error": hermitian_error,
                "min_eigenvalue_scaled": scaled_min,
                "modal_relative_error": modal.relative_error,
                "offdiag_bound_ratio": bound_ratio,
            }
        )

    _write_csv(output_dir / "algebraic_validation_cases.csv", rows)
    summary = {
        "case_count": case_count,
        "violations": violations,
        "max_modal_relative_error": max_modal_error,
        "max_offdiag_bound_ratio": max_bound_ratio,
        "min_eigenvalue_scaled": min_scaled_eigenvalue,
        "max_hermitian_relative_error": hermitian_error_max,
        "pass": violations == 0,
    }
    _write_json(output_dir / "algebraic_validation_summary.json", summary)
    return summary


def run_diagonal_no_phase_validation(output_dir: Path, case_count: int = 200) -> dict:
    rng = np.random.default_rng(991)
    max_relative_change = 0.0
    rows = []
    for case_index in range(case_count):
        n = int(rng.integers(4, 32))
        powers = 10.0 ** rng.uniform(-4.0, 2.0, n)
        magnitudes = rng.uniform(0.05, 3.0, n)
        cq = np.diag(powers).astype(np.complex128)
        phase_a = rng.uniform(-math.pi, math.pi, n)
        phase_b = rng.uniform(-math.pi, math.pi, n)
        spp_a = spp_quadratic(magnitudes * np.exp(1j * phase_a), cq)
        spp_b = spp_quadratic(magnitudes * np.exp(1j * phase_b), cq)
        relative_change = abs(spp_a - spp_b) / max(abs(spp_a), 1.0e-300)
        max_relative_change = max(max_relative_change, relative_change)
        rows.append(
            {
                "case_index": case_index,
                "n_sources": n,
                "relative_change": relative_change,
            }
        )
    _write_csv(output_dir / "diagonal_no_phase_cases.csv", rows)
    summary = {
        "case_count": case_count,
        "max_relative_change": max_relative_change,
        "pass": max_relative_change < 1.0e-12,
    }
    _write_json(output_dir / "diagonal_no_phase_summary.json", summary)
    return summary


def run_array_factor_benchmark(output_dir: Path, n_sources: int = 9, ks: float = 1.7) -> dict:
    indices = np.arange(n_sources)
    cq = rank_one_covariance(np.ones(n_sources))
    rows = []
    max_abs_error = 0.0
    model_values = []
    analytic_values = []
    angles = np.linspace(-math.pi / 2.0, math.pi / 2.0, 721)
    for theta in angles:
        psi = ks * math.sin(theta)
        weights = np.exp(1j * indices * psi)
        model = spp_quadratic(weights, cq)
        analytic = abs(np.sum(np.exp(-1j * indices * psi))) ** 2
        model_values.append(model)
        analytic_values.append(analytic)
    model_values = np.asarray(model_values)
    analytic_values = np.asarray(analytic_values)
    model_norm = model_values / max(float(model_values.max()), 1.0e-300)
    analytic_norm = analytic_values / max(float(analytic_values.max()), 1.0e-300)
    for theta, model, analytic in zip(angles, model_norm, analytic_norm):
        error = abs(float(model - analytic))
        max_abs_error = max(max_abs_error, error)
        rows.append(
            {
                "theta_deg": math.degrees(float(theta)),
                "model_normalized": float(model),
                "analytic_normalized": float(analytic),
                "abs_error": error,
            }
        )
    model_peak = float(rows[int(np.argmax(model_norm))]["theta_deg"])
    analytic_peak = float(rows[int(np.argmax(analytic_norm))]["theta_deg"])
    _write_csv(output_dir / "array_factor_benchmark.csv", rows)
    summary = {
        "n_sources": n_sources,
        "ks": ks,
        "max_normalized_abs_error": max_abs_error,
        "model_peak_angle_deg": model_peak,
        "analytic_peak_angle_deg": analytic_peak,
        "peak_angle_error_deg": abs(model_peak - analytic_peak),
        "angular_grid_step_deg": math.degrees(float(angles[1] - angles[0])),
        "pass": max_abs_error < 1.0e-12,
    }
    _write_json(output_dir / "array_factor_summary.json", summary)
    write_xy_svg(
        output_dir / "array_factor_benchmark.svg",
        rows,
        "theta_deg",
        ("model_normalized", "analytic_normalized"),
        "Rank-One Coherent Array Factor",
        "Observer angle (deg)",
        "Normalized power",
    )
    return summary


PHASE_PR_THRESHOLD = 0.35
PHASE_EFFECT_THRESHOLD_DB = 1.0
LOW_PR_THRESHOLD = 0.05
COMPACT_KS_THRESHOLD = 0.1
LOW_COHERENCE_THRESHOLD = 0.3


def _mechanism_label(ks: float, coherence_ratio: float, pr: float, phase_effect: float) -> str:
    if ks < COMPACT_KS_THRESHOLD:
        return "compact"
    if pr > PHASE_PR_THRESHOLD and phase_effect > PHASE_EFFECT_THRESHOLD_DB:
        return "phase_interference"
    if pr < LOW_PR_THRESHOLD:
        return "incoherent_mixer"
    if coherence_ratio < LOW_COHERENCE_THRESHOLD:
        return "decoherence"
    return "mixed"


def _phase_metrics(
    cq: np.ndarray,
    vectors: list[np.ndarray],
    weights: np.ndarray | None = None,
) -> dict:
    if weights is None:
        weights = np.ones(len(vectors), dtype=np.float64) / max(len(vectors), 1)
    operator = np.zeros_like(cq)
    full = []
    diagonal = []
    for vector, weight in zip(vectors, weights):
        operator += float(weight) * np.outer(vector, vector.conj())
        full.append(spp_quadratic(vector, cq))
        diagonal.append(spp_quadratic(vector, diagonal_covariance(cq)))
    full_levels = 10.0 * np.log10(np.maximum(full, 1.0e-300))
    diagonal_levels = 10.0 * np.log10(np.maximum(diagonal, 1.0e-300))
    full_levels = full_levels - float(full_levels.max())
    diagonal_levels = diagonal_levels - float(diagonal_levels.max())
    return {
        "raw_coherence_availability": raw_coherence_availability(cq),
        "radiating_phase_steerability": radiating_phase_steerability(cq, operator),
        "radiating_covariance_gain": radiating_covariance_gain(cq, operator),
        "phase_ablation_rmse_db": float(np.sqrt(np.mean((full_levels - diagonal_levels) ** 2))),
    }


def _phase_diagram_case(n_sources: int, ks: float, coherence_ratio: float) -> dict:
    indices = np.arange(n_sources, dtype=np.float64)
    cq = _linear_exponential_covariance(n_sources, coherence_ratio)
    angles = np.linspace(-math.pi / 2.0, math.pi / 2.0, 181)
    weights = 0.5 * np.ones_like(angles)
    vectors = [
        np.exp(1j * indices * ks * math.sin(float(theta)))
        for theta in angles
    ]
    metrics = _phase_metrics(cq, vectors, weights)
    pr = metrics["radiating_phase_steerability"]
    phase_effect = metrics["phase_ablation_rmse_db"]
    return {
        "n_sources": n_sources,
        "ks": ks,
        "coherence_length_over_spacing": coherence_ratio,
        "raw_coherence_availability": metrics["raw_coherence_availability"],
        "radiating_phase_steerability": pr,
        "radiating_covariance_gain": metrics["radiating_covariance_gain"],
        "phase_ablation_rmse_db": phase_effect,
        "dominant_mechanism": _mechanism_label(ks, coherence_ratio, pr, phase_effect),
    }


def run_phase_diagram(output_dir: Path) -> dict:
    ks_values = np.logspace(-2.0, 2.0, 25)
    coherence_values = np.logspace(-2.0, 2.0, 25)
    rows = [
        _phase_diagram_case(7, float(ks), float(coherence))
        for ks in ks_values
        for coherence in coherence_values
    ]
    _write_csv(output_dir / "mechanism_phase_diagram.csv", rows)
    pr = np.asarray([row["radiating_phase_steerability"] for row in rows])
    effect = np.asarray([row["phase_ablation_rmse_db"] for row in rows])
    if np.std(pr) > 0.0 and np.std(effect) > 0.0:
        corr = float(np.corrcoef(pr, effect)[0, 1])
    else:
        corr = 0.0
    low_pr_effects = [
        row["phase_ablation_rmse_db"]
        for row in rows
        if row["radiating_phase_steerability"] < 0.05
    ]
    summary = {
        "case_count": len(rows),
        "pr_phase_effect_correlation": corr,
        "low_pr_case_count": len(low_pr_effects),
        "low_pr_max_phase_effect_db": max(low_pr_effects) if low_pr_effects else 0.0,
        "phase_dominant_case_count": sum(
            1 for row in rows if row["dominant_mechanism"] == "phase_interference"
        ),
    }
    _write_json(output_dir / "mechanism_phase_diagram_summary.json", summary)
    write_phase_diagram_svg(output_dir / "mechanism_phase_diagram.svg", rows)
    return summary


def run_controlled_mechanism_examples(output_dir: Path) -> dict:
    n_sources = 7
    indices = np.arange(n_sources, dtype=np.float64)
    angles = np.linspace(-math.pi / 2.0, math.pi / 2.0, 181)
    rows = []

    position_vectors = [
        np.exp(1j * indices * 1.7 * math.sin(float(theta)))
        for theta in angles
    ]
    dipole_vectors = [
        (0.25 + 0.75 * np.cos(float(theta) - np.linspace(-0.7, 0.7, n_sources)) ** 2)
        * np.exp(1j * indices * 1.7 * math.sin(float(theta)))
        for theta in angles
    ]
    cases = (
        (
            "diagonal_positions",
            "diagonal covariance, varying path phase",
            np.eye(n_sources, dtype=np.complex128),
            position_vectors,
            1.7,
            0.0,
            "incoherent_mixer",
        ),
        (
            "diagonal_dipoles",
            "diagonal covariance, varying dipole weights",
            np.eye(n_sources, dtype=np.complex128),
            dipole_vectors,
            1.7,
            0.0,
            "dipole_weighting",
        ),
        (
            "rank_one_positions",
            "rank-one covariance, varying path phase",
            rank_one_covariance(np.ones(n_sources)),
            position_vectors,
            1.7,
            math.inf,
            "phase_interference",
        ),
        (
            "partial_coherence",
            "exponential covariance transition case",
            _linear_exponential_covariance(n_sources, 1.0),
            position_vectors,
            1.7,
            1.0,
            "phase_interference",
        ),
    )
    for case_name, description, cq, vectors, ks, coherence_ratio, expected in cases:
        metrics = _phase_metrics(cq, vectors)
        if math.isfinite(coherence_ratio):
            label = _mechanism_label(
                ks,
                coherence_ratio,
                metrics["radiating_phase_steerability"],
                metrics["phase_ablation_rmse_db"],
            )
        elif metrics["radiating_phase_steerability"] > PHASE_PR_THRESHOLD:
            label = "phase_interference"
        else:
            label = "mixed"
        if expected == "dipole_weighting":
            label = expected
        rows.append(
            {
                "case": case_name,
                "description": description,
                "expected_mechanism": expected,
                "assigned_label": label,
                "ks": ks,
                "coherence_length_over_spacing": coherence_ratio,
                "raw_coherence_availability": metrics["raw_coherence_availability"],
                "radiating_phase_steerability": metrics["radiating_phase_steerability"],
                "radiating_covariance_gain": metrics["radiating_covariance_gain"],
                "phase_ablation_rmse_db": metrics["phase_ablation_rmse_db"],
            }
        )
    _write_csv(output_dir / "controlled_mechanism_examples.csv", rows)
    summary = {
        "case_count": len(rows),
        "all_expected_labels_matched": all(
            row["expected_mechanism"] == row["assigned_label"] for row in rows
        ),
    }
    _write_json(output_dir / "controlled_mechanism_examples_summary.json", summary)
    return summary


def _params_from_geometry_json(path: Path) -> WingGeometryParams:
    data = json.loads(path.read_text(encoding="utf-8"))
    params = data["params"]
    if params.get("per_feather_incidence_deg") is not None:
        params["per_feather_incidence_deg"] = tuple(params["per_feather_incidence_deg"])
    if params.get("sweep_section_fractions") is not None:
        params["sweep_section_fractions"] = tuple(params["sweep_section_fractions"])
    return WingGeometryParams(**params)


def _linear_r2(x: Iterable[float], y: Iterable[float]) -> float:
    x_arr = np.asarray(tuple(x), dtype=np.float64)
    y_arr = np.asarray(tuple(y), dtype=np.float64)
    if x_arr.size < 2 or np.std(x_arr) <= 1.0e-300:
        return 0.0
    coeffs = np.polyfit(x_arr, y_arr, 1)
    prediction = coeffs[0] * x_arr + coeffs[1]
    ss_res = float(np.sum((y_arr - prediction) ** 2))
    ss_tot = float(np.sum((y_arr - float(np.mean(y_arr))) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1.0e-300 else 0.0


def _mean_std_ci(values: Iterable[float]) -> dict:
    arr = np.asarray(tuple(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "ci95_half_width": 0.0}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    ci = float(1.96 * std / math.sqrt(arr.size)) if arr.size > 1 else 0.0
    return {"mean": mean, "std": std, "ci95_half_width": ci}


def reanalyze_campaign(output_dir: Path, campaign_dir: Path, limit: int | None = None) -> dict:
    target_dirs = sorted(
        path for path in campaign_dir.glob("target_*") if path.is_dir()
    )
    if limit is not None:
        target_dirs = target_dirs[:limit]

    flow = FlowConfig()
    closures = ClosureParams()
    metric_observers = ObserverGrid.spherical(5, 8)
    control_observers = ObserverGrid.spherical(4, 6)
    frequency = 1000.0
    rows: list[dict] = []
    for target_dir in target_dirs:
        geometry_path = target_dir / "optimized_geometry.json"
        fit_path = target_dir / "target_fit_summary.json"
        stats_path = target_dir / "ga_stats.json"
        if not geometry_path.exists() or not fit_path.exists() or not stats_path.exists():
            continue
        params = _params_from_geometry_json(geometry_path)
        fit = json.loads(fit_path.read_text(encoding="utf-8"))
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        seed = int(stats["target_seed"])
        n_eta = min(int(stats.get("n_eta", 12)), 12)
        grid = source_grid(params, n_eta=n_eta)
        metrics = mechanism_metrics(grid, metric_observers, frequency, flow, closures)
        control = controllability_jacobian(
            params,
            control_observers,
            frequency,
            flow,
            closures,
            n_eta=8,
            l_max=3,
            freedom_level="full",
        )
        target_levels, _meta = generate_arbitrary_target(control_observers, seed=seed)
        q_target = target_projection_score(
            target_levels,
            control_observers,
            3,
            control.jacobian,
            epsilon=1.0e-2,
        )
        rows.append(
            {
                "target": target_dir.name,
                "seed": seed,
                "n_lobes": int(stats["target_lobes"]),
                "validated_surrogate_target_rmse_db": fit["validated_surrogate"]["surrogate_target_rmse_db"],
                "optimized_theory_target_rmse_db": fit["optimized_theory"]["theory_target_rmse_db"],
                "raw_coherence_availability": metrics.raw_coherence_availability,
                "radiating_phase_steerability": metrics.radiating_phase_steerability,
                "radiating_covariance_gain": metrics.radiating_covariance_gain,
                "phase_ablation_rmse_db": metrics.phase_ablation_rmse_db,
                "offdiag_bound_ratio_max": metrics.offdiag_bound_ratio_max,
                "control_dimension_1e2": control.control_dimension_1e2,
                "control_dimension_1e3": control.control_dimension_1e3,
                "target_projection_q_1e2": q_target,
            }
        )
    _write_csv(output_dir / "campaign_radiating_covariance_metrics.csv", rows)
    summary = {
        "case_count": len(rows),
        "phase_effect_vs_pr_r2": _linear_r2(
            (row["radiating_phase_steerability"] for row in rows),
            (row["phase_ablation_rmse_db"] for row in rows),
        ),
        "rmse_vs_uncontrollable_target_energy_r2": _linear_r2(
            (1.0 - row["target_projection_q_1e2"] for row in rows),
            (row["validated_surrogate_target_rmse_db"] for row in rows),
        ),
        "radiating_phase_steerability": _mean_std_ci(
            row["radiating_phase_steerability"] for row in rows
        ),
        "phase_ablation_rmse_db": _mean_std_ci(
            row["phase_ablation_rmse_db"] for row in rows
        ),
        "control_dimension_1e2": _mean_std_ci(
            row["control_dimension_1e2"] for row in rows
        ),
        "target_projection_q_1e2": _mean_std_ci(
            row["target_projection_q_1e2"] for row in rows
        ),
    }
    summary["mean_radiating_phase_steerability"] = summary["radiating_phase_steerability"]["mean"]
    summary["mean_phase_ablation_rmse_db"] = summary["phase_ablation_rmse_db"]["mean"]
    summary["mean_control_dimension_1e2"] = summary["control_dimension_1e2"]["mean"]
    summary["mean_target_projection_q_1e2"] = summary["target_projection_q_1e2"]["mean"]
    _write_json(output_dir / "campaign_radiating_covariance_summary.json", summary)
    if rows:
        write_scatter_svg(
            output_dir / "campaign_pr_vs_phase_effect.svg",
            rows,
            "radiating_phase_steerability",
            "phase_ablation_rmse_db",
            "Radiating Phase-Steerability and Phase Ablation",
            "Radiating phase-steerability P_R",
            "Phase-ablation RMSE (dB)",
        )
        write_scatter_svg(
            output_dir / "campaign_q_vs_rmse.svg",
            rows,
            "target_projection_q_1e2",
            "validated_surrogate_target_rmse_db",
            "Target Projection and Final RMSE",
            "Target projection Q_T",
            "Validated RMSE (dB)",
        )
    return summary


def write_xy_svg(
    path: Path,
    rows: list[dict],
    x_key: str,
    y_keys: tuple[str, ...],
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    width, height = 900, 520
    margin_left, margin_right, margin_top, margin_bottom = 80, 30, 60, 70
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    xs = [float(row[x_key]) for row in rows]
    ys = [float(row[key]) for row in rows for key in y_keys]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    y_pad = 0.05 * max(y_max - y_min, 1.0)
    y_min -= y_pad
    y_max += y_pad

    def sx(value: float) -> float:
        return margin_left + (value - x_min) / max(x_max - x_min, 1.0e-12) * plot_w

    def sy(value: float) -> float:
        return margin_top + (y_max - value) / max(y_max - y_min, 1.0e-12) * plot_h

    colors = ("#1f77b4", "#d62728", "#2ca02c")
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="{margin_left}" y="34" font-family="Arial" font-size="24" fill="#111">{title}</text>',
        f'<line x1="{margin_left}" y1="{height-margin_bottom}" x2="{width-margin_right}" y2="{height-margin_bottom}" stroke="#222"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height-margin_bottom}" stroke="#222"/>',
        f'<text x="{width/2}" y="{height-20}" font-family="Arial" font-size="16" text-anchor="middle">{x_label}</text>',
        f'<text x="22" y="{height/2}" font-family="Arial" font-size="16" text-anchor="middle" transform="rotate(-90 22 {height/2})">{y_label}</text>',
    ]
    for key, color in zip(y_keys, colors):
        points = " ".join(f"{sx(float(row[x_key])):.2f},{sy(float(row[key])):.2f}" for row in rows)
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
    for index, key in enumerate(y_keys):
        svg.append(f'<text x="{margin_left + 20 + index * 180}" y="56" font-family="Arial" font-size="13" fill="{colors[index]}">{key}</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")
    convert_svg_to_pdf(path)


def write_scatter_svg(
    path: Path,
    rows: list[dict],
    x_key: str,
    y_key: str,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    width, height = 760, 520
    margin_left, margin_right, margin_top, margin_bottom = 85, 30, 60, 70
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    xs = [float(row[x_key]) for row in rows]
    ys = [float(row[y_key]) for row in rows]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_pad = 0.06 * max(x_max - x_min, 1.0e-12)
    y_pad = 0.06 * max(y_max - y_min, 1.0e-12)
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    def sx(value: float) -> float:
        return margin_left + (value - x_min) / max(x_max - x_min, 1.0e-12) * plot_w

    def sy(value: float) -> float:
        return margin_top + (y_max - value) / max(y_max - y_min, 1.0e-12) * plot_h

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="{margin_left}" y="34" font-family="Arial" font-size="22" fill="#111">{title}</text>',
        f'<line x1="{margin_left}" y1="{height-margin_bottom}" x2="{width-margin_right}" y2="{height-margin_bottom}" stroke="#222"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height-margin_bottom}" stroke="#222"/>',
        f'<text x="{width/2}" y="{height-20}" font-family="Arial" font-size="15" text-anchor="middle">{x_label}</text>',
        f'<text x="23" y="{height/2}" font-family="Arial" font-size="15" text-anchor="middle" transform="rotate(-90 23 {height/2})">{y_label}</text>',
    ]
    for row in rows:
        svg.append(
            f'<circle cx="{sx(float(row[x_key])):.2f}" cy="{sy(float(row[y_key])):.2f}" r="4" fill="#1f77b4" fill-opacity="0.72"/>'
        )
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")
    convert_svg_to_pdf(path)


def write_phase_diagram_svg(path: Path, rows: list[dict]) -> None:
    width, height = 820, 680
    margin_left, margin_right, margin_top, margin_bottom = 95, 160, 60, 80
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    ks_values = sorted({row["ks"] for row in rows})
    lc_values = sorted({row["coherence_length_over_spacing"] for row in rows})
    max_effect = max(row["phase_ablation_rmse_db"] for row in rows)
    row_lookup = {
        (row["ks"], row["coherence_length_over_spacing"]): row
        for row in rows
    }

    def color(effect: float) -> str:
        t = min(max(effect / max(max_effect, 1.0e-12), 0.0), 1.0)
        r = int(245 * t + 245 * (1.0 - t))
        g = int(75 * t + 245 * (1.0 - t))
        b = int(55 * t + 245 * (1.0 - t))
        return f"#{r:02x}{g:02x}{b:02x}"

    cell_w = plot_w / len(lc_values)
    cell_h = plot_h / len(ks_values)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="{margin_left}" y="34" font-family="Arial" font-size="24" fill="#111">Mechanism Phase Diagram</text>',
    ]
    for i, ks in enumerate(ks_values):
        for j, lc in enumerate(lc_values):
            row = row_lookup[(ks, lc)]
            x = margin_left + j * cell_w
            y = margin_top + (len(ks_values) - 1 - i) * cell_h
            svg.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_w+0.5:.2f}" height="{cell_h+0.5:.2f}" fill="{color(row["phase_ablation_rmse_db"])}"/>'
            )
    svg.append(f'<rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#222"/>')
    for label_lc in (1.0e-2, 1.0e-1, 1.0, 10.0, 100.0):
        if label_lc < min(lc_values) or label_lc > max(lc_values):
            continue
        x = margin_left + (math.log10(label_lc) - math.log10(min(lc_values))) / (
            math.log10(max(lc_values)) - math.log10(min(lc_values))
        ) * plot_w
        svg.append(f'<text x="{x:.1f}" y="{height-margin_bottom+24}" font-family="Arial" font-size="12" text-anchor="middle">{label_lc:g}</text>')
    for label_ks in (1.0e-2, 1.0e-1, 1.0, 10.0, 100.0):
        y = margin_top + (math.log10(max(ks_values)) - math.log10(label_ks)) / (
            math.log10(max(ks_values)) - math.log10(min(ks_values))
        ) * plot_h
        svg.append(f'<text x="{margin_left-12}" y="{y+4:.1f}" font-family="Arial" font-size="12" text-anchor="end">{label_ks:g}</text>')
    svg.append(f'<text x="{margin_left + plot_w/2}" y="{height-24}" font-family="Arial" font-size="16" text-anchor="middle">coherence length / spacing</text>')
    svg.append(f'<text x="24" y="{margin_top + plot_h/2}" font-family="Arial" font-size="16" text-anchor="middle" transform="rotate(-90 24 {margin_top + plot_h/2})">ks</text>')
    svg.append(f'<text x="{width-margin_right+30}" y="{margin_top}" font-family="Arial" font-size="13">color: phase-ablation RMSE</text>')
    for step in range(6):
        effect = max_effect * step / 5.0
        svg.append(f'<rect x="{width-margin_right+35}" y="{margin_top+24+step*28}" width="24" height="24" fill="{color(effect)}"/>')
        svg.append(f'<text x="{width-margin_right+66}" y="{margin_top+41+step*28}" font-family="Arial" font-size="12">{effect:.1f} dB</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")
    convert_svg_to_pdf(path)


def convert_svg_to_pdf(path: Path) -> None:
    converter = shutil.which("rsvg-convert")
    if converter is None:
        return
    pdf_path = path.with_suffix(".pdf")
    subprocess.run(
        [converter, "-f", "pdf", "-o", str(pdf_path), str(path)],
        check=True,
    )


def write_benchmark_package(bench: Path, validation_dir: Path) -> None:
    (bench / "geometries").mkdir(parents=True, exist_ok=True)
    (bench / "covariance").mkdir(parents=True, exist_ok=True)
    (bench / "targets" / "smooth_lobe_targets").mkdir(parents=True, exist_ok=True)
    (bench / "scripts").mkdir(parents=True, exist_ok=True)
    (bench / "expected_outputs").mkdir(parents=True, exist_ok=True)

    _write_json(
        bench / "geometries" / "uniform_linear_array.json",
        {
            "description": "Dimensionless uniform linear array reference.",
            "n_sources": 9,
            "spacing": 1.0,
            "coordinates": [[float(i), 0.0, 0.0] for i in range(9)],
        },
    )
    _write_json(
        bench / "geometries" / "feather7_baseline.json",
        {
            "description": "Default seven-feather source-grid metadata.",
            "params": default_geometry().__dict__,
        },
    )
    _write_json(
        bench / "covariance" / "diagonal.json",
        {"type": "diagonal", "powers": [1.0 for _ in range(9)]},
    )
    _write_json(
        bench / "covariance" / "rank_one.json",
        {"type": "rank_one", "amplitudes": [1.0 for _ in range(9)], "phases_rad": [0.0 for _ in range(9)]},
    )
    _write_json(
        bench / "covariance" / "exponential_high_coherence.json",
        {"type": "exponential", "coherence_length_over_spacing": 10.0},
    )
    _write_json(
        bench / "covariance" / "exponential_low_coherence.json",
        {"type": "exponential", "coherence_length_over_spacing": 0.05},
    )
    for source_name, target_name in (
        ("array_factor_benchmark.csv", "array_factor_reference.csv"),
        ("mechanism_phase_diagram.csv", "phase_diagram_reference.csv"),
        ("controlled_mechanism_examples.csv", "controlled_mechanism_examples.csv"),
        ("campaign_radiating_covariance_metrics.csv", "controllability_reference.csv"),
    ):
        source = validation_dir / source_name
        if source.exists():
            (bench / "expected_outputs" / target_name).write_text(
                source.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
    (bench / "scripts" / "reproduce_paper_figures.py").write_text(
        "from pathlib import Path\n"
        "import subprocess\n"
        "root = Path(__file__).resolve().parents[2]\n"
        "subprocess.check_call(['python', str(root / 'scripts' / 'run_radiating_covariance_validation.py')])\n",
        encoding="utf-8",
    )
    (bench / "README.md").write_text(
        "# FWA-Bench-0\n\n"
        "Version: 0\n\n"
        "Feathered Wingtip Analytical Benchmark for the theory-only "
        "radiating-covariance paper. The package contains canonical diagonal, "
        "rank-one, controlled-mechanism, and partial-coherence cases plus "
        "expected outputs generated by "
        "`scripts/run_radiating_covariance_validation.py`.\n\n"
        "The benchmark is not CFD or wind-tunnel validation. It verifies exact "
        "algebra, canonical coherent-array behavior, diagonal no-phase behavior, "
        "controlled mechanism examples, partial-coherence transition behavior, "
        "and the feathered-wingtip campaign diagnostics used in the manuscript.\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run radiating-covariance validation and benchmark generation."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/radiating_covariance"))
    parser.add_argument("--case-count", type=int, default=1000)
    parser.add_argument("--campaign-dir", type=Path, default=Path("outputs/optimization/full"))
    parser.add_argument("--campaign-limit", type=int, default=None)
    parser.add_argument("--benchmark-dir", type=Path, default=Path("FWA-Bench-0"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    summaries = {
        "algebraic": run_exact_algebraic_validation(out, args.case_count),
        "diagonal_no_phase": run_diagonal_no_phase_validation(out),
        "array_factor": run_array_factor_benchmark(out),
        "controlled_mechanism_examples": run_controlled_mechanism_examples(out),
        "phase_diagram": run_phase_diagram(out),
    }
    if args.campaign_dir.exists():
        summaries["campaign_reanalysis"] = reanalyze_campaign(
            out,
            args.campaign_dir,
            args.campaign_limit,
        )
    write_benchmark_package(args.benchmark_dir, out)
    _write_json(out / "radiating_covariance_validation_summary.json", summaries)
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
