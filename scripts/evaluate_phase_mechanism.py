import sys
import os
import csv
import json
import torch
import argparse
import numpy as np
from pathlib import Path
from scipy import stats

# Add workspace src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase_tuned_feathering.geometry import WingGeometryParams, source_grid
from phase_tuned_feathering.observers import ObserverGrid
from phase_tuned_feathering.acoustics_torch import evaluate_spp_torch
from phase_tuned_feathering.closures import ClosureParams, FlowConfig

def calculate_angular_metrics(levels_exp, levels_zero):
    # Diff at each observer angle
    diff = levels_exp - levels_zero
    mean_diff = float(torch.mean(diff).item())
    std_diff = float(torch.std(diff).item())
    max_dev = float(torch.max(torch.abs(diff - mean_diff)).item())
    
    # Centered RMSE
    levels_exp_centered = levels_exp - torch.mean(levels_exp)
    levels_zero_centered = levels_zero - torch.mean(levels_zero)
    rmse_centered = float(torch.sqrt(torch.mean((levels_exp_centered - levels_zero_centered) ** 2)).item())
    
    # Pearson Correlation Coefficient
    var_exp = torch.sum(levels_exp_centered ** 2)
    var_zero = torch.sum(levels_zero_centered ** 2)
    cov = torch.sum(levels_exp_centered * levels_zero_centered)
    denom = torch.sqrt(var_exp * var_zero)
    correlation = float((cov / denom).item() if denom > 1.0e-12 else 1.0)
    
    return mean_diff, std_diff, max_dev, rmse_centered, correlation

def calculate_tost_equivalence(list_A, list_B, delta=0.25):
    diffs = [a - b for a, b in zip(list_A, list_B)]
    n = len(diffs)
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    sem = std_diff / math.sqrt(n)
    
    # 95% Confidence Interval
    ci = stats.t.interval(0.95, df=n-1, loc=mean_diff, scale=sem)
    
    # TOST Equivalence Test (H0: |mean_diff| >= delta)
    t1 = (mean_diff + delta) / sem
    t2 = (mean_diff - delta) / sem
    p1 = 1.0 - stats.t.cdf(t1, df=n-1)
    p2 = stats.t.cdf(t2, df=n-1)
    p_tost = max(p1, p2)
    
    return mean_diff, ci, p_tost

import math # Ensure math is imported inside evaluate_phase_mechanism.py

def main():
    parser = argparse.ArgumentParser(description="Evaluate phase mechanism and model equivalence.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="outputs/optimization/full",
        help="Input optimization directory containing target runs."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/phase_mechanism_summary.json",
        help="Path to save the summary JSON file."
    )
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[1]
    input_dir_path = root_dir / args.input_dir
    output_path = root_dir / args.output

    if not input_dir_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir_path}")

    target_dirs = sorted([d for d in input_dir_path.iterdir() if d.is_dir() and d.name.startswith("target_")])
    
    if not target_dirs:
        raise ValueError(f"No target directories (target_*) found in {input_dir_path}")

    observers = ObserverGrid.spherical(10, 15)  # Dense observer grid
    flow = FlowConfig()
    frequency = 1000.0  # 1 kHz
    
    target_mean_shifts = []
    target_std_shifts = []
    target_max_devs = []
    target_rmse_centered = []
    target_correlations = []
    
    for target_dir in target_dirs:
        geom_path = target_dir / "optimized_geometry.json"
        if not geom_path.exists():
            continue
            
        with open(geom_path, "r") as f:
            geom_dict = json.load(f)
            
        params_dict = geom_dict["params"]
        if "per_feather_incidence_deg" in params_dict and params_dict["per_feather_incidence_deg"] is not None:
            params_dict["per_feather_incidence_deg"] = tuple(params_dict["per_feather_incidence_deg"])
        if "sweep_section_fractions" in params_dict and params_dict["sweep_section_fractions"] is not None:
            params_dict["sweep_section_fractions"] = tuple(params_dict["sweep_section_fractions"])
            
        # Reconstruct geometry
        geom = WingGeometryParams(**params_dict)
        grid = source_grid(geom, n_eta=32)
        
        # Convert to PyTorch tensors
        points_t = torch.tensor(grid.points, dtype=torch.float32).unsqueeze(0)
        chords_t = torch.tensor(grid.chords, dtype=torch.float32).unsqueeze(0)
        incidence_deg_t = torch.tensor(grid.incidence_deg, dtype=torch.float32).unsqueeze(0)
        loading_directions_t = torch.tensor(grid.loading_directions, dtype=torch.float32).unsqueeze(0)
        weights_t = torch.tensor(grid.weights, dtype=torch.float32).unsqueeze(0)
        
        # Evaluate with exponential (partially coherent)
        closures_exp = ClosureParams(coherence_model="exponential")
        spp_exp = evaluate_spp_torch(
            points_t, chords_t, incidence_deg_t, loading_directions_t, weights_t,
            observers=observers, frequencies_hz=frequency, flow=flow, closures=closures_exp
        )
        levels_exp = 10.0 * torch.log10(spp_exp[0] / (flow.p_ref ** 2))
        
        # Evaluate with zero coherence (removes phase offsets)
        closures_zero = ClosureParams(coherence_model="zero")
        spp_zero = evaluate_spp_torch(
            points_t, chords_t, incidence_deg_t, loading_directions_t, weights_t,
            observers=observers, frequencies_hz=frequency, flow=flow, closures=closures_zero
        )
        levels_zero = 10.0 * torch.log10(spp_zero[0] / (flow.p_ref ** 2))
        
        # Calculate angular shape metrics
        mean_sh, std_sh, max_dev, rmse_c, corr = calculate_angular_metrics(levels_exp, levels_zero)
        target_mean_shifts.append(mean_sh)
        target_std_shifts.append(std_sh)
        target_max_devs.append(max_dev)
        target_rmse_centered.append(rmse_c)
        target_correlations.append(corr)

    # Load campaign summary CSV for model equivalence statistics
    campaign_csv_path = root_dir / "outputs" / "optimization" / "campaign_summary.csv"
    if not campaign_csv_path.exists():
        raise FileNotFoundError(f"Campaign summary file not found: {campaign_csv_path}")

    # Parse campaign summary
    # Group results by seed and freedom_level
    campaign_data = {}
    with open(campaign_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seed = int(row["seed"])
            level = row["freedom_level"]
            rmse = float(row["validated_surrogate_target_rmse_db"])
            campaign_data.setdefault(seed, {})[level] = rmse

    seeds = sorted(campaign_data.keys())
    full_rmses = [campaign_data[s]["full"] for s in seeds if "full" in campaign_data[s]]
    no_delay_rmses = [campaign_data[s]["no_delay"] for s in seeds if "no_delay" in campaign_data[s]]
    zero_coh_rmses = [campaign_data[s]["zero_coherence"] for s in seeds if "zero_coherence" in campaign_data[s]]

    # Paired differences and TOST equivalence
    eq_delta = 0.25 # Equivalence margin of 0.25 dB
    
    mean_d_delay, ci_delay, p_eq_delay = calculate_tost_equivalence(full_rmses, no_delay_rmses, eq_delta)
    mean_d_zero, ci_zero, p_eq_zero = calculate_tost_equivalence(full_rmses, zero_coh_rmses, eq_delta)

    results = {
        "coherence_ablation_angular_metrics": {
            "mean_spl_difference_db": float(np.mean(target_mean_shifts)),
            "mean_angular_std_difference_db": float(np.mean(target_std_shifts)),
            "mean_max_angular_deviation_db": float(np.mean(target_max_devs)),
            "mean_centered_rmse_db": float(np.mean(target_rmse_centered)),
            "mean_correlation": float(np.mean(target_correlations))
        },
        "model_equivalence": {
            "equivalence_margin_db": eq_delta,
            "full_vs_no_delay": {
                "mean_paired_difference_db": mean_d_delay,
                "ci_95_lower": ci_delay[0],
                "ci_95_upper": ci_delay[1],
                "tost_p_value": p_eq_delay,
                "is_equivalent": bool(p_eq_delay < 0.05)
            },
            "full_vs_zero_coherence": {
                "mean_paired_difference_db": mean_d_zero,
                "ci_95_lower": ci_zero[0],
                "ci_95_upper": ci_zero[1],
                "tost_p_value": p_eq_zero,
                "is_equivalent": bool(p_eq_zero < 0.05)
            }
        }
    }

    print("--- COHERENCE PHASE REMOVAL SHAPE METRICS (MEAN ACROSS TARGETS) ---")
    print(f"Mean SPL level shift:               {results['coherence_ablation_angular_metrics']['mean_spl_difference_db']:.3f} dB")
    print(f"Angular standard deviation of shift: {results['coherence_ablation_angular_metrics']['mean_angular_std_difference_db']:.3f} dB")
    print(f"Max angular deviation of shift:      {results['coherence_ablation_angular_metrics']['mean_max_angular_deviation_db']:.3f} dB")
    print(f"Centered shape RMSE:                 {results['coherence_ablation_angular_metrics']['mean_centered_rmse_db']:.3f} dB")
    print(f"Normalized directivity correlation:  {results['coherence_ablation_angular_metrics']['mean_correlation']:.5f}")

    print("\n--- MODEL EQUIVALENCE AND PAIRED CONTRASTS ---")
    print(f"Equivalence margin (delta): {eq_delta:.2f} dB")
    print(f"Full vs No Delay:       Mean Difference = {mean_d_delay:.3f} dB, 95% CI = [{ci_delay[0]:.3f}, {ci_delay[1]:.3f}] dB, TOST p-val = {p_eq_delay:.4f} (Equivalent: {results['model_equivalence']['full_vs_no_delay']['is_equivalent']})")
    print(f"Full vs Zero Coherence: Mean Difference = {mean_d_zero:.3f} dB, 95% CI = [{ci_zero[0]:.3f}, {ci_zero[1]:.3f}] dB, TOST p-val = {p_eq_zero:.4f} (Equivalent: {results['model_equivalence']['full_vs_zero_coherence']['is_equivalent']})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved summary results to: {output_path}")

if __name__ == "__main__":
    main()
