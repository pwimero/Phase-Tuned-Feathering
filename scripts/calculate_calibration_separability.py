import os
import csv
import json
import math
import argparse
from pathlib import Path
import numpy as np

def weighted_stats(errors, weights):
    if not errors:
        return 0.0, 0.0, 0.0
    total_weight = sum(weights)
    if total_weight <= 0:
        return 0.0, 0.0, 0.0
    bias = sum(e * w for e, w in zip(errors, weights)) / total_weight
    mae = sum(abs(e) * w for e, w in zip(errors, weights)) / total_weight
    rmse = math.sqrt(sum((e ** 2) * w for e, w in zip(errors, weights)) / total_weight)
    return bias, mae, rmse

def calculate_centered_metrics(sim_levels, th_levels, weights):
    total_w = sum(weights)
    if total_w <= 0:
        return 0.0, 1.0
        
    mean_sim = sum(s * w for s, w in zip(sim_levels, weights)) / total_w
    mean_th = sum(t * w for t, w in zip(th_levels, weights)) / total_w
    
    sim_centered = [s - mean_sim for s in sim_levels]
    th_centered = [t - mean_th for t in th_levels]
    
    # RMSE of centered levels
    errors_centered = [t_c - s_c for t_c, s_c in zip(th_centered, sim_centered)]
    rmse_shape = math.sqrt(sum((e ** 2) * w for e, w in zip(errors_centered, weights)) / total_w)
    
    # Pearson Correlation Coefficient
    var_sim = sum((s_c ** 2) * w for s_c in sim_centered for w in [weights[sim_centered.index(s_c)]])
    var_th = sum((t_c ** 2) * w for t_c in th_centered for w in [weights[th_centered.index(t_c)]])
    
    cov = sum(s_c * t_c * w for s_c, t_c, w in zip(sim_centered, th_centered, weights))
    
    denom = math.sqrt(var_sim * var_th)
    correlation = cov / denom if denom > 1.0e-12 else 1.0
    
    return rmse_shape, correlation

def main():
    parser = argparse.ArgumentParser(description="Calculate calibration separability metrics.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="outputs/optimization/full",
        help="Input optimization directory containing target runs."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/calibration_separability_summary.json",
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

    cal_err_before = []
    cal_wt_before = []
    cal_err_after = []
    cal_wt_after = []
    
    val_err_before = []
    val_wt_before = []
    val_err_after = []
    val_wt_after = []

    # Per-target shape metrics
    target_cal_rmse_shape_before = []
    target_cal_rmse_shape_after = []
    target_cal_corr_before = []
    target_cal_corr_after = []
    
    target_val_rmse_shape_before = []
    target_val_rmse_shape_after = []
    target_val_corr_before = []
    target_val_corr_after = []
    
    for target_dir in target_dirs:
        summary_path = target_dir / "validation_summary.json"
        csv_path = target_dir / "theory_vs_simulation.csv"
        if not summary_path.exists() or not csv_path.exists():
            continue
            
        with open(summary_path, "r") as f:
            summary = json.load(f)
            
        offset = summary["all"]["theory_level_offset_db"]
        
        # Read CSV data for this target
        target_cal_sim = []
        target_cal_th_after = []
        target_cal_th_before = []
        target_cal_wt = []
        
        target_val_sim = []
        target_val_th_after = []
        target_val_th_before = []
        target_val_wt = []
        
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                split = row["split"]
                sim = float(row["simulated_level_db"])
                th = float(row["theory_level_db"])
                err_after = float(row["error_db"])
                weight = float(row["weight"])
                
                th_before = th - offset
                err_before = th_before - sim
                
                if split == "calibration":
                    cal_err_before.append(err_before)
                    cal_wt_before.append(weight)
                    cal_err_after.append(err_after)
                    cal_wt_after.append(weight)
                    
                    target_cal_sim.append(sim)
                    target_cal_th_after.append(th)
                    target_cal_th_before.append(th_before)
                    target_cal_wt.append(weight)
                elif split == "validation":
                    val_err_before.append(err_before)
                    val_wt_before.append(weight)
                    val_err_after.append(err_after)
                    val_wt_after.append(weight)
                    
                    target_val_sim.append(sim)
                    target_val_th_after.append(th)
                    target_val_th_before.append(th_before)
                    target_val_wt.append(weight)
                    
        # Compute centered metrics for this target
        if target_cal_sim:
            rmse_s_b, corr_b = calculate_centered_metrics(target_cal_sim, target_cal_th_before, target_cal_wt)
            rmse_s_a, corr_a = calculate_centered_metrics(target_cal_sim, target_cal_th_after, target_cal_wt)
            target_cal_rmse_shape_before.append(rmse_s_b)
            target_cal_rmse_shape_after.append(rmse_s_a)
            target_cal_corr_before.append(corr_b)
            target_cal_corr_after.append(corr_a)
            
        if target_val_sim:
            rmse_s_b, corr_b = calculate_centered_metrics(target_val_sim, target_val_th_before, target_val_wt)
            rmse_s_a, corr_a = calculate_centered_metrics(target_val_sim, target_val_th_after, target_val_wt)
            target_val_rmse_shape_before.append(rmse_s_b)
            target_val_rmse_shape_after.append(rmse_s_a)
            target_val_corr_before.append(corr_b)
            target_val_corr_after.append(corr_a)

    cal_bias_b, cal_mae_b, cal_rmse_b = weighted_stats(cal_err_before, cal_wt_before)
    val_bias_b, val_mae_b, val_rmse_b = weighted_stats(val_err_before, val_wt_before)
    
    cal_bias_a, cal_mae_a, cal_rmse_a = weighted_stats(cal_err_after, cal_wt_after)
    val_bias_a, val_mae_a, val_rmse_a = weighted_stats(val_err_after, val_wt_after)

    results = {
        "before_calibration": {
            "calibration": {"bias_db": cal_bias_b, "mae_db": cal_mae_b, "rmse_db": cal_rmse_b},
            "validation": {"bias_db": val_bias_b, "mae_db": val_mae_b, "rmse_db": val_rmse_b}
        },
        "after_calibration": {
            "calibration": {"bias_db": cal_bias_a, "mae_db": cal_mae_a, "rmse_db": cal_rmse_a},
            "validation": {"bias_db": val_bias_a, "mae_db": val_mae_a, "rmse_db": val_rmse_a}
        },
        "shape_metrics": {
            "calibration": {
                "mean_rmse_shape_before": float(np.mean(target_cal_rmse_shape_before)),
                "mean_rmse_shape_after": float(np.mean(target_cal_rmse_shape_after)),
                "mean_correlation_before": float(np.mean(target_cal_corr_before)),
                "mean_correlation_after": float(np.mean(target_cal_corr_after))
            },
            "validation": {
                "mean_rmse_shape_before": float(np.mean(target_val_rmse_shape_before)),
                "mean_rmse_shape_after": float(np.mean(target_val_rmse_shape_after)),
                "mean_correlation_before": float(np.mean(target_val_corr_before)),
                "mean_correlation_after": float(np.mean(target_val_corr_after))
            }
        }
    }

    print("--- BEFORE SCALAR CALIBRATION ---")
    print(f"Calibration Split: Bias = {cal_bias_b:.2f} dB, MAE = {cal_mae_b:.2f} dB, RMSE = {cal_rmse_b:.2f} dB")
    print(f"Validation Split:  Bias = {val_bias_b:.2f} dB, MAE = {val_mae_b:.2f} dB, RMSE = {val_rmse_b:.2f} dB")
    
    print("\n--- AFTER SCALAR CALIBRATION ---")
    print(f"Calibration Split: Bias = {cal_bias_a:.2f} dB, MAE = {cal_mae_a:.2f} dB, RMSE = {cal_rmse_a:.2f} dB")
    print(f"Validation Split:  Bias = {val_bias_a:.2f} dB, MAE = {val_mae_a:.2f} dB, RMSE = {val_rmse_a:.2f} dB")

    print("\n--- SHAPE PRESERVATION METRICS (MEAN ACROSS TARGETS) ---")
    print(f"Calibration Split shape RMSE: Before = {results['shape_metrics']['calibration']['mean_rmse_shape_before']:.2f} dB, After = {results['shape_metrics']['calibration']['mean_rmse_shape_after']:.2f} dB")
    print(f"Calibration Split correlation: Before = {results['shape_metrics']['calibration']['mean_correlation_before']:.4f}, After = {results['shape_metrics']['calibration']['mean_correlation_after']:.4f}")
    print(f"Validation Split shape RMSE:  Before = {results['shape_metrics']['validation']['mean_rmse_shape_before']:.2f} dB, After = {results['shape_metrics']['validation']['mean_rmse_shape_after']:.2f} dB")
    print(f"Validation Split correlation:  Before = {results['shape_metrics']['validation']['mean_correlation_before']:.4f}, After = {results['shape_metrics']['validation']['mean_correlation_after']:.4f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved summary results to: {output_path}")

if __name__ == "__main__":
    main()
