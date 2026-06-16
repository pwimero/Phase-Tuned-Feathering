import os
import csv
import json
import math
from pathlib import Path

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

def main():
    root_dir = Path("/Users/primero/Library/Mobile Documents/com~apple~CloudDocs/CODE PROJECTS/Phase Tuned Feathering/outputs/optimization/full")
    target_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith("target_")])
    
    cal_err_before = []
    cal_wt_before = []
    cal_err_after = []
    cal_wt_after = []
    
    val_err_before = []
    val_wt_before = []
    val_err_after = []
    val_wt_after = []
    
    for target_dir in target_dirs:
        summary_path = target_dir / "validation_summary.json"
        csv_path = target_dir / "theory_vs_simulation.csv"
        if not summary_path.exists() or not csv_path.exists():
            continue
            
        with open(summary_path, "r") as f:
            summary = json.load(f)
            
        offset = summary["all"]["theory_level_offset_db"]
        
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
                elif split == "validation":
                    val_err_before.append(err_before)
                    val_wt_before.append(weight)
                    val_err_after.append(err_after)
                    val_wt_after.append(weight)

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
        }
    }

    print("--- BEFORE SCALAR CALIBRATION ---")
    print(f"Calibration Split: Bias = {cal_bias_b:.2f} dB, MAE = {cal_mae_b:.2f} dB, RMSE = {cal_rmse_b:.2f} dB")
    print(f"Validation Split:  Bias = {val_bias_b:.2f} dB, MAE = {val_mae_b:.2f} dB, RMSE = {val_rmse_b:.2f} dB")
    
    print("\n--- AFTER SCALAR CALIBRATION ---")
    print(f"Calibration Split: Bias = {cal_bias_a:.2f} dB, MAE = {cal_mae_a:.2f} dB, RMSE = {cal_rmse_a:.2f} dB")
    print(f"Validation Split:  Bias = {val_bias_a:.2f} dB, MAE = {val_mae_a:.2f} dB, RMSE = {val_rmse_a:.2f} dB")

    output_path = Path("/Users/primero/Library/Mobile Documents/com~apple~CloudDocs/CODE PROJECTS/Phase Tuned Feathering/outputs/calibration_separability_summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved summary results to: {output_path}")

if __name__ == "__main__":
    main()
