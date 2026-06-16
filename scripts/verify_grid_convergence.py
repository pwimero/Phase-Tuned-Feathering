import sys
import os
import json
import torch
import math
import random
from pathlib import Path
from dataclasses import replace
import numpy as np

# Add workspace src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase_tuned_feathering.geometry import WingGeometryParams, source_grid, default_geometry
from phase_tuned_feathering.observers import ObserverGrid
from phase_tuned_feathering.acoustics_torch import evaluate_spp_torch
from phase_tuned_feathering.closures import ClosureParams, FlowConfig

def main():
    root_dir = Path(__file__).resolve().parents[1]
    output_path = root_dir / "outputs" / "grid_convergence_summary.json"

    # 1. Observers: Full spherical grid (5 theta x 6 phi = 30 observers)
    observers = ObserverGrid.spherical(5, 6)
    flow = FlowConfig()
    
    # 2. Geometries
    geom_baseline = default_geometry()
    
    # Load optimized geometry for target seed 19
    geom_opt = geom_baseline
    opt_geom_path = root_dir / "outputs" / "optimization" / "full" / "target_019" / "optimized_geometry.json"
    if opt_geom_path.exists():
        with open(opt_geom_path, "r") as f:
            geom_dict = json.load(f)
        params_dict = geom_dict["params"]
        if "per_feather_incidence_deg" in params_dict and params_dict["per_feather_incidence_deg"] is not None:
            params_dict["per_feather_incidence_deg"] = tuple(params_dict["per_feather_incidence_deg"])
        if "sweep_section_fractions" in params_dict and params_dict["sweep_section_fractions"] is not None:
            params_dict["sweep_section_fractions"] = tuple(params_dict["sweep_section_fractions"])
        geom_opt = WingGeometryParams(**params_dict)
    
    # Create randomized perturbed geometry
    random.seed(1234)
    geom_perturbed = replace(
        geom_baseline,
        y_spacing_scale=float(geom_baseline.y_spacing_scale + random.uniform(-0.1, 0.1)),
        wing_1_tip_sweep=float(geom_baseline.wing_1_tip_sweep + random.uniform(-0.1, 0.1)),
        wing_7_tip_sweep=float(geom_baseline.wing_7_tip_sweep + random.uniform(-0.1, 0.1)),
        wing_1_root_z_translation=float(geom_baseline.wing_1_root_z_translation + random.uniform(-0.05, 0.05)),
        wing_7_root_z_translation=float(geom_baseline.wing_7_root_z_translation + random.uniform(-0.05, 0.05)),
        min_tip_chord_scale=float(geom_baseline.min_tip_chord_scale + random.uniform(-0.02, 0.02))
    )
    
    geometries = {
        "baseline": geom_baseline,
        "optimized": geom_opt,
        "perturbed": geom_perturbed
    }
    
    # 3. Parameters
    frequencies = [250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]
    coherence_models = ["zero", "full", "exponential"]
    resolutions = [8, 16, 32, 64, 128, 256]
    ref_res = 256
    
    # Storage for errors: key resolution -> list of errors
    res_rel_errors = {r: [] for r in resolutions[:-1]}
    res_db_errors = {r: [] for r in resolutions[:-1]}
    
    print("Running convergence checks across parameter space...")
    
    for geom_name, geom in geometries.items():
        for model in coherence_models:
            closures = ClosureParams(coherence_model=model)
            for freq in frequencies:
                # Precompute grids for all resolutions to save time
                grids = {}
                spps = {}
                for r in resolutions:
                    grid = source_grid(geom, n_eta=r)
                    
                    points_t = torch.tensor(grid.points, dtype=torch.float32).unsqueeze(0)
                    chords_t = torch.tensor(grid.chords, dtype=torch.float32).unsqueeze(0)
                    incidence_deg_t = torch.tensor(grid.incidence_deg, dtype=torch.float32).unsqueeze(0)
                    loading_directions_t = torch.tensor(grid.loading_directions, dtype=torch.float32).unsqueeze(0)
                    weights_t = torch.tensor(grid.weights, dtype=torch.float32).unsqueeze(0)
                    
                    spp = evaluate_spp_torch(
                        points_t, chords_t, incidence_deg_t, loading_directions_t, weights_t,
                        observers=observers, frequencies_hz=freq, flow=flow, closures=closures
                    )[0]
                    spps[r] = spp
                
                # Reference exact solution
                spp_ref = spps[ref_res]
                
                for r in resolutions[:-1]:
                    spp_r = spps[r]
                    # Relative error at each observer
                    rel_err = torch.abs(spp_r - spp_ref) / torch.clamp(spp_ref, min=1.0e-30)
                    # SPL error in dB
                    db_r = 10.0 * torch.log10(torch.clamp(spp_r, min=1.0e-30) / (flow.p_ref ** 2))
                    db_ref = 10.0 * torch.log10(torch.clamp(spp_ref, min=1.0e-30) / (flow.p_ref ** 2))
                    db_err = torch.abs(db_r - db_ref)
                    
                    res_rel_errors[r].extend(rel_err.tolist())
                    res_db_errors[r].extend(db_err.tolist())
                    
    # Aggregated metrics per resolution
    summary = {}
    for r in resolutions[:-1]:
        rel_errs = res_rel_errors[r]
        db_errs = res_db_errors[r]
        
        max_rel = float(np.max(rel_errs))
        rms_rel = float(math.sqrt(np.mean([x**2 for x in rel_errs])))
        max_db = float(np.max(db_errs))
        rms_db = float(math.sqrt(np.mean([x**2 for x in db_errs])))
        
        summary[r] = {
            "max_relative_error": max_rel,
            "rms_relative_error": rms_rel,
            "max_spl_error_db": max_db,
            "rms_spl_error_db": rms_db
        }
        
    # Empirical convergence rates
    rates = {}
    for i in range(len(resolutions) - 2):
        r_curr = resolutions[i]
        r_next = resolutions[i+1]
        
        err_curr = summary[r_curr]["rms_relative_error"]
        err_next = summary[r_next]["rms_relative_error"]
        
        rate = math.log2(err_curr / err_next) if err_next > 1.0e-15 else 2.0
        rates[r_next] = rate
        summary[r_next]["empirical_rate"] = rate

    summary[resolutions[0]]["empirical_rate"] = None
    
    print("\n--- GRID CONVERGENCE SUMMARY ---")
    print(f"{'n_eta':<8}{'Max Rel Err':<15}{'RMS Rel Err':<15}{'Max SPL Err (dB)':<18}{'RMS SPL Err (dB)':<18}{'Empirical Rate':<15}")
    for r in resolutions[:-1]:
        rate_str = f"{summary[r]['empirical_rate']:.2f}" if summary[r]['empirical_rate'] is not None else "---"
        print(f"{r:<8}{summary[r]['max_relative_error']:<15.3e}{summary[r]['rms_relative_error']:<15.3e}{summary[r]['max_spl_error_db']:<18.4f}{summary[r]['rms_spl_error_db']:<18.4f}{rate_str:<15}")

    results = {
        "resolutions": resolutions[:-1],
        "metrics_by_resolution": summary
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved summary results to: {output_path}")

if __name__ == "__main__":
    main()
