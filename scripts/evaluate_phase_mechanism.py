import sys
import os
import json
import torch
import numpy as np
from pathlib import Path

# Add workspace src to path
sys.path.insert(0, "/Users/primero/Library/Mobile Documents/com~apple~CloudDocs/CODE PROJECTS/Phase Tuned Feathering/src")

from phase_tuned_feathering.geometry import WingGeometryParams, source_grid
from phase_tuned_feathering.observers import ObserverGrid
from phase_tuned_feathering.acoustics_torch import evaluate_spp_torch
from phase_tuned_feathering.closures import ClosureParams, FlowConfig

def main():
    root_dir = Path("/Users/primero/Library/Mobile Documents/com~apple~CloudDocs/CODE PROJECTS/Phase Tuned Feathering/outputs/optimization/full")
    target_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith("target_")])
    
    observers = ObserverGrid.spherical(10, 15)  # Dense observer grid
    flow = FlowConfig()
    frequency = 1000.0  # 1 kHz
    
    diffs = []
    
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
        
        # Calculate RMSE difference in SPL across observers
        diff = torch.sqrt(torch.mean((levels_exp - levels_zero) ** 2))
        diffs.append(diff.item())
        
    mean_diff = float(np.mean(diffs))
    max_diff = float(np.max(diffs))
    min_diff = float(np.min(diffs))

    results = {
        "mean_spl_difference_db": mean_diff,
        "max_spl_difference_db": max_diff,
        "min_spl_difference_db": min_diff,
        "target_differences_db": diffs
    }

    print(f"Aggregated over {len(diffs)} targets:")
    print(f"Mean SPL difference when phase offsets are removed: {mean_diff:.3f} dB")
    print(f"Max SPL difference when phase offsets are removed:  {max_diff:.3f} dB")
    print(f"Min SPL difference when phase offsets are removed:  {min_diff:.3f} dB")

    output_path = Path("/Users/primero/Library/Mobile Documents/com~apple~CloudDocs/CODE PROJECTS/Phase Tuned Feathering/outputs/phase_mechanism_summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved summary results to: {output_path}")

if __name__ == "__main__":
    main()
