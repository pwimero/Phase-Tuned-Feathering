import sys
import random
from pathlib import Path
from dataclasses import replace

# Add workspace src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase_tuned_feathering.geometry import WingGeometryParams, source_grid, default_geometry
from phase_tuned_feathering.closures import (
    ClosureParams,
    FlowConfig,
    source_cross_spectral_matrix,
    is_hermitian,
    cholesky_psd_check
)
from phase_tuned_feathering.observers import ObserverGrid
from phase_tuned_feathering.acoustics import evaluate_spp

def test_cross_spectral_matrix_is_psd_for_all_coherence_models():
    flow = FlowConfig()
    frequencies = (0.0, 250.0, 1000.0, 4000.0, 8000.0)
    
    # 1. Test baseline geometry
    params_baseline = default_geometry()
    grid_baseline = source_grid(params_baseline, n_eta=16)
    
    # 2. Test randomized geometries
    random.seed(42)
    randomized_geometries = []
    for _ in range(3):
        geom_perturbed = replace(
            params_baseline,
            y_spacing_scale=float(params_baseline.y_spacing_scale + random.uniform(-0.1, 0.1)),
            wing_1_tip_sweep=float(params_baseline.wing_1_tip_sweep + random.uniform(-0.1, 0.1)),
            wing_7_tip_sweep=float(params_baseline.wing_7_tip_sweep + random.uniform(-0.1, 0.1)),
            wing_1_root_z_translation=float(params_baseline.wing_1_root_z_translation + random.uniform(-0.05, 0.05)),
            wing_7_root_z_translation=float(params_baseline.wing_7_root_z_translation + random.uniform(-0.05, 0.05)),
            min_tip_chord_scale=float(params_baseline.min_tip_chord_scale + random.uniform(-0.02, 0.02))
        )
        randomized_geometries.append(geom_perturbed)
        
    all_geometries = [params_baseline] + randomized_geometries
    
    for geom in all_geometries:
        grid = source_grid(geom, n_eta=16)
        for model in ("zero", "full", "exponential"):
            closures = ClosureParams(coherence_model=model)
            for frequency in frequencies:
                cq = source_cross_spectral_matrix(grid, frequency, flow, closures)
                assert is_hermitian(cq, tolerance=1.0e-8), f"Failed Hermitian check for {model} model at {frequency} Hz"
                assert cholesky_psd_check(cq, tolerance=1.0e-9), f"Failed PSD check for {model} model at {frequency} Hz"

def test_quadrature_convergence_rate():
    # Verify that integration error converges quadratically (rate near 2.0)
    # as discretization n_eta increases
    geom = default_geometry()
    observers = ObserverGrid(
        directions=((0.0, 0.0, 1.0),),
        weights=(1.0,)
    )
    flow = FlowConfig()
    closures = ClosureParams(coherence_model="exponential")
    frequency = 1000.0
    
    spps = {}
    for n_eta in (32, 64, 128, 256):
        grid = source_grid(geom, n_eta=n_eta)
        res = evaluate_spp(grid, observers, [frequency], flow, closures)
        spps[n_eta] = res.spp[0][0]
        
    # Differences
    diff_32_64 = abs(spps[32] - spps[64])
    diff_64_128 = abs(spps[64] - spps[128])
    diff_128_256 = abs(spps[128] - spps[256])
    
    # Rates
    rate_1 = math_log2(diff_32_64 / diff_64_128) if diff_64_128 > 1.0e-20 else 2.0
    rate_2 = math_log2(diff_64_128 / diff_128_256) if diff_128_256 > 1.0e-20 else 2.0
    
    print(f"Convergence rates: {rate_1:.3f}, {rate_2:.3f}")
    
    # Assert that convergence rate is close to 2.0
    assert 1.5 < rate_1 < 2.5
    assert 1.5 < rate_2 < 2.5

def math_log2(x):
    return math.log(x) / math.log(2.0)

import math

if __name__ == "__main__":
    print("Running test_cross_spectral_matrix_is_psd_for_all_coherence_models...")
    test_cross_spectral_matrix_is_psd_for_all_coherence_models()
    print("test_cross_spectral_matrix_is_psd_for_all_coherence_models passed!")
    
    print("\nRunning test_quadrature_convergence_rate...")
    test_quadrature_convergence_rate()
    print("test_quadrature_convergence_rate passed!")
    
    print("\nAll tests passed successfully!")
