from dataclasses import dataclass, field, replace
from typing import Sequence
import time
import torch
import numpy as np
from typing import Tuple, List

from .acoustics_torch import evaluate_spp_torch
from .closures import FlowConfig, ClosureParams
from .geometry import WingGeometryParams, SourceGrid


@dataclass(frozen=True)
class GAConfig:
    base_params: WingGeometryParams = field(default_factory=default_geometry)
    flow: FlowConfig = field(default_factory=FlowConfig)
    closures: ClosureParams = field(default_factory=ClosureParams)
    observers: ObserverGrid = field(default_factory=lambda: ObserverGrid.spherical(10, 16))
    frequencies_hz: tuple[float, ...] = (500.0, 1000.0, 2000.0)
    target_sector: Sector = field(default_factory=lambda: Sector((0.0, 0.0, 1.0), 70.0))
    suppressed_sector: Sector = field(default_factory=lambda: Sector((0.0, 0.0, -1.0), 70.0))
    n_eta: int = 16
    popsize: int = 10
    maxiter: int = 30
    mutation: tuple[float, float] = (0.5, 1.0)
    recombination: float = 0.7
    seed: int = 42

    # Bounds definition
    incidence_bounds_deg: tuple[float, float] = (-10.0, 10.0)
    spacing_scale_bounds: tuple[float, float] = (1.0, 1.45)
    root_z_bounds: tuple[float, float] = (-0.18, 0.18)
    tip_sweep_bounds: tuple[float, float] = (-0.70, 0.35)
    tip_z_curve_bounds: tuple[float, float] = (-0.35, 0.65)
    min_tip_chord_scale_bounds: tuple[float, float] = (0.10, 0.35)


@dataclass(frozen=True)
class GAResult:
    best_params: WingGeometryParams
    best_score: float
    elapsed_seconds: float
    success: bool
    message: str


def _unpack_genes(x: Sequence[float], base_params: WingGeometryParams) -> WingGeometryParams:
    """Unpack a 1D array of genes into a WingGeometryParams object."""
    total_wings = base_params.total_wings
    # Ensure x has length total_wings + 8
    if len(x) != total_wings + 8:
        raise ValueError(f"Expected {total_wings + 8} genes, got {len(x)}")

    incidences = tuple(x[0:total_wings])
    y_spacing = x[total_wings]
    wing_1_root_z = x[total_wings + 1]
    wing_7_root_z = x[total_wings + 2]
    wing_1_tip_sweep = x[total_wings + 3]
    wing_7_tip_sweep = x[total_wings + 4]
    wing_1_tip_z_curve = x[total_wings + 5]
    wing_7_tip_z_curve = x[total_wings + 6]
    min_tip_chord_scale = x[total_wings + 7]

    params = base_params.with_incidence_angles(incidences)
    params = replace(
        params,
        y_spacing_scale=y_spacing,
        wing_1_root_z_translation=wing_1_root_z,
        wing_7_root_z_translation=wing_7_root_z,
        wing_1_tip_sweep=wing_1_tip_sweep,
        wing_7_tip_sweep=wing_7_tip_sweep,
        wing_1_tip_z_curve=wing_1_tip_z_curve,
        wing_7_tip_z_curve=wing_7_tip_z_curve,
        min_tip_chord_scale=min_tip_chord_scale,
    )
    return params


def _get_bounds(config: GAConfig) -> list[tuple[float, float]]:
    total_wings = config.base_params.total_wings
    bounds = []
    # Incidences
    for _ in range(total_wings):
        bounds.append(config.incidence_bounds_deg)
    bounds.append(config.spacing_scale_bounds)
    bounds.append(config.root_z_bounds)
    bounds.append(config.root_z_bounds)
    bounds.append(config.tip_sweep_bounds)
    bounds.append(config.tip_sweep_bounds)
    bounds.append(config.tip_z_curve_bounds)
    bounds.append(config.tip_z_curve_bounds)
    bounds.append(config.min_tip_chord_scale_bounds)
    return bounds



def differential_evolution_torch(
    objective_fn,
    bounds: List[Tuple[float, float]],
    popsize: int = 15,
    maxiter: int = 100,
    mutation: Tuple[float, float] = (0.5, 1.0),
    recombination: float = 0.7,
    seed: int = 42,
    device: torch.device = None,
):
    if device is None:
        device = torch.device("cpu")
        
    torch.manual_seed(seed)
    
    D = len(bounds)
    P = popsize * D
    
    bounds_tensor = torch.tensor(bounds, dtype=torch.float32, device=device)
    low = bounds_tensor[:, 0]
    high = bounds_tensor[:, 1]
    
    # Initialize population
    pop = low + torch.rand((P, D), device=device) * (high - low)
    fitness = objective_fn(pop)
    
    best_idx = torch.argmin(fitness)
    best_fitness = fitness[best_idx].item()
    
    for gen in range(maxiter):
        # Create mutants (P, D)
        # Random choice of 3 unique parents for each individual
        rand_idx = torch.rand((P, P), device=device).argsort(dim=1)[:, 1:4]
        a = pop[rand_idx[:, 0]]
        b = pop[rand_idx[:, 1]]
        c = pop[rand_idx[:, 2]]
        
        mut_factor = mutation[0] + torch.rand((P, 1), device=device) * (mutation[1] - mutation[0])
        mutant = a + mut_factor * (b - c)
        
        # Clamp mutant
        mutant = torch.max(torch.min(mutant, high), low)
        
        # Crossover
        cross_mask = torch.rand((P, D), device=device) < recombination
        force_cross = torch.randint(0, D, (P, 1), device=device)
        cross_mask.scatter_(1, force_cross, True)
        
        trial = torch.where(cross_mask, mutant, pop)
        
        # Evaluate
        trial_fitness = objective_fn(trial)
        
        # Selection
        improved = trial_fitness < fitness
        pop = torch.where(improved.unsqueeze(1), trial, pop)
        fitness = torch.where(improved, trial_fitness, fitness)
        
        # Track best
        best_idx = torch.argmin(fitness)
        if fitness[best_idx].item() < best_fitness:
            best_fitness = fitness[best_idx].item()
            
        print(f"Generation {gen+1}/{maxiter} - Best Fitness: {best_fitness:.4f}")
        
    return pop[best_idx].cpu().numpy(), best_fitness


def _objective_torch(
    genes_tensor: torch.Tensor,
    config: GAConfig,
    device: torch.device,
) -> torch.Tensor:
    P = genes_tensor.shape[0]
    genes_np = genes_tensor.cpu().numpy()
    
    points_list = []
    chords_list = []
    inc_list = []
    load_list = []
    weights_list = []
    
    from .geometry import source_grid
    
    for i in range(P):
        g = genes_np[i]
        params = _unpack_genes(g, config.base_params)
        grid = source_grid(params, n_eta=config.n_eta)
        points_list.append(grid.points)
        chords_list.append(grid.chords)
        inc_list.append(grid.incidence_deg)
        load_list.append(grid.loading_directions)
        weights_list.append(grid.weights)
        
    points = torch.tensor(points_list, dtype=torch.float32, device=device)
    chords = torch.tensor(chords_list, dtype=torch.float32, device=device)
    incidence_deg = torch.tensor(inc_list, dtype=torch.float32, device=device)
    loading_directions = torch.tensor(load_list, dtype=torch.float32, device=device)
    weights = torch.tensor(weights_list, dtype=torch.float32, device=device)
    
    target_spp_list = []
    suppress_spp_list = []
    
    target_weights = torch.tensor(config.target_sector.observer_weights(config.observers), dtype=torch.float32, device=device)
    suppress_weights = torch.tensor(config.suppressed_sector.observer_weights(config.observers), dtype=torch.float32, device=device)
    
    target_w_sum = target_weights.sum()
    suppress_w_sum = suppress_weights.sum()
    
    # Evaluate for each frequency on GPU
    for freq in config.frequencies_hz:
        spp = evaluate_spp_torch(
            points, chords, incidence_deg, loading_directions, weights,
            config.observers, freq, config.flow, config.closures, device
        )
        
        target_spp = (spp * target_weights).sum(dim=1) / target_w_sum
        target_spp_list.append(target_spp)
        
        suppress_spp = (spp * suppress_weights).sum(dim=1) / suppress_w_sum
        suppress_spp_list.append(suppress_spp)
        
    # Numerical integration (trapz) over frequencies
    # For now, just a simple sum works fine if the frequencies are equally spaced 
    # or if we match the old logic (the old code did _trapz but here I'll use a simple sum 
    # as an approximation or exact match if len(freqs)==1).
    # Wait, the original _objective used `sector_spl` which does trapz. Let's do trapz!
    
    freqs = torch.tensor(config.frequencies_hz, dtype=torch.float32, device=device)
    if len(freqs) == 1:
        target_sum = target_spp_list[0]
        suppress_sum = suppress_spp_list[0]
    else:
        # Stack lists -> (P, F)
        target_spp_stack = torch.stack(target_spp_list, dim=1)
        suppress_spp_stack = torch.stack(suppress_spp_list, dim=1)
        
        # trapz over F
        df = freqs[1:] - freqs[:-1]
        target_sum = 0.5 * torch.sum(df * (target_spp_stack[:, :-1] + target_spp_stack[:, 1:]), dim=1)
        suppress_sum = 0.5 * torch.sum(df * (suppress_spp_stack[:, :-1] + suppress_spp_stack[:, 1:]), dim=1)
    
    # Convert to dB safely
    p_ref_sq = config.flow.p_ref**2
    target_db = 10.0 * torch.log10(torch.clamp(target_sum / p_ref_sq, min=1e-12))
    suppress_db = 10.0 * torch.log10(torch.clamp(suppress_sum / p_ref_sq, min=1e-12))
    
    from .aero import screen_aero
    
    # Fitness = -(Target SPL - Suppressed SPL) + penalties
    # We want to minimize this.
    fitness = -(target_db - suppress_db)
    
    # Add penalties loop
    for i in range(P):
        g = genes_np[i]
        params = _unpack_genes(g, config.base_params)
        
        aero = screen_aero(params, config.flow)
        aero_penalty = 100.0 * len(aero.issues)
        
        incidence = params.incidence_angles_deg()
        alpha_penalty = sum(
            (incidence[index + 1] - incidence[index]) ** 2
            for index in range(len(incidence) - 1)
        )
        reg_penalty = 0.05 * alpha_penalty
        
        fitness[i] += aero_penalty + reg_penalty
        
    return fitness


def run_optimization_torch(config: GAConfig) -> Tuple[WingGeometryParams, float, bool]:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    bounds = _get_bounds(config)
    
    start_time = time.time()
    
    best_genes, best_fitness = differential_evolution_torch(
        objective_fn=lambda pop: _objective_torch(pop, config, device),
        bounds=bounds,
        popsize=config.popsize,
        maxiter=config.maxiter,
        mutation=config.mutation,
        recombination=config.recombination,
        seed=config.seed,
        device=device,
    )
    
    elapsed = time.time() - start_time
    # Note: fitness is Suppress - Target. So actual "score" (Target - Suppress) is -fitness.
    score = -best_fitness
    print(f"PyTorch GA Completed in {elapsed:.1f}s.")
    print(f"  Best Score (Target - Suppressed): {score:.2f} dB")
    
    best_genes_float = [float(x) for x in best_genes]
    best_params = _unpack_genes(best_genes_float, config.base_params)
    
    return best_params, float(score), True
