from dataclasses import dataclass, field, replace
from typing import Sequence
from functools import lru_cache
import math
import numpy as np
import time
import torch
from typing import Tuple, List

from .acoustics_torch import evaluate_spp_torch
from .closures import FlowConfig, ClosureParams
from .geometry import WingGeometryParams, default_geometry
from .observers import ObserverGrid
import concurrent.futures


@dataclass(frozen=True)
class GAConfig:
    base_params: WingGeometryParams = field(default_factory=default_geometry)
    flow: FlowConfig = field(default_factory=FlowConfig)
    closures: ClosureParams = field(default_factory=ClosureParams)
    observers: ObserverGrid = field(default_factory=lambda: ObserverGrid.spherical(8, 10))
    frequencies_hz: tuple[float, ...] = (500.0, 1000.0, 2000.0)
    target_pattern_db: tuple[float, ...] = field(default_factory=tuple)
    n_eta: int = 16
    popsize: int = 10
    maxiter: int = 0
    mutation: tuple[float, float] = (0.5, 1.0)
    recombination: float = 0.7
    seed: int = 42
    patience: int = 15

    # Expanded where the recent campaign repeatedly saturated the bounds.
    incidence_bounds_deg: tuple[float, float] = (-20.0, 20.0)
    spacing_scale_bounds: tuple[float, float] = (0.65, 2.10)
    root_z_bounds: tuple[float, float] = (-0.45, 0.45)
    tip_sweep_bounds: tuple[float, float] = (-1.10, 0.80)
    tip_z_curve_bounds: tuple[float, float] = (-0.75, 1.00)
    min_tip_chord_scale_bounds: tuple[float, float] = (0.06, 0.55)
    intersection_penalty: float = 1.0e6
    intersection_n_eta: int = 8
    coarse_n_eta: int = 8
    coarse_observer_stride: int = 2
    coarse_frequency_stride: int = 2
    refinement_top_fraction: float = 0.35
    refinement_min_candidates: int = 12


@dataclass(frozen=True)
class GAResult:
    best_params: WingGeometryParams
    best_score: float
    elapsed_seconds: float
    success: bool
    message: str


@dataclass(frozen=True)
class _FitnessContext:
    observers: ObserverGrid
    observer_dirs: torch.Tensor
    frequencies_hz: tuple[float, ...]
    frequency_tensor: torch.Tensor
    target_normalized: torch.Tensor
    flow: FlowConfig
    closures: ClosureParams
    scales: torch.Tensor


def _incidence_basis(total_wings: int) -> tuple[tuple[float, ...], ...]:
    if total_wings <= 0:
        raise ValueError("total_wings must be positive.")
    rows: list[tuple[float, ...]] = []
    scale0 = math.sqrt(1.0 / total_wings)
    for wing_index in range(total_wings):
        row: list[float] = []
        for mode in range(total_wings):
            if mode == 0:
                row.append(scale0)
            else:
                row.append(
                    math.sqrt(2.0 / total_wings)
                    * math.cos(
                        math.pi
                        * (wing_index + 0.5)
                        * mode
                        / total_wings
                    )
                )
        rows.append(tuple(row))
    return tuple(rows)


def _decode_incidence_shape_coeffs(
    coeffs: Sequence[float],
    total_wings: int,
) -> tuple[float, ...]:
    if len(coeffs) != total_wings:
        raise ValueError(
            f"Expected {total_wings} incidence coefficients, got {len(coeffs)}."
        )
    basis = _incidence_basis(total_wings)
    values: list[float] = []
    for wing_index in range(total_wings):
        values.append(
            sum(basis[wing_index][mode] * float(coeffs[mode]) for mode in range(total_wings))
        )
    return tuple(values)


def _incidence_coeff_bounds(config: GAConfig) -> tuple[tuple[float, float], ...]:
    total_wings = config.base_params.total_wings
    max_abs_incidence = max(
        abs(config.incidence_bounds_deg[0]),
        abs(config.incidence_bounds_deg[1]),
    )
    basis = _incidence_basis(total_wings)
    bounds: list[tuple[float, float]] = []
    for mode in range(total_wings):
        l1_norm = sum(abs(basis[wing_index][mode]) for wing_index in range(total_wings))
        coeff_limit = max_abs_incidence * l1_norm
        bounds.append((-coeff_limit, coeff_limit))
    return tuple(bounds)


def _unpack_genes(x: Sequence[float], base_params: WingGeometryParams) -> WingGeometryParams:
    """Unpack a 1D array of genes into a WingGeometryParams object."""
    total_wings = base_params.total_wings
    extra_gene_count = 11
    if len(x) != total_wings + extra_gene_count:
        raise ValueError(f"Expected {total_wings + extra_gene_count} genes, got {len(x)}")

    incidences = _decode_incidence_shape_coeffs(x[0:total_wings], total_wings)
    y_spacing = x[total_wings]
    wing_1_root_z = x[total_wings + 1]
    mid_root_z = x[total_wings + 2]
    wing_7_root_z = x[total_wings + 3]
    wing_1_tip_sweep = x[total_wings + 4]
    mid_tip_sweep = x[total_wings + 5]
    wing_7_tip_sweep = x[total_wings + 6]
    wing_1_tip_z_curve = x[total_wings + 7]
    mid_tip_z_curve = x[total_wings + 8]
    wing_7_tip_z_curve = x[total_wings + 9]
    min_tip_chord_scale = x[total_wings + 10]

    params = base_params.with_incidence_angles(incidences)
    params = replace(
        params,
        y_spacing_scale=y_spacing,
        wing_1_root_z_translation=wing_1_root_z,
        mid_wing_root_z_translation=mid_root_z,
        wing_7_root_z_translation=wing_7_root_z,
        wing_1_tip_sweep=wing_1_tip_sweep,
        mid_wing_tip_sweep=mid_tip_sweep,
        wing_7_tip_sweep=wing_7_tip_sweep,
        wing_1_tip_z_curve=wing_1_tip_z_curve,
        mid_wing_tip_z_curve=mid_tip_z_curve,
        wing_7_tip_z_curve=wing_7_tip_z_curve,
        min_tip_chord_scale=min_tip_chord_scale,
        min_feather_root_gap=0.0,
    )
    return params


def _get_bounds(config: GAConfig) -> list[tuple[float, float]]:
    total_wings = config.base_params.total_wings
    bounds = []
    # Incidence shape coefficients in a full-rank orthonormal basis.
    bounds.extend(_incidence_coeff_bounds(config))
    bounds.append(config.spacing_scale_bounds)
    bounds.append(config.root_z_bounds)
    bounds.append(config.root_z_bounds)
    bounds.append(config.root_z_bounds)
    bounds.append(config.tip_sweep_bounds)
    bounds.append(config.tip_sweep_bounds)
    bounds.append(config.tip_sweep_bounds)
    bounds.append(config.tip_z_curve_bounds)
    bounds.append(config.tip_z_curve_bounds)
    bounds.append(config.tip_z_curve_bounds)
    bounds.append(config.min_tip_chord_scale_bounds)
    return bounds


def _coarse_observer_indices(config: GAConfig) -> tuple[int, ...]:
    stride = max(config.coarse_observer_stride, 1)
    indices = tuple(range(0, len(config.observers.directions), stride))
    if not indices:
        return (0,)
    return indices


def _coarse_observers(config: GAConfig) -> ObserverGrid:
    indices = _coarse_observer_indices(config)
    return ObserverGrid(
        tuple(config.observers.directions[index] for index in indices),
        tuple(config.observers.weights[index] for index in indices),
    )


def _coarse_target_pattern(config: GAConfig) -> tuple[float, ...]:
    indices = _coarse_observer_indices(config)
    return tuple(config.target_pattern_db[index] for index in indices)


def _coarse_frequencies(config: GAConfig) -> tuple[float, ...]:
    stride = max(config.coarse_frequency_stride, 1)
    coarse = tuple(config.frequencies_hz[index] for index in range(0, len(config.frequencies_hz), stride))
    if not coarse:
        return config.frequencies_hz
    if coarse[-1] != config.frequencies_hz[-1]:
        coarse = coarse + (config.frequencies_hz[-1],)
    return coarse


def _segments_intersect(
    a0: tuple[float, float],
    a1: tuple[float, float],
    b0: tuple[float, float],
    b1: tuple[float, float],
    tolerance: float = 1.0e-10,
) -> bool:
    def orientation(
        p: tuple[float, float],
        q: tuple[float, float],
        r: tuple[float, float],
    ) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_segment(
        p: tuple[float, float],
        q: tuple[float, float],
        r: tuple[float, float],
    ) -> bool:
        return (
            min(p[0], r[0]) - tolerance <= q[0] <= max(p[0], r[0]) + tolerance
            and min(p[1], r[1]) - tolerance <= q[1] <= max(p[1], r[1]) + tolerance
        )

    o1 = orientation(a0, a1, b0)
    o2 = orientation(a0, a1, b1)
    o3 = orientation(b0, b1, a0)
    o4 = orientation(b0, b1, a1)

    if o1 * o2 < -tolerance and o3 * o4 < -tolerance:
        return True
    if abs(o1) <= tolerance and on_segment(a0, b0, a1):
        return True
    if abs(o2) <= tolerance and on_segment(a0, b1, a1):
        return True
    if abs(o3) <= tolerance and on_segment(b0, a0, b1):
        return True
    if abs(o4) <= tolerance and on_segment(b0, a1, b1):
        return True
    return False


def _projected_source_line_intersection_count(
    points: tuple[tuple[float, float, float], ...],
    feather_ids: tuple[int, ...],
) -> int:
    by_feather: dict[int, list[tuple[float, float]]] = {}
    for point, feather_id in zip(points, feather_ids):
        by_feather.setdefault(feather_id, []).append((point[0], point[1]))

    count = 0
    feathers = sorted(by_feather)
    for left_index, left_feather in enumerate(feathers):
        left_points = by_feather[left_feather]
        for right_feather in feathers[left_index + 1:]:
            right_points = by_feather[right_feather]
            for a_index in range(len(left_points) - 1):
                a0 = left_points[a_index]
                a1 = left_points[a_index + 1]
                for b_index in range(len(right_points) - 1):
                    b0 = right_points[b_index]
                    b1 = right_points[b_index + 1]
                    if _segments_intersect(a0, a1, b0, b1):
                        count += 1
    return count


@lru_cache(maxsize=4096)
def _eval_cpu_worker_cached(
    genes_key: tuple[float, ...],
    base_params: WingGeometryParams,
    n_eta: int,
    intersection_penalty: float,
    intersection_n_eta: int,
):
    from .geometry import source_grid

    params = _unpack_genes(genes_key, base_params)
    grid = source_grid(params, n_eta=n_eta)
    intersection_grid = source_grid(params, n_eta=intersection_n_eta)
    intersection_count = _projected_source_line_intersection_count(
        intersection_grid.points,
        intersection_grid.feather_ids,
    )

    return (
        np.asarray(grid.points, dtype=np.float32),
        np.asarray(grid.chords, dtype=np.float32),
        np.asarray(grid.incidence_deg, dtype=np.float32),
        np.asarray(grid.loading_directions, dtype=np.float32),
        np.asarray(grid.weights, dtype=np.float32),
        float(intersection_penalty * intersection_count),
    )


def _eval_cpu_worker(g, base_params, n_eta, intersection_penalty, intersection_n_eta):
    return _eval_cpu_worker_cached(
        tuple(float(value) for value in g),
        base_params,
        n_eta,
        intersection_penalty,
        intersection_n_eta,
    )


def _sample_distinct_parents(population_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    self_indices = torch.arange(population_size, device=device)

    a = torch.randint(0, population_size - 1, (population_size,), device=device)
    a = a + (a >= self_indices).to(torch.long)

    b = torch.randint(0, population_size, (population_size,), device=device)
    invalid_b = (b == self_indices) | (b == a)
    while torch.any(invalid_b):
        b[invalid_b] = torch.randint(0, population_size, (int(invalid_b.sum().item()),), device=device)
        invalid_b = (b == self_indices) | (b == a)

    c = torch.randint(0, population_size, (population_size,), device=device)
    invalid_c = (c == self_indices) | (c == a) | (c == b)
    while torch.any(invalid_c):
        c[invalid_c] = torch.randint(0, population_size, (int(invalid_c.sum().item()),), device=device)
        invalid_c = (c == self_indices) | (c == a) | (c == b)

    return a, b, c


def _make_fitness_context(
    observers: ObserverGrid,
    frequencies_hz: tuple[float, ...],
    target_pattern_db: tuple[float, ...],
    flow: FlowConfig,
    closures: ClosureParams,
    device: torch.device,
) -> _FitnessContext:
    observer_dirs = torch.as_tensor(observers.directions, dtype=torch.float32, device=device)
    frequency_tensor = torch.as_tensor(frequencies_hz, dtype=torch.float32, device=device)
    target_tensor = torch.as_tensor(target_pattern_db, dtype=torch.float32, device=device)
    target_normalized = target_tensor - torch.max(target_tensor)
    scales = torch.tensor(
        [closures.coherence_x, closures.coherence_y, closures.coherence_z],
        dtype=torch.float32,
        device=device,
    )
    return _FitnessContext(
        observers=observers,
        observer_dirs=observer_dirs,
        frequencies_hz=frequencies_hz,
        frequency_tensor=frequency_tensor,
        target_normalized=target_normalized,
        flow=flow,
        closures=closures,
        scales=scales,
    )


@torch.no_grad()
def differential_evolution_torch(
    objective_fn,
    bounds: List[Tuple[float, float]],
    popsize: int = 15,
    maxiter: int = 0,
    mutation: Tuple[float, float] = (0.5, 1.0),
    recombination: float = 0.7,
    seed: int = 42,
    device: torch.device = None,
    patience: int = 10,
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
    
    no_improve_count = 0
    gen = 0

    while True:
        gen += 1
        parent_a, parent_b, parent_c = _sample_distinct_parents(P, device)
        a = pop[parent_a]
        b = pop[parent_b]
        c = pop[parent_c]
        
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
        current_best = fitness[best_idx].item()
        
        if current_best < best_fitness - 1e-5:
            best_fitness = current_best
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        if maxiter > 0:
            print(f"Generation {gen}/{maxiter} - Best Fitness: {best_fitness:.4f}")
        else:
            print(f"Generation {gen} - Best Fitness: {best_fitness:.4f}")
        
        if no_improve_count >= patience:
            print(f"Early stopping at generation {gen}: No improvement in {patience} generations.")
            break

        if maxiter > 0 and gen >= maxiter:
            print(f"Reached generation cap at {gen}.")
            break
        
    return pop[best_idx].cpu().numpy(), best_fitness


@torch.no_grad()
def _evaluate_population_fitness(
    results: list[tuple],
    context: _FitnessContext,
    device: torch.device,
) -> torch.Tensor:
    points_np = np.stack([r[0] for r in results], axis=0)
    chords_np = np.stack([r[1] for r in results], axis=0)
    incidence_np = np.stack([r[2] for r in results], axis=0)
    loading_np = np.stack([r[3] for r in results], axis=0)
    weights_np = np.stack([r[4] for r in results], axis=0)
    penalties_np = np.asarray([r[5] for r in results], dtype=np.float32)

    points = torch.from_numpy(points_np).to(device=device)
    chords = torch.from_numpy(chords_np).to(device=device)
    incidence_deg = torch.from_numpy(incidence_np).to(device=device)
    loading_directions = torch.from_numpy(loading_np).to(device=device)
    weights = torch.from_numpy(weights_np).to(device=device)

    spp = evaluate_spp_torch(
        points,
        chords,
        incidence_deg,
        loading_directions,
        weights,
        observers=None,
        frequencies_hz=context.frequency_tensor,
        flow=context.flow,
        closures=context.closures,
        device=device,
        observer_dirs=context.observer_dirs,
        scales=context.scales,
    )

    if context.frequency_tensor.numel() == 1:
        spp_sum = spp
    else:
        df = context.frequency_tensor[1:] - context.frequency_tensor[:-1]
        df = df.unsqueeze(0).unsqueeze(-1)
        spp_sum = 0.5 * torch.sum(df * (spp[:, :-1, :] + spp[:, 1:, :]), dim=1)

    p_ref_sq = context.flow.p_ref ** 2
    spp_db = 10.0 * torch.log10(torch.clamp(spp_sum / p_ref_sq, min=1e-12))
    sim_peak, _ = torch.max(spp_db, dim=1, keepdim=True)
    sim_normalized = spp_db - sim_peak

    target_normalized = context.target_normalized.unsqueeze(0).expand(len(results), -1)

    fitness = torch.mean((sim_normalized - target_normalized) ** 2, dim=1)
    penalty_tensor = torch.from_numpy(penalties_np).to(device=device)
    return fitness + penalty_tensor


@torch.no_grad()
def _objective_torch(
    genes_tensor: torch.Tensor,
    config: GAConfig,
    device: torch.device,
    executor: concurrent.futures.ProcessPoolExecutor,
    coarse_context: _FitnessContext,
    fine_context: _FitnessContext,
) -> torch.Tensor:
    P = genes_tensor.shape[0]
    genes_np = genes_tensor.cpu().numpy()

    coarse_results = list(executor.map(
        _eval_cpu_worker,
        [genes_np[i] for i in range(P)],
        [config.base_params] * P,
        [config.coarse_n_eta] * P,
        [config.intersection_penalty] * P,
        [config.intersection_n_eta] * P,
    ))
    coarse_fitness = _evaluate_population_fitness(
        coarse_results,
        coarse_context,
        device,
    )

    refine_count = max(
        1,
        min(
            P,
            max(
                config.refinement_min_candidates,
                math.ceil(P * config.refinement_top_fraction),
            ),
        ),
    )
    refine_indices = torch.argsort(coarse_fitness)[:refine_count]
    fitness = coarse_fitness.clone()

    selected = refine_indices.cpu().tolist()
    fine_results = list(executor.map(
        _eval_cpu_worker,
        [genes_np[index] for index in selected],
        [config.base_params] * refine_count,
        [config.n_eta] * refine_count,
        [config.intersection_penalty] * refine_count,
        [config.intersection_n_eta] * refine_count,
    ))
    fine_fitness = _evaluate_population_fitness(
        fine_results,
        fine_context,
        device,
    )
    fitness[refine_indices] = fine_fitness
    return fitness


def run_optimization_torch(config: GAConfig) -> Tuple[WingGeometryParams, float, bool]:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    bounds = _get_bounds(config)
    coarse_context = _make_fitness_context(
        _coarse_observers(config),
        _coarse_frequencies(config),
        _coarse_target_pattern(config),
        config.flow,
        config.closures,
        device,
    )
    fine_context = _make_fitness_context(
        config.observers,
        config.frequencies_hz,
        config.target_pattern_db,
        config.flow,
        config.closures,
        device,
    )

    start_time = time.time()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        best_genes, best_fitness = differential_evolution_torch(
            objective_fn=lambda pop: _objective_torch(
                pop,
                config,
                device,
                executor,
                coarse_context,
                fine_context,
            ),
            bounds=bounds,
            popsize=config.popsize,
            maxiter=config.maxiter,
            mutation=config.mutation,
            recombination=config.recombination,
            seed=config.seed,
            device=device,
            patience=config.patience,
        )
    
    elapsed = time.time() - start_time
    print(f"PyTorch GA Completed in {elapsed:.1f}s.")
    print(f"  Best Score (Normalized MSE): {best_fitness:.2f} dB^2")
    
    best_genes_float = [float(x) for x in best_genes]
    best_params = _unpack_genes(best_genes_float, config.base_params)
    
    return best_params, float(best_fitness), True
