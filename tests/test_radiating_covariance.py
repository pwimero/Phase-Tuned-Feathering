import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase_tuned_feathering.closures import ClosureParams, FlowConfig
from phase_tuned_feathering.geometry import default_geometry, source_grid
from phase_tuned_feathering.observers import ObserverGrid
from phase_tuned_feathering.operators import (
    diagonal_covariance,
    mechanism_metrics,
    modal_radiation_decomposition,
    rank_one_covariance,
    raw_coherence_availability,
    radiation_operator,
    radiating_phase_steerability,
    source_csd_matrix,
    spp_quadratic,
    transfer_vector,
)


def test_modal_radiation_identity_matches_quadratic_form():
    flow = FlowConfig()
    grid = source_grid(default_geometry(), n_eta=12)
    observers = ObserverGrid.spherical(4, 6)
    closures = ClosureParams(coherence_model="exponential")
    cq = source_csd_matrix(grid, 1000.0, flow, closures)

    for direction in observers.directions[::5]:
        weights = transfer_vector(grid, direction, 1000.0, flow)
        modal = modal_radiation_decomposition(weights, cq, flow)
        assert modal.relative_error < 1.0e-10


def test_implemented_covariance_models_are_psd_in_double_precision():
    flow = FlowConfig()
    grid = source_grid(default_geometry(), n_eta=10)
    for model in ("zero", "full", "exponential"):
        for frequency in (0.0, 250.0, 1000.0, 4000.0):
            cq = source_csd_matrix(
                grid,
                frequency,
                flow,
                ClosureParams(coherence_model=model),
            )
            eig = np.linalg.eigvalsh(0.5 * (cq + cq.conj().T))
            norm = max(np.linalg.norm(cq, 2), 1.0)
            assert float(eig.min()) >= -1.0e-10 * norm


def test_diagonal_covariance_removes_position_phase_interference():
    rng = np.random.default_rng(11)
    n = 16
    source_power = 10.0 ** rng.uniform(-2.0, 1.0, n)
    cq = np.diag(source_power).astype(np.complex128)
    magnitudes = rng.uniform(0.1, 2.0, n)

    phase_a = rng.uniform(-math.pi, math.pi, n)
    phase_b = rng.uniform(-math.pi, math.pi, n)
    weights_a = magnitudes * np.exp(1j * phase_a)
    weights_b = magnitudes * np.exp(1j * phase_b)

    spp_a = spp_quadratic(weights_a, cq)
    spp_b = spp_quadratic(weights_b, cq)
    assert abs(spp_a - spp_b) / max(abs(spp_a), 1.0e-300) < 1.0e-14


def test_rank_one_covariance_recovers_uniform_linear_array_factor():
    n_sources = 9
    ks = 1.7
    source_amplitudes = np.ones(n_sources)
    cq = rank_one_covariance(source_amplitudes)
    source_indices = np.arange(n_sources)
    angles = np.linspace(-math.pi / 2.0, math.pi / 2.0, 181)

    model = []
    analytic = []
    for theta in angles:
        psi = ks * math.sin(theta)
        weights = np.exp(1j * source_indices * psi)
        model.append(spp_quadratic(weights, cq, FlowConfig(observer_radius=1.0)))
        analytic.append(abs(np.sum(np.exp(-1j * source_indices * psi))) ** 2)

    model = np.asarray(model) / max(model)
    analytic = np.asarray(analytic) / max(analytic)
    assert float(np.max(np.abs(model - analytic))) < 1.0e-12


def test_offdiagonal_interference_bound_is_never_violated():
    rng = np.random.default_rng(21)
    for _ in range(100):
        n = 12
        raw = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        cq = raw @ raw.conj().T
        weights = rng.normal(size=n) + 1j * rng.normal(size=n)
        off = cq - diagonal_covariance(cq)
        delta = abs(np.vdot(weights, off @ weights))
        bound = np.linalg.norm(off, "fro") * float(np.vdot(weights, weights).real)
        assert delta <= bound * (1.0 + 1.0e-12)


def test_radiating_phase_index_tracks_phase_ablation_directionally():
    flow = FlowConfig()
    observers = ObserverGrid.spherical(5, 8)
    grid = source_grid(default_geometry(), n_eta=12)

    low_coherence = ClosureParams(
        coherence_model="exponential",
        coherence_x=0.01,
        coherence_y=0.01,
        coherence_z=0.01,
    )
    high_coherence = ClosureParams(
        coherence_model="exponential",
        coherence_x=20.0,
        coherence_y=20.0,
        coherence_z=20.0,
    )

    low = mechanism_metrics(grid, observers, 1000.0, flow, low_coherence)
    high = mechanism_metrics(grid, observers, 1000.0, flow, high_coherence)

    assert high.radiating_phase_steerability > low.radiating_phase_steerability
    assert high.phase_ablation_rmse_db > low.phase_ablation_rmse_db


def test_raw_and_radiating_indices_are_bounded_for_physical_case():
    flow = FlowConfig()
    observers = ObserverGrid.spherical(4, 6)
    grid = source_grid(default_geometry(), n_eta=8)
    cq = source_csd_matrix(grid, 1000.0, flow, ClosureParams())
    operator = radiation_operator(grid, observers, 1000.0, flow)

    p0 = raw_coherence_availability(cq)
    pr = radiating_phase_steerability(cq, operator)
    assert 0.0 <= p0 <= 1.0 + 1.0e-12
    assert 0.0 <= pr <= 1.0 + 1.0e-12
