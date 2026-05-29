import math
import torch
from dataclasses import dataclass

from .closures import FlowConfig, ClosureParams
from .geometry import SourceGrid
from .observers import ObserverGrid


def evaluate_spp_torch(
    points: torch.Tensor,               # (P, N, 3)
    chords: torch.Tensor,               # (P, N)
    incidence_deg: torch.Tensor,        # (P, N)
    loading_directions: torch.Tensor,   # (P, N, 3)
    quad_weights: torch.Tensor,         # (P, N)
    observers: ObserverGrid,
    frequency_hz: float,
    flow: FlowConfig | None = None,
    closures: ClosureParams | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Evaluates the Spp acoustic metric over a batch of P array geometries.
    Returns: Spp tensor of shape (P, O) where O is the number of observers.
    """
    if flow is None:
        flow = FlowConfig()
    if closures is None:
        closures = ClosureParams()
    if device is None:
        device = points.device

    P, N, _ = points.shape
    O = len(observers.directions)

    omega = 2.0 * math.pi * frequency_hz
    u_c = closures.u_c if closures.u_c is not None else closures.beta * flow.u_inf

    # 1. Source Autospectra
    chi = omega * chords * closures.strouhal_scale / u_c  # (P, N)
    
    # frequency_shape
    freq_shape = chi**2 / ((1.0 + chi**2)**(7.0/3.0))
    freq_shape = torch.where(chi == 0.0, torch.zeros_like(chi), freq_shape)

    # incidence_amplitude_factor
    alpha = torch.deg2rad(incidence_deg)
    alpha_ref = math.radians(closures.incidence_ref_deg)
    amplitude_factor = 1.0 + closures.incidence_amplitude_coeff * (alpha - alpha_ref)**2

    autospectra = (
        closures.cq
        * (flow.rho0**2)
        * (flow.u_inf**5)
        * (chords**2)
        * freq_shape
        * amplitude_factor
    )
    autospectra = torch.clamp(autospectra, min=0.0)  # (P, N)

    # 2. Coherence Matrix
    model = closures.coherence_model.lower()
    
    # Incidence delay
    delay = closures.incidence_delay_per_rad * (alpha - alpha_ref)  # (P, N)
    
    # Distance matrix (P, N, N)
    scales = torch.tensor(
        [closures.coherence_x, closures.coherence_y, closures.coherence_z],
        dtype=points.dtype, device=device
    ).view(1, 1, 1, 3)
    
    # points: (P, N, 3) -> (P, N, 1, 3) and (P, 1, N, 3)
    diff = (points.unsqueeze(2) - points.unsqueeze(1)) / scales  # (P, N, N, 3)
    distance = torch.sum(torch.abs(diff), dim=-1)  # (P, N, N)
    
    if model == "exponential":
        magnitude = torch.exp(-omega * distance / u_c)
    elif model == "full":
        magnitude = torch.ones_like(distance)
    elif model == "zero":
        magnitude = torch.eye(N, device=device).unsqueeze(0).expand(P, N, N)
    else:
        raise ValueError("Invalid coherence model")

    # Phase delay matrix
    phase_diff = omega * (delay.unsqueeze(2) - delay.unsqueeze(1))  # (P, N, N)
    
    gamma = magnitude * torch.polar(torch.ones_like(magnitude), phase_diff)  # (P, N, N) complex

    # Cross-spectral matrix Cq (P, N, N)
    auto_sqrt = torch.sqrt(autospectra)  # (P, N)
    Cq = auto_sqrt.unsqueeze(2) * auto_sqrt.unsqueeze(1) * gamma

    # 3. Transfer Weights for each observer
    obs_dirs = torch.tensor(observers.directions, dtype=points.dtype, device=device) # (O, 3)
    
    # kernel: (O,)
    mach = flow.u_inf / flow.c0
    denominator = torch.clamp((1.0 - mach * obs_dirs[:, 0])**2, min=1.0e-12)
    kernel = (omega / flow.c0) / denominator

    # loading_projection: sum of loading_direction * observer_direction
    # loading_directions: (P, N, 3)
    # obs_dirs: (O, 3)
    # Result: (P, N, O)
    loading_proj = torch.einsum("pni,oi->pno", loading_directions, obs_dirs)

    # phase = wavenumber * dot(point, obs_dir)
    wave_number = omega / flow.c0
    phase = wave_number * torch.einsum("pni,oi->pno", points, obs_dirs)

    # magnitude: (P, N, O)
    mag = kernel.unsqueeze(0).unsqueeze(0) * chords.unsqueeze(-1) * loading_proj * quad_weights.unsqueeze(-1)
    
    # weights: (P, N, O) complex
    w = mag * torch.polar(torch.ones_like(mag), phase)

    # 4. Quadratic Form Spp = w^H Cq w
    # w is (P, N, O)
    # w^H is (P, O, N)
    w_H = torch.conj(w.transpose(1, 2))
    
    # Cq is (P, N, N)
    # Cq @ w -> (P, N, O)
    Cq_w = torch.einsum("pnm,pmo->pno", Cq, w)
    
    # w^H @ (Cq @ w) -> (P, O, O) but we only want the diagonals, or just dot product
    # Spp(o) = sum_n ( w^H(o, n) * Cq_w(n, o) )
    spp_complex = torch.einsum("pon,pno->po", w_H, Cq_w)
    
    # factor
    radius = max(flow.observer_radius, 1.0e-12)
    factor = 1.0 / ((4.0 * math.pi * radius)**2)

    spp_real = torch.clamp(spp_complex.real, min=0.0) * factor
    return spp_real
