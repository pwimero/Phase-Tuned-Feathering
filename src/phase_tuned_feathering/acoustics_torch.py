import math
import torch

from .closures import FlowConfig, ClosureParams
from .observers import ObserverGrid


@torch.no_grad()
def evaluate_spp_torch(
    points: torch.Tensor,               # (P, N, 3)
    chords: torch.Tensor,               # (P, N)
    incidence_deg: torch.Tensor,        # (P, N)
    loading_directions: torch.Tensor,   # (P, N, 3)
    quad_weights: torch.Tensor,         # (P, N)
    observers: ObserverGrid | None,
    frequencies_hz: float | torch.Tensor,
    flow: FlowConfig | None = None,
    closures: ClosureParams | None = None,
    device: torch.device | None = None,
    observer_dirs: torch.Tensor | None = None,
    scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Evaluates the Spp acoustic metric over a batch of P array geometries.
    Returns:
      - (P, O) when one frequency is provided
      - (P, F, O) when a frequency tensor with F values is provided
    """
    if flow is None:
        flow = FlowConfig()
    if closures is None:
        closures = ClosureParams()
    if device is None:
        device = points.device

    P, N, _ = points.shape
    if observer_dirs is None:
        if observers is None:
            raise ValueError("Either observers or observer_dirs must be provided.")
        observer_dirs = torch.as_tensor(observers.directions, dtype=points.dtype, device=device)
    else:
        observer_dirs = observer_dirs.to(device=device, dtype=points.dtype)
    O = observer_dirs.shape[0]

    frequencies = torch.as_tensor(frequencies_hz, dtype=points.dtype, device=device).reshape(-1)
    F = frequencies.numel()
    omega = (2.0 * math.pi * frequencies).view(1, F, 1)
    u_c = closures.u_c if closures.u_c is not None else closures.beta * flow.u_inf

    # 1. Source Autospectra
    chord_term = chords.unsqueeze(1)
    chi = omega * chord_term * closures.strouhal_scale / u_c  # (P, F, N)

    freq_shape = chi**2 / ((1.0 + chi**2)**(7.0 / 3.0))
    freq_shape = torch.where(chi == 0.0, torch.zeros_like(chi), freq_shape)

    alpha = torch.deg2rad(incidence_deg)
    alpha_ref = math.radians(closures.incidence_ref_deg)
    amplitude_factor = 1.0 + closures.incidence_amplitude_coeff * (alpha - alpha_ref) ** 2

    autospectra = (
        closures.cq
        * (flow.rho0 ** 2)
        * (flow.u_inf ** 5)
        * (chords.unsqueeze(1) ** 2)
        * freq_shape
        * amplitude_factor.unsqueeze(1)
    )
    autospectra = torch.clamp(autospectra, min=0.0)  # (P, F, N)
    auto_sqrt = torch.sqrt(autospectra)  # (P, F, N)

    # 2. Coherence Matrix Setup (Skipped for zero-coherence fast path)
    model = closures.coherence_model.lower()
    
    if model != "zero":
        delay = closures.incidence_delay_per_rad * (alpha - alpha_ref)  # (P, N)
        if scales is None:
            scales = torch.tensor(
                [closures.coherence_x, closures.coherence_y, closures.coherence_z],
                dtype=points.dtype,
                device=device,
            ).view(1, 1, 1, 3)
        else:
            scales = scales.to(device=device, dtype=points.dtype).view(1, 1, 1, 3)

        diff = (points.unsqueeze(2) - points.unsqueeze(1)) / scales  # (P, N, N, 3)
        normalized_distance = torch.sqrt(torch.sum(diff * diff, dim=-1))  # (P, N, N)
        delay_diff = delay.unsqueeze(2) - delay.unsqueeze(1)  # (P, N, N)

        if model == "exponential":
            frequency_factor = (
                frequencies.view(1, F, 1, 1)
                / max(closures.coherence_ref_hz, 1.0e-12)
            )
            magnitude = torch.exp(-frequency_factor * normalized_distance.unsqueeze(1))
        elif model == "full":
            magnitude = torch.ones((P, F, N, N), dtype=points.dtype, device=device)
        else:
            raise ValueError("Invalid coherence model")

        phase_diff = omega.view(1, F, 1, 1) * delay_diff.unsqueeze(1)  # (P, F, N, N)
        gamma_real = magnitude * torch.cos(phase_diff)
        gamma_imag = magnitude * torch.sin(phase_diff)

    # 3. Transfer Weights for each observer
    mach = flow.u_inf / flow.c0
    denominator = torch.clamp((1.0 - mach * observer_dirs[:, 0]) ** 2, min=1.0e-12)
    kernel = (omega.view(F, 1) / flow.c0) / denominator.view(1, O)  # (F, O)

    # Use hardware-accelerated batch matrix multiplication
    loading_proj = loading_directions @ observer_dirs.T  # (P, N, O)
    point_proj = points @ observer_dirs.T  # (P, N, O)
    wave_number = omega.view(F, 1, 1) / flow.c0
    phase = wave_number.unsqueeze(0) * point_proj.unsqueeze(1)  # (P, F, N, O)

    # Pre-combine chords and quadrature weights in 2D space before unsqueezing
    chords_weights = (chords * quad_weights).unsqueeze(1).unsqueeze(-1)  # (P, 1, N, 1)
    magnitude_w = (
        kernel.unsqueeze(0).unsqueeze(2)
        * chords_weights
        * loading_proj.unsqueeze(1)
    )
    w_real = magnitude_w * torch.cos(phase)
    w_imag = magnitude_w * torch.sin(phase)

    # 4. Quadratic Form Spp = Re[w^H Cq w]
    # Mathematically exact tensor scaling refactoring (scales w directly by auto_sqrt)
    w_scaled_real = w_real * auto_sqrt.unsqueeze(-1)
    w_scaled_imag = w_imag * auto_sqrt.unsqueeze(-1)

    if model == "zero":
        # Fast path: Bypasses all (P, F, N, N) allocations and matrix multiplications
        spp_real = torch.sum(w_scaled_real * w_scaled_real + w_scaled_imag * w_scaled_imag, dim=2)
    else:
        # Use hardware-accelerated batched torch.matmul
        cq_w_real = (
            torch.matmul(gamma_real, w_scaled_real)
            - torch.matmul(gamma_imag, w_scaled_imag)
        )
        cq_w_imag = (
            torch.matmul(gamma_real, w_scaled_imag)
            + torch.matmul(gamma_imag, w_scaled_real)
        )
        spp_real = torch.sum(w_scaled_real * cq_w_real + w_scaled_imag * cq_w_imag, dim=2)

    radius = max(flow.observer_radius, 1.0e-12)
    factor = 1.0 / ((4.0 * math.pi * radius) ** 2)
    spp_real = torch.clamp(spp_real, min=0.0) * factor
    if F == 1:
        return spp_real[:, 0, :]
    return spp_real
