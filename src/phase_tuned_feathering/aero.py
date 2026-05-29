"""Low-cost aerodynamic feasibility screening.

This module deliberately does not perform aeroacoustic validation. It provides
fast constraints and sanity checks for the model-first paper workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .closures import FlowConfig
from .geometry import WingGeometryParams, feather_root_properties


@dataclass(frozen=True)
class AeroScreenResult:
    feasible: bool
    issues: tuple[str, ...]
    values: dict[str, float]
    method: str


def _thin_airfoil_proxy(alpha_deg: float) -> tuple[float, float]:
    alpha_rad = math.radians(alpha_deg)
    cl = 2.0 * math.pi * alpha_rad
    cd = 0.010 + 0.020 * cl * cl
    return cl, cd


def screen_aero(
    params: WingGeometryParams,
    flow: FlowConfig | None = None,
    stall_angle_deg: float = 14.0,
    stall_margin_deg: float = 2.0,
    max_abs_incidence_deg: float = 12.0,
    min_reynolds: float = 5.0e4,
    max_reynolds: float = 5.0e6,
    min_root_gap: float | None = None,
) -> AeroScreenResult:
    flow = FlowConfig() if flow is None else flow
    roots = feather_root_properties(params)
    method = "thin-airfoil-proxy"
    issues: list[str] = []
    values: dict[str, float] = {}

    cl_values: list[float] = []
    cd_values: list[float] = []
    reynolds_values: list[float] = []
    for root in roots:
        if abs(root.incidence_deg) > max_abs_incidence_deg:
            issues.append(
                f"feather {root.feather_index} incidence exceeds "
                f"{max_abs_incidence_deg:.1f} deg"
            )
        if abs(root.incidence_deg) > stall_angle_deg - stall_margin_deg:
            issues.append(
                f"feather {root.feather_index} incidence violates stall margin"
            )
        re = flow.rho0 * flow.u_inf * root.chord / max(flow.dynamic_viscosity, 1.0e-12)
        reynolds_values.append(re)
        if re < min_reynolds or re > max_reynolds:
            issues.append(
                f"feather {root.feather_index} Reynolds number outside screen bounds"
            )
        if root.feather_index > 1:
            required_gap = params.min_feather_root_gap if min_root_gap is None else min_root_gap
            if root.root_gap < required_gap:
                issues.append(f"feather {root.feather_index} root gap below minimum")

        cl, cd = _thin_airfoil_proxy(root.incidence_deg)
        cl_values.append(cl)
        cd_values.append(cd)

    values["cl_mean_proxy"] = sum(cl_values) / len(cl_values)
    values["cd_mean_proxy"] = sum(cd_values) / len(cd_values)
    values["cl_cd_proxy"] = values["cl_mean_proxy"] / max(values["cd_mean_proxy"], 1.0e-12)
    values["re_min"] = min(reynolds_values)
    values["re_max"] = max(reynolds_values)
    values["max_abs_incidence_deg"] = max(abs(root.incidence_deg) for root in roots)
    values["min_root_gap"] = min(root.root_gap for root in roots[1:]) if len(roots) > 1 else 0.0

    return AeroScreenResult(
        feasible=not issues,
        issues=tuple(issues),
        values=values,
        method=method,
    )
