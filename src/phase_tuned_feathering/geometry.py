"""Shared geometry rules for Fusion CAD and the research model.

All values in this module are SI units. The original Fusion script used
centimeters directly because Fusion 360's API interprets real-valued geometry
inputs as centimeters. The default values below are therefore the current
Fusion constants divided by 100.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable, Sequence


M_TO_CM = 100.0
CM_TO_M = 0.01


def meters_to_cm(value: float) -> float:
    return value * M_TO_CM


def cm_to_meters(value: float) -> float:
    return value * CM_TO_M


def _as_tuple(values: Iterable[float]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


def _unit(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = _norm(vector)
    if length <= 0.0:
        raise ValueError("Cannot normalize a zero-length vector.")
    return tuple(component / length for component in vector)  # type: ignore[return-value]


@dataclass(frozen=True)
class WingGeometryParams:
    """Parametric feathered half-wing geometry in SI units."""

    center_wing_chord: float = 0.30
    wing_1_chord: float = 0.24
    wing_7_chord: float = 0.18
    half_span: float = 1.60
    y_spacing_scale: float = 1.10
    additional_wings: int = 6
    mid_wing_span_scale: float = 2.0
    wing_1_tip_sweep: float = -0.55
    wing_7_tip_sweep: float = 0.20
    sweep_section_fractions: tuple[float, ...] = (
        0.0,
        0.25,
        0.5,
        0.75,
        0.8,
        0.9,
        0.95,
        0.98,
        0.99,
    )
    sweep_curve_exponent: float = 2.5
    wing_1_root_z_translation: float = 0.12
    wing_7_root_z_translation: float = -0.12
    wing_1_tip_z_curve: float = 0.50
    wing_7_tip_z_curve: float = -0.20
    z_curve_exponent: float = 2.5
    thickness_ratio: float = 0.12
    point_count: int = 80
    component_name: str = "NACA0012 Half Wing Single Root Airfoil"
    root_span_station: float = 0.0
    trailing_edge_thickness: float = 0.005
    max_trailing_edge_thickness_to_chord: float = 0.08
    min_tip_chord_scale: float = 0.12
    feather_incidence_deg: float = 6.0
    root_foil_incidence_deg: float = 6.0
    min_feather_root_gap: float = 0.03
    root_airfoil_length: float = 0.35
    root_feather_gap: float = 0.40
    root_airfoil_chord_scale: float = 1.0
    root_airfoil_y_translation: float = 0.0
    transition_sections: int = 3
    transition_profile_sample_count: int = 100
    transition_fairing_wall_thickness: float = 0.0
    transition_fairing_footprint_margin: float = 0.005
    unify_bodies_for_export: bool = True
    per_feather_incidence_deg: tuple[float, ...] | None = None

    @property
    def total_wings(self) -> int:
        return self.additional_wings + 1

    def incidence_angles_deg(self) -> tuple[float, ...]:
        if self.per_feather_incidence_deg is None:
            return tuple(self.feather_incidence_deg for _ in range(self.total_wings))
        values = _as_tuple(self.per_feather_incidence_deg)
        if len(values) != self.total_wings:
            raise ValueError(
                "per_feather_incidence_deg must match total_wings "
                f"({self.total_wings})."
            )
        return values

    def with_incidence_angles(self, incidence_deg: Sequence[float]) -> "WingGeometryParams":
        return replace(self, per_feather_incidence_deg=_as_tuple(incidence_deg))

    @property
    def fusion_incidence_pivot_fraction_from_le(self) -> float:
        """Current Fusion rotation pivot as a fraction from leading edge to TE.

        The CAD script uses ``pivot_y = trailing_edge_y - 0.75 * chord``. In
        standard airfoil convention, where 0 is the leading edge and 1 is the
        trailing edge, that is 0.25 chord. This property makes the convention
        explicit so the source-line model does not hide the CAD assumption.
        """

        return 0.25


@dataclass(frozen=True)
class FeatherRoot:
    feather_index: int
    chord: float
    trailing_edge_y: float
    z: float
    root_gap: float
    incidence_deg: float


@dataclass(frozen=True)
class FeatherSection:
    """A section of a single feather at span fraction eta."""

    feather_index: int
    eta: float
    chord: float
    span_station: float
    trailing_edge_y: float
    z: float
    incidence_deg: float
    sweep_offset: float
    z_offset: float
    chord_scale: float
    tip_span_station: float
    tip_sweep_offset: float
    tip_z_curve_offset: float

    def fusion_midline_point(
        self,
        source_chord_fraction: float = 1.0,
    ) -> tuple[float, float, float]:
        """Return a source point in Fusion coordinates.

        ``source_chord_fraction`` follows standard airfoil convention:
        0.0 is the leading edge and 1.0 is the trailing edge. The current Fusion
        incidence pivot is therefore 0.25, because the CAD script rotates about
        ``trailing_edge_y - 0.75 * chord``.
        """

        if source_chord_fraction < 0.0 or source_chord_fraction > 1.0:
            raise ValueError("source_chord_fraction must be between 0 and 1.")
        y = self.trailing_edge_y - self.chord * (1.0 - source_chord_fraction)
        y_rot, z_rot = rotate_yz_about_incidence_pivot(
            y,
            self.z,
            self.trailing_edge_y,
            self.z,
            self.chord,
            self.incidence_deg,
        )
        return (self.span_station, y_rot, z_rot)

    def paper_midline_point(
        self,
        source_chord_fraction: float = 1.0,
    ) -> tuple[float, float, float]:
        """Return source point in paper coordinates.

        Mapping: paper x = Fusion Y, paper y = Fusion X, paper z = Fusion Z.
        """

        fusion_x, fusion_y, fusion_z = self.fusion_midline_point(source_chord_fraction)
        return (fusion_y, fusion_x, fusion_z)

    def loading_direction(self) -> tuple[float, float, float]:
        """Approximate local dipole/loading direction in paper coordinates."""

        angle = math.radians(self.incidence_deg)
        return _unit((math.sin(angle), 0.0, math.cos(angle)))


@dataclass(frozen=True)
class SourceGrid:
    points: tuple[tuple[float, float, float], ...]
    weights: tuple[float, ...]
    feather_ids: tuple[int, ...]
    etas: tuple[float, ...]
    chords: tuple[float, ...]
    incidence_deg: tuple[float, ...]
    loading_directions: tuple[tuple[float, float, float], ...]

    @property
    def n(self) -> int:
        return len(self.points)


def default_geometry() -> WingGeometryParams:
    return WingGeometryParams()


def validate_geometry(params: WingGeometryParams) -> None:
    if params.center_wing_chord <= 0.0 or params.half_span <= 0.0:
        raise ValueError("center_wing_chord and half_span must be positive.")
    if params.wing_1_chord <= 0.0 or params.wing_7_chord <= 0.0:
        raise ValueError("edge wing chords must be positive.")
    if params.additional_wings < 0:
        raise ValueError("additional_wings must be zero or greater.")
    if params.mid_wing_span_scale < 1.0:
        raise ValueError("mid_wing_span_scale must be at least 1.0.")
    if params.sweep_curve_exponent <= 0.0 or params.z_curve_exponent <= 0.0:
        raise ValueError("curve exponents must be positive.")
    if params.min_tip_chord_scale <= 0.0:
        raise ValueError("min_tip_chord_scale must be positive.")
    if params.y_spacing_scale < 0.0:
        raise ValueError("y_spacing_scale must be non-negative.")
    if params.root_airfoil_length <= 0.0:
        raise ValueError("root_airfoil_length must be positive.")
    if params.root_airfoil_chord_scale <= 0.0:
        raise ValueError("root_airfoil_chord_scale must be positive.")
    if params.min_feather_root_gap < 0.0:
        raise ValueError("min_feather_root_gap must be non-negative.")
    if any(value < 0.0 or value > 1.0 for value in params.sweep_section_fractions):
        raise ValueError("sweep_section_fractions must stay in [0, 1].")
    if tuple(sorted(params.sweep_section_fractions)) != params.sweep_section_fractions:
        raise ValueError("sweep_section_fractions must be sorted.")
    params.incidence_angles_deg()


def wing_chord(params: WingGeometryParams, feather_index: int) -> float:
    validate_geometry(params)
    total = params.total_wings
    if feather_index < 1 or feather_index > total:
        raise ValueError("feather_index is out of range.")
    if total == 1:
        return params.center_wing_chord

    midpoint = (total + 1) / 2.0
    if feather_index <= midpoint:
        progress = (midpoint - feather_index) / max(midpoint - 1.0, 1.0)
        return params.center_wing_chord + (
            params.wing_1_chord - params.center_wing_chord
        ) * progress

    progress = (feather_index - midpoint) / max(total - midpoint, 1.0)
    return params.center_wing_chord + (
        params.wing_7_chord - params.center_wing_chord
    ) * progress


def tip_span_station(params: WingGeometryParams, feather_index: int) -> float:
    validate_geometry(params)
    total = params.total_wings
    if total == 1:
        return params.half_span
    midpoint = (total + 1) / 2.0
    edge_distance = midpoint - 1.0
    normalized_distance = abs(feather_index - midpoint) / edge_distance
    scale = params.mid_wing_span_scale - (
        params.mid_wing_span_scale - 1.0
    ) * normalized_distance
    return params.half_span * scale


def tip_sweep_offset(params: WingGeometryParams, feather_index: int) -> float:
    validate_geometry(params)
    total = params.total_wings
    if total == 1:
        return 0.0
    midpoint = (total + 1) / 2.0
    if feather_index <= midpoint:
        progress = (midpoint - feather_index) / max(midpoint - 1.0, 1.0)
        return params.wing_1_tip_sweep * progress
    progress = (feather_index - midpoint) / max(total - midpoint, 1.0)
    return params.wing_7_tip_sweep * progress


def tip_z_curve_offset(params: WingGeometryParams, feather_index: int) -> float:
    validate_geometry(params)
    total = params.total_wings
    if total == 1:
        return 0.0
    midpoint = (total + 1) / 2.0
    if feather_index <= midpoint:
        progress = (midpoint - feather_index) / max(midpoint - 1.0, 1.0)
        return params.wing_1_tip_z_curve * progress
    progress = (feather_index - midpoint) / max(total - midpoint, 1.0)
    return params.wing_7_tip_z_curve * progress


def root_z_translation(params: WingGeometryParams, feather_index: int) -> float:
    validate_geometry(params)
    total = params.total_wings
    if total == 1:
        return 0.0
    progress = (feather_index - 1) / max(total - 1, 1)
    return params.wing_1_root_z_translation + (
        params.wing_7_root_z_translation - params.wing_1_root_z_translation
    ) * progress


def curved_sweep_offset(
    params: WingGeometryParams,
    total_tip_sweep: float,
    span_fraction: float,
) -> float:
    if span_fraction < 0.0 or span_fraction > 1.0:
        raise ValueError("span_fraction must be between 0 and 1.")
    return total_tip_sweep * (span_fraction ** params.sweep_curve_exponent)


def curved_z_offset(
    params: WingGeometryParams,
    total_tip_z_curve: float,
    span_fraction: float,
) -> float:
    if span_fraction < 0.0 or span_fraction > 1.0:
        raise ValueError("span_fraction must be between 0 and 1.")
    return total_tip_z_curve * (span_fraction ** params.z_curve_exponent)


def chord_scale(params: WingGeometryParams, span_fraction: float) -> float:
    rounding_start_fraction = 0.8
    if span_fraction < rounding_start_fraction:
        return 1.0
    progress = (span_fraction - rounding_start_fraction) / (
        1.0 - rounding_start_fraction
    )
    scale = math.sqrt(max(1.0 - progress * progress, 0.0))
    return max(scale, params.min_tip_chord_scale)


def feather_root_gap(params: WingGeometryParams, chord: float) -> float:
    nominal_gap = chord * max(params.y_spacing_scale - 1.0, 0.0)
    return max(nominal_gap, params.min_feather_root_gap)


def feather_root_properties(params: WingGeometryParams) -> tuple[FeatherRoot, ...]:
    validate_geometry(params)
    roots: list[FeatherRoot] = []
    incidences = params.incidence_angles_deg()
    for index in range(1, params.total_wings + 1):
        chord = wing_chord(params, index)
        if index == 1:
            trailing_edge_y = 0.0
            root_gap = 0.0
        else:
            root_gap = feather_root_gap(params, chord)
            trailing_edge_y = roots[index - 2].trailing_edge_y + chord + root_gap
        roots.append(
            FeatherRoot(
                feather_index=index,
                chord=chord,
                trailing_edge_y=trailing_edge_y,
                z=root_z_translation(params, index),
                root_gap=root_gap,
                incidence_deg=incidences[index - 1],
            )
        )
    return tuple(roots)


def rotate_yz_about_incidence_pivot(
    y: float,
    z: float,
    trailing_edge_y: float,
    pivot_z: float,
    chord: float,
    incidence_deg: float,
) -> tuple[float, float]:
    if incidence_deg == 0.0:
        return y, z
    pivot_y = trailing_edge_y - 0.75 * chord
    angle = -math.radians(incidence_deg)
    dy = y - pivot_y
    dz = z - pivot_z
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return (
        pivot_y + dy * cos_a - dz * sin_a,
        pivot_z + dy * sin_a + dz * cos_a,
    )


def feather_section(
    params: WingGeometryParams,
    feather_index: int,
    span_fraction: float,
) -> FeatherSection:
    validate_geometry(params)
    roots = feather_root_properties(params)
    root = roots[feather_index - 1]
    tip_span = tip_span_station(params, feather_index)
    tip_sweep = tip_sweep_offset(params, feather_index)
    tip_z = tip_z_curve_offset(params, feather_index)
    scale = chord_scale(params, span_fraction)
    current_chord = root.chord * scale
    base_y = root.trailing_edge_y + curved_sweep_offset(params, tip_sweep, span_fraction)
    y_adjustment = 0.5 * (root.chord - current_chord)
    section_y = base_y - y_adjustment
    section_z = root.z + curved_z_offset(params, tip_z, span_fraction)
    return FeatherSection(
        feather_index=feather_index,
        eta=span_fraction,
        chord=current_chord,
        span_station=tip_span * span_fraction,
        trailing_edge_y=section_y,
        z=section_z,
        incidence_deg=root.incidence_deg,
        sweep_offset=section_y - root.trailing_edge_y,
        z_offset=section_z - root.z,
        chord_scale=scale,
        tip_span_station=tip_span,
        tip_sweep_offset=tip_sweep,
        tip_z_curve_offset=tip_z,
    )


def feather_sections(
    params: WingGeometryParams,
    span_fractions: Sequence[float] | None = None,
) -> tuple[FeatherSection, ...]:
    fractions = (
        params.sweep_section_fractions if span_fractions is None else tuple(span_fractions)
    )
    return tuple(
        feather_section(params, feather_index, eta)
        for feather_index in range(1, params.total_wings + 1)
        for eta in fractions
    )


def _trapezoid_etas(n_eta: int) -> tuple[float, ...]:
    if n_eta < 2:
        raise ValueError("n_eta must be at least 2 for line quadrature.")
    return tuple(index / (n_eta - 1) for index in range(n_eta))


def _trapezoid_eta_weight(index: int, n_eta: int) -> float:
    delta = 1.0 / (n_eta - 1)
    return 0.5 * delta if index == 0 or index == n_eta - 1 else delta


def _source_point(
    params: WingGeometryParams,
    feather_index: int,
    eta: float,
    source_chord_fraction: float,
) -> tuple[float, float, float]:
    return feather_section(params, feather_index, eta).paper_midline_point(
        source_chord_fraction
    )


def _arc_length_jacobian(
    params: WingGeometryParams,
    feather_index: int,
    eta: float,
    source_chord_fraction: float,
) -> float:
    h = 1e-5
    if eta <= h:
        p0 = _source_point(params, feather_index, eta, source_chord_fraction)
        p1 = _source_point(params, feather_index, eta + h, source_chord_fraction)
        return _norm(tuple((p1[i] - p0[i]) / h for i in range(3)))  # type: ignore[arg-type]
    if eta >= 1.0 - h:
        p0 = _source_point(params, feather_index, eta - h, source_chord_fraction)
        p1 = _source_point(params, feather_index, eta, source_chord_fraction)
        return _norm(tuple((p1[i] - p0[i]) / h for i in range(3)))  # type: ignore[arg-type]
    p0 = _source_point(params, feather_index, eta - h, source_chord_fraction)
    p1 = _source_point(params, feather_index, eta + h, source_chord_fraction)
    return _norm(tuple((p1[i] - p0[i]) / (2.0 * h) for i in range(3)))  # type: ignore[arg-type]


def source_grid(
    params: WingGeometryParams | None = None,
    n_eta: int = 64,
    source_chord_fraction: float = 1.0,
) -> SourceGrid:
    """Build the flattened acoustic source grid.

    By default, sources lie on the rotated trailing-edge midline of each
    feather. Use ``source_chord_fraction=0.25`` to place the source on the
    current Fusion incidence pivot.
    """

    params = default_geometry() if params is None else params
    validate_geometry(params)
    if source_chord_fraction < 0.0 or source_chord_fraction > 1.0:
        raise ValueError("source_chord_fraction must be between 0 and 1.")

    etas = _trapezoid_etas(n_eta)
    points: list[tuple[float, float, float]] = []
    weights: list[float] = []
    feather_ids: list[int] = []
    eta_values: list[float] = []
    chords: list[float] = []
    incidence_values: list[float] = []
    directions: list[tuple[float, float, float]] = []

    for feather_index in range(1, params.total_wings + 1):
        for eta_index, eta in enumerate(etas):
            section = feather_section(params, feather_index, eta)
            points.append(section.paper_midline_point(source_chord_fraction))
            jacobian = _arc_length_jacobian(
                params, feather_index, eta, source_chord_fraction
            )
            weights.append(_trapezoid_eta_weight(eta_index, n_eta) * jacobian)
            feather_ids.append(feather_index)
            eta_values.append(eta)
            chords.append(section.chord)
            incidence_values.append(section.incidence_deg)
            directions.append(section.loading_direction())

    return SourceGrid(
        points=tuple(points),
        weights=tuple(weights),
        feather_ids=tuple(feather_ids),
        etas=tuple(eta_values),
        chords=tuple(chords),
        incidence_deg=tuple(incidence_values),
        loading_directions=tuple(directions),
    )


def fusion_parameter_values(params: WingGeometryParams | None = None) -> dict[str, object]:
    """Return Fusion-script constants in centimeter units."""

    params = default_geometry() if params is None else params
    validate_geometry(params)
    return {
        "CENTER_WING_CHORD": meters_to_cm(params.center_wing_chord),
        "WING_1_CHORD": meters_to_cm(params.wing_1_chord),
        "WING_7_CHORD": meters_to_cm(params.wing_7_chord),
        "HALF_SPAN": meters_to_cm(params.half_span),
        "Y_SPACING_SCALE": params.y_spacing_scale,
        "ADDITIONAL_WINGS": params.additional_wings,
        "MID_WING_SPAN_SCALE": params.mid_wing_span_scale,
        "WING_1_TIP_SWEEP": meters_to_cm(params.wing_1_tip_sweep),
        "WING_7_TIP_SWEEP": meters_to_cm(params.wing_7_tip_sweep),
        "SWEEP_SECTION_FRACTIONS": params.sweep_section_fractions,
        "SWEEP_CURVE_EXPONENT": params.sweep_curve_exponent,
        "WING_1_ROOT_Z_TRANSLATION": meters_to_cm(params.wing_1_root_z_translation),
        "WING_7_ROOT_Z_TRANSLATION": meters_to_cm(params.wing_7_root_z_translation),
        "WING_1_TIP_Z_CURVE": meters_to_cm(params.wing_1_tip_z_curve),
        "WING_7_TIP_Z_CURVE": meters_to_cm(params.wing_7_tip_z_curve),
        "Z_CURVE_EXPONENT": params.z_curve_exponent,
        "THICKNESS_RATIO": params.thickness_ratio,
        "POINT_COUNT": params.point_count,
        "COMPONENT_NAME": params.component_name,
        "ROOT_SPAN_STATION": meters_to_cm(params.root_span_station),
        "TRAILING_EDGE_THICKNESS": meters_to_cm(params.trailing_edge_thickness),
        "MAX_TRAILING_EDGE_THICKNESS_TO_CHORD": (
            params.max_trailing_edge_thickness_to_chord
        ),
        "MIN_TIP_CHORD_SCALE": params.min_tip_chord_scale,
        "FEATHER_INCIDENCE_DEG": params.feather_incidence_deg,
        "ROOT_FOIL_INCIDENCE_DEG": params.root_foil_incidence_deg,
        "MIN_FEATHER_ROOT_GAP": meters_to_cm(params.min_feather_root_gap),
        "ROOT_AIRFOIL_LENGTH": meters_to_cm(params.root_airfoil_length),
        "ROOT_FEATHER_GAP": meters_to_cm(params.root_feather_gap),
        "ROOT_AIRFOIL_CHORD_SCALE": params.root_airfoil_chord_scale,
        "ROOT_AIRFOIL_Y_TRANSLATION": meters_to_cm(
            params.root_airfoil_y_translation
        ),
        "TRANSITION_SECTIONS": params.transition_sections,
        "TRANSITION_PROFILE_SAMPLE_COUNT": params.transition_profile_sample_count,
        "TRANSITION_FAIRING_WALL_THICKNESS": meters_to_cm(
            params.transition_fairing_wall_thickness
        ),
        "TRANSITION_FAIRING_FOOTPRINT_MARGIN": meters_to_cm(
            params.transition_fairing_footprint_margin
        ),
        "UNIFY_BODIES_FOR_EXPORT": params.unify_bodies_for_export,
    }


def section_to_fusion_cm(section: FeatherSection) -> dict[str, float]:
    return {
        "chord": meters_to_cm(section.chord),
        "span_station": meters_to_cm(section.span_station),
        "trailing_edge_y": meters_to_cm(section.trailing_edge_y),
        "z": meters_to_cm(section.z),
        "incidence_deg": section.incidence_deg,
        "sweep_offset": meters_to_cm(section.sweep_offset),
        "z_offset": meters_to_cm(section.z_offset),
        "tip_span_station": meters_to_cm(section.tip_span_station),
        "tip_sweep_offset": meters_to_cm(section.tip_sweep_offset),
        "tip_z_curve_offset": meters_to_cm(section.tip_z_curve_offset),
    }


def root_to_fusion_cm(root: FeatherRoot) -> dict[str, float]:
    return {
        "chord": meters_to_cm(root.chord),
        "y": meters_to_cm(root.trailing_edge_y),
        "z": meters_to_cm(root.z),
        "root_gap": meters_to_cm(root.root_gap),
        "incidence_deg": root.incidence_deg,
    }
