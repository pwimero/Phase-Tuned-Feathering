import math
from pathlib import Path
import sys
import traceback

import adsk.core  # type: ignore
import adsk.fusion  # type: ignore


try:
    _SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    _SCRIPT_DIR = Path.cwd()

_SRC_DIR = _SCRIPT_DIR.parents[1] / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from phase_tuned_feathering.geometry import (
    default_geometry,
    feather_root_properties,
    feather_section,
    fusion_parameter_values,
    root_to_fusion_cm,
    section_to_fusion_cm,
)
from phase_tuned_feathering.io import (
    write_geometry_metadata_json,
    write_source_grid_csv,
)
from phase_tuned_feathering.visualization import write_simulator_geometry_renders


GEOMETRY_PARAMS = default_geometry()
_FUSION_VALUES = fusion_parameter_values(GEOMETRY_PARAMS)


# NOTE: Fusion 360's API uses centimeters internally. All numeric values
# in this script are passed directly to the API without conversion.
# The dimensions below are the final physical model size; no downstream
# geometry rescaling is required after STEP export.
# This script creates a half-wing with span along +X.
# Each airfoil section lies in a YZ-parallel plane, with the chord on -Y.
# Wing numbering is front to back: wing 1 is frontmost and wing 7 is backmost.
# The feather roots lie on the global YZ plane at x = 0.
#
# Main-root strategy implemented here:
# - Build one single NACA0012 root airfoil body that spans in X.
# - Its baseline chord is the Y distance between wing 1's leading edge and
#   the last wing's trailing edge.
# - ROOT_AIRFOIL_CHORD_SCALE lets you enlarge that root airfoil uniformly.
#   Because the thickness ratio stays constant, increasing the chord also
#   increases the root airfoil thickness in Z.
# - Build each feather as its own loft from x = 0, then union the bodies into
#   one export-ready solid for inspection and meshing.

CENTER_WING_CHORD = _FUSION_VALUES["CENTER_WING_CHORD"]
WING_1_CHORD = _FUSION_VALUES["WING_1_CHORD"]
WING_7_CHORD = _FUSION_VALUES["WING_7_CHORD"]
HALF_SPAN = _FUSION_VALUES["HALF_SPAN"]
Y_SPACING_SCALE = _FUSION_VALUES["Y_SPACING_SCALE"]
ADDITIONAL_WINGS = _FUSION_VALUES["ADDITIONAL_WINGS"]
MID_WING_SPAN_SCALE = _FUSION_VALUES["MID_WING_SPAN_SCALE"]
WING_1_TIP_SWEEP = _FUSION_VALUES["WING_1_TIP_SWEEP"]
WING_7_TIP_SWEEP = _FUSION_VALUES["WING_7_TIP_SWEEP"]
# Avoid an ultra-tiny final tip section; it was making feather 7 especially
# fragile and prone to bad export triangles.
SWEEP_SECTION_FRACTIONS = _FUSION_VALUES["SWEEP_SECTION_FRACTIONS"]
SWEEP_CURVE_EXPONENT = _FUSION_VALUES["SWEEP_CURVE_EXPONENT"]
WING_1_ROOT_Z_TRANSLATION = _FUSION_VALUES["WING_1_ROOT_Z_TRANSLATION"]
WING_7_ROOT_Z_TRANSLATION = _FUSION_VALUES["WING_7_ROOT_Z_TRANSLATION"]
WING_1_TIP_Z_CURVE = _FUSION_VALUES["WING_1_TIP_Z_CURVE"]
WING_7_TIP_Z_CURVE = _FUSION_VALUES["WING_7_TIP_Z_CURVE"]
Z_CURVE_EXPONENT = _FUSION_VALUES["Z_CURVE_EXPONENT"]
THICKNESS_RATIO = _FUSION_VALUES["THICKNESS_RATIO"]
POINT_COUNT = _FUSION_VALUES["POINT_COUNT"]
COMPONENT_NAME = _FUSION_VALUES["COMPONENT_NAME"]
ROOT_SPAN_STATION = _FUSION_VALUES["ROOT_SPAN_STATION"]

# Meshing-safe settings retained from the successful fix.
TRAILING_EDGE_THICKNESS = _FUSION_VALUES["TRAILING_EDGE_THICKNESS"]
# Prevent very small feathers from inheriting an unrealistically blunt
# absolute trailing edge.
MAX_TRAILING_EDGE_THICKNESS_TO_CHORD = _FUSION_VALUES[
    "MAX_TRAILING_EDGE_THICKNESS_TO_CHORD"
]
MIN_TIP_CHORD_SCALE = _FUSION_VALUES["MIN_TIP_CHORD_SCALE"]
FEATHER_INCIDENCE_DEG = _FUSION_VALUES["FEATHER_INCIDENCE_DEG"]
ROOT_FOIL_INCIDENCE_DEG = _FUSION_VALUES["ROOT_FOIL_INCIDENCE_DEG"]

# Keep the aft feathers from packing too tightly at the root plane.
MIN_FEATHER_ROOT_GAP = _FUSION_VALUES["MIN_FEATHER_ROOT_GAP"]

# Main root settings.
ROOT_AIRFOIL_LENGTH = _FUSION_VALUES["ROOT_AIRFOIL_LENGTH"]

# Gap between the feather roots at x = 0 and the main-root rear face.
# Set > 0 to create a physical separation along the span axis.
ROOT_FEATHER_GAP = _FUSION_VALUES["ROOT_FEATHER_GAP"]

# Increase this above 1.0 to make the main root airfoil larger.
# This scales the chord about the center of the feather Y-footprint, which also
# increases the airfoil thickness because THICKNESS_RATIO stays constant.
ROOT_AIRFOIL_CHORD_SCALE = _FUSION_VALUES["ROOT_AIRFOIL_CHORD_SCALE"]

# Translate the main root airfoil in the Y direction (along the chord). Positive is towards the trailing edge.
ROOT_AIRFOIL_Y_TRANSLATION = _FUSION_VALUES["ROOT_AIRFOIL_Y_TRANSLATION"]

# Number of intermediate sections in each transition loft (more = smoother curve).
TRANSITION_SECTIONS = _FUSION_VALUES["TRANSITION_SECTIONS"]
TRANSITION_PROFILE_SAMPLE_COUNT = _FUSION_VALUES["TRANSITION_PROFILE_SAMPLE_COUNT"]
TRANSITION_FAIRING_WALL_THICKNESS = _FUSION_VALUES[
    "TRANSITION_FAIRING_WALL_THICKNESS"
]
# Clearance applied to the sampled fairing envelope so it stays strictly
# outside the feather-root profiles in both chordwise extent and thickness.
TRANSITION_FAIRING_FOOTPRINT_MARGIN = _FUSION_VALUES[
    "TRANSITION_FAIRING_FOOTPRINT_MARGIN"
]

# Export as a single solid to avoid coincident/interior faces between the root,
# fairing, and feather bodies.
UNIFY_BODIES_FOR_EXPORT = _FUSION_VALUES["UNIFY_BODIES_FOR_EXPORT"]
EXPORT_GEOMETRY_METADATA = True
METADATA_SOURCE_GRID_N_ETA = 64

def _naca0012_half_thickness(chord, x_over_c):
    return 5.0 * THICKNESS_RATIO * chord * (
        0.2969 * math.sqrt(max(x_over_c, 0.0))
        - 0.1260 * x_over_c
        - 0.3516 * x_over_c**2
        + 0.2843 * x_over_c**3
        - 0.1036 * x_over_c**4
    )


def _effective_te_half_thickness(chord):
    capped_te = min(
        TRAILING_EDGE_THICKNESS,
        chord * MAX_TRAILING_EDGE_THICKNESS_TO_CHORD,
    )
    return 0.5 * capped_te


def _to_sketch_space(sketch, x, y, z):
    return sketch.modelToSketchSpace(
        adsk.core.Point3D.create(x, y, z)
    )


def _rotate_yz_about_pivot(y, z, pivot_y, pivot_z, incidence_deg):
    if incidence_deg == 0.0:
        return y, z

    # Positive incidence is nose-up (leading edge moves to +Z).
    angle = -math.radians(incidence_deg)
    dy = y - pivot_y
    dz = z - pivot_z
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return (
        pivot_y + dy * cos_a - dz * sin_a,
        pivot_z + dy * sin_a + dz * cos_a,
    )


def _airfoil_point_collections(
    sketch,
    chord,
    span_station,
    y_offset,
    z_offset,
    point_count=POINT_COUNT,
    incidence_deg=0.0,
):
    upper = adsk.core.ObjectCollection.create()
    lower = adsk.core.ObjectCollection.create()

    te_half_thickness = _effective_te_half_thickness(chord)
    pivot_y = y_offset - 0.75 * chord
    pivot_z = z_offset

    def _make_point(y, z):
        y_rot, z_rot = _rotate_yz_about_pivot(
            y, z, pivot_y, pivot_z, incidence_deg
        )
        return sketch.sketchPoints.add(
            _to_sketch_space(sketch, span_station, y_rot, z_rot)
        )

    trailing_edge_upper = _make_point(
        y_offset,
        z_offset + te_half_thickness,
    )
    trailing_edge_lower = _make_point(
        y_offset,
        z_offset - te_half_thickness,
    )
    leading_edge = _make_point(
        y_offset - chord,
        z_offset,
    )

    upper.add(trailing_edge_upper)
    for index in range(point_count - 1, 0, -1):
        x_over_c = 0.5 * (1.0 - math.cos(math.pi * index / point_count))
        base_thickness = _naca0012_half_thickness(chord, x_over_c)
        thickness = max(base_thickness, te_half_thickness)
        y = y_offset - chord * (1.0 - x_over_c)
        upper.add(_make_point(y, z_offset + thickness))
    upper.add(leading_edge)

    lower.add(leading_edge)
    for index in range(1, point_count):
        x_over_c = 0.5 * (1.0 - math.cos(math.pi * index / point_count))
        base_thickness = _naca0012_half_thickness(chord, x_over_c)
        thickness = max(base_thickness, te_half_thickness)
        y = y_offset - chord * (1.0 - x_over_c)
        lower.add(_make_point(y, z_offset - thickness))
    lower.add(trailing_edge_lower)

    return upper, lower, trailing_edge_upper, trailing_edge_lower


def _create_offset_plane(component, base_plane, offset, name):
    plane_input = component.constructionPlanes.createInput()
    plane_input.setByOffset(base_plane, adsk.core.ValueInput.createByReal(offset))
    plane = component.constructionPlanes.add(plane_input)
    plane.name = name
    return plane


def _create_airfoil_profile(
    component,
    plane,
    sketch_name,
    chord,
    span_station,
    y_offset,
    z_offset,
    incidence_deg=0.0,
):
    sketch = component.sketches.add(plane)
    sketch.name = sketch_name

    if chord < 0.1:
        local_point_count = 20
    elif chord < 1.0:
        local_point_count = 40
    else:
        local_point_count = POINT_COUNT

    upper_points, lower_points, trailing_edge_upper, trailing_edge_lower = _airfoil_point_collections(
        sketch,
        chord,
        span_station,
        y_offset,
        z_offset,
        local_point_count,
        incidence_deg,
    )

    sketch.sketchCurves.sketchFittedSplines.add(upper_points)
    sketch.sketchCurves.sketchFittedSplines.add(lower_points)
    sketch.sketchCurves.sketchLines.addByTwoPoints(trailing_edge_lower, trailing_edge_upper)

    if sketch.profiles.count != 1:
        raise RuntimeError(
            f"Expected exactly 1 profile for {sketch_name}, got {sketch.profiles.count}."
        )

    return sketch.profiles.item(0)


def _join_body_into_target(component, target_body, tool_body, feature_name):
    if target_body == tool_body:
        return target_body

    tool_bodies = adsk.core.ObjectCollection.create()
    tool_bodies.add(tool_body)

    combine_features = component.features.combineFeatures
    combine_input = combine_features.createInput(target_body, tool_bodies)
    combine_input.operation = adsk.fusion.FeatureOperations.JoinFeatureOperation
    combine_input.isKeepToolBodies = False
    combine_feature = combine_features.add(combine_input)
    combine_feature.name = feature_name
    return target_body


def _smoothstep(value):
    clamped = min(max(value, 0.0), 1.0)
    return clamped * clamped * (3.0 - 2.0 * clamped)


def _shared_root_props_cm():
    return [
        root_to_fusion_cm(root)
        for root in feather_root_properties(GEOMETRY_PARAMS)
    ]


def _shared_section_cm(wing_number, span_fraction):
    return section_to_fusion_cm(
        feather_section(GEOMETRY_PARAMS, wing_number, span_fraction)
    )


def _export_geometry_metadata():
    if not EXPORT_GEOMETRY_METADATA:
        return

    write_geometry_metadata_json(
        _SCRIPT_DIR / "geometry_metadata.json",
        GEOMETRY_PARAMS,
    )
    write_source_grid_csv(
        _SCRIPT_DIR / "source_grid.csv",
        GEOMETRY_PARAMS,
        n_eta=METADATA_SOURCE_GRID_N_ETA,
    )
    write_simulator_geometry_renders(
        _SCRIPT_DIR,
        GEOMETRY_PARAMS,
        n_eta=METADATA_SOURCE_GRID_N_ETA,
    )


def _build_single_root_airfoil(component, root_plane, feather_root_props):
    loft_features = component.features.loftFeatures

    wing_1_le_y = feather_root_props[0]['y'] - feather_root_props[0]['chord']
    wing_last_te_y = feather_root_props[-1]['y']

    baseline_root_chord = wing_last_te_y - wing_1_le_y
    baseline_center_y = 0.5 * (wing_1_le_y + wing_last_te_y)

    main_root_chord = baseline_root_chord * ROOT_AIRFOIL_CHORD_SCALE
    main_root_y_offset = baseline_center_y + 0.5 * main_root_chord + ROOT_AIRFOIL_Y_TRANSLATION

    min_root_z = min(p['z'] for p in feather_root_props)
    max_root_z = max(p['z'] for p in feather_root_props)
    main_root_z_offset = 0.5 * (min_root_z + max_root_z)

    root_front_plane = _create_offset_plane(
        component,
        root_plane,
        -ROOT_AIRFOIL_LENGTH,
        "Main Root Front Plane",
    )
    # Root rear face pulled back by the gap so feathers at x=0 are separated.
    root_rear_x = -ROOT_FEATHER_GAP
    root_rear_plane = _create_offset_plane(
        component,
        root_plane,
        root_rear_x,
        "Main Root Rear Plane",
    )

    root_front_profile = _create_airfoil_profile(
        component,
        root_front_plane,
        "Main Root Front Airfoil",
        main_root_chord,
        -ROOT_AIRFOIL_LENGTH,
        main_root_y_offset,
        main_root_z_offset,
        ROOT_FOIL_INCIDENCE_DEG,
    )
    root_rear_profile = _create_airfoil_profile(
        component,
        root_rear_plane,
        "Main Root Rear Airfoil",
        main_root_chord,
        root_rear_x,
        main_root_y_offset,
        main_root_z_offset,
        ROOT_FOIL_INCIDENCE_DEG,
    )

    loft_input = loft_features.createInput(adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
    loft_input.isSolid = True
    loft_input.loftSections.add(root_front_profile)
    loft_input.loftSections.add(root_rear_profile)
    loft = loft_features.add(loft_input)
    loft.name = "Main Root Airfoil Loft"

    if loft.bodies.count != 1:
        raise RuntimeError("Expected a single main-root airfoil body.")

    body = loft.bodies.item(0)
    body.name = "Main Root Airfoil"
    return body



def _fill_missing_samples(values):
    valid_indices = [index for index, value in enumerate(values) if value is not None]
    if not valid_indices:
        raise RuntimeError("Unable to construct transition fairing envelope.")

    filled = list(values)
    first_valid = valid_indices[0]
    first_value = filled[first_valid]
    for index in range(first_valid):
        filled[index] = first_value

    for left_index, right_index in zip(valid_indices, valid_indices[1:]):
        left_value = filled[left_index]
        right_value = filled[right_index]
        gap = right_index - left_index
        if gap <= 1:
            continue

        for step in range(1, gap):
            blend = step / gap
            filled[left_index + step] = left_value + (right_value - left_value) * blend

    last_valid = valid_indices[-1]
    last_value = filled[last_valid]
    for index in range(last_valid + 1, len(filled)):
        filled[index] = last_value

    return filled


def _airfoil_surface_curves(chord, y_offset, z_offset, incidence_deg, point_count=POINT_COUNT):
    pivot_y = y_offset - 0.75 * chord
    pivot_z = z_offset
    te_half_thickness = _effective_te_half_thickness(chord)

    upper = []
    lower = []

    le_y, le_z = _rotate_yz_about_pivot(
        y_offset - chord,
        z_offset,
        pivot_y,
        pivot_z,
        incidence_deg,
    )
    upper.append((le_y, le_z))
    lower.append((le_y, le_z))

    for index in range(1, point_count):
        x_over_c = 0.5 * (1.0 - math.cos(math.pi * index / point_count))
        base_thickness = _naca0012_half_thickness(chord, x_over_c)
        thickness = max(base_thickness, te_half_thickness)
        y = y_offset - chord * (1.0 - x_over_c)

        upper_y, upper_z = _rotate_yz_about_pivot(
            y,
            z_offset + thickness,
            pivot_y,
            pivot_z,
            incidence_deg,
        )
        lower_y, lower_z = _rotate_yz_about_pivot(
            y,
            z_offset - thickness,
            pivot_y,
            pivot_z,
            incidence_deg,
        )
        upper.append((upper_y, upper_z))
        lower.append((lower_y, lower_z))

    te_upper_y, te_upper_z = _rotate_yz_about_pivot(
        y_offset,
        z_offset + te_half_thickness,
        pivot_y,
        pivot_z,
        incidence_deg,
    )
    te_lower_y, te_lower_z = _rotate_yz_about_pivot(
        y_offset,
        z_offset - te_half_thickness,
        pivot_y,
        pivot_z,
        incidence_deg,
    )
    upper.append((te_upper_y, te_upper_z))
    lower.append((te_lower_y, te_lower_z))

    upper.sort(key=lambda point: point[0])
    lower.sort(key=lambda point: point[0])
    return upper, lower


def _curve_z_at_y(curve_points, sample_y):
    if sample_y < curve_points[0][0] or sample_y > curve_points[-1][0]:
        return None

    for (y0, z0), (y1, z1) in zip(curve_points, curve_points[1:]):
        if y0 <= sample_y <= y1:
            dy = y1 - y0
            if abs(dy) < 1e-9:
                return 0.5 * (z0 + z1)
            blend = (sample_y - y0) / dy
            return z0 + (z1 - z0) * blend

    if abs(sample_y - curve_points[-1][0]) < 1e-9:
        return curve_points[-1][1]
    return None


def _create_fairing_profile(component, plane, sketch_name, span_station, section_props):
    sketch = component.sketches.add(plane)
    sketch.name = sketch_name

    section_curves = [
        _airfoil_surface_curves(
            prop['chord'],
            prop['y'],
            prop['z'],
            prop.get('incidence_deg', 0.0),
        )
        for prop in section_props
    ]
    all_y_values = [
        y
        for upper_curve, lower_curve in section_curves
        for y, _ in upper_curve + lower_curve
    ]
    y_min = min(all_y_values)
    y_max = max(all_y_values)
    if y_max <= y_min:
        raise RuntimeError(f"Invalid fairing profile range for {sketch_name}.")

    margin = max(TRANSITION_FAIRING_FOOTPRINT_MARGIN, 0.0)
    y_min -= margin
    y_max += margin

    sample_count = max(TRANSITION_PROFILE_SAMPLE_COUNT, 8)
    y_values = [
        y_min + (y_max - y_min) * index / sample_count
        for index in range(sample_count + 1)
    ]

    upper_values = []
    lower_values = []
    for sample_y in y_values:
        section_upper = None
        section_lower = None

        for upper_curve, lower_curve in section_curves:
            current_upper = _curve_z_at_y(upper_curve, sample_y)
            current_lower = _curve_z_at_y(lower_curve, sample_y)
            if current_upper is None or current_lower is None:
                continue

            if section_upper is None or current_upper > section_upper:
                section_upper = current_upper
            if section_lower is None or current_lower < section_lower:
                section_lower = current_lower

        upper_values.append(section_upper)
        lower_values.append(section_lower)

    upper_values = _fill_missing_samples(upper_values)
    lower_values = _fill_missing_samples(lower_values)
    upper_values = [value + margin for value in upper_values]
    lower_values = [value - margin for value in lower_values]

    leading_edge_z = 0.5 * (upper_values[0] + lower_values[0])
    upper_values[0] = leading_edge_z
    lower_values[0] = leading_edge_z

    upper_points = adsk.core.ObjectCollection.create()
    lower_points = adsk.core.ObjectCollection.create()
    trailing_edge_upper = None
    trailing_edge_lower = None
    leading_edge_upper = None
    leading_edge_lower = None

    for index in range(sample_count, -1, -1):
        point = sketch.sketchPoints.add(
            _to_sketch_space(
                sketch,
                span_station,
                y_values[index],
                upper_values[index],
            )
        )
        if index == sample_count:
            trailing_edge_upper = point
        if index == 0:
            leading_edge_upper = point
        upper_points.add(point)

    for index in range(sample_count + 1):
        point = sketch.sketchPoints.add(
            _to_sketch_space(
                sketch,
                span_station,
                y_values[index],
                lower_values[index],
            )
        )
        if index == 0:
            leading_edge_lower = point
        if index == sample_count:
            trailing_edge_lower = point
        lower_points.add(point)

    sketch.sketchCurves.sketchFittedSplines.add(upper_points)
    sketch.sketchCurves.sketchFittedSplines.add(lower_points)
    if abs(upper_values[0] - lower_values[0]) > 1e-6:
        sketch.sketchCurves.sketchLines.addByTwoPoints(
            leading_edge_upper,
            leading_edge_lower,
        )
    sketch.sketchCurves.sketchLines.addByTwoPoints(
        trailing_edge_lower,
        trailing_edge_upper,
    )

    if sketch.profiles.count != 1:
        raise RuntimeError(
            f"Expected exactly 1 profile for {sketch_name}, got {sketch.profiles.count}."
        )

    return sketch.profiles.item(0)


def _blended_section_props(
    feather_root_props,
    blend,
    start_chord,
    start_y,
    start_z,
    start_incidence_deg=0.0,
):
    section_props = []
    for feather in feather_root_props:
        section_props.append({
            'chord': start_chord + (feather['chord'] - start_chord) * blend,
            'y': start_y + (feather['y'] - start_y) * blend,
            'z': start_z + (feather['z'] - start_z) * blend,
            'incidence_deg': start_incidence_deg + (
                feather.get('incidence_deg', 0.0) - start_incidence_deg
            ) * blend,
        })
    return section_props


def _hollow_transition_fairing(component, fairing_body):
    if TRANSITION_FAIRING_WALL_THICKNESS <= 0.0:
        return

    outboard_face = None
    outboard_x = None
    for face in fairing_body.faces:
        point = face.pointOnFace
        if outboard_face is None or point.x > outboard_x:
            outboard_face = face
            outboard_x = point.x

    if outboard_face is None:
        raise RuntimeError("Unable to find the outboard face for fairing hollowing.")

    shell_entities = adsk.core.ObjectCollection.create()
    shell_entities.add(outboard_face)

    shell_features = component.features.shellFeatures
    shell_input = shell_features.createInput(shell_entities, False)
    shell_input.insideThickness = adsk.core.ValueInput.createByReal(
        TRANSITION_FAIRING_WALL_THICKNESS
    )
    shell_features.add(shell_input)


def _build_transition_fairing(component, root_plane, feather_root_props):
    """Build one smooth fairing body that covers the root-to-feather transition."""
    if ROOT_FEATHER_GAP <= 0.0:
        return None

    loft_features = component.features.loftFeatures
    start_x = -ROOT_AIRFOIL_LENGTH
    transition_span = abs(start_x)

    w1_le_y = feather_root_props[0]['y'] - feather_root_props[0]['chord']
    wN_te_y = feather_root_props[-1]['y']
    base_chord = (wN_te_y - w1_le_y) * ROOT_AIRFOIL_CHORD_SCALE
    base_center_y = 0.5 * (w1_le_y + wN_te_y)
    root_y = base_center_y + 0.5 * base_chord + ROOT_AIRFOIL_Y_TRANSLATION
    min_z = min(p['z'] for p in feather_root_props)
    max_z = max(p['z'] for p in feather_root_props)
    root_z = 0.5 * (min_z + max_z)

    loft_input = loft_features.createInput(
        adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
    loft_input.isSolid = True

    start_plane = _create_offset_plane(
        component,
        root_plane,
        start_x,
        "Transition Fairing Start Plane",
    )
    start_profile = _create_airfoil_profile(
        component,
        start_plane,
        "Transition Fairing Start",
        base_chord,
        start_x,
        root_y,
        root_z,
        ROOT_FOIL_INCIDENCE_DEG,
    )
    loft_input.loftSections.add(start_profile)

    for k in range(1, TRANSITION_SECTIONS + 1):
        t = k / (TRANSITION_SECTIONS + 1)
        blend = _smoothstep(t)
        x = start_x + transition_span * t
        section_props = _blended_section_props(
            feather_root_props,
            blend,
            base_chord,
            root_y,
            root_z,
            ROOT_FOIL_INCIDENCE_DEG,
        )

        section_plane = _create_offset_plane(
            component,
            root_plane,
            x,
            f"Transition Fairing Plane {k}",
        )
        section_profile = _create_fairing_profile(
            component,
            section_plane,
            f"Transition Fairing Section {k}",
            x,
            section_props,
        )
        loft_input.loftSections.add(section_profile)

    loft_input.loftSections.add(
        _create_fairing_profile(
            component,
            root_plane,
            "Transition Fairing End",
            ROOT_SPAN_STATION,
            feather_root_props,
        )
    )

    loft = loft_features.add(loft_input)
    loft.name = "Transition Fairing Loft"

    if loft.bodies.count != 1:
        raise RuntimeError("Expected a single transition fairing body.")

    fairing_body = loft.bodies.item(0)
    fairing_body.name = "Transition Fairing Body"
    _hollow_transition_fairing(component, fairing_body)
    return fairing_body


def _build_wing(design):
    if CENTER_WING_CHORD <= 0.0 or HALF_SPAN <= 0.0:
        raise ValueError("CENTER_WING_CHORD and HALF_SPAN must be greater than zero.")
    if ADDITIONAL_WINGS < 0:
        raise ValueError("ADDITIONAL_WINGS must be zero or greater.")
    if Y_SPACING_SCALE < 0.0:
        raise ValueError("Y_SPACING_SCALE must be zero or greater.")
    if ROOT_AIRFOIL_LENGTH <= 0.0:
        raise ValueError("ROOT_AIRFOIL_LENGTH must be greater than zero.")
    if ROOT_AIRFOIL_CHORD_SCALE <= 0.0:
        raise ValueError("ROOT_AIRFOIL_CHORD_SCALE must be greater than zero.")
    if MIN_TIP_CHORD_SCALE <= 0.0:
        raise ValueError("MIN_TIP_CHORD_SCALE must be greater than zero.")
    if TRAILING_EDGE_THICKNESS <= 0.0:
        raise ValueError("TRAILING_EDGE_THICKNESS must be greater than zero.")
    if MAX_TRAILING_EDGE_THICKNESS_TO_CHORD <= 0.0:
        raise ValueError("MAX_TRAILING_EDGE_THICKNESS_TO_CHORD must be greater than zero.")
    if MIN_FEATHER_ROOT_GAP < 0.0:
        raise ValueError("MIN_FEATHER_ROOT_GAP must be zero or greater.")
    if TRANSITION_FAIRING_FOOTPRINT_MARGIN < 0.0:
        raise ValueError("TRANSITION_FAIRING_FOOTPRINT_MARGIN must be zero or greater.")

    wing_component = design.rootComponent
    root_plane = wing_component.yZConstructionPlane
    loft_features = wing_component.features.loftFeatures

    shared_root_props = _shared_root_props_cm()
    total_wings = len(shared_root_props)
    wing_chords = [root['chord'] for root in shared_root_props]

    feather_root_props = []
    feather_root_profiles = []
    feather_bodies = []

    for i in range(1, total_wings + 1):
        root = shared_root_props[i - 1]
        chord = root['chord']
        y_offset = root['y']
        z_offset = root['z']
        root_gap = root['root_gap']
        incidence_deg = root['incidence_deg']

        feather_root_props.append({
            'chord': chord,
            'y': y_offset,
            'z': z_offset,
            'root_gap': root_gap,
            'incidence_deg': incidence_deg,
        })
        profile = _create_airfoil_profile(
            wing_component,
            root_plane,
            f"Feather Root {i} | chord {chord:.2f} | gap {root_gap:.2f}",
            chord,
            ROOT_SPAN_STATION,
            y_offset,
            z_offset,
            incidence_deg,
        )
        feather_root_profiles.append(profile)

    main_root_body = _build_single_root_airfoil(wing_component, root_plane, feather_root_props)
    transition_fairing = _build_transition_fairing(
        wing_component,
        root_plane,
        feather_root_props,
    )

    for i in range(1, total_wings + 1):
        root_gap = feather_root_props[i - 1]['root_gap']
        incidence_deg = feather_root_props[i - 1]['incidence_deg']

        terminal_section = _shared_section_cm(i, SWEEP_SECTION_FRACTIONS[-1])
        tip_sweep_offset = terminal_section['tip_sweep_offset']
        tip_z_curve_offset = terminal_section['tip_z_curve_offset']
        terminal_tip_chord = terminal_section['chord']

        loft_input = loft_features.createInput(adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
        loft_input.isSolid = True
        loft_input.loftSections.add(feather_root_profiles[i - 1])

        for span_fraction in SWEEP_SECTION_FRACTIONS[1:]:
            shared_section = _shared_section_cm(i, span_fraction)
            span_station = shared_section['span_station']
            current_chord = shared_section['chord']
            section_y_offset = shared_section['trailing_edge_y']
            section_z_offset = shared_section['z']
            incidence_deg = shared_section['incidence_deg']

            section_plane = _create_offset_plane(
                wing_component,
                root_plane,
                span_station,
                f"Section Plane {i}-{span_fraction}",
            )
            section_profile = _create_airfoil_profile(
                wing_component,
                section_plane,
                f"Section Airfoil {i}-{span_fraction}",
                current_chord,
                span_station,
                section_y_offset,
                section_z_offset,
                incidence_deg,
            )
            loft_input.loftSections.add(section_profile)

        loft = loft_features.add(loft_input)
        loft.name = (
            f"Feather {i} Loft | root gap {root_gap:.2f} | "
            f"tip chord {terminal_tip_chord:.2f} | sweep {tip_sweep_offset:.2f} | "
            f"tip z {tip_z_curve_offset:.2f}"
        )
        if loft.bodies.count != 1:
            raise RuntimeError(f"Expected 1 feather body for wing {i}, got {loft.bodies.count}.")
        feather_body = loft.bodies.item(0)
        feather_body.name = (
            f"Feather {i} Body | root gap {root_gap:.2f} | tip chord {terminal_tip_chord:.2f}"
        )
        feather_bodies.append(feather_body)

    if UNIFY_BODIES_FOR_EXPORT:
        unified_body = main_root_body
        if transition_fairing:
            unified_body = _join_body_into_target(
                wing_component,
                unified_body,
                transition_fairing,
                "Join Transition Fairing",
            )
        for feather_body in feather_bodies:
            unified_body = _join_body_into_target(
                wing_component,
                unified_body,
                feather_body,
                f"Join {feather_body.name}",
            )
        unified_body.name = "Feathered Wing Unified"
        expected_bodies = 1
    else:
        expected_bodies = 1 + total_wings + (1 if transition_fairing else 0)

    if wing_component.bRepBodies.count != expected_bodies:
        raise RuntimeError(
            f"Expected {expected_bodies} bodies after build, "
            f"got {wing_component.bRepBodies.count}."
        )


def run(context):
    app = adsk.core.Application.get()
    ui = app.userInterface

    try:
        app.documents.add(adsk.core.DocumentTypes.FusionDesignDocumentType)
        design = adsk.fusion.Design.cast(app.activeProduct)
        if not design:
            raise RuntimeError("Unable to create or access a Fusion design document.")

        _build_wing(design)
        _export_geometry_metadata()
        app.activeViewport.fit()
    except Exception:
        ui.messageBox(f"Failed:\n{traceback.format_exc()}")
