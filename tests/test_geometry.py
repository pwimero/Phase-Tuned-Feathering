import math
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase_tuned_feathering.geometry import (  # noqa: E402
    default_geometry,
    feather_root_properties,
    feather_section,
    fusion_parameter_values,
    meters_to_cm,
    source_grid,
    tip_span_station,
    tip_sweep_offset,
    tip_z_curve_offset,
    wing_chord,
)


class GeometryTests(unittest.TestCase):
    def test_default_geometry_matches_current_fusion_constants(self):
        params = default_geometry()
        self.assertEqual(params.total_wings, 7)
        expected_chords = (0.24, 0.26, 0.28, 0.30, 0.26, 0.22, 0.18)
        for index, expected in enumerate(expected_chords, start=1):
            self.assertAlmostEqual(wing_chord(params, index), expected)

        roots = feather_root_properties(params)
        self.assertAlmostEqual(roots[0].trailing_edge_y, 0.0)
        self.assertAlmostEqual(roots[1].root_gap, 0.03)
        self.assertAlmostEqual(roots[-1].root_gap, 0.03)
        self.assertAlmostEqual(roots[0].z, 0.12)
        self.assertAlmostEqual(roots[-1].z, -0.12)

        self.assertAlmostEqual(tip_span_station(params, 1), 1.60)
        self.assertAlmostEqual(tip_span_station(params, 4), 3.20)
        self.assertAlmostEqual(tip_span_station(params, 7), 1.60)
        self.assertAlmostEqual(tip_sweep_offset(params, 1), -0.55)
        self.assertAlmostEqual(tip_sweep_offset(params, 4), 0.0)
        self.assertAlmostEqual(tip_sweep_offset(params, 7), 0.20)
        self.assertAlmostEqual(tip_z_curve_offset(params, 1), 0.50)
        self.assertAlmostEqual(tip_z_curve_offset(params, 7), -0.20)

    def test_fusion_unit_conversion(self):
        params = default_geometry()
        values = fusion_parameter_values(params)
        self.assertAlmostEqual(values["CENTER_WING_CHORD"], 30.0)
        self.assertAlmostEqual(values["HALF_SPAN"], 160.0)
        self.assertAlmostEqual(values["TRAILING_EDGE_THICKNESS"], 0.5)
        self.assertAlmostEqual(meters_to_cm(params.root_feather_gap), 40.0)

    def test_source_grid_mapping_and_incidence_pivot(self):
        params = default_geometry()
        grid = source_grid(params, n_eta=5, source_chord_fraction=1.0)
        self.assertEqual(grid.n, 35)
        first_section = feather_section(params, 1, 0.0)
        fusion_x, fusion_y, fusion_z = first_section.fusion_midline_point(1.0)
        self.assertAlmostEqual(grid.points[0][0], fusion_y)
        self.assertAlmostEqual(grid.points[0][1], fusion_x)
        self.assertAlmostEqual(grid.points[0][2], fusion_z)

        pivot_grid = source_grid(params, n_eta=2, source_chord_fraction=0.25)
        root = feather_root_properties(params)[0]
        self.assertAlmostEqual(pivot_grid.points[0][0], root.trailing_edge_y - 0.75 * root.chord)
        self.assertAlmostEqual(pivot_grid.points[0][2], root.z)
        self.assertAlmostEqual(params.fusion_incidence_pivot_fraction_from_le, 0.25)

    def test_section_tip_shape(self):
        params = default_geometry()
        section = feather_section(params, 1, 0.99)
        progress = (0.99 - 0.8) / 0.2
        expected_scale = max(math.sqrt(1.0 - progress**2), params.min_tip_chord_scale)
        self.assertAlmostEqual(section.chord, params.wing_1_chord * expected_scale)
        self.assertLess(section.trailing_edge_y, 0.0)
        self.assertGreater(section.z, params.wing_1_root_z_translation)


if __name__ == "__main__":
    unittest.main()
