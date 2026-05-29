from dataclasses import replace
from pathlib import Path
import tempfile
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase_tuned_feathering import (  # noqa: E402
    ClosureParams,
    FlowConfig,
    ObserverGrid,
    OptimizationConfig,
    Sector,
    band_spl,
    default_geometry,
    directivity,
    evaluate_spp,
    optimize_stage,
    screen_aero,
    sector_spl,
    source_grid,
    total_acoustic_proxy,
    write_geometry_metadata_json,
    write_source_grid_csv,
)


class WorkflowTests(unittest.TestCase):
    def test_metrics_are_finite(self):
        params = replace(default_geometry(), additional_wings=1)
        grid = source_grid(params, n_eta=3)
        observers = ObserverGrid.six_axis()
        flow = FlowConfig()
        result = evaluate_spp(grid, observers, (400.0, 800.0), flow, ClosureParams())
        spl = band_spl(result, flow)
        self.assertEqual(len(spl), len(observers.directions))
        self.assertTrue(all(value == value for value in spl))

        directivity_result = directivity(result)
        for row in directivity_result.spp:
            mean = sum(value * weight for value, weight in zip(row, observers.weights))
            mean /= sum(observers.weights)
            self.assertAlmostEqual(mean, 1.0)

        top_sector = Sector((0.0, 0.0, 1.0), 80.0)
        self.assertTrue(sector_spl(result, top_sector, flow) == sector_spl(result, top_sector, flow))
        self.assertGreaterEqual(total_acoustic_proxy(result), 0.0)

    def test_aero_screen_default_is_feasible(self):
        result = screen_aero(default_geometry(), FlowConfig())
        self.assertTrue(result.feasible, result.issues)
        self.assertIn("cl_cd_proxy", result.values)

    def test_optimization_stage_runs(self):
        params = replace(default_geometry(), additional_wings=1)
        config = OptimizationConfig(
            base_params=params,
            observers=ObserverGrid.six_axis(),
            frequencies_hz=(500.0,),
            iterations=3,
            n_eta=3,
            target_sector=Sector((0.0, 0.0, 1.0), 95.0),
            suppressed_sector=Sector((0.0, 0.0, -1.0), 95.0),
        )
        result = optimize_stage(1, config)
        self.assertEqual(result.stage, 1)
        self.assertEqual(len(result.history), 3)
        self.assertEqual(len(result.best_params.incidence_angles_deg()), params.total_wings)

    def test_metadata_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            json_path = write_geometry_metadata_json(output_dir / "geometry.json", default_geometry())
            csv_path = write_source_grid_csv(
                output_dir / "source.csv",
                replace(default_geometry(), additional_wings=1),
                n_eta=3,
            )
            self.assertTrue(json_path.exists())
            self.assertTrue(csv_path.exists())
            self.assertIn("coordinate_mapping", json_path.read_text(encoding="utf-8"))
            self.assertIn("paper_x_m", csv_path.read_text(encoding="utf-8").splitlines()[0])


if __name__ == "__main__":
    unittest.main()
