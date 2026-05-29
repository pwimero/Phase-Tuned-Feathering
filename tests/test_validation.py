from argparse import Namespace
from contextlib import redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase_tuned_feathering.geometry import default_geometry  # noqa: E402
from phase_tuned_feathering.observers import ObserverGrid  # noqa: E402
from phase_tuned_feathering.pipeline import run_pipeline  # noqa: E402
from phase_tuned_feathering.validation import (  # noqa: E402
    compare_theory_to_simulation,
    generate_synthetic_simulation_dataset,
    load_simulation_csv,
    write_comparison_csv,
    write_simulation_csv,
    write_summary_json,
)


class ValidationTests(unittest.TestCase):
    def test_synthetic_simulation_can_be_compared(self):
        params = replace(default_geometry(), additional_wings=1)
        simulation = generate_synthetic_simulation_dataset(
            params=params,
            observers=ObserverGrid.six_axis(),
            frequencies_hz=(500.0, 1000.0),
            n_eta=3,
        )
        result = compare_theory_to_simulation(simulation, params=params, n_eta=3)
        self.assertEqual(len(result.rows), len(simulation.records))
        self.assertIn("all", result.summary)
        self.assertIn("validation", result.summary)
        self.assertGreater(result.summary["all"]["count"], 0.0)
        self.assertGreaterEqual(result.summary["all"]["rmse_db"], 0.0)

    def test_simulation_csv_round_trip_and_exports(self):
        params = replace(default_geometry(), additional_wings=1)
        simulation = generate_synthetic_simulation_dataset(
            params=params,
            observers=ObserverGrid.six_axis(),
            frequencies_hz=(500.0,),
            n_eta=3,
        )
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            simulation_path = write_simulation_csv(directory_path / "sim.csv", simulation)
            loaded = load_simulation_csv(simulation_path)
            result = compare_theory_to_simulation(loaded, params=params, n_eta=3)
            comparison_path = write_comparison_csv(directory_path / "comparison.csv", result)
            summary_path = write_summary_json(directory_path / "summary.json", result)
            self.assertTrue(comparison_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertIn("error_db", comparison_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("rmse_db", summary_path.read_text(encoding="utf-8"))

    def test_pipeline_entrypoint_writes_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            stream = io.StringIO()
            with redirect_stdout(stream):
                paths = run_pipeline(
                    Namespace(
                        simulation_csv=None,
                        output_dir=Path(directory),
                        n_eta=3,
                        source_chord_fraction=1.0,
                        synthetic_seed=3,
                    )
                )
            for path in paths.values():
                self.assertTrue(Path(path).exists())


if __name__ == "__main__":
    unittest.main()
