from dataclasses import replace
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase_tuned_feathering.acoustics import (  # noqa: E402
    evaluate_spp,
    evaluate_spp_reference,
    level2_deterministic_pressure,
)
from phase_tuned_feathering.closures import (  # noqa: E402
    ClosureParams,
    FlowConfig,
    cholesky_psd_check,
    is_hermitian,
    source_autospectra,
    source_cross_spectral_matrix,
    transfer_weights,
)
from phase_tuned_feathering.geometry import default_geometry, source_grid  # noqa: E402
from phase_tuned_feathering.observers import ObserverGrid  # noqa: E402


class AcousticTests(unittest.TestCase):
    def _small_grid(self):
        params = replace(default_geometry(), additional_wings=1)
        return source_grid(params, n_eta=3)

    def test_cross_spectral_matrix_is_hermitian_psd(self):
        grid = self._small_grid()
        flow = FlowConfig()
        closures = ClosureParams(coherence_model="exponential")
        matrix = source_cross_spectral_matrix(grid, 750.0, flow, closures)
        self.assertTrue(is_hermitian(matrix))
        self.assertTrue(cholesky_psd_check(matrix, tolerance=1.0e-8))

    def test_spp_nonnegative_and_reference_matches(self):
        grid = self._small_grid()
        observers = ObserverGrid.six_axis()
        frequencies = (300.0, 900.0)
        result = evaluate_spp(grid, observers, frequencies)
        reference = evaluate_spp_reference(grid, observers, frequencies)
        for row, reference_row in zip(result.spp, reference.spp):
            for value, reference_value in zip(row, reference_row):
                self.assertGreaterEqual(value, 0.0)
                self.assertAlmostEqual(value, reference_value, delta=max(1.0e-24, abs(value) * 1.0e-12))

    def test_zero_coherence_removes_cross_terms(self):
        grid = self._small_grid()
        flow = FlowConfig()
        closures = ClosureParams(coherence_model="zero")
        matrix = source_cross_spectral_matrix(grid, 500.0, flow, closures)
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                if i == j:
                    self.assertGreater(value.real, 0.0)
                else:
                    self.assertEqual(value, 0.0j)

    def test_full_coherence_recovers_deterministic_limit(self):
        grid = self._small_grid()
        observer = ObserverGrid.from_directions(((0.0, 0.0, 1.0),))
        flow = FlowConfig()
        closures = ClosureParams(coherence_model="full")
        frequency = 500.0
        result = evaluate_spp(grid, observer, (frequency,), flow, closures)
        pressure = level2_deterministic_pressure(grid, observer, frequency, flow, closures)[0]
        expected = abs(pressure) ** 2
        self.assertAlmostEqual(result.spp[0][0], expected, delta=max(1.0e-24, expected * 1.0e-12))

    def test_low_frequency_limit_is_smooth(self):
        grid = self._small_grid()
        observer = ObserverGrid.from_directions(((0.0, 0.0, 1.0),))
        low = evaluate_spp(grid, observer, (100.0,))
        lower = evaluate_spp(grid, observer, (90.0,))
        self.assertGreaterEqual(low.spp[0][0], 0.0)
        self.assertGreaterEqual(lower.spp[0][0], 0.0)
        self.assertLess(abs(low.spp[0][0] - lower.spp[0][0]), max(low.spp[0][0], lower.spp[0][0], 1.0e-30))

    def test_full_coherence_manual_quadratic_form(self):
        grid = self._small_grid()
        flow = FlowConfig()
        closures = ClosureParams(coherence_model="full")
        direction = (0.0, 0.0, 1.0)
        frequency = 500.0
        autospectra = source_autospectra(grid, frequency, flow, closures)
        weights = transfer_weights(grid, direction, frequency, flow)
        coherent_sum = sum(weights[index] * autospectra[index] ** 0.5 for index in range(grid.n))
        result = evaluate_spp(grid, ObserverGrid.from_directions((direction,)), (frequency,), flow, closures)
        radius_factor = 1.0 / ((4.0 * 3.141592653589793 * flow.observer_radius) ** 2)
        self.assertAlmostEqual(result.spp[0][0], radius_factor * abs(coherent_sum) ** 2)


if __name__ == "__main__":
    unittest.main()
