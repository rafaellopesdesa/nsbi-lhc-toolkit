"""Mathematical examples for the scan geometry, independent of head training."""

import unittest

import numpy as np

from utils_extended_pairs_diagnostics import threshold_intervals


class ThresholdIntervalsTest(unittest.TestCase):
    def test_nonquadratic_double_well_has_two_components(self):
        # (x^2 - 1)^2 <= 1/4 has four known crossings and two islands.
        x = np.linspace(-2, 2, 4001)
        result = threshold_intervals(x, (x**2 - 1)**2, 0.25)
        self.assertEqual(len(result), 2)
        actual = [(v['lower'], v['upper']) for v in result]
        expected = [(-np.sqrt(1.5), -np.sqrt(0.5)),
                    (np.sqrt(0.5), np.sqrt(1.5))]
        np.testing.assert_allclose(actual, expected, atol=2e-6)
        self.assertFalse(any(v['lower_at_edge'] or v['upper_at_edge'] for v in result))

    def test_boundary_and_upper_truncation(self):
        result = threshold_intervals([0, 1, 2, 3], [0, 2, 2, 0], 1)
        self.assertEqual(result, [
            dict(lower=0., upper=.5, lower_at_edge=True, upper_at_edge=False),
            dict(lower=2.5, upper=3., lower_at_edge=False, upper_at_edge=True)])

    def test_empty_and_full_sets(self):
        self.assertEqual(threshold_intervals([0, 1, 2], [2, 3, 2], 1), [])
        self.assertEqual(threshold_intervals([0, 1, 2], [0, 1, 0], 1), [
            dict(lower=0., upper=2., lower_at_edge=True, upper_at_edge=True)])

    def test_exact_crossing_and_tangent_equality(self):
        result = threshold_intervals([0, 1, 2], [2, 1, 2], 1)
        self.assertEqual(result, [dict(lower=1., upper=1.,
                                     lower_at_edge=False, upper_at_edge=False)])
        result = threshold_intervals([0, 1, 2, 3], [2, 1, 1, 2], 1)
        self.assertEqual(result, [dict(lower=1., upper=2.,
                                     lower_at_edge=False, upper_at_edge=False)])

    def test_uneven_grid_known_linear_crossings(self):
        result = threshold_intervals([0, 2, 5], [3, -1, 5], 0)
        np.testing.assert_allclose([result[0]['lower'], result[0]['upper']], [1.5, 2.5])


if __name__ == '__main__':
    unittest.main()
