"""MC integration errors are checked independently of neural-network training."""

import unittest

import numpy as np
from scipy.optimize import brentq

from utils_extended_pairs_population import (
    independent_root_difference, population_root_mc, score_mc_moments,
)


def fit(qs, qb, mu, ls, lb):
    def score(nu):
        return -ls + mu * ls * np.mean(qs / (1 + nu * qs)) + lb * np.mean(qb / (1 + nu * qb))
    return 0. if score(0) <= 0 else brentq(score, 0, 20)


class PopulationMCTest(unittest.TestCase):
    def test_analytic_score_variance_and_curvature(self):
        # Two equiprobable support points: their moments can be evaluated directly.
        qs, qb = np.array([.1, .5]), np.array([.1, .3])
        mu, ls, lb = 2., 2., 8.
        nu = fit(qs, qb, mu, ls, lb)
        result = population_root_mc(nu, mu, qs, qb, ls, lb)
        ws, wb = qs / (1 + nu * qs), qb / (1 + nu * qb)
        variance = (mu*ls)**2 * (ws[1]-ws[0])**2/4 + lb**2 * (wb[1]-wb[0])**2/4
        information = mu*ls * (ws[0]**2 + ws[1]**2)/2 + lb * (wb[0]**2 + wb[1]**2)/2
        self.assertEqual(result['status'], 'interior')
        self.assertAlmostEqual(result['score'], 0., places=11)
        self.assertAlmostEqual(result['score_se']**2, variance)
        self.assertAlmostEqual(result['information'], information)
        self.assertAlmostEqual(result['root_se'], np.sqrt(variance)/information)
        self.assertAlmostEqual(result['signal_variance_fraction'] + result['background_variance_fraction'], 1.)

    def test_inverse_square_root_bank_size_scaling(self):
        qs = np.tile([.1, .5], 100)
        qb = np.tile([.1, .3], 100)
        first = score_mc_moments(1., 2., qs, qb, 2., 8.)
        larger = score_mc_moments(1., 2., np.tile(qs, 4), np.tile(qb, 4), 2., 8.)
        # Exact finite-sample correction from ddof=1; tends to 1/sqrt(4).
        self.assertAlmostEqual(larger['score_se']/first['score_se'], np.sqrt(199/799))
        self.assertAlmostEqual(larger['information'], first['information'])

    def test_repeated_independent_banks_match_root_variance(self):
        rng = np.random.default_rng(1829)
        roots, variances = [], []
        for _ in range(800):
            qs = rng.choice([.1, .5], size=400)
            qb = rng.choice([.1, .3], size=600)
            root = fit(qs, qb, 2., 2., 8.)
            result = population_root_mc(root, 2., qs, qb, 2., 8.)
            self.assertEqual(result['status'], 'interior')
            roots.append(root)
            variances.append(result['root_se']**2)
        ratio = np.var(roots, ddof=1) / np.mean(variances)
        self.assertLess(abs(ratio-1), .15)

    def test_boundary_score_and_independent_difference(self):
        qs, qb = np.array([.1, .5]), np.array([.1, .3])
        root = fit(qs, qb, 0., 2., 8.)
        boundary = population_root_mc(root, 0., qs, qb, 2., 8.)
        self.assertEqual(boundary['status'], 'boundary')
        self.assertLess(boundary['score_at_zero'], 0)
        self.assertAlmostEqual(boundary['signal_score_variance'], 0.)
        self.assertAlmostEqual(boundary['score_at_zero_se'], .8)
        self.assertTrue(np.isnan(boundary['root_se']))
        self.assertGreater(boundary['local_root_scale'], 0.)
        interior = population_root_mc(fit(qs, qb, 2., 2., 8.), 2., qs, qb, 2., 8.)
        self.assertTrue(np.isnan(independent_root_difference(interior, boundary)['difference_over_mc_se']))
        # Use explicitly supplied independent-bank estimates for the difference rule.
        pair = independent_root_difference(dict(root=1.2, root_se=.1, status='interior'),
                                           dict(root=1., root_se=.2, status='interior'))
        self.assertAlmostEqual(pair['difference_se'], np.sqrt(.05))
        self.assertAlmostEqual(pair['difference_over_mc_se'], .2/np.sqrt(.05))


if __name__ == '__main__':
    unittest.main()
