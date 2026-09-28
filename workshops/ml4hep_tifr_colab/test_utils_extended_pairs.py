"""Small scientific checks for the unbinned teachers and learned interfaces.

Run from this directory with: python -m unittest test_utils_extended_pairs
"""
import unittest

import numpy as np
import torch

import utils_extended_pairs as ep


class ExtendedPairsTests(unittest.TestCase):
    def test_counting_model_has_analytic_mle_and_no_upper_cap(self):
        for n, q, lam_s in [(80, .1, 10), (130, .1, 10), (80, .5, 2)]:
            events = np.full(n, q)
            expected = max(0., n/lam_s - 1/q)
            self.assertAlmostEqual(ep.exact_fit(events, lam_s), expected, places=9)
            queries = np.array([0., 1., 3.])
            loglik = lambda nu: -nu*lam_s + n*np.log1p(nu*q)
            np.testing.assert_allclose(ep.exact_statistic(events, queries, lam_s),
                2*(loglik(expected)-loglik(queries)), atol=1e-10)
        self.assertEqual(ep.exact_fit([], 2), 0)

    def test_normalized_reference_population_closes_to_identity(self):
        # Two event types with exactly normalized process masses.
        ps, pb = np.array([.8, .2]), np.array([.3, .7])
        lam_s, lam_b = 4., 20.
        q = lam_s/lam_b * ps/pb
        for mu in [0., .2, 1., 3.]:
            result = ep.exact_response(q, q, mu, lam_s, lam_b,
                signal_weights=ps, background_weights=pb)
            self.assertAlmostEqual(result, mu, places=9)

    def test_pooling_is_permutation_invariant_and_ignores_padding(self):
        torch.manual_seed(1)
        model = ep.PairPosterior(ep.PairEncoder(2, embed_dim=4), n_classes=3)
        x = torch.randn(5, 2, 2)
        n = torch.full((5,), 2)
        domain = torch.zeros(5, dtype=torch.long)
        torch.testing.assert_close(model(x, n, domain), model(x.flip(1), n, domain))
        n.fill_(1)
        altered = x.clone()
        altered[:, 1] += 100
        torch.testing.assert_close(model(x, n, domain), model(altered, n, domain))
        self.assertEqual(ep.encode_events(model.encoder, np.empty((0, 2))).shape, (0, 4))

    def test_metadata_never_enters_heads_and_scaler_uses_training_only(self):
        torch.manual_seed(3)
        rng = np.random.default_rng(3)
        z = rng.normal(size=(12, 2))
        fitted = np.abs(z[:, 0])
        queries = rng.uniform(0, 3, (12, 3))
        train = dict(z=z, nu_hat=fitted, nu=queries,
            t=(queries-fitted[:, None])**2, generation_seconds=np.asarray(2.),
            domain=np.ones(12), mu=np.full(12, 99.))
        val = dict(train, z=z+10)
        model = ep.FastHeads(2, width=8)
        ep.train_heads(model, train, val, epochs=0)
        np.testing.assert_allclose(model.z_mean.numpy(), z.mean(0), atol=1e-7)
        estimate, statistic = ep.predict_heads(model, z, queries)
        self.assertEqual(estimate.shape, (12,))
        self.assertEqual(statistic.shape, (12, 3))
        self.assertTrue(np.all(statistic >= 0))

    def test_reference_posterior_matches_normalized_discrete_bank(self):
        # Reference events are sampled with separately normalized S/B weights.
        # Their normalization ratio must enter the posterior benchmark.
        signal = np.array([2., 1., 4.])
        background = np.array([3., 2., 1.])
        ps, pb = signal/signal.sum(), background/background.sum()
        fractions = np.array([0.1, 0.5, 0.9])
        indices = np.array([[0, 2], [1, 0], [2, 1]])
        n = np.array([2, 1, 2])
        lam_s, lam_b = 2., 17.
        q = lam_s/lam_b * signal[indices]/background[indices]
        scale = lam_b/lam_s * background.sum()/signal.sum()
        posterior = ep.reference_fraction_posterior(q, n, fractions, scale)
        expected = []
        for row, size in zip(indices, n):
            probability = np.prod([
                (1-fractions)*pb[event] + fractions*ps[event]
                for event in row[:size]
            ], axis=0)
            expected.append(probability/probability.sum())
        np.testing.assert_allclose(posterior, expected, atol=1e-14)
        q[1, 1] = 1e8  # A masked second event cannot affect a singleton.
        np.testing.assert_allclose(
            ep.reference_fraction_posterior(q, n, fractions, scale), expected,
            atol=1e-14)

    def test_signed_root_loss_preserves_sign_and_boundary_is_reachable(self):
        model = ep.RefinedHeads(1, width=4, mode='signed_root')
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.statistic[-1].bias.fill_(-2.)
        # The exact root is -2 on one side of the minimum and +3 on the other.
        # A constant -2 prediction therefore has root-MSE (0 + 25)/2 = 12.5.
        table = dict(z=np.array([[0.], [1.]]), nu_hat=np.ones(2),
            nu=np.array([[0., 2.], [0., 2.]]),
            t=np.array([[4., 9.], [4., 9.]]))
        history = ep.train_refined_heads(model, table, table, epochs=0)
        self.assertAlmostEqual(history[0]['val_statistic'], 12.5)
        estimate, statistic = ep.predict_heads(model, table['z'], table['nu'])
        np.testing.assert_array_equal(estimate, 0.)
        np.testing.assert_array_equal(statistic, 4.)
        self.assertAlmostEqual(history[0]['val'],
            history[0]['val_estimator'] + history[0]['val_statistic'])

    def test_refined_training_uses_only_training_scaler_and_keeps_old_head_api(self):
        torch.manual_seed(8)
        rng = np.random.default_rng(8)
        z = rng.normal(size=(12, 2))
        fitted = np.maximum(z[:, 0], 0.)
        queries = rng.uniform(0, 3, (12, 3))
        delta = queries-fitted[:, None]
        # This target is explicitly nonquadratic and asymmetric.
        train = dict(z=z, nu_hat=fitted, nu=queries,
            t=delta**2 * (1 + 0.3*delta + 0.2*delta**2),
            domain=np.ones(12), mu=np.full(12, 99.))
        val = dict(train, z=z+10)
        for mode in ['signed_root', 'direct']:
            model = ep.RefinedHeads(2, width=8, mode=mode)
            history = ep.train_refined_heads(model, train, val, epochs=1,
                                            batch_size=6)
            np.testing.assert_allclose(model.z_mean.numpy(), z.mean(0), atol=1e-7)
            self.assertTrue(np.isfinite(history[-1]['val_statistic']))
            estimate, statistic = ep.predict_heads(model, z, queries)
            self.assertEqual(estimate.shape, (12,))
            self.assertEqual(statistic.shape, (12, 3))
            self.assertTrue(np.all(statistic >= 0))
        old = ep.FastHeads(2, width=8)
        restored = ep.FastHeads(2, width=8)
        restored.load_state_dict(old.state_dict())
        a = ep.predict_heads(old, z, queries)
        b = ep.predict_heads(restored, z, queries)
        for left, right in zip(a, b):
            np.testing.assert_array_equal(left, right)


if __name__ == '__main__':
    unittest.main()
