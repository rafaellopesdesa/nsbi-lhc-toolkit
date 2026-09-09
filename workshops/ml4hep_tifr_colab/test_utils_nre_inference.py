"""Fast, training-free regression tests for the NRE Asimov example."""

import tempfile
import unittest

import numpy as np
from scipy.special import ndtr

from utils_nre_inference import (
    binned_asimov, bounded_muhat_bin_probabilities, build_compression,
    finite_reference_asimov, fit_weighted_q, log_likelihood_relative,
    q0_bin_probabilities, raw_simulator_asimov, ratio_to_q,
    run_toys, run_toys_cached, simulator_bin_probabilities,
    validate_compression,
    simulator_score_diagnostic, summarize_score_diagnostics,
)


class TestNREInference(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2026)
        self.ratios = np.exp(rng.normal(size=(257, 2)))
        self.yields = np.array([12.0, 30.0])

    def test_simulator_score_and_mc_error_match_direct_calculation(self):
        signal, background = self.ratios[:100], self.ratios[100:]
        normalizers = [0.8, 1.3]
        for mu in (0.0, 1.0, 2.0):
            result = simulator_score_diagnostic(signal, background, self.yields, mu, normalizers)
            gs, gb = [ratio_to_q(bank, self.yields, normalizers) for bank in (signal, background)]
            gs, gb = gs / (1 + mu * gs), gb / (1 + mu * gb)
            score = -12 + mu * 12 * gs.mean() + 30 * gb.mean()
            variance = (mu * 12)**2 * gs.var(ddof=1) / len(gs) + 30**2 * gb.var(ddof=1) / len(gb)
            information = mu * 12 * np.mean(gs**2) + 30 * np.mean(gb**2)
            self.assertAlmostEqual(result["score_at_truth"], score)
            self.assertAlmostEqual(result["score_mc_se"]**2, variance)
            self.assertAlmostEqual(result["information_at_truth"], information)
            self.assertAlmostEqual(result["linearized_shift_mc_se"], np.sqrt(variance) / information)
            asimov = raw_simulator_asimov(signal, background, self.yields, mu, normalizers=normalizers)
            self.assertAlmostEqual(result["score_at_truth"], asimov["score_at_truth"])
        with self.assertRaises(ValueError):
            simulator_score_diagnostic(signal[:1], background, self.yields)

    def test_mc_errors_match_independent_bank_scatter(self):
        rng = np.random.default_rng(89)
        records = [simulator_score_diagnostic(np.exp(rng.normal(size=(64, 2))),
                                             np.exp(rng.normal(size=(128, 2))), self.yields)
                   for _ in range(400)]
        predicted_variance = np.mean([row["score_mc_se"]**2 for row in records])
        empirical_variance = np.var([row["score_at_truth"] for row in records], ddof=1)
        self.assertLess(abs(predicted_variance / empirical_variance - 1), 0.15)
        summary = summarize_score_diagnostics(records)
        self.assertAlmostEqual(summary["score_mc_se"]**2, predicted_variance / 400)
        self.assertAlmostEqual(summary["score_between_bank_se"]**2, empirical_variance / 400)
        with self.assertRaises(ValueError):
            summarize_score_diagnostics(records[:1])
        with self.assertRaises(ValueError):
            summarize_score_diagnostics([records[0], {**records[1], "normalizers": [2, 1]}])

    def test_score_diagnostic_does_not_recenter_misspecification(self):
        ratios = np.tile([2.0, 1.0], (20, 1))
        result = simulator_score_diagnostic(ratios, ratios, [10, 50])
        self.assertGreater(result["score_at_truth"], 1)
        self.assertLess(result["score_mc_se"], 1e-12)

    def test_finite_reference_global_maximum(self):
        for m in (1, 2, 11, len(self.ratios)):
            for mu in (0.0, 0.2, 1.0, 3.2):
                scan = np.unique(np.r_[np.linspace(0, 6, 81), mu])
                result = finite_reference_asimov(self.ratios[:m], self.yields, mu, scan)
                self.assertAlmostEqual(result["mu_hat"], mu, places=11)
                self.assertAlmostEqual(result["score_at_truth"], 0.0, places=10)
                self.assertAlmostEqual(result["t_at_truth"], 0.0, places=10)
                self.assertTrue(np.all(result["t_scan"] >= 0))
                self.assertAlmostEqual(result["total_weight"], mu * 12 + 30, places=10)
                self.assertEqual(result["n_integration_events"], m)

    def test_direct_global_likelihood_inequality(self):
        z = self.ratios.mean(axis=0)
        normalized = self.ratios / z
        intensity_truth = (normalized @ self.yields) / len(self.ratios)
        q = ratio_to_q(self.ratios, self.yields, z)
        ell_truth = log_likelihood_relative(q, intensity_truth, 1.0, self.yields[0])
        for mu in (0.0, 0.1, 0.8, 1.1, 3.0, 20.0):
            intensity = (mu * self.yields[0] * normalized[:, 0] + self.yields[1] * normalized[:, 1]) / len(self.ratios)
            kl = np.sum(intensity - intensity_truth - intensity_truth * np.log(intensity / intensity_truth))
            delta = log_likelihood_relative(q, intensity_truth, mu, self.yields[0]) - ell_truth
            self.assertAlmostEqual(delta, -kl, places=10)
            self.assertLessEqual(delta, 1e-10)

    def test_raw_simulator_does_not_force_truth(self):
        # A deliberately misspecified constant ratio produces a pseudo-true 3.5,
        # independent of integration size. No normalization is fitted to S/B data.
        ratios = np.tile([2.0, 1.0], (10, 1))
        small = raw_simulator_asimov(ratios, ratios, [10.0, 50.0], mu_true=1.0)
        large = raw_simulator_asimov(np.repeat(ratios, 5, axis=0), ratios, [10.0, 50.0], mu_true=1.0)
        self.assertAlmostEqual(small["mu_hat"], 3.5, places=10)
        self.assertAlmostEqual(large["mu_hat"], small["mu_hat"], places=10)
        self.assertGreater(small["t_at_truth"], 0)
        self.assertEqual(small["normalizers"], [1.0, 1.0])

    def test_empty_zero_weight_and_zero_signal_events(self):
        for q, weights in (([], None), ([1.0, 2.0], [0.0, 0.0]), ([0.0, 0.0], None)):
            result = fit_weighted_q(q, weights, signal_yield=5)
            self.assertEqual(result["mu_hat"], 0)
            self.assertEqual(result["q0"], 0)
            self.assertTrue(np.isinf(result["sigma_curvature"]))

    def test_vectorized_fit_matches_individual(self):
        rng = np.random.default_rng(8)
        q = np.exp(rng.normal(size=31))
        weights = rng.poisson(3, size=(9, 31))
        weights[0] = 0
        batch = fit_weighted_q(q, weights, 10)
        individual = [fit_weighted_q(q, w, 10) for w in weights]
        for key in batch:
            np.testing.assert_allclose(batch[key], [r[key] for r in individual], rtol=1e-12)

    def test_no_arbitrary_fit_upper_bound_or_ratio_clipping(self):
        result = fit_weighted_q([1.0], [1e8], signal_yield=1.0)
        self.assertAlmostEqual(result["mu_hat"], 1e8 - 1, places=6)
        self.assertEqual(ratio_to_q([[1e12, 1.0]], [1.0, 1.0])[0], 1e12)
        with self.assertRaises(ValueError):
            ratio_to_q([[1.0, 0.0]], [1.0, 1.0])
        with self.assertRaises(ValueError):
            finite_reference_asimov([[np.nan, 1.0]], [1.0, 1.0])

    def test_compression_normalization_and_closure(self):
        for bins in (1, 2, 17, 500):
            compressed = build_compression(self.ratios, self.yields, bins)
            self.assertLessEqual(compressed["n_bins"], min(bins, len(self.ratios)))
            self.assertAlmostEqual(np.sum(compressed["p_signal"]), 1.0)
            self.assertAlmostEqual(np.sum(compressed["p_background"]), 1.0)
            self.assertTrue(np.all(compressed["p_background"] > 0))
            expected = binned_asimov(compressed)
            self.assertAlmostEqual(expected["mu_hat"], 1.0, places=11)
            self.assertAlmostEqual(expected["score_at_truth"], 0.0, places=11)
        with self.assertRaises(ValueError):
            build_compression(self.ratios, self.yields, normalizers=[1.0, 1.0])

    def test_compression_ties_and_extreme_bins(self):
        repeated = np.tile([[0.0, 1.0], [1.0, 1.0], [1e20, 1.0]], (50, 1))
        compressed = build_compression(repeated, [1.0, 1.0], n_bins=100)
        self.assertLessEqual(compressed["n_bins"], 3)
        self.assertTrue(np.all(compressed["p_background"] > 0))
        probabilities = simulator_bin_probabilities([[0.0, 1.0], [1e30, 1.0]], [[1.0, 1.0]], compressed)
        self.assertAlmostEqual(np.sum(probabilities["p_signal"]), 1.0)
        self.assertAlmostEqual(np.sum(probabilities["p_background"]), 1.0)

    def test_log_bins_do_not_saturate_extreme_positive_q(self):
        # Every positive q exceeds the resolution of q/(1+q), except the final
        # event; direct logarithmic bins must still distinguish these tails.
        ratios = np.array([[1.0, 1e-100], [1.0, 1e-200], [1.0, 1e-300], [1.0, 1.0]])
        compressed = build_compression(ratios, [1.0, 1.0], n_bins=4)
        self.assertEqual(compressed["n_bins"], 4)
        self.assertTrue(np.all(np.diff(compressed["bin_edges_log_q"]) > 0))
        self.assertTrue(np.all(np.isfinite(compressed["q"])))
        result = fit_weighted_q([1e300], [0.0], signal_yield=1.0)
        self.assertEqual(result["mu_hat"], 0.0)
        self.assertTrue(np.isinf(result["sigma_curvature"]))

    def test_hybrid_log_bins_resolve_equal_reference_discovery_tails(self):
        rng = np.random.default_rng(81723)
        u = rng.beta(0.5, 0.5, size=262144)
        # Exact equal-REF identity rS+rB=2; this is not an unphysical case of
        # independent unbounded component-ratio fluctuations.
        ratios = 2.0 * np.column_stack((u, 1.0 - u))
        yields = [611.6454, 152779.58]
        scan = np.linspace(0, 2, 41)
        expected = finite_reference_asimov(ratios, yields, 1.0, scan)
        compressed = build_compression(ratios, yields, n_bins=2048)
        binned = binned_asimov(compressed, 1.0, scan)
        paired = validate_compression(compressed, ratios, n_toys=8, seed=890)
        profile_error = np.max(np.abs(expected["t_scan"] - binned["t_scan"])) / max(1, np.max(expected["t_scan"]))
        mu_error = paired["rms_delta_mu_hat"] / expected["sigma_curvature"]
        q_error = paired["rms_delta_q0"] / max(1, np.sqrt(expected["q0_asimov"]))
        self.assertLessEqual(compressed["n_bins"], 2048)
        self.assertTrue(np.all(compressed["p_background"] > 0))
        self.assertLess(profile_error, 0.005)
        self.assertLess(mu_error, 0.02)
        self.assertLess(q_error, 0.02)

    def test_paired_compression_exact_for_constant_ratio(self):
        ratios = np.tile([2.0, 3.0], (100, 1))
        compressed = build_compression(ratios, [5.0, 10.0], 32)
        check = validate_compression(compressed, ratios, n_toys=12)
        np.testing.assert_allclose(check["mu_hat_event"], check["mu_hat_binned"], atol=1e-12)
        np.testing.assert_allclose(check["q0_event"], check["q0_binned"], atol=1e-12)
        with self.assertRaises(ValueError):
            validate_compression(compressed, ratios * 2, n_toys=1)

    def test_one_bin_toy_mean_variance_and_poisson_fit(self):
        compressed = build_compression([[2.0, 3.0]], [100.0, 50.0], 1)
        toys = run_toys(compressed, n_toys=20000, seed=70)
        self.assertLess(abs(np.mean(toys["mu_hat"]) - 1), 0.004)
        self.assertLess(abs(np.var(toys["mu_hat"]) - 150 / 100**2), 0.0005)
        rng = np.random.default_rng(70)
        counts = rng.poisson(150, size=20000)
        expected_mu = np.maximum((counts - 50) / 100, 0)
        expected_q0 = np.where(counts > 50, 2 * (counts * np.log(counts / 50) - counts + 50), 0)
        np.testing.assert_allclose(toys["mu_hat"], expected_mu, atol=1e-13)
        np.testing.assert_allclose(toys["q0"], expected_q0, atol=1e-10)
        empty = run_toys(compressed, n_toys=0)
        self.assertEqual(empty["mu_hat"].shape, (0,))

    def test_toy_sources_frozen_likelihood_and_reproducibility(self):
        compressed = build_compression(self.ratios, self.yields, 17)
        ps = simulator_bin_probabilities(self.ratios[:100], self.ratios[-100:], compressed)
        a = run_toys(compressed, n_toys=21, seed=9, source="simulator", probabilities=ps, batch_size=3)
        b = run_toys(compressed, n_toys=21, seed=9, source="simulator", probabilities=ps, batch_size=7)
        np.testing.assert_array_equal(a["mu_hat"], b["mu_hat"])
        np.testing.assert_array_equal(a["q0"], b["q0"])
        self.assertEqual(a["model_sha256"], compressed["model_sha256"])
        with self.assertRaises(ValueError):
            run_toys(compressed, probabilities=ps)
        with self.assertRaises(ValueError):
            run_toys(compressed, source="simulator")

    def test_asymptotic_zero_atoms_and_total_probability(self):
        edges = np.array([0, 1e-12, 1, np.inf])
        muh = bounded_muhat_bin_probabilities(edges, 1, 0.5)
        q0 = q0_bin_probabilities(edges, 4)
        self.assertAlmostEqual(muh[0], ndtr(-2), places=10)
        self.assertAlmostEqual(q0[0], ndtr(-2), places=6)
        self.assertAlmostEqual(np.sum(muh), 1)
        self.assertAlmostEqual(np.sum(q0), 1)
        null = q0_bin_probabilities([0, 1e-16, np.inf], 0)
        self.assertAlmostEqual(null[0], 0.5, places=7)
        for func in (lambda e: bounded_muhat_bin_probabilities(e, 0, 1), lambda e: q0_bin_probabilities(e, 0)):
            masses = func([-np.inf, -1, 0, 1, np.inf])
            self.assertEqual(masses[0], 0)
            self.assertEqual(masses[1], 0)
            self.assertAlmostEqual(np.sum(masses), 1)
            self.assertAlmostEqual(func([-1, 0])[0], 0.5)

    def test_toy_cache_extends_without_changing_previous_draws(self):
        compressed = build_compression(self.ratios, self.yields, 11)
        with tempfile.TemporaryDirectory() as directory:
            first = run_toys_cached(directory, compressed, n_toys=9, shard_size=5, seed=11)
            extended = run_toys_cached(directory, compressed, n_toys=16, shard_size=5, seed=11)
            again = run_toys_cached(directory, compressed, n_toys=9, shard_size=5, seed=11)
            np.testing.assert_array_equal(first["mu_hat"], extended["mu_hat"][:9])
            np.testing.assert_array_equal(first["q0"], extended["q0"][:9])
            np.testing.assert_array_equal(first["q0"], again["q0"])
            self.assertEqual(extended["n_reused_shards"], 2)
            self.assertEqual(again["n_reused_shards"], 2)
            new_truth = run_toys_cached(directory, compressed, mu_true=0.0, n_toys=9, shard_size=5, seed=11)
            self.assertNotEqual(new_truth["cache_directory"], first["cache_directory"])


if __name__ == "__main__":
    unittest.main()
