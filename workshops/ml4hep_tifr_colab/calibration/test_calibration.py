"""Small numerical checks for the calibration likelihood and training interfaces.

Run in this directory with ``python -m unittest test_calibration``.  These use
synthetic finite reference banks and tiny CPU networks, never the notebook's
large training samples or saved user checkpoints.
"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.stats import multivariate_normal
import torch

import model
import sampling
from utils_distributions import smearing_parameters


def reference_bank(size=31):
    """Two distinguishable, normalized processes with mild shape variations."""
    x = np.linspace(-2.5, 2.5, size)
    anchors = np.empty((size, 2, 3))
    for process, shift in enumerate((0.7, -0.7)):
        for index, alpha in enumerate((-1., 0., 1.)):
            anchors[:, process, index] = np.exp(
                -0.5 * (x - shift - 0.25 * alpha) ** 2
                + 0.06 * alpha * (process + 1) * np.sin(2. * x)
            )
    return anchors / anchors.mean(axis=0)


def finite_bank_config(anchors, epsilon=0.):
    from interpolation import polynomial_normalization

    return dict(model.DEFAULT_CONFIG, signal_yield=4., background_yield=40.,
                morph_normalization=polynomial_normalization(anchors).tolist(),
                model_epsilon=epsilon)


class NormalizedInterpolationTests(unittest.TestCase):
    def test_tail_positivity_repair_preserves_anchor_derivatives(self):
        from interpolation import (
            derivative_polynomial, evaluate_polynomial, numpy_coefficients,
            tensor_coefficients, tensor_morph,
        )

        ordinary = reference_bank()
        standard, diagnostics = numpy_coefficients(ordinary, return_diagnostics=True)
        self.assertFalse(diagnostics["repair_mask"].any())
        np.testing.assert_array_equal(standard, numpy_coefficients(ordinary, ensure_positive=False))
        for index, alpha in ((0, -1.), (2, 1.)):
            logarithm = np.log(ordinary[..., index] / ordinary[..., 1])
            np.testing.assert_allclose(evaluate_polynomial(standard, alpha), ordinary[..., index],
                                       atol=1e-13)
            np.testing.assert_allclose(derivative_polynomial(standard, alpha),
                                       alpha * ordinary[..., index] * logarithm, atol=1e-13)
            second = standard[..., 2:] * np.arange(2, 9) * np.arange(1, 8)
            np.testing.assert_allclose(evaluate_polynomial(second, alpha),
                                       ordinary[..., index] * logarithm ** 2, atol=2e-13)
        tail = np.array([7.49314668e-16, 1.16555663e-13, 4.21906219e-12])
        anchors = np.array([[tail / tail[1], [0.8, 1., 1.2]]])
        original = numpy_coefficients(anchors, ensure_positive=False)
        repaired, diagnostics = numpy_coefficients(anchors, return_diagnostics=True)
        self.assertTrue(diagnostics["repair_mask"][0, 0])
        self.assertFalse(diagnostics["repair_mask"][0, 1])
        grid = np.linspace(-1., 1., 2001)[:, None]
        self.assertLess(evaluate_polynomial(original, grid).min(), 0.)
        self.assertGreater(evaluate_polynomial(repaired, grid).min(), 0.)
        tensor_anchors = torch.as_tensor(anchors, dtype=torch.float64)
        fast = tensor_coefficients(tensor_anchors, diagnostics["repair_amplitude"])
        np.testing.assert_allclose(fast.numpy(), repaired, rtol=1e-13, atol=1e-13)
        tensor_values = tensor_morph(tensor_anchors, torch.as_tensor(grid), coefficients=fast)
        np.testing.assert_allclose(tensor_values.numpy(), evaluate_polynomial(repaired, grid),
                                   rtol=1e-9, atol=1e-12)
        # The explicit bubble must not alter nominal first-order sensitivity
        # or the C2 matching to the exponential extrapolations.
        for alpha in (-1., 0., 1.):
            np.testing.assert_allclose(evaluate_polynomial(repaired, alpha),
                                       evaluate_polynomial(original, alpha), atol=1e-12)
            np.testing.assert_allclose(derivative_polynomial(repaired, alpha),
                                       derivative_polynomial(original, alpha), atol=1e-11)
        for alpha in (-1., 1.):
            second_original = original[..., 2:] * np.arange(2, 9) * np.arange(1, 8)
            second_repaired = repaired[..., 2:] * np.arange(2, 9) * np.arange(1, 8)
            np.testing.assert_allclose(evaluate_polynomial(second_repaired, alpha),
                                       evaluate_polynomial(second_original, alpha), atol=1e-10)

    def test_anchor_autograd_retains_exponential_first_and_second_derivatives(self):
        from interpolation import tensor_morph

        anchors = reference_bank(7)
        tensors = torch.as_tensor(anchors, dtype=torch.float64)
        for index, alpha in ((0, -1.), (2, 1.)):
            parameter = torch.tensor(alpha, dtype=torch.float64, requires_grad=True)
            value = tensor_morph(tensors, parameter).sum()
            first, = torch.autograd.grad(value, parameter, create_graph=True)
            second, = torch.autograd.grad(first, parameter)
            logarithm = np.log(anchors[..., index] / anchors[..., 1])
            self.assertAlmostEqual(first.item(), (alpha * anchors[..., index] * logarithm).sum(), places=11)
            self.assertAlmostEqual(second.item(), (anchors[..., index] * logarithm ** 2).sum(), places=11)

    def test_ill_conditioned_extreme_anchors_fail_before_likelihood_evaluation(self):
        from interpolation import numpy_coefficients

        with self.assertRaises(FloatingPointError):
            numpy_coefficients(np.array([[[9e-4, .0175, 2.69e6], [1., 1., 1.]]]))

    def test_anchors_and_intermediate_normalization_including_misspecification(self):
        anchors = reference_bank()
        for epsilon in (0., 0.25):
            config = finite_bank_config(anchors, epsilon)
            for alpha in np.linspace(-1., 1., 15):
                shaped = model.morph(anchors, alpha, config)
                self.assertTrue(np.all(shaped > 0.))
                np.testing.assert_allclose(shaped.mean(0), 1., atol=3e-14)
                good = model.morph(anchors, alpha, dict(config, model_epsilon=0.))
                np.testing.assert_allclose(shaped[:, 0], (1. - epsilon) * good[:, 0]
                                           + epsilon * good[:, 1], atol=2e-14)
            for index, alpha in enumerate((-1., 0., 1.)):
                expected = anchors[:, :, index].copy()
                expected[:, 0] = ((1. - epsilon) * expected[:, 0]
                                  + epsilon * expected[:, 1])
                np.testing.assert_allclose(model.morph(anchors, alpha, config), expected,
                                           rtol=2e-13, atol=2e-14)

    def test_numpy_torch_values_and_normalization_gradients(self):
        from interpolation import tensor_morph

        anchors = reference_bank()
        config = finite_bank_config(anchors, epsilon=0.13)
        tensor_anchors = torch.as_tensor(anchors, dtype=torch.float64)
        weights = np.linspace(0.4, 1.9, anchors.shape[0])[:, None] * np.array([[0.7, 1.3]])
        for alpha in (-0.73, -0.2, 0., 0.31, 0.83):
            parameter = torch.tensor(alpha, dtype=torch.float64, requires_grad=True)
            shaped = tensor_morph(tensor_anchors, parameter, config)
            np.testing.assert_allclose(shaped.detach().numpy(), model.morph(anchors, alpha, config),
                                       rtol=3e-12, atol=2e-13)
            (shaped * torch.as_tensor(weights)).sum().backward()
            step = 1e-5
            numerical = np.sum(weights * (
                model.morph(anchors, alpha + step, config)
                - model.morph(anchors, alpha - step, config)
            )) / (2. * step)
            self.assertAlmostEqual(parameter.grad.item(), numerical, delta=2e-7)
            np.testing.assert_allclose(model.morph_derivative(anchors, alpha, config), (
                model.morph(anchors, alpha + step, config)
                - model.morph(anchors, alpha - step, config)
            ) / (2. * step), rtol=3e-7, atol=2e-9)

    def test_same_bank_asimov_recovers_truth_with_and_without_signal_mixing(self):
        anchors = reference_bank(53)
        for epsilon in (0., 0.25):
            config = finite_bank_config(anchors, epsilon)
            for nu, alpha in ((0.6, -0.43), (1., 0.), (1.8, 0.61)):
                weights = model.asimov_weights(anchors, nu, alpha, config)
                self.assertAlmostEqual(weights.sum(), nu * 4. + 40., places=11)
                fitted = model.fit(anchors, alpha, config, weights=weights)
                self.assertTrue(fitted["success"], fitted)
                np.testing.assert_allclose([fitted["nu"], fitted["alpha"]], [nu, alpha], atol=2e-5)

    def test_empty_experiment_has_known_boundary_fit(self):
        config = finite_bank_config(reference_bank())
        anchors = np.empty((0, 2, 3))
        for auxiliary in (-1.4, 0.2, 1.5):
            fitted = model.fit(anchors, auxiliary, config)
            self.assertAlmostEqual(fitted["nu"], 0., places=8)
            self.assertAlmostEqual(fitted["alpha"], np.clip(auxiliary, -1., 1.), places=7)
            self.assertTrue(fitted["success"])


class ReferenceSamplingTests(unittest.TestCase):
    def test_resampling_follows_normalized_morphed_mixture(self):
        from utils import ReferenceSampler

        anchors = reference_bank(19)
        config = finite_bank_config(anchors, epsilon=0.27)
        x = np.broadcast_to(np.arange(len(anchors))[:, None], (len(anchors), 5)).copy()
        sampler = ReferenceSampler(x, anchors)
        mu, alpha, size = 1.4, 0.37, 80_000
        expected = model.intensity(anchors, mu, alpha, config)
        expected /= expected.sum()
        toy = sampler.sample_experiment(mu, alpha, config, np.random.default_rng(96),
                                        epsilon=0.27, n=size)
        indices = toy["x"][:, 0].astype(int)
        observed = np.bincount(indices, minlength=len(anchors)) / size
        self.assertTrue(np.all(np.abs(observed - expected)
                               < 6. * np.sqrt(expected * (1. - expected) / size)))
        np.testing.assert_array_equal(toy["anchors"], anchors[indices])
        self.assertAlmostEqual(toy["proposal_ess"], 1. / np.sum(expected ** 2), places=10)
        empty = sampler.sample_experiment(mu, alpha, config, np.random.default_rng(97),
                                          epsilon=0.27, n=0)
        self.assertEqual(empty["anchors"].shape, (0, 2, 3))
        self.assertEqual(empty["x"].shape, (0, 5))


class PhysicalSamplingTests(unittest.TestCase):
    def test_continuous_detector_density_matches_analytic_mixture(self):
        rng = np.random.default_rng(91)
        x = rng.normal(2., 2., size=(23, 5))
        scale, resolution = smearing_parameters()
        for process in ("signal", "background"):
            components = sampling._components(process)
            for alpha in (-1., -0.4, 0., 0.37, 1.):
                response = scale * (1. + 0.1 * alpha)
                expected = sum(
                    fraction * multivariate_normal.pdf(
                        x, mean=mean * response,
                        cov=covariance * np.outer(response, response)
                        + np.diag(resolution ** 2),
                    )
                    for fraction, mean, covariance in components
                ) / sum(component[0] for component in components)
                np.testing.assert_allclose(
                    sampling.process_density(x, process, alpha), expected,
                    rtol=2e-13, atol=1e-15,
                )

    def test_continuous_sampling_reproduces_mean_and_empty_shape(self):
        rng = np.random.default_rng(92)
        scale, resolution = smearing_parameters()
        alpha, size = 0.43, 40_000
        for process in ("signal", "background"):
            components = sampling._components(process)
            response = scale * (1. + 0.1 * alpha)
            fractions = np.array([component[0] for component in components])
            fractions /= fractions.sum()
            means = np.array([component[1] * response for component in components])
            expected = fractions @ means
            variance = sum(
                fraction * (
                    np.diag(covariance) * response ** 2 + resolution ** 2
                    + (mean * response - expected) ** 2
                )
                for fraction, (_, mean, covariance) in zip(fractions, components)
            )
            samples = sampling.sample_process(process, size, rng, alpha)
            self.assertTrue(np.all(np.abs(samples.mean(0) - expected)
                                   < 6. * np.sqrt(variance / size)))
            self.assertEqual(sampling.sample_process(process, 0, rng, alpha).shape, (0, 5))
            self.assertEqual(sampling.process_density(np.empty((0, 5)), process, alpha).shape, (0,))
        empty_config = dict(model.DEFAULT_CONFIG, signal_yield=0., background_yield=0.)
        toy = sampling.sample_experiment(0., alpha, empty_config, rng)
        self.assertEqual(toy["x"].shape, (0, 5))
        self.assertTrue(-2. <= toy["auxiliary"] <= 2.)


class AmortizedInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.anchors = reference_bank(23)
        self.config = finite_bank_config(self.anchors, epsilon=0.11)
        self.experiments = []
        rng = np.random.default_rng(102)
        for index, size in enumerate((0, 6, 17, 9, 13, 7)):
            self.experiments.append(dict(
                anchors=self.anchors[rng.integers(len(self.anchors), size=size)],
                auxiliary=float(rng.uniform(-1., 1.)), mu=0.2 + 0.3 * index,
                alpha_gen=0., source="reference",
            ))

    def test_packed_likelihood_matches_numpy_and_has_finite_gradients(self):
        from inference import centered_nll, pack_experiments

        data = pack_experiments(self.experiments, "cpu")
        nu = torch.linspace(0.3, 2., len(self.experiments), requires_grad=True)
        alpha = torch.tensor([0., -0.4, 0.6, -0.8, 0.1, 0.], requires_grad=True)
        losses = centered_nll(**data, nu=nu, alpha=alpha, config=self.config)
        expected = []
        for toy, q, a in zip(self.experiments, nu.detach().numpy(), alpha.detach().numpy()):
            expected.append(model.nll(toy["anchors"], q, a, toy["auxiliary"], self.config)
                            - model.nll(toy["anchors"], 1., 0., toy["auxiliary"], self.config))
        np.testing.assert_allclose(losses.detach().numpy(), expected, atol=2e-6, rtol=2e-6)
        losses.sum().backward()
        self.assertTrue(torch.isfinite(nu.grad).all())
        self.assertTrue(torch.isfinite(alpha.grad).all())
        # The empty experiment contributes only the expected count and constraint.
        self.assertAlmostEqual(nu.grad[0].item(), self.config["signal_yield"], places=6)
        self.assertAlmostEqual(alpha.grad[0].item(), -self.experiments[0]["auxiliary"], places=6)

    def test_tiny_training_checkpoint_and_query_independent_global(self):
        from inference import Inference, load_inference, save_inference, train_profiles, train_response

        response, response_history = train_response(self.anchors, self.anchors, self.config,
                                                    steps=3, batch_mu=2, seed=103)
        profile, profile_history = train_profiles(
            self.experiments[:4], self.experiments[4:], self.anchors, self.config,
            epochs=2, batch_size=2, seed=104,
        )
        self.assertTrue(np.isfinite([row["loss"] for row in response_history]).all())
        self.assertTrue(np.isfinite([row["train"] for row in profile_history]).all())
        inference = Inference(response, profile, self.config, epsilon=0.11)
        toy = self.experiments[2]
        scalar = inference.evaluate(toy["anchors"], toy["auxiliary"], 0.8)
        vector = inference.evaluate(toy["anchors"], toy["auxiliary"], np.array([2., 0.8, 0.1]))
        for key in ("global_nu", "global_alpha", "nll_global"):
            self.assertAlmostEqual(scalar[key], vector[key], places=12)
        self.assertAlmostEqual(scalar["statistic"], vector["statistic"][1], places=10)
        own = inference.evaluate(toy["anchors"], toy["auxiliary"], scalar["global_nu"])
        self.assertAlmostEqual(own["statistic"], 0., places=9)
        self.assertTrue(np.isfinite(inference.evaluate(
            self.experiments[0]["anchors"], self.experiments[0]["auxiliary"], 1.
        )["statistic"]))
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            base_config = dict(self.config, model_epsilon=0.)
            (run / "config.json").write_text(json.dumps(base_config))
            (run / "misspecification.json").write_text(json.dumps({"epsilon": 0.11}))
            (run / "hybrid").mkdir()
            (run / "hybrid/manifest.json").write_text(json.dumps({"model_id": "test-hybrid"}))
            save_inference(folder, inference, metadata={
                "config": self.config, "epsilon": 0.11, "hybrid_model_id": "test-hybrid"
            })
            loaded = load_inference(folder)
            restored = loaded.evaluate(toy["anchors"], toy["auxiliary"], np.array([2., 0.8, 0.1]))
            np.testing.assert_allclose(restored["statistic"], vector["statistic"], atol=1e-12)
            np.testing.assert_allclose(loaded.response([0., 0.8, 2.]), inference.response([0., 0.8, 2.]))
            (run / "config.json").write_text(json.dumps(dict(base_config, background_yield=41.)))
            with self.assertRaises(ValueError):
                load_inference(folder)

    def test_global_nuisance_candidate_reused_at_its_own_minimum(self):
        from inference import Inference, ProfileNetwork, ResponseNetwork

        # A pure counting model has an analytic interior MLE: nu=(45-40)/4.
        # Deliberately make the conditional nuisance head worse than the global
        # head so this catches a positive statistic at the reported minimum.
        anchors = np.ones((45, 2, 3))
        config = finite_bank_config(anchors)
        profile = ProfileNetwork(anchors[:2], config, event_width=4, head_width=4)
        response = ResponseNetwork(config, width=4)
        with torch.no_grad():
            for parameter in profile.parameters():
                parameter.zero_()
            profile.global_head[-1].bias.copy_(torch.tensor([
                np.log((1.25 / 3.) / (1. - 1.25 / 3.)), np.arctanh(0.2)
            ]))
            profile.conditional_head[-1].bias.fill_(float(np.arctanh(-0.7)))
        inference = Inference(response, profile, config, epsilon=0.)
        estimate = inference.evaluate(anchors, 0.2, 0.8)
        self.assertAlmostEqual(estimate["global_nu"], 1.25, places=6)
        self.assertAlmostEqual(estimate["global_alpha"], 0.2, places=6)
        own = inference.evaluate(anchors, 0.2, estimate["global_nu"])
        self.assertAlmostEqual(own["statistic"], 0., places=11)
        self.assertGreater(own["conditional_head_T_at_global"], 0.8)
        self.assertAlmostEqual(own["candidate_T_at_global"], 0.04, places=8)
        # Reusing the global nuisance at every query avoids a point jump when
        # the global and conditional heads disagree.
        near = inference.evaluate(anchors, 0.2, estimate["global_nu"] + np.array([-1e-4, 1e-4]))
        self.assertLess(np.max(np.abs(near["statistic"])), 1e-6)


class StatisticFlowTests(unittest.TestCase):
    def test_tiny_three_stage_training_and_checkpoint(self):
        from statistic_flows import ComposedCDF, load_flow, train_flow

        rng = np.random.default_rng(93)
        mu = np.linspace(0., 2., 80)
        values = rng.gamma(shape=1.2, scale=1. + 0.2 * mu)
        values[::8] = 0.
        options = dict(bins=8, hidden=8, epochs=2, batch_size=32)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = folder + "/q.pt"
            q, history = train_flow(mu, values, statistic=True, checkpoint=checkpoint, **options)
            self.assertTrue(np.isfinite(history).all())
            np.testing.assert_allclose(load_flow(checkpoint).cdf(mu, values), q.cdf(mu, values))
            u = ComposedCDF(q).pit(mu, values, rng)
            k, _ = train_flow(mu, u, seed=94, **options)
            v = ComposedCDF(q, k).pit(mu, values, rng)
            g, _ = train_flow(mu, v, seed=95, **options)
            calibrated = ComposedCDF(q, k, g)
            probabilities = np.linspace(0.3, 0.99, 23)
            quantiles = calibrated.ppf(0.8, probabilities)
            np.testing.assert_allclose(calibrated.cdf(0.8, quantiles), probabilities, atol=1e-12)
            self.assertTrue(np.all(np.diff(quantiles) > 0.))
            self.assertEqual(float(calibrated.cdf(0.8, 0., left=True)), 0.)
            self.assertGreater(float(calibrated.cdf(0.8, 0.)), 0.)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
