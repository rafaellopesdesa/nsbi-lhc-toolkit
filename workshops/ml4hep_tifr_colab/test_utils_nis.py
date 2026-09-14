"""Exercise 6 checks: python -m unittest test_utils_nis."""

import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
import torch

import utils_nf
from utils_nis import (
    ensemble_log_prob_x, ensemble_sample_x, mixture_log_weights, mix_quadratures,
)


class Gaussian(torch.nn.Module):
    def __init__(self, location=0.0):
        super().__init__()
        self.location = torch.nn.Parameter(torch.tensor(float(location)))

    def log_prob(self, batch):
        return -0.5 * (batch[:, 0] - self.location)**2 - 0.5 * np.log(2 * np.pi)

    def sample(self, n):
        return self.location + torch.randn(n, 1)


def gaussian_member(location, scale):
    return {
        "flow": Gaussian(location), "features": ["x"],
        "scaler": utils_nf.Standardizer(np.array([0.0]), np.array([scale])),
    }


class TestNIS(unittest.TestCase):
    def test_ensemble_density_and_samples_agree(self):
        members = [gaussian_member(-2, 1), gaussian_member(1, 0.5), gaussian_member(2, 2)]
        values = np.linspace(-16, 24, 20_001)[:, None]
        density = np.exp(ensemble_log_prob_x(members, values, batch_size=4096))
        expected = np.mean([
            np.exp(-0.5 * (values[:, 0] / scale - location)**2)
            / (scale * np.sqrt(2 * np.pi))
            for location, scale in [(-2, 1), (1, 0.5), (2, 2)]
        ], axis=0)
        np.testing.assert_allclose(density, expected, rtol=2e-5, atol=1e-7)
        self.assertAlmostEqual(trapezoid(density, values[:, 0]), 1.0, places=5)

        torch.manual_seed(19)
        samples = ensemble_sample_x(members, 120_000, batch_size=4096)[:, 0]
        mean = np.mean([-2, 0.5, 4])
        second_moment = np.mean([1 + 4, 0.25 + 0.25, 4 + 16])
        self.assertAlmostEqual(float(samples.mean()), mean, delta=0.03)
        self.assertAlmostEqual(float(np.mean(samples**2)), second_moment, delta=0.12)
        # A short prefix must also sample the mixture, not one member's block.
        self.assertAlmostEqual(float(samples[:10_000].mean()), mean, delta=0.12)

    def test_single_member_preserves_original_sampling_seed(self):
        member = gaussian_member(1, 2)
        torch.manual_seed(71)
        expected = utils_nf.flow_sample_x(member, 123, batch_size=17)
        torch.manual_seed(71)
        actual = ensemble_sample_x([member], 123, batch_size=17)
        np.testing.assert_array_equal(actual, expected)

    def test_mixture_weights_and_endpoints(self):
        log_ratio = np.array([-1000.0, 0.0, 1000.0])
        np.testing.assert_array_equal(mixture_log_weights(log_ratio, 0.0), -log_ratio)
        np.testing.assert_array_equal(mixture_log_weights(log_ratio, 1.0), 0)
        for epsilon in (0.01, 0.1, 0.5):
            weights = np.exp(mixture_log_weights(log_ratio, epsilon))
            self.assertTrue(np.isfinite(weights).all())
            self.assertTrue(np.all(weights <= 1.0 / epsilon * (1 + 1e-12)))
            np.testing.assert_allclose(mixture_log_weights(np.zeros(3), epsilon), 0,
                                       atol=1e-15)

    def test_paired_mixture_preserves_prefixes_and_endpoints(self):
        reference = {"signal": np.arange(6.0), "background": np.arange(6.0) + 1,
                     "log_g_over_q": np.zeros(6)}
        proposal = {key: value + 10 for key, value in reference.items()}
        uniforms = np.array([0.05, 0.7, 0.3, 0.01, 0.9, 0.2])
        for epsilon in (0.0, 0.1, 1.0):
            full = mix_quadratures(reference, proposal, uniforms, epsilon)
            expected = np.where(uniforms < epsilon, reference["signal"], proposal["signal"])
            np.testing.assert_array_equal(full[0], expected)
            short_q = {key: value[:3] for key, value in reference.items()}
            short_g = {key: value[:3] for key, value in proposal.items()}
            shortened = mix_quadratures(short_q, short_g, uniforms[:3], epsilon)
            for actual, short in zip(full, shortened):
                np.testing.assert_array_equal(actual[:3], short)

    def test_actual_weighted_trainer_uses_global_weights(self):
        gradients = []

        class RecordingAdamW(torch.optim.AdamW):
            def step(self, closure=None):
                gradients.append(float(self.param_groups[0]["params"][0].grad))
                # Hold the model fixed to compare minibatch gradients.

        values = np.array([-2., -1., 0., 1., 3., 5., 8.], dtype=np.float32)
        weights = np.array([1., 1., 10., 20., 3., 50., 2.])
        frame = pd.DataFrame({"x": values})
        seed, batch_size, fraction = 37, 2, 0.28
        scaler = utils_nf.Standardizer.fit(values[:, None], sample_weights=weights)
        train_loader, _ = utils_nf._make_loaders(
            scaler.transform(values[:, None]), sample_weights=weights,
            validation_fraction=fraction, batch_size=batch_size, seed=seed,
        )
        train_x, train_w = train_loader.dataset.tensors
        expected = float(torch.mean(-train_w * train_x[:, 0]))
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(utils_nf, "build_flow", return_value=Gaussian()), \
                patch.object(utils_nf, "_save_flow", side_effect=lambda path, *args, **kwargs: path), \
                patch.object(utils_nf.torch.optim, "AdamW", RecordingAdamW):
            utils_nf.train_flow(
                "test", frame, features=["x"], model_dir=directory,
                model_config={"flow_type": "realnvp", "n_features": 1},
                training_config={"batch_size": batch_size, "n_epochs": 1,
                                 "learning_rate": 0.0, "min_learning_rate": 0.0,
                                 "validation_fraction": fraction, "gradient_clip": None},
                device=torch.device("cpu"), sample_weights=weights,
                load_if_available=False, seed=seed,
            )
        sizes = [min(batch_size, len(train_x) - start)
                 for start in range(0, len(train_x), batch_size)]
        self.assertAlmostEqual(float(np.dot(gradients, sizes) / len(train_x)), expected, places=5)

    def test_notebook_code_parses(self):
        path = Path(__file__).with_name(
            "Exercise_6_NeuralImportanceSampling_Asimov_SameSampleNorm.ipynb"
        )
        for index, cell in enumerate(json.loads(path.read_text())["cells"]):
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]), filename=f"cell_{index}")


if __name__ == "__main__":
    unittest.main()
