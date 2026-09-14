"""Regression checks for Exercise 6 NIS tuning (python -m unittest test_utils_nis)."""

import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import utils_nf
from utils_nis import (
    conditional_variance_terms, epsilon_diagnostics,
    mixture_log_weights, mix_quadratures,
)


class TestNISTuning(unittest.TestCase):
    def test_conditional_gradient_includes_acceptance(self):
        log_ratio = np.array([-1.7, -0.3, 0.5, 2.1, -0.8])
        a2 = np.array([0.3, 2.0, 0.1, 4.0, 1.2])
        value, coefficient = conditional_variance_terms(log_ratio, a2, 0.1)
        finite_difference = []
        for index in range(len(log_ratio)):
            step = np.zeros(len(log_ratio))
            step[index] = 1.0e-5
            plus, _ = conditional_variance_terms(log_ratio + step, a2, 0.1)
            minus, _ = conditional_variance_terms(log_ratio - step, a2, 0.1)
            finite_difference.append((plus - minus) / 2.0e-5)
        np.testing.assert_allclose(coefficient / len(log_ratio), finite_difference,
                                   rtol=1e-7, atol=1e-8)
        shifted, shifted_coefficient = conditional_variance_terms(log_ratio + 1000, a2, 0.1)
        np.testing.assert_allclose(shifted, value, rtol=1e-11)
        np.testing.assert_allclose(shifted_coefficient, coefficient, atol=1e-11)
        self.assertAlmostEqual(float(coefficient.sum()), 0.0, places=12)

    def test_defensive_weights_and_direct_limit(self):
        log_ratio = np.array([-1000.0, 0.0, 1000.0])
        for epsilon in (0.01, 0.1, 0.5, 1.0):
            rho = np.exp(mixture_log_weights(log_ratio, epsilon))
            self.assertTrue(np.isfinite(rho).all())
            self.assertTrue(np.all(rho <= 1.0 / epsilon * (1 + 1e-12)))
        np.testing.assert_array_equal(mixture_log_weights(log_ratio, 1.0), 0)
        table = epsilon_diagnostics(np.zeros(3), np.array([1, 2, 3]),
                                    np.array([3, 1, 4]), (0.01, 0.1, 1.0))
        np.testing.assert_allclose(table["predicted_scan_gain"], 1)
        np.testing.assert_allclose(table["predicted_q0_gain"], 1)

    def test_paired_mixture_and_prefixes(self):
        reference = {"signal": np.arange(6.0), "background": np.arange(6.0) + 1,
                     "log_g_over_q": np.zeros(6)}
        proposal = {key: value + 10 for key, value in reference.items()}
        u = np.array([0.05, 0.7, 0.3, 0.01, 0.9, 0.2])
        signal, background, log_w = mix_quadratures(reference, proposal, u, 0.1)
        np.testing.assert_array_equal(signal, np.where(u < 0.1, reference["signal"], proposal["signal"]))
        short_q = {key: value[:3] for key, value in reference.items()}
        short_g = {key: value[:3] for key, value in proposal.items()}
        shortened = mix_quadratures(short_q, short_g, u[:3], 0.1)
        for full, short in zip((signal, background, log_w), shortened):
            np.testing.assert_array_equal(full[:3], short)

    def test_actual_weighted_trainer_uses_global_weights(self):
        class Gaussian(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.location = torch.nn.Parameter(torch.tensor(0.0))

            def log_prob(self, batch):
                return -0.5 * (batch[:, 0] - self.location)**2

        gradients = []

        class RecordingAdamW(torch.optim.AdamW):
            def step(self, closure=None):
                gradients.append(float(self.param_groups[0]["params"][0].grad))
                # Hold the model fixed so minibatch gradients can be compared.

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
        model = Gaussian()
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(utils_nf, "build_flow", return_value=model), \
                patch.object(utils_nf, "_save_flow", side_effect=lambda path, *args, **kwargs: path), \
                patch.object(utils_nf.torch.optim, "AdamW", RecordingAdamW):
            utils_nf.train_flow(
                "test", frame, features=["x"], model_dir=directory,
                model_config={"flow_type": "realnvp", "n_features": 1},
                training_config={"batch_size": batch_size, "n_epochs": 1,
                                 "learning_rate": 0.0, "min_learning_rate": 0.0,
                                 "validation_fraction": fraction, "gradient_clip": float("inf")},
                device=torch.device("cpu"), sample_weights=weights,
                load_if_available=False, seed=seed,
            )
        sizes = [min(batch_size, len(train_x) - start)
                 for start in range(0, len(train_x), batch_size)]
        actual = float(np.dot(gradients, sizes) / len(train_x))
        self.assertAlmostEqual(actual, expected, places=5)

    def test_notebook_code_parses_and_selection_precedes_test(self):
        path = Path(__file__).with_name(
            "Exercise_6_NeuralImportanceSampling_Asimov_SameSampleNorm.ipynb"
        )
        notebook = json.loads(path.read_text())
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]), filename=f"cell_{index}")
        cells = ["".join(cell["source"]) for cell in notebook["cells"]]
        self.assertIn('DEFENSIVE_REFERENCE_FRACTION = float(selected["epsilon"])', cells[14])
        self.assertNotIn('DEFENSIVE_REFERENCE_FRACTION =', cells[16])
        self.assertNotIn('DEFENSIVE_REFERENCE_FRACTION =', cells[22])


if __name__ == "__main__":
    unittest.main()
