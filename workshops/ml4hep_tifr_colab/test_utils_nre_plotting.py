"""Small synthetic tests: no training, simulator bank, or GPU is required."""

import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

import utils_nre_plotting as plotting


class NREPlottingTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    @staticmethod
    def result(mean=1.0, sigma=0.2):
        grid = np.linspace(0, 2, 101)
        return {
            "mu_hat": mean,
            "scan_mu": grid,
            "t_scan": ((grid - mean) / sigma) ** 2,
            "sigma_curvature": sigma,
            "q0_asimov": (mean / sigma) ** 2,
        }

    def test_boundary_probability_is_in_first_bin(self):
        result = self.result(mean=0.1, sigma=0.2)
        mu_edges = np.array([0.0, 0.01, 10.0])
        q_edges = np.array([0.0, 0.01, 1000.0])
        mu, q0 = plotting._prediction_bin_probabilities(result, mu_edges, q_edges)
        self.assertAlmostEqual(float(mu.sum()), 1.0)
        self.assertAlmostEqual(float(q0.sum()), 1.0)
        self.assertAlmostEqual(mu[0], norm.cdf((0.01 - 0.1) / 0.2))
        self.assertAlmostEqual(q0[0], norm.cdf(0.1 - 0.5))
        self.assertGreater(mu[0], norm.cdf(-0.5))
        self.assertGreater(q0[0], norm.cdf(-0.5))

    def test_invalid_ratios_rejected(self):
        with self.assertRaises(ValueError):
            plotting._ratios((np.array([1.0, -1.0]), np.ones(2)))
        with self.assertRaises(ValueError):
            plotting._ratios(np.ones((2, 3)))

    def test_raw_boundary_predictions_omitted_but_true_null_retained(self):
        raw_boundary = dict(self.result(0), construction="raw_simulator")
        raw_near_boundary = dict(self.result(0.01), construction="raw_simulator", q0_asimov=1e-15)
        genuine_null = dict(self.result(0), construction="finite_reference", mu_true=0.0)
        toys = {"mu_hat": np.array([0, 0, 0.1, 0.2]), "q0": np.array([0, 0, 0.25, 1.0])}
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(output):
            fig = plotting.plot_toy_comparison(
                toys, toys, [raw_boundary, raw_near_boundary, genuine_null],
                ["Raw boundary", "Raw tiny q0", "Normalized true null"],
                "Boundary check", directory, "boundary_check",
            )
            for ax in fig.axes:
                labels = [text.get_text() for text in ax.get_legend().get_texts()]
                self.assertNotIn("Raw boundary", labels)
                self.assertNotIn("Raw tiny q0", labels)
                self.assertIn("Normalized true null", labels)
            note = "\n".join(text.get_text() for text in fig.axes[0].texts)
            self.assertIn("Raw boundary prediction omitted: Raw boundary", note)
            self.assertIn("Raw boundary prediction omitted: Raw tiny q0", note)
        self.assertIn("Omitting both asymptotic overlays for Raw boundary", output.getvalue())
        self.assertIn("Omitting both asymptotic overlays for Raw tiny q0", output.getvalue())

    def test_all_figures_export_and_run_standalone(self):
        rng = np.random.default_rng(42)
        ratios = np.exp(rng.normal(size=(180, 2)))
        histories = {
            key: [{"member": 0, "train_loss": [0.7, 0.6, 0.55],
                   "validation_loss": [0.72, 0.64, 0.60], "best_epoch": 2}]
            for key in ("signal", "background")
        }
        raw = [[self.result(0.9), self.result(1.03)], [self.result(1.1), self.result(0.98)]]
        corrected = [[self.result(), self.result()], [self.result(), self.result()]]
        toy_mu = np.maximum(0, rng.normal(1, 0.2, 1000))
        toys = {"mu_hat": toy_mu, "q0": (toy_mu / 0.2) ** 2}
        # Include exact zero values so log-axis export exercises boundary bins.
        toys["mu_hat"][:2] = 0.0
        toys["q0"][:2] = 0.0
        initial_font_size = plt.rcParams["font.size"]
        with tempfile.TemporaryDirectory() as directory:
            plotting.plot_training(histories, directory)
            plotting.plot_ratio_validation(ratios, ratios[:80], ratios[80:], directory)
            plotting.plot_mle_convergence({"sizes": [100, 1000], "raw": raw, "corrected": corrected}, directory)
            plotting.plot_asimov_scans(raw[0], [100, 1000], "Direct simulator Asimov", directory, "scans")
            fig = plotting.plot_toy_comparison(toys, toys, [self.result()], ["Corrected Asimov"],
                                              "Toy comparison", directory, "toys")
            self.assertEqual(fig.axes[1].get_yscale(), "log")
            validation = {
                "source": "simulator", "mu_hat_event": toy_mu[:20],
                "mu_hat_binned": toy_mu[:20] + 1e-5,
                "q0_event": toys["q0"][:20], "q0_binned": toys["q0"][:20] + 1e-4,
            }
            plotting.plot_compression_validation(validation, directory)
            simulator_pdf = Path(directory) / "nre_compression_validation_simulator.pdf"
            simulator_script = simulator_pdf.with_suffix(".py")
            simulator_content = simulator_script.read_bytes()
            validation["source"] = "model"
            plotting.plot_compression_validation(validation, directory)
            self.assertTrue((Path(directory) / "nre_compression_validation_model.pdf").is_file())
            self.assertEqual(simulator_script.read_bytes(), simulator_content)
            self.assertTrue(simulator_pdf.is_file())
            script_files = sorted(Path(directory).glob("*.py"))
            self.assertEqual(len(script_files), 8)
            self.assertEqual(plt.rcParams["font.size"], initial_font_size)
            exported = (Path(directory) / "toys.py").read_text()
            self.assertIn("Simulator toys", exported)
            self.assertIn("NRE-model toys", exported)
            self.assertIn("Zero-boundary mass", exported)
            for script in script_files:
                self.assertTrue(script.with_suffix(".pdf").is_file())
                env = dict(os.environ, MPLBACKEND="Agg")
                env.pop("PYTHONPATH", None)
                # A different CWD and no tutorial PYTHONPATH establish that
                # each script has its arrays and does not import the helpers.
                completed = subprocess.run([sys.executable, str(script)], cwd=directory,
                                           env=env, capture_output=True, text=True, timeout=60)
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                self.assertTrue(script.with_suffix(".png").is_file())
                self.assertTrue(script.with_suffix(".pdf").stat().st_size > 1000)


if __name__ == "__main__":
    unittest.main()
