"""Numerical checks for frozen-model diagnostics, without training dependencies.

Run here with ``python -m unittest test_ratio_diagnostics``. Only NumPy, SciPy,
pandas and Matplotlib are needed; no model files, Torch, or ONNX runtime are used.
"""

from types import SimpleNamespace
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

from ratio_diagnostics import _calibration_bins, diagnostic_report
from sampling import process_density, process_log_density, sample_process
from utils import RatioEnsemble


class RatioDiagnosticsTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_arithmetic_ratio_ensemble_and_paired_oracle_loss(self):
        rng = np.random.default_rng(441)
        # For N(1,1)/N(0,1), the exact log ratio is x - 1/2.
        exact_num = rng.normal(.5, 1, 500)
        exact_den = rng.normal(-.5, 1, 500)
        members_num = exact_num + np.log([.5, 1.5])[:, None]
        members_den = exact_den + np.log([.5, 1.5])[:, None]
        result = diagnostic_report(members_num, members_den, exact_num, exact_den)
        summary = result["summary"].set_index("model")
        ensemble = summary.loc["Ensemble"]
        # The arithmetic mean of these ratios is exact; a mean of logits or
        # classifier probabilities would fail this cancellation.
        self.assertLess(abs(ensemble.mean_log_error_num), 1e-14)
        self.assertLess(ensemble.rms_log_error_den, 1e-14)
        self.assertLess(abs(ensemble.bce_excess_exact), 1e-14)
        self.assertLess(ensemble.bce_excess_exact_se, 1e-14)
        self.assertEqual(summary.loc["Exact ratio", "bce_excess_exact"], 0)
        self.assertEqual(summary.loc["Exact ratio", "bce_excess_exact_se"], 0)
        self.assertAlmostEqual(summary.loc["Member 1", "mean_log_error_num"], np.log(.5))
        for _, rows in result["bins"].query("diagnostic == 'calibration'").groupby(["view", "model"]):
            self.assertEqual(rows.n_num.sum(), 500)
            self.assertEqual(rows.n_den.sum(), 500)
            self.assertAlmostEqual(rows.mass_balanced.sum(), 1)

    def test_empty_and_single_class_bins_include_score_endpoints(self):
        numerator = expit(np.array([-1000., 1000., 1000.]))
        denominator = expit(np.array([-1000., -1000., 0.]))
        bins = _calibration_bins(numerator, denominator, np.array([0., .25, .5, .75, 1.]))
        np.testing.assert_array_equal(bins.n_num, [1, 0, 0, 2])
        np.testing.assert_array_equal(bins.n_den, [2, 0, 1, 0])
        self.assertTrue(np.isnan(bins.fraction_num.iloc[1]))
        self.assertTrue(np.isnan(bins.fraction_num_low95.iloc[1]))
        # Even an observed fraction of exactly 0 or 1 has a nonzero interval.
        self.assertGreater(bins.fraction_num_high95.iloc[2], 0)
        self.assertLess(bins.fraction_num_low95.iloc[3], 1)
        self.assertEqual(bins.n_total.sum(), 6)

    def test_reject_invalid_shapes_and_nonfinite_log_ratios(self):
        valid = (np.zeros((2, 4)), np.zeros((2, 4)), np.zeros(4), np.zeros(4))
        for index, replacement in (
            (0, np.zeros(4)), (1, np.zeros((2, 3))), (2, np.zeros((1, 4))),
            (3, np.array([0., np.nan, 0., 0.])),
            (0, np.full((2, 4), np.inf)),
        ):
            args = list(valid)
            args[index] = replacement
            with self.subTest(index=index, shape=replacement.shape):
                with self.assertRaises(ValueError):
                    diagnostic_report(*args)
        with self.assertRaises(ValueError):
            diagnostic_report(*valid, score_edges=[0., .5, .5, 1.])

    def test_exact_log_density_matches_pdf_and_survives_underflow(self):
        rng = np.random.default_rng(20261001)
        for process in ("signal", "background"):
            for alpha in (-1., 0., 1.):
                with self.subTest(process=process, alpha=alpha):
                    events = sample_process(process, 200, rng, alpha)
                    np.testing.assert_allclose(
                        np.exp(process_log_density(events, process, alpha)),
                        process_density(events, process, alpha), rtol=4e-14, atol=0)
                    far = np.full((3, 5), 1e4)
                    self.assertTrue(np.all(process_density(far, process, alpha) == 0))
                    self.assertTrue(np.all(np.isfinite(process_log_density(far, process, alpha))))
                    self.assertEqual(process_log_density(events[:0], process, alpha).shape, (0,))
                    self.assertEqual(process_log_density(events[:1], process, alpha).shape, (1,))

    def test_frozen_member_predictions_preserve_preprocessing_and_chunking(self):
        class Scaler:
            def transform(self, frame):
                if list(frame.columns) != [f"x{i}" for i in range(1, 6)]:
                    raise AssertionError("Feature order changed")
                return frame.to_numpy() * 2

        class Session:
            def __init__(self, shift):
                self.shift, self.calls = shift, []

            def get_inputs(self):
                return [SimpleNamespace(name="features")]

            def run(self, _, mapping):
                values = mapping["features"]
                if values.dtype != np.float32:
                    raise AssertionError("ONNX inputs must be float32")
                self.calls.append(len(values))
                return [(values[:, :1] + self.shift).astype(np.float32)]

        events = np.arange(35, dtype=np.float32).reshape(7, 5) / 10
        sessions = [Session(shift) for shift in (-1, 0, 1)]
        ensemble = RatioEnsemble([(Scaler(), session) for session in sessions])
        logs = ensemble.member_log_ratios(events, batch_size=3)
        self.assertEqual(logs.shape, (3, 7))
        self.assertEqual(logs.dtype, np.float64)
        self.assertTrue(all(session.calls == [3, 3, 1] for session in sessions))
        expected = np.stack([(events[:, 0] * 2 + shift).astype(np.float32) for shift in (-1, 0, 1)])
        np.testing.assert_array_equal(logs, expected)
        np.testing.assert_array_equal(np.exp(logs).mean(axis=0), ensemble(events, batch_size=3))
        self.assertEqual(ensemble.member_log_ratios(events[:0]).shape, (3, 0))
        for batch_size in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                ensemble.member_log_ratios(events, batch_size=batch_size)
        with self.assertRaises(ValueError):
            RatioEnsemble([]).member_log_ratios(events)
        with self.assertRaises(ValueError):
            RatioEnsemble([(Scaler(), Session(np.inf))]).member_log_ratios(events)


if __name__ == "__main__":
    unittest.main()
