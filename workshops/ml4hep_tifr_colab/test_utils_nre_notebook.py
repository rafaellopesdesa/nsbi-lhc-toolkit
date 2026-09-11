"""Execute Exercise 12's analysis cells with a deterministic test predictor.

This checks notebook wiring, caches, inference and figure export without a GPU,
ONNX selection artifact, or neural training. It does not test learned-model
accuracy. The actual simulator is used with an accept-all test selection.

Run beside the notebook: python -m unittest test_utils_nre_notebook -v
"""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class TestSelection:
    fingerprint = "notebook-test-selection-v1"
    yields = np.array([5.0, 20.0])

    def __call__(self, features):
        return np.ones(len(features), dtype=bool)


class TestPredictor:
    fingerprint = "notebook-test-predictor-v1"
    histories = {
        component: [dict(train_loss=[0.7, 0.65], validation_loss=[0.71, 0.67],
                         learning_rate=[1e-4, 1e-5], selected_epoch=2,
                         selected_learning_rate=1e-5, best_validation_loss=0.67,
                         best_epoch=2, seed=1, member=0)]
        for component in ("signal", "background")
    }

    def __call__(self, features):
        probability = 0.1 + 0.8 / (1.0 + np.exp(-0.4 * features[:, 0]))
        return np.column_stack((2.0 * probability, 2.0 * (1.0 - probability)))


class NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(
            (Path(__file__).parent / "Exercise_12_NREAsimov.ipynb").read_text()
        )

    def test_notebook_without_errors_and_compilable_cells(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        ids = [cell["id"] for cell in self.notebook["cells"]]
        self.assertEqual(len(ids), len(set(ids)))
        for index, cell in enumerate(self.notebook["cells"]):
            if cell["cell_type"] == "code":
                self.assertFalse(any(output["output_type"] == "error" for output in cell["outputs"]))
                compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")

    def test_full_configuration_and_gpu_guard(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            namespace = {"WORK_DIR": Path(tmp), "ARTIFACT_ROOT": Path(tmp) / "artifacts"}
            source = "".join(self.notebook["cells"][3]["source"])
            fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
            with patch.dict("sys.modules", {"torch": fake_torch}):
                exec(source, namespace)
            self.assertEqual(namespace["N_TRAIN_PER_CLASS"], 5_000_000)
            self.assertEqual(namespace["DEPLOYMENT_REF_EVENTS"], 5_000_000)
            self.assertEqual(namespace["SIMULATOR_TOY_BANK_EVENTS"], 5_000_000)
            self.assertFalse(namespace["TRAIN_NRE"])
            cfg = namespace["TRAINING_CONFIG"]
            self.assertEqual((cfg["epochs"], cfg["ensemble_size"], cfg["device"]), (140, 4, "cuda"))
            self.assertIsNone(cfg["patience"])
            self.assertEqual(cfg["checkpoint_selection"], "last")
            self.assertEqual(namespace["FIGURE_SCRIPT_DIR"].name, "full")
            self.assertEqual(namespace["MC_SIZES"][-1], 2_000_000)
            fake_torch.cuda.is_available = lambda: False
            with patch.dict("sys.modules", {"torch": fake_torch}), self.assertRaisesRegex(RuntimeError, "GPU runtime"):
                exec(source, namespace)
            self.assertFalse((namespace["ARTIFACT_ROOT"] / "banks").exists())

    def test_analysis_cells_with_mock_training(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            namespace = {
                "Path": Path,
                "WORK_DIR": Path(tmp),
                "ARTIFACT_ROOT": Path(tmp) / "artifacts",
                "FIGURE_SCRIPT_DIR": Path(tmp) / "figures",
                "display": lambda value: None,
            }
            for index, cell in enumerate(self.notebook["cells"]):
                if cell["cell_type"] != "code" or index == 2:
                    continue  # Do not mount Drive, clone, or install packages.
                source = "".join(cell["source"])
                if index == 3:
                    source = source.replace('PROFILE = "FULL"', 'PROFILE = "SMOKE"')
                exec(compile(source, f"notebook-cell-{index}", "exec"), namespace)
                if index == 3:
                    namespace["load_exercise5_selection"] = lambda *args, **kwargs: TestSelection()
                    namespace["load_nre"] = lambda *args, **kwargs: TestPredictor()
                    def unexpected_training(*args, **kwargs):
                        self.fail("The default figure workflow must not open training banks or train.")
                    namespace["train_nre"] = unexpected_training
                    namespace["cached_features"] = unexpected_training
                plt.close("all")

            self.assertEqual(namespace["PROFILE"], "SMOKE")
            self.assertLess(np.max(np.abs(namespace["corrected_mles"] - 1.0)), 1e-7)
            for name in ("model_toys", "simulator_toys"):
                self.assertEqual(len(namespace[name]["mu_hat"]), namespace["N_TOYS"])
            self.assertEqual(set(namespace["compression_checks"]), {"model", "simulator"})
            figures = namespace["FIGURE_SCRIPT_DIR"]
            scripts = list(figures.glob("*.py"))
            self.assertEqual(len(scripts), 13)
            for script in scripts:
                self.assertTrue(script.with_suffix(".pdf").is_file())
                compile(script.read_text(), str(script), "exec")
            for name in ("asimov_convergence.csv", "training_summary.csv",
                         "simulator_score_closure.csv", "simulator_score_closure_summary.csv"):
                self.assertTrue((namespace["RESULT_DIR"] / name).is_file())
            self.assertEqual(len(namespace["SCORE_DIAGNOSTICS"]), namespace["MC_REPETITIONS"])
            self.assertEqual(namespace["score_summary"]["n_banks"], namespace["MC_REPETITIONS"])
            self.assertEqual(namespace["FIGURE_SCRIPT_DIR"].name, "smoke")

    def test_compression_refines_without_retraining(self):
        # Exercise the FULL-only control flow with small synthetic banks. The
        # first paired check deliberately fails; the second delegates to the
        # real validator, with its reported numerical errors controlled here.
        import utils_nre_inference as inference
        ratios = TestPredictor()(np.random.default_rng(12).normal(size=(256, 5)))
        deployment = inference.finite_reference_asimov(ratios, TestSelection.yields)
        calls = []

        def validate(*args, **kwargs):
            result = inference.validate_compression(*args, **kwargs)
            calls.append(result["source"])
            result["rms_delta_mu_hat"] = 1.0 if len(calls) <= 2 else 0.0
            result["rms_delta_q0"] = 0.0
            return result

        namespace = {
            "np": np, "plt": plt, "ARTIFACT_ROOT": None,
            "selection": TestSelection(), "nre": TestPredictor(),
            "SIMULATOR_TOY_BANK_EVENTS": 256, "SEED": 12,
            "cached_ratios": lambda *args, **kwargs: ratios,
            "deployment_ratios": ratios, "DEPLOYMENT": deployment,
            "YIELDS": TestSelection.yields, "MU_TRUE": 1.0,
            "SCAN_MU": deployment["scan_mu"], "N_Q_BINS": 32,
            "MAX_Q_BINS": 64, "N_VALIDATION_TOYS": 2, "PROFILE": "FULL",
            "build_compression": inference.build_compression,
            "simulator_bin_probabilities": inference.simulator_bin_probabilities,
            "validate_compression": validate,
            "binned_asimov": lambda *args, **kwargs: deployment,
            "plot_compression_validation": lambda *args: None,
            "FIGURE_SCRIPT_DIR": None,
        }
        with contextlib.redirect_stdout(io.StringIO()):
            exec("".join(self.notebook["cells"][14]["source"]), namespace)
        self.assertEqual(namespace["effective_bins"], 64)
        self.assertEqual(calls, ["model", "simulator", "model", "simulator"])
        self.assertTrue(namespace["COMPRESSION_OK"])


if __name__ == "__main__":
    unittest.main()
