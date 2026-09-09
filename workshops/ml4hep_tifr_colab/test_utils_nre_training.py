"""Fast, synthetic Exercise-12 training/simulator/cache checks.

Run: python -m unittest test_utils_nre_training -v
PyTorch tests skip when it is not installed; no shared artifacts are accessed.
"""

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import utils_nre as nre


class AcceptAll:
    fingerprint = "test-all-selected-v1"
    yields = np.array([10.0, 2500.0])

    def __call__(self, features):
        return np.ones(len(features), dtype=bool)


class FixedRatios:
    fingerprint = "test-fixed-ratios-v1"

    def __call__(self, features):
        return np.column_stack((np.exp(0.01 * features[:, 0]), np.exp(-0.01 * features[:, 0])))


class SamplingTests(unittest.TestCase):
    def collect(self, role="test", component="signal", n=101, seed=15):
        return np.concatenate(list(nre.selected_feature_chunks(AcceptAll(), role, component, n, seed, 32)))

    def test_reproducibility_exact_counts_and_semantic_seed_independence(self):
        values = self.collect()
        self.assertEqual(values.shape, (101, 5))
        self.assertEqual(values.dtype, np.float32)
        np.testing.assert_array_equal(values, self.collect())
        self.assertFalse(np.array_equal(values, self.collect(role="validation")))
        self.assertFalse(np.array_equal(values, self.collect(component="background")))
        self.assertFalse(np.array_equal(values, self.collect(seed=16)))

    def test_fixed_role_smaller_samples_are_exact_prefixes(self):
        for component in ("signal", "background", "reference"):
            np.testing.assert_array_equal(self.collect(component=component, n=45),
                                          self.collect(component=component, n=123)[:45])

    def test_reference_mixes_selected_not_inclusive_components(self):
        def simulator(component, n, rng):
            values = rng.random((n, 5)).astype(np.float32)
            values[:, 0] = 1.0 if component == "signal" else -1.0
            return values

        def unequal_acceptance(values):
            return values[:, 1] < np.where(values[:, 0] > 0, 0.9, 0.1)

        with patch.object(nre, "simulate_reconstructed", side_effect=simulator):
            values = np.concatenate(list(nre.selected_feature_chunks(
                unequal_acceptance, "reference-check", "reference", 20_000, 923, 1024
            )))
        self.assertTrue(unequal_acceptance(values).all())
        self.assertLess(abs(np.mean(values[:, 0] > 0) - 0.5), 0.02)

    def test_invalid_inputs_and_no_acceptance(self):
        with self.assertRaises(ValueError):
            self.collect(n=0)
        with self.assertRaises(ValueError):
            self.collect(role="")
        with self.assertRaises(ValueError):
            self.collect(component="invalid")
        with self.assertRaises(RuntimeError):
            list(nre.selected_feature_chunks(lambda x: np.zeros(len(x), dtype=bool),
                                             "zero", "signal", 2, 9, 2))


class CacheTests(unittest.TestCase):
    def test_reuse_and_content_corruption_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = nre.cached_features(tmp, AcceptAll(), "train", "signal", 27, 99, 16)
            self.assertFalse(first.flags.writeable)
            with patch.object(nre, "selected_feature_chunks", side_effect=AssertionError("must reuse")):
                second = nre.cached_features(tmp, AcceptAll(), "train", "signal", 27, 99, 16)
            np.testing.assert_array_equal(first, second)
            bank = next((Path(tmp) / "banks" / "features").glob("*/values.npy"))
            with bank.open("r+b") as handle:
                handle.seek(-1, 2)
                value = handle.read(1)
                handle.seek(-1, 2)
                handle.write(bytes([value[0] ^ 1]))
            with self.assertRaisesRegex(nre.CacheError, "Refusing to overwrite"):
                nre.cached_features(tmp, AcceptAll(), "train", "signal", 27, 99, 16)

    def test_ratio_cache_stores_only_ratios_and_distinguishes_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = nre.cached_ratios(tmp, AcceptAll(), FixedRatios(), "toys", "reference", 51, 99, 16)
            self.assertEqual(values.shape, (51, 2))
            self.assertEqual(values.dtype, np.float64)
            self.assertTrue(np.isfinite(values).all() and np.all(values > 0))
            self.assertFalse((Path(tmp) / "banks" / "features").exists())
            alternative = FixedRatios()
            alternative.fingerprint = "other-model"
            nre.cached_ratios(tmp, AcceptAll(), alternative, "toys", "reference", 51, 99, 16)
            self.assertEqual(len(list((Path(tmp) / "banks" / "ratios").iterdir())), 2)

    def test_atomic_failure_cleans_only_own_staging_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "artifact"
            unrelated = Path(tmp) / "prior-training"
            unrelated.mkdir()
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                with nre._new_artifact(artifact):
                    raise RuntimeError("intentional")
            self.assertFalse(artifact.exists())
            self.assertEqual(list(Path(tmp).iterdir()), [unrelated])

    def test_existing_locks_are_not_stolen(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "artifact.lock"
            lock.touch()
            with self.assertRaisesRegex(nre.CacheError, "locked"):
                with nre._new_artifact(Path(tmp) / "artifact"):
                    self.fail("cannot enter locked artifact")
            self.assertTrue(lock.exists())


class SelectionTests(unittest.TestCase):
    def test_presel_uses_dataframe_and_odds_not_raw_score(self):
        class Scaler:
            def transform(self, frame):
                self.columns = tuple(frame.columns)
                return frame.to_numpy()

        class Session:
            def get_inputs(self):
                return [SimpleNamespace(name="in")]

            def get_outputs(self):
                return [SimpleNamespace(name="out")]

            def run(self, names, feeds):
                return [feeds["in"][:, :1]]

        scaler = Scaler()
        selected = nre.Exercise5Selection(Session(), scaler, 2.0, [1.0, 250.0], "mock", {})
        features = np.zeros((4, 5), dtype=np.float32)
        features[:, 0] = [0.0, 0.5, 0.7, 1.0]
        np.testing.assert_array_equal(selected(features), [False, False, True, True])
        self.assertEqual(scaler.columns, nre.FEATURES)

    def test_missing_shared_models_never_retrains(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "will not retrain"):
                nre.load_exercise5_selection(tmp)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_conflicting_metadata_stops_before_loading_onnx(self):
        with tempfile.TemporaryDirectory() as tmp:
            modeldir = Path(tmp) / "models_PRESEL"
            modeldir.mkdir()
            (modeldir / "model0.onnx").touch()
            (modeldir / "model_scaler0.bin").touch()
            bankdir = Path(tmp) / "simulator_toy_banks_hybrid"
            bankdir.mkdir()
            for index, cut in enumerate((2.0, 3.0)):
                np.savez(bankdir / f"signal_selected_q_{index}.npz", presel_ratio_cut=cut,
                         lam_sig=1.0, lam_bkg=250.0)
            with self.assertRaisesRegex(nre.CacheError, "inconsistent"):
                nre.load_exercise5_selection(tmp)


class ConfigTests(unittest.TestCase):
    def test_defaults_match_topology_and_optimizer_schedule(self):
        cfg = nre._training_config(None)
        self.assertEqual((cfg["ensemble_size"], cfg["hidden_layers"], cfg["hidden_features"]), (4, 4, 1024))
        self.assertEqual(cfg["activation"], "swish")
        self.assertEqual((cfg["scheduler_step"], cfg["scheduler_gamma"]), (10, 0.01))

    def test_bad_training_parameters_stop(self):
        for settings in ({"epochs": 0}, {"batch_size": 4.5}, {"learning_rate": float("nan")},
                         {"scheduler_gamma": 2.0}, {"device": "tpu"}, {"activation": "unknown"},
                         {"not_a_setting": 3}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                nre._training_config(settings)

    def test_reused_validation_bank_rejected_before_importing_torch(self):
        rng = np.random.default_rng(71)
        signal, background, reference = [rng.normal(size=(8, 5)) for _ in range(3)]
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "independent banks"):
            nre.train_nre(tmp, signal, background, reference, signal.copy(), background.copy(), reference.copy())


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed")
class TorchSmokeTests(unittest.TestCase):
    def test_small_training_member_resume_and_ratio_averaging(self):
        import torch

        rng = np.random.default_rng(17)
        arrays = [rng.normal(size=(24, 5)).astype(np.float32) for _ in range(6)]
        cfg = {"ensemble_size": 2, "hidden_layers": 1, "hidden_features": 8,
               "epochs": 2, "batch_size": 16, "device": "cpu", "prediction_batch_size": 7}
        with tempfile.TemporaryDirectory() as tmp:
            predictor = nre.train_nre(tmp, *arrays, config=cfg, seed=18)
            output = predictor(arrays[3])
            self.assertEqual(output.shape, (24, 2))
            self.assertEqual(output.dtype, np.float64)
            self.assertTrue(np.all(output > 0))
            with patch.object(nre, "_train_member", side_effect=AssertionError("must reuse members")):
                reused = nre.train_nre(tmp, *arrays, config=cfg, seed=18)
            np.testing.assert_array_equal(output, reused(arrays[3]))
            self.assertEqual(predictor.fingerprint, reused.fingerprint)
            # Directly distinguish arithmetic ratio averaging from score averaging.
            with torch.no_grad():
                for index, model in enumerate(predictor.models["signal"]):
                    for parameter in model.parameters():
                        parameter.zero_()
                    model[-1].bias.fill_(float(index * 2))
            np.testing.assert_allclose(predictor(arrays[3])[:, 0], (1 + np.exp(2)) / 2, rtol=1e-7)
            saved = next((Path(tmp) / "training").glob("*/signal/member_00/weights.npz"))
            with saved.open("ab") as handle:
                handle.write(b"corrupt")
            with self.assertRaises(nre.CacheError):
                nre.train_nre(tmp, *arrays, config=cfg, seed=18)


if __name__ == "__main__":
    unittest.main()
