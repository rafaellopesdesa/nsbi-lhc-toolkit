"""Regression checks for the 02 -> 03 checkpoint handoff; no NN training."""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

import utils_jana as jana
import utils_jana_reuse as reuse


class TrainingAttempt(Exception):
    pass


class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        theta, x = np.zeros((4, 5), dtype=np.float32), np.zeros((4, 8), dtype=np.float32)
        bank = types.SimpleNamespace(path=self.root / "bank.npz", role="test", seed=1,
                                     theta=theta, content_fingerprint="content", file_sha256="file")
        self.data = types.SimpleNamespace(
            theta_train=theta, x_train=x, theta_validation=theta[:2], x_validation=x[:2],
            theta_pilot=theta[:2], x_pilot=x[:2], theta_shape=theta[:2], x_shape=x[:2],
            training_indices=np.arange(4), validation_indices=np.arange(2),
            validation_mode="external_paper", paper_exact_validation_protocol=True,
            training_simulator_calls=4, shape_simulator_calls=2, pilot_simulator_calls=2,
            validation_simulator_calls=2, total_simulator_calls=10,
            master=bank, shape_bank=bank, pilot_bank=bank, validation_bank=bank,
            split_path=None, split_fingerprint=None,
        )
        self.prepare = self.enterContext(patch.object(jana, "prepare_training_slice", return_value=self.data))
        self.enterContext(patch.object(jana, "validate_legacy_runtime", return_value={"test_runtime": True}))
        # This is the first operation AFTER cache validation in the real driver.
        # Any attempted training fails the test before TensorFlow is imported.
        self.training = self.enterContext(patch.object(jana, "_consume_benchmark_shape_rows", side_effect=TrainingAttempt))
        self.original_train = jana.train_exact_jana
        self.enterContext(patch.object(jana, "train_exact_jana", self.original_train))

    def arguments(self, budget):
        return dict(artifact_root=self.root, budget=budget, seed=31082027,
                    shape_bank_path=self.root / "shape.npz", pilot_bank_path=self.root / "pilot.npz",
                    validation_bank_path=self.root / "validation.npz")

    def complete_checkpoint(self, budget, batch_size):
        # Obtain the real driver's contract from tiny stand-in bank arrays,
        # stopping before training. No hand-written copy of its contract logic.
        captured = {}
        mapping_hash = jana._mapping_sha256

        def capture(payload):
            if "optimization" in payload:
                captured.update(copy.deepcopy(payload))
            return mapping_hash(payload)

        with patch.object(jana, "_mapping_sha256", side_effect=capture):
            with self.assertRaises(TrainingAttempt):
                self.original_train(self.root / "bank.npz", batch_size=batch_size, **self.arguments(budget))
        run = jana.default_run_directory(self.root, budget=budget, seed=31082027)
        prefix = run / "joint_jana_checkpoint"
        records = []
        for suffix in (".index", ".data-00000-of-00001"):
            path = Path(str(prefix) + suffix)
            path.write_bytes(b"test checkpoint bytes")
            records.append(dict(relative_path=path.name, bytes=path.stat().st_size, sha256=jana._sha256_file(path)))
        manifest = {**captured, "status": "complete", "checkpoint_prefix": prefix.name,
                    "checkpoint_files": records, "training_contract_sha256": mapping_hash(captured)}
        manifest["checkpoint_artifact_sha256"] = jana._checkpoint_artifact_sha256(manifest)
        jana._atomic_write_json(run / "checkpoint_manifest.json", manifest)
        self.training.reset_mock()
        return run, manifest

    def test_batch_1024_checkpoint_fails_old_path_but_reuses_without_training(self):
        run, manifest = self.complete_checkpoint(1_000_000, 1024)
        before = {p.name: p.read_bytes() for p in run.iterdir() if p.is_file()}
        with self.assertRaisesRegex(RuntimeError, "incompatible or incomplete"):
            self.original_train(self.root / "bank.npz", **self.arguments(1_000_000))
        reuse.install_reuse_hook()
        loaded = jana.train_exact_jana(self.root / "bank.npz", **self.arguments(1_000_000))
        self.assertEqual(loaded, manifest)
        self.assertEqual(before, {p.name: p.read_bytes() for p in run.iterdir() if p.is_file()})
        self.training.assert_not_called()

    def test_smaller_budgets_keep_the_batch_32_contract(self):
        for budget in (10_000, 100_000):
            _, manifest = self.complete_checkpoint(budget, 32)
            with patch.object(jana, "train_exact_jana", self.original_train):
                reuse.install_reuse_hook()
                self.assertEqual(jana.train_exact_jana(self.root / "bank.npz", **self.arguments(budget)), manifest)
            self.training.assert_not_called()

    def test_correction_rebuild_flag_cannot_retrain_flow(self):
        _, manifest = self.complete_checkpoint(1_000_000, 1024)
        reuse.install_reuse_hook()
        loaded = jana.train_exact_jana(self.root / "bank.npz", load_if_available=False, force=True,
                                      **self.arguments(1_000_000))
        self.assertEqual(loaded, manifest)
        self.training.assert_not_called()

    def test_corrupt_checkpoint_is_rejected_without_training(self):
        run, _ = self.complete_checkpoint(1_000_000, 1024)
        (run / "joint_jana_checkpoint.index").write_bytes(b"damaged")
        reuse.install_reuse_hook()
        with self.assertRaisesRegex(RuntimeError, "No complete, intact"):
            jana.train_exact_jana(self.root / "bank.npz", **self.arguments(1_000_000))
        self.training.assert_not_called()

    def test_other_training_contract_changes_still_rejected(self):
        self.complete_checkpoint(1_000_000, 1024)
        reuse.install_reuse_hook()
        with self.assertRaisesRegex(RuntimeError, "incompatible or incomplete"):
            jana.train_exact_jana(self.root / "bank.npz", epochs=2, **self.arguments(1_000_000))
        self.training.assert_not_called()

    def test_missing_checkpoint_cannot_start_training(self):
        reuse.install_reuse_hook()
        with self.assertRaisesRegex(RuntimeError, "03 will not retrain"):
            jana.train_exact_jana(self.root / "bank.npz", **self.arguments(1_000_000))
        self.prepare.assert_not_called()
        self.training.assert_not_called()

    def test_preflight_reports_missing_seed_without_touching_resume(self):
        self.complete_checkpoint(1_000_000, 1024)
        missing = jana.default_run_directory(self.root, budget=1_000_000, seed=31082028)
        state = missing / "resume" / "epoch_000042" / "state.json"
        state.parent.mkdir(parents=True)
        state.write_text('{"completed_epoch": 42}')
        reuse.require_completed_checkpoints(self.root, [(1_000_000, 31082027)])
        with self.assertRaisesRegex(RuntimeError, "budget=1000000/seed=31082028"):
            reuse.require_completed_checkpoints(self.root, [(1_000_000, 31082027), (1_000_000, 31082028)])
        self.assertEqual(state.read_text(), '{"completed_epoch": 42}')
        self.training.assert_not_called()

    def test_subprocess_error_includes_the_underlying_traceback(self):
        options = self.arguments(1_000_000)
        options.pop("budget")
        options.pop("seed")
        failed = subprocess.CompletedProcess([], 1, stdout="loading saved run", stderr="RuntimeError: incompatible checkpoint")
        with patch.object(reuse.subprocess, "run", return_value=failed) as launched:
            with self.assertRaisesRegex(RuntimeError, "RuntimeError: incompatible checkpoint"):
                reuse.launch_saved_campaign("/venv/bin/python", master_bank_path=self.root / "bank.npz",
                                            budgets=[1_000_000], seeds=[31082027], **options)
        command = launched.call_args.args[0]
        self.assertEqual(Path(command[2]).name, "utils_jana_reuse.py")
        self.assertNotIn("--force", command)
        self.assertNotIn("--no-load-if-available", command)

    def test_hybrid_installs_reuse_and_corrected_evaluation_before_campaign(self):
        # Exercise the real entry's wiring without importing the PyTorch stack
        # or generating banks. Stop at the legacy campaign boundary.
        source = ast.parse(Path(__file__).with_name("utils.py").read_text())
        function = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "run_exact_jana_corrections")
        code = ast.Module(body=[function], type_ignores=[])
        corrected_evaluation, corrected_export = object(), object()
        calls = []

        def campaign(*args, **kwargs):
            self.assertIs(jana.launch_isolated_campaign, reuse.launch_saved_campaign)
            self.assertIs(jana._launch_isolated_evaluation, corrected_evaluation)
            self.assertIs(jana._launch_isolated_ratio_export, corrected_export)
            self.assertEqual(calls, ["preflight", "preserve"])
            raise TrainingAttempt

        namespace = {"__name__": "test_entry", "Path": Path,
                     "_config_module": lambda: types.SimpleNamespace(campaign_signature=lambda c: "test"),
                     "_exact_jana_budget_seed_groups": lambda *args, **kwargs: [(1_000_000, (31082027,))],
                     "_launch_checkpoint_compatible_jana_evaluation": corrected_evaluation,
                     "_launch_checkpoint_compatible_jana_ratio_export": corrected_export,
                     "_preserve_nonreusable_exact_jana_evaluations": lambda **kwargs: calls.append("preserve")}
        exec(compile("from __future__ import annotations\n" + ast.unparse(code), "entry", "exec"), namespace)
        with patch.object(jana, "launch_isolated_campaign"), patch.object(jana, "_launch_isolated_evaluation"), \
             patch.object(jana, "_launch_isolated_ratio_export"), patch.object(jana, "run_exact_jana_campaign", side_effect=campaign), \
             patch.object(reuse, "require_completed_checkpoints", side_effect=lambda *args: calls.append("preflight")), \
             patch("utils_jana_gpu.activate_runtime_hooks"):
            with self.assertRaises(TrainingAttempt):
                namespace["run_exact_jana_corrections"](self.root, {}, factorizations=("multiclass", "binary"),
                                                       budgets_to_run=(1_000_000,), ml_seeds_to_run=(31082027,), load_if_available=True)

    def test_notebook_03_sources_match_generator_and_compile(self):
        import generate_notebooks
        notebook = json.loads(Path(__file__).with_name("03_SLCP_hybrid.ipynb").read_text())
        expected = {c["id"]: c["source"] for c in generate_notebooks.NOTEBOOKS["03_SLCP_hybrid.ipynb"]}
        for cell in notebook["cells"]:
            if cell.get("id") in expected:
                self.assertEqual(cell["source"], expected[cell["id"]])
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), cell.get("id", "empty"), "exec")


if __name__ == "__main__":
    unittest.main()
