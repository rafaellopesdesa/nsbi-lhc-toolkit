"""Run with unittest in modern Python and in the pinned JANA interpreter.

The TensorFlow integration test is skipped only when TensorFlow is absent;
it tests real pinned BayesFlow/Adam save-resume on tiny throwaway arrays.
No campaign artifacts, Drive files or scientific trainings are used.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import config
import utils_jana as jana
import utils_jana_gpu as gpu
import utils_jana_training as training


HERE = Path(__file__).parent


def extracted_helpers():
    # Completion/caching helpers do not need the modern PyTorch stack.
    names = {"_exact_jana_ml_seeds_for_budget", "_exact_jana_budget_seed_groups",
             "_expected_ml_seeds_for_method", "_filter_exact_jana_grid",
             "_read_json_mapping", "_unused_recovery_path", "_preserve_evaluation_after_retraining"}
    tree = ast.parse((HERE / "utils.py").read_text())
    code = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
    namespace = {"pd": pd, "Path": Path, "json": json}
    exec(compile("from __future__ import annotations\n" + ast.unparse(code), "helpers", "exec"), namespace)
    return namespace


class ConfigurationTests(unittest.TestCase):
    def test_fingerprinted_driver_and_config_unchanged(self):
        for name, expected in {
            "utils_jana.py": "a289cc136de70bdde5fffa3be174b88ded4929699dffb9108fbc81357efdda5c",
            "requirements_jana.txt": "8033c33495df40123e05e577efb7f9fadc6085795753a51bf69a37386a2a11a2",
            "config.py": "c7f0fddf1e5d63928439efcf201cddc8ac84f28f8943ddd41e5b970309ec04e5",
        }.items():
            self.assertEqual(hashlib.sha256((HERE / name).read_bytes()).hexdigest(), expected)

    def test_three_seeds_for_every_method_and_budget(self):
        helpers = extracted_helpers()
        campaign = config.campaign_config()
        self.assertEqual(config.campaign_signature(campaign), "sha256-e455fa167513")
        for method in campaign["methods"]:
            for budget in campaign["budgets"]:
                self.assertEqual(helpers["_expected_ml_seeds_for_method"](campaign, method=method, budget=budget), config.DEFAULT_ML_SEEDS)
        self.assertEqual(helpers["_exact_jana_budget_seed_groups"](campaign, budgets=(1_000_000,), requested_seeds=(31_082_028,)), ((1_000_000, (31_082_028,)),))
        frame = pd.DataFrame({"budget": [1_000_000] * 3, "ml_seed": config.DEFAULT_ML_SEEDS})
        self.assertEqual(len(helpers["_filter_exact_jana_grid"](frame, campaign)), 3)

    def test_notebook_and_generator_agree(self):
        import generate_notebooks
        actual = json.loads((HERE / "02_SLCP_JANA.ipynb").read_text())
        self.assertEqual(actual["cells"], generate_notebooks.NOTEBOOKS["02_SLCP_JANA.ipynb"])
        sources = {cell["id"]: "".join(cell["source"]) for cell in actual["cells"]}
        self.assertIn("require_gpu=True", sources["paper-02-jana-environment"])
        for cell in actual["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), cell["id"], "exec")

    def test_gpu_paths_are_child_only(self):
        env = {"LD_LIBRARY_PATH": "/colab/modern", "PAPER_SUMMARY_JANA_CUDA_LIBRARY_PATH": "/legacy/cublas:/legacy/cudnn",
               "PAPER_SUMMARY_JANA_CUDA_DATA_DIR": "/legacy/cuda_nvcc", "PATH": "/usr/bin",
               "XLA_FLAGS": "--xla_gpu_cuda_data_dir=/colab/cuda12 --xla_cpu_enable_fast_math=false"}
        result = gpu.subprocess_environment(env)
        self.assertEqual(result["LD_LIBRARY_PATH"], "/legacy/cublas:/legacy/cudnn:/colab/modern")
        self.assertEqual(env["LD_LIBRARY_PATH"], "/colab/modern")
        self.assertEqual(result["PYTHONUNBUFFERED"], "1")
        self.assertEqual(result["XLA_FLAGS"], "--xla_cpu_enable_fast_math=false --xla_gpu_cuda_data_dir=/legacy/cuda_nvcc")
        self.assertEqual(result["PATH"], "/legacy/cuda_nvcc/bin:/usr/bin")
        self.assertEqual(result, gpu.subprocess_environment(result))

    def test_hooks_idempotent_and_opt_in(self):
        module = types.SimpleNamespace(_isolated_jana_subprocess_env=lambda: {"LD_LIBRARY_PATH": "/old"}, launch_isolated_campaign="original")
        with patch.dict(os.environ, {}, clear=True):
            gpu.activate_runtime_hooks(module)
            self.assertEqual(module.launch_isolated_campaign, "original")
        with patch.dict(os.environ, {"PAPER_SUMMARY_JANA_REQUIRE_GPU": "1"}):
            gpu.activate_runtime_hooks(module)
            first = module._isolated_jana_subprocess_env
            gpu.activate_runtime_hooks(module)
            self.assertIs(module._isolated_jana_subprocess_env, first)
            self.assertEqual(module._isolated_jana_subprocess_env()["LD_LIBRARY_PATH"], "/old")
            self.assertIs(module.launch_isolated_campaign, gpu.launch_gpu_campaign)

    def test_no_gpu_is_an_error(self):
        fake_tf = types.SimpleNamespace(__version__="2.12.0", config=types.SimpleNamespace(list_physical_devices=lambda kind: []))
        with patch.dict("sys.modules", {"tensorflow": fake_tf}):
            with self.assertRaisesRegex(RuntimeError, "Refusing CPU"):
                gpu.check_tensorflow_gpu()

    def test_gpu_installer_keeps_venv_symlink_and_isolates_libraries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base_python"
            base.touch()
            interpreter = root / "env" / "bin" / "python"
            interpreter.parent.mkdir(parents=True)
            interpreter.symlink_to(base)
            cuda_root = root / "nvidia" / "cuda_nvcc"
            for name in ("bin/ptxas", "nvvm/libdevice/libdevice.10.bc"):
                path = cuda_root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            responses = [
                subprocess.CompletedProcess([], 0, stdout="Test GPU, driver", stderr=""),
                subprocess.CompletedProcess([], 0),
                subprocess.CompletedProcess([], 0, stdout=json.dumps({"libraries": ["/legacy/lib"], "cuda_root": str(cuda_root)})),
                subprocess.CompletedProcess([], 0),
            ]
            with patch.dict(os.environ, {}, clear=True), patch.object(jana, "_isolated_jana_subprocess_env", side_effect=lambda: os.environ.copy()), patch.object(gpu, "activate_runtime_hooks"), patch.object(gpu.subprocess, "run", side_effect=responses) as calls:
                self.assertEqual(gpu.prepare_gpu_environment(interpreter), interpreter)
                commands = [call.args[0] for call in calls.call_args_list]
                self.assertEqual(commands[1][0], str(interpreter))
                self.assertIn("--no-deps", commands[1])
                self.assertEqual(commands[-1][0], str(interpreter))
                self.assertNotIn("LD_LIBRARY_PATH", os.environ)
                child = calls.call_args_list[-1].kwargs["env"]
                self.assertEqual(child["LD_LIBRARY_PATH"], "/legacy/lib")
                self.assertIn(str(cuda_root), child["XLA_FLAGS"])

    def test_gpu_launcher_uses_existing_campaign_cli(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jana._atomic_write_json(root / "jana_paper" / "campaign_manifest.json", {"status": "test"})
            with patch.object(gpu.subprocess, "run") as launched:
                result = gpu.launch_gpu_campaign("/venv/bin/python", artifact_root=root,
                    master_bank_path=root / "master.npz", shape_bank_path=root / "shape.npz",
                    pilot_bank_path=root / "pilot.npz", validation_bank_path=root / "validation.npz",
                    budgets=[1_000_000], seeds=config.DEFAULT_ML_SEEDS, profile="PAPER")
            self.assertEqual(result, {"status": "test"})
            command = launched.call_args.args[0]
            self.assertEqual(command[:2], ["/venv/bin/python", "-u"])
            options = jana._cli_parser().parse_args(command[3:])
            self.assertEqual(options.budgets, [1_000_000])
            self.assertEqual(options.seeds, list(config.DEFAULT_ML_SEEDS))
            self.assertEqual(options.batch_size, 1024)

    def test_gpu_launcher_scopes_batch_override_to_1m(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jana._atomic_write_json(root / "jana_paper" / "campaign_manifest.json", {"status": "test"})
            with patch.object(gpu.subprocess, "run") as launched:
                gpu.launch_gpu_campaign("/venv/bin/python", artifact_root=root,
                    master_bank_path=root / "master.npz", shape_bank_path=root / "shape.npz",
                    pilot_bank_path=root / "pilot.npz", validation_bank_path=root / "validation.npz",
                    budgets=[10_000, 100_000, 1_000_000], seeds=config.DEFAULT_ML_SEEDS,
                    profile="PAPER", load_if_available=True)
            self.assertEqual(launched.call_count, 2)
            actual = {}
            for call in launched.call_args_list:
                options = jana._cli_parser().parse_args(call.args[0][3:])
                actual.update({budget: options.batch_size for budget in options.budgets})
                self.assertEqual(options.seeds, list(config.DEFAULT_ML_SEEDS))
                self.assertEqual(options.profile, "PAPER")
                self.assertFalse(options.force)
                self.assertFalse(options.no_load_if_available)
                self.assertTrue(call.kwargs["check"])
            self.assertEqual(actual, {10_000: 32, 100_000: 32, 1_000_000: 1024})

    def test_old_evaluation_retired_after_retraining(self):
        helper = extracted_helpers()["_preserve_evaluation_after_retraining"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            output = root / "results" / "standardized"
            run.mkdir()
            output.mkdir(parents=True)
            jana._atomic_write_json(run / "checkpoint_manifest.json", {"checkpoint_artifact_sha256": "new", "training_contract_sha256": "contract"})
            jana._atomic_write_json(output / "evaluation_manifest.json", {"checkpoint_artifact_sha256": "old", "checkpoint_contract_sha256": "contract"})
            helper(run, output)
            self.assertFalse(output.exists())
            self.assertEqual(len(list(output.parent.glob("standardized.recovery-*"))), 1)
            output.mkdir()
            jana._atomic_write_json(output / "evaluation_manifest.json", {"checkpoint_artifact_sha256": "new", "checkpoint_contract_sha256": "contract"})
            helper(run, output)
            self.assertTrue(output.exists())


class FakeCheckpoint:
    optimizer = types.SimpleNamespace(iterations=types.SimpleNamespace(numpy=lambda: 2))

    def write(self, prefix):
        Path(prefix + ".index").write_bytes(b"index")
        Path(prefix + ".data-00000-of-00001").write_bytes(b"weights-and-optimizer")


class PersistenceTests(unittest.TestCase):
    def test_save_fallback_retention_and_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = {"epochs": 3, "seed": 1}
            digest = jana._mapping_sha256(contract)
            state = None
            for epoch in range(1, 4):
                state = training.save_epoch(root, FakeCheckpoint(), epoch=epoch, contract=contract,
                    train_frame=pd.DataFrame({"Epoch": [epoch], "Loss": [1.]}),
                    val_frame=pd.DataFrame({"Epoch": [epoch], "Loss": [2.]}), started_utc="start", previous=state)
            self.assertEqual(len(training._generations(root)), 2)
            self.assertEqual(len(state["history_files"]), 6)
            self.assertEqual(training.find_resume(root, digest)["completed_epoch"], 3)
            (root / state["checkpoint_files"][0]["path"]).write_bytes(b"broken")
            self.assertEqual(training.find_resume(root, digest)["completed_epoch"], 2)
            with self.assertRaisesRegex(RuntimeError, "Incompatible"):
                training.find_resume(root, "different")
            # Non-canonical archives must not participate in resume or pruning.
            junk = root / "resume" / "epoch_000004.replaced-1"
            junk.mkdir()
            (junk / "state.json").write_text("{}")
            self.assertEqual(len(training._generations(root)), 2)

    def test_pending_write_does_not_hide_previous_epoch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = {"epochs": 2}
            arguments = dict(contract=contract, train_frame=pd.DataFrame({"x": [1]}), val_frame=pd.DataFrame({"x": [2]}), started_utc="start")
            state = training.save_epoch(root, FakeCheckpoint(), epoch=1, previous=None, **arguments)
            broken = FakeCheckpoint()
            def fail(prefix):
                Path(prefix + ".index").write_bytes(b"partial")
                raise OSError("interrupted write")
            broken.write = fail
            with self.assertRaises(OSError):
                training.save_epoch(root, broken, epoch=2, previous=state, **arguments)
            self.assertEqual(training.find_resume(root, jana._mapping_sha256(contract))["completed_epoch"], 1)

    def test_rng_and_path_safety(self):
        saved = training._rng_state()
        expected = (random.random(), np.random.rand())
        training._restore_rng(json.loads(json.dumps(saved)))
        self.assertEqual(expected, (random.random(), np.random.rand()))
        self.assertFalse(training._record_valid({"path": "../outside", "bytes": 0, "sha256": ""}, HERE))


@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "Requires the isolated TensorFlow/BayesFlow environment")
class TensorFlowResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tensorflow as tf
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

    def test_real_jana_resume_restores_weights_adam_and_epoch(self):
        import tensorflow as tf
        from bayesflow.trainers import Trainer

        self.assertEqual(tf.__version__, "2.12.0")
        rng = np.random.default_rng(123)
        arrays = {"prior_draws": rng.normal(size=(64, 5)).astype("float32"),
                  "sim_data": rng.normal(size=(64, 8)).astype("float32") * 30}
        validation = {key: value[:16] for key, value in arrays.items()}

        def make_trainer():
            jana.seed_everything(17)
            model = jana.build_exact_jana()
            jana._run_trainer_pilot(model, arrays["prior_draws"][:2], arrays["sim_data"][:2])
            return Trainer(amortizer=model.joint, default_lr=jana.DEFAULT_LEARNING_RATE,
                           configurator=jana.configure_joint, memory=False)

        with tempfile.TemporaryDirectory() as temporary:
            context = {"run_directory": Path(temporary), "seed": 17, "budget": 64}
            first = make_trainer()
            real_save = training.save_epoch
            snapshots = {}

            def interrupt(*args, **kwargs):
                state = real_save(*args, **kwargs)
                snapshots["model"] = [value.numpy().copy() for value in first.amortizer.variables]
                snapshots["adam"] = [value.numpy().copy() for value in first.optimizer.variables()]
                raise KeyboardInterrupt("simulated Colab interruption after epoch 1")

            with patch.object(training, "save_epoch", side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    training.train_resumable(first, arrays, epochs=3, batch_size=32, validation_sims=validation, context=context)
            second = make_trainer()
            original_step = second._train_step
            checked = []

            def checked_step(*args, **kwargs):
                if not checked:
                    self.assertEqual(int(second.optimizer.iterations.numpy()), 2)
                    self.assertEqual(len(second.amortizer.variables), len(snapshots["model"]))
                    self.assertEqual(len(second.optimizer.variables()), len(snapshots["adam"]))
                    for actual, expected in zip(second.amortizer.variables, snapshots["model"]):
                        np.testing.assert_array_equal(actual.numpy(), expected)
                    for actual, expected in zip(second.optimizer.variables(), snapshots["adam"]):
                        np.testing.assert_array_equal(actual.numpy(), expected)
                    checked.append(True)
                return original_step(*args, **kwargs)

            second._train_step = checked_step
            history = training.train_resumable(second, arrays, epochs=3, batch_size=32, validation_sims=validation, context=context)
            self.assertTrue(checked)
            self.assertEqual(int(second.optimizer.iterations.numpy()), 6)
            self.assertEqual(list(history["val_losses"]["Epoch"]), [1, 2, 3])
            self.assertEqual(len(history["train_losses"]), 6)
            self.assertEqual(float(second.optimizer._learning_rate(6)), 0.)

    def test_training_entrypoint_final_manifest_load_and_cache(self):
        rng = np.random.default_rng(44)
        original = jana.train_exact_jana
        try:
            training.install_training_hook()
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                bank = root / "bank.npz"
                theta = rng.normal(size=(32, 5)).astype("float32")
                x = rng.normal(size=(32, 8)).astype("float32") * 30
                jana._atomic_savez(bank, theta=theta, x=x, role=np.asarray("master"), seed=np.asarray(44))
                kwargs = dict(artifact_root=root, budget=32, seed=29,
                    shape_bank_path=None, pilot_bank_path=None, validation_bank_path=None,
                    validation_mode="inside_budget", training_indices=np.arange(16),
                    validation_indices=np.arange(16, 32), epochs=2)
                manifest = jana.train_exact_jana(bank, **kwargs)
                run = Path(manifest["run_directory"])
                self.assertEqual(manifest["status"], "complete")
                self.assertTrue(jana._manifest_checkpoint_valid(run, manifest["training_contract_sha256"]))
                self.assertEqual(manifest["driver_source_sha256"], jana._sha256_file(HERE / "utils_jana.py"))
                loaded = jana.load_exact_jana(run)
                values = jana.evaluate_nominal_log_likelihood(loaded, theta[:2], x[:2])
                self.assertTrue(np.isfinite(values).all())
                sidecar_before = (run / "training_execution.json").read_bytes()
                with patch.object(training, "train_resumable", side_effect=AssertionError("must reuse final checkpoint")):
                    reused = jana.train_exact_jana(bank, **kwargs)
                self.assertEqual(reused["checkpoint_artifact_sha256"], manifest["checkpoint_artifact_sha256"])
                self.assertEqual(sidecar_before, (run / "training_execution.json").read_bytes())
        finally:
            jana.train_exact_jana = original


if __name__ == "__main__":
    unittest.main()
