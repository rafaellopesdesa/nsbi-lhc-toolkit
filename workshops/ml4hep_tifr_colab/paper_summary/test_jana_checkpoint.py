"""Regression tests for seeded restoration, export, and stale inference caches.

Run with unittest in modern Python and in the pinned JANA interpreter. The
BayesFlow tests save tiny synthetic fixtures, without fitting neural networks.
"""
from __future__ import annotations

import importlib.util
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import utils_jana as jana
import utils_jana_checkpoint as recovery


class CacheTests(unittest.TestCase):
    def test_old_results_rejected_and_preserved_without_touching_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / 'joint_jana_checkpoint.index'
            checkpoint.write_bytes(b'untouched training')
            output = root / 'standardized'
            output.mkdir()
            arrays = output / 'route.npz'
            arrays.write_bytes(b'old diagnostics')
            manifest_path = output / 'evaluation_manifest.json'
            manifest = dict(schema=jana.JANA_EVALUATION_SCHEMA, status='complete',
                            output_directory=str(output), output_files=[{
                                'relative_path': arrays.name, 'sha256': jana._sha256_file(arrays)}])
            jana._atomic_write_json(manifest_path, manifest)
            with patch.object(jana, '_evaluation_output_manifest_valid', jana._evaluation_output_manifest_valid):
                recovery.install_evaluation_cache_guard()
                self.assertFalse(jana._evaluation_output_manifest_valid(manifest_path))
                recovery.stamp_inference_manifest(manifest_path)
                self.assertTrue(jana._evaluation_output_manifest_valid(manifest_path))
                self.assertIsNone(recovery.preserve_old_inference(output, manifest_path.name))
                # Simulate an old loader's completed output again.
                jana._atomic_write_json(manifest_path, manifest)
                archive = recovery.preserve_old_inference(output, manifest_path.name)
                self.assertFalse(output.exists())
                self.assertEqual((archive / arrays.name).read_bytes(), b'old diagnostics')
                self.assertEqual(checkpoint.read_bytes(), b'untouched training')

    def test_partial_ratio_export_is_archived(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'ratio'
            output.mkdir()
            (output / 'partial.npz').write_bytes(b'partial')
            archived = recovery.preserve_old_inference(output, 'manifest.json')
            self.assertEqual((archived / 'partial.npz').read_bytes(), b'partial')


@unittest.skipUnless(importlib.util.find_spec('tensorflow') and importlib.util.find_spec('bayesflow'),
                     'requires pinned TensorFlow/BayesFlow runtime')
class RestoreIntegrationTests(unittest.TestCase):
    def setUp(self):
        import tensorflow as tf
        from utils_jana_evaluation import _install_bayesflow_numerical_guards
        self.tf = tf
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.seed = 31082027
        self.enterContext(patch.object(jana, 'load_exact_jana', jana.load_exact_jana))
        _install_bayesflow_numerical_guards()
        jana.seed_everything(self.seed)
        self.model = jana.build_exact_jana()
        jana._materialize_variables(self.model)
        # Make the density anisotropic, so incorrect rotations cannot hide
        # behind an isotropic standard-normal density. No training is needed.
        for layer in self.model.joint.submodules:
            if type(layer).__name__ == 'ActNorm':
                layer.scale.assign(np.linspace(0.8, 1.4, layer.scale.shape[0]).astype(np.float32))
                layer.bias.assign(np.linspace(-0.2, 0.3, layer.bias.shape[0]).astype(np.float32))
        prefix = self.root / 'joint_jana_checkpoint'
        tf.train.Checkpoint(amortizer=self.model.joint).write(str(prefix))
        records = [dict(relative_path=p.name, bytes=p.stat().st_size, sha256=jana._sha256_file(p))
                   for p in jana._checkpoint_files(prefix)]
        self.manifest = dict(schema=jana.JANA_RUNTIME_SCHEMA, status='complete', seed=self.seed,
                             checkpoint_prefix=prefix.name, checkpoint_files=records,
                             training_contract_sha256='synthetic-fixture')
        self.manifest['checkpoint_artifact_sha256'] = jana._checkpoint_artifact_sha256(self.manifest)
        jana._atomic_write_json(self.root / 'checkpoint_manifest.json', self.manifest)
        rng = np.random.default_rng(34)
        self.theta = rng.uniform(-2, 2, (16, 5)).astype(np.float32)
        self.x = rng.normal(size=(16, 8)).astype(np.float32)

    @staticmethod
    def rotations(model):
        return [layer.W.numpy() for layer in model.joint.submodules if type(layer).__name__ == 'Orthogonal']

    def test_repeated_restoration_matches_saved_model_after_unrelated_rng_use(self):
        expected_rotations = self.rotations(self.model)
        self.assertEqual(len(expected_rotations), 10)
        # Establish the historical defect: these tensors are not checkpointed.
        self.assertFalse(any('/W/' in name for name, _ in self.tf.train.list_variables(
            str(self.root / 'joint_jana_checkpoint'))))
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        expected_p = jana.evaluate_nominal_log_posterior(self.model, self.theta, self.x)
        expected_l = jana.evaluate_nominal_log_likelihood(self.model, self.theta, self.x)
        expected_draws = jana.sample_nominal_posterior(self.model, self.x[:2], n_samples=8, seed=765)
        recovery.install_checkpoint_restore_hook()
        installed = jana.load_exact_jana
        recovery.install_checkpoint_restore_hook()
        self.assertIs(jana.load_exact_jana, installed)
        for ambient_seed in (12, 9123):
            jana.seed_everything(ambient_seed)
            loaded = jana.load_exact_jana(self.root, strict_runtime=False)
            for expected, actual in zip(expected_rotations, self.rotations(loaded)):
                np.testing.assert_array_equal(actual, expected)
            np.testing.assert_allclose(jana.evaluate_nominal_log_posterior(loaded, self.theta, self.x), expected_p,
                                       rtol=2e-5, atol=2e-5)
            np.testing.assert_allclose(jana.evaluate_nominal_log_likelihood(loaded, self.theta, self.x), expected_l,
                                       rtol=2e-5, atol=2e-5)
            np.testing.assert_allclose(jana.sample_nominal_posterior(loaded, self.x[:2], n_samples=8, seed=765),
                                       expected_draws, rtol=2e-5, atol=2e-5)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.iterdir()})

    def test_ratio_export_entry_roundtrip_and_reuse(self):
        import utils_jana_ratio_export as exporter
        rng = np.random.default_rng(125)
        for role, n in [('master', 16), ('jana_validation', 300)]:
            theta = rng.uniform(-2, 2, (n, 5)).astype(np.float32)
            x = rng.normal(size=(n, 8)).astype(np.float32)
            np.savez(self.root / f'{role}.npz', theta=theta, x=x, role=np.asarray(role))
        master = jana.load_slcp_bank(self.root / 'master.npz')
        validation = jana.load_slcp_bank(self.root / 'jana_validation.npz')
        self.manifest.update(master_bank={'content_sha256': master.content_fingerprint},
                             validation_bank={'content_sha256': validation.content_fingerprint},
                             requested_budget=16, training_rows=16, paper_exact_validation_protocol=True,
                             training_index_sha256=jana._array_sha256(np.arange(16, dtype=np.int64)),
                             training_array_sha256=jana._array_sha256(master.theta, master.x))
        jana._atomic_write_json(self.root / 'checkpoint_manifest.json', self.manifest)
        # Export the original model as the numerical reference.
        export_model = replace(self.model, manifest=self.manifest, run_directory=self.root)
        expected = jana.export_nominal_ratio_class_bank(
            export_model, master_bank_path=master.path, validation_bank_path=validation.path,
            budget=16, seed=self.seed, output_directory=self.root / 'reference', context_batch_size=128)
        before = {p.name: p.read_bytes() for p in self.root.glob('joint_jana_checkpoint.*')}
        args = ['export-ratio-bank', '--run-directory', str(self.root), '--master-bank', str(master.path),
                '--validation-bank', str(validation.path), '--budget', '16', '--seed', str(self.seed),
                '--output-directory', str(self.root / 'export'), '--context-batch-size', '128', '--allow-runtime-drift']
        self.assertEqual(exporter.main(args), 0)
        current = json.loads((self.root / 'export/manifest.json').read_text())
        self.assertTrue(recovery.inference_manifest_current(current))
        with np.load(expected['arrays_path']) as reference, np.load(current['arrays_path']) as actual:
            for key in ('train_S', 'train_P', 'train_L', 'validation_P', 'validation_L'):
                np.testing.assert_allclose(actual[key], reference[key], rtol=2e-5, atol=2e-5)
        # A rerun must reuse the export without drawing a replacement bank.
        with patch.object(jana, 'sample_nominal_posterior', side_effect=AssertionError('cache missed')):
            self.assertEqual(exporter.main(args), 0)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.glob('joint_jana_checkpoint.*')})


if __name__ == '__main__':
    unittest.main()
