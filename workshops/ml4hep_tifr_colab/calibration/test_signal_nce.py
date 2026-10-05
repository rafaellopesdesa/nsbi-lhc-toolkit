"""Small CPU checks: python -m unittest test_signal_nce -v."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import signal_nce as nce


class TestSignalNCE(unittest.TestCase):
    def test_gaussian_initialization_and_bounded_residual(self):
        model = nce.SignalNCE([1., -1.], [2., .5], width=8, layers=1)
        x = np.array([[0., 1.], [2., 0.]], dtype='float32')
        expected = -.5 * (((x - [1., -1.]) / [2., .5])**2 + np.log(2*np.pi)).sum(1)
        np.testing.assert_allclose(nce.log_density(model, x), expected, rtol=1e-6)
        with torch.no_grad():
            model.net[-1].bias.fill_(10000.)
        np.testing.assert_allclose(nce.log_density(model, x)-expected, 30., rtol=1e-6)

    def test_resume_reuse_and_provenance(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(9)
        signal = rng.normal(1., 1., (160, 2)).astype('float32')
        noise = rng.normal(0., 2., (160, 2)).astype('float32')
        def logq(x):
            return -.5 * ((x/2)**2 + np.log(2*np.pi)).sum(1) - 2*np.log(2.)
        settings = dict(seed=19, epochs=3, batch_size=32, width=8, layers=2,
                        bound=30., learning_rate=.001)
        args = (signal, noise, logq(signal), logq(noise))
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            full, resumed = Path(directory)/'full', Path(directory)/'resumed'
            model, history = nce.train_signal(*args, full, settings, {'reference': 'test'})
            real_save = nce.atomic_save
            def interrupted_save(value, path):
                real_save(value, path)
                if Path(path).name == 'last.pt' and value['epoch'] == 1:
                    raise InterruptedError('Simulated runtime disconnect')
            with patch.object(nce, 'atomic_save', interrupted_save):
                with self.assertRaises(InterruptedError):
                    nce.train_signal(*args, resumed, settings, {'reference': 'test'})
            restored, resumed_history = nce.train_signal(*args, resumed, settings, {'reference': 'test'})
            self.assertEqual(history, resumed_history)
            np.testing.assert_array_equal(nce.log_density(model, signal), nce.log_density(restored, signal))
            checkpoint_time = (resumed/'last.pt').stat().st_mtime_ns
            nce.train_signal(*args, resumed, settings, {'reference': 'test'})
            self.assertEqual(checkpoint_time, (resumed/'last.pt').stat().st_mtime_ns)
            with self.assertRaises(ValueError):
                nce.train_signal(*args, resumed, settings, {'reference': 'changed'})
            best = torch.load(full/'best.pt', weights_only=False)
            self.assertEqual(best['val_bce'], min(row['val_bce'] for row in history))


if __name__ == '__main__':
    unittest.main()
