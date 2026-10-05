"""CPU checks for mixture density, sampling and NCE monitor isolation."""
import contextlib
import io
import tempfile
import unittest
import numpy as np
import torch
from signal_nce import train_signal, log_density, signal_envelope
from signal_mixture_noise import mixture_log_prob, sample_mixture, broad_monitor
from signal_tail_diagnostics import gaussian_log_density


class GaussianReference:
    def log_prob(self, x, batch_size=8192):
        return gaussian_log_density(x, np.zeros(x.shape[1]), np.ones(x.shape[1]))
    def sample(self, n, seed, batch_size=8192):
        return np.random.default_rng(seed).normal(size=(n, 2)).astype('float32')


class TestMixtureNoise(unittest.TestCase):
    def test_density_on_both_classes_and_tails(self):
        ref = GaussianReference()
        mean, std = np.array([1., -1.]), np.array([.8, 1.2])
        signal = np.random.default_rng(8).normal(size=(100, 2)) + mean
        noise = sample_mixture(100, ref, mean, std, 19)
        for x in [signal, noise, np.array([[100., -100.]])]:
            actual = mixture_log_prob(x, ref, mean, std, batch_size=17)
            q = ref.log_prob(x)
            g = gaussian_log_density(x, mean, 2*std)
            expected = np.logaddexp(np.log(.8)+q, np.log(.2)+g)
            np.testing.assert_allclose(actual, expected, rtol=1e-13)
            self.assertTrue(np.isfinite(actual).all())
        self.assertGreater(abs(actual[0]-q[0]), 100)

    def test_component_counts_shuffle_and_seed(self):
        class MarkedReference(GaussianReference):
            def sample(self, n, seed, batch_size=8192):
                return np.full((n, 2), 99., dtype='float32')
        ref = MarkedReference()
        x = sample_mixture(100, ref, np.zeros(2), np.ones(2), 19)
        mask = np.all(x == 99, axis=1)
        self.assertEqual(mask.sum(), 80)
        self.assertFalse(mask[:80].all())
        np.testing.assert_array_equal(x, sample_mixture(100, ref, np.zeros(2), np.ones(2), 19))
        with self.assertRaises(ValueError):
            sample_mixture(101, ref, np.zeros(2), np.ones(2), 19)

    def test_monitor_does_not_change_training(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(19)
        x = rng.normal(size=(100, 2)).astype('float32')
        mean, std = signal_envelope(x, 19)
        ref = GaussianReference()
        noise = sample_mixture(100, ref, mean, std, 21)
        a, b = [mixture_log_prob(v, ref, mean, std) for v in [x, noise]]
        settings = dict(seed=19, epochs=2, batch_size=20, width=8, layers=1,
                        bound=30., learning_rate=.001)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            first, h1 = train_signal(x, noise, a, b, tmp+'/plain', settings, {})
            second, h2 = train_signal(x, noise, a, b, tmp+'/monitor', settings, {},
                                     monitor=broad_monitor(mean, std, n=100, every=1))
        np.testing.assert_array_equal(log_density(first, x), log_density(second, x))
        self.assertEqual([v['val_bce'] for v in h1], [v['val_bce'] for v in h2])
        self.assertTrue(all('broad_log_mean' in v for v in h2))
        np.testing.assert_array_equal(second.mean.numpy(), mean)
        np.testing.assert_array_equal(second.std.numpy(), std)


if __name__ == '__main__':
    unittest.main()
