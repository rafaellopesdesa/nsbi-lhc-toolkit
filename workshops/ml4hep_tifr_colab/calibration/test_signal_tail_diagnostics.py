"""Small numerical checks: python -m unittest test_signal_tail_diagnostics -v."""
import unittest
import numpy as np
import torch
from flow_reference import ReferenceFlow, build_flow
from signal_nce import SignalNCE
from signal_tail_diagnostics import weight_summary, tail_table, precision_probe


class TestSignalTailDiagnostics(unittest.TestCase):
    def test_weights_and_event_identity(self):
        flat = weight_summary(np.zeros(100))
        self.assertAlmostEqual(flat['mean'], 1.)
        self.assertAlmostEqual(flat['ess'], 100.)
        extreme = np.zeros(100)
        extreme[17] = 30.
        stats = weight_summary(extreme)
        self.assertGreater(stats['largest_weight_fraction'], .99999)
        self.assertLess(stats['ess'], 1.001)
        x = np.arange(200).reshape(100, 2).astype(float)
        table = tail_table('test', x, np.zeros(100), extreme, np.zeros(100),
                           np.zeros(100), np.zeros(2), np.ones(2), top_k=2)
        self.assertEqual(table.iloc[0].event_index, 17)
        self.assertEqual(table.iloc[0].candidate_minus_exact_logp, 30.)
        self.assertEqual(table.iloc[0].x1, x[17, 0])

    def test_real_spline_probe_does_not_mutate_models(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        config = dict(n_features=2, n_coupling_layers=2, hidden_features=8,
                      hidden_layers=1, spline_num_bins=4, spline_tail_bound=5.)
        ref = ReferenceFlow(build_flow(config), np.zeros(2), np.ones(2), config)
        model = SignalNCE(np.zeros(2), np.ones(2), width=8, layers=1)
        before = {k:v.clone() for k,v in ref.flow.state_dict().items()}
        x = ref.sample(12, 19)
        frame, generation = precision_probe(ref, model, x)
        self.assertTrue(np.isfinite(frame.to_numpy()).all())
        self.assertLess(frame.x_roundtrip_max_std64.max(), 1e-9)
        self.assertLess(generation.iloc[1].max_logq_disagreement, 1e-9)
        for k,v in ref.flow.state_dict().items():
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)
        self.assertEqual(next(model.parameters()).dtype, torch.float32)
        self.assertEqual(next(ref.flow.parameters()).dtype, torch.float32)


if __name__ == '__main__':
    unittest.main()
