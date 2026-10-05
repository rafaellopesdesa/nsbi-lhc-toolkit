"""Read-only tail and numerical checks for the saved signal NCE experiment."""
import copy
import math
import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp


def weight_summary(logweights):
    """Unclipped importance-weight statistics, with stable intermediate scaling."""
    v = np.asarray(logweights, dtype=np.float64)
    if v.ndim != 1 or len(v) < 2 or not np.isfinite(v).all():
        raise ValueError('At least two finite log weights are required.')
    total = logsumexp(v)
    log_mean = total - np.log(len(v))
    fractions = np.exp(v - total)
    scaled = np.exp(v - log_mean)
    rel_se = scaled.std(ddof=1) / np.sqrt(len(v))
    with np.errstate(over='ignore'):
        mean = float(np.exp(log_mean))
    return dict(n=len(v), log_mean=float(log_mean), mean=mean,
                relative_mc_se=float(rel_se), mc_se=float(mean * rel_se),
                ess=float(1 / np.square(fractions).sum()),
                max_log_weight=float(v.max()), largest_weight_fraction=float(fractions.max()),
                top_10_weight_fraction=float(np.sort(fractions)[-10:].sum()))


def gaussian_log_density(x, mean, std):
    z = (np.asarray(x, dtype=np.float64) - mean) / std
    return -.5 * (z*z + np.log(2*np.pi)).sum(-1) - np.log(std).sum()


def tail_table(bank, x, logq, candidate_logr, baseline_logr, exact_logr, mean, std, top_k=20):
    """Union of top events by each ratio; indices refer to the original bank."""
    values = dict(candidate=np.asarray(candidate_logr), baseline=np.asarray(baseline_logr),
                  exact=np.asarray(exact_logr))
    if any(len(v) != len(x) or not np.isfinite(v).all() for v in values.values()):
        raise ValueError('Invalid event arrays.')
    k = min(top_k, len(x))
    if k < 1:
        raise ValueError('top_k must be positive.')
    tops = {name: np.argsort(v)[-k:][::-1] for name, v in values.items()}
    indices = np.unique(np.concatenate(list(tops.values())))
    indices = indices[np.argsort(values['candidate'][indices])[::-1]]
    table = pd.DataFrame(dict(bank=bank, event_index=indices))
    for j in range(x.shape[1]):
        table[f'x{j+1}'] = x[indices, j]
    table['log_q32'] = np.asarray(logq)[indices]
    table['log_p_exact'] = (np.asarray(exact_logr) + logq)[indices]
    table['f_candidate32'] = (np.asarray(candidate_logr) + logq)[indices]
    table['log_g'] = gaussian_log_density(x[indices], mean, std)
    table['candidate_minus_exact_logp'] = table['f_candidate32'] - table['log_p_exact']
    table['bounded_residual_plus_offset'] = table['f_candidate32'] - table['log_g']
    table['gaussian_radius_squared'] = (((x[indices]-mean)/std)**2).sum(-1)
    for name, v in values.items():
        rank = {int(index): i+1 for i, index in enumerate(tops[name])}
        table[f'{name}_top_rank'] = [rank.get(int(index), np.nan) for index in indices]
        table[f'{name}_log_ratio'] = v[indices]
        table[f'{name}_weight_fraction'] = np.exp(v[indices]-logsumexp(v))
    return table


@torch.no_grad()
def precision_probe(reference, candidate, x):
    """Copies on CPU: no changes to live models, device, dtype or checkpoints.

    Evaluate the *same stored physical coordinates* in both precisions. Promoting
    x to double cannot recover coordinates already rounded during generation.
    The x->z->x and z->x->z checks use the nflows transform convention.
    """
    mean, std = reference.mean, reference.std
    flow = copy.deepcopy(reference.flow).cpu().eval()
    model = copy.deepcopy(candidate).cpu().eval()
    result = {}
    latent = np.random.default_rng(8202610).normal(size=(512, x.shape[1]))
    generative_rows = []
    for suffix, dtype in [('32', torch.float32), ('64', torch.float64)]:
        flow = flow.to(dtype=dtype)
        model = model.to(dtype=dtype)
        numpy_dtype = np.float32 if suffix == '32' else np.float64
        standardized = (np.asarray(x, dtype=numpy_dtype)-mean.astype(numpy_dtype))/std.astype(numpy_dtype)
        t = torch.as_tensor(standardized, dtype=dtype)
        z, forward_logdet = flow._transform(t)
        reconstructed, inverse_logdet = flow._transform.inverse(z)
        q = flow._distribution.log_prob(z) + forward_logdet - np.log(std.astype(numpy_dtype)).sum()
        result[f'log_q_cpu{suffix}'] = q.numpy()
        result[f'f_cpu{suffix}'] = model(torch.as_tensor(x, dtype=dtype)).numpy()
        result[f'x_roundtrip_max_std{suffix}'] = (reconstructed-t).abs().amax(1).numpy()
        result[f'logdet_roundtrip_abs{suffix}'] = (forward_logdet+inverse_logdet).abs().numpy()
        result[f'latent_radius_squared{suffix}'] = z.square().sum(1).numpy()
        # A controlled generative check retains the original latent coordinates.
        z0 = torch.as_tensor(latent, dtype=dtype)
        generated, inverse_ld = flow._transform.inverse(z0)
        z1, forward_ld = flow._transform(generated)
        sample_logq = flow._distribution.log_prob(z0)-inverse_ld
        reevaluated_logq = flow._distribution.log_prob(z1)+forward_ld
        generative_rows.append(dict(precision=suffix, n=len(latent),
            max_latent_roundtrip=float((z1-z0).abs().max()),
            max_logdet_cancellation=float((inverse_ld+forward_ld).abs().max()),
            max_logq_disagreement=float((reevaluated_logq-sample_logq).abs().max())))
    frame = pd.DataFrame(result)
    frame['delta_logq_64_minus_32'] = frame.log_q_cpu64-frame.log_q_cpu32
    frame['delta_f_64_minus_32'] = frame.f_cpu64-frame.f_cpu32
    frame['candidate_logr_cpu64'] = frame.f_cpu64-frame.log_q_cpu64
    return frame, pd.DataFrame(generative_rows)
