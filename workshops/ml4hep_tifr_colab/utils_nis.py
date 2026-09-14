"""Density ensembles and sampling weights for Exercise 6."""

import numpy as np
from scipy.special import logsumexp
import torch

from utils_nf import flow_log_prob_x, flow_sample_x


def ensemble_log_prob_x(members, values, batch_size=65_536):
    """Log density of the equally weighted flow ensemble, before PRESEL."""
    if len(members) == 1:
        return flow_log_prob_x(members[0], values, batch_size=batch_size)
    log_densities = np.stack([
        flow_log_prob_x(member, values, batch_size=batch_size)
        for member in members
    ])
    return logsumexp(log_densities, axis=0) - np.log(len(members))


def ensemble_sample_x(members, n, batch_size=65_536):
    """Choose a flow uniformly for each event and preserve the event order."""
    if len(members) == 1:
        return flow_sample_x(members[0], n, batch_size=batch_size)
    choices = torch.randint(len(members), (n,)).numpy()
    values = np.empty((n, len(members[0]["features"])), dtype=np.float32)
    for index, member in enumerate(members):
        positions = np.flatnonzero(choices == index)
        values[positions] = flow_sample_x(member, len(positions), batch_size=batch_size)
    return values


def mixture_log_weights(log_g_over_q, epsilon):
    """Log q/g_epsilon, with both q and g conditioned on PRESEL."""
    log_g_over_q = np.asarray(log_g_over_q, dtype=np.float64)
    if epsilon == 0.0:
        return -log_g_over_q
    if epsilon == 1.0:
        return np.zeros_like(log_g_over_q)
    return -np.logaddexp(
        np.log(epsilon), np.log1p(-epsilon) + log_g_over_q
    )


def mix_quadratures(reference, proposal, uniforms, epsilon):
    """Use common q/g samples and mixture uniforms for all epsilon choices."""
    use_reference = uniforms < epsilon
    signal = np.where(use_reference, reference["signal"], proposal["signal"])
    background = np.where(
        use_reference, reference["background"], proposal["background"]
    )
    log_ratio = np.where(
        use_reference, reference["log_g_over_q"], proposal["log_g_over_q"]
    )
    return signal, background, mixture_log_weights(log_ratio, epsilon)
