"""Analytic detector-smeared likelihood for the Exercise 12 truth curve."""

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from utils_distributions import (
    background_components, signal_components, smearing_parameters,
)
from utils_nre import _fingerprint, cached_ratios
from utils_nre_inference import raw_simulator_asimov


class AnalyticRatios:
    """Selected p_s / p_ref, with the same acceptance convention as Exercise 5."""

    def __init__(self, efficiencies):
        self.efficiencies = np.asarray(efficiencies, dtype=np.float64)
        self.fingerprint = _fingerprint({
            "version": "analytic-reco-ratios-v1",
            "efficiencies": self.efficiencies.tolist(),
        })
        scale, resolution = smearing_parameters()
        self.components = []
        for components in (signal_components(), background_components()):
            self.components.append([
                (np.log(frac), multivariate_normal(
                    mean=scale * mean,
                    cov=np.outer(scale, scale) * cov + np.diag(resolution**2),
                ))
                for frac, mean, cov in components
            ])

    def __call__(self, x):
        log_p = np.column_stack([
            logsumexp([log_frac + density.logpdf(x) for log_frac, density in components], axis=0)
            - np.log(efficiency)
            for components, efficiency in zip(self.components, self.efficiencies)
        ])
        log_reference = np.logaddexp(log_p[:, 0], log_p[:, 1]) - np.log(2.0)
        return np.exp(log_p - log_reference[:, None])


def analytic_truth_asimov(root, selection, inclusive_yields, mu_true, scan_mu,
                          n_per_component=5_000_000, seed=210_926):
    """Conventional Asimov scan using analytic densities and fresh selected MC."""
    yields = np.asarray(selection.yields, dtype=np.float64)
    predictor = AnalyticRatios(yields / np.asarray(inclusive_yields, dtype=np.float64))
    signal, background = [
        cached_ratios(root, selection, predictor, "analytic_truth_asimov", component,
                      n_per_component, seed + index)
        for index, component in enumerate(("signal", "background"))
    ]
    result = raw_simulator_asimov(signal, background, yields, mu_true, scan_mu)
    result["construction"] = "analytic_truth"
    return result
