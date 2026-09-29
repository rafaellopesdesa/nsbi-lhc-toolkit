"""Monte Carlo precision of the scalar simulator population fit.

The signal and background arrays must be independent, unweighted IID banks.
The learned likelihood and its normalizers are held fixed. These expressions
do not apply to the reference banks, whose two processes share weighted draws.
"""

import numpy as np


def score_mc_moments(nu, mu, q_signal, q_background, lam_signal, lam_background):
    """Score, curvature and integration variance at a fixed reference parameter."""
    s = np.asarray(q_signal, dtype=float)
    b = np.asarray(q_background, dtype=float)
    ws = s / (1 + nu * s)
    wb = b / (1 + nu * b)
    signal_yield = mu * lam_signal
    signal_variance = signal_yield**2 * ws.var(ddof=1) / len(ws)
    background_variance = lam_background**2 * wb.var(ddof=1) / len(wb)
    return dict(
        score=float(-lam_signal + signal_yield * ws.mean() + lam_background * wb.mean()),
        information=float(signal_yield * np.mean(ws**2) + lam_background * np.mean(wb**2)),
        signal_score_variance=float(signal_variance),
        background_score_variance=float(background_variance),
        score_se=float(np.sqrt(signal_variance + background_variance)),
    )


def population_root_mc(nu, mu, q_signal, q_background, lam_signal, lam_background):
    """Evaluate MC precision at an already fitted root; do not refit the model.

    At an interior root, ``root_se`` uses the delta method. At the constrained
    boundary it is NaN: ``local_root_scale`` only translates score uncertainty
    into parameter units and is not a symmetric error on the boundary fit.
    """
    at_root = score_mc_moments(nu, mu, q_signal, q_background, lam_signal, lam_background)
    at_zero = score_mc_moments(0., mu, q_signal, q_background, lam_signal, lam_background)
    local_scale = at_root['score_se'] / at_root['information']
    total_variance = at_root['signal_score_variance'] + at_root['background_score_variance']
    signal_fraction = (at_root['signal_score_variance'] / total_variance
                       if total_variance > 0 else np.nan)
    return dict(
        root=float(nu), status='interior' if nu > 0 else 'boundary',
        signal_events=len(q_signal), background_events=len(q_background),
        **at_root,
        local_root_scale=float(local_scale),
        root_se=float(local_scale) if nu > 0 else np.nan,
        signal_variance_fraction=float(signal_fraction),
        background_variance_fraction=float(1 - signal_fraction),
        score_at_zero=at_zero['score'], score_at_zero_se=at_zero['score_se'],
    )


def independent_root_difference(first, second):
    """Pointwise difference for independent banks conditional on a fixed model."""
    interior = first['status'] == second['status'] == 'interior'
    difference = first['root'] - second['root']
    se = np.hypot(first['root_se'], second['root_se']) if interior else np.nan
    return dict(
        root_difference=float(difference),
        difference_se=float(se),
        difference_over_mc_se=float(difference / se) if interior and se > 0 else np.nan,
        status='both interior' if interior else 'boundary involved',
    )
