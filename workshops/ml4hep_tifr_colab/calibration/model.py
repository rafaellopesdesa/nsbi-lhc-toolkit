"""The small, fully unbinned likelihood shared by the calibration notebooks.

Anchor axis order is [down, nominal, up], and process order is [signal,
background].  Each anchor is a normalized process/reference density ratio.
The nuisance changes shape only; the expected process yields stay fixed.
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

FEATURES = [f"x{i}" for i in range(1, 6)]
DEFAULT_CONFIG = {
    "signal_yield": 100.0,
    "background_yield": 10000.0,
    "nu_bounds": [0.0, 3.0],
    "alpha_bounds": [-1.0, 1.0],
    "mu_range": [0.0, 2.0],
    "epsilon": 0.05,
    "seed": 13001,
}


def save_config(config, run):
    """Save this run's explicit settings."""
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def load_config(run):
    return json.loads((Path(run) / "config.json").read_text())


def morph(anchors, alpha):
    """Piecewise-linear interpolation, defined for -1 <= alpha <= 1."""
    anchors = np.asarray(anchors)
    return (
        (1.0 - abs(alpha)) * anchors[..., 1]
        + max(alpha, 0.0) * anchors[..., 2]
        + max(-alpha, 0.0) * anchors[..., 0]
    )


def bad_anchors(anchors, epsilon):
    """Replace signal by (1-epsilon)*signal + epsilon*background."""
    mixed = np.array(anchors, copy=True)
    mixed[..., 0, :] = (
        (1.0 - epsilon) * anchors[..., 0, :]
        + epsilon * anchors[..., 1, :]
    )
    return mixed


def intensity(anchors, nu, alpha, config):
    """Extended-model intensity divided by the common reference density."""
    ratios = morph(anchors, alpha)
    return (
        nu * config["signal_yield"] * ratios[..., 0]
        + config["background_yield"] * ratios[..., 1]
    )


def nll(anchors, nu, alpha, auxiliary, config, weights=None):
    """Negative log likelihood, omitting parameter-independent terms.

    `weights=None` means an ordinary unbinned experiment.  Asimov weights
    represent expected event counts, so only the event sum is weighted.
    The Gaussian auxiliary constraint is included exactly once.
    """
    anchors = np.asarray(anchors, dtype=np.float64)
    event_log = np.log(intensity(anchors, nu, alpha, config))
    event_sum = event_log.sum() if weights is None else np.dot(weights, event_log)
    expected = nu * config["signal_yield"] + config["background_yield"]
    return float(expected - event_sum + 0.5 * (auxiliary - alpha) ** 2)


def asimov_weights(anchors, mu, alpha, config):
    """Reference-sampled quadrature weights for the expected event sum."""
    return intensity(anchors, mu, alpha, config) / len(anchors)


def fit(anchors, auxiliary, config, weights=None, fixed_nu=None):
    """Direct numerical fit, checking both nuisance branches and their join.

    The interpolation has a kink at alpha=0.  Optimize the two smooth
    branches separately, and explicitly include alpha=0 and the endpoints.
    For a global fit, three POI starting points reduce sensitivity to local
    optima.  This is the numerical validation reference, not a fit-label
    generator for training the amortized networks.
    """
    anchors = np.asarray(anchors, dtype=np.float64)
    event_weights = np.ones(len(anchors)) if weights is None else np.asarray(weights)
    signal_yield, background_yield = config["signal_yield"], config["background_yield"]
    nu_low, nu_high = config["nu_bounds"]
    alpha_low, alpha_high = config["alpha_bounds"]
    nu_start = 0.5 * (nu_low + nu_high) if fixed_nu is None else float(fixed_nu)
    # A parameter-independent shift makes the optimizer's stopping tolerance
    # meaningful even when the full event sum is very large.
    offset = nll(anchors, nu_start, 0.0, auxiliary, config, weights)
    candidates = []

    def objective(nu, alpha, slope):
        ratios = morph(anchors, alpha)
        density = nu * signal_yield * ratios[:, 0] + background_yield * ratios[:, 1]
        value = (nu * signal_yield + background_yield
                 - np.dot(event_weights, np.log(density))
                 + 0.5 * (auxiliary - alpha) ** 2 - offset)
        weighted_inverse = event_weights / density
        grad_nu = signal_yield - np.dot(weighted_inverse, signal_yield * ratios[:, 0])
        grad_alpha = alpha - auxiliary - np.dot(
            weighted_inverse, nu * signal_yield * slope[:, 0] + background_yield * slope[:, 1]
        )
        return value, np.array([grad_nu, grad_alpha])

    for bounds, anchor_index, sign in [((alpha_low, 0.0), 0, -1.0), ((0.0, alpha_high), 2, 1.0)]:
        slope = sign * (anchors[:, :, anchor_index] - anchors[:, :, 1])
        if fixed_nu is None:
            def branch_objective(parameters):
                return objective(parameters[0], parameters[1], slope)

            for start in np.linspace(nu_low, nu_high, 3):
                result = minimize(
                    branch_objective, [start, np.mean(bounds)], jac=True,
                    method="L-BFGS-B", bounds=[(nu_low, nu_high), bounds],
                    options={"ftol": 1e-12, "gtol": 1e-7, "maxiter": 200},
                )
                candidates.append((result.fun + offset, *result.x, bool(result.success)))
        else:
            def branch_objective(parameters):
                value, gradient = objective(fixed_nu, parameters[0], slope)
                return value, gradient[1:]

            result = minimize(
                branch_objective, [np.mean(bounds)], jac=True,
                method="L-BFGS-B", bounds=[bounds],
                options={"ftol": 1e-12, "gtol": 1e-7, "maxiter": 200},
            )
            candidates.append((result.fun + offset, fixed_nu, result.x[0], bool(result.success)))

    # Include the nondifferentiable join and both outer nuisance boundaries.
    for alpha in [alpha_low, 0.0, alpha_high]:
        if fixed_nu is None:
            result = minimize_scalar(
                lambda nu: nll(anchors, nu, alpha, auxiliary, config, weights) - offset,
                bounds=(nu_low, nu_high), method="bounded", options={"xatol": 1e-10},
            )
            for nu in [nu_low, result.x, nu_high]:
                candidates.append((nll(anchors, nu, alpha, auxiliary, config, weights), nu, alpha, True))
        else:
            candidates.append((nll(anchors, fixed_nu, alpha, auxiliary, config, weights), fixed_nu, alpha, True))
    value, nu, alpha, success = min(candidates, key=lambda candidate: candidate[0])
    return {"nu": float(nu), "alpha": float(alpha), "nll": float(value), "success": success}
