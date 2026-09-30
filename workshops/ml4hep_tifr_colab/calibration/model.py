"""The normalized, fully unbinned likelihood shared by the notebooks.

Anchor order is [down, nominal, up], process order [signal, background].
Inputs always contain the GOOD process/reference ratios. The configured
bad-signal mixture is applied after normalized nuisance interpolation.
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

try:  # Notebooks import helpers directly; tests may import the package.
    from .interpolation import (
        derivative_polynomial, evaluate_polynomial, normalization_coefficients,
        numpy_coefficients, numpy_raw_derivative, numpy_raw_morph,
    )
except ImportError:
    from interpolation import (
        derivative_polynomial, evaluate_polynomial, normalization_coefficients,
        numpy_coefficients, numpy_raw_derivative, numpy_raw_morph,
    )

FEATURES = [f"x{i}" for i in range(1, 6)]
DEFAULT_CONFIG = {
    "signal_yield": 100.0,
    "background_yield": 1000.0,
    "nu_bounds": [0.0, 3.0],
    "alpha_bounds": [-1.0, 1.0],
    "mu_range": [0.0, 2.0],
    "epsilon": 0.05,
    "model_epsilon": 0.0,
    "interpolation_version": "normalized_exp_poly_c2_positive_v1",
    "seed": 13001,
}


def save_config(config, run):
    """Save this run's explicit settings."""
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def load_config(run):
    return json.loads((Path(run) / "config.json").read_text())


def with_model_epsilon(config, epsilon):
    """Return a config for the deliberate normalized signal/background mix."""
    epsilon = float(epsilon)
    if not 0 <= epsilon <= 1:
        raise ValueError('model_epsilon must lie in [0,1].')
    return {**config, 'model_epsilon': epsilon}


def _mix_processes(values, config):
    epsilon = float(config.get('model_epsilon', 0.0))
    if not 0 <= epsilon <= 1:
        raise ValueError('model_epsilon must lie in [0,1].')
    mixed = np.array(values, copy=True)
    mixed[..., 0] = (1-epsilon)*values[..., 0] + epsilon*values[..., 1]
    return mixed


def morph(anchors, alpha, config=None, coefficients=None):
    """Smooth process morph, normalized using the fixed reference bank.

    Supplying config requires its explicit ``morph_normalization`` table.
    ``config=None`` returns the raw interpolation for normalization diagnostics.
    Passing precomputed coefficients avoids repeated positivity certification.
    """
    if config is not None and np.any(np.abs(np.asarray(alpha)) > 1):
        raise ValueError('The cached normalized model is defined on alpha in [-1,1].')
    values = numpy_raw_morph(anchors, alpha, coefficients=coefficients)
    if config is None:
        return values
    denominator = evaluate_polynomial(normalization_coefficients(config), alpha)
    if np.any(denominator <= 0):
        raise FloatingPointError('Nonpositive exp-poly normalization.')
    return _mix_processes(values / denominator, config)


def morph_derivative(anchors, alpha, config=None, coefficients=None):
    """Nuisance derivative including the derivative of normalization."""
    coefficients = numpy_coefficients(anchors) if coefficients is None else coefficients
    derivative = numpy_raw_derivative(anchors, alpha, coefficients)
    if config is None:
        return derivative
    raw = numpy_raw_morph(anchors, alpha, coefficients)
    normalization = normalization_coefficients(config)
    denominator = evaluate_polynomial(normalization, alpha)
    denominator_derivative = derivative_polynomial(normalization, alpha)
    values = derivative / denominator - raw * denominator_derivative / denominator**2
    return _mix_processes(values, config)


def bad_anchors(anchors, epsilon):
    """Anchor-only diagnostic mixture; do not pass it to the likelihood.

    The exp-poly interpolation does not commute with process mixing. For fits
    and toys, retain good anchors and use ``with_model_epsilon`` instead.
    """
    mixed = np.array(anchors, copy=True)
    mixed[..., 0, :] = ((1.0-epsilon)*anchors[..., 0, :] + epsilon*anchors[..., 1, :])
    return mixed


def intensity(anchors, nu, alpha, config, coefficients=None):
    """Extended intensity divided by the common reference density."""
    ratios = morph(anchors, alpha, config, coefficients=coefficients)
    return (nu * config['signal_yield'] * ratios[..., 0]
            + config['background_yield'] * ratios[..., 1])


def nll(anchors, nu, alpha, auxiliary, config, weights=None, coefficients=None):
    """Extended NLL, omitting parameter-independent event-density terms."""
    anchors = np.asarray(anchors, dtype=np.float64)
    density = intensity(anchors, nu, alpha, config, coefficients)
    if np.any(density <= 0) or not np.all(np.isfinite(density)):
        raise FloatingPointError('The normalized likelihood has a nonpositive/nonfinite intensity.')
    event_log = np.log(density)
    event_sum = event_log.sum() if weights is None else np.dot(weights, event_log)
    expected = nu*config['signal_yield'] + config['background_yield']
    return float(expected - event_sum + .5*(auxiliary-alpha)**2)


def asimov_weights(anchors, mu, alpha, config, coefficients=None):
    """Reference-sampled quadrature weights for the expected event sum."""
    return intensity(anchors, mu, alpha, config, coefficients) / len(anchors)


def _projected_gradient(parameters, gradient, bounds):
    gradient = np.asarray(gradient, dtype=float).copy()
    for i, (value, (low, high)) in enumerate(zip(parameters, bounds)):
        tolerance = 1e-8 * max(1., abs(low), abs(high))
        if value <= low+tolerance and gradient[i] > 0:
            gradient[i] = 0.
        if value >= high-tolerance and gradient[i] < 0:
            gradient[i] = 0.
    return gradient


def fit(anchors, auxiliary, config, weights=None, fixed_nu=None, coefficients=None):
    """Numerical validation fit of the smooth, normalized likelihood.

    Multiple starts and explicit faces of the parameter box protect against
    local optima. The optimized objective uses event log-density ratios to a
    fixed baseline; it never subtracts two large full-event NLLs. Success is
    checked using the projected gradient at the returned solution, including
    boundary solutions, rather than trusting an optimizer termination flag.
    Coefficients are certified once per experiment, outside the optimizer.
    """
    anchors = np.asarray(anchors, dtype=np.float64)
    coefficients = numpy_coefficients(anchors) if coefficients is None else coefficients
    event_weights = np.ones(len(anchors)) if weights is None else np.asarray(weights, dtype=np.float64)
    signal_yield, background_yield = config['signal_yield'], config['background_yield']
    nu_bounds, alpha_bounds = tuple(config['nu_bounds']), tuple(config['alpha_bounds'])
    if alpha_bounds[0] < -1 or alpha_bounds[1] > 1:
        raise ValueError('The normalized exp-poly coefficient cache supports [-1,1].')
    if fixed_nu is not None and not nu_bounds[0] <= fixed_nu <= nu_bounds[1]:
        raise ValueError('fixed_nu lies outside nu_bounds.')
    nu_reference = float(np.clip(1., *nu_bounds))
    alpha_reference = float(np.clip(0., *alpha_bounds))
    baseline = intensity(anchors, nu_reference, alpha_reference, config, coefficients)
    offset = nll(anchors, nu_reference, alpha_reference, auxiliary, config, weights, coefficients)
    candidates = []

    def objective(nu, alpha):
        ratios = morph(anchors, alpha, config, coefficients)
        slopes = morph_derivative(anchors, alpha, config, coefficients)
        density = nu*signal_yield*ratios[:, 0] + background_yield*ratios[:, 1]
        if np.any(density <= 0) or not np.all(np.isfinite(density)):
            return np.inf, np.array([np.nan, np.nan])
        value = ((nu-nu_reference)*signal_yield
                 - np.dot(event_weights, np.log(density / baseline))
                 + .5*((auxiliary-alpha)**2-(auxiliary-alpha_reference)**2))
        weighted_inverse = event_weights / density
        gradient = np.array([
            signal_yield - np.dot(weighted_inverse, signal_yield*ratios[:, 0]),
            alpha-auxiliary - np.dot(weighted_inverse, nu*signal_yield*slopes[:, 0]+background_yield*slopes[:, 1]),
        ])
        return float(value), gradient

    options = {'ftol': 1e-14, 'gtol': 1e-7, 'maxiter': 500, 'maxls': 40}

    def optimize(start, free_indices, fixed):
        bounds = [nu_bounds, alpha_bounds]
        def free_objective(parameters):
            full = np.array(fixed, dtype=float)
            full[free_indices] = parameters
            value, gradient = objective(*full)
            return value, gradient[free_indices]
        result = minimize(free_objective, start, jac=True, method='L-BFGS-B',
                          bounds=[bounds[i] for i in free_indices], options=options)
        point = np.array(fixed, dtype=float)
        point[free_indices] = result.x
        value, gradient = objective(*point)
        candidates.append((value, point, gradient, bool(result.success), str(result.message)))

    alpha_starts = np.linspace(*alpha_bounds, 5)
    if fixed_nu is None:
        for nu in np.linspace(*nu_bounds, 3):
            for alpha in alpha_starts[::2]:
                optimize([nu, alpha], [0, 1], [nu, alpha])
        # Optimize every face as well as the two-dimensional interior.
        for nu in nu_bounds:
            for alpha in alpha_starts[::2]:
                optimize([alpha], [1], [nu, alpha])
        for alpha in alpha_bounds:
            optimize([nu_reference], [0], [nu_reference, alpha])
        free = [0, 1]
        fit_bounds = [nu_bounds, alpha_bounds]
    else:
        for alpha in alpha_starts:
            optimize([alpha], [1], [fixed_nu, alpha])
        for alpha in alpha_bounds:
            value, gradient = objective(fixed_nu, alpha)
            candidates.append((value, np.array([fixed_nu, alpha]), gradient, True, 'Explicit nuisance boundary'))
        free = [1]
        fit_bounds = [alpha_bounds]
    value, point, gradient, optimizer_success, message = min(candidates, key=lambda item: item[0])
    projected = _projected_gradient(point[free], gradient[free], fit_bounds)
    gradient_norm = float(np.max(np.abs(projected)))
    success = bool(np.isfinite(value) and gradient_norm <= 1e-4)
    return {'nu': float(point[0]), 'alpha': float(point[1]), 'nll': float(value+offset),
            'centered_nll': float(value), 'success': success,
            'projected_gradient_norm': gradient_norm, 'optimizer_success': optimizer_success,
            'message': message, 'n_starts': len(candidates)}
