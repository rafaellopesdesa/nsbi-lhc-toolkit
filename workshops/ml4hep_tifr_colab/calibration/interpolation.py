"""Smooth, normalized shape interpolation shared by NumPy and PyTorch.

The central degree-six polynomial is HistFactory's polynomial/exponential
interpolation. Rare tail configurations can make that polynomial negative.
Only those configurations receive an explicit nonnegative degree-eight bump,
``lambda * alpha**2 * (1-alpha**2)**3``. It preserves the three anchors,
the nominal first derivative, and the value/first/second derivatives at +/-1.
There is no clipping of queried densities. Coefficients and their integrals
can be prepared once, before likelihood evaluation or training.
"""

from math import comb

import numpy as np

N_COEFFICIENTS = 9
_BUMP = np.array([0., 0., 1., 0., -3., 0., 3., 0., -1.])


def _bernstein_matrix(left, right, degree=6):
    """Map ascending power coefficients to Bernstein coefficients."""
    matrix = np.zeros((degree + 1, degree + 1))
    for j in range(degree + 1):
        for k in range(degree + 1):
            matrix[j, k] = sum(
                comb(k, i) * left ** (k - i) * (right - left) ** i
                * comb(j, i) / comb(degree, i)
                for i in range(min(j, k) + 1)
            )
    return matrix


_BERNSTEIN = [_bernstein_matrix(a, b) for a, b in
              [(-1., -.5), (-.5, 0.), (0., .5), (.5, 1.)]]


def _minimum_polynomial(coefficients):
    derivative = np.arange(1, len(coefficients)) * coefficients[1:]
    roots = np.polynomial.polynomial.polyroots(np.trim_zeros(derivative, 'b')) if np.any(derivative) else []
    points = [-1., 0., 1.]
    points.extend(float(root.real) for root in roots
                  if abs(root.imag) < 1e-8 and -1. < root.real < 1.)
    values = np.polynomial.polynomial.polyval(points, coefficients)
    index = int(np.argmin(values))
    return float(values[index]), float(points[index])


def numpy_coefficients(anchors, ensure_positive=True, return_diagnostics=False):
    """Return ascending polynomial coefficients, with final axis length nine.

    ``anchors[..., :]`` is [down, nominal, up]. Input densities/ratios must
    be strictly positive. Positive Bernstein coefficients certify most rows;
    only uncertified rows need stationary-point checks. A positivity repair
    is used only when the paper polynomial has a nonpositive interior minimum.
    The returned coefficients are independent of the queried nuisance value.
    """
    anchors = np.asarray(anchors, dtype=np.float64)
    if anchors.shape[-1] != 3 or not np.all(np.isfinite(anchors)) or np.any(anchors <= 0):
        raise ValueError('Exp-poly requires finite, strictly positive down/nominal/up anchors.')
    down, nominal, up = np.moveaxis(anchors, -1, 0)
    # Multiplying by the nominal before forming coefficients avoids huge
    # intermediate up/nominal ratios in small-density tails.
    log_hi = np.log(up) - np.log(nominal)
    log_lo = np.log(down) - np.log(nominal)
    s0, a0 = (up + down) / 2, (up - down) / 2
    s1, a1 = (up * log_hi - down * log_lo) / 2, (up * log_hi + down * log_lo) / 2
    s2, a2 = (up * log_hi**2 + down * log_lo**2) / 2, (up * log_hi**2 - down * log_lo**2) / 2
    coefficients = np.stack([
        nominal,
        (15*a0 - 7*s1 + a2) / 8,
        (-24*nominal + 24*s0 - 9*a1 + s2) / 8,
        (-5*a0 + 5*s1 - a2) / 4,
        (12*nominal - 12*s0 + 7*a1 - s2) / 4,
        (3*a0 - 3*s1 + a2) / 8,
        (-8*nominal + 8*s0 - 5*a1 + s2) / 8,
        np.zeros_like(nominal), np.zeros_like(nominal),
    ], axis=-1)
    repair = np.zeros_like(nominal, dtype=bool)
    amplitude = np.zeros_like(nominal)
    if ensure_positive:
        flat = coefficients.reshape(-1, N_COEFFICIENTS)
        anchor_flat = anchors.reshape(-1, 3)
        scale = anchor_flat.max(axis=-1)
        scaled = flat[:, :7] / scale[:, None]
        certified = np.ones(len(flat), dtype=bool)
        roundoff_margin = 100 * 64 * np.finfo(float).eps * np.sum(np.abs(scaled), axis=-1)
        for matrix in _BERNSTEIN:
            certified &= np.min(scaled @ matrix.T, axis=-1) > roundoff_margin
        for i in np.flatnonzero(~certified):
            polynomial = flat[i] / scale[i]
            minimum, location = _minimum_polynomial(polynomial)
            if minimum > 0:
                error_bound = 64 * np.finfo(float).eps * np.sum(np.abs(polynomial))
                if error_bound >= .01 * min(minimum, anchor_flat[i].min() / scale[i]):
                    raise FloatingPointError('Exp-poly is ill-conditioned in double precision; inspect extreme anchor-ratio tails.')
                continue
            if abs(location) == 1 or location == 0:
                # Bumps cannot repair invalid anchors. A failure here usually
                # indicates anchor dynamic range beyond double precision.
                raise FloatingPointError('Exp-poly anchor cancellation exceeds double precision; inspect density-ratio tails.')
            bump = location**2 * (1-location**2)**3
            amount = max(1.05 * -minimum / bump, 1e-10)
            margin = 1e-10 * anchor_flat[i].min() / scale[i]
            for _ in range(60):
                candidate = polynomial + amount * _BUMP
                new_minimum, _ = _minimum_polynomial(candidate)
                error_bound = 64 * np.finfo(float).eps * np.sum(np.abs(candidate))
                if error_bound >= .01 * anchor_flat[i].min() / scale[i]:
                    raise FloatingPointError('The positivity repair is ill-conditioned in double precision; inspect extreme anchor-ratio tails.')
                if new_minimum > max(margin, 100 * error_bound):
                    flat[i] = candidate * scale[i]
                    repair.reshape(-1)[i] = True
                    amplitude.reshape(-1)[i] = amount * scale[i]
                    break
                amount *= 2
            else:
                raise FloatingPointError('Could not construct a positive exp-poly extension.')
    if return_diagnostics:
        return coefficients, {'repair_mask': repair, 'repair_amplitude': amplitude}
    return coefficients


def evaluate_polynomial(coefficients, alpha):
    """Evaluate coefficients; alpha broadcasts over all axes except process.

    For event/process coefficients (N,2,9), scalar alpha or an (N,) alpha
    vector works. (1,N,2,9) also supports an (B,1) nuisance array.
    """
    coefficients = np.asarray(coefficients)
    alpha = np.asarray(alpha)[..., None]
    value = np.zeros(np.broadcast_shapes(coefficients.shape[:-1], alpha.shape), dtype=np.result_type(coefficients, alpha))
    for coefficient in np.moveaxis(coefficients, -1, 0)[::-1]:
        value = value * alpha + coefficient
    return value


def derivative_polynomial(coefficients, alpha):
    coefficients = np.asarray(coefficients)
    return evaluate_polynomial(coefficients[..., 1:] * np.arange(1, coefficients.shape[-1]), alpha)


def polynomial_normalization(anchors, coefficients=None):
    """Same-reference-sample integral coefficients, shape (process,9)."""
    coefficients = numpy_coefficients(anchors) if coefficients is None else np.asarray(coefficients)
    return coefficients.mean(axis=0)


def numpy_raw_morph(anchors, alpha, coefficients=None):
    """Unnormalized smooth morph; exponential extrapolation outside +/-1."""
    anchors = np.asarray(anchors, dtype=np.float64)
    coefficients = numpy_coefficients(anchors) if coefficients is None else coefficients
    values = evaluate_polynomial(coefficients, alpha)
    a = np.asarray(alpha)[..., None]
    nominal = anchors[..., 1]
    if np.any(a > 1):
        log_hi = np.log(anchors[..., 2]) - np.log(nominal)
        values = np.where(a > 1, nominal * np.exp(np.clip(a, 1, None) * log_hi), values)
    if np.any(a < -1):
        log_lo = np.log(anchors[..., 0]) - np.log(nominal)
        values = np.where(a < -1, nominal * np.exp(np.clip(-a, 1, None) * log_lo), values)
    # Preserve exact anchors despite Horner cancellation in very small tails.
    if np.any(a == -1):
        values = np.where(a == -1, anchors[..., 0], values)
    if np.any(a == 0):
        values = np.where(a == 0, nominal, values)
    if np.any(a == 1):
        values = np.where(a == 1, anchors[..., 2], values)
    return values


def numpy_raw_derivative(anchors, alpha, coefficients=None):
    anchors = np.asarray(anchors, dtype=np.float64)
    coefficients = numpy_coefficients(anchors) if coefficients is None else coefficients
    values = derivative_polynomial(coefficients, alpha)
    a = np.asarray(alpha)[..., None]
    if np.any(np.abs(a) >= 1):
        raw = numpy_raw_morph(anchors, alpha, coefficients)
        log_hi = np.log(anchors[..., 2]) - np.log(anchors[..., 1])
        log_lo = np.log(anchors[..., 0]) - np.log(anchors[..., 1])
        values = np.where(a >= 1, raw * log_hi, values)
        values = np.where(a <= -1, -raw * log_lo, values)
    return values


def normalization_coefficients(config):
    if 'morph_normalization' not in config:
        raise ValueError('Missing exp-poly normalization: run notebook 02 for this TAG before inference.')
    coefficients = np.asarray(config['morph_normalization'], dtype=np.float64)
    if coefficients.shape != (2, N_COEFFICIENTS) or not np.all(np.isfinite(coefficients)):
        raise ValueError('morph_normalization must contain two finite degree-eight coefficient vectors.')
    return coefficients


def tensor_coefficients(anchors, repair_amplitude=None):
    """Build coefficients on device from anchors and pre-certified bump sizes.

    Storing only the two bump amplitudes per event avoids retaining eighteen
    float64 polynomial coefficients throughout a large toy-training cache.
    """
    import torch
    if repair_amplitude is None:
        return torch.as_tensor(numpy_coefficients(anchors.detach().cpu().numpy()),
                               dtype=torch.float64, device=anchors.device)
    anchors = anchors.to(torch.float64)
    down, nominal, up = anchors.unbind(dim=-1)
    log_hi, log_lo = torch.log(up) - torch.log(nominal), torch.log(down) - torch.log(nominal)
    s0, a0 = (up + down) / 2, (up - down) / 2
    s1, a1 = (up*log_hi-down*log_lo)/2, (up*log_hi+down*log_lo)/2
    s2, a2 = (up*log_hi.square()+down*log_lo.square())/2, (up*log_hi.square()-down*log_lo.square())/2
    bump = torch.as_tensor(repair_amplitude, dtype=torch.float64, device=anchors.device)
    return torch.stack((nominal, (15*a0-7*s1+a2)/8,
                        (-24*nominal+24*s0-9*a1+s2)/8+bump,
                        (-5*a0+5*s1-a2)/4,
                        (12*nominal-12*s0+7*a1-s2)/4-3*bump,
                        (3*a0-3*s1+a2)/8,
                        (-8*nominal+8*s0-5*a1+s2)/8+3*bump,
                        torch.zeros_like(nominal), -bump), dim=-1)


def tensor_morph(anchors, alpha, config=None, coefficients=None):
    """Differentiable normalized morph of frozen good anchors.

    Precompute coefficients once with ``numpy_coefficients`` and pass a tensor
    during training. The fallback is convenient for small checks, but performs
    the NumPy positivity certification and must not be used in an inner loop.
    Normalization is differentiated together with the numerator. The optional
    bad-signal mixture is applied after normalizing each good process.
    """
    import torch
    if coefficients is None:
        coefficients = torch.as_tensor(numpy_coefficients(anchors.detach().cpu().numpy()), dtype=torch.float64, device=anchors.device)
    else:
        coefficients = coefficients.to(dtype=torch.float64, device=anchors.device)
    alpha = torch.as_tensor(alpha, device=anchors.device).to(torch.float64)
    a = alpha.unsqueeze(-1)
    value = torch.zeros_like(coefficients[..., 0])
    for k in range(coefficients.shape[-1] - 1, -1, -1):
        value = value * a + coefficients[..., k]
    # Use the exponential jets at the two exact anchors. Unlike replacing
    # values by constants, these expressions retain the correct first and
    # second derivatives while avoiding polynomial cancellation in tails.
    precise_anchors = anchors.to(torch.float64)
    nominal = precise_anchors[..., 1]
    log_hi = torch.log(precise_anchors[..., 2]) - torch.log(nominal)
    log_lo = torch.log(precise_anchors[..., 0]) - torch.log(nominal)
    upper_exponent = torch.where(a >= 1, a-1, torch.zeros_like(a))
    lower_exponent = torch.where(a <= -1, -a-1, torch.zeros_like(a))
    upper = precise_anchors[..., 2] * torch.exp(upper_exponent * log_hi)
    lower = precise_anchors[..., 0] * torch.exp(lower_exponent * log_lo)
    value = torch.where(a >= 1, upper, value)
    value = torch.where(a <= -1, lower, value)
    if config is not None:
        norm = torch.as_tensor(normalization_coefficients(config), dtype=torch.float64, device=anchors.device)
        denominator = torch.zeros_like(norm[..., 0])
        for k in range(norm.shape[-1] - 1, -1, -1):
            denominator = denominator * a + norm[..., k]
        value = value / denominator
        epsilon = float(config.get('model_epsilon', 0.0))
        value = torch.stack(((1-epsilon)*value[..., 0] + epsilon*value[..., 1], value[..., 1]), dim=-1)
    return value
