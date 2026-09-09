"""NumPy/SciPy inference helpers for Exercise 12 (no training dependencies).

Every fit uses a nonnegative signal strength and a *frozen* extended likelihood.
Finite-reference normalization is part of the model definition, not a correction
applied separately to each observed or simulated toy dataset.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.special import expit, ndtr


INFERENCE_VERSION = "nre-asimov-v2"
COMPRESSION_ALGORITHM = "hybrid_quantile_logq_25_75_v1"


def _ratios(values, name="ratios"):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not len(values):
        raise ValueError(f"{name} must be a nonempty (n_events, 2) array [rS, rB].")
    if not np.all(np.isfinite(values)) or np.any(values[:, 0] < 0) or np.any(values[:, 1] <= 0):
        raise ValueError(f"{name} must be finite, with rS >= 0 and rB > 0; values are never clipped.")
    return values


def _yields(values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (2,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("yields must contain finite positive [S, B].")
    return values


def _normalizers(values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (2,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("normalizers must contain finite positive [Z_S, Z_B].")
    return values


def _nonnegative(value, name):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return value


def _integer(value, name, minimum=0):
    result = int(value)
    if result != value or result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return result


def _array_hash(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def ratio_to_q(ratios, yields, normalizers=(1.0, 1.0)):
    """The per-event sufficient statistic q = S*rS/(B*rB). No clipping."""
    ratios = _ratios(ratios)
    signal, background = _yields(yields)
    z_signal, z_background = _normalizers(normalizers)
    q = (ratios[:, 0] / ratios[:, 1]) * ((signal / background) * (z_background / z_signal))
    if not np.all(np.isfinite(q)):
        raise ValueError("The likelihood ratio overflowed; inspect the ratio model (no clipping was applied).")
    return q


def _q_weights(q, weights):
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 1 or not np.all(np.isfinite(q)) or np.any(q < 0):
        raise ValueError("q must be a finite nonnegative one-dimensional array.")
    if weights is None:
        weights = np.ones_like(q)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim not in (1, 2) or weights.shape[-1] != len(q):
        raise ValueError("weights must have shape (n_events,) or (n_datasets, n_events).")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("weights must be finite and nonnegative.")
    return q, weights


def _score_factors(q, mu):
    # q/(1 + mu*q), avoiding overflow in the product for large positive mu*q.
    inverse = np.full_like(q, np.inf)
    np.divide(1.0, q, out=inverse, where=q > 0)
    return 1.0 / (mu[..., None] + inverse)


def _log_factors(q, mu):
    # logaddexp is the overflow-safe equivalent of log1p(mu*q).
    log_q = np.full_like(q, -np.inf)
    np.log(q, out=log_q, where=q > 0)
    log_mu = np.full_like(mu, -np.inf, dtype=np.float64)
    np.log(mu, out=log_mu, where=mu > 0)
    return np.logaddexp(0.0, log_mu[..., None] + log_q)


def log_likelihood_relative(q, weights, mu, signal_yield):
    """ell(mu)-ell(0) = -mu*S + sum(w*log(1+mu*q))."""
    q, weights = _q_weights(q, weights)
    signal_yield = float(signal_yield)
    if not np.isfinite(signal_yield) or signal_yield <= 0:
        raise ValueError("signal_yield must be positive and finite.")
    mu = np.asarray(mu, dtype=np.float64)
    if not np.all(np.isfinite(mu)) or np.any(mu < 0):
        raise ValueError("mu must be finite and nonnegative.")
    return -mu * signal_yield + np.sum(weights * _log_factors(q, mu), axis=-1)


def fit_weighted_q(q, weights=None, signal_yield=1.0, *, iterations=64):
    """Fit one or a batch of weighted datasets by monotone score bisection.

    There is no arbitrary upper fit bound. At an interior optimum mu*S <=
    sum(weights), which supplies a valid data-dependent bracketing bound.
    Empty datasets and zero-weight datasets correctly give mu_hat=q0=0.
    """
    q, weights = _q_weights(q, weights)
    signal_yield = float(signal_yield)
    if not np.isfinite(signal_yield) or signal_yield <= 0:
        raise ValueError("signal_yield must be finite and positive.")
    iterations = _integer(iterations, "iterations", 1)
    scalar = weights.ndim == 1
    matrix = weights[None, :] if scalar else weights
    score_zero = np.sum(matrix * q, axis=1) - signal_yield
    if not np.all(np.isfinite(score_zero)):
        raise ValueError("The score overflowed; inspect event ratios and weights.")
    interior = score_zero > 0
    low = np.zeros(len(matrix), dtype=np.float64)
    high = np.sum(matrix, axis=1) / signal_yield
    if not np.all(np.isfinite(high)):
        raise ValueError("The total dataset weight is not representable.")
    # A finite bound follows from concavity; no guessed fit range can truncate mu_hat.
    high = np.where(interior, high, 0.0)
    for _ in range(iterations):
        mid = low + 0.5 * (high - low)
        score = np.sum(matrix * _score_factors(q, mid), axis=1) - signal_yield
        above = (score > 0) & interior
        low = np.where(above, mid, low)
        high = np.where(above, high, mid)
    mu_hat = low + 0.5 * (high - low)
    ell_hat = -mu_hat * signal_yield + np.sum(matrix * _log_factors(q, mu_hat), axis=1)
    # Multiply by sqrt(weight) first so zero-weight extreme-ratio events do
    # not create the undefined intermediate 0 * inf when q**2 overflows.
    information = np.sum((np.sqrt(matrix) * _score_factors(q, mu_hat)) ** 2, axis=1)
    sigma = np.full_like(information, np.inf)
    np.divide(1.0, np.sqrt(information), out=sigma, where=information > 0)
    result = {"mu_hat": mu_hat, "q0": np.maximum(2.0 * ell_hat, 0.0),
              "log_likelihood_at_mle_relative_zero": ell_hat,
              "score_at_zero": score_zero, "information_at_mle": information,
              "sigma_curvature": sigma}
    return {key: float(value[0]) for key, value in result.items()} if scalar else result


def _asimov_summary(q, weights, yields, mu_true, scan_mu, normalizers):
    signal, background = _yields(yields)
    mu_true = _nonnegative(mu_true, "mu_true")
    scan_mu = np.asarray(scan_mu, dtype=np.float64)
    if scan_mu.ndim != 1 or not np.all(np.isfinite(scan_mu)) or np.any(scan_mu < 0):
        raise ValueError("scan_mu must be a finite nonnegative one-dimensional array.")
    fit = fit_weighted_q(q, weights, signal)
    truth_ll = float(log_likelihood_relative(q, weights, mu_true, signal))
    # Iterate over scan points to avoid a scan_size * reference_size allocation.
    scan_ll = np.array([log_likelihood_relative(q, weights, mu, signal) for mu in scan_mu])
    score_at_truth = float(np.sum(weights * _score_factors(q, np.array(mu_true))) - signal)
    information_at_truth = float(np.sum((np.sqrt(weights) * _score_factors(q, np.array(mu_true))) ** 2))
    q0 = fit["q0"]
    result = dict(fit)
    result.update({
        "mu_true": mu_true, "yields": [float(signal), float(background)],
        "normalizers": np.asarray(normalizers, dtype=np.float64).tolist(),
        "score_at_truth": score_at_truth,
        "t_at_truth": max(0.0, 2.0 * (fit["log_likelihood_at_mle_relative_zero"] - truth_ll)),
        "scan_mu": scan_mu, "t_scan": np.maximum(2.0 * (fit["log_likelihood_at_mle_relative_zero"] - scan_ll), 0.0),
        "q0_asimov": q0, "information_at_truth": information_at_truth,
        "sigma_at_truth": 1.0 / np.sqrt(information_at_truth) if information_at_truth > 0 else np.inf,
        # The truth-centred expression is meaningful only under closure/Wald assumptions.
        "sigma_q0": mu_true / np.sqrt(q0) if q0 > 0 and mu_true > 0 else np.nan,
        "sigma_q0_fitted_center": fit["mu_hat"] / np.sqrt(q0) if q0 > 0 else np.nan,
        "total_weight": float(np.sum(weights)),
        "n_integration_events": len(q),
    })
    return result


def finite_reference_asimov(ratios, yields, mu_true=1.0, scan_mu=None):
    """Exactly closing finite-reference Asimov construction.

    Z_s = mean_REF(r_s), rbar_s = r_s/Z_s and
    w_i = [mu_true*S*rbar_S(i)+B*rbar_B(i)]/M. Both the Asimov weights
    and the fitted likelihood use these same, parameter-independent Z_s.
    """
    ratios = _ratios(ratios)
    yields = _yields(yields)
    mu_true = _nonnegative(mu_true, "mu_true")
    normalizers = _normalizers(np.mean(ratios, axis=0))
    normalized = ratios / normalizers
    weights = (mu_true * yields[0] * normalized[:, 0] + yields[1] * normalized[:, 1]) / len(ratios)
    q = ratio_to_q(ratios, yields, normalizers)
    if scan_mu is None:
        scan_mu = np.linspace(0.0, max(2.0, 2.0 * mu_true), 101)
    result = _asimov_summary(q, weights, yields, mu_true, scan_mu, normalizers)
    result.update({"construction": "finite_reference", "n_reference": len(ratios),
                   "reference_sha256": _array_hash(ratios)})
    tolerance = 2e-10 * max(1.0, yields[0])
    if abs(result["score_at_truth"]) > tolerance or not np.isclose(result["mu_hat"], mu_true, atol=1e-9, rtol=1e-9):
        raise ArithmeticError("Finite-reference score closure failed; do not use this result.")
    return result


def raw_simulator_asimov(signal_ratios, background_ratios, yields, mu_true=1.0,
                         scan_mu=None, normalizers=(1.0, 1.0)):
    """Usual weighted S/B simulator bank fitted with one frozen NRE model.

    This estimator converges to the simulator-expected likelihood, whose maximum
    can be pseudo-true rather than mu_true if the learned ratios are imperfect.
    Finite-integration fluctuations must not be called guaranteed finite-M bias.
    """
    signal_ratios = _ratios(signal_ratios, "signal_ratios")
    background_ratios = _ratios(background_ratios, "background_ratios")
    yields = _yields(yields)
    mu_true = _nonnegative(mu_true, "mu_true")
    normalizers = _normalizers(normalizers)
    q = np.concatenate((ratio_to_q(signal_ratios, yields, normalizers),
                        ratio_to_q(background_ratios, yields, normalizers)))
    weights = np.concatenate((np.full(len(signal_ratios), mu_true * yields[0] / len(signal_ratios)),
                              np.full(len(background_ratios), yields[1] / len(background_ratios))))
    if scan_mu is None:
        scan_mu = np.linspace(0.0, max(2.0, 2.0 * mu_true), 101)
    result = _asimov_summary(q, weights, yields, mu_true, scan_mu, normalizers)
    result.update({"construction": "raw_simulator", "n_signal": len(signal_ratios),
                   "n_background": len(background_ratios)})
    return result


def simulator_score_diagnostic(signal_ratios, background_ratios, yields, mu_true=1.0,
                               normalizers=(1.0, 1.0)):
    """Expected score and integration SE from independent, fixed-size S/B banks.

    g(x)=q/(1+mu_true*q), U=-S+mu_true*S*mean_S(g)+B*mean_B(g).
    Var_MC(U)=(mu_true*S)^2*var_S(g)/nS+B^2*var_B(g)/nB, using
    unbiased sample variances. This is NOT the Poisson-toy score variance.
    Errors are conditional on the frozen NN and deployment normalizers; they
    do not include training or deployment-reference uncertainty. U/I estimates
    the pseudo-true displacement only to first order near an interior optimum.
    Its reported SE treats I as fixed (valid to leading order near score
    closure), omitting curvature variance/covariance away from closure.
    """
    signal, background = _yields(yields)
    mu_true = _nonnegative(mu_true, "mu_true")
    normalizers = _normalizers(normalizers)
    banks = [_ratios(signal_ratios), _ratios(background_ratios)]
    if any(len(bank) < 2 for bank in banks):
        raise ValueError("Score MC errors need at least two events in each independent bank.")
    score, variance, information = -signal, 0.0, 0.0
    for bank, weight in zip(banks, (mu_true * signal, background)):
        q = ratio_to_q(bank, yields, normalizers)
        g = _score_factors(q, np.array(mu_true))
        score += weight * np.mean(g)
        variance += weight**2 * np.var(g, ddof=1) / len(g)
        information += weight * np.mean(g**2)
    if not np.isfinite([score, variance, information]).all() or information <= 0:
        raise ValueError("The score diagnostic needs finite moments and positive information.")
    standard_error = float(np.sqrt(variance))
    return {"n_signal": len(banks[0]), "n_background": len(banks[1]),
            "mu_true": mu_true, "yields": [float(signal), float(background)],
            "normalizers": normalizers.tolist(), "score_at_truth": float(score),
            "score_mc_se": standard_error, "information_at_truth": float(information),
            "linearized_shift": float(score / information),
            "linearized_shift_mc_se": float(standard_error / information)}


def summarize_score_diagnostics(diagnostics):
    """Equal-weight mean of independent-bank scores for one frozen likelihood.

    Report both propagated within-bank integration errors and the empirical
    between-bank standard error. The latter has only R-1 degrees of freedom.
    Independence must come from the caller's simulation streams, not labels.
    """
    if len(diagnostics) < 2:
        raise ValueError("Use at least two independent banks to measure between-bank scatter.")
    for item in diagnostics[1:]:
        for key in ("mu_true", "yields", "normalizers"):
            if not np.array_equal(item[key], diagnostics[0][key]):
                raise ValueError("All score diagnostics must use the same frozen likelihood and truth.")
    scores = np.array([item["score_at_truth"] for item in diagnostics], dtype=float)
    errors = np.array([item["score_mc_se"] for item in diagnostics], dtype=float)
    information = np.array([item["information_at_truth"] for item in diagnostics], dtype=float)
    if not np.isfinite([scores, errors, information]).all() or np.any(errors < 0) or np.any(information <= 0):
        raise ValueError("Invalid score diagnostics.")
    count = len(scores)
    mean_score, mean_information = float(scores.mean()), float(information.mean())
    mc_se = float(np.sqrt(np.sum(errors**2)) / count)
    between_se = float(np.std(scores, ddof=1) / np.sqrt(count))
    return {"n_banks": count, "score_mean": mean_score, "score_mc_se": mc_se,
            "score_between_bank_se": between_se, "information_mean": mean_information,
            "linearized_shift": mean_score / mean_information,
            "linearized_shift_mc_se": mc_se / mean_information,
            "linearized_shift_between_bank_se": between_se / mean_information}


def _log_q(q):
    """A monotone binning coordinate retaining q=0 and arbitrarily large q."""
    q = np.asarray(q, dtype=np.float64)
    result = np.full_like(q, -np.inf)
    np.log(q, out=result, where=q > 0)
    return result


def _bin_indices(q, edges):
    # The -inf/+inf outer bins include every event, including q=0. Binning
    # directly in log(q), rather than q/(1+q), avoids rounding large q to 1.
    return np.searchsorted(edges[1:-1], _log_q(q), side="right")


def _compression_hash(compression):
    return _array_hash(compression["bin_edges_log_q"], compression["p_signal"],
                       compression["p_background"], compression["q"],
                       compression["normalizers"], compression["yields"])


def build_compression(reference_ratios, yields, n_bins=1024, normalizers=None):
    """Construct an exact binned model of a finite, normalized REF likelihood.

    A hybrid grid uses 25% quantile and 75% uniform-log(q) resolution, retaining
    bulk resolution and resolving the discovery-sensitive tails. Bin intensities
    are integrated component weights, NOT averages of q. Sampling
    and fitting this model is exact; the reduction from event-level q must be
    checked with ``validate_compression`` before interpreting large toy studies.
    """
    ratios = _ratios(reference_ratios, "reference_ratios")
    yields = _yields(yields)
    n_bins = _integer(n_bins, "n_bins", 1)
    actual_normalizers = _normalizers(np.mean(ratios, axis=0))
    if normalizers is None:
        normalizers = actual_normalizers
    normalizers = _normalizers(normalizers)
    if not np.allclose(normalizers, actual_normalizers, atol=1e-12, rtol=1e-10):
        raise ValueError("Compression must use the same-reference normalizers, not a different REF bank.")
    q_reference = ratio_to_q(ratios, yields, normalizers)
    log_q = _log_q(q_reference)
    # The union has <=n_bins bins before any empty/tied bins are merged.
    # Quantile midpoints retain bulk resolution, but quantiles alone can put
    # a huge q range in one rare tail bin and bias the discovery statistic.
    quantile_bins = max(1, min(n_bins, len(log_q)) // 4)
    log_bins = n_bins - quantile_bins + 1
    sorted_log_q = np.sort(log_q)
    cuts = np.unique(np.linspace(0, len(log_q), quantile_bins + 1, dtype=int)[1:-1])
    quantile_interior = []
    for cut in cuts:
        left, right = sorted_log_q[cut - 1], sorted_log_q[cut]
        if left < right:
            # An edge at the smallest positive q separates any exact q=0 atom.
            midpoint = left + 0.5 * (right - left) if np.isfinite(left) else right
            # If no floating-point midpoint exists, the right boundary is valid.
            quantile_interior.append(midpoint if midpoint > left else right)
    finite_log_q = log_q[np.isfinite(log_q)]
    log_interior = np.linspace(np.min(finite_log_q), np.max(finite_log_q), log_bins + 1)[1:-1]
    edges = np.unique(np.r_[-np.inf, quantile_interior, log_interior, np.inf])
    indices = _bin_indices(q_reference, edges)
    occupied = np.flatnonzero(np.bincount(indices, minlength=len(edges) - 1))
    # Merge empty regions into an adjacent occupied bin; never drop any support.
    edges = np.r_[-np.inf, edges[occupied[1:]], np.inf]
    indices = _bin_indices(q_reference, edges)
    p_signal = np.bincount(indices, weights=ratios[:, 0] / normalizers[0], minlength=len(edges) - 1) / len(ratios)
    p_background = np.bincount(indices, weights=ratios[:, 1] / normalizers[1], minlength=len(edges) - 1) / len(ratios)
    if np.any(p_background <= 0):
        raise ValueError("A REF bin has zero background support; use fewer bins or inspect tied/extreme ratios.")
    # Remove only floating-point summation drift, not model normalization error.
    p_signal /= np.sum(p_signal)
    p_background /= np.sum(p_background)
    q = yields[0] * p_signal / (yields[1] * p_background)
    result = {"bin_edges_log_q": edges, "bin_edges_u": expit(edges),
              "p_signal": p_signal, "p_background": p_background,
              "q": q, "normalizers": normalizers, "yields": yields,
              "n_bins": len(q), "requested_n_bins": n_bins, "n_reference": len(ratios),
              "reference_sha256": _array_hash(ratios), "inference_version": INFERENCE_VERSION,
              "compression_algorithm": COMPRESSION_ALGORITHM}
    result["model_sha256"] = _compression_hash(result)
    return result


def simulator_bin_probabilities(signal_ratios, background_ratios, compression):
    """Independent simulator probabilities in the frozen deployed model's bins."""
    signal_ratios = _ratios(signal_ratios, "signal_ratios")
    background_ratios = _ratios(background_ratios, "background_ratios")
    p = []
    for ratios in (signal_ratios, background_ratios):
        indices = _bin_indices(ratio_to_q(ratios, compression["yields"], compression["normalizers"]), compression["bin_edges_log_q"])
        p.append(np.bincount(indices, minlength=compression["n_bins"]).astype(np.float64) / len(ratios))
    result = {"p_signal": p[0], "p_background": p[1], "n_signal": len(signal_ratios),
              "n_background": len(background_ratios), "model_sha256": compression["model_sha256"],
              "source_sha256": _array_hash(signal_ratios, background_ratios)}
    result["probabilities_sha256"] = _array_hash(*p)
    return result


def _toy_probabilities(compression, source, probabilities):
    if source not in ("model", "simulator"):
        raise ValueError("source must be 'model' or 'simulator'.")
    if source == "model":
        if probabilities is not None:
            raise ValueError("Model toys cannot use alternate generation probabilities.")
        ps, pb = compression["p_signal"], compression["p_background"]
    else:
        if probabilities is None:
            raise ValueError("Simulator toys require independent simulator bin probabilities.")
        if isinstance(probabilities, dict):
            if probabilities.get("model_sha256", compression["model_sha256"]) != compression["model_sha256"]:
                raise ValueError("Simulator probabilities belong to a different deployed model.")
            ps, pb = probabilities["p_signal"], probabilities["p_background"]
        else:
            ps, pb = probabilities
    ps, pb = np.asarray(ps, dtype=np.float64), np.asarray(pb, dtype=np.float64)
    for p in (ps, pb):
        if p.shape != np.asarray(compression["q"]).shape or not np.all(np.isfinite(p)) or np.any(p < 0) or not np.isclose(np.sum(p), 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError("Toy component probabilities must be normalized, finite, nonnegative, and match the bins.")
    return ps, pb


def run_toys(compression, mu_true=1.0, n_toys=10000, seed=12345, *, source="model",
             probabilities=None, batch_size=128):
    """Independent extended Poisson toys fitted with the SAME frozen likelihood."""
    mu_true = _nonnegative(mu_true, "mu_true")
    n_toys = _integer(n_toys, "n_toys")
    batch_size = _integer(batch_size, "batch_size", 1)
    ps, pb = _toy_probabilities(compression, source, probabilities)
    signal, background = _yields(compression["yields"])
    rates = mu_true * signal * ps + background * pb
    rng = np.random.default_rng(seed)
    fields = {key: np.empty(n_toys) for key in ("mu_hat", "q0", "sigma_curvature")}
    for start in range(0, n_toys, batch_size):
        stop = min(start + batch_size, n_toys)
        counts = rng.poisson(rates, size=(stop - start, len(rates)))
        fits = fit_weighted_q(compression["q"], counts, signal)
        for key in fields:
            fields[key][start:stop] = fits[key]
    fields.update({"mu_true": mu_true, "n_toys": n_toys, "seed": seed, "source": source,
                   "model_sha256": compression["model_sha256"],
                   "probabilities_sha256": _array_hash(ps, pb), "n_bins": len(rates),
                   "inference_version": INFERENCE_VERSION})
    return fields


def binned_asimov(compression, mu_true=1.0, scan_mu=None, *, source="model", probabilities=None):
    """Expected scan of the deployed binned model (or its simulator expectation)."""
    ps, pb = _toy_probabilities(compression, source, probabilities)
    signal, background = _yields(compression["yields"])
    mu_true = _nonnegative(mu_true, "mu_true")
    if scan_mu is None:
        scan_mu = np.linspace(0.0, max(2.0, 2.0 * mu_true), 101)
    weights = mu_true * signal * ps + background * pb
    result = _asimov_summary(compression["q"], weights, compression["yields"], mu_true,
                             scan_mu, compression["normalizers"])
    result.update({"construction": f"binned_{source}", "model_sha256": compression["model_sha256"]})
    return result


def validate_compression(compression, reference_ratios, mu_true=1.0, n_toys=16, seed=81234,
                         *, signal_ratios=None, background_ratios=None):
    """Paired event-level vs binned fits on identical small validation toys.

    Model toys sample the weighted discrete REF measure; simulator toys sample
    independent S/B banks. This deliberately costs more than compressed toys,
    but its small paired sample tests the numerical approximation, not coverage.
    """
    reference_ratios = _ratios(reference_ratios, "reference_ratios")
    if _array_hash(reference_ratios) != compression["reference_sha256"]:
        raise ValueError("Validation REF differs from the deployed finite-reference model.")
    mu_true = _nonnegative(mu_true, "mu_true")
    n_toys = _integer(n_toys, "n_toys", 1)
    signal, background = _yields(compression["yields"])
    if (signal_ratios is None) != (background_ratios is None):
        raise ValueError("Provide both simulator component banks or neither.")
    if signal_ratios is None:
        banks = (reference_ratios, reference_ratios)
        probabilities = (reference_ratios[:, 0] / np.sum(reference_ratios[:, 0]),
                         reference_ratios[:, 1] / np.sum(reference_ratios[:, 1]))
        source = "model"
    else:
        banks = (_ratios(signal_ratios), _ratios(background_ratios))
        probabilities = (None, None)
        source = "simulator"
    q_banks = [ratio_to_q(bank, compression["yields"], compression["normalizers"]) for bank in banks]
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(n_toys):
        n_signal, n_background = rng.poisson([mu_true * signal, background])
        pieces = [qbank[rng.choice(len(qbank), size=int(n), p=p)]
                  for qbank, n, p in zip(q_banks, (n_signal, n_background), probabilities)]
        q_events = np.concatenate(pieces)
        exact = fit_weighted_q(q_events, signal_yield=signal)
        counts = np.bincount(_bin_indices(q_events, compression["bin_edges_log_q"]), minlength=compression["n_bins"])
        binned = fit_weighted_q(compression["q"], counts, signal)
        records.append((exact["mu_hat"], binned["mu_hat"], exact["q0"], binned["q0"]))
    records = np.asarray(records)
    return {"source": source, "n_toys": n_toys, "mu_hat_event": records[:, 0],
            "mu_hat_binned": records[:, 1], "q0_event": records[:, 2], "q0_binned": records[:, 3],
            "max_abs_delta_mu_hat": float(np.max(np.abs(records[:, 1] - records[:, 0]))),
            "rms_delta_mu_hat": float(np.sqrt(np.mean((records[:, 1] - records[:, 0]) ** 2))),
            "max_abs_delta_q0": float(np.max(np.abs(records[:, 3] - records[:, 2]))),
            "rms_delta_q0": float(np.sqrt(np.mean((records[:, 3] - records[:, 2]) ** 2))),
            "model_sha256": compression["model_sha256"]}


def _probabilities_from_cdf(edges, cdf):
    edges = np.asarray(edges, dtype=np.float64)
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.isnan(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError("Histogram edges must be strictly increasing (infinite outer edges are allowed).")
    upper = cdf(edges[1:])
    lower = cdf(edges[:-1])
    # A point mass at zero belongs to the bin whose left edge is zero, too.
    lower = np.where(edges[:-1] <= 0.0, 0.0, lower)
    probabilities = upper - lower
    # If a bin ends at zero, numpy.histogram puts the atom in the next bin,
    # except when zero is the final right edge.
    if len(probabilities) > 1:
        probabilities[:-1] = np.where(edges[1:-1] == 0.0, 0.0, probabilities[:-1])
    return np.maximum(probabilities, 0.0)


def bounded_muhat_bin_probabilities(edges, mu_center=1.0, sigma=1.0):
    """Bin masses of max(0, Normal(mu_center, sigma)); includes the zero atom."""
    mu_center = float(mu_center)
    sigma = float(sigma)
    if not np.isfinite(mu_center) or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("mu_center must be finite and sigma must be finite and positive.")
    return _probabilities_from_cdf(edges, lambda x: np.where(x >= 0, ndtr((x - mu_center) / sigma), 0.0))


def q0_bin_probabilities(edges, q0_asimov):
    """Wald/Cowan q0 bin masses: F(q)=Phi(sqrt(q)-sqrt(q0_A)), q>=0.

    This approximation assumes the model-based regular/Wald regime. Finite
    normalization guarantees stationarity, not this asymptotic approximation
    or simulator coverage. The atom at zero is included exactly in the bins.
    """
    q0_asimov = _nonnegative(q0_asimov, "q0_asimov")
    return _probabilities_from_cdf(edges, lambda x: np.where(x >= 0, ndtr(np.sqrt(np.maximum(x, 0)) - np.sqrt(q0_asimov)), 0.0))


def _atomic_json(path, document):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run_toys_cached(cache_directory, compression, mu_true=1.0, n_toys=10000, seed=12345,
                    *, source="model", probabilities=None, shard_size=1000, batch_size=128):
    """Resumable, content-addressed toy shards; never modifies training artifacts.

    The cache identity includes model/bin/probability arrays and inference
    version, source, truth, seed and shard size. n_toys is a requested prefix;
    increasing it reuses all earlier full shards without changing their draws.
    Concurrent writers are not supported: use distinct cache directories or
    run this notebook's toy cell in one session at a time.
    """
    n_toys = _integer(n_toys, "n_toys")
    shard_size = _integer(shard_size, "shard_size", 1)
    seed = _integer(seed, "seed")
    mu_true = _nonnegative(mu_true, "mu_true")
    ps, pb = _toy_probabilities(compression, source, probabilities)
    contract = {"inference_version": INFERENCE_VERSION, "model_sha256": _compression_hash(compression),
                "compression_algorithm": compression["compression_algorithm"],
                "reference_sha256": compression.get("reference_sha256"),
                "probabilities_sha256": _array_hash(ps, pb), "source": source,
                "simulator_source_sha256": probabilities.get("source_sha256") if isinstance(probabilities, dict) else None,
                "mu_true": mu_true, "seed": seed, "shard_size": shard_size}
    identity = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    directory = Path(cache_directory) / f"{source}_{identity[:20]}"
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    old = {}
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as stream:
            old = json.load(stream)
        if old.get("contract") != contract:
            raise RuntimeError("Toy cache contract mismatch; existing artifacts were left untouched.")
    arrays = {key: [] for key in ("mu_hat", "q0", "sigma_curvature")}
    reused = 0
    n_shards = (n_toys + shard_size - 1) // shard_size
    for index in range(n_shards):
        path = directory / f"shard_{index:06d}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as archive:
                if str(archive["cache_identity"]) != identity or int(archive["shard_index"]) != index:
                    raise RuntimeError(f"Toy shard identity mismatch: {path}; file was left untouched.")
                result = {key: archive[key].copy() for key in arrays}
            reused += 1
        else:
            shard_seed = int(np.random.SeedSequence([seed, index]).generate_state(1, dtype=np.uint64)[0])
            result = run_toys(compression, mu_true, shard_size, shard_seed, source=source,
                              probabilities=probabilities, batch_size=batch_size)
            descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=directory)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    np.savez_compressed(stream, **{key: result[key] for key in arrays},
                                        cache_identity=np.array(identity), shard_index=np.array(index))
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        for key in arrays:
            value = np.asarray(result[key])
            if value.shape != (shard_size,) or np.any(np.isnan(value)) or np.any(value < 0) or (key != "sigma_curvature" and not np.all(np.isfinite(value))):
                raise RuntimeError(f"Invalid {key} in toy shard {path}; existing files were left untouched.")
            arrays[key].append(value)
        _atomic_json(manifest_path, {"contract": contract, "cache_identity": identity,
                                     "requested_n_toys": n_toys,
                                     "n_completed_shards": max(index + 1, old.get("n_completed_shards", 0))})
    result = {key: np.concatenate(value)[:n_toys] if value else np.empty(0) for key, value in arrays.items()}
    result.update({"mu_true": mu_true, "n_toys": n_toys, "seed": seed, "source": source,
                   "model_sha256": compression["model_sha256"], "probabilities_sha256": _array_hash(ps, pb),
                   "n_bins": compression["n_bins"], "cache_directory": str(directory),
                   "n_reused_shards": reused, "n_shards": n_shards, "inference_version": INFERENCE_VERSION})
    return result
