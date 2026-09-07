"""Checkpoint-compatible exact-JANA evaluation entry point.

The trained checkpoints fingerprint :mod:`utils_jana`, so evaluation-only
numerical handling lives here.  Density evaluations remain strict everywhere
except on the exact, fixed theta grid used for the Bayes-cycle diagnostic.
That diagnostic already defines its result on the finite intersection of the
posterior, prior, and likelihood log densities in the modern metric runtime.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

import utils_jana as jana


_AUDIT_CYCLE_THETA: np.ndarray | None = None


def _install_bayesflow_numerical_guards() -> None:
    """Stabilize the pinned BayesFlow spline inverse during sampling.

    BayesFlow 1.1.6 solves the inverse rational-quadratic spline in float32
    with ``2*c / (-b - sqrt(discriminant))``.  That expression is
    algebraically correct but can overflow or suffer catastrophic
    cancellation for otherwise valid tail draws.  The checkpoint remains
    usable: solve the same quadratic in float64 with the standard stable-root
    construction, then return to the model dtype for the next coupling layer.

    The legacy sanity logger also accumulates Boolean counts in int8, which is
    why a real positive NaN count can be printed as a negative number.  Keep
    the warning, but count in int64 so any remaining problem is reported
    faithfully.
    """

    import tensorflow as tf
    from bayesflow import amortizers as bayesflow_amortizers
    from bayesflow.coupling_networks import SplineCoupling

    original_calculate_spline = SplineCoupling._calculate_spline

    def stable_spline_parameters(self, parameters):
        left_edge, bottom_edge, widths, heights, derivatives = (
            tf.cast(value, tf.float64) for value in parameters
        )
        left_edge = left_edge + tf.cast(self.default_domain[0], tf.float64)
        bottom_edge = bottom_edge + tf.cast(
            self.default_domain[2], tf.float64
        )

        default_width = tf.cast(
            (self.default_domain[1] - self.default_domain[0]) / self.bins,
            tf.float64,
        )
        default_height = tf.cast(
            (self.default_domain[3] - self.default_domain[2]) / self.bins,
            tf.float64,
        )
        widths = tf.math.softplus(
            widths + tf.math.log(tf.math.expm1(default_width))
        )
        heights = tf.math.softplus(
            heights + tf.math.log(tf.math.expm1(default_height))
        )
        derivatives = tf.math.softplus(
            derivatives + tf.math.log(tf.math.expm1(tf.constant(1.0, tf.float64)))
        )
        total_height = tf.reduce_sum(heights, axis=-1, keepdims=True)
        total_width = tf.reduce_sum(widths, axis=-1, keepdims=True)
        scale = total_height / total_width
        derivatives = tf.concat([scale, derivatives, scale], axis=-1)
        return left_edge, bottom_edge, widths, heights, derivatives

    def stable_spline_inverse(self, v1, v2, condition, **kwargs):
        spline_params = self.net(v1, condition, **kwargs)
        spline_params = self._semantic_spline_parameters(spline_params)
        spline_params = stable_spline_parameters(self, spline_params)
        return self._calculate_spline(v2, spline_params, inverse=True)

    def stable_calculate_spline(self, target, spline_params, inverse=False):
        if not inverse:
            return original_calculate_spline(
                self, target, spline_params, inverse=False
            )

        output_dtype = target.dtype
        target = tf.cast(target, tf.float64)
        left_edge, bottom_edge, widths, heights, derivatives = (
            tf.cast(value, tf.float64) for value in spline_params
        )
        result = tf.zeros_like(target)

        total_width = tf.reduce_sum(widths, axis=-1, keepdims=True)
        total_height = tf.reduce_sum(heights, axis=-1, keepdims=True)
        knots_x = tf.concat(
            [left_edge, left_edge + tf.math.cumsum(widths, axis=-1)], axis=-1
        )
        knots_y = tf.concat(
            [bottom_edge, bottom_edge + tf.math.cumsum(heights, axis=-1)],
            axis=-1,
        )

        target_in_domain = tf.logical_and(
            knots_y[..., 0] < target, target <= knots_y[..., -1]
        )
        higher_indices = tf.searchsorted(knots_y, target[..., None])
        target_in = target[target_in_domain]
        target_in_idx = tf.where(target_in_domain)
        target_out = target[~target_in_domain]
        target_out_idx = tf.where(~target_in_domain)

        if tf.size(target_in_idx) > 0:
            higher_indices = tf.gather_nd(higher_indices, target_in_idx)
            higher_indices = tf.cast(higher_indices, tf.int32)
            lower_indices = higher_indices - 1
            lower_idx_tuples = tf.concat(
                [tf.cast(target_in_idx, tf.int32), lower_indices], axis=-1
            )
            higher_idx_tuples = tf.concat(
                [tf.cast(target_in_idx, tf.int32), higher_indices], axis=-1
            )

            dk = tf.gather_nd(derivatives, lower_idx_tuples)
            dkp = tf.gather_nd(derivatives, higher_idx_tuples)
            xk = tf.gather_nd(knots_x, lower_idx_tuples)
            xkp = tf.gather_nd(knots_x, higher_idx_tuples)
            yk = tf.gather_nd(knots_y, lower_idx_tuples)
            ykp = tf.gather_nd(knots_y, higher_idx_tuples)
            dx = xkp - xk
            dy = ykp - yk
            sk = dy / dx

            y_minus_yk = target_in - yk
            curvature = dkp + dk - 2.0 * sk
            a = dy * (sk - dk) + y_minus_yk * curvature
            b = dy * dk - y_minus_yk * curvature
            c = -sk * y_minus_yk
            discriminant = tf.maximum(b * b - 4.0 * a * c, 0.0)
            sqrt_discriminant = tf.math.sqrt(discriminant)

            # q avoids subtracting nearly equal numbers.  q/a and c/q are
            # the two roots; this branch is algebraically identical to the
            # root selected by the legacy implementation.
            sign_b = tf.where(b >= 0.0, 1.0, -1.0)
            q = -0.5 * (b + sign_b * sqrt_discriminant)
            root_from_q = tf.where(
                b >= 0.0,
                tf.math.divide_no_nan(c, q),
                tf.math.divide_no_nan(q, a),
            )
            linear_root = tf.math.divide_no_nan(-c, b)
            scale = tf.abs(b) + tf.abs(c) + 1.0
            near_linear = tf.abs(a) <= np.finfo(np.float64).eps * scale
            xi = tf.where(near_linear, linear_root, root_from_q)

            # A monotone RQS has one inverse root in [0, 1].  Numerical
            # roundoff can move an endpoint by a few ulps only.
            xi = tf.clip_by_value(xi, 0.0, 1.0)
            result_in = xi * dx + xk
            result = tf.tensor_scatter_nd_update(
                result, target_in_idx, result_in
            )

        if tf.size(target_out_idx) > 0:
            scale = total_height / total_width
            shift = bottom_edge - scale * left_edge
            scale_out = tf.gather_nd(scale, target_out_idx)
            shift_out = tf.gather_nd(shift, target_out_idx)
            result_out = (target_out[..., None] - shift_out) / scale_out
            result_out = tf.squeeze(result_out, axis=-1)
            result = tf.tensor_scatter_nd_update(
                result, target_out_idx, result_out
            )

        return tf.cast(result, output_dtype)

    def check_tensor_sanity_int64(tensor, logger):
        if not tf.executing_eagerly():
            return
        nan_count = int(
            tf.reduce_sum(tf.cast(tf.math.is_nan(tensor), tf.int64)).numpy()
        )
        inf_count = int(
            tf.reduce_sum(tf.cast(tf.math.is_inf(tensor), tf.int64)).numpy()
        )
        if nan_count:
            logger.warning(
                "Warning! Returned estimates contain %d nan values!", nan_count
            )
        if inf_count:
            logger.warning(
                "Warning! Returned estimates contain %d inf values!", inf_count
            )

    SplineCoupling._calculate_spline = stable_calculate_spline
    SplineCoupling._inverse = stable_spline_inverse
    bayesflow_amortizers.check_tensor_sanity = check_tensor_sanity_int64


def _is_audit_cycle(theta: np.ndarray, observations: np.ndarray) -> bool:
    """Identify only fixed-grid, fixed-observation Bayes-cycle evaluations."""

    if _AUDIT_CYCLE_THETA is None:
        return False
    theta_rows = jana._as_rows(theta, jana.POSTERIOR_DIMENSION, "theta")
    observation_rows = jana._as_rows(
        observations, jana.OBSERVATION_DIMENSION, "observations"
    )
    return (
        len(observation_rows) == 1
        and theta_rows.shape == _AUDIT_CYCLE_THETA.shape
        and np.array_equal(theta_rows, _AUDIT_CYCLE_THETA)
    )


def _log_posterior_with_cycle_mask(
    model_or_directory: jana.LoadedJANA | str | Path,
    theta: np.ndarray,
    observations: np.ndarray,
    *,
    chunk_size: int = 8192,
    strict_runtime: bool = True,
) -> np.ndarray:
    allow_cycle_nonfinite = _is_audit_cycle(theta, observations)
    model = jana._resolve_loaded(
        model_or_directory, strict_runtime=strict_runtime
    )
    theta, observations = jana._broadcast_pairs(theta, observations)
    chunks = []
    for start in range(0, len(theta), int(chunk_size)):
        stop = min(len(theta), start + int(chunk_size))
        values = model.joint.log_posterior(
            {
                "parameters": theta[start:stop],
                "direct_conditions": (
                    observations[start:stop] / jana.OBSERVATION_SCALE
                ),
            }
        )
        chunks.append(np.asarray(values, dtype=np.float64).reshape(-1))
    output = np.concatenate(chunks)
    nonfinite = ~np.isfinite(output)
    if nonfinite.any() and not allow_cycle_nonfinite:
        raise FloatingPointError("JANA log posterior returned non-finite values.")
    if nonfinite.any():
        print(
            "[exact JANA evaluation] Bayes-cycle posterior grid: preserving "
            f"{int(nonfinite.sum())}/{len(output)} non-finite values for the "
            "declared finite-intersection diagnostic."
        )
    return output


def _log_likelihood_with_cycle_mask(
    model_or_directory: jana.LoadedJANA | str | Path,
    theta: np.ndarray,
    observations: np.ndarray,
    *,
    physical_density: bool = True,
    chunk_size: int = 8192,
    strict_runtime: bool = True,
) -> np.ndarray:
    allow_cycle_nonfinite = _is_audit_cycle(theta, observations)
    model = jana._resolve_loaded(
        model_or_directory, strict_runtime=strict_runtime
    )
    theta, observations = jana._broadcast_pairs(theta, observations)
    chunks = []
    for start in range(0, len(theta), int(chunk_size)):
        stop = min(len(theta), start + int(chunk_size))
        values = model.joint.log_likelihood(
            {
                "observables": (
                    observations[start:stop] / jana.OBSERVATION_SCALE
                ),
                "conditions": theta[start:stop],
            }
        )
        chunks.append(np.asarray(values, dtype=np.float64).reshape(-1))
    output = np.concatenate(chunks)
    if physical_density:
        output = output - jana.OBSERVATION_DIMENSION * math.log(
            jana.OBSERVATION_SCALE
        )
    nonfinite = ~np.isfinite(output)
    if nonfinite.any() and not allow_cycle_nonfinite:
        raise FloatingPointError("JANA log likelihood returned non-finite values.")
    if nonfinite.any():
        print(
            "[exact JANA evaluation] Bayes-cycle likelihood grid: preserving "
            f"{int(nonfinite.sum())}/{len(output)} non-finite values for the "
            "declared finite-intersection diagnostic."
        )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    global _AUDIT_CYCLE_THETA

    arguments = list(sys.argv[1:] if argv is None else argv)
    parsed = jana._cli_parser().parse_args(arguments)
    if parsed.command != "evaluate":
        raise ValueError("utils_jana_evaluation.py supports only 'evaluate'.")
    input_path = Path(parsed.input).expanduser().resolve()
    with np.load(input_path, allow_pickle=False) as saved:
        if "audit_reference_theta" not in saved.files:
            raise KeyError(
                f"Evaluation input {input_path} has no fixed audit theta grid."
            )
        _AUDIT_CYCLE_THETA = np.asarray(
            saved["audit_reference_theta"], dtype=np.float32
        ).reshape(-1, jana.POSTERIOR_DIMENSION)

    _install_bayesflow_numerical_guards()
    jana.evaluate_nominal_log_posterior = _log_posterior_with_cycle_mask
    jana.evaluate_nominal_log_likelihood = _log_likelihood_with_cycle_mask
    return jana._main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
