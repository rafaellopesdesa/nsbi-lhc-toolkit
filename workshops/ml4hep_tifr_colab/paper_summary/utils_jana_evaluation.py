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

    jana.evaluate_nominal_log_posterior = _log_posterior_with_cycle_mask
    jana.evaluate_nominal_log_likelihood = _log_likelihood_with_cycle_mask
    return jana._main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
