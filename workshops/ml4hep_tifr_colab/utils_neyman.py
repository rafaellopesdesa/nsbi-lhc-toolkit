"""Utilities for the amortized Neyman-construction tutorial (Exercise 11).

The helpers in this module keep the notebook focused on the statistical
construction.  They provide six pieces of infrastructure:

* the one-dimensional likelihood-ratio compression used in Exercise 5;
* resumable, batch-wise pseudo-experiment generation;
* reconstruction of pooled and split simulator templates from Exercise 5's
  held-out density-evaluation arrays;
* bounded-memory, resumable simulator-template streaming with exact nested
  selected-event checkpoints;
* numerical conditional-density and conditional-PIT primitives; and
* an LF2I-style Bernoulli coverage auditor trained with ordinary BCE.

No helper changes the fitted likelihood.  Simulator toys may use different
signal/background bin probabilities, but they are always fitted with the
frozen hNDE likelihood-ratio values supplied by the notebook.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.integrate import cumulative_trapezoid, trapezoid
from torch.utils.data import DataLoader, TensorDataset

from utils_hnpe import (
    ArrayStandardizer,
    ratio_classifier_ensemble_logit,
    sample_spline_flow,
    spline_flow_log_prob,
)


def build_compressed_q_model(
    q_values: np.ndarray,
    signal_weights: np.ndarray,
    background_weights: np.ndarray,
    *,
    lam_signal: float,
    lam_background: float,
    n_bins: int = 512,
    tail_quantile: float = 1.0e-6,
    probability_floor: float = 1.0e-15,
) -> dict[str, np.ndarray]:
    """Compress the Exercise 5 scalar event ratio into a normalized model."""

    q_values = np.asarray(q_values, dtype=np.float64).reshape(-1)
    signal_weights = np.asarray(signal_weights, dtype=np.float64).reshape(-1)
    background_weights = np.asarray(
        background_weights, dtype=np.float64
    ).reshape(-1)
    if not (
        len(q_values) == len(signal_weights) == len(background_weights)
    ):
        raise ValueError("q_values and both weight arrays must have equal length.")
    if len(q_values) < int(n_bins):
        raise ValueError("The reference sample must exceed the number of bins.")
    if not np.isfinite(q_values).all() or np.any(q_values <= 0.0):
        raise ValueError("q_values must be finite and strictly positive.")
    for name, weights in [
        ("signal_weights", signal_weights),
        ("background_weights", background_weights),
    ]:
        if not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise ValueError(f"{name} must be finite and non-negative.")
        if not float(weights.sum()) > 0.0:
            raise ValueError(f"{name} must have positive total weight.")

    log_q = np.log(q_values)
    lower, upper = np.quantile(
        log_q, [float(tail_quantile), 1.0 - float(tail_quantile)]
    )
    interior = np.linspace(lower, upper, int(n_bins) - 1)
    edges = np.concatenate(([-np.inf], interior, [np.inf]))

    def probability(weights: np.ndarray) -> np.ndarray:
        values = np.histogram(log_q, bins=edges, weights=weights)[0].astype(
            np.float64
        )
        values += float(probability_floor)
        return values / values.sum()

    signal_probability = probability(signal_weights)
    background_probability = probability(background_weights)
    compressed_q = (
        float(lam_signal)
        / float(lam_background)
        * signal_probability
        / background_probability
    )
    return {
        "q": compressed_q,
        "signal_probability": signal_probability,
        "background_probability": background_probability,
        "log_q_edges": edges,
    }


def asimov_test_statistic(
    test_mu: float,
    truth_mu: float,
    q_values: np.ndarray,
    expected_counts: np.ndarray,
    *,
    lam_signal: float,
) -> float:
    """Evaluate the expected compressed or weighted unbinned statistic."""

    test_mu = float(test_mu)
    truth_mu = float(truth_mu)
    q_values = np.asarray(q_values, dtype=np.float64)
    expected_counts = np.asarray(expected_counts, dtype=np.float64)
    if q_values.shape != expected_counts.shape:
        raise ValueError("q_values and expected_counts must have equal shape.")
    value = 2.0 * (
        (test_mu - truth_mu) * float(lam_signal)
        - np.sum(
            expected_counts
            * (
                np.log1p(test_mu * q_values)
                - np.log1p(truth_mu * q_values)
            )
        )
    )
    return max(0.0, float(value))


def _array_fingerprint(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for values in arrays:
        values = np.ascontiguousarray(np.asarray(values))
        digest.update(str(values.dtype).encode("utf-8"))
        digest.update(str(values.shape).encode("utf-8"))
        digest.update(values.view(np.uint8))
    return digest.hexdigest()


def run_cached_toy_ensemble(
    *,
    cache_dir: str | Path,
    n_toys: int,
    batch_size: int,
    seed: int,
    mu_range: tuple[float, float],
    signal_probability: np.ndarray,
    background_probability: np.ndarray,
    lam_signal: float,
    lam_background: float,
    likelihood_q: np.ndarray,
    fit_batch: Callable[
        [np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray, np.ndarray],
    ],
    fixed_mu: float | None = None,
    fit_fingerprint: str = "unspecified_v1",
    fit_runtime_fingerprint: str = "unspecified_runtime_v1",
    generation_mu_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    test_mu_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    generation_mu_fingerprint: str = "identity_v1",
    test_mu_fingerprint: str = "identity_v1",
) -> dict[str, np.ndarray]:
    """Generate and fit Poisson toys in resumable, deterministic shards.

    ``fit_batch`` receives ``(counts, test_mu)`` and returns
    ``(mu_hat, t_mu, fitted_score)``.  ``fit_fingerprint`` must be bumped when
    its numerical definition changes.  The stored ``mu`` is always the
    physical Neyman parameter.  Optional transforms may separately change the
    parameter used to generate a surrogate toy and the likelihood coordinate
    used in the statistic numerator.  Their fingerprints are part of the
    cache provenance. A partial shard set requires an exact fit-runtime
    fingerprint match, while a complete immutable shard set may be loaded on
    another runtime. Count matrices exist only for one batch and are never
    written to disk.
    """

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = cache_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    n_toys = int(n_toys)
    batch_size = int(batch_size)
    if n_toys < 1 or batch_size < 1:
        raise ValueError("n_toys and batch_size must be positive.")
    fit_runtime_fingerprint = str(fit_runtime_fingerprint)
    if not fit_runtime_fingerprint:
        raise ValueError("fit_runtime_fingerprint must be non-empty.")
    signal_probability = np.asarray(signal_probability, dtype=np.float64)
    background_probability = np.asarray(
        background_probability, dtype=np.float64
    )
    likelihood_q = np.asarray(likelihood_q, dtype=np.float64)
    if signal_probability.ndim == 1:
        signal_probability = signal_probability[None, :]
    if background_probability.ndim == 1:
        background_probability = background_probability[None, :]
    if signal_probability.shape != background_probability.shape:
        raise ValueError("Signal/background probabilities must have equal shape.")
    if signal_probability.ndim != 2:
        raise ValueError("Template probabilities must be one- or two-dimensional.")
    if likelihood_q.shape != signal_probability.shape[1:]:
        raise ValueError("likelihood_q must match the template binning.")
    if (
        not np.isfinite(signal_probability).all()
        or not np.isfinite(background_probability).all()
        or np.any(signal_probability < 0.0)
        or np.any(background_probability < 0.0)
    ):
        raise ValueError("Template probabilities must be finite and non-negative.")
    if not np.allclose(signal_probability.sum(axis=1), 1.0):
        raise ValueError("Every signal template must sum to one.")
    if not np.allclose(background_probability.sum(axis=1), 1.0):
        raise ValueError("Every background template must sum to one.")
    if generation_mu_transform is not None and str(
        generation_mu_fingerprint
    ) == "identity_v1":
        raise ValueError(
            "A non-identity generation_mu_transform needs an explicit "
            "generation_mu_fingerprint."
        )
    if test_mu_transform is not None and str(
        test_mu_fingerprint
    ) == "identity_v1":
        raise ValueError(
            "A non-identity test_mu_transform needs an explicit "
            "test_mu_fingerprint."
        )

    manifest = {
        "version": 6,
        "n_toys": n_toys,
        "batch_size": batch_size,
        "seed": int(seed),
        "mu_range": [float(mu_range[0]), float(mu_range[1])],
        "fixed_mu": None if fixed_mu is None else float(fixed_mu),
        "lam_signal": float(lam_signal),
        "lam_background": float(lam_background),
        "template_fingerprint": _array_fingerprint(
            signal_probability, background_probability
        ),
        "likelihood_fingerprint": _array_fingerprint(likelihood_q),
        "fit_fingerprint": str(fit_fingerprint),
        "fit_runtime_fingerprint": fit_runtime_fingerprint,
        "has_generation_mu_transform": generation_mu_transform is not None,
        "has_test_mu_transform": test_mu_transform is not None,
        "generation_mu_fingerprint": str(generation_mu_fingerprint),
        "test_mu_fingerprint": str(test_mu_fingerprint),
    }
    manifest_path = cache_dir / "manifest.json"
    n_batches = int(math.ceil(n_toys / batch_size))
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        stable_existing = dict(existing)
        stable_expected = dict(manifest)
        recorded_runtime = stable_existing.pop(
            "fit_runtime_fingerprint", None
        )
        stable_expected.pop("fit_runtime_fingerprint", None)
        if stable_existing != stable_expected:
            raise RuntimeError(
                f"Toy cache {cache_dir} was made with a different "
                "configuration. Remove that versioned directory or choose "
                "a new RUN_TAG."
            )
        complete = all(
            (shard_dir / f"batch_{index:05d}.npz").exists()
            for index in range(n_batches)
        )
        if recorded_runtime != fit_runtime_fingerprint and not complete:
            raise RuntimeError(
                f"The partial toy cache {cache_dir} was created with a "
                "different JAX/runtime fingerprint. Resume it on a matching "
                "runtime or choose a new versioned cache."
            )
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    for batch_index in range(n_batches):
        start = batch_index * batch_size
        stop = min(n_toys, start + batch_size)
        shard_path = shard_dir / f"batch_{batch_index:05d}.npz"
        if shard_path.exists():
            continue
        rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), int(batch_index)])
        )
        if fixed_mu is None:
            mu = rng.uniform(
                float(mu_range[0]), float(mu_range[1]), size=stop - start
            )
        else:
            mu = np.full(stop - start, float(fixed_mu), dtype=np.float64)
        generation_mu = (
            mu
            if generation_mu_transform is None
            else np.asarray(generation_mu_transform(mu), dtype=np.float64)
        )
        test_mu = (
            mu
            if test_mu_transform is None
            else np.asarray(test_mu_transform(mu), dtype=np.float64)
        )
        generation_mu = np.broadcast_to(
            generation_mu, mu.shape
        ).astype(np.float64, copy=False)
        test_mu = np.broadcast_to(test_mu, mu.shape).astype(
            np.float64, copy=False
        )
        if (
            not np.isfinite(generation_mu).all()
            or np.any(generation_mu < 0.0)
            or not np.isfinite(test_mu).all()
            or np.any(test_mu < 0.0)
        ):
            raise ValueError(
                "Generation and tested likelihood coordinates must be "
                "finite and non-negative."
            )
        template_index = rng.integers(
            0, len(signal_probability), size=stop - start
        )
        batch_signal_probability = signal_probability[template_index]
        batch_background_probability = background_probability[template_index]
        poisson_mean = (
            generation_mu[:, None]
            * float(lam_signal)
            * batch_signal_probability
            + float(lam_background) * batch_background_probability
        )
        counts = rng.poisson(poisson_mean)
        mu_hat, t_mu, fitted_score = fit_batch(counts, test_mu)
        payload = {
            "mu": mu.astype(np.float32),
            "generation_mu": generation_mu.astype(np.float32),
            "test_mu": test_mu.astype(np.float32),
            "template_index": template_index.astype(np.int32),
            "mu_hat": np.asarray(mu_hat, dtype=np.float32),
            "t_mu": np.asarray(t_mu, dtype=np.float32),
            "n_events": counts.sum(axis=1).astype(np.int32),
            "fitted_score": np.asarray(fitted_score, dtype=np.float32),
        }
        payload["payload_fingerprint"] = np.asarray(
            _array_fingerprint(*(payload[name] for name in sorted(payload)))
        )
        temporary_path = shard_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary_path, **payload)
        temporary_path.replace(shard_path)
        if (batch_index + 1) % max(1, n_batches // 10) == 0:
            print(f"  completed {stop:,}/{n_toys:,} toys in {cache_dir.name}")

    chunks: dict[str, list[np.ndarray]] = {
        name: []
        for name in [
            "mu",
            "generation_mu",
            "test_mu",
            "template_index",
            "mu_hat",
            "t_mu",
            "n_events",
            "fitted_score",
        ]
    }
    for batch_index in range(n_batches):
        start = batch_index * batch_size
        stop = min(n_toys, start + batch_size)
        expected_rows = stop - start
        with np.load(
            shard_dir / f"batch_{batch_index:05d}.npz",
            allow_pickle=False,
        ) as shard:
            required = set(chunks) | {"payload_fingerprint"}
            missing = required - set(shard.files)
            if missing:
                raise RuntimeError(
                    f"Toy shard {batch_index} is missing {sorted(missing)}."
                )
            values = {
                name: np.asarray(shard[name])
                for name in chunks
            }
            bad_shapes = {
                name: value.shape
                for name, value in values.items()
                if value.shape != (expected_rows,)
            }
            if bad_shapes:
                raise RuntimeError(
                    f"Toy shard {batch_index} has invalid shapes: {bad_shapes}."
                )
            observed_fingerprint = str(
                np.asarray(shard["payload_fingerprint"]).item()
            )
            expected_fingerprint = _array_fingerprint(
                *(values[name] for name in sorted(values))
            )
            if observed_fingerprint != expected_fingerprint:
                raise RuntimeError(
                    f"Toy shard {batch_index} failed its content fingerprint."
                )
            if (
                not all(np.isfinite(values[name]).all() for name in [
                    "mu", "generation_mu", "test_mu", "mu_hat", "t_mu",
                    "fitted_score",
                ])
                or np.any(values["n_events"] < 0)
                or np.any(values["template_index"] < 0)
                or np.any(values["template_index"] >= len(signal_probability))
            ):
                raise RuntimeError(
                    f"Toy shard {batch_index} contains invalid values."
                )
            for name in chunks:
                chunks[name].append(values[name])
    result = {
        name: np.concatenate(parts)[:n_toys]
        for name, parts in chunks.items()
    }
    if len(result["mu"]) != n_toys:
        raise RuntimeError("Toy cache did not reconstruct the requested row count.")
    return result


def simulator_templates_from_exercise5(
    *,
    weights_path: str | Path,
    ratio_signal_path: str | Path,
    ratio_background_path: str | Path,
    log_q_edges: np.ndarray,
    lam_signal: float,
    lam_background: float,
    seed: int,
    source_ratio_normalization: Mapping[str, float] | None = None,
    target_ratio_normalization: Mapping[str, float] | None = None,
    probability_floor: float = 1.0e-15,
) -> dict[str, Any]:
    """Build pooled and diagnostic split templates from Exercise 5 arrays.

    Exercise 5 concatenates background evaluation rows followed by signal
    evaluation rows.  The generator assigns one constant event weight per
    process, so the unique contiguous weight change recovers that boundary
    without assuming a fixed reservoir size.  The optional normalization
    mappings transport ratios saved under Exercise 5's finite reference draw
    to the normalization used by the frozen Exercise 11 likelihood.
    """

    weights = np.load(weights_path).astype(np.float64, copy=False).reshape(-1)
    ratio_signal = np.load(ratio_signal_path).astype(
        np.float64, copy=False
    ).reshape(-1)
    ratio_background = np.load(ratio_background_path).astype(
        np.float64, copy=False
    ).reshape(-1)
    if not (
        len(weights) == len(ratio_signal) == len(ratio_background)
    ):
        raise ValueError("Exercise 5 density arrays have inconsistent lengths.")
    if len(weights) < 16:
        raise ValueError("Exercise 5 density arrays are unexpectedly small.")
    if (
        not np.isfinite(weights).all()
        or not np.isfinite(ratio_signal).all()
        or not np.isfinite(ratio_background).all()
        or np.any(weights <= 0.0)
        or np.any(ratio_signal <= 0.0)
        or np.any(ratio_background <= 0.0)
    ):
        raise ValueError("Exercise 5 density arrays must be finite and positive.")

    changes = np.flatnonzero(
        ~np.isclose(weights[1:], weights[:-1], rtol=1.0e-12, atol=0.0)
    ) + 1
    if len(changes) != 1:
        raise RuntimeError(
            "Could not infer the background/signal boundary from "
            "weights_asimov.npy. Re-run Exercise 5 with the tutorial "
            "generator and default constant process weights."
        )
    boundary = int(changes[0])
    if not (4 <= boundary <= len(weights) - 4):
        raise RuntimeError("The inferred simulator process boundary is invalid.")
    if not float(np.median(weights[:boundary])) > float(
        np.median(weights[boundary:])
    ):
        raise RuntimeError(
            "Exercise 5 arrays are not ordered as background then signal."
        )

    background_weight_sum = float(weights[:boundary].sum())
    signal_weight_sum = float(weights[boundary:].sum())
    yield_closure = {
        "background": background_weight_sum / float(lam_background) - 1.0,
        "signal": signal_weight_sum / float(lam_signal) - 1.0,
    }
    if max(abs(value) for value in yield_closure.values()) > 5.0e-3:
        raise RuntimeError(
            "The Exercise 5 held-out weights do not close to the reconstructed "
            f"post-selection yields: {yield_closure}. Check the PRESEL state."
        )

    if (source_ratio_normalization is None) != (
        target_ratio_normalization is None
    ):
        raise ValueError(
            "source_ratio_normalization and target_ratio_normalization "
            "must be supplied together."
        )
    log_q_scale_correction = 0.0
    if source_ratio_normalization is not None:
        normalization_values = {}
        for mapping_name, mapping in [
            ("source", source_ratio_normalization),
            ("target", target_ratio_normalization),
        ]:
            for process in ("signal", "background"):
                try:
                    value = float(mapping[process])
                except (KeyError, TypeError) as error:
                    raise ValueError(
                        f"{mapping_name}_ratio_normalization needs positive "
                        "signal/background entries."
                    ) from error
                if not np.isfinite(value) or value <= 0.0:
                    raise ValueError(
                        f"Invalid {mapping_name} normalization for {process}."
                    )
                normalization_values[(mapping_name, process)] = value
        # The saved arrays contain raw_ratio / source_normalization.  Convert
        # them to raw_ratio / target_normalization before forming q_s / q_b.
        log_q_scale_correction = (
            math.log(normalization_values[("source", "signal")])
            - math.log(normalization_values[("target", "signal")])
            - math.log(normalization_values[("source", "background")])
            + math.log(normalization_values[("target", "background")])
        )

    log_q = (
        np.log(float(lam_signal) / float(lam_background))
        + np.log(ratio_signal)
        - np.log(ratio_background)
        + log_q_scale_correction
    )
    edges = np.asarray(log_q_edges, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    slices = {
        "background": np.arange(0, boundary),
        "signal": np.arange(boundary, len(weights)),
    }
    templates: dict[str, Any] = {
        "n_background": boundary,
        "n_signal": len(weights) - boundary,
        "background_weight_sum": background_weight_sum,
        "signal_weight_sum": signal_weight_sum,
        "background_yield_closure": yield_closure["background"],
        "signal_yield_closure": yield_closure["signal"],
        "log_q_scale_correction": log_q_scale_correction,
    }
    for process, indices in slices.items():
        pooled_counts = np.histogram(
            log_q[indices], bins=edges
        )[0].astype(np.int64)
        pooled_probability = np.histogram(
            log_q[indices], bins=edges, weights=weights[indices]
        )[0].astype(np.float64)
        pooled_probability += float(probability_floor)
        pooled_probability /= pooled_probability.sum()
        templates[f"pooled_{process}_counts"] = pooled_counts
        templates[f"pooled_{process}_probability"] = pooled_probability

        order = indices[rng.permutation(len(indices))]
        split = len(order) // 2
        subsets = {
            "calibration": order[:split],
            "audit": order[split:],
        }
        for subset_name, subset_indices in subsets.items():
            counts = np.histogram(
                log_q[subset_indices], bins=edges
            )[0].astype(np.int64)
            probability = np.histogram(
                log_q[subset_indices],
                bins=edges,
                weights=weights[subset_indices],
            )[0].astype(np.float64)
            probability += float(probability_floor)
            probability /= probability.sum()
            templates[f"{subset_name}_{process}_counts"] = counts
            templates[f"{subset_name}_{process}_probability"] = probability
            templates[f"{subset_name}_{process}_events"] = len(subset_indices)
    return templates


_STREAMED_TEMPLATE_CACHE_VERSION = 2


def _streamed_template_manifest(
    *,
    process: str,
    selected_checkpoints: np.ndarray,
    batch_size: int,
    seed: int,
    feature_names: Sequence[str],
    feature_edges: np.ndarray,
    log_q_edges: np.ndarray,
    presel_ratio_cut: float,
    recipe_fingerprint: str,
) -> dict[str, Any]:
    """Return the exact provenance contract for a streamed template cache."""

    return {
        "cache_version": _STREAMED_TEMPLATE_CACHE_VERSION,
        "process": str(process),
        "selected_checkpoints": [
            int(value) for value in np.asarray(selected_checkpoints).reshape(-1)
        ],
        "batch_size": int(batch_size),
        "seed": int(seed),
        "feature_names": [str(name) for name in feature_names],
        "feature_edges_fingerprint": _array_fingerprint(feature_edges),
        "log_q_edges_fingerprint": _array_fingerprint(log_q_edges),
        "presel_ratio_cut": float(presel_ratio_cut),
        "recipe_fingerprint": str(recipe_fingerprint),
    }


def _streamed_template_state_fingerprint(
    manifest_json: str,
    *,
    runtime_fingerprint: str,
    selected_events: int,
    generated_events: int,
    passing_events: int,
    next_batch_index: int,
    n_completed_checkpoints: int,
    current_q_counts: np.ndarray,
    current_feature_counts: np.ndarray,
    checkpoint_q_counts: np.ndarray,
    checkpoint_feature_counts: np.ndarray,
) -> str:
    digest = hashlib.sha256(str(manifest_json).encode("utf-8"))
    digest.update(str(runtime_fingerprint).encode("utf-8"))
    for value in (
        selected_events,
        generated_events,
        passing_events,
        next_batch_index,
        n_completed_checkpoints,
    ):
        digest.update(np.asarray(int(value), dtype=np.int64).tobytes())
    for values in (
        current_q_counts,
        current_feature_counts,
        checkpoint_q_counts,
        checkpoint_feature_counts,
    ):
        values = np.ascontiguousarray(np.asarray(values, dtype=np.int64))
        digest.update(str(values.shape).encode("utf-8"))
        digest.update(values.view(np.uint8))
    return digest.hexdigest()


def _validate_streamed_template_state(
    saved: Mapping[str, Any],
    *,
    expected_manifest_json: str,
    selected_checkpoints: np.ndarray,
    n_features: int,
    n_feature_bins: int,
    n_q_bins: int,
) -> dict[str, Any]:
    """Validate a complete or resumable streamed-template state."""

    required = {
        "manifest_json",
        "runtime_fingerprint",
        "selected_events",
        "generated_events",
        "passing_events",
        "next_batch_index",
        "n_completed_checkpoints",
        "current_q_counts",
        "current_feature_counts",
        "checkpoint_q_counts",
        "checkpoint_feature_counts",
        "state_fingerprint",
    }
    missing = required - set(saved.keys())
    if missing:
        raise RuntimeError(
            "The streamed-template cache is incomplete: "
            f"missing {sorted(missing)}."
        )
    manifest_json = str(np.asarray(saved["manifest_json"]).item())
    if manifest_json != expected_manifest_json:
        raise RuntimeError(
            "The streamed-template cache has different provenance. Choose a "
            "new versioned cache path or remove only this incompatible cache."
        )
    runtime_fingerprint = str(
        np.asarray(saved["runtime_fingerprint"]).item()
    )
    if not runtime_fingerprint:
        raise RuntimeError(
            "The streamed-template cache has an empty runtime fingerprint."
        )

    scalar_names = [
        "selected_events",
        "generated_events",
        "passing_events",
        "next_batch_index",
        "n_completed_checkpoints",
    ]
    scalar_values = {}
    for name in scalar_names:
        values = np.asarray(saved[name])
        if values.shape != () or values.dtype != np.dtype(np.int64):
            raise RuntimeError(
                f"The streamed-template cache field {name!r} is not an "
                "int64 scalar."
            )
        scalar_values[name] = int(values.item())
    selected_events = scalar_values["selected_events"]
    generated_events = scalar_values["generated_events"]
    passing_events = scalar_values["passing_events"]
    next_batch_index = scalar_values["next_batch_index"]
    n_completed = scalar_values["n_completed_checkpoints"]

    def int64_array(name: str) -> np.ndarray:
        values = np.asarray(saved[name])
        if values.dtype != np.dtype(np.int64):
            raise RuntimeError(
                f"The streamed-template cache field {name!r} is not int64."
            )
        return values

    current_q = int64_array("current_q_counts")
    current_feature = int64_array("current_feature_counts")
    checkpoint_q = int64_array("checkpoint_q_counts")
    checkpoint_feature = int64_array("checkpoint_feature_counts")
    checkpoints = np.asarray(selected_checkpoints, dtype=np.int64)

    expected_shapes = {
        "current_q_counts": (n_q_bins,),
        "current_feature_counts": (n_features, n_feature_bins),
        "checkpoint_q_counts": (len(checkpoints), n_q_bins),
        "checkpoint_feature_counts": (
            len(checkpoints), n_features, n_feature_bins
        ),
    }
    observed_shapes = {
        "current_q_counts": current_q.shape,
        "current_feature_counts": current_feature.shape,
        "checkpoint_q_counts": checkpoint_q.shape,
        "checkpoint_feature_counts": checkpoint_feature.shape,
    }
    mismatched_shapes = {
        name: (observed_shapes[name], shape)
        for name, shape in expected_shapes.items()
        if observed_shapes[name] != shape
    }
    if mismatched_shapes:
        raise RuntimeError(
            "The streamed-template cache has invalid array shapes: "
            f"{mismatched_shapes}."
        )
    if (
        selected_events < 0
        or generated_events < 0
        or passing_events < selected_events
        or passing_events > generated_events
        or next_batch_index < 0
        or not 0 <= n_completed <= len(checkpoints)
        or np.any(current_q < 0)
        or np.any(current_feature < 0)
        or np.any(checkpoint_q < 0)
        or np.any(checkpoint_feature < 0)
    ):
        raise RuntimeError("The streamed-template cache has invalid counters.")
    manifest_batch_size = int(json.loads(manifest_json)["batch_size"])
    if generated_events != next_batch_index * manifest_batch_size:
        raise RuntimeError(
            "The streamed-template generated count is inconsistent with its "
            "completed raw batches."
        )
    if int(current_q.sum()) != selected_events:
        raise RuntimeError("The current q histogram lost selected events.")
    if not np.all(current_feature.sum(axis=1) == selected_events):
        raise RuntimeError("A current feature histogram lost selected events.")
    for index in range(n_completed):
        target = int(checkpoints[index])
        if int(checkpoint_q[index].sum()) != target:
            raise RuntimeError("A completed q checkpoint has the wrong size.")
        if not np.all(checkpoint_feature[index].sum(axis=1) == target):
            raise RuntimeError(
                "A completed feature checkpoint has the wrong size."
            )
    if n_completed > 1:
        if np.any(np.diff(checkpoint_q[:n_completed], axis=0) < 0) or np.any(
            np.diff(checkpoint_feature[:n_completed], axis=0) < 0
        ):
            raise RuntimeError(
                "Nested streamed-template checkpoints are not monotone."
            )
    if n_completed and int(checkpoints[n_completed - 1]) == selected_events:
        if not np.array_equal(
            checkpoint_q[n_completed - 1], current_q
        ) or not np.array_equal(
            checkpoint_feature[n_completed - 1], current_feature
        ):
            raise RuntimeError(
                "The latest streamed checkpoint differs from the running state."
            )
    if n_completed < len(checkpoints):
        if np.any(checkpoint_q[n_completed:]) or np.any(
            checkpoint_feature[n_completed:]
        ):
            raise RuntimeError(
                "An incomplete streamed checkpoint contains nonzero counts."
            )
    expected_completed = int(np.searchsorted(
        checkpoints, selected_events, side="right"
    ))
    if expected_completed != n_completed:
        raise RuntimeError(
            "The streamed-template checkpoint counter is inconsistent."
        )

    observed_fingerprint = str(np.asarray(saved["state_fingerprint"]).item())
    expected_fingerprint = _streamed_template_state_fingerprint(
        manifest_json,
        runtime_fingerprint=runtime_fingerprint,
        selected_events=selected_events,
        generated_events=generated_events,
        passing_events=passing_events,
        next_batch_index=next_batch_index,
        n_completed_checkpoints=n_completed,
        current_q_counts=current_q,
        current_feature_counts=current_feature,
        checkpoint_q_counts=checkpoint_q,
        checkpoint_feature_counts=checkpoint_feature,
    )
    if observed_fingerprint != expected_fingerprint:
        raise RuntimeError(
            "The streamed-template cache failed its content fingerprint."
        )
    return {
        "manifest_json": manifest_json,
        "runtime_fingerprint": runtime_fingerprint,
        "selected_events": selected_events,
        "generated_events": generated_events,
        "passing_events": passing_events,
        "next_batch_index": next_batch_index,
        "n_completed_checkpoints": n_completed,
        "current_q_counts": current_q,
        "current_feature_counts": current_feature,
        "checkpoint_q_counts": checkpoint_q,
        "checkpoint_feature_counts": checkpoint_feature,
        "state_fingerprint": observed_fingerprint,
    }


def _save_streamed_template_state(
    cache_path: Path,
    *,
    state: Mapping[str, Any],
) -> None:
    """Atomically save a small resumable state on local or Drive storage."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state)
    payload["state_fingerprint"] = _streamed_template_state_fingerprint(
        str(payload["manifest_json"]),
        runtime_fingerprint=str(payload["runtime_fingerprint"]),
        selected_events=int(payload["selected_events"]),
        generated_events=int(payload["generated_events"]),
        passing_events=int(payload["passing_events"]),
        next_batch_index=int(payload["next_batch_index"]),
        n_completed_checkpoints=int(payload["n_completed_checkpoints"]),
        current_q_counts=payload["current_q_counts"],
        current_feature_counts=payload["current_feature_counts"],
        checkpoint_q_counts=payload["checkpoint_q_counts"],
        checkpoint_feature_counts=payload["checkpoint_feature_counts"],
    )
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, cache_path)


def load_streamed_simulator_template(
    *,
    cache_path: str | Path,
    process: str,
    selected_checkpoints: Sequence[int],
    batch_size: int,
    seed: int,
    feature_names: Sequence[str],
    feature_edges: np.ndarray,
    log_q_edges: np.ndarray,
    presel_ratio_cut: float,
    recipe_fingerprint: str,
    expose_counts: bool = True,
) -> dict[str, Any]:
    """Validate and load a complete streamed simulator template.

    With ``expose_counts=False`` only provenance and scalar counters are
    returned.  Exercise 11 uses that mode to certify its independently seeded
    audit cache without inspecting the audit law before freezing the method.
    """

    cache_path = Path(cache_path)
    checkpoints = np.asarray(selected_checkpoints, dtype=np.int64).reshape(-1)
    edges = np.asarray(feature_edges, dtype=np.float64)
    q_edges = np.asarray(log_q_edges, dtype=np.float64).reshape(-1)
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)
    if edges.ndim != 2 or len(feature_names) != edges.shape[0]:
        raise ValueError("feature_edges must have one row per feature name.")
    manifest = _streamed_template_manifest(
        process=process,
        selected_checkpoints=checkpoints,
        batch_size=batch_size,
        seed=seed,
        feature_names=feature_names,
        feature_edges=edges,
        log_q_edges=q_edges,
        presel_ratio_cut=presel_ratio_cut,
        recipe_fingerprint=recipe_fingerprint,
    )
    manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    with np.load(cache_path, allow_pickle=False) as saved:
        state = _validate_streamed_template_state(
            saved,
            expected_manifest_json=manifest_json,
            selected_checkpoints=checkpoints,
            n_features=len(feature_names),
            n_feature_bins=edges.shape[1] - 1,
            n_q_bins=len(q_edges) - 1,
        )
    if state["selected_events"] != int(checkpoints[-1]):
        raise RuntimeError(
            f"The streamed {process} template is only partially complete: "
            f"{state['selected_events']:,}/{int(checkpoints[-1]):,}."
        )
    result = {
        "cache_path": cache_path,
        "process": str(process),
        "selected_checkpoints": checkpoints,
        "selected_events": state["selected_events"],
        "generated_events": state["generated_events"],
        "passing_events": state["passing_events"],
        "acceptance": state["passing_events"] / state["generated_events"],
        "runtime_fingerprint": state["runtime_fingerprint"],
        "fingerprint": state["state_fingerprint"],
    }
    if expose_counts:
        result.update({
            "q_counts": state["checkpoint_q_counts"],
            "feature_counts": state["checkpoint_feature_counts"],
        })
    return result


def run_streamed_simulator_template(
    *,
    cache_path: str | Path,
    process: str,
    selected_checkpoints: Sequence[int],
    batch_size: int,
    seed: int,
    feature_names: Sequence[str],
    feature_edges: np.ndarray,
    log_q_edges: np.ndarray,
    sample_batch: Callable[[int, np.random.Generator], np.ndarray],
    preselection_ratio: Callable[[np.ndarray], np.ndarray],
    log_q_evaluator: Callable[[np.ndarray], np.ndarray],
    presel_ratio_cut: float,
    recipe_fingerprint: str,
    runtime_fingerprint: str,
    save_every_batches: int = 10,
    progress_every_batches: int = 10,
    expose_counts: bool = True,
) -> dict[str, Any]:
    """Stream an exact-size post-selection simulator template to a cache.

    Raw feature batches and selected events are discarded immediately after
    their q and feature histograms are updated.  Milestones are nested prefixes
    of one deterministic selected-event stream.  Every raw batch has an
    independent seed derived from ``(seed, batch_index)``; an interrupted run
    can therefore resume from the last atomic summary without saving mutable
    RNG state or duplicating events. A partial cache additionally requires an
    exact runtime fingerprint match. A complete immutable cache may be loaded
    on a different runtime; its recorded runtime remains part of its content
    commitment and is returned to the caller.
    """

    cache_path = Path(cache_path)
    checkpoints = np.asarray(selected_checkpoints, dtype=np.int64).reshape(-1)
    feature_names = [str(name) for name in feature_names]
    edges = np.asarray(feature_edges, dtype=np.float64)
    q_edges = np.asarray(log_q_edges, dtype=np.float64).reshape(-1)
    batch_size = int(batch_size)
    save_every_batches = int(save_every_batches)
    progress_every_batches = int(progress_every_batches)
    runtime_fingerprint = str(runtime_fingerprint)
    if (
        len(checkpoints) < 1
        or np.any(checkpoints <= 0)
        or np.any(np.diff(checkpoints) <= 0)
    ):
        raise ValueError("selected_checkpoints must be positive and increasing.")
    if batch_size < 1 or save_every_batches < 1 or progress_every_batches < 1:
        raise ValueError("Batch and reporting intervals must be positive.")
    if not runtime_fingerprint:
        raise ValueError("runtime_fingerprint must be non-empty.")
    if edges.ndim != 2 or edges.shape[0] != len(feature_names):
        raise ValueError("feature_edges must have one row per feature name.")
    if edges.shape[1] < 3 or not np.all(np.diff(edges, axis=1) > 0.0):
        raise ValueError("Every feature edge row must be strictly increasing.")
    if len(q_edges) < 3 or not np.all(np.diff(q_edges) > 0.0):
        raise ValueError("log_q_edges must be strictly increasing.")
    manifest = _streamed_template_manifest(
        process=process,
        selected_checkpoints=checkpoints,
        batch_size=batch_size,
        seed=seed,
        feature_names=feature_names,
        feature_edges=edges,
        log_q_edges=q_edges,
        presel_ratio_cut=presel_ratio_cut,
        recipe_fingerprint=recipe_fingerprint,
    )
    manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    n_features = len(feature_names)
    n_feature_bins = edges.shape[1] - 1
    n_q_bins = len(q_edges) - 1

    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as saved:
            state = _validate_streamed_template_state(
                saved,
                expected_manifest_json=manifest_json,
                selected_checkpoints=checkpoints,
                n_features=n_features,
                n_feature_bins=n_feature_bins,
                n_q_bins=n_q_bins,
            )
        print(
            f"Resuming streamed {process} template at "
            f"{state['selected_events']:,}/{int(checkpoints[-1]):,} selected "
            f"events ({state['generated_events']:,} generated)."
        )
        if (
            state["selected_events"] < int(checkpoints[-1])
            and state["runtime_fingerprint"] != runtime_fingerprint
        ):
            raise RuntimeError(
                f"The partial streamed {process} template was created with "
                "different inference/runtime numerics. Resume on a matching "
                "runtime or start a new versioned cache."
            )
    else:
        state = {
            "manifest_json": manifest_json,
            "runtime_fingerprint": runtime_fingerprint,
            "selected_events": 0,
            "generated_events": 0,
            "passing_events": 0,
            "next_batch_index": 0,
            "n_completed_checkpoints": 0,
            "current_q_counts": np.zeros(n_q_bins, dtype=np.int64),
            "current_feature_counts": np.zeros(
                (n_features, n_feature_bins), dtype=np.int64
            ),
            "checkpoint_q_counts": np.zeros(
                (len(checkpoints), n_q_bins), dtype=np.int64
            ),
            "checkpoint_feature_counts": np.zeros(
                (len(checkpoints), n_features, n_feature_bins), dtype=np.int64
            ),
        }
        _save_streamed_template_state(cache_path, state=state)

    target = int(checkpoints[-1])
    started = time.monotonic()
    selected_at_start = int(state["selected_events"])
    while int(state["selected_events"]) < target:
        batch_index = int(state["next_batch_index"])
        completed_checkpoint_this_batch = False
        rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), batch_index])
        )
        values = np.asarray(sample_batch(batch_size, rng), dtype=np.float32)
        if values.shape != (batch_size, n_features):
            raise RuntimeError(
                f"sample_batch returned {values.shape}, expected "
                f"{(batch_size, n_features)}."
            )
        presel = np.asarray(
            preselection_ratio(values), dtype=np.float64
        ).reshape(-1)
        if len(presel) != batch_size:
            raise RuntimeError("preselection_ratio returned the wrong length.")
        passes = np.isfinite(presel) & (presel >= float(presel_ratio_cut))
        passing_values = values[passes]
        state["generated_events"] = int(state["generated_events"]) + batch_size
        state["passing_events"] = int(state["passing_events"]) + len(
            passing_values
        )
        remaining = target - int(state["selected_events"])
        if len(passing_values) > remaining:
            passing_values = passing_values[:remaining]
        if len(passing_values):
            log_q = np.asarray(
                log_q_evaluator(passing_values), dtype=np.float64
            ).reshape(-1)
            if len(log_q) != len(passing_values) or not np.isfinite(log_q).all():
                raise RuntimeError("log_q_evaluator returned invalid values.")

            offset = 0
            while offset < len(passing_values):
                checkpoint_index = int(state["n_completed_checkpoints"])
                checkpoint_target = int(checkpoints[checkpoint_index])
                take = min(
                    len(passing_values) - offset,
                    checkpoint_target - int(state["selected_events"]),
                )
                stop = offset + take
                q_segment = log_q[offset:stop]
                feature_segment = passing_values[offset:stop]
                state["current_q_counts"] += np.histogram(
                    q_segment, bins=q_edges
                )[0].astype(np.int64)
                for feature_index in range(n_features):
                    state["current_feature_counts"][feature_index] += np.histogram(
                        feature_segment[:, feature_index],
                        bins=edges[feature_index],
                    )[0].astype(np.int64)
                state["selected_events"] = int(state["selected_events"]) + take
                offset = stop
                if int(state["selected_events"]) == checkpoint_target:
                    state["checkpoint_q_counts"][checkpoint_index] = state[
                        "current_q_counts"
                    ]
                    state["checkpoint_feature_counts"][checkpoint_index] = state[
                        "current_feature_counts"
                    ]
                    state["n_completed_checkpoints"] = checkpoint_index + 1
                    completed_checkpoint_this_batch = True
                    print(
                        f"  {process}: completed nested selected-event "
                        f"checkpoint {checkpoint_target:,}."
                    )
        state["next_batch_index"] = batch_index + 1
        should_report = (
            state["next_batch_index"] % progress_every_batches == 0
            or int(state["selected_events"]) == target
        )
        should_save = (
            state["next_batch_index"] % save_every_batches == 0
            or completed_checkpoint_this_batch
            or should_report
        )
        if should_save:
            _save_streamed_template_state(cache_path, state=state)
        if should_report:
            elapsed = max(time.monotonic() - started, 1.0e-9)
            new_selected = int(state["selected_events"]) - selected_at_start
            rate = new_selected / elapsed
            remaining_seconds = (
                (target - int(state["selected_events"])) / rate
                if rate > 0.0 else math.inf
            )
            print(
                f"  {process}: {int(state['selected_events']):,}/{target:,} "
                f"selected from {int(state['generated_events']):,} generated; "
                f"acceptance={int(state['passing_events']) / int(state['generated_events']):.3%}; "
                f"session rate={rate:,.0f} selected/s; "
                f"ETA={remaining_seconds / 60.0:,.1f} min."
            )
        del values, presel, passes, passing_values

    _save_streamed_template_state(cache_path, state=state)
    return load_streamed_simulator_template(
        cache_path=cache_path,
        process=process,
        selected_checkpoints=checkpoints,
        batch_size=batch_size,
        seed=seed,
        feature_names=feature_names,
        feature_edges=edges,
        log_q_edges=q_edges,
        presel_ratio_cut=presel_ratio_cut,
        recipe_fingerprint=recipe_fingerprint,
        expose_counts=expose_counts,
    )


def sample_truncated_spline_flow(
    flow_pack: Mapping[str, Any],
    context: np.ndarray,
    *,
    lower_bound: float,
    seed: int,
    max_rounds: int = 100,
) -> tuple[np.ndarray, float]:
    """Draw one conditional scalar-flow value per context above a boundary."""

    context = np.asarray(context, dtype=np.float32)
    if context.ndim == 1:
        context = context[:, None]
    if context.ndim != 2:
        raise ValueError("context must be one- or two-dimensional.")
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    values = np.empty(len(context), dtype=np.float32)
    pending = np.arange(len(context))
    total_draws = 0
    for _ in range(int(max_rounds)):
        if len(pending) == 0:
            break
        draws = sample_spline_flow(
            flow_pack, 1, context=context[pending]
        ).reshape(-1)
        if len(draws) != len(pending):
            raise RuntimeError("Conditional flow returned an unexpected shape.")
        total_draws += len(draws)
        valid = np.isfinite(draws) & (draws >= float(lower_bound))
        values[pending[valid]] = draws[valid]
        pending = pending[~valid]
    if len(pending):
        raise RuntimeError(
            "Conditional-flow rejection sampling did not finish. Increase "
            "the log-statistic offset or inspect the trained flow."
        )
    rejection_fraction = 1.0 - len(context) / total_draws
    return values, float(rejection_fraction)


def conditional_density_grid(
    flow_pack: Mapping[str, Any],
    ratio_stages: Sequence[Sequence[Mapping[str, Any]]],
    mu_values: np.ndarray,
    flow_coordinate_grid: np.ndarray,
    *,
    max_abs_log_ratio: float = 15.0,
    batch_size: int = 65_536,
) -> dict[str, np.ndarray]:
    """Normalize a ratio-corrected conditional density by 1D quadrature."""

    mu_values = np.asarray(mu_values, dtype=np.float32).reshape(-1)
    coordinate = np.asarray(flow_coordinate_grid, dtype=np.float64).reshape(-1)
    if len(mu_values) < 1 or len(coordinate) < 4:
        raise ValueError("At least one mu and four quadrature points are required.")
    if not np.all(np.diff(coordinate) > 0.0):
        raise ValueError("flow_coordinate_grid must be strictly increasing.")
    n_mu = len(mu_values)
    n_coordinate = len(coordinate)
    context = np.repeat(mu_values, n_coordinate)[:, None]
    target = np.tile(coordinate.astype(np.float32), n_mu)[:, None]
    log_density = spline_flow_log_prob(
        flow_pack,
        target,
        context=context,
        batch_size=batch_size,
    ).astype(np.float64)
    ratio_input = np.column_stack([context[:, 0], target[:, 0]])
    ratio_clip_fraction = []
    ratio_log_range = []
    for ensemble in ratio_stages:
        log_ratio = ratio_classifier_ensemble_logit(
            list(ensemble), ratio_input, batch_size=batch_size
        )
        ratio_clip_fraction.append(
            float(np.mean(np.abs(log_ratio) > float(max_abs_log_ratio)))
        )
        ratio_log_range.append(
            [float(np.min(log_ratio)), float(np.max(log_ratio))]
        )
        log_density += np.clip(
            log_ratio, -float(max_abs_log_ratio), float(max_abs_log_ratio)
        )
    log_density = log_density.reshape(n_mu, n_coordinate)
    row_maximum = np.max(log_density, axis=1, keepdims=True)
    scaled_density = np.exp(log_density - row_maximum)
    scaled_normalization = trapezoid(
        scaled_density, x=coordinate, axis=1
    )
    if (
        not np.isfinite(scaled_normalization).all()
        or np.any(scaled_normalization <= 0.0)
    ):
        raise FloatingPointError("Conditional quadrature normalization failed.")
    density = scaled_density / scaled_normalization[:, None]
    cdf = cumulative_trapezoid(
        density, x=coordinate, axis=1, initial=0.0
    )
    cdf /= cdf[:, -1, None]
    log_normalization = (
        row_maximum[:, 0] + np.log(scaled_normalization)
    )
    return {
        "density": density,
        "cdf": cdf,
        "log_normalization": log_normalization,
        "ratio_clip_fraction": np.asarray(
            ratio_clip_fraction, dtype=np.float64
        ),
        "ratio_log_range": np.asarray(ratio_log_range, dtype=np.float64).reshape(
            -1, 2
        ),
    }


def conditional_quantiles(
    cdf: np.ndarray,
    flow_coordinate_grid: np.ndarray,
    probabilities: Sequence[float],
) -> np.ndarray:
    """Invert one row-wise numerical CDF for each requested probability."""

    cdf = np.asarray(cdf, dtype=np.float64)
    coordinate = np.asarray(flow_coordinate_grid, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if cdf.ndim != 2 or cdf.shape[1] != len(coordinate):
        raise ValueError("cdf shape does not match flow_coordinate_grid.")
    if np.any((probabilities <= 0.0) | (probabilities >= 1.0)):
        raise ValueError("Quantile probabilities must lie strictly in (0, 1).")
    return np.asarray(
        [
            [np.interp(probability, row, coordinate) for probability in probabilities]
            for row in cdf
        ],
        dtype=np.float64,
    )


def conditional_row_quantiles(
    cdf: np.ndarray,
    coordinate_grid: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    """Invert each conditional CDF at row-dependent probabilities.

    ``probabilities`` may have shape ``(n_rows,)`` or ``(n_rows, n_levels)``.
    This is useful after a calibration map supplies a different probability
    level for every parameter point.
    """

    cdf = np.asarray(cdf, dtype=np.float64)
    coordinate = np.asarray(coordinate_grid, dtype=np.float64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if cdf.ndim != 2 or cdf.shape[1] != len(coordinate):
        raise ValueError("cdf shape does not match coordinate_grid.")
    squeeze = probabilities.ndim == 1
    if squeeze:
        probabilities = probabilities[:, None]
    if probabilities.ndim != 2 or probabilities.shape[0] != cdf.shape[0]:
        raise ValueError(
            "probabilities must have one row per conditional CDF."
        )
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("Probabilities must lie in [0, 1].")
    result = np.asarray(
        [
            [np.interp(probability, row, coordinate) for probability in levels]
            for row, levels in zip(cdf, probabilities)
        ],
        dtype=np.float64,
    )
    return result[:, 0] if squeeze else result


def conditional_cdf_values(
    cdf: np.ndarray,
    context_grid: np.ndarray,
    coordinate_grid: np.ndarray,
    context: np.ndarray,
    coordinate: np.ndarray,
) -> np.ndarray:
    """Bilinearly interpolate a scalar-context conditional CDF.

    Coordinates below or above the tabulated support return zero or one.
    Context values must lie within the design grid.  The helper is vectorized
    so it can transform large toy ensembles without a Python loop.
    """

    cdf = np.asarray(cdf, dtype=np.float64)
    context_grid = np.asarray(context_grid, dtype=np.float64).reshape(-1)
    coordinate_grid = np.asarray(
        coordinate_grid, dtype=np.float64
    ).reshape(-1)
    context = np.asarray(context, dtype=np.float64).reshape(-1)
    coordinate = np.asarray(coordinate, dtype=np.float64).reshape(-1)
    if len(context) != len(coordinate):
        raise ValueError("context and coordinate must have equal length.")
    if cdf.shape != (len(context_grid), len(coordinate_grid)):
        raise ValueError("cdf shape does not match the supplied grids.")
    if len(context_grid) < 2 or len(coordinate_grid) < 2:
        raise ValueError("Both interpolation grids require at least two points.")
    if not np.all(np.diff(context_grid) > 0.0) or not np.all(
        np.diff(coordinate_grid) > 0.0
    ):
        raise ValueError("Interpolation grids must be strictly increasing.")
    tolerance = 1.0e-10 * max(1.0, np.max(np.abs(context_grid)))
    if np.any(context < context_grid[0] - tolerance) or np.any(
        context > context_grid[-1] + tolerance
    ):
        raise ValueError("Context values lie outside the conditional CDF grid.")

    context_clipped = np.clip(context, context_grid[0], context_grid[-1])
    context_index = np.searchsorted(
        context_grid, context_clipped, side="right"
    ) - 1
    context_index = np.clip(context_index, 0, len(context_grid) - 2)
    context_low = context_grid[context_index]
    context_high = context_grid[context_index + 1]
    context_fraction = (context_clipped - context_low) / (
        context_high - context_low
    )

    coordinate_clipped = np.clip(
        coordinate, coordinate_grid[0], coordinate_grid[-1]
    )
    coordinate_index = np.searchsorted(
        coordinate_grid, coordinate_clipped, side="right"
    ) - 1
    coordinate_index = np.clip(
        coordinate_index, 0, len(coordinate_grid) - 2
    )
    coordinate_low = coordinate_grid[coordinate_index]
    coordinate_high = coordinate_grid[coordinate_index + 1]
    coordinate_fraction = (coordinate_clipped - coordinate_low) / (
        coordinate_high - coordinate_low
    )

    lower_context = (
        (1.0 - coordinate_fraction)
        * cdf[context_index, coordinate_index]
        + coordinate_fraction * cdf[context_index, coordinate_index + 1]
    )
    upper_context = (
        (1.0 - coordinate_fraction)
        * cdf[context_index + 1, coordinate_index]
        + coordinate_fraction
        * cdf[context_index + 1, coordinate_index + 1]
    )
    result = (
        (1.0 - context_fraction) * lower_context
        + context_fraction * upper_context
    )
    result = np.where(coordinate < coordinate_grid[0], 0.0, result)
    result = np.where(coordinate > coordinate_grid[-1], 1.0, result)
    return np.clip(result, 0.0, 1.0)


def conditional_ratio_grid(
    ratio_ensemble: Sequence[Mapping[str, Any]],
    context_values: np.ndarray,
    coordinate_grid: np.ndarray,
    *,
    max_abs_log_ratio: float = 15.0,
    batch_size: int = 65_536,
) -> dict[str, np.ndarray]:
    """Normalize classifier odds over one scalar coordinate.

    The base measure is Lebesgue measure on ``coordinate_grid``.  In Exercise
    11 this grid is ``[0, 1]``, so the base density is exactly Uniform(0, 1)
    and the normalized odds estimate the simulator PIT density.

    ``context_values`` may contain any number of parameter columns.  Only the
    quadrature coordinate remains one-dimensional, which is the key scaling
    advantage of calibration in PIT space.
    """

    context_values = np.asarray(context_values, dtype=np.float32)
    if context_values.ndim == 1:
        context_values = context_values[:, None]
    coordinate = np.asarray(coordinate_grid, dtype=np.float64).reshape(-1)
    if context_values.ndim != 2 or len(context_values) < 1:
        raise ValueError("context_values must be a non-empty matrix.")
    if len(coordinate) < 4 or not np.all(np.diff(coordinate) > 0.0):
        raise ValueError(
            "coordinate_grid needs at least four strictly increasing points."
        )
    n_context = len(context_values)
    n_coordinate = len(coordinate)
    repeated_context = np.repeat(context_values, n_coordinate, axis=0)
    tiled_coordinate = np.tile(
        coordinate.astype(np.float32), n_context
    )[:, None]
    ratio_input = np.column_stack([repeated_context, tiled_coordinate])
    raw_log_ratio = ratio_classifier_ensemble_logit(
        list(ratio_ensemble), ratio_input, batch_size=batch_size
    ).astype(np.float64)
    clip_fraction = float(
        np.mean(np.abs(raw_log_ratio) > float(max_abs_log_ratio))
    )
    log_ratio_range = np.asarray(
        [np.min(raw_log_ratio), np.max(raw_log_ratio)], dtype=np.float64
    )
    log_ratio = np.clip(
        raw_log_ratio, -float(max_abs_log_ratio), float(max_abs_log_ratio)
    ).reshape(n_context, n_coordinate)
    row_maximum = np.max(log_ratio, axis=1, keepdims=True)
    scaled_density = np.exp(log_ratio - row_maximum)
    scaled_normalization = trapezoid(
        scaled_density, x=coordinate, axis=1
    )
    if not np.isfinite(scaled_normalization).all() or np.any(
        scaled_normalization <= 0.0
    ):
        raise FloatingPointError("Conditional ratio normalization failed.")
    density = scaled_density / scaled_normalization[:, None]
    cdf = cumulative_trapezoid(
        density, x=coordinate, axis=1, initial=0.0
    )
    cdf /= cdf[:, -1, None]
    return {
        "density": density,
        "cdf": cdf,
        "log_normalization": (
            row_maximum[:, 0] + np.log(scaled_normalization)
        ),
        "ratio_clip_fraction": clip_fraction,
        "ratio_log_range": log_ratio_range,
    }


def conservative_empirical_quantile(
    values: np.ndarray,
    probability: float | Sequence[float],
) -> np.ndarray:
    """Finite-sample upper quantile for a Neyman acceptance cutoff.

    For ``n`` calibration toys the one-indexed order is
    ``ceil(probability * (n + 1))``, clipped to ``[1, n]``.  This is the
    standard split-conformal/Neyman finite-sample choice.  Ties can make the
    resulting non-randomized test conservative.
    """

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    probability = np.asarray(probability, dtype=np.float64)
    if len(values) < 1 or not np.isfinite(values).all():
        raise ValueError("values must be a non-empty finite array.")
    if np.any((probability <= 0.0) | (probability >= 1.0)):
        raise ValueError("probability must lie strictly in (0, 1).")
    ordered = np.sort(values)
    order = np.ceil(probability * (len(ordered) + 1)).astype(np.int64)
    order = np.clip(order, 1, len(ordered))
    return ordered[order - 1]


def importance_effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("weights must be finite and non-negative.")
    total = float(weights.sum())
    if not total > 0.0:
        raise ValueError("weights must have positive total.")
    normalized = weights / total
    return float(1.0 / np.sum(normalized**2))


class _CoverageAuditor(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_features: int,
        hidden_layers: int,
        initial_probability: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        n_input = int(n_features)
        for _ in range(int(hidden_layers)):
            layers.extend(
                [nn.Linear(n_input, int(hidden_features)), nn.SiLU()]
            )
            n_input = int(hidden_features)
        output = nn.Linear(n_input, 1)
        nn.init.zeros_(output.weight)
        initial_probability = float(
            np.clip(initial_probability, 1.0e-4, 1.0 - 1.0e-4)
        )
        nn.init.constant_(
            output.bias,
            math.log(initial_probability / (1.0 - initial_probability)),
        )
        layers.append(output)
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def _build_coverage_auditor(
    config: Mapping[str, Any], device: torch.device
) -> nn.Module:
    return _CoverageAuditor(
        n_features=int(config["n_features"]),
        hidden_features=int(config["hidden_features"]),
        hidden_layers=int(config["hidden_layers"]),
        initial_probability=float(config.get("initial_probability", 0.95)),
    ).to(device)


def _torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_coverage_auditor(
    checkpoint: str | Path, device: torch.device
) -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    saved = _torch_load(checkpoint, device)
    config = dict(saved["config"])
    model = _build_coverage_auditor(config, device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return {
        "model": model,
        "scaler": ArrayStandardizer(
            mean=np.asarray(saved["mean"], dtype=np.float32),
            std=np.asarray(saved["std"], dtype=np.float32),
        ),
        "config": config,
        "history": saved.get("history", {}),
        "training_fingerprint": saved.get("training_fingerprint"),
        "validation_indices": np.asarray(
            saved.get("validation_indices", []), dtype=np.int64
        ),
        "test_indices": np.asarray(
            saved.get("test_indices", []), dtype=np.int64
        ),
        "checkpoint": checkpoint,
    }


def train_coverage_auditor(
    mu: np.ndarray,
    covered: np.ndarray,
    *,
    checkpoint: str | Path,
    model_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    device: torch.device,
    seed: int,
    load_if_available: bool = True,
) -> dict[str, Any]:
    """Fit ``P(covered=1 | mu)`` with natural-frequency, unweighted BCE.

    Early stopping uses a validation split.  A second held-out test split is
    stored in the checkpoint so the fitted curve can be compared with fixed
    and constant predictors on rows that did not select the stopping epoch.
    """

    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    mu = np.asarray(mu, dtype=np.float32).reshape(-1, 1)
    covered = np.asarray(covered, dtype=np.float32).reshape(-1)
    if len(mu) != len(covered) or len(mu) < 20:
        raise ValueError("mu/covered must contain at least twenty matched rows.")
    if not np.isfinite(mu).all() or not np.isin(covered, [0.0, 1.0]).all():
        raise ValueError("mu must be finite and covered must be binary.")
    fingerprint_configuration = json.dumps(
        {
            "model_config": dict(model_config),
            "training_config": dict(training_config),
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    training_fingerprint = _array_fingerprint(
        mu,
        covered,
        np.asarray([seed], dtype=np.int64),
        np.asarray([fingerprint_configuration]),
    )
    if load_if_available and checkpoint.exists():
        print(f"Loading coverage auditor from {checkpoint}")
        loaded = load_coverage_auditor(checkpoint, device)
        if loaded.get("training_fingerprint") != training_fingerprint:
            raise RuntimeError(
                "The cached coverage auditor used different rows, labels, "
                "seed, or training settings. Use a new checkpoint path or "
                "disable LOAD_IF_AVAILABLE."
            )
        configuration_mismatch = {
            name: (loaded["config"].get(name), expected)
            for name, expected in model_config.items()
            if loaded["config"].get(name) != expected
        }
        if configuration_mismatch:
            raise RuntimeError(
                "The cached coverage auditor has a different architecture: "
                f"{configuration_mismatch}. Use a new checkpoint path."
            )
        return loaded

    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(mu))
    validation_fraction = float(
        training_config.get("validation_fraction", 0.2)
    )
    test_fraction = float(training_config.get("test_fraction", 0.2))
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1.")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be strictly between 0 and 1.")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError(
            "validation_fraction + test_fraction must be smaller than one."
        )
    n_validation = min(
        len(mu) - 2,
        max(1, int(round(validation_fraction * len(mu)))),
    )
    n_test = min(
        len(mu) - n_validation - 1,
        max(1, int(round(test_fraction * len(mu)))),
    )
    validation_indices = order[:n_validation]
    test_indices = order[n_validation : n_validation + n_test]
    training_indices = order[n_validation + n_test :]
    scaler = ArrayStandardizer.fit(mu[training_indices])
    mu_scaled = scaler.transform(mu)
    mu_tensor = torch.tensor(mu_scaled, dtype=torch.float32)
    covered_tensor = torch.tensor(covered, dtype=torch.float32)
    generator = torch.Generator().manual_seed(int(seed))
    training_loader = DataLoader(
        TensorDataset(
            mu_tensor[training_indices], covered_tensor[training_indices]
        ),
        batch_size=int(training_config.get("batch_size", 2048)),
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        TensorDataset(
            mu_tensor[validation_indices], covered_tensor[validation_indices]
        ),
        batch_size=int(training_config.get("batch_size", 2048)),
        shuffle=False,
    )

    config = dict(model_config)
    config["n_features"] = 1
    config["initial_probability"] = float(
        np.mean(covered[training_indices])
    )
    model = _build_coverage_auditor(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 1.0e-3)),
        weight_decay=float(training_config.get("weight_decay", 1.0e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training_config.get("lr_scheduler_factor", 0.3)),
        patience=int(training_config.get("lr_scheduler_patience", 5)),
        min_lr=float(training_config.get("min_learning_rate", 1.0e-6)),
    )
    n_epochs = int(training_config.get("n_epochs", 200))
    patience = int(training_config.get("patience", 20))
    gradient_clip = float(training_config.get("gradient_clip", 5.0))
    best_validation = math.inf
    best_state = None
    stale_epochs = 0
    history = {"train": [], "validation": [], "learning_rate": []}
    for epoch in range(1, n_epochs + 1):
        model.train()
        train_numerator = 0.0
        train_rows = 0
        for mu_batch, covered_batch in training_loader:
            mu_batch = mu_batch.to(device)
            covered_batch = covered_batch.to(device)
            loss = F.binary_cross_entropy_with_logits(
                model(mu_batch), covered_batch
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            train_numerator += float(loss.detach().cpu()) * len(mu_batch)
            train_rows += len(mu_batch)

        model.eval()
        validation_numerator = 0.0
        validation_rows = 0
        with torch.no_grad():
            for mu_batch, covered_batch in validation_loader:
                mu_batch = mu_batch.to(device)
                covered_batch = covered_batch.to(device)
                loss = F.binary_cross_entropy_with_logits(
                    model(mu_batch), covered_batch
                )
                validation_numerator += float(loss.detach().cpu()) * len(
                    mu_batch
                )
                validation_rows += len(mu_batch)
        train_loss = train_numerator / train_rows
        validation_loss = validation_numerator / validation_rows
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history["train"].append(train_loss)
        history["validation"].append(validation_loss)
        history["learning_rate"].append(learning_rate)
        scheduler.step(validation_loss)
        if validation_loss < best_validation - 1.0e-6:
            best_validation = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"  auditor epoch {epoch:03d}: train={train_loss:.6f}, "
                f"validation={validation_loss:.6f}, lr={learning_rate:.2e}"
            )
        if stale_epochs >= patience:
            print(f"  auditor early stopping after {epoch} epochs")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(mu_tensor[test_indices].to(device))
        test_loss = float(F.binary_cross_entropy_with_logits(
            test_logits,
            covered_tensor[test_indices].to(device),
        ).detach().cpu())
    history["test"] = [test_loss]
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config,
            "mean": scaler.mean,
            "std": scaler.std,
            "history": history,
            "training_fingerprint": training_fingerprint,
            "validation_indices": validation_indices,
            "test_indices": test_indices,
        },
        checkpoint,
    )
    print(f"Saved coverage auditor to {checkpoint}")
    return load_coverage_auditor(checkpoint, device)


@torch.no_grad()
def coverage_auditor_probability(
    pack: Mapping[str, Any],
    mu: np.ndarray,
    *,
    batch_size: int = 65_536,
) -> np.ndarray:
    mu = np.asarray(mu, dtype=np.float32).reshape(-1, 1)
    model = pack["model"]
    device = next(model.parameters()).device
    model.eval()
    chunks = []
    for start in range(0, len(mu), int(batch_size)):
        tensor = torch.tensor(
            pack["scaler"].transform(mu[start : start + int(batch_size)]),
            dtype=torch.float32,
            device=device,
        )
        chunks.append(torch.sigmoid(model(tensor)).cpu().numpy())
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)


def wilson_interval(
    successes: int | np.ndarray,
    trials: int | np.ndarray,
    *,
    z: float = 1.959963984540054,
) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for one or many binomial proportions."""

    successes = np.asarray(successes, dtype=np.float64)
    trials = np.asarray(trials, dtype=np.float64)
    if np.any(trials <= 0.0) or np.any(successes < 0.0) or np.any(
        successes > trials
    ):
        raise ValueError("Require 0 <= successes <= trials and trials > 0.")
    probability = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (probability + z**2 / (2.0 * trials)) / denominator
    half_width = (
        z
        / denominator
        * np.sqrt(
            probability * (1.0 - probability) / trials
            + z**2 / (4.0 * trials**2)
        )
    )
    return center - half_width, center + half_width


def binned_coverage(
    mu: np.ndarray,
    covered: np.ndarray,
    *,
    edges: np.ndarray,
) -> dict[str, np.ndarray]:
    """Equal-definition binomial coverage summary with Wilson intervals."""

    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    covered = np.asarray(covered, dtype=bool).reshape(-1)
    edges = np.asarray(edges, dtype=np.float64)
    if len(mu) != len(covered):
        raise ValueError("mu and covered must have equal length.")
    bin_index = np.clip(np.digitize(mu, edges) - 1, 0, len(edges) - 2)
    trials = np.bincount(bin_index, minlength=len(edges) - 1)
    successes = np.bincount(
        bin_index, weights=covered.astype(np.int64), minlength=len(edges) - 1
    )
    if np.any(trials == 0):
        raise ValueError("Every requested coverage bin must be populated.")
    lower, upper = wilson_interval(successes, trials)
    return {
        "center": 0.5 * (edges[:-1] + edges[1:]),
        "trials": trials,
        "successes": successes,
        "coverage": successes / trials,
        "lower": lower,
        "upper": upper,
    }
