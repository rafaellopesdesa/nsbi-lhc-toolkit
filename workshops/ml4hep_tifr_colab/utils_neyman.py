"""Utilities for the amortized Neyman-construction tutorial (Exercise 11).

The helpers in this module keep the notebook focused on the statistical
construction.  They provide five pieces of infrastructure:

* the one-dimensional likelihood-ratio compression used in Exercise 5;
* resumable, batch-wise pseudo-experiment generation;
* reconstruction of two disjoint simulator templates from Exercise 5's
  held-out density-evaluation arrays;
* numerical conditional-density primitives and quantiles; and
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
) -> dict[str, np.ndarray]:
    """Generate and fit Poisson toys in resumable, deterministic shards.

    ``fit_batch`` receives ``(counts, test_mu)`` and returns
    ``(mu_hat, t_mu, fitted_score)``.  ``fit_fingerprint`` must be bumped when
    its numerical definition changes. Count matrices exist only for one batch
    and are never written to disk.
    """

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = cache_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    n_toys = int(n_toys)
    batch_size = int(batch_size)
    if n_toys < 1 or batch_size < 1:
        raise ValueError("n_toys and batch_size must be positive.")
    signal_probability = np.asarray(signal_probability, dtype=np.float64)
    background_probability = np.asarray(
        background_probability, dtype=np.float64
    )
    likelihood_q = np.asarray(likelihood_q, dtype=np.float64)
    if signal_probability.shape != background_probability.shape:
        raise ValueError("Signal/background probabilities must have equal shape.")
    if likelihood_q.shape != signal_probability.shape:
        raise ValueError("likelihood_q must match the template binning.")
    if not np.isclose(signal_probability.sum(), 1.0):
        raise ValueError("Signal probabilities must sum to one.")
    if not np.isclose(background_probability.sum(), 1.0):
        raise ValueError("Background probabilities must sum to one.")

    manifest = {
        "version": 3,
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
    }
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing != manifest:
            raise RuntimeError(
                f"Toy cache {cache_dir} was made with a different "
                "configuration. Remove that versioned directory or choose "
                "a new RUN_TAG."
            )
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    n_batches = int(math.ceil(n_toys / batch_size))
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
        poisson_mean = (
            mu[:, None] * float(lam_signal) * signal_probability[None, :]
            + float(lam_background) * background_probability[None, :]
        )
        counts = rng.poisson(poisson_mean)
        mu_hat, t_mu, fitted_score = fit_batch(counts, mu)
        payload = {
            "mu": mu.astype(np.float32),
            "mu_hat": np.asarray(mu_hat, dtype=np.float32),
            "t_mu": np.asarray(t_mu, dtype=np.float32),
            "n_events": counts.sum(axis=1).astype(np.int32),
            "fitted_score": np.asarray(fitted_score, dtype=np.float32),
        }
        temporary_path = shard_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary_path, **payload)
        temporary_path.replace(shard_path)
        if (batch_index + 1) % max(1, n_batches // 10) == 0:
            print(f"  completed {stop:,}/{n_toys:,} toys in {cache_dir.name}")

    chunks: dict[str, list[np.ndarray]] = {
        name: []
        for name in ["mu", "mu_hat", "t_mu", "n_events", "fitted_score"]
    }
    for batch_index in range(n_batches):
        with np.load(shard_dir / f"batch_{batch_index:05d}.npz") as shard:
            for name in chunks:
                chunks[name].append(np.asarray(shard[name]))
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
    probability_floor: float = 1.0e-15,
) -> dict[str, Any]:
    """Split Exercise 5's held-out simulator arrays into calibration/audit.

    Exercise 5 concatenates background evaluation rows followed by signal
    evaluation rows.  The generator assigns one constant event weight per
    process, so the unique contiguous weight change recovers that boundary
    without assuming a fixed reservoir size.
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

    log_q = (
        np.log(float(lam_signal) / float(lam_background))
        + np.log(ratio_signal)
        - np.log(ratio_background)
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
    }
    for process, indices in slices.items():
        order = indices[rng.permutation(len(indices))]
        split = len(order) // 2
        subsets = {
            "calibration": order[:split],
            "audit": order[split:],
        }
        for subset_name, subset_indices in subsets.items():
            probability = np.histogram(
                log_q[subset_indices],
                bins=edges,
                weights=weights[subset_indices],
            )[0].astype(np.float64)
            probability += float(probability_floor)
            probability /= probability.sum()
            templates[f"{subset_name}_{process}_probability"] = probability
            templates[f"{subset_name}_{process}_events"] = len(subset_indices)
    return templates


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
    """Fit ``P(covered=1 | mu)`` with natural-frequency, unweighted BCE."""

    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    mu = np.asarray(mu, dtype=np.float32).reshape(-1, 1)
    covered = np.asarray(covered, dtype=np.float32).reshape(-1)
    if len(mu) != len(covered) or len(mu) < 20:
        raise ValueError("mu/covered must contain at least twenty matched rows.")
    if not np.isfinite(mu).all() or not np.isin(covered, [0.0, 1.0]).all():
        raise ValueError("mu must be finite and covered must be binary.")
    training_fingerprint = _array_fingerprint(mu, covered)
    if load_if_available and checkpoint.exists():
        print(f"Loading coverage auditor from {checkpoint}")
        loaded = load_coverage_auditor(checkpoint, device)
        if loaded.get("training_fingerprint") != training_fingerprint:
            raise RuntimeError(
                "The cached coverage auditor was trained on different audit "
                "labels. Bump RUN_TAG or disable LOAD_IF_AVAILABLE."
            )
        configuration_mismatch = {
            name: (loaded["config"].get(name), expected)
            for name, expected in model_config.items()
            if loaded["config"].get(name) != expected
        }
        if configuration_mismatch:
            raise RuntimeError(
                "The cached coverage auditor has a different architecture: "
                f"{configuration_mismatch}. Bump RUN_TAG."
            )
        return loaded

    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(mu))
    validation_fraction = float(
        training_config.get("validation_fraction", 0.2)
    )
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1.")
    n_validation = min(
        len(mu) - 1,
        max(1, int(round(validation_fraction * len(mu)))),
    )
    validation_indices = order[:n_validation]
    training_indices = order[n_validation:]
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
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config,
            "mean": scaler.mean,
            "std": scaler.std,
            "history": history,
            "training_fingerprint": training_fingerprint,
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
