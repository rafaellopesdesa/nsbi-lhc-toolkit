"""Hybrid hNPE--hNDE helpers for the ``sbibm`` benchmark exercise.

The benchmark in Lueckmann et al. counts calls to the simulator.  This module
therefore keeps simulation, model training, posterior construction, and metric
evaluation separate.  A single cached simulation bank can be shared by the
posterior and likelihood routes without hiding additional simulator calls.

Exercise 10 deliberately keeps its improved training machinery in this module:
the spline flows use learned LU mixing, the simulation bank is genuinely
cross-fitted, and paired classifier rows receive an honest grouped validation
split.  The likelihood route uses the conditional-reference utilities in
:mod:`utils_dual_hnde`; the standard hNDE code used by Exercises 5--8 is not
imported or modified.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.special import logsumexp
from scipy.stats import kstest
from torch.utils.data import DataLoader, TensorDataset

from utils_dual_hnde import (
    conditional_flow_c2st,
    conditional_log_normalizer,
    conditional_normalization_diagnostics,
    conditional_residual_log_ratio,
    importance_tail_summary,
    ratio_member_tail_diagnostics,
    train_conditional_log_normalizer,
)
from utils_hnpe import (
    ArrayStandardizer,
    ratio_classifier_ensemble_logit,
    sample_spline_flow,
    spline_flow_log_prob,
    train_ratio_classifier,
)


PAPER_RESULTS_URL = (
    "https://raw.githubusercontent.com/sbi-benchmark/results/main/"
    "benchmarking_sbi/results/main_paper.csv"
)
GAUSSIAN_MIXTURE_RERUN_URL = (
    "https://raw.githubusercontent.com/sbi-benchmark/results/main/"
    "gaussian_mixture_rerun/results/gaussian_mixture_rerun.csv"
)
PAPER_ALGORITHMS = (
    "REJ-ABC",
    "SMC-ABC",
    "NLE",
    "NPE",
    "NRE",
    "SNLE",
    "SNPE",
    "SNRE",
)


@dataclass(frozen=True)
class TaskRecommendation:
    """Static information used to choose a suitable hybrid construction."""

    task: str
    display_name: str
    dim_parameters: int
    dim_data: int
    data_type: str
    recommended_method: str
    colab_status: str
    reason: str


TASK_RECOMMENDATIONS = (
    TaskRecommendation(
        "gaussian_linear",
        "Gaussian Linear",
        10,
        10,
        "continuous",
        "dual hNPE--hNDE",
        "ready",
        "A normalization and scaling control; the posterior is nearly solved.",
    ),
    TaskRecommendation(
        "gaussian_linear_uniform",
        "Gaussian Linear Uniform",
        10,
        10,
        "continuous, bounded parameters",
        "dual hNPE--hNDE",
        "ready",
        "Tests exact bounded support through the parameter logit transform.",
    ),
    TaskRecommendation(
        "gaussian_mixture",
        "Gaussian Mixture",
        2,
        2,
        "continuous mixture",
        "dual hNPE--hNDE",
        "ready",
        "A compact heavy-tail test; corrected v1.1 benchmark results are used.",
    ),
    TaskRecommendation(
        "two_moons",
        "Two Moons",
        2,
        2,
        "continuous, multimodal",
        "dual hNPE--hNDE",
        "primary target",
        "The paper's diagnostic example has curved modes and clear headroom.",
    ),
    TaskRecommendation(
        "slcp",
        "SLCP",
        5,
        8,
        "continuous, four-mode posterior",
        "dual hNPE--hNDE",
        "stretch target",
        "The best published mean C2ST at 100k is still far from 0.5.",
    ),
    TaskRecommendation(
        "slcp_distractors",
        "SLCP Distractors",
        5,
        100,
        "continuous with 92 distractors",
        "hNPE; optional dual",
        "expensive",
        "hNPE avoids making a 100-dimensional observation flow the bottleneck.",
    ),
    TaskRecommendation(
        "bernoulli_glm",
        "Bernoulli GLM",
        10,
        10,
        "discrete-derived sufficient statistics",
        "hNPE",
        "ready",
        "A conditional posterior can use the summaries directly; a continuous "
        "hNDE reference is not an exact probability-mass model.",
    ),
    TaskRecommendation(
        "bernoulli_glm_raw",
        "Bernoulli GLM Raw",
        10,
        100,
        "binary",
        "hNPE",
        "expensive",
        "The raw observation is discrete and 100-dimensional.",
    ),
    TaskRecommendation(
        "sir",
        "SIR",
        2,
        10,
        "count time series",
        "hNPE",
        "launcher-ready",
        "The task's legacy diffeqtorch solver is replaced by an audited Python "
        "integration of the same SIR equations; count data are dequantized.",
    ),
    TaskRecommendation(
        "lotka_volterra",
        "Lotka--Volterra",
        4,
        20,
        "continuous LogNormal-noised trajectories",
        "hNPE",
        "launcher-ready",
        "The task's legacy diffeqtorch solver is replaced by an audited Python "
        "integration of the same equations and LogNormal observation model.",
    ),
)


def task_recommendation_table() -> pd.DataFrame:
    """Return the ten paper tasks and the recommended Exercise 10 route."""

    return pd.DataFrame([item.__dict__ for item in TASK_RECOMMENDATIONS])


def seed_everything(seed: int) -> None:
    """Seed NumPy, Python, and PyTorch without changing determinism settings."""

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_lu_mixed_spline(
    *,
    n_features: int,
    context_features: int,
    n_coupling_layers: int,
    hidden_features: int,
    hidden_layers: int,
    spline_num_bins: int,
    spline_tail_bound: float,
    dropout_probability: float,
) -> torch.nn.Module:
    """Build an NSF whose coupling layers are separated by learned LU mixing.

    This mirrors the transform topology used by the ``sbi`` NSF benchmark:
    alternating masks alone decide which coordinates are transformed, while
    every coupling is followed by an invertible learned linear map.  In
    particular, no reversal is combined with an alternating two-dimensional
    mask, which would repeatedly transform the same original coordinate.
    """

    try:
        from nflows.distributions.normal import StandardNormal
        from nflows.flows.base import Flow
        from nflows.nn.nets import ResidualNet
        from nflows.transforms.base import CompositeTransform
        from nflows.transforms.coupling import (
            PiecewiseRationalQuadraticCouplingTransform,
        )
        from nflows.transforms.lu import LULinear
        from nflows.utils.torchutils import create_alternating_binary_mask
    except ImportError as exc:
        raise ImportError(
            "Exercise 10 requires nflows. Install it with `pip install nflows`."
        ) from exc

    def make_net(in_features: int, out_features: int) -> torch.nn.Module:
        return ResidualNet(
            in_features=in_features,
            out_features=out_features,
            hidden_features=int(hidden_features),
            context_features=(int(context_features) or None),
            num_blocks=int(hidden_layers),
            activation=F.relu,
            dropout_probability=float(dropout_probability),
            use_batch_norm=False,
        )

    transforms = []
    for layer_index in range(int(n_coupling_layers)):
        mask = create_alternating_binary_mask(
            features=int(n_features),
            even=(layer_index % 2 == 0),
        )
        transforms.extend(
            [
                PiecewiseRationalQuadraticCouplingTransform(
                    mask=mask,
                    transform_net_create_fn=make_net,
                    num_bins=int(spline_num_bins),
                    tails="linear",
                    tail_bound=float(spline_tail_bound),
                    apply_unconditional_transform=False,
                ),
                LULinear(int(n_features), identity_init=True),
            ]
        )
    return Flow(
        transform=CompositeTransform(transforms),
        distribution=StandardNormal(shape=[int(n_features)]),
    )


def _torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _build_lu_flow_from_config(
    config: Mapping[str, Any], device: torch.device
) -> torch.nn.Module:
    if str(config.get("architecture", "")) != "lu_mixed_nsf_v2":
        raise ValueError(
            "This checkpoint is not an Exercise 10 LU-mixed v2 flow. "
            "Use the v2 run tag rather than an Exercise 9/v1 checkpoint."
        )
    return _build_lu_mixed_spline(
        n_features=int(config["n_features"]),
        context_features=int(config.get("context_features", 0)),
        n_coupling_layers=int(config["n_coupling_layers"]),
        hidden_features=int(config["hidden_features"]),
        hidden_layers=int(config["hidden_layers"]),
        spline_num_bins=int(config["spline_num_bins"]),
        spline_tail_bound=float(config["spline_tail_bound"]),
        dropout_probability=float(config.get("dropout_probability", 0.0)),
    ).to(device)


def load_lu_mixed_spline_flow(
    checkpoint: str | Path, device: torch.device
) -> dict[str, Any]:
    """Load a benchmark-only LU-mixed spline flow."""

    checkpoint = Path(checkpoint)
    saved = _torch_load(checkpoint, device)
    config = dict(saved["config"])
    flow = _build_lu_flow_from_config(config, device)
    flow.load_state_dict(saved["state_dict"])
    flow.eval()
    context_scaler = None
    if saved.get("context_mean") is not None:
        context_scaler = ArrayStandardizer(
            mean=np.asarray(saved["context_mean"], dtype=np.float32),
            std=np.asarray(saved["context_std"], dtype=np.float32),
        )
    return {
        "flow": flow,
        "target_scaler": ArrayStandardizer(
            mean=np.asarray(saved["target_mean"], dtype=np.float32),
            std=np.asarray(saved["target_std"], dtype=np.float32),
        ),
        "context_scaler": context_scaler,
        "config": config,
        "checkpoint": checkpoint,
        "history": saved.get("history", {}),
    }


def train_lu_mixed_spline_flow(
    target: np.ndarray,
    *,
    context: np.ndarray | None,
    checkpoint: str | Path,
    model_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    device: torch.device,
    seed: int,
    load_if_available: bool = True,
) -> dict[str, Any]:
    """Train the independent Exercise 10 LU-mixed spline architecture."""

    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if load_if_available and checkpoint.exists():
        print(f"Loading LU-mixed spline flow from {checkpoint}")
        return load_lu_mixed_spline_flow(checkpoint, device)

    target = _as_float32_2d(target, "target")
    if len(target) < 4:
        raise ValueError("At least four target rows are required.")
    if context is not None:
        context = _as_float32_2d(context, "context")
        if len(context) != len(target):
            raise ValueError("context must contain one row per target row.")

    config = dict(model_config)
    config.update(
        {
            "architecture": "lu_mixed_nsf_v2",
            "n_features": int(target.shape[1]),
            "context_features": 0 if context is None else int(context.shape[1]),
        }
    )
    target_scaler = ArrayStandardizer.fit(target)
    target_scaled = target_scaler.transform(target)
    context_scaler = None
    context_scaled = None
    if context is not None:
        context_scaler = ArrayStandardizer.fit(context)
        context_scaled = context_scaler.transform(context)

    validation_fraction = float(training_config.get("validation_fraction", 0.1))
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one.")
    n_validation = min(
        len(target_scaled) - 1,
        max(1, int(round(validation_fraction * len(target_scaled)))),
    )
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(len(target_scaled), generator=generator).numpy()
    validation_indices = order[:n_validation]
    training_indices = order[n_validation:]

    target_tensor = torch.tensor(target_scaled, dtype=torch.float32)
    if context_scaled is None:
        training_dataset = TensorDataset(target_tensor[training_indices])
        validation_dataset = TensorDataset(target_tensor[validation_indices])
    else:
        context_tensor = torch.tensor(context_scaled, dtype=torch.float32)
        training_dataset = TensorDataset(
            target_tensor[training_indices], context_tensor[training_indices]
        )
        validation_dataset = TensorDataset(
            target_tensor[validation_indices], context_tensor[validation_indices]
        )

    batch_size = int(training_config["batch_size"])
    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    seed_everything(seed)
    flow = _build_lu_flow_from_config(config, device)
    optimizer = torch.optim.AdamW(
        flow.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training_config.get("lr_scheduler_factor", 0.3)),
        patience=int(training_config.get("lr_scheduler_patience", 3)),
        min_lr=float(training_config.get("min_learning_rate", 1.0e-6)),
    )
    n_epochs = int(training_config["n_epochs"])
    patience = int(training_config.get("patience", n_epochs))
    gradient_clip = float(training_config.get("gradient_clip", 5.0))
    min_delta = float(training_config.get("min_delta", 1.0e-4))
    best_validation = math.inf
    best_state = None
    stale_epochs = 0
    history = {"train": [], "validation": [], "learning_rate": []}
    print(
        f"Training {'conditional' if context is not None else 'unconditional'} "
        f"LU-mixed spline flow on {len(training_dataset):,} rows"
    )

    for epoch in range(1, n_epochs + 1):
        flow.train()
        train_losses = []
        for batch in training_loader:
            target_batch = batch[0].to(device)
            context_batch = batch[1].to(device) if len(batch) == 2 else None
            loss = -flow.log_prob(target_batch, context=context_batch).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), gradient_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        flow.eval()
        validation_losses = []
        with torch.no_grad():
            for batch in validation_loader:
                target_batch = batch[0].to(device)
                context_batch = batch[1].to(device) if len(batch) == 2 else None
                loss = -flow.log_prob(target_batch, context=context_batch).mean()
                validation_losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(train_losses))
        validation_loss = float(np.mean(validation_losses))
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history["train"].append(train_loss)
        history["validation"].append(validation_loss)
        history["learning_rate"].append(learning_rate)
        scheduler.step(validation_loss)

        if validation_loss < best_validation - min_delta:
            best_validation = validation_loss
            best_state = copy.deepcopy(flow.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        print(
            f"  epoch {epoch:03d}/{n_epochs}: lr={learning_rate:.3e}, "
            f"train={train_loss:.4f}, validation={validation_loss:.4f}"
        )
        if stale_epochs >= patience:
            print(f"  early stopping after {epoch} epochs")
            break

    if best_state is not None:
        flow.load_state_dict(best_state)
    flow.eval()
    torch.save(
        {
            "state_dict": flow.state_dict(),
            "config": config,
            "target_mean": target_scaler.mean,
            "target_std": target_scaler.std,
            "context_mean": None if context_scaler is None else context_scaler.mean,
            "context_std": None if context_scaler is None else context_scaler.std,
            "history": history,
        },
        checkpoint,
    )
    print(f"Saved LU-mixed spline flow to {checkpoint}")
    return load_lu_mixed_spline_flow(checkpoint, device)


def budget_label(num_simulations: int) -> str:
    """Convert a paper simulation budget to the unicode labels in its CSV."""

    labels = {1_000: "10³", 10_000: "10⁴", 100_000: "10⁵"}
    try:
        return labels[int(num_simulations)]
    except KeyError as exc:
        raise ValueError("The paper budgets are 1,000, 10,000, and 100,000.") from exc


def load_paper_results(*, corrected_gaussian_mixture: bool = True) -> pd.DataFrame:
    """Load the official result tables maintained by ``sbi-benchmark``.

    ``sbibm`` v1.1 fixed the Gaussian Mixture simulator.  By default the
    corresponding official rerun replaces the superseded rows from the paper
    table, while the other nine tasks remain exactly those in the manuscript.
    """

    main = pd.read_csv(PAPER_RESULTS_URL)
    if not corrected_gaussian_mixture:
        return main
    rerun = pd.read_csv(GAUSSIAN_MIXTURE_RERUN_URL)
    return pd.concat(
        [main.loc[main["task"] != "gaussian_mixture"], rerun],
        ignore_index=True,
    )


def summarize_paper_results(
    results: pd.DataFrame,
    *,
    num_simulations: int,
    task: str | None = None,
) -> pd.DataFrame:
    """Average C2ST over the ten observations and attach a normal 95% CI."""

    selected = results.loc[results["num_simulations"] == budget_label(num_simulations)]
    if task is not None:
        selected = selected.loc[selected["task"] == task]
    selected = selected.loc[selected["algorithm"].isin(PAPER_ALGORITHMS)]
    summary = (
        selected.groupby(["task", "algorithm"], as_index=False)["C2ST"]
        .agg(mean="mean", std="std", count="count")
        .sort_values(["task", "mean"])
        .reset_index(drop=True)
    )
    summary["ci95"] = 1.96 * summary["std"] / np.sqrt(summary["count"])
    summary["num_simulations"] = int(num_simulations)
    summary["is_best_published"] = summary.groupby("task")["mean"].transform(
        lambda values: values == values.min()
    )
    return summary


def best_published_targets(
    results: pd.DataFrame, *, num_simulations: int
) -> pd.DataFrame:
    """Return the lowest published mean C2ST for every task."""

    summary = summarize_paper_results(results, num_simulations=num_simulations)
    return summary.loc[summary["is_best_published"]].reset_index(drop=True)


@dataclass
class SimulationBank:
    """A bank of unique simulator calls used by all hybrid estimators."""

    task_name: str
    theta: np.ndarray
    x: np.ndarray
    seed: int

    @property
    def num_simulations(self) -> int:
        return int(len(self.theta))


def _as_float32_2d(values: Any, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")
    if len(values) and not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values.")
    return values


def simulate_benchmark_bank(
    task: Any,
    num_simulations: int,
    *,
    seed: int,
    chunk_size: int = 20_000,
) -> SimulationBank:
    """Generate exactly ``num_simulations`` prior-predictive pairs."""

    num_simulations = int(num_simulations)
    if num_simulations < 4:
        raise ValueError("At least four simulations are required.")
    seed_everything(seed)
    prior = task.get_prior()
    simulator = task.get_simulator(max_calls=num_simulations)
    theta_chunks, x_chunks = [], []
    for start in range(0, num_simulations, int(chunk_size)):
        current = min(int(chunk_size), num_simulations - start)
        theta = prior(num_samples=current)
        x = simulator(theta)
        theta_chunks.append(theta.detach().cpu().numpy().astype(np.float32))
        x_chunks.append(task.flatten_data(x).detach().cpu().numpy().astype(np.float32))
    return SimulationBank(
        task_name=str(task.name),
        theta=np.concatenate(theta_chunks),
        x=np.concatenate(x_chunks),
        seed=int(seed),
    )


def load_or_simulate_bank(
    task: Any,
    num_simulations: int,
    *,
    cache_path: str | Path,
    seed: int,
    chunk_size: int = 20_000,
) -> SimulationBank:
    """Load an exact-budget bank or generate and cache it as compressed NumPy."""

    cache_path = Path(cache_path)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as saved:
            bank = SimulationBank(
                task_name=str(saved["task_name"].item()),
                theta=_as_float32_2d(saved["theta"], "theta"),
                x=_as_float32_2d(saved["x"], "x"),
                seed=int(saved["seed"].item()),
            )
        if bank.task_name != str(task.name):
            raise ValueError(f"Cached task {bank.task_name!r} does not match {task.name!r}.")
        if bank.num_simulations != int(num_simulations):
            raise ValueError("Cached bank has a different simulation budget.")
        print(f"Loaded {bank.num_simulations:,} simulations from {cache_path}")
        return bank
    bank = simulate_benchmark_bank(
        task,
        num_simulations,
        seed=seed,
        chunk_size=chunk_size,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        task_name=np.asarray(bank.task_name),
        theta=bank.theta,
        x=bank.x,
        seed=np.asarray(bank.seed),
    )
    print(f"Saved exactly {bank.num_simulations:,} simulations to {cache_path}")
    return bank


def split_simulation_bank(
    bank: SimulationBank,
    *,
    seed: int,
    fold: int = 0,
    n_folds: int = 5,
) -> dict[str, np.ndarray]:
    """Return one complementary split of a genuine K-fold cross-fit.

    The held-out fold trains the density-ratio correction; all other folds
    train the reference flow.  Across all K fits every simulation is used once
    for ratio estimation and K-1 times for flow estimation.  The same ``seed``
    must be used for every fold so that the partitions remain complementary.
    """

    n_folds = int(n_folds)
    fold = int(fold)
    if n_folds < 2:
        raise ValueError("n_folds must be at least two.")
    if n_folds > bank.num_simulations:
        raise ValueError("n_folds cannot exceed the number of simulations.")
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold must lie in [0, {n_folds}).")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(bank.num_simulations)
    partitions = [
        np.asarray(values, dtype=np.int64)
        for values in np.array_split(order, n_folds)
    ]
    ratio_indices = partitions[fold]
    flow_indices = np.concatenate(
        [values for index, values in enumerate(partitions) if index != fold]
    )
    if np.intersect1d(flow_indices, ratio_indices).size:
        raise RuntimeError("Cross-fit flow and ratio subsets unexpectedly overlap.")
    return {
        "flow_indices": flow_indices,
        "ratio_indices": ratio_indices,
        "theta_flow": bank.theta[flow_indices],
        "x_flow": bank.x[flow_indices],
        "theta_ratio": bank.theta[ratio_indices],
        "x_ratio": bank.x[ratio_indices],
    }


@dataclass(frozen=True)
class ParameterTransform:
    """Map constrained benchmark parameters to an unconstrained flow space."""

    kind: str
    low: np.ndarray | None = None
    high: np.ndarray | None = None
    clip: float = 1.0e-6

    def forward(self, theta: np.ndarray) -> np.ndarray:
        theta = _as_float32_2d(theta, "theta").astype(np.float64)
        if self.kind == "identity":
            return theta.astype(np.float32)
        if self.kind == "log":
            if np.any(theta <= 0.0):
                raise ValueError("Log-transformed parameters must be positive.")
            return np.log(theta).astype(np.float32)
        if self.kind == "bounded_logit":
            low = np.asarray(self.low, dtype=np.float64)
            high = np.asarray(self.high, dtype=np.float64)
            unit = (theta - low) / (high - low)
            unit = np.clip(unit, self.clip, 1.0 - self.clip)
            return (np.log(unit) - np.log1p(-unit)).astype(np.float32)
        raise ValueError(f"Unknown transform kind {self.kind!r}.")

    def inverse(self, latent: np.ndarray) -> np.ndarray:
        latent = _as_float32_2d(latent, "latent").astype(np.float64)
        if self.kind == "identity":
            return latent.astype(np.float32)
        if self.kind == "log":
            return np.exp(np.clip(latent, -80.0, 80.0)).astype(np.float32)
        if self.kind == "bounded_logit":
            low = np.asarray(self.low, dtype=np.float64)
            high = np.asarray(self.high, dtype=np.float64)
            unit = np.empty_like(latent)
            positive = latent >= 0.0
            unit[positive] = 1.0 / (1.0 + np.exp(-latent[positive]))
            exp_latent = np.exp(latent[~positive])
            unit[~positive] = exp_latent / (1.0 + exp_latent)
            unit = np.clip(unit, self.clip, 1.0 - self.clip)
            return (low + (high - low) * unit).astype(np.float32)
        raise ValueError(f"Unknown transform kind {self.kind!r}.")

    def log_abs_det_inverse(self, latent: np.ndarray) -> np.ndarray:
        """Return ``log |d theta / d latent|`` for every row."""

        latent = _as_float32_2d(latent, "latent").astype(np.float64)
        if self.kind == "identity":
            return np.zeros(len(latent), dtype=np.float64)
        if self.kind == "log":
            return latent.sum(axis=1)
        if self.kind == "bounded_logit":
            width = np.asarray(self.high, dtype=np.float64) - np.asarray(
                self.low, dtype=np.float64
            )
            per_dimension = (
                np.log(width)
                - np.logaddexp(0.0, -latent)
                - np.logaddexp(0.0, latent)
            )
            return per_dimension.sum(axis=1)
        raise ValueError(f"Unknown transform kind {self.kind!r}.")


def infer_parameter_transform(task: Any) -> ParameterTransform:
    """Infer identity, logarithmic, or bounded-logit coordinates from a task."""

    params = task.get_prior_params()
    if "low" in params and "high" in params:
        return ParameterTransform(
            "bounded_logit",
            low=params["low"].detach().cpu().numpy().astype(np.float64),
            high=params["high"].detach().cpu().numpy().astype(np.float64),
        )
    if str(task.name) in {"sir", "lotka_volterra"}:
        return ParameterTransform("log")
    return ParameterTransform("identity")


def _prior_log_prob_latent(
    task: Any, transform: ParameterTransform, latent: np.ndarray
) -> np.ndarray:
    theta = transform.inverse(latent)
    distribution = task.get_prior_dist()
    with torch.no_grad():
        log_probability = distribution.log_prob(torch.as_tensor(theta)).detach().cpu().numpy()
    return np.asarray(log_probability, dtype=np.float64) + transform.log_abs_det_inverse(
        latent
    )


def _sample_prior_latent(
    task: Any,
    transform: ParameterTransform,
    n_samples: int,
    *,
    seed: int,
) -> np.ndarray:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        theta = task.get_prior()(num_samples=int(n_samples)).detach().cpu().numpy()
    return transform.forward(theta)


def _sample_defensive_latent_matched(
    q_phi: Mapping[str, Any],
    task: Any,
    transform: ParameterTransform,
    context: np.ndarray,
    *,
    epsilon: float,
    seed: int,
) -> np.ndarray:
    seed_everything(seed)
    context = _as_float32_2d(context, "context")
    flow_draws = sample_spline_flow(q_phi, 1, context=context)[:, 0, :]
    prior_draws = _sample_prior_latent(
        task, transform, len(context), seed=int(seed) + 1
    )
    use_prior = np.random.default_rng(int(seed) + 2).random(len(context)) < epsilon
    flow_draws[use_prior] = prior_draws[use_prior]
    return flow_draws.astype(np.float32)


def _validation_score(pack: Mapping[str, Any]) -> float:
    values = np.asarray(pack.get("history", {}).get("validation", []), dtype=float)
    return float(values.min()) if len(values) else math.inf


def _train_ratio_ensemble(
    positive: np.ndarray,
    negative: np.ndarray,
    *,
    checkpoint_stem: Path,
    ensemble_size: int,
    model_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    device: torch.device,
    seed: int,
    load_if_available: bool,
    retrain_outliers: bool,
    paired_group_ids: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    packs = []
    checkpoints = []
    for member in range(int(ensemble_size)):
        checkpoint = checkpoint_stem.with_name(
            f"{checkpoint_stem.name}_member{member}.pt"
        )
        checkpoints.append(checkpoint)
        print(f"\nRatio ensemble member {member + 1}/{ensemble_size}")
        packs.append(
            train_ratio_classifier(
                positive,
                negative,
                checkpoint=checkpoint,
                model_config=model_config,
                training_config=training_config,
                device=device,
                seed=int(seed) + 100 * member,
                load_if_available=load_if_available,
                paired_group_ids=paired_group_ids,
            )
        )
    if retrain_outliers and len(packs) >= 3:
        scores = np.asarray([_validation_score(pack) for pack in packs])
        finite = np.isfinite(scores)
        if finite.any():
            median = float(np.median(scores[finite]))
            mad = float(np.median(np.abs(scores[finite] - median)))
            threshold = median + max(0.01, 4.0 * mad)
            for member in np.flatnonzero((scores > threshold) | ~finite):
                print(
                    f"Retraining validation-loss outlier member {member}: "
                    f"score={scores[member]:.5f}, threshold={threshold:.5f}"
                )
                packs[member] = train_ratio_classifier(
                    positive,
                    negative,
                    checkpoint=checkpoints[member],
                    model_config=model_config,
                    training_config=training_config,
                    device=device,
                    seed=int(seed) + 10_000 + 100 * int(member),
                    load_if_available=False,
                    paired_group_ids=paired_group_ids,
                )
    return packs


@dataclass
class HybridBenchmarkModel:
    """One flow/ratio split of the benchmark hybrid construction."""

    task: Any
    transform: ParameterTransform
    q_phi: Mapping[str, Any]
    r_p_ensemble: list[Mapping[str, Any]]
    q_eta: Mapping[str, Any] | None
    r_l_ensemble: list[Mapping[str, Any]] | None
    hnde_log_normalizer: Mapping[str, Any] | None
    defensive_epsilon: float
    fold: int
    n_folds: int
    split_seed: int
    num_simulations: int

    @property
    def has_hnde(self) -> bool:
        return (
            self.q_eta is not None
            and bool(self.r_l_ensemble)
            and self.hnde_log_normalizer is not None
        )


def train_hybrid_benchmark_model(
    task: Any,
    bank: SimulationBank,
    *,
    model_dir: str | Path,
    flow_model_config: Mapping[str, Any],
    flow_training_config: Mapping[str, Any],
    ratio_model_config: Mapping[str, Any],
    ratio_training_config: Mapping[str, Any],
    device: torch.device,
    seed: int,
    split_seed: int,
    ensemble_size: int = 4,
    defensive_epsilon: float = 0.02,
    fold: int = 0,
    n_folds: int = 5,
    train_hnde: bool = True,
    load_if_available: bool = True,
    retrain_outliers: bool = True,
    normalizer_contexts: int = 2_048,
    normalizer_reference_per_context: int = 64,
) -> HybridBenchmarkModel:
    """Train exact-budget hNPE and conditional-reference hNDE routes."""

    if bank.task_name != str(task.name):
        raise ValueError("The simulation bank belongs to a different task.")
    if not 0.0 < float(defensive_epsilon) < 1.0:
        raise ValueError("defensive_epsilon must be strictly between zero and one.")
    model_dir = Path(model_dir) / f"fold{int(fold)}"
    model_dir.mkdir(parents=True, exist_ok=True)
    split = split_simulation_bank(
        bank,
        seed=int(split_seed),
        fold=int(fold),
        n_folds=int(n_folds),
    )
    transform = infer_parameter_transform(task)
    z_flow = transform.forward(split["theta_flow"])
    z_ratio = transform.forward(split["theta_ratio"])

    q_phi = train_lu_mixed_spline_flow(
        z_flow,
        context=split["x_flow"],
        checkpoint=model_dir / "q_phi_conditional_posterior.pt",
        model_config=flow_model_config,
        training_config=flow_training_config,
        device=device,
        seed=int(seed) + 11,
        load_if_available=load_if_available,
    )
    z_negative = _sample_defensive_latent_matched(
        q_phi,
        task,
        transform,
        split["x_ratio"],
        epsilon=float(defensive_epsilon),
        seed=int(seed) + 21,
    )
    r_p_positive = np.column_stack([z_ratio, split["x_ratio"]])
    r_p_negative = np.column_stack([z_negative, split["x_ratio"]])
    r_p_ensemble = _train_ratio_ensemble(
        r_p_positive,
        r_p_negative,
        checkpoint_stem=model_dir / "r_p_posterior_residual",
        ensemble_size=ensemble_size,
        model_config=ratio_model_config,
        training_config=ratio_training_config,
        device=device,
        seed=int(seed) + 31,
        load_if_available=load_if_available,
        retrain_outliers=retrain_outliers,
        paired_group_ids=split["ratio_indices"],
    )

    q_eta = None
    r_l_ensemble = None
    hnde_log_normalizer = None
    if train_hnde:
        q_eta = train_lu_mixed_spline_flow(
            split["x_flow"],
            context=z_flow,
            checkpoint=model_dir / "q_eta_conditional_likelihood_reference.pt",
            model_config=flow_model_config,
            training_config=flow_training_config,
            device=device,
            seed=int(seed) + 41,
            load_if_available=load_if_available,
        )
        seed_everything(int(seed) + 51)
        x_negative = sample_spline_flow(
            q_eta,
            1,
            context=z_ratio,
        )[:, 0, :]
        r_l_positive = np.column_stack([z_ratio, split["x_ratio"]])
        r_l_negative = np.column_stack([z_ratio, x_negative])
        r_l_ensemble = _train_ratio_ensemble(
            r_l_positive,
            r_l_negative,
            checkpoint_stem=model_dir / "r_c_conditional_residual",
            ensemble_size=ensemble_size,
            model_config=ratio_model_config,
            training_config=ratio_training_config,
            device=device,
            seed=int(seed) + 61,
            load_if_available=load_if_available,
            retrain_outliers=retrain_outliers,
            paired_group_ids=split["ratio_indices"],
        )
        normalizer_context = _sample_prior_latent(
            task,
            transform,
            int(normalizer_contexts),
            seed=int(seed) + 71,
        )
        hnde_log_normalizer = train_conditional_log_normalizer(
            q_eta,
            r_l_ensemble,
            normalizer_context,
            checkpoint=model_dir / "r_c_conditional_log_normalizer.pt",
            device=device,
            seed=int(seed) + 72,
            n_reference=int(normalizer_reference_per_context),
            load_if_available=load_if_available,
        )

    return HybridBenchmarkModel(
        task=task,
        transform=transform,
        q_phi=q_phi,
        r_p_ensemble=r_p_ensemble,
        q_eta=q_eta,
        r_l_ensemble=r_l_ensemble,
        hnde_log_normalizer=hnde_log_normalizer,
        defensive_epsilon=float(defensive_epsilon),
        fold=int(fold),
        n_folds=int(n_folds),
        split_seed=int(split_seed),
        num_simulations=bank.num_simulations,
    )


@dataclass
class NPEBaselineModel:
    """All-data NSF baseline with the architecture used in the paper era."""

    task: Any
    transform: ParameterTransform
    q_phi: Mapping[str, Any]
    num_simulations: int


def train_official_style_nsf_baseline(
    task: Any,
    bank: SimulationBank,
    *,
    checkpoint: str | Path,
    model_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    device: torch.device,
    seed: int,
    load_if_available: bool = True,
) -> NPEBaselineModel:
    """Train a small, correctly mixed conditional NSF on the complete bank."""

    if bank.task_name != str(task.name):
        raise ValueError("The simulation bank belongs to a different task.")
    transform = infer_parameter_transform(task)
    q_phi = train_lu_mixed_spline_flow(
        transform.forward(bank.theta),
        context=bank.x,
        checkpoint=checkpoint,
        model_config=model_config,
        training_config=training_config,
        device=device,
        seed=int(seed),
        load_if_available=load_if_available,
    )
    return NPEBaselineModel(
        task=task,
        transform=transform,
        q_phi=q_phi,
        num_simulations=bank.num_simulations,
    )


def _normalized_weights(log_weights: np.ndarray) -> np.ndarray:
    log_weights = np.asarray(log_weights, dtype=np.float64)
    finite = np.isfinite(log_weights)
    if not finite.any():
        raise RuntimeError("All importance weights are non-finite.")
    log_weights = np.where(finite, log_weights, -np.inf)
    return np.exp(log_weights - logsumexp(log_weights))


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    return float(1.0 / np.sum(weights**2))


def _resample(
    values: np.ndarray, weights: np.ndarray, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    indices = rng.choice(
        len(values),
        size=int(n_samples),
        replace=True,
        p=np.asarray(weights, dtype=np.float64),
    )
    return values[indices]


@dataclass
class PosteriorDraws:
    """Unweighted posterior samples plus the proposals used to obtain them."""

    samples: dict[str, np.ndarray]
    proposals: np.ndarray
    weights: dict[str, np.ndarray]
    diagnostics: pd.DataFrame


def draw_hybrid_posterior(
    models: HybridBenchmarkModel | Sequence[HybridBenchmarkModel],
    observation: np.ndarray,
    *,
    baseline_model: NPEBaselineModel | None = None,
    n_proposal: int = 200_000,
    n_samples: int = 10_000,
    seed: int,
) -> PosteriorDraws:
    """Draw NPE, hNPE, hNDE, and predeclared dual-consensus posteriors.

    The dual consensus is the normalized geometric mean of the hNPE and hNDE
    importance weights.  If both routes are exact it is the same posterior;
    with finite networks it averages their log-density errors.  This rule is
    fixed before looking at benchmark reference samples.
    """

    if isinstance(models, HybridBenchmarkModel):
        models = [models]
    models = list(models)
    if not models:
        raise ValueError("At least one hybrid model is required.")
    if len({str(model.task.name) for model in models}) != 1:
        raise ValueError("All folds must belong to the same benchmark task.")
    if baseline_model is not None and str(baseline_model.task.name) != str(
        models[0].task.name
    ):
        raise ValueError("The baseline and hybrid folds must use the same task.")
    observation = _as_float32_2d(np.atleast_2d(observation), "observation")
    if len(observation) != 1:
        raise ValueError("Exactly one observation is required.")
    rng = np.random.default_rng(int(seed))
    proposal_chunks, npe_chunks = [], []
    log_weight_chunks: dict[str, list[np.ndarray]] = {"hNPE": []}
    if all(model.has_hnde for model in models):
        log_weight_chunks.update({"hNDE": [], "dual hNPE--hNDE": []})
    diagnostic_rows = []

    counts = np.full(len(models), int(n_proposal) // len(models), dtype=int)
    counts[: int(n_proposal) % len(models)] += 1
    for model_index, (model, count) in enumerate(zip(models, counts)):
        seed_everything(int(seed) + 1000 * model_index)
        z_npe = sample_spline_flow(model.q_phi, int(count), context=observation)
        npe_chunks.append(model.transform.inverse(z_npe))
        z_prior = _sample_prior_latent(
            model.task,
            model.transform,
            int(count),
            seed=int(seed) + 1000 * model_index + 1,
        )
        use_prior = rng.random(int(count)) < model.defensive_epsilon
        z_proposal = z_npe.copy()
        z_proposal[use_prior] = z_prior[use_prior]
        theta_proposal = model.transform.inverse(z_proposal)
        x_repeated = np.repeat(observation, int(count), axis=0)
        inputs = np.column_stack([z_proposal, x_repeated])
        log_r_p = ratio_classifier_ensemble_logit(model.r_p_ensemble, inputs)

        log_q_phi = spline_flow_log_prob(
            model.q_phi, z_proposal, context=x_repeated
        )
        log_prior_z = _prior_log_prob_latent(
            model.task, model.transform, z_proposal
        )
        log_q_defensive = np.logaddexp(
            np.log1p(-model.defensive_epsilon) + log_q_phi,
            np.log(model.defensive_epsilon) + log_prior_z,
        )
        proposal_chunks.append(theta_proposal)
        log_weight_chunks["hNPE"].append(log_r_p)
        row = {
            "fold": model.fold,
            "proposal_rows": int(count),
            "E_q_rP": float(np.exp(logsumexp(log_r_p) - np.log(count))),
        }
        row.update(
            {
                f"fold_{name}_hNPE": value
                for name, value in importance_tail_summary(log_r_p).items()
            }
        )

        if "hNDE" in log_weight_chunks:
            log_r_l = conditional_residual_log_ratio(
                model.r_l_ensemble,
                model.hnde_log_normalizer,
                z_proposal,
                x_repeated,
            )
            log_q_eta = spline_flow_log_prob(
                model.q_eta,
                x_repeated,
                context=z_proposal,
            )
            log_weight_l = (
                log_prior_z
                + log_q_eta
                + log_r_l
                - log_q_defensive
            )
            log_weight_chunks["hNDE"].append(log_weight_l)
            # Normalize before combining because the two ratios differ by an
            # observation-dependent constant.
            log_p_normalized = log_r_p - logsumexp(log_r_p)
            log_l_normalized = log_weight_l - logsumexp(log_weight_l)
            log_weight_dual = 0.5 * (
                log_p_normalized + log_l_normalized
            )
            log_weight_chunks["dual hNPE--hNDE"].append(log_weight_dual)
            row.update(
                {
                    f"fold_{name}_hNDE": value
                    for name, value in importance_tail_summary(
                        log_weight_l
                    ).items()
                }
            )
            row.update(
                {
                    f"fold_{name}_dual": value
                    for name, value in importance_tail_summary(
                        log_weight_dual
                    ).items()
                }
            )
            row["hNPE_hNDE_log_weight_rms"] = float(
                np.sqrt(
                    np.mean(
                        (
                            log_p_normalized
                            - log_l_normalized
                            - np.mean(log_p_normalized - log_l_normalized)
                        )
                        ** 2
                    )
                )
            )
        diagnostic_rows.append(row)

    proposals = np.concatenate(proposal_chunks, axis=0)
    npe_candidates = np.concatenate(npe_chunks, axis=0)
    samples = {
        "NPE reference": npe_candidates[
            rng.choice(len(npe_candidates), int(n_samples), replace=False)
        ]
    }
    if baseline_model is not None:
        seed_everything(int(seed) + 900_000)
        z_baseline = sample_spline_flow(
            baseline_model.q_phi,
            int(n_samples),
            context=observation,
        )
        samples["all-data NSF baseline"] = baseline_model.transform.inverse(
            z_baseline
        )
    combined_weights = {}
    for method, chunks in log_weight_chunks.items():
        # Every fold is an independently normalized importance estimate of the
        # same posterior and receives equal mixture weight.
        fold_weights = [
            _normalized_weights(chunk) / len(chunks) for chunk in chunks
        ]
        weights = np.concatenate(fold_weights)
        weights = weights / weights.sum()
        combined_weights[method] = weights
        samples[method] = _resample(proposals, weights, int(n_samples), rng)
    diagnostics = pd.DataFrame(diagnostic_rows)
    for method, weights in combined_weights.items():
        global_ess = effective_sample_size(weights)
        diagnostics[f"global_ESS_{method}"] = global_ess
        diagnostics[f"global_ESS_fraction_{method}"] = global_ess / len(weights)
    return PosteriorDraws(
        samples=samples,
        proposals=proposals,
        weights=combined_weights,
        diagnostics=diagnostics,
    )


def evaluate_posterior_samples(
    task: Any,
    num_observation: int,
    samples: Mapping[str, np.ndarray],
    *,
    seed: int = 1,
) -> pd.DataFrame:
    """Compute the paper's z-scored, five-fold C2ST against 10k references."""

    from sbibm.metrics import c2st

    reference = task.get_reference_posterior_samples(
        num_observation=int(num_observation)
    ).detach().cpu().float()
    rows = []
    for method, values in samples.items():
        candidate = torch.as_tensor(
            _as_float32_2d(values, method), dtype=torch.float32
        )
        if len(candidate) != len(reference):
            raise ValueError(
                f"{method} has {len(candidate):,} samples; the paper comparison "
                f"requires {len(reference):,}."
            )
        score = float(
            c2st(
                reference,
                candidate,
                seed=int(seed),
                n_folds=5,
                z_score=True,
            ).item()
        )
        rows.append(
            {
                "task": str(task.name),
                "num_observation": int(num_observation),
                "algorithm": method,
                "C2ST": score,
            }
        )
    return pd.DataFrame(rows)


def evaluate_observation_suite(
    models: HybridBenchmarkModel | Sequence[HybridBenchmarkModel],
    *,
    baseline_model: NPEBaselineModel | None = None,
    observations: Iterable[int] = range(1, 11),
    n_proposal: int = 200_000,
    n_samples: int = 10_000,
    seed: int = 1,
    keep_draws: bool = False,
) -> tuple[pd.DataFrame, dict[int, PosteriorDraws]]:
    """Evaluate a pre-trained amortized model on several official observations.

    Full proposal arrays can be large.  They are discarded after each metric by
    default; ``keep_draws=True`` is useful for detailed diagnostics on a short
    observation list.
    """

    if isinstance(models, HybridBenchmarkModel):
        model_list = [models]
    else:
        model_list = list(models)
    task = model_list[0].task
    result_frames = []
    draws_by_observation = {}
    for num_observation in observations:
        print(f"\nEvaluating official observation {num_observation}")
        observation = task.get_observation(num_observation=int(num_observation))
        draws = draw_hybrid_posterior(
            model_list,
            observation.detach().cpu().numpy(),
            baseline_model=baseline_model,
            n_proposal=n_proposal,
            n_samples=n_samples,
            seed=int(seed) + 10_000 * int(num_observation),
        )
        metrics = evaluate_posterior_samples(
            task,
            int(num_observation),
            draws.samples,
            seed=int(seed) + int(num_observation),
        )
        metrics["num_simulations"] = model_list[0].num_simulations
        for method, weights in draws.weights.items():
            tail = importance_tail_summary(
                np.log(np.maximum(weights, np.finfo(float).tiny))
            )
            selection = metrics["algorithm"] == method
            metrics.loc[selection, "ESS"] = tail["ESS"]
            metrics.loc[selection, "ESS_fraction"] = tail["ESS_fraction"]
            metrics.loc[selection, "max_weight_fraction"] = tail[
                "max_weight_fraction"
            ]
            metrics.loc[selection, "pareto_k"] = tail["pareto_k"]
        result_frames.append(metrics)
        if keep_draws:
            draws_by_observation[int(num_observation)] = draws
    return pd.concat(result_frames, ignore_index=True), draws_by_observation


def crossfit_coverage_table(
    bank: SimulationBank,
    *,
    n_folds: int,
    split_seed: int,
) -> pd.DataFrame:
    """Verify that every row has exactly the intended K-fold role counts."""

    ratio_counts = np.zeros(bank.num_simulations, dtype=np.int16)
    flow_counts = np.zeros(bank.num_simulations, dtype=np.int16)
    rows = []
    for fold in range(int(n_folds)):
        split = split_simulation_bank(
            bank,
            seed=int(split_seed),
            fold=fold,
            n_folds=int(n_folds),
        )
        ratio_counts[split["ratio_indices"]] += 1
        flow_counts[split["flow_indices"]] += 1
        rows.append(
            {
                "fold": fold,
                "flow_rows": len(split["flow_indices"]),
                "ratio_rows": len(split["ratio_indices"]),
                "overlap_rows": int(
                    np.intersect1d(
                        split["flow_indices"], split["ratio_indices"]
                    ).size
                ),
            }
        )
    if not np.all(ratio_counts == 1):
        raise RuntimeError("Cross-fitting did not use every row exactly once for ratios.")
    if not np.all(flow_counts == int(n_folds) - 1):
        raise RuntimeError(
            "Cross-fitting did not use every row K-1 times for flow training."
        )
    table = pd.DataFrame(rows)
    table["ratio_coverage_min"] = int(ratio_counts.min())
    table["ratio_coverage_max"] = int(ratio_counts.max())
    table["flow_coverage_min"] = int(flow_counts.min())
    table["flow_coverage_max"] = int(flow_counts.max())
    return table


def training_diagnostics(
    models: HybridBenchmarkModel | Sequence[HybridBenchmarkModel],
    bank: SimulationBank,
    *,
    max_rows_per_fold: int = 10_000,
    seed: int = 1,
) -> pd.DataFrame:
    """Evaluate pre-benchmark flow and normalization diagnostics.

    Only simulation-bank pairs and neural validation histories are used.  No
    reference posterior sample or C2ST result enters these diagnostics.
    """

    if isinstance(models, HybridBenchmarkModel):
        models = [models]
    rows = []
    for model_index, model in enumerate(models):
        split = split_simulation_bank(
            bank,
            seed=model.split_seed,
            fold=model.fold,
            n_folds=model.n_folds,
        )
        rng = np.random.default_rng(int(seed) + 1000 * model_index)
        n_check = min(int(max_rows_per_fold), len(split["ratio_indices"]))
        chosen = rng.choice(len(split["ratio_indices"]), n_check, replace=False)
        theta = split["theta_ratio"][chosen]
        x = split["x_ratio"][chosen]
        z = model.transform.forward(theta)
        q_phi_nll = -float(
            np.mean(spline_flow_log_prob(model.q_phi, z, context=x))
        )
        z_negative = _sample_defensive_latent_matched(
            model.q_phi,
            model.task,
            model.transform,
            x,
            epsilon=model.defensive_epsilon,
            seed=int(seed) + 10_000 + model_index,
        )
        log_r_p = ratio_classifier_ensemble_logit(
            model.r_p_ensemble,
            np.column_stack([z_negative, x]),
        )
        row = {
            "fold": model.fold,
            "flow_rows": len(split["flow_indices"]),
            "ratio_rows": len(split["ratio_indices"]),
            "q_phi_heldout_nll": q_phi_nll,
            "rP_validation_split": ",".join(
                sorted(
                    {
                        str(
                            pack.get("history", {}).get(
                                "split_strategy", "unknown"
                            )
                        )
                        for pack in model.r_p_ensemble
                    }
                )
            ),
            "rP_validation_bce": float(
                np.mean([_validation_score(pack) for pack in model.r_p_ensemble])
            ),
            "E_q_rP": float(np.exp(logsumexp(log_r_p) - np.log(n_check))),
        }
        if model.has_hnde:
            row["q_eta_heldout_nll"] = -float(
                np.mean(
                    spline_flow_log_prob(
                        model.q_eta,
                        x,
                        context=z,
                    )
                )
            )
            seed_everything(int(seed) + 20_000 + model_index)
            x_reference = sample_spline_flow(
                model.q_eta,
                1,
                context=z,
            )[:, 0, :]
            c2st = conditional_flow_c2st(
                z,
                x,
                x_reference,
                seed=int(seed) + 21_000 + model_index,
            )
            row.update({f"q_eta_{key}": value for key, value in c2st.items()})
            raw_log_r_l = ratio_classifier_ensemble_logit(
                model.r_l_ensemble,
                np.column_stack([z, x_reference]),
            )
            log_r_l = raw_log_r_l - conditional_log_normalizer(
                model.hnde_log_normalizer,
                z,
            )
            row["rC_validation_split"] = ",".join(
                sorted(
                    {
                        str(
                            pack.get("history", {}).get(
                                "split_strategy", "unknown"
                            )
                        )
                        for pack in model.r_l_ensemble
                    }
                )
            )
            row["rL_validation_bce"] = float(
                np.mean([_validation_score(pack) for pack in model.r_l_ensemble])
            )
            row["E_qeta_rC_corrected_mixture"] = float(
                np.exp(logsumexp(log_r_l) - np.log(len(log_r_l)))
            )
            n_normalization_context = min(128, len(z))
            normalization_indices = rng.choice(
                len(z),
                n_normalization_context,
                replace=False,
            )
            normalization = conditional_normalization_diagnostics(
                model.q_eta,
                model.r_l_ensemble,
                model.hnde_log_normalizer,
                z[normalization_indices],
                n_reference=64,
                seed=int(seed) + 22_000 + model_index,
            )
            row["raw_Z_mean"] = float(normalization["raw_Z"].mean())
            row["raw_Z_max_abs_error"] = float(
                np.max(np.abs(normalization["raw_Z"] - 1.0))
            )
            row["corrected_Z_mean"] = float(
                normalization["corrected_Z"].mean()
            )
            row["corrected_Z_max_abs_error"] = float(
                np.max(np.abs(normalization["corrected_Z"] - 1.0))
            )
        rows.append(row)
    return pd.DataFrame(rows)


def benchmark_ratio_tail_diagnostics(
    models: HybridBenchmarkModel | Sequence[HybridBenchmarkModel],
    bank: SimulationBank,
    *,
    max_rows_per_fold: int = 10_000,
    seed: int = 1,
) -> pd.DataFrame:
    """Report member-wise ratio tails on denominator-distribution draws."""

    if isinstance(models, HybridBenchmarkModel):
        models = [models]
    rows = []
    for model_index, model in enumerate(models):
        split = split_simulation_bank(
            bank,
            seed=model.split_seed,
            fold=model.fold,
            n_folds=model.n_folds,
        )
        rng = np.random.default_rng(int(seed) + 1000 * model_index)
        n_check = min(int(max_rows_per_fold), len(split["ratio_indices"]))
        chosen = rng.choice(len(split["ratio_indices"]), n_check, replace=False)
        x = split["x_ratio"][chosen]
        z = model.transform.forward(split["theta_ratio"][chosen])
        z_negative = _sample_defensive_latent_matched(
            model.q_phi,
            model.task,
            model.transform,
            x,
            epsilon=model.defensive_epsilon,
            seed=int(seed) + 10_000 + model_index,
        )
        rows.append(
            ratio_member_tail_diagnostics(
                model.r_p_ensemble,
                np.column_stack([z_negative, x]),
                path="hNPE posterior residual",
                fold=model.fold,
            )
        )
        if model.has_hnde:
            seed_everything(int(seed) + 20_000 + model_index)
            x_reference = sample_spline_flow(
                model.q_eta,
                1,
                context=z,
            )[:, 0, :]
            rows.append(
                ratio_member_tail_diagnostics(
                    model.r_l_ensemble,
                    np.column_stack([z, x_reference]),
                    path="conditional hNDE residual",
                    fold=model.fold,
                )
            )
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def simulation_based_calibration_diagnostics(
    models: HybridBenchmarkModel | Sequence[HybridBenchmarkModel],
    bank: SimulationBank,
    *,
    n_cases_per_fold: int = 32,
    n_proposal: int = 512,
    seed: int = 1,
) -> pd.DataFrame:
    """Compute lightweight cross-fit SBC ranks for every hybrid route."""

    if isinstance(models, HybridBenchmarkModel):
        models = [models]
    models = list(models)
    rank_chunks: dict[str, list[np.ndarray]] = {
        "NPE reference": [],
        "hNPE": [],
    }
    if all(model.has_hnde for model in models):
        rank_chunks.update({"hNDE": [], "dual hNPE--hNDE": []})

    for model_index, model in enumerate(models):
        split = split_simulation_bank(
            bank,
            seed=model.split_seed,
            fold=model.fold,
            n_folds=model.n_folds,
        )
        rng = np.random.default_rng(int(seed) + 1000 * model_index)
        n_cases = min(int(n_cases_per_fold), len(split["ratio_indices"]))
        chosen = rng.choice(len(split["ratio_indices"]), n_cases, replace=False)
        z_true = model.transform.forward(split["theta_ratio"][chosen])
        x_cases = split["x_ratio"][chosen]
        seed_everything(int(seed) + 2000 * model_index)
        z_npe = sample_spline_flow(
            model.q_phi,
            int(n_proposal),
            context=x_cases,
        )
        if z_npe.ndim == 2:
            z_npe = z_npe[None, :, :]
        z_prior = _sample_prior_latent(
            model.task,
            model.transform,
            n_cases * int(n_proposal),
            seed=int(seed) + 2000 * model_index + 1,
        ).reshape(n_cases, int(n_proposal), -1)
        use_prior = (
            rng.random((n_cases, int(n_proposal))) < model.defensive_epsilon
        )
        z_proposal = np.where(use_prior[:, :, None], z_prior, z_npe)
        z_flat = z_proposal.reshape(-1, z_proposal.shape[-1])
        x_flat = np.repeat(x_cases, int(n_proposal), axis=0)
        inputs = np.column_stack([z_flat, x_flat])
        log_r_p = ratio_classifier_ensemble_logit(
            model.r_p_ensemble,
            inputs,
        ).reshape(n_cases, int(n_proposal))
        weights_p = np.exp(log_r_p - logsumexp(log_r_p, axis=1, keepdims=True))
        rank_chunks["NPE reference"].append(
            np.mean(z_npe <= z_true[:, None, :], axis=1)
        )
        rank_chunks["hNPE"].append(
            np.sum(
                weights_p[:, :, None]
                * (z_proposal <= z_true[:, None, :]),
                axis=1,
            )
        )

        if "hNDE" in rank_chunks:
            log_q_phi = spline_flow_log_prob(
                model.q_phi,
                z_flat,
                context=x_flat,
            ).reshape(n_cases, int(n_proposal))
            log_prior = _prior_log_prob_latent(
                model.task,
                model.transform,
                z_flat,
            ).reshape(n_cases, int(n_proposal))
            log_q_defensive = np.logaddexp(
                np.log1p(-model.defensive_epsilon) + log_q_phi,
                np.log(model.defensive_epsilon) + log_prior,
            )
            log_r_l = conditional_residual_log_ratio(
                model.r_l_ensemble,
                model.hnde_log_normalizer,
                z_flat,
                x_flat,
            ).reshape(n_cases, int(n_proposal))
            log_q_eta = spline_flow_log_prob(
                model.q_eta,
                x_flat,
                context=z_flat,
            ).reshape(n_cases, int(n_proposal))
            log_weights_l = (
                log_prior + log_q_eta + log_r_l - log_q_defensive
            )
            weights_l = np.exp(
                log_weights_l
                - logsumexp(log_weights_l, axis=1, keepdims=True)
            )
            rank_chunks["hNDE"].append(
                np.sum(
                    weights_l[:, :, None]
                    * (z_proposal <= z_true[:, None, :]),
                    axis=1,
                )
            )
            log_weights_dual = 0.5 * (
                log_r_p
                - logsumexp(log_r_p, axis=1, keepdims=True)
                + log_weights_l
                - logsumexp(log_weights_l, axis=1, keepdims=True)
            )
            weights_dual = np.exp(
                log_weights_dual
                - logsumexp(log_weights_dual, axis=1, keepdims=True)
            )
            rank_chunks["dual hNPE--hNDE"].append(
                np.sum(
                    weights_dual[:, :, None]
                    * (z_proposal <= z_true[:, None, :]),
                    axis=1,
                )
            )

    rows = []
    for method, chunks in rank_chunks.items():
        ranks = np.concatenate(chunks, axis=0)
        for dimension in range(ranks.shape[1]):
            statistic, p_value = kstest(ranks[:, dimension], "uniform")
            rows.append(
                {
                    "algorithm": method,
                    "dimension": dimension,
                    "num_ranks": len(ranks),
                    "rank_mean": float(np.mean(ranks[:, dimension])),
                    "rank_variance": float(np.var(ranks[:, dimension])),
                    "KS_uniform": float(statistic),
                    "KS_pvalue": float(p_value),
                }
            )
    return pd.DataFrame(rows)


def posterior_predictive_diagnostics(
    task: Any,
    posterior_samples: np.ndarray,
    observation: np.ndarray,
    *,
    n_simulations: int = 2_000,
    seed: int = 1,
) -> pd.DataFrame:
    """Run an optional posterior-predictive check with explicitly extra calls.

    These simulator calls are diagnostic and must not be counted as part of
    the official 100k training budget or used to tune a reported benchmark
    after C2ST/reference samples have been examined.
    """

    posterior_samples = _as_float32_2d(
        posterior_samples, "posterior_samples"
    )
    observation = _as_float32_2d(np.atleast_2d(observation), "observation")
    if len(observation) != 1:
        raise ValueError("Exactly one observation is required.")
    n_simulations = min(int(n_simulations), len(posterior_samples))
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(len(posterior_samples), n_simulations, replace=False)
    seed_everything(seed)
    simulator = task.get_simulator(max_calls=n_simulations)
    x_predictive = simulator(
        torch.as_tensor(posterior_samples[chosen], dtype=torch.float32)
    )
    x_predictive = (
        task.flatten_data(x_predictive).detach().cpu().numpy().astype(np.float64)
    )
    center = x_predictive.mean(axis=0)
    scale = x_predictive.std(axis=0, ddof=1)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    z_residual = (observation[0] - center) / scale
    rows = [
        {
            "dimension": dimension,
            "observed": float(observation[0, dimension]),
            "predictive_mean": float(center[dimension]),
            "predictive_std": float(scale[dimension]),
            "z_residual": float(z_residual[dimension]),
            "extra_simulator_calls": n_simulations,
        }
        for dimension in range(len(center))
    ]
    return pd.DataFrame(rows)


def summarize_our_results(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize Exercise 10 C2ST results using the paper convention."""

    summary = (
        results.groupby(["task", "num_simulations", "algorithm"], as_index=False)[
            "C2ST"
        ]
        .agg(mean="mean", std="std", count="count")
        .sort_values(["task", "mean"])
        .reset_index(drop=True)
    )
    summary["ci95"] = 1.96 * summary["std"] / np.sqrt(summary["count"])
    return summary


def compare_with_paper(
    our_results: pd.DataFrame,
    paper_results: pd.DataFrame,
    *,
    num_simulations: int,
    headline_method: str = "dual hNPE--hNDE",
) -> pd.DataFrame:
    """Compare a predeclared method with the best published mean C2ST."""

    ours = summarize_our_results(our_results)
    ours = ours.loc[
        (ours["num_simulations"] == int(num_simulations))
        & (ours["algorithm"] == headline_method)
    ].copy()
    targets = best_published_targets(
        paper_results, num_simulations=int(num_simulations)
    ).rename(
        columns={
            "algorithm": "best_published_algorithm",
            "mean": "best_published_mean",
            "ci95": "best_published_ci95",
        }
    )
    comparison = ours.merge(
        targets[
            [
                "task",
                "best_published_algorithm",
                "best_published_mean",
                "best_published_ci95",
            ]
        ],
        on="task",
        how="left",
    )
    comparison["delta_C2ST"] = (
        comparison["mean"] - comparison["best_published_mean"]
    )
    comparison["beats_published_mean"] = comparison["delta_C2ST"] < 0.0
    return comparison
