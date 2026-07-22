"""Hybrid hNPE--hNDE helpers for the ``sbibm`` benchmark exercise.

The benchmark in Lueckmann et al. counts calls to the simulator.  This module
therefore keeps simulation, model training, posterior construction, and metric
evaluation separate.  A single cached simulation bank can be shared by the
posterior and likelihood routes without hiding additional simulator calls.

The neural building blocks are imported from :mod:`utils_hnpe`, so Exercise 10
uses exactly the quadratic-spline flows, raw ratio ensembles, and learning-rate
machinery introduced in Exercise 9.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp

from utils_hnpe import (
    ratio_classifier_ensemble_logit,
    sample_spline_flow,
    spline_flow_log_prob,
    train_ratio_classifier,
    train_spline_flow,
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
        "Julia backend required",
        "The official simulator uses diffeqtorch/Julia; hNDE would need a "
        "discrete or dequantized reference model.",
    ),
    TaskRecommendation(
        "lotka_volterra",
        "Lotka--Volterra",
        4,
        20,
        "count time series",
        "hNPE",
        "Julia backend required",
        "The official simulator uses diffeqtorch/Julia and is costly.",
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
    flow_fraction: float = 0.5,
    seed: int,
    fold: int = 0,
) -> dict[str, np.ndarray]:
    """Split a bank into statistically separate flow and ratio subsets.

    ``fold=1`` swaps the two subsets.  Training both folds is cross-fitting: it
    doubles neural training but performs no additional simulations.
    """

    if not 0.0 < float(flow_fraction) < 1.0:
        raise ValueError("flow_fraction must be strictly between zero and one.")
    if int(fold) not in (0, 1):
        raise ValueError("fold must be zero or one.")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(bank.num_simulations)
    n_flow = int(round(float(flow_fraction) * bank.num_simulations))
    n_flow = min(bank.num_simulations - 2, max(2, n_flow))
    flow_indices, ratio_indices = order[:n_flow], order[n_flow:]
    if int(fold) == 1:
        flow_indices, ratio_indices = ratio_indices, flow_indices
    return {
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
    defensive_epsilon: float
    fold: int
    num_simulations: int

    @property
    def has_hnde(self) -> bool:
        return self.q_eta is not None and bool(self.r_l_ensemble)


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
    ensemble_size: int = 4,
    defensive_epsilon: float = 0.02,
    flow_fraction: float = 0.5,
    fold: int = 0,
    train_hnde: bool = True,
    load_if_available: bool = True,
    retrain_outliers: bool = True,
) -> HybridBenchmarkModel:
    """Train an exact-budget hNPE and, optionally, its dual hNDE route."""

    if bank.task_name != str(task.name):
        raise ValueError("The simulation bank belongs to a different task.")
    if not 0.0 < float(defensive_epsilon) < 1.0:
        raise ValueError("defensive_epsilon must be strictly between zero and one.")
    model_dir = Path(model_dir) / f"fold{int(fold)}"
    model_dir.mkdir(parents=True, exist_ok=True)
    split = split_simulation_bank(
        bank,
        flow_fraction=flow_fraction,
        seed=int(seed),
        fold=int(fold),
    )
    transform = infer_parameter_transform(task)
    z_flow = transform.forward(split["theta_flow"])
    z_ratio = transform.forward(split["theta_ratio"])

    q_phi = train_spline_flow(
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
    )

    q_eta = None
    r_l_ensemble = None
    if train_hnde:
        q_eta = train_spline_flow(
            split["x_flow"],
            context=None,
            checkpoint=model_dir / "q_eta_observation_reference.pt",
            model_config=flow_model_config,
            training_config=flow_training_config,
            device=device,
            seed=int(seed) + 41,
            load_if_available=load_if_available,
        )
        seed_everything(int(seed) + 51)
        x_negative = sample_spline_flow(q_eta, len(z_ratio))
        r_l_positive = np.column_stack([z_ratio, split["x_ratio"]])
        r_l_negative = np.column_stack([z_ratio, x_negative])
        r_l_ensemble = _train_ratio_ensemble(
            r_l_positive,
            r_l_negative,
            checkpoint_stem=model_dir / "r_l_likelihood",
            ensemble_size=ensemble_size,
            model_config=ratio_model_config,
            training_config=ratio_training_config,
            device=device,
            seed=int(seed) + 61,
            load_if_available=load_if_available,
            retrain_outliers=retrain_outliers,
        )

    return HybridBenchmarkModel(
        task=task,
        transform=transform,
        q_phi=q_phi,
        r_p_ensemble=r_p_ensemble,
        q_eta=q_eta,
        r_l_ensemble=r_l_ensemble,
        defensive_epsilon=float(defensive_epsilon),
        fold=int(fold),
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

        if "hNDE" in log_weight_chunks:
            log_r_l = ratio_classifier_ensemble_logit(model.r_l_ensemble, inputs)
            log_weight_l = log_prior_z + log_r_l - log_q_defensive
            log_weight_chunks["hNDE"].append(log_weight_l)
            # Normalize before combining because the two ratios differ by an
            # observation-dependent constant.
            log_p_normalized = log_r_p - logsumexp(log_r_p)
            log_l_normalized = log_weight_l - logsumexp(log_weight_l)
            log_weight_chunks["dual hNPE--hNDE"].append(
                0.5 * (log_p_normalized + log_l_normalized)
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
        diagnostics[f"ESS_{method}"] = effective_sample_size(weights)
        diagnostics[f"ESS_fraction_{method}"] = effective_sample_size(weights) / len(
            weights
        )
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
        for method in draws.weights:
            metrics.loc[
                metrics["algorithm"] == method, "ESS"
            ] = effective_sample_size(draws.weights[method])
        result_frames.append(metrics)
        if keep_draws:
            draws_by_observation[int(num_observation)] = draws
    return pd.concat(result_frames, ignore_index=True), draws_by_observation


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
