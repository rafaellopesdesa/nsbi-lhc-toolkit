"""Conditional-reference hNDE helpers for Exercises 9 and 10.

This module is deliberately separate from the standard hNDE workflow used by
Exercises 5--8.  It implements the likelihood-side member of the dual
hNPE--hNDE construction,

    p(x | theta) = q_eta(x | theta) c(x, theta) / Z_C(theta),

where the conditional flow supplies a normalized approximate likelihood and a
paired classifier learns only the residual correction.  The utilities below
also provide the diagnostics introduced after the SBIBM study: honest grouped
validation, conditional-flow C2ST, parameter-dependent normalization checks,
member-wise ratio-tail inspection, and a Pareto-k importance-tail estimate.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import logsumexp
from scipy.stats import binomtest, genpareto
from torch.utils.data import DataLoader, TensorDataset

from utils_hnpe import (
    ArrayStandardizer,
    ratio_classifier_ensemble_logit,
    ratio_classifier_logit,
    sample_spline_flow,
)


def _as_2d_float32(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")
    if len(values) and not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values.")
    return values


def _torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


class _LogNormalizerMLP(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_features: int,
        hidden_layers: int,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        n_input = int(n_features)
        for _ in range(int(hidden_layers)):
            layers.extend(
                [
                    nn.Linear(n_input, int(hidden_features)),
                    nn.SiLU(),
                ]
            )
            n_input = int(hidden_features)
        layers.append(nn.Linear(n_input, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def _build_log_normalizer(
    config: Mapping[str, Any], device: torch.device
) -> nn.Module:
    return _LogNormalizerMLP(
        n_features=int(config["n_features"]),
        hidden_features=int(config.get("hidden_features", 64)),
        hidden_layers=int(config.get("hidden_layers", 2)),
    ).to(device)


def load_conditional_log_normalizer(
    checkpoint: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    """Load a parameter-dependent residual partition-function regressor."""

    checkpoint = Path(checkpoint)
    saved = _torch_load(checkpoint, device)
    config = dict(saved["config"])
    model = _build_log_normalizer(config, device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return {
        "model": model,
        "context_scaler": ArrayStandardizer(
            mean=np.asarray(saved["context_mean"], dtype=np.float32),
            std=np.asarray(saved["context_std"], dtype=np.float32),
        ),
        "target_mean": float(saved["target_mean"]),
        "target_std": float(saved["target_std"]),
        "config": config,
        "checkpoint": checkpoint,
        "history": saved.get("history", {}),
    }


@torch.no_grad()
def conditional_log_normalizer(
    pack: Mapping[str, Any],
    context: np.ndarray,
    *,
    batch_size: int = 65_536,
) -> np.ndarray:
    """Predict ``log Z_C(theta)`` in the original normalization coordinates."""

    context = _as_2d_float32(context, "context")
    model = pack["model"]
    device = next(model.parameters()).device
    model.eval()
    chunks = []
    for start in range(0, len(context), int(batch_size)):
        stop = start + int(batch_size)
        tensor = torch.tensor(
            pack["context_scaler"].transform(context[start:stop]),
            dtype=torch.float32,
            device=device,
        )
        prediction = model(tensor).detach().cpu().numpy()
        chunks.append(
            prediction * float(pack["target_std"]) + float(pack["target_mean"])
        )
    if not chunks:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(chunks).astype(np.float64)


def estimate_conditional_log_normalization(
    q_eta: Mapping[str, Any],
    ratio_packs: Sequence[Mapping[str, Any]],
    context: np.ndarray,
    *,
    n_reference: int = 64,
    seed: int = 1,
    context_chunk_size: int = 256,
) -> np.ndarray:
    """Estimate raw ``log E_q[c]`` at many parameter points.

    No simulator calls are made.  Each context receives independent samples
    from the frozen conditional reference flow.
    """

    context = _as_2d_float32(context, "context")
    n_reference = int(n_reference)
    if n_reference < 2:
        raise ValueError("n_reference must be at least two.")
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    targets = []
    for start in range(0, len(context), int(context_chunk_size)):
        context_chunk = context[start : start + int(context_chunk_size)]
        draws = sample_spline_flow(
            q_eta,
            n_reference,
            context=context_chunk,
        )
        if draws.ndim == 2:
            draws = draws[None, :, :]
        repeated_context = np.repeat(context_chunk, n_reference, axis=0)
        inputs = np.column_stack(
            [repeated_context, draws.reshape(-1, draws.shape[-1])]
        )
        log_ratio = ratio_classifier_ensemble_logit(
            list(ratio_packs),
            inputs,
        ).reshape(len(context_chunk), n_reference)
        targets.append(logsumexp(log_ratio, axis=1) - np.log(n_reference))
    if not targets:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(targets).astype(np.float64)


def train_conditional_log_normalizer(
    q_eta: Mapping[str, Any],
    ratio_packs: Sequence[Mapping[str, Any]],
    context: np.ndarray,
    *,
    checkpoint: str | Path,
    device: torch.device,
    seed: int,
    n_reference: int = 64,
    model_config: Mapping[str, Any] | None = None,
    training_config: Mapping[str, Any] | None = None,
    load_if_available: bool = True,
) -> dict[str, Any]:
    """Fit the smooth parameter-dependent correction ``log Z_C(theta)``."""

    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if load_if_available and checkpoint.exists():
        print(f"Loading conditional log normalizer from {checkpoint}")
        return load_conditional_log_normalizer(checkpoint, device)

    context = _as_2d_float32(context, "context")
    if len(context) < 16:
        raise ValueError("At least sixteen normalization contexts are required.")
    log_targets = estimate_conditional_log_normalization(
        q_eta,
        ratio_packs,
        context,
        n_reference=n_reference,
        seed=seed,
    ).astype(np.float32)
    model_config = dict(model_config or {})
    model_config.setdefault("hidden_features", 64)
    model_config.setdefault("hidden_layers", 2)
    model_config["n_features"] = int(context.shape[1])
    training_config = dict(training_config or {})
    validation_fraction = float(training_config.get("validation_fraction", 0.2))
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one.")
    batch_size = int(training_config.get("batch_size", 128))
    n_epochs = int(training_config.get("n_epochs", 300))
    learning_rate = float(training_config.get("learning_rate", 2.0e-3))
    weight_decay = float(training_config.get("weight_decay", 1.0e-4))
    patience = int(training_config.get("patience", 30))

    rng = np.random.default_rng(int(seed) + 1)
    order = rng.permutation(len(context))
    n_validation = min(
        len(context) - 1,
        max(1, int(round(validation_fraction * len(context)))),
    )
    validation_indices = order[:n_validation]
    training_indices = order[n_validation:]
    context_scaler = ArrayStandardizer.fit(context[training_indices])
    context_scaled = context_scaler.transform(context)
    target_mean = float(
        np.mean(log_targets[training_indices], dtype=np.float64)
    )
    target_std = max(
        float(np.std(log_targets[training_indices], dtype=np.float64)),
        1.0e-3,
    )
    targets_scaled = ((log_targets - target_mean) / target_std).astype(np.float32)
    context_tensor = torch.tensor(context_scaled, dtype=torch.float32)
    target_tensor = torch.tensor(targets_scaled, dtype=torch.float32)
    generator = torch.Generator().manual_seed(int(seed) + 2)
    training_loader = DataLoader(
        TensorDataset(
            context_tensor[training_indices],
            target_tensor[training_indices],
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        TensorDataset(
            context_tensor[validation_indices],
            target_tensor[validation_indices],
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    torch.manual_seed(int(seed) + 3)
    model = _build_log_normalizer(model_config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    best_validation = math.inf
    best_state = None
    stale_epochs = 0
    history = {
        "train": [],
        "validation": [],
        "raw_log_z_mean": target_mean,
        "raw_log_z_std": float(np.std(log_targets, dtype=np.float64)),
        "n_context": int(len(context)),
        "n_reference_per_context": int(n_reference),
    }
    print(
        "Training conditional log normalizer on "
        f"{len(training_indices):,} parameter points "
        f"({n_reference} reference draws each)"
    )
    for epoch in range(1, n_epochs + 1):
        model.train()
        train_losses = []
        for context_batch, target_batch in training_loader:
            context_batch = context_batch.to(device)
            target_batch = target_batch.to(device)
            loss = F.mse_loss(model(context_batch), target_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        validation_losses = []
        with torch.no_grad():
            for context_batch, target_batch in validation_loader:
                context_batch = context_batch.to(device)
                target_batch = target_batch.to(device)
                validation_losses.append(
                    float(
                        F.mse_loss(
                            model(context_batch), target_batch
                        ).detach().cpu()
                    )
                )
        train_loss = float(np.mean(train_losses))
        validation_loss = float(np.mean(validation_losses))
        history["train"].append(train_loss)
        history["validation"].append(validation_loss)
        if validation_loss < best_validation - 1.0e-5:
            best_validation = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 25 == 0:
            print(
                f"  epoch {epoch:03d}/{n_epochs}: "
                f"train={train_loss:.5f}, validation={validation_loss:.5f}"
            )
        if stale_epochs >= patience:
            print(f"  early stopping after {epoch} epochs")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": model_config,
            "context_mean": context_scaler.mean,
            "context_std": context_scaler.std,
            "target_mean": target_mean,
            "target_std": target_std,
            "history": history,
        },
        checkpoint,
    )
    print(f"Saved conditional log normalizer to {checkpoint}")
    return load_conditional_log_normalizer(checkpoint, device)


def conditional_residual_log_ratio(
    ratio_packs: Sequence[Mapping[str, Any]],
    normalizer: Mapping[str, Any],
    context: np.ndarray,
    observation: np.ndarray,
) -> np.ndarray:
    """Evaluate ``log c(x,theta) - log Z_C(theta)``."""

    context = _as_2d_float32(context, "context")
    observation = _as_2d_float32(observation, "observation")
    if len(context) != len(observation):
        raise ValueError("context and observation must have equal row counts.")
    raw = ratio_classifier_ensemble_logit(
        list(ratio_packs),
        np.column_stack([context, observation]),
    )
    return raw - conditional_log_normalizer(normalizer, context)


def conditional_normalization_diagnostics(
    q_eta: Mapping[str, Any],
    ratio_packs: Sequence[Mapping[str, Any]],
    normalizer: Mapping[str, Any],
    context: np.ndarray,
    *,
    n_reference: int = 128,
    seed: int = 1,
) -> pd.DataFrame:
    """Return raw and corrected normalization at held-out parameter points."""

    context = _as_2d_float32(context, "context")
    raw_log_z = estimate_conditional_log_normalization(
        q_eta,
        ratio_packs,
        context,
        n_reference=n_reference,
        seed=seed,
    )
    modeled_log_z = conditional_log_normalizer(normalizer, context)
    return pd.DataFrame(
        {
            "context_index": np.arange(len(context), dtype=int),
            "raw_log_Z": raw_log_z,
            "modeled_log_Z": modeled_log_z,
            "corrected_log_Z": raw_log_z - modeled_log_z,
            "raw_Z": np.exp(np.clip(raw_log_z, -50.0, 50.0)),
            "corrected_Z": np.exp(
                np.clip(raw_log_z - modeled_log_z, -50.0, 50.0)
            ),
        }
    )


def conditional_flow_c2st(
    context: np.ndarray,
    simulator_observation: np.ndarray,
    reference_observation: np.ndarray,
    *,
    seed: int = 1,
    test_fraction: float = 0.3,
) -> dict[str, float]:
    """Classifier two-sample test with matched contexts kept in one split."""

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score

    context = _as_2d_float32(context, "context")
    simulator_observation = _as_2d_float32(
        simulator_observation, "simulator_observation"
    )
    reference_observation = _as_2d_float32(
        reference_observation, "reference_observation"
    )
    if not (
        len(context)
        == len(simulator_observation)
        == len(reference_observation)
    ):
        raise ValueError("C2ST arrays must contain the same number of rows.")
    if len(context) < 20:
        raise ValueError("At least twenty paired rows are required for C2ST.")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(context))
    n_test = min(
        len(context) - 1,
        max(1, int(round(float(test_fraction) * len(context)))),
    )
    test_groups = order[:n_test]
    train_groups = order[n_test:]

    def rows(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        positive = np.column_stack(
            [context[groups], simulator_observation[groups]]
        )
        negative = np.column_stack(
            [context[groups], reference_observation[groups]]
        )
        values = np.concatenate([positive, negative], axis=0)
        labels = np.concatenate(
            [np.ones(len(groups)), np.zeros(len(groups))]
        ).astype(int)
        shuffle = rng.permutation(len(values))
        return values[shuffle], labels[shuffle]

    x_train, y_train = rows(train_groups)
    x_test, y_test = rows(test_groups)
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=150,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=int(seed),
    )
    classifier.fit(x_train, y_train)
    probability = classifier.predict_proba(x_test)[:, 1]
    prediction = probability >= 0.5
    accuracy = float(accuracy_score(y_test, prediction))
    auc = float(roc_auc_score(y_test, probability))
    symmetric_auc = max(auc, 1.0 - auc)
    correct = int(np.sum(prediction == y_test))
    p_value = float(
        binomtest(correct, n=len(y_test), p=0.5, alternative="greater").pvalue
    )
    return {
        "c2st_accuracy": accuracy,
        "c2st_auc": auc,
        "c2st_symmetric_auc": symmetric_auc,
        "c2st_binomial_pvalue": p_value,
        "c2st_test_rows": int(len(y_test)),
    }


def ratio_member_tail_diagnostics(
    ratio_packs: Sequence[Mapping[str, Any]],
    values: np.ndarray,
    *,
    path: str,
    fold: int | None = None,
) -> pd.DataFrame:
    """Inspect member tails and domination of arithmetic ratio averaging."""

    values = _as_2d_float32(values, "values")
    member_logits = np.stack(
        [
            ratio_classifier_logit(
                pack,
                values,
                calibrated=False,
            )
            for pack in ratio_packs
        ],
        axis=0,
    )
    member_log_share = member_logits - logsumexp(
        member_logits, axis=0, keepdims=True
    )
    dominant = np.argmax(member_logits, axis=0)
    rows = []
    for member, logits in enumerate(member_logits):
        rows.append(
            {
                "fold": fold,
                "path": str(path),
                "member": int(member),
                "validation_bce": float(
                    np.min(
                        np.asarray(
                            ratio_packs[member]
                            .get("history", {})
                            .get("validation", [np.nan]),
                            dtype=float,
                        )
                    )
                ),
                "log_ratio_q50": float(np.quantile(logits, 0.50)),
                "log_ratio_q95": float(np.quantile(logits, 0.95)),
                "log_ratio_q99": float(np.quantile(logits, 0.99)),
                "log_ratio_q999": float(np.quantile(logits, 0.999)),
                "log_ratio_max": float(np.max(logits)),
                "mean_arithmetic_share": float(
                    np.mean(np.exp(member_log_share[member]))
                ),
                "dominant_fraction": float(np.mean(dominant == member)),
            }
        )
    return pd.DataFrame(rows)


def pareto_k_from_log_weights(log_weights: np.ndarray) -> float:
    """Estimate the generalized-Pareto shape of the largest raw weights.

    This is a lightweight diagnostic rather than a PSIS replacement.  Values
    below about 0.5 are usually benign, 0.5--0.7 deserve attention, and values
    above 0.7 indicate an unstable importance tail.
    """

    log_weights = np.asarray(log_weights, dtype=np.float64).ravel()
    log_weights = log_weights[np.isfinite(log_weights)]
    if len(log_weights) < 20:
        return math.nan
    weights = np.exp(log_weights - np.max(log_weights))
    tail_length = min(
        max(20, int(3.0 * np.sqrt(len(weights)))),
        max(20, len(weights) // 5),
    )
    if tail_length >= len(weights):
        tail_length = len(weights) - 1
    ordered = np.sort(weights)
    threshold = ordered[-tail_length - 1]
    excess = ordered[-tail_length:] - threshold
    if np.ptp(excess) <= np.finfo(float).eps:
        return 0.0
    try:
        shape, _, _ = genpareto.fit(excess, floc=0.0)
    except Exception:
        return math.nan
    return float(shape)


def importance_tail_summary(log_weights: np.ndarray) -> dict[str, float]:
    """Return ESS, maximum normalized weight, and Pareto-k."""

    log_weights = np.asarray(log_weights, dtype=np.float64).ravel()
    finite = np.isfinite(log_weights)
    if not finite.any():
        return {
            "ESS": 0.0,
            "ESS_fraction": 0.0,
            "max_weight_fraction": math.nan,
            "pareto_k": math.nan,
        }
    safe = np.where(finite, log_weights, -np.inf)
    weights = np.exp(safe - logsumexp(safe))
    ess = float(1.0 / np.sum(weights**2))
    return {
        "ESS": ess,
        "ESS_fraction": ess / len(weights),
        "max_weight_fraction": float(np.max(weights)),
        "pareto_k": pareto_k_from_log_weights(safe),
    }
