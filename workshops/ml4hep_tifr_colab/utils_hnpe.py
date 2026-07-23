"""Small hNPE/hNDE training helpers for Exercise 9.

The two density estimators in the exercise use the same rational-quadratic
spline coupling construction as ``utils_nf.py``.  This module adds the one
feature that the earlier exercises did not need: conditional density
estimation for an NPE, together with a compact neural density-ratio estimator.

The functions intentionally accept and return NumPy arrays.  Keeping the
PyTorch and ``nflows`` details here makes the Bayesian identities in the
notebook easier to read without hiding any statistical step.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class ArrayStandardizer:
    """Column-wise affine standardization for two-dimensional arrays."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "ArrayStandardizer":
        values = _as_2d_float32(values, "values")
        mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = values.std(axis=0, dtype=np.float64).astype(np.float32)
        std = np.where(std > 1.0e-6, std, 1.0).astype(np.float32)
        return cls(mean=mean, std=std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = _as_2d_float32(values, "values")
        return ((values - self.mean) / self.std).astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        values = _as_2d_float32(values, "values")
        return (values * self.std + self.mean).astype(np.float32)

    @property
    def log_det_to_standard(self) -> float:
        return float(-np.log(self.std).sum())


def _as_2d_float32(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")
    if len(values) and not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values.")
    return values


def _build_quadratic_spline(
    *,
    n_features: int,
    context_features: int,
    n_coupling_layers: int,
    hidden_features: int,
    hidden_layers: int,
    spline_num_bins: int,
    spline_tail_bound: float,
    dropout_probability: float,
) -> nn.Module:
    """Build the workshop rational-quadratic spline coupling flow."""
    try:
        from nflows.distributions.normal import StandardNormal
        from nflows.flows.base import Flow
        from nflows.nn.nets import ResidualNet
        from nflows.transforms.base import CompositeTransform
        from nflows.transforms.coupling import (
            PiecewiseRationalQuadraticCouplingTransform,
        )
        from nflows.transforms.permutations import ReversePermutation
        from nflows.utils.torchutils import create_alternating_binary_mask
    except ImportError as exc:
        raise ImportError(
            "Exercise 9 requires nflows. Install it with `pip install nflows`."
        ) from exc

    def make_net(in_features: int, out_features: int) -> nn.Module:
        return ResidualNet(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            context_features=(context_features or None),
            num_blocks=hidden_layers,
            activation=F.relu,
            dropout_probability=dropout_probability,
            use_batch_norm=False,
        )

    transforms = []
    for layer_index in range(n_coupling_layers):
        mask = create_alternating_binary_mask(
            features=n_features,
            even=(layer_index % 2 == 0),
        )
        transforms.append(
            PiecewiseRationalQuadraticCouplingTransform(
                mask=mask,
                transform_net_create_fn=make_net,
                num_bins=spline_num_bins,
                tails="linear",
                tail_bound=spline_tail_bound,
                apply_unconditional_transform=False,
            )
        )
        if layer_index + 1 < n_coupling_layers:
            transforms.append(ReversePermutation(features=n_features))

    return Flow(
        transform=CompositeTransform(transforms),
        distribution=StandardNormal(shape=[n_features]),
    )


def _flow_checkpoint(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _build_flow_from_config(
    config: Mapping[str, Any], device: torch.device
) -> nn.Module:
    flow = _build_quadratic_spline(
        n_features=int(config["n_features"]),
        context_features=int(config.get("context_features", 0)),
        n_coupling_layers=int(config["n_coupling_layers"]),
        hidden_features=int(config["hidden_features"]),
        hidden_layers=int(config["hidden_layers"]),
        spline_num_bins=int(config["spline_num_bins"]),
        spline_tail_bound=float(config["spline_tail_bound"]),
        dropout_probability=float(config.get("dropout_probability", 0.0)),
    )
    return flow.to(device)


def load_spline_flow(
    checkpoint: str | Path, device: torch.device
) -> dict[str, Any]:
    """Load a flow trained by :func:`train_spline_flow`."""
    checkpoint = Path(checkpoint)
    saved = _torch_load(checkpoint, device)
    config = dict(saved["config"])
    flow = _build_flow_from_config(config, device)
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


def train_spline_flow(
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
    """Train an unconditional or conditional quadratic-spline flow."""
    checkpoint = _flow_checkpoint(checkpoint)
    if load_if_available and checkpoint.exists():
        print(f"Loading spline flow from {checkpoint}")
        return load_spline_flow(checkpoint, device)

    target = _as_2d_float32(target, "target")
    if len(target) < 4:
        raise ValueError("At least four target rows are required.")
    if context is not None:
        context = _as_2d_float32(context, "context")
        if len(context) != len(target):
            raise ValueError("context must contain one row per target row.")

    config = dict(model_config)
    config["n_features"] = int(target.shape[1])
    config["context_features"] = 0 if context is None else int(context.shape[1])
    target_scaler = ArrayStandardizer.fit(target)
    target_scaled = target_scaler.transform(target)
    context_scaler = None
    if context is not None:
        context_scaler = ArrayStandardizer.fit(context)
        context_scaled = context_scaler.transform(context)

    validation_fraction = float(training_config.get("validation_fraction", 0.2))
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
    if context is None:
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

    flow = _build_flow_from_config(config, device)
    optimizer = torch.optim.AdamW(
        flow.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training_config.get("lr_scheduler_factor", 0.3)),
        patience=int(training_config.get("lr_scheduler_patience", 2)),
        min_lr=float(training_config.get("min_learning_rate", 1.0e-6)),
    )
    n_epochs = int(training_config["n_epochs"])
    patience = int(training_config.get("patience", n_epochs))
    gradient_clip = float(training_config.get("gradient_clip", 5.0))
    best_validation = math.inf
    best_state = None
    stale_epochs = 0
    history = {"train": [], "validation": []}

    print(
        f"Training {'conditional' if context is not None else 'unconditional'} "
        f"quadratic-spline flow on {len(training_dataset):,} rows"
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
        history["train"].append(train_loss)
        history["validation"].append(validation_loss)
        scheduler.step(validation_loss)

        if validation_loss < best_validation - 1.0e-4:
            best_validation = validation_loss
            best_state = copy.deepcopy(flow.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        print(
            f"  epoch {epoch:02d}/{n_epochs}: train={train_loss:.4f}, "
            f"validation={validation_loss:.4f}"
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
    print(f"Saved spline flow to {checkpoint}")
    return load_spline_flow(checkpoint, device)


@torch.no_grad()
def spline_flow_log_prob(
    flow_pack: Mapping[str, Any],
    target: np.ndarray,
    *,
    context: np.ndarray | None = None,
    batch_size: int = 65_536,
) -> np.ndarray:
    """Evaluate a flow density in the original target coordinates."""
    target = _as_2d_float32(target, "target")
    conditional = flow_pack["context_scaler"] is not None
    if conditional:
        if context is None:
            raise ValueError("This flow requires context.")
        context = _as_2d_float32(context, "context")
        if len(context) != len(target):
            raise ValueError("context must contain one row per target row.")
    elif context is not None:
        raise ValueError("This flow is unconditional; context must be None.")

    flow = flow_pack["flow"]
    device = next(flow.parameters()).device
    target_scaler = flow_pack["target_scaler"]
    context_scaler = flow_pack["context_scaler"]
    flow.eval()
    chunks = []
    for start in range(0, len(target), int(batch_size)):
        stop = start + int(batch_size)
        target_tensor = torch.tensor(
            target_scaler.transform(target[start:stop]),
            dtype=torch.float32,
            device=device,
        )
        context_tensor = None
        if conditional:
            context_tensor = torch.tensor(
                context_scaler.transform(context[start:stop]),
                dtype=torch.float32,
                device=device,
            )
        log_probability = flow.log_prob(
            target_tensor, context=context_tensor
        ).detach().cpu().numpy()
        chunks.append(log_probability + target_scaler.log_det_to_standard)
    if not chunks:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(chunks)


@torch.no_grad()
def sample_spline_flow(
    flow_pack: Mapping[str, Any],
    n_samples: int,
    *,
    context: np.ndarray | None = None,
    batch_size: int = 16_384,
) -> np.ndarray:
    """Draw samples in original coordinates.

    For an unconditional flow the result has shape ``(n_samples, d)``.  For a
    conditional flow, a one-row context also returns ``(n_samples, d)``.  With
    multiple context rows, ``n_samples`` samples are drawn per row and the
    result has shape ``(n_context, n_samples, d)``.
    """
    n_samples = int(n_samples)
    if n_samples < 1:
        raise ValueError("n_samples must be positive.")
    flow = flow_pack["flow"]
    device = next(flow.parameters()).device
    target_scaler = flow_pack["target_scaler"]
    context_scaler = flow_pack["context_scaler"]
    flow.eval()

    if context_scaler is None:
        if context is not None:
            raise ValueError("This flow is unconditional; context must be None.")
        chunks = []
        remaining = n_samples
        while remaining:
            current = min(int(batch_size), remaining)
            values = flow.sample(current).detach().cpu().numpy()
            chunks.append(target_scaler.inverse(values))
            remaining -= current
        return np.concatenate(chunks, axis=0)

    if context is None:
        raise ValueError("This flow requires context.")
    context = _as_2d_float32(context, "context")
    if len(context) == 1:
        context_scaled = context_scaler.transform(context)
        context_tensor = torch.tensor(
            context_scaled, dtype=torch.float32, device=device
        )
        chunks = []
        remaining = n_samples
        while remaining:
            current = min(int(batch_size), remaining)
            values = flow.sample(current, context=context_tensor)
            values = values[0].detach().cpu().numpy()
            chunks.append(target_scaler.inverse(values))
            remaining -= current
        return np.concatenate(chunks, axis=0)

    # Many contexts are the matched-pair use case.  Batch context rows rather
    # than the total number of returned values.
    context_batch_size = max(1, int(batch_size) // n_samples)
    chunks = []
    for start in range(0, len(context), context_batch_size):
        context_tensor = torch.tensor(
            context_scaler.transform(context[start : start + context_batch_size]),
            dtype=torch.float32,
            device=device,
        )
        values = flow.sample(n_samples, context=context_tensor)
        shape = values.shape
        values = values.detach().cpu().numpy().reshape(-1, shape[-1])
        values = target_scaler.inverse(values).reshape(
            shape[0], shape[1], shape[2]
        )
        chunks.append(values)
    return np.concatenate(chunks, axis=0)


class _RatioMLP(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_features: int,
        hidden_layers: int,
        dropout_probability: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        n_input = n_features
        for _ in range(hidden_layers):
            layers.extend(
                [
                    nn.Linear(n_input, hidden_features),
                    nn.SiLU(),
                    nn.Dropout(dropout_probability),
                ]
            )
            n_input = hidden_features
        layers.append(nn.Linear(n_input, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def _build_ratio_model(
    config: Mapping[str, Any], device: torch.device
) -> nn.Module:
    return _RatioMLP(
        n_features=int(config["n_features"]),
        hidden_features=int(config["hidden_features"]),
        hidden_layers=int(config["hidden_layers"]),
        dropout_probability=float(config.get("dropout_probability", 0.0)),
    ).to(device)


def load_ratio_classifier(
    checkpoint: str | Path, device: torch.device
) -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    saved = _torch_load(checkpoint, device)
    config = dict(saved["config"])
    model = _build_ratio_model(config, device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return {
        "model": model,
        "scaler": ArrayStandardizer(
            mean=np.asarray(saved["mean"], dtype=np.float32),
            std=np.asarray(saved["std"], dtype=np.float32),
        ),
        "config": config,
        "calibration_slope": float(saved.get("calibration_slope", 1.0)),
        "calibration_intercept": float(saved.get("calibration_intercept", 0.0)),
        "checkpoint": checkpoint,
        "history": saved.get("history", {}),
    }


def _save_ratio_classifier(pack: Mapping[str, Any]) -> None:
    torch.save(
        {
            "state_dict": pack["model"].state_dict(),
            "config": dict(pack["config"]),
            "mean": pack["scaler"].mean,
            "std": pack["scaler"].std,
            "calibration_slope": float(pack.get("calibration_slope", 1.0)),
            "calibration_intercept": float(
                pack.get("calibration_intercept", 0.0)
            ),
            "history": pack.get("history", {}),
        },
        Path(pack["checkpoint"]),
    )


def train_ratio_classifier(
    positive: np.ndarray,
    negative: np.ndarray,
    *,
    checkpoint: str | Path,
    model_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    device: torch.device,
    seed: int,
    load_if_available: bool = True,
    paired_group_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    """Train a balanced neural classifier whose logit estimates log p/q.

    ``paired_group_ids`` is used by the dual hNPE--hNDE exercises.  Each
    positive/negative row pair is generated from one simulator index and must
    remain on the same side of the train/validation boundary.  Splitting the
    concatenated class rows independently leaks the shared parameter or
    observation into validation and can make a memorizing network appear to
    generalize.
    """
    checkpoint = _flow_checkpoint(checkpoint)
    if load_if_available and checkpoint.exists():
        print(f"Loading ratio classifier from {checkpoint}")
        return load_ratio_classifier(checkpoint, device)
    positive = _as_2d_float32(positive, "positive")
    negative = _as_2d_float32(negative, "negative")
    if positive.shape[1] != negative.shape[1]:
        raise ValueError("positive and negative must have the same columns.")
    if paired_group_ids is not None:
        paired_group_ids = np.asarray(paired_group_ids)
        if len(positive) != len(negative):
            raise ValueError(
                "paired_group_ids requires one negative row per positive row."
            )
        if paired_group_ids.ndim != 1 or len(paired_group_ids) != len(positive):
            raise ValueError(
                "paired_group_ids must contain one one-dimensional id per "
                "positive/negative row pair."
            )
        n_per_class = len(positive)
    else:
        n_per_class = min(len(positive), len(negative))
    if n_per_class < 4:
        raise ValueError("At least four rows per class are required.")
    rng = np.random.default_rng(seed)
    if paired_group_ids is None:
        positive = positive[rng.choice(len(positive), n_per_class, replace=False)]
        negative = negative[rng.choice(len(negative), n_per_class, replace=False)]
    values = np.concatenate([positive, negative], axis=0)
    labels = np.concatenate(
        [np.ones(n_per_class), np.zeros(n_per_class)]
    ).astype(np.float32)

    validation_fraction = float(training_config.get("validation_fraction", 0.2))
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one.")
    split_strategy = "independent_rows"
    n_training_groups = None
    n_validation_groups = None
    if paired_group_ids is None:
        order = rng.permutation(len(values))
        n_validation = min(
            len(values) - 1,
            max(2, int(round(validation_fraction * len(values)))),
        )
        validation_indices = order[:n_validation]
        training_indices = order[n_validation:]
    else:
        unique_groups = np.unique(paired_group_ids)
        if len(unique_groups) < 2:
            raise ValueError("At least two distinct paired groups are required.")
        group_order = unique_groups[rng.permutation(len(unique_groups))]
        n_validation_groups = min(
            len(group_order) - 1,
            max(1, int(round(validation_fraction * len(group_order)))),
        )
        validation_groups = group_order[:n_validation_groups]
        training_groups = group_order[n_validation_groups:]
        row_groups = np.concatenate([paired_group_ids, paired_group_ids])
        validation_mask = np.isin(row_groups, validation_groups)
        training_mask = np.isin(row_groups, training_groups)
        if np.any(validation_mask & training_mask):
            raise RuntimeError("Paired group split unexpectedly overlaps.")
        if not np.all(validation_mask | training_mask):
            raise RuntimeError("Paired group split lost classifier rows.")
        validation_indices = np.flatnonzero(validation_mask)
        training_indices = np.flatnonzero(training_mask)
        split_strategy = "paired_groups"
        n_training_groups = int(len(training_groups))
        n_validation_groups = int(len(validation_groups))
    # Fit preprocessing on training rows only.  This is a much smaller effect
    # than paired-row leakage, but keeps the validation boundary fully honest.
    scaler = ArrayStandardizer.fit(values[training_indices])
    values = scaler.transform(values)
    values_tensor = torch.tensor(values, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    training_dataset = TensorDataset(
        values_tensor[training_indices], labels_tensor[training_indices]
    )
    validation_dataset = TensorDataset(
        values_tensor[validation_indices], labels_tensor[validation_indices]
    )
    generator = torch.Generator().manual_seed(int(seed))
    training_loader = DataLoader(
        training_dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=False,
    )

    config = dict(model_config)
    config["n_features"] = int(values.shape[1])
    model = _build_ratio_model(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    scheduler_name = str(
        training_config.get("lr_scheduler", "plateau")
    ).lower()
    if scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(training_config.get("lr_scheduler_patience", 10)),
            gamma=float(training_config.get("lr_scheduler_factor", 0.01)),
        )
        scheduler_uses_metric = False
    elif scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(training_config.get("lr_scheduler_factor", 0.3)),
            patience=int(training_config.get("lr_scheduler_patience", 2)),
            min_lr=float(training_config.get("min_learning_rate", 1.0e-6)),
        )
        scheduler_uses_metric = True
    else:
        raise ValueError("lr_scheduler must be 'step' or 'plateau'.")
    n_epochs = int(training_config["n_epochs"])
    patience = int(training_config.get("patience", n_epochs))
    gradient_clip = float(training_config.get("gradient_clip", 5.0))
    best_validation = math.inf
    best_state = None
    stale_epochs = 0
    history = {
        "train": [],
        "validation": [],
        "learning_rate": [],
        "split_strategy": split_strategy,
        "n_training_groups": n_training_groups,
        "n_validation_groups": n_validation_groups,
    }
    print(
        f"Training balanced ratio classifier on {n_per_class:,} rows per class "
        f"with {split_strategy} validation"
    )

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_losses = []
        for values_batch, labels_batch in training_loader:
            values_batch = values_batch.to(device)
            labels_batch = labels_batch.to(device)
            loss = F.binary_cross_entropy_with_logits(
                model(values_batch), labels_batch
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        validation_losses = []
        with torch.no_grad():
            for values_batch, labels_batch in validation_loader:
                values_batch = values_batch.to(device)
                labels_batch = labels_batch.to(device)
                loss = F.binary_cross_entropy_with_logits(
                    model(values_batch), labels_batch
                )
                validation_losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(train_losses))
        validation_loss = float(np.mean(validation_losses))
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history["train"].append(train_loss)
        history["validation"].append(validation_loss)
        history["learning_rate"].append(learning_rate)
        if scheduler_uses_metric:
            scheduler.step(validation_loss)
        else:
            scheduler.step()
        if validation_loss < best_validation - 1.0e-5:
            best_validation = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        print(
            f"  epoch {epoch:02d}/{n_epochs}: lr={learning_rate:.3e}, "
            f"train={train_loss:.4f}, validation={validation_loss:.4f}"
        )
        if stale_epochs >= patience:
            print(f"  early stopping after {epoch} epochs")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    pack = {
        "model": model,
        "scaler": scaler,
        "config": config,
        "calibration_slope": 1.0,
        "calibration_intercept": 0.0,
        "checkpoint": checkpoint,
        "history": history,
    }
    _save_ratio_classifier(pack)
    print(f"Saved ratio classifier to {checkpoint}")
    return pack


@torch.no_grad()
def ratio_classifier_logit(
    pack: Mapping[str, Any],
    values: np.ndarray,
    *,
    calibrated: bool = True,
    batch_size: int = 65_536,
) -> np.ndarray:
    """Return raw or calibrated balanced-class logits."""
    values = _as_2d_float32(values, "values")
    model = pack["model"]
    device = next(model.parameters()).device
    model.eval()
    chunks = []
    for start in range(0, len(values), int(batch_size)):
        values_tensor = torch.tensor(
            pack["scaler"].transform(values[start : start + int(batch_size)]),
            dtype=torch.float32,
            device=device,
        )
        chunks.append(model(values_tensor).detach().cpu().numpy())
    if not chunks:
        return np.empty(0, dtype=np.float32)
    logits = np.concatenate(chunks).astype(np.float64)
    if calibrated:
        logits = (
            float(pack.get("calibration_slope", 1.0)) * logits
            + float(pack.get("calibration_intercept", 0.0))
        )
    return logits


def ratio_classifier_ensemble_logit(
    packs: list[Mapping[str, Any]],
    values: np.ndarray,
    *,
    batch_size: int = 65_536,
) -> np.ndarray:
    """Return the log of the arithmetic-mean ratio from raw classifiers.

    Each member estimates a positive density ratio through ``exp(logit)``.
    Averaging the ratios, rather than the logits, matches the ensemble
    convention used in Exercise 5.  No post-hoc calibration is applied.
    """
    if not packs:
        raise ValueError("At least one ratio-classifier pack is required.")
    member_logits = np.stack(
        [
            ratio_classifier_logit(
                pack,
                values,
                calibrated=False,
                batch_size=batch_size,
            )
            for pack in packs
        ],
        axis=0,
    )
    maximum = np.max(member_logits, axis=0)
    return maximum + np.log(
        np.mean(np.exp(member_logits - maximum[None, :]), axis=0)
    )


def ratio_classifier_ensemble_ratio(
    packs: list[Mapping[str, Any]],
    values: np.ndarray,
    *,
    max_abs_log_ratio: float = 20.0,
    batch_size: int = 65_536,
) -> np.ndarray:
    """Return the uncalibrated arithmetic-mean density ratio of an ensemble."""
    log_ratio = ratio_classifier_ensemble_logit(
        packs,
        values,
        batch_size=batch_size,
    )
    return np.exp(np.clip(log_ratio, -max_abs_log_ratio, max_abs_log_ratio))


def calibrate_ratio_classifier(
    pack: dict[str, Any],
    positive: np.ndarray,
    negative: np.ndarray,
    *,
    max_iter: int = 100,
) -> dict[str, Any]:
    """Fit an affine logit calibration on an independent balanced sample."""
    positive = _as_2d_float32(positive, "positive")
    negative = _as_2d_float32(negative, "negative")
    raw_logits = np.concatenate(
        [
            ratio_classifier_logit(pack, positive, calibrated=False),
            ratio_classifier_logit(pack, negative, calibrated=False),
        ]
    )
    labels = np.concatenate(
        [np.ones(len(positive)), np.zeros(len(negative))]
    ).astype(np.float64)
    logits_tensor = torch.tensor(raw_logits, dtype=torch.float64)
    labels_tensor = torch.tensor(labels, dtype=torch.float64)
    slope = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    intercept = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [slope, intercept], max_iter=int(max_iter), line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        calibrated_logits = slope * logits_tensor + intercept
        loss = F.binary_cross_entropy_with_logits(
            calibrated_logits, labels_tensor
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    pack["calibration_slope"] = float(slope.detach())
    pack["calibration_intercept"] = float(intercept.detach())
    _save_ratio_classifier(pack)
    print(
        "Affine logit calibration: "
        f"slope={pack['calibration_slope']:.4f}, "
        f"intercept={pack['calibration_intercept']:.4f}"
    )
    return pack


def ratio_classifier_ratio(
    pack: Mapping[str, Any],
    values: np.ndarray,
    *,
    max_abs_log_ratio: float = 20.0,
    batch_size: int = 65_536,
) -> np.ndarray:
    """Convert calibrated balanced-class logits into positive density ratios."""
    log_ratio = ratio_classifier_logit(
        pack, values, calibrated=True, batch_size=batch_size
    )
    return np.exp(np.clip(log_ratio, -max_abs_log_ratio, max_abs_log_ratio))
