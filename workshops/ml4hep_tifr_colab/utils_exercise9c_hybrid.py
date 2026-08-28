"""Runtime for Exercise 9c: intermediate flows plus large residual-ratio models.

The notebooks are intentionally thin launchers.  Keeping the scientific logic
here makes the ten SBIBM campaigns use exactly the same bank split, model,
checkpoint, ratio-arithmetic, metric, and plotting code.

The design is deliberately asymmetric:

* one intermediate, normalized conditional flow is a proposal with full support;
* fresh, much larger simulator banks train plain wide CE classifiers;
* a single 3-class S/P/L model is compared with independent S/P and S/L
  binary models;
* only direct float64 softmax probability quotients are deployed;
* validation selects checkpoints and a fourth, fresh bank is audit-only.
"""

from __future__ import annotations

import copy
import gc
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist, pdist
from scipy.special import logsumexp

from utils_benchmark import infer_parameter_transform, load_or_simulate_bank
from utils_exercise9c_contract import (
    ALL_TASKS,
    CAMPAIGN_SCHEMA,
    FLOW_MODEL_CONFIG,
    FLOW_TRAINING_POLICY,
    INITIAL_LEARNING_RATE,
    JANA_PAPER_COMMIT,
    LEARNING_RATE_DROP_FACTOR,
    METHOD_BINARY,
    METHOD_COLORS,
    METHOD_FLOW,
    METHOD_LABELS,
    METHOD_MULTICLASS,
    METHODS,
    METRIC_SCHEMA,
    MINIMUM_LEARNING_RATE,
    PROFILES,
    campaign_run_tag,
    campaign_signature,
    normalize_profile,
    validate_seed,
)
from utils_hnpe import (
    install_nflows_rqs_float64_retry,
    sample_spline_flow_ensemble,
    spline_flow_ensemble_log_prob,
    train_spline_flow_ensemble,
)
from utils_plotting import export_standalone_figure_script


DISCRETE_TASKS = {"bernoulli_glm", "bernoulli_glm_raw", "sir"}
CLASS_S, CLASS_P, CLASS_L = 0, 1, 2


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % 2**32)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def require_finite_rows(task_name: str, stage: str, values: Any) -> np.ndarray:
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array.reshape(-1, 1)
    flat = array.reshape(len(array), -1)
    finite = np.isfinite(flat).all(axis=1)
    if not finite.all():
        raise FloatingPointError(
            f"{task_name} [{stage}]: {int((~finite).sum())}/{len(flat)} "
            "rows contain non-finite values"
        )
    return array


def dequantization_widths(task_name: str, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if task_name not in DISCRETE_TASKS:
        return np.zeros(values.shape[1], dtype=np.float32)
    if task_name != "bernoulli_glm":
        return np.ones(values.shape[1], dtype=np.float32)
    widths = []
    for column in values.T:
        unique = np.unique(np.round(column[: min(len(column), 20_000)], 7))
        gaps = np.diff(unique)
        gaps = gaps[gaps > 1.0e-7]
        robust_scale = np.std(column, dtype=np.float64)
        width = np.median(gaps) if len(gaps) else 0.02 * robust_scale
        width = np.clip(width, 1.0e-4, max(1.0e-4, 0.10 * robust_scale))
        widths.append(width)
    return np.asarray(widths, dtype=np.float32)


def dequantize(values: np.ndarray, widths: np.ndarray, seed: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    widths = np.asarray(widths, dtype=np.float32)
    if not np.any(widths):
        return values.copy()
    rng = np.random.default_rng(int(seed))
    return (values + (rng.random(values.shape) - 0.5) * widths).astype(np.float32)


def _safe_torch_load(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _export_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    export_standalone_figure_script(
        fig, script_name=f"{stem}.py", output_dir=output_dir
    )
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    print("Exported", output_dir / f"{stem}.py")


def normalized_positive_weights(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if (
        not np.isfinite(values).all()
        or np.any(values < 0.0)
        or not np.any(values > 0.0)
    ):
        raise FloatingPointError("Invalid positive importance weights")
    # Scale before summing so several finite near-maximal float64 ratios do
    # not overflow. Numerically negligible weights may underflow to zero;
    # self-normalized resampling and ESS both permit exact zero mass.
    scaled = values / values.max()
    total = scaled.sum(dtype=np.float64)
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError("Invalid scaled importance-weight sum")
    return scaled / total


def effective_sample_size(weights: np.ndarray) -> float:
    weights = normalized_positive_weights(weights)
    return float(1.0 / np.square(weights).sum())


def robust_pooled_limits(
    arrays: Sequence[np.ndarray], column: int, padding: float = 0.05
) -> tuple[float, float]:
    values = np.concatenate(
        [np.asarray(array, dtype=np.float64)[:, int(column)] for array in arrays]
    )
    values = values[np.isfinite(values)]
    low, high = np.quantile(values, [0.005, 0.995])
    width = float(high - low)
    if not np.isfinite(width) or width <= 0.0:
        width = max(abs(float(low)), 1.0) * 0.1
    return float(low - padding * width), float(high + padding * width)


@dataclass
class PreparedBank:
    role: str
    theta: np.ndarray
    z: np.ndarray
    x: np.ndarray
    seed: int
    cache_path: Path


@dataclass
class ClassBank:
    simulator: np.ndarray
    posterior_reference: np.ndarray
    likelihood_reference: np.ndarray
    provenance: dict[str, Any]

    def __len__(self) -> int:
        return len(self.simulator)

    @property
    def input_dim(self) -> int:
        return int(self.simulator.shape[1])

    def classes(self, family: str) -> tuple[np.ndarray, ...]:
        if family == "multiclass":
            return (
                self.simulator,
                self.posterior_reference,
                self.likelihood_reference,
            )
        if family == "posterior_binary":
            return self.simulator, self.posterior_reference
        if family == "likelihood_binary":
            return self.simulator, self.likelihood_reference
        raise ValueError(f"Unknown classifier family {family!r}")


class PlainCEMLP(nn.Module):
    """Plain ReLU MLP: logits only, with no implicit regularization."""

    def __init__(
        self, input_dim: int, width: int, hidden_layers: int, n_classes: int
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = int(input_dim)
        for _ in range(int(hidden_layers)):
            layers.extend([nn.Linear(current, int(width)), nn.ReLU()])
            current = int(width)
        layers.append(nn.Linear(current, int(n_classes)))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def flow_model_config() -> dict[str, Any]:
    return dict(FLOW_MODEL_CONFIG)


def flow_training_config(campaign: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(FLOW_TRAINING_POLICY)
    config.update(
        {
            "batch_size": int(campaign["flow_batch_size"]),
            "n_epochs": int(campaign["flow_epochs"]),
        }
    )
    return config


def draw_flow_mixture(
    packs: Sequence[Mapping[str, Any]],
    n_samples: int,
    contexts: np.ndarray,
    seed: int,
    *,
    allocation: str = "iid",
) -> np.ndarray:
    contexts = np.atleast_2d(np.asarray(contexts, dtype=np.float32))
    seed_everything(seed)
    values = sample_spline_flow_ensemble(
        packs,
        int(n_samples),
        context=contexts,
        seed=int(seed),
        batch_size=16_384,
        allocation=allocation,
    )
    values = np.asarray(values, dtype=np.float32)
    n_features = int(np.asarray(packs[0]["target_scaler"].mean).size)
    if len(contexts) == 1:
        values = values.reshape(1, int(n_samples), n_features)
    expected = (len(contexts), int(n_samples), n_features)
    if values.shape != expected:
        raise RuntimeError(f"Unexpected flow sample shape {values.shape}; expected {expected}")
    if not np.isfinite(values).all():
        raise FloatingPointError("Flow sampling returned non-finite values")
    return values


def build_class_bank(
    bank: PreparedBank,
    q_phi: Sequence[Mapping[str, Any]],
    q_eta: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    chunk_size: int = 32_768,
) -> ClassBank:
    """Build matched S/P/L groups without ever mixing bank roles."""

    n_rows = len(bank.z)
    z_p = np.empty_like(bank.z, dtype=np.float32)
    x_l = np.empty_like(bank.x, dtype=np.float32)
    for chunk, start in enumerate(range(0, n_rows, int(chunk_size))):
        stop = min(n_rows, start + int(chunk_size))
        z_p[start:stop] = draw_flow_mixture(
            q_phi, 1, bank.x[start:stop], seed + 2 * chunk + 1
        )[:, 0, :]
        x_l[start:stop] = draw_flow_mixture(
            q_eta, 1, bank.z[start:stop], seed + 2 * chunk + 2
        )[:, 0, :]
        if chunk == 0 or (chunk + 1) % 10 == 0 or stop == n_rows:
            print(f"  {bank.role} S/P/L groups: {stop:,}/{n_rows:,}")
    simulator = np.column_stack([bank.z, bank.x]).astype(np.float32)
    posterior_reference = np.column_stack([z_p, bank.x]).astype(np.float32)
    likelihood_reference = np.column_stack([bank.z, x_l]).astype(np.float32)
    for name, values in (
        ("S", simulator),
        ("P", posterior_reference),
        ("L", likelihood_reference),
    ):
        require_finite_rows(bank.role, f"class {name}", values)
    return ClassBank(
        simulator=simulator,
        posterior_reference=posterior_reference,
        likelihood_reference=likelihood_reference,
        provenance={
            "role": bank.role,
            "simulator_seed": bank.seed,
            "cache_path": str(bank.cache_path),
            "rows": n_rows,
            "class_sampling_seed": int(seed),
            "z_dim": int(bank.z.shape[1]),
            "x_dim": int(bank.x.shape[1]),
        },
    )


def fit_common_transform(bank: ClassBank) -> tuple[np.ndarray, np.ndarray]:
    """Fit one class-symmetric transform shared by all three comparisons."""

    arrays = bank.classes("multiclass")
    count = sum(len(values) for values in arrays)
    total = np.zeros(bank.input_dim, dtype=np.float64)
    total2 = np.zeros(bank.input_dim, dtype=np.float64)
    for values in arrays:
        for start in range(0, len(values), 65_536):
            chunk = np.asarray(values[start : start + 65_536], dtype=np.float64)
            total += chunk.sum(axis=0)
            total2 += np.square(chunk).sum(axis=0)
    center = total / count
    variance = np.maximum(total2 / count - np.square(center), 0.0)
    scale = np.sqrt(variance)
    scale = np.where(scale > 1.0e-6, scale, 1.0)
    return center.astype(np.float32), scale.astype(np.float32)


def _classifier_config(
    family: str,
    input_dim: int,
    n_classes: int,
    campaign: Mapping[str, Any],
    campaign_signature_value: str,
) -> dict[str, Any]:
    return {
        "family": family,
        "input_dim": int(input_dim),
        "n_classes": int(n_classes),
        "width": int(campaign["classifier_width"]),
        "hidden_layers": int(campaign["classifier_layers"]),
        "members": int(campaign["classifier_members"]),
        "row_batch_budget": int(campaign["classifier_row_batch_budget"]),
        "steps": int(campaign["classifier_steps"]),
        "validation_interval": int(campaign["classifier_validation_interval"]),
        "initial_learning_rate": INITIAL_LEARNING_RATE,
        "minimum_learning_rate": MINIMUM_LEARNING_RATE,
        "learning_rate_schedule": "drops_at_40_70_90_percent",
        "objective": "equal_prior_cross_entropy_only",
        "checkpoint_objective": "fresh_bank_validation_cross_entropy_only",
        "regularization": "none",
        "input_transform": "common_class_symmetric_column_mean_std_v1",
        "campaign_signature": campaign_signature_value,
    }


def _learning_rate_for_step(step: int, total_steps: int) -> float:
    fractions = (0.40, 0.70, 0.90)
    exponent = sum(int(step) >= max(1, int(round(total_steps * f))) for f in fractions)
    return max(
        MINIMUM_LEARNING_RATE,
        INITIAL_LEARNING_RATE * LEARNING_RATE_DROP_FACTOR**exponent,
    )


def _set_learning_rate(
    optimizer: torch.optim.Optimizer, step: int, total_steps: int
) -> float:
    value = _learning_rate_for_step(step, total_steps)
    for group in optimizer.param_groups:
        group["lr"] = value
    return value


def _group_batch(
    arrays: Sequence[np.ndarray],
    indices: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    values = np.stack([array[indices] for array in arrays], axis=1)
    values = (values - center) / scale
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def _group_ce(model: nn.Module, groups: torch.Tensor) -> torch.Tensor:
    n_groups, n_classes, n_features = groups.shape
    logits = model(groups.reshape(-1, n_features)).reshape(
        n_groups, n_classes, n_classes
    )
    labels = torch.arange(n_classes, device=groups.device).repeat(n_groups)
    return F.cross_entropy(logits.reshape(-1, n_classes), labels)


@torch.no_grad()
def _validation_ce(
    model: nn.Module,
    arrays: Sequence[np.ndarray],
    indices: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
    *,
    chunk_size: int = 1_024,
) -> float:
    model.eval()
    total = 0.0
    rows = 0
    for start in range(0, len(indices), int(chunk_size)):
        local = indices[start : start + int(chunk_size)]
        groups = _group_batch(arrays, local, center, scale, device)
        value = _group_ce(model, groups)
        total += float(value.detach().cpu()) * len(local)
        rows += len(local)
    return total / max(1, rows)


def classifier_checkpoint_paths(
    model_dir: Path, campaign: Mapping[str, Any]
) -> list[Path]:
    paths = []
    for family in ("multiclass", "posterior_binary", "likelihood_binary"):
        paths.extend(
            model_dir / family / f"member_{member:02d}.pt"
            for member in range(int(campaign["classifier_members"]))
        )
    return paths


def _load_classifier_family(
    family: str,
    checkpoint_dir: Path,
    campaign: Mapping[str, Any],
    campaign_signature_value: str,
    input_dim: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    n_classes = 3 if family == "multiclass" else 2
    expected_config = _classifier_config(
        family, input_dim, n_classes, campaign, campaign_signature_value
    )
    packs = []
    for member in range(int(campaign["classifier_members"])):
        checkpoint = checkpoint_dir / f"member_{member:02d}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        saved = _safe_torch_load(checkpoint, device)
        if saved.get("config") != expected_config or saved.get("member") != member:
            raise RuntimeError(
                f"Classifier checkpoint contract mismatch: {checkpoint}. "
                "Use a new run tag rather than mixing campaigns."
            )
        center = np.asarray(saved["center"], dtype=np.float32)
        scale = np.asarray(saved["scale"], dtype=np.float32)
        if (
            center.shape != (input_dim,)
            or scale.shape != (input_dim,)
            or not np.isfinite(center).all()
            or not np.isfinite(scale).all()
            or np.any(scale <= 0.0)
        ):
            raise RuntimeError(f"Invalid classifier transform in {checkpoint}")
        model = PlainCEMLP(
            input_dim,
            int(campaign["classifier_width"]),
            int(campaign["classifier_layers"]),
            n_classes,
        ).to(device)
        model.load_state_dict(saved["state_dict"], strict=True)
        model.eval()
        packs.append(
            {
                "model": model,
                "center": center,
                "scale": scale,
                "history": saved["history"],
                "family": family,
                "member": member,
                "checkpoint": checkpoint,
                "data_provenance": saved.get("data_provenance", {}),
            }
        )
        print("Loaded frozen", checkpoint)
    return packs


def train_classifier_family(
    family: str,
    train_bank: ClassBank,
    validation_bank: ClassBank,
    checkpoint_dir: Path,
    campaign: Mapping[str, Any],
    campaign_signature_value: str,
    center: np.ndarray,
    scale: np.ndarray,
    *,
    seed: int,
    device: torch.device,
    load_if_available: bool,
) -> list[dict[str, Any]]:
    arrays = train_bank.classes(family)
    validation_arrays = validation_bank.classes(family)
    n_classes = len(arrays)
    config = _classifier_config(
        family, train_bank.input_dim, n_classes, campaign, campaign_signature_value
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    all_present = all(
        (checkpoint_dir / f"member_{member:02d}.pt").is_file()
        for member in range(int(campaign["classifier_members"]))
    )
    if load_if_available and all_present:
        return _load_classifier_family(
            family,
            checkpoint_dir,
            campaign,
            campaign_signature_value,
            train_bank.input_dim,
            device,
        )
    if load_if_available and any(
        (checkpoint_dir / f"member_{member:02d}.pt").is_file()
        for member in range(int(campaign["classifier_members"]))
    ):
        raise RuntimeError(
            f"Partial classifier ensemble in {checkpoint_dir}. Refusing to mix "
            "old and newly trained members; finish or remove that run explicitly."
        )

    validation_rng = np.random.default_rng(int(seed) + 13)
    n_validation = min(len(validation_bank), 20_000)
    validation_index = validation_rng.choice(
        len(validation_bank), n_validation, replace=False
    )
    group_batch = max(
        1, int(campaign["classifier_row_batch_budget"]) // n_classes
    )
    total_steps = int(campaign["classifier_steps"])
    validation_interval = int(campaign["classifier_validation_interval"])
    patience = int(campaign["classifier_patience_validations"])
    packs = []
    for member in range(int(campaign["classifier_members"])):
        checkpoint = checkpoint_dir / f"member_{member:02d}.pt"
        member_seed = int(seed) + 10_007 * member
        seed_everything(member_seed)
        model = PlainCEMLP(
            train_bank.input_dim,
            int(campaign["classifier_width"]),
            int(campaign["classifier_layers"]),
            n_classes,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LEARNING_RATE)
        rng = np.random.default_rng(member_seed + 1)
        history = {
            "step": [],
            "train_ce": [],
            "validation_ce": [],
            "learning_rate": [],
            "selected_step": None,
            "selected_validation_ce": None,
        }
        best_state = None
        best_value = math.inf
        best_step = 0
        stale = 0
        running_losses: list[float] = []
        print(
            f"\n{family} member {member + 1}/{campaign['classifier_members']}: "
            f"{total_steps:,} CE steps, {group_batch:,} matched groups/step"
        )
        for step in range(1, total_steps + 1):
            learning_rate = _set_learning_rate(optimizer, step - 1, total_steps)
            index = rng.integers(0, len(train_bank), size=group_batch)
            groups = _group_batch(arrays, index, center, scale, device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = _group_ce(model, groups)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite {family} CE")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_losses.append(float(loss.detach().cpu()))
            if step % validation_interval == 0 or step == total_steps:
                value = _validation_ce(
                    model,
                    validation_arrays,
                    validation_index,
                    center,
                    scale,
                    device,
                )
                history["step"].append(step)
                history["train_ce"].append(float(np.mean(running_losses)))
                history["validation_ce"].append(value)
                history["learning_rate"].append(learning_rate)
                running_losses.clear()
                print(
                    f"  step {step:6d}/{total_steps}: "
                    f"train={history['train_ce'][-1]:.5f}, "
                    f"fresh-val={value:.5f}, lr={learning_rate:.1e}"
                )
                if value < best_value - 1.0e-6:
                    best_value = value
                    best_state = copy.deepcopy(model.state_dict())
                    best_step = step
                    stale = 0
                else:
                    stale += 1
                if stale >= patience:
                    print(f"  early stop after {step:,} steps")
                    break
        if best_state is None:
            raise RuntimeError(f"{family} never produced a finite checkpoint")
        model.load_state_dict(best_state)
        model.eval()
        history["selected_step"] = best_step
        history["selected_validation_ce"] = best_value
        torch.save(
            {
                "state_dict": best_state,
                "config": config,
                "center": center,
                "scale": scale,
                "history": history,
                "member": member,
                "member_seed": member_seed,
                "data_provenance": {
                    "training": train_bank.provenance,
                    "validation": validation_bank.provenance,
                },
            },
            checkpoint,
        )
        packs.append(
            {
                "model": model,
                "center": center,
                "scale": scale,
                "history": history,
                "family": family,
                "member": member,
                "checkpoint": checkpoint,
                "data_provenance": {
                    "training": train_bank.provenance,
                    "validation": validation_bank.provenance,
                },
            }
        )
    return packs


@torch.no_grad()
def predict_probability_ratio(
    packs: Sequence[Mapping[str, Any]],
    points: np.ndarray,
    *,
    numerator: int = 0,
    denominator: int = 1,
    batch_size: int = 16_384,
    return_members: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Arithmetic mean of member-wise direct float64 softmax quotients."""

    if not packs:
        raise ValueError("At least one classifier member is required")
    points = np.asarray(points, dtype=np.float32)
    member_outputs = []
    tiny = torch.finfo(torch.float64).tiny
    for pack in packs:
        chunks = []
        pack["model"].eval()
        for start in range(0, len(points), int(batch_size)):
            values = (
                points[start : start + int(batch_size)] - pack["center"]
            ) / pack["scale"]
            tensor = torch.as_tensor(values, dtype=torch.float32, device=next(pack["model"].parameters()).device)
            probability = torch.softmax(pack["model"](tensor).to(torch.float64), dim=1)
            ratio = probability[:, int(numerator)] / probability[:, int(denominator)].clamp_min(tiny)
            chunks.append(ratio.detach().cpu().numpy())
        member_outputs.append(np.concatenate(chunks))
    members = np.stack(member_outputs, axis=0)
    # Divide each member before summing to avoid overflow when several valid
    # direct quotients lie near float64's maximum.
    mean = np.sum(members / len(members), axis=0, dtype=np.float64)
    if not np.isfinite(mean).all() or np.any(mean <= 0.0):
        raise FloatingPointError("Invalid direct softmax probability quotient")
    return (mean, members) if return_members else mean


def predict_multiclass_ratios(
    packs: Sequence[Mapping[str, Any]],
    points: np.ndarray,
    *,
    return_members: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[
    tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]
]:
    posterior = predict_probability_ratio(
        packs, points, numerator=CLASS_S, denominator=CLASS_P,
        return_members=return_members,
    )
    likelihood = predict_probability_ratio(
        packs, points, numerator=CLASS_S, denominator=CLASS_L,
        return_members=return_members,
    )
    return posterior, likelihood


@torch.no_grad()
def predict_probabilities(
    packs: Sequence[Mapping[str, Any]],
    points: np.ndarray,
    *,
    batch_size: int = 16_384,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    member_probabilities = []
    for pack in packs:
        chunks = []
        model = pack["model"]
        model.eval()
        model_device = next(model.parameters()).device
        for start in range(0, len(points), int(batch_size)):
            values = (
                points[start : start + int(batch_size)] - pack["center"]
            ) / pack["scale"]
            tensor = torch.as_tensor(values, dtype=torch.float32, device=model_device)
            chunks.append(
                torch.softmax(model(tensor).to(torch.float64), dim=1)
                .detach()
                .cpu()
                .numpy()
            )
        member_probabilities.append(np.concatenate(chunks))
    members = np.stack(member_probabilities, axis=0)
    return members.mean(axis=0), members


def _c2st_equal(
    left: np.ndarray, right: np.ndarray, seed: int, max_samples: int
) -> float:
    from sbibm.metrics import c2st

    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    n = min(len(left), len(right), int(max_samples))
    if n < 32:
        return float("nan")
    rng = np.random.default_rng(int(seed))
    left = left[rng.choice(len(left), n, replace=False)]
    right = right[rng.choice(len(right), n, replace=False)]
    return float(
        c2st(
            torch.as_tensor(left),
            torch.as_tensor(right),
            seed=int(seed) % (2**32 - 1),
            n_folds=5,
            z_score=True,
        ).item()
    )


def _reference_standardization(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference, dtype=np.float64)
    mean = reference.mean(axis=0)
    std = reference.std(axis=0)
    return mean, np.where(std > 1.0e-8, std, 1.0)


def _reference_bandwidth2(reference: np.ndarray) -> float:
    pilot = np.asarray(reference, dtype=np.float64)[:1_000]
    distances = pdist(pilot, metric="sqeuclidean")
    distances = distances[distances > 0]
    return max(float(np.median(distances)) if len(distances) else 1.0, 1.0e-8)


def _mmd2(
    left: np.ndarray, right: np.ndarray, bandwidth2: float, chunk: int = 256
) -> float:
    def kernel_mean(first: np.ndarray, second: np.ndarray) -> float:
        total = 0.0
        count = 0
        for start in range(0, len(first), int(chunk)):
            distance2 = cdist(
                first[start : start + int(chunk)], second, metric="sqeuclidean"
            )
            total += float(np.exp(-distance2 / (2.0 * bandwidth2)).sum())
            count += distance2.size
        return total / max(1, count)

    return max(
        0.0,
        kernel_mean(left, left)
        + kernel_mean(right, right)
        - 2.0 * kernel_mean(left, right),
    )


def paired_distribution_metrics(
    reference: np.ndarray,
    candidates: Mapping[str, np.ndarray],
    *,
    seed: int,
    max_samples: int,
) -> dict[str, dict[str, float]]:
    if tuple(candidates) != METHODS:
        raise ValueError(f"Expected ordered candidates {METHODS}")
    reference = np.asarray(reference, dtype=np.float64)
    n = min(
        len(reference),
        int(max_samples),
        *(len(candidates[method]) for method in METHODS),
    )
    if n < 32:
        raise RuntimeError("Too few rows for paired C2ST/MMD")
    rng = np.random.default_rng(int(seed))
    reference_equal = reference[rng.choice(len(reference), n, replace=False)]
    candidate_equal = {}
    for index, method in enumerate(METHODS):
        local = np.random.default_rng(int(seed) + 1_009 * (index + 1))
        values = np.asarray(candidates[method], dtype=np.float64)
        candidate_equal[method] = values[
            local.choice(len(values), n, replace=False)
        ]
    mean, std = _reference_standardization(reference)
    ref_std = (reference_equal - mean) / std
    bandwidth2 = _reference_bandwidth2(ref_std)
    results = {}
    for index, method in enumerate(METHODS):
        candidate_std = (candidate_equal[method] - mean) / std
        mmd2 = _mmd2(ref_std, candidate_std, bandwidth2)
        results[method] = {
            "C2ST": _c2st_equal(
                reference_equal,
                candidate_equal[method],
                int(seed) + 20_000 + index,
                n,
            ),
            "MMD2": mmd2,
            "MMD": math.sqrt(mmd2),
            "MMD_bandwidth2": bandwidth2,
            "metric_rows": n,
        }
    return results


def flow_audit(
    q_phi: Sequence[Mapping[str, Any]],
    q_eta: Sequence[Mapping[str, Any]],
    audit_bank: PreparedBank,
) -> pd.DataFrame:
    posterior_log_prob = spline_flow_ensemble_log_prob(
        q_phi, audit_bank.z, context=audit_bank.x
    )
    likelihood_log_prob = spline_flow_ensemble_log_prob(
        q_eta, audit_bank.x, context=audit_bank.z
    )
    rows = []
    for name, packs, target, log_probability in (
        ("q_phi", q_phi, audit_bank.z, posterior_log_prob),
        ("q_eta", q_eta, audit_bank.x, likelihood_log_prob),
    ):
        scaler = packs[0]["target_scaler"]
        standardized = (target - scaler.mean) / scaler.std
        tail = np.any(np.abs(standardized) > 5.0, axis=1)
        rows.append(
            {
                "flow": name,
                "audit_rows": len(target),
                "mean_nll": float(-np.mean(log_probability)),
                "median_nll": float(-np.median(log_probability)),
                "p99_nll": float(np.quantile(-log_probability, 0.99)),
                "finite_log_probability_fraction": float(
                    np.mean(np.isfinite(log_probability))
                ),
                "outside_standardized_target_tail_bound_fraction": float(np.mean(tail)),
                "linear_tails_preserve_support": True,
            }
        )
    return pd.DataFrame(rows)


def plot_flow_training_and_audit(
    q_phi: Sequence[Mapping[str, Any]],
    q_eta: Sequence[Mapping[str, Any]],
    audit_frame: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.7), constrained_layout=True)
    for label, pack, color in (
        (r"$q_\phi(z\mid x)$", q_phi[0], METHOD_COLORS[METHOD_FLOW]),
        (r"$q_\eta(x\mid z)$", q_eta[0], "#B279A2"),
    ):
        history = pack.get("history", {})
        axes[0].plot(history.get("train", []), alpha=0.75, label=f"{label} train", color=color)
        axes[0].plot(history.get("validation", []), ls="--", label=f"{label} val", color=color)
        axes[1].step(
            np.arange(1, len(history.get("learning_rate", [])) + 1),
            history.get("learning_rate", []),
            where="post",
            label=label,
            color=color,
        )
    axes[0].set(title="modest-flow learning curves", xlabel="epoch", ylabel="NLL")
    axes[1].set(title="flow learning rate", xlabel="epoch", ylabel="Adam LR", yscale="log")
    axes[2].bar(
        audit_frame["flow"], audit_frame["mean_nll"],
        color=[METHOD_COLORS[METHOD_FLOW], "#B279A2"],
    )
    axes[2].set(title="fresh audit-bank NLL", ylabel="mean NLL")
    for ax in axes:
        ax.grid(alpha=0.25)
        if ax.lines:
            ax.legend(fontsize=7)
    _export_figure(fig, output_dir, "flow_training_and_fresh_audit")
    plt.show()


def _expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> tuple[float, np.ndarray, np.ndarray]:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confidence, bin_accuracy, weights = [], [], []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lower) & (
            confidence <= upper if upper == 1.0 else confidence < upper
        )
        if mask.any():
            bin_confidence.append(float(confidence[mask].mean()))
            bin_accuracy.append(float(correct[mask].mean()))
            weights.append(float(mask.mean()))
    ece = float(
        np.sum(np.asarray(weights) * np.abs(np.asarray(bin_accuracy) - np.asarray(bin_confidence)))
    )
    return ece, np.asarray(bin_confidence), np.asarray(bin_accuracy)


def _binary_roc(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores)[::-1]
    positives = max(1, int(labels.sum()))
    negatives = max(1, int((~labels).sum()))
    true_positive = np.cumsum(labels[order]) / positives
    false_positive = np.cumsum(~labels[order]) / negatives
    true_positive = np.concatenate([[0.0], true_positive, [1.0]])
    false_positive = np.concatenate([[0.0], false_positive, [1.0]])
    auc = float(np.trapz(true_positive, false_positive))
    return false_positive, true_positive, auc


def audit_ratio_models(
    audit_bank: ClassBank,
    classifiers: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    seed: int,
    max_metric_samples: int,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(int(seed))
    n_groups = min(len(audit_bank), 20_000)
    index = rng.choice(len(audit_bank), n_groups, replace=False)
    family_rows = []
    calibration_curves = {}
    confusion_matrices = {}
    roc_curves = {}
    for family, packs in classifiers.items():
        arrays = audit_bank.classes(family)
        points = np.concatenate([array[index] for array in arrays], axis=0)
        labels = np.concatenate(
            [np.full(n_groups, label, dtype=np.int64) for label in range(len(arrays))]
        )
        probabilities, member_probabilities = predict_probabilities(packs, points)
        selected = np.clip(probabilities[np.arange(len(labels)), labels], 1.0e-300, 1.0)
        predictions = probabilities.argmax(axis=1)
        confusion = np.zeros((len(arrays), len(arrays)), dtype=np.float64)
        for truth in range(len(arrays)):
            mask = labels == truth
            for predicted in range(len(arrays)):
                confusion[truth, predicted] = np.mean(predictions[mask] == predicted)
        ece, confidence, accuracy = _expected_calibration_error(probabilities, labels)
        calibration_curves[family] = (confidence, accuracy)
        confusion_matrices[family] = confusion
        family_rocs = []
        for class_index in range(len(arrays)):
            family_rocs.append(
                _binary_roc(labels == class_index, probabilities[:, class_index])
            )
        roc_curves[family] = family_rocs
        member_ce = [
            float(-np.log(np.clip(member[np.arange(len(labels)), labels], 1.0e-300, 1.0)).mean())
            for member in member_probabilities
        ]
        family_rows.append(
            {
                "family": family,
                "audit_groups": n_groups,
                "audit_CE": float(-np.log(selected).mean()),
                "audit_accuracy": float(np.mean(predictions == labels)),
                "audit_ECE": ece,
                "audit_macro_AUC": float(
                    np.mean([curve[2] for curve in family_rocs])
                ),
                "member_CE_mean": float(np.mean(member_ce)),
                "member_CE_std": float(np.std(member_ce)),
                "audit_bank_used_for_selection": False,
            }
        )

    fig, axes = plt.subplots(3, 3, figsize=(11.8, 10.4), constrained_layout=True)
    for column, family in enumerate(("multiclass", "posterior_binary", "likelihood_binary")):
        matrix = confusion_matrices[family]
        image = axes[0, column].imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues")
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                axes[0, column].text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", fontsize=8)
        axes[0, column].set(
            title=f"{family.replace('_', ' ')} confusion",
            xlabel="predicted class",
            ylabel="true class",
            xticks=range(matrix.shape[1]),
            yticks=range(matrix.shape[0]),
        )
        confidence, accuracy = calibration_curves[family]
        axes[1, column].plot([0, 1], [0, 1], "k--", lw=1)
        axes[1, column].plot(confidence, accuracy, "o-")
        axes[1, column].set(
            title=f"{family.replace('_', ' ')} calibration",
            xlabel="mean confidence",
            ylabel="empirical accuracy",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        axes[1, column].grid(alpha=0.25)
        axes[2, column].plot([0, 1], [0, 1], "k--", lw=1)
        for class_index, (false_positive, true_positive, auc) in enumerate(roc_curves[family]):
            axes[2, column].plot(
                false_positive,
                true_positive,
                label=f"class {class_index} (AUC={auc:.3f})",
            )
        axes[2, column].set(
            title=f"{family.replace('_', ' ')} one-vs-rest ROC",
            xlabel="false-positive rate",
            ylabel="true-positive rate",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        axes[2, column].legend(fontsize=7)
        axes[2, column].grid(alpha=0.25)
    fig.colorbar(image, ax=axes[0, :], shrink=0.72, label="row fraction")
    _export_figure(fig, output_dir, "audit_confusion_and_calibration")
    plt.show()

    joint_points = np.concatenate(
        [
            audit_bank.simulator[index],
            audit_bank.posterior_reference[index],
            audit_bank.likelihood_reference[index],
        ]
    )
    multi_p, multi_l = predict_multiclass_ratios(
        classifiers["multiclass"], joint_points, return_members=True
    )
    multi_p_mean, multi_p_members = multi_p
    multi_l_mean, multi_l_members = multi_l
    binary_p_mean, binary_p_members = predict_probability_ratio(
        classifiers["posterior_binary"], joint_points, return_members=True
    )
    binary_l_mean, binary_l_members = predict_probability_ratio(
        classifiers["likelihood_binary"], joint_points, return_members=True
    )

    agreement_rows = []
    for route, left, right, left_members, right_members in (
        ("posterior_S_over_P", multi_p_mean, binary_p_mean, multi_p_members, binary_p_members),
        ("likelihood_S_over_L", multi_l_mean, binary_l_mean, multi_l_members, binary_l_members),
    ):
        log_left = np.log(left)
        log_right = np.log(right)
        agreement_rows.append(
            {
                "route": route,
                "audit_points": len(left),
                "pearson_log_ratio": float(np.corrcoef(log_left, log_right)[0, 1]),
                "rms_log_ratio_difference": float(np.sqrt(np.mean(np.square(log_left - log_right)))),
                "multiclass_log_ratio_p01": float(np.quantile(log_left, 0.01)),
                "multiclass_log_ratio_p99": float(np.quantile(log_left, 0.99)),
                "binary_log_ratio_p01": float(np.quantile(log_right, 0.01)),
                "binary_log_ratio_p99": float(np.quantile(log_right, 0.99)),
                "multiclass_member_log_std_mean": float(np.mean(np.std(np.log(left_members), axis=0))),
                "binary_member_log_std_mean": float(np.mean(np.std(np.log(right_members), axis=0))),
            }
        )

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.4), constrained_layout=True)
    show = rng.choice(len(joint_points), min(20_000, len(joint_points)), replace=False)
    ratio_specs = (
        ("posterior", multi_p_mean, binary_p_mean, multi_p_members, binary_p_members),
        ("likelihood", multi_l_mean, binary_l_mean, multi_l_members, binary_l_members),
    )
    for column, (route, left, right, left_members, right_members) in enumerate(ratio_specs):
        x = np.log(left[show])
        y = np.log(right[show])
        low, high = np.quantile(np.concatenate([x, y]), [0.005, 0.995])
        axes[0, column].scatter(x, y, s=3, alpha=0.12, rasterized=True)
        axes[0, column].plot([low, high], [low, high], "k--", lw=1)
        axes[0, column].set(
            title=f"{route}: multiclass vs binary",
            xlabel="multiclass log probability quotient",
            ylabel="binary log probability quotient",
            xlim=(low, high),
            ylim=(low, high),
        )
        axes[1, column].hist(
            np.std(np.log(left_members[:, show]), axis=0), bins=50,
            density=True, histtype="step", lw=2, color=METHOD_COLORS[METHOD_MULTICLASS],
            label="multiclass ensemble",
        )
        axes[1, column].hist(
            np.std(np.log(right_members[:, show]), axis=0), bins=50,
            density=True, histtype="step", lw=2, color=METHOD_COLORS[METHOD_BINARY],
            label="binary ensemble",
        )
        axes[1, column].set(
            title=f"{route}: member disagreement",
            xlabel="memberwise log-ratio standard deviation",
            ylabel="density",
        )
        axes[1, column].legend(fontsize=8)
        for row in range(2):
            axes[row, column].grid(alpha=0.25)
    _export_figure(fig, output_dir, "audit_multiclass_binary_ratio_agreement")
    plt.show()

    closure_rows = []
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.2), constrained_layout=True)
    closure_specs = (
        (
            "posterior",
            audit_bank.simulator[index],
            audit_bank.posterior_reference[index],
            multi_p_mean[n_groups : 2 * n_groups],
            binary_p_mean[n_groups : 2 * n_groups],
            0,
        ),
        (
            "likelihood",
            audit_bank.simulator[index],
            audit_bank.likelihood_reference[index],
            multi_l_mean[2 * n_groups : 3 * n_groups],
            binary_l_mean[2 * n_groups : 3 * n_groups],
            int(audit_bank.provenance["z_dim"]),
        ),
    )
    for row_index, (route, truth, proposal, multi_ratio, binary_ratio, column) in enumerate(closure_specs):
        before = _c2st_equal(truth, proposal, seed + 101 * row_index, max_metric_samples)
        local_rng = np.random.default_rng(seed + 500 + row_index)
        corrected = {}
        for method, ratio in (
            (METHOD_MULTICLASS, multi_ratio),
            (METHOD_BINARY, binary_ratio),
        ):
            weights = normalized_positive_weights(ratio)
            selected = local_rng.choice(len(proposal), len(proposal), replace=True, p=weights)
            corrected[method] = proposal[selected]
            closure_rows.append(
                {
                    "route": route,
                    "method": method,
                    "audit_rows": len(proposal),
                    "proposal_C2ST_before": before,
                    "weighted_C2ST_after": _c2st_equal(
                        truth, corrected[method], seed + 700 + 11 * row_index + len(closure_rows), max_metric_samples
                    ),
                    "ESS": effective_sample_size(weights),
                    "ESS_fraction": effective_sample_size(weights) / len(weights),
                    "max_weight": float(weights.max()),
                    "ratio_p01": float(np.quantile(ratio, 0.01)),
                    "ratio_median": float(np.median(ratio)),
                    "ratio_p99": float(np.quantile(ratio, 0.99)),
                }
            )
        limits = robust_pooled_limits([truth, proposal, *corrected.values()], column)
        bins = np.linspace(*limits, 60)
        axes[row_index, 0].hist(truth[:, column], bins=bins, density=True, histtype="step", lw=2, color="black", label="S truth")
        axes[row_index, 0].hist(proposal[:, column], bins=bins, density=True, histtype="step", lw=1.5, color=METHOD_COLORS[METHOD_FLOW], label="uncorrected proposal")
        for method in (METHOD_MULTICLASS, METHOD_BINARY):
            axes[row_index, 0].hist(corrected[method][:, column], bins=bins, density=True, histtype="step", lw=2, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        axes[row_index, 0].set(title=f"{route} closure marginal", xlabel=f"joint feature {column + 1}", ylabel="density")
        route_frame = pd.DataFrame(closure_rows).query("route == @route")
        axes[row_index, 1].bar(
            ["before", "multiclass", "binary"],
            [before, *route_frame["weighted_C2ST_after"].tolist()],
            color=[METHOD_COLORS[METHOD_FLOW], METHOD_COLORS[METHOD_MULTICLASS], METHOD_COLORS[METHOD_BINARY]],
        )
        axes[row_index, 1].axhline(0.5, color="black", ls="--", lw=1)
        axes[row_index, 1].set(title=f"{route} fresh-bank closure", ylabel="C2ST")
        axes[row_index, 0].legend(fontsize=7)
        axes[row_index, 0].grid(alpha=0.2)
        axes[row_index, 1].grid(axis="y", alpha=0.2)
    _export_figure(fig, output_dir, "audit_reweighting_closure")
    plt.show()
    return pd.DataFrame(family_rows), pd.DataFrame(agreement_rows + closure_rows)


def plot_classifier_histories(
    classifiers: Mapping[str, Sequence[Mapping[str, Any]]], output_dir: Path
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.7), constrained_layout=True)
    colors = {
        "multiclass": METHOD_COLORS[METHOD_MULTICLASS],
        "posterior_binary": METHOD_COLORS[METHOD_BINARY],
        "likelihood_binary": "#E45756",
    }
    for family, packs in classifiers.items():
        for member, pack in enumerate(packs):
            history = pack["history"]
            label = family.replace("_", " ") if member == 0 else None
            axes[0].plot(history["step"], history["train_ce"], alpha=0.45, color=colors[family], label=label)
            axes[1].plot(history["step"], history["validation_ce"], alpha=0.55, color=colors[family], label=label)
            axes[2].step(history["step"], history["learning_rate"], where="post", alpha=0.45, color=colors[family], label=label)
    axes[0].set(title="step-window training CE", xlabel="optimizer step", ylabel="CE")
    axes[1].set(title="fresh-bank validation CE", xlabel="optimizer step", ylabel="CE")
    axes[2].set(title="progressive learning rate", xlabel="optimizer step", ylabel="Adam LR", yscale="log")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    _export_figure(fig, output_dir, "classifier_learning_curves_all_members")
    plt.show()


def observation_contexts(
    task: Any,
    observation: np.ndarray,
    widths: np.ndarray,
    *,
    n_jitters: int,
    seed: int,
) -> np.ndarray:
    count = int(n_jitters) if str(task.name) in DISCRETE_TASKS else 1
    contexts = np.repeat(
        np.asarray(observation, dtype=np.float32).reshape(1, -1), count, axis=0
    )
    if count > 1:
        contexts = dequantize(contexts, widths, seed)
    return require_finite_rows(str(task.name), "observation contexts", contexts).astype(np.float32)


def draw_posterior_comparison(
    task: Any,
    transform: Any,
    observation: np.ndarray,
    widths: np.ndarray,
    q_phi: Sequence[Mapping[str, Any]],
    classifiers: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    n_proposal: int,
    n_posterior: int,
    n_jitters: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, np.ndarray]]:
    contexts = observation_contexts(
        task, observation, widths, n_jitters=n_jitters, seed=seed + 1
    )
    draws_per_context = max(256, int(math.ceil(int(n_proposal) / len(contexts))))
    z = draw_flow_mixture(
        q_phi, draws_per_context, contexts, seed + 2, allocation="iid"
    )
    z_flat = z.reshape(-1, z.shape[-1])
    x_flat = np.repeat(
        contexts[:, None, :], draws_per_context, axis=1
    ).reshape(-1, contexts.shape[-1])
    theta = require_finite_rows(
        str(task.name), "inverse-transformed q_phi candidates", transform.inverse(z_flat)
    ).astype(np.float32)
    points = np.column_stack([z_flat, x_flat]).astype(np.float32)
    multiclass_ratio, _ = predict_multiclass_ratios(
        classifiers["multiclass"], points
    )
    binary_ratio = predict_probability_ratio(
        classifiers["posterior_binary"], points
    )
    ratios = {
        METHOD_MULTICLASS: np.asarray(multiclass_ratio, dtype=np.float64),
        METHOD_BINARY: np.asarray(binary_ratio, dtype=np.float64),
    }
    outputs = {}
    direct_rng = np.random.default_rng(int(seed) + 3)
    outputs[METHOD_FLOW] = theta[
        direct_rng.choice(
            len(theta), size=int(n_posterior), replace=int(n_posterior) > len(theta)
        )
    ]
    diagnostics = []
    raw_weights = {}
    for method_index, method in enumerate((METHOD_MULTICLASS, METHOD_BINARY)):
        odds = ratios[method].reshape(len(contexts), draws_per_context)
        context_weights = [
            normalized_positive_weights(row) / len(contexts) for row in odds
        ]
        weights = normalized_positive_weights(np.concatenate(context_weights))
        rng = np.random.default_rng(int(seed) + 4 + method_index)
        selected = rng.choice(len(theta), size=int(n_posterior), replace=True, p=weights)
        outputs[method] = theta[selected]
        raw_weights[method] = weights
        diagnostics.append(
            {
                "method": method,
                "proposal_rows": len(theta),
                "observation_jitters": len(contexts),
                "ESS": effective_sample_size(weights),
                "ESS_fraction": effective_sample_size(weights) / len(weights),
                "max_weight": float(weights.max()),
                "log_ratio_p01": float(np.quantile(np.log(ratios[method]), 0.01)),
                "log_ratio_median": float(np.median(np.log(ratios[method]))),
                "log_ratio_p99": float(np.quantile(np.log(ratios[method]), 0.99)),
                "posthoc_context_log_normalizer_rms": float(
                    np.sqrt(np.mean(np.square(
                        logsumexp(np.log(odds), axis=1) - math.log(odds.shape[1])
                    )))
                ),
            }
        )
    diagnostics.insert(
        0,
        {
            "method": METHOD_FLOW,
            "proposal_rows": len(theta),
            "observation_jitters": len(contexts),
            "ESS": float(len(theta)),
            "ESS_fraction": 1.0,
            "max_weight": 1.0 / len(theta),
            "log_ratio_p01": 0.0,
            "log_ratio_median": 0.0,
            "log_ratio_p99": 0.0,
            "posthoc_context_log_normalizer_rms": 0.0,
        },
    )
    return outputs, pd.DataFrame(diagnostics), raw_weights


def draw_predictive_comparison(
    transform: Any,
    posterior: Mapping[str, np.ndarray],
    q_eta: Sequence[Mapping[str, Any]],
    classifiers: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    n_outputs: int,
    n_candidates: int,
    seed: int,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    outputs = {}
    diagnostic_rows = []
    for method_index, method in enumerate(METHODS):
        rng = np.random.default_rng(int(seed) + 1_000 * method_index)
        posterior_theta = np.asarray(posterior[method], dtype=np.float32)
        theta = posterior_theta[
            rng.choice(
                len(posterior_theta),
                size=int(n_outputs),
                replace=int(n_outputs) > len(posterior_theta),
            )
        ]
        z = transform.forward(theta).astype(np.float32)
        if method == METHOD_FLOW:
            x = draw_flow_mixture(q_eta, 1, z, seed + 1_000 * method_index + 1)[:, 0, :]
            outputs[method] = (theta, x.astype(np.float32))
            diagnostic_rows.append(
                {
                    "method": method,
                    "candidate_count": 1,
                    "candidate_ESS_mean": 1.0,
                    "candidate_ESS_fraction_mean": 1.0,
                    "candidate_ESS_fraction_min": 1.0,
                    "candidate_max_weight_mean": 1.0,
                    "candidate_max_weight_worst": 1.0,
                }
            )
            continue
        x_candidates = draw_flow_mixture(
            q_eta,
            int(n_candidates),
            z,
            seed + 1_000 * method_index + 1,
            allocation="iid",
        )
        z_repeat = np.repeat(z[:, None, :], int(n_candidates), axis=1)
        points = np.concatenate([z_repeat, x_candidates], axis=2).reshape(
            -1, z.shape[1] + x_candidates.shape[2]
        )
        if method == METHOD_MULTICLASS:
            _, ratio = predict_multiclass_ratios(classifiers["multiclass"], points)
        else:
            ratio = predict_probability_ratio(classifiers["likelihood_binary"], points)
        odds = np.asarray(ratio, dtype=np.float64).reshape(
            int(n_outputs), int(n_candidates)
        )
        scaled_odds = odds / odds.max(axis=1, keepdims=True)
        weights = scaled_odds / scaled_odds.sum(axis=1, keepdims=True)
        if not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise FloatingPointError("Invalid predictive ratio weights")
        cdf = np.cumsum(weights, axis=1)
        selected = np.minimum(
            np.sum(cdf < rng.random(int(n_outputs))[:, None], axis=1),
            int(n_candidates) - 1,
        )
        x = x_candidates[np.arange(int(n_outputs)), selected]
        ess = 1.0 / np.square(weights).sum(axis=1)
        outputs[method] = (theta, x.astype(np.float32))
        diagnostic_rows.append(
            {
                "method": method,
                "candidate_count": int(n_candidates),
                "candidate_ESS_mean": float(ess.mean()),
                "candidate_ESS_fraction_mean": float((ess / n_candidates).mean()),
                "candidate_ESS_fraction_min": float((ess / n_candidates).min()),
                "candidate_max_weight_mean": float(weights.max(axis=1).mean()),
                "candidate_max_weight_worst": float(weights.max()),
                "posthoc_candidate_log_normalizer_rms": float(
                    np.sqrt(np.mean(np.square(
                        logsumexp(np.log(odds), axis=1) - math.log(odds.shape[1])
                    )))
                ),
            }
        )
    return outputs, pd.DataFrame(diagnostic_rows)


def plot_posterior_comparison(
    reference: np.ndarray,
    candidates: Mapping[str, np.ndarray],
    labels: Sequence[str],
    *,
    observation_number: int,
    task_name: str,
    seed: int,
    output_dir: Path,
) -> None:
    reference = np.asarray(reference)
    candidate_arrays = {method: np.asarray(candidates[method]) for method in METHODS}
    pooled = [reference, *(candidate_arrays[method] for method in METHODS)]
    rng = np.random.default_rng(int(seed))
    if reference.shape[1] == 2:
        xlim = robust_pooled_limits(pooled, 0)
        ylim = robust_pooled_limits(pooled, 1)
        fig, axes = plt.subplots(1, 6, figsize=(21.0, 3.7), constrained_layout=True)
        panels = [(reference, "official reference", "black")]
        panels.extend(
            (candidate_arrays[method], METHOD_LABELS[method], METHOD_COLORS[method])
            for method in METHODS
        )
        for ax, (values, title, color) in zip(axes[:4], panels):
            selected = rng.choice(len(values), min(len(values), 10_000), replace=False)
            outside = np.mean(
                (values[:, 0] < xlim[0]) | (values[:, 0] > xlim[1])
                | (values[:, 1] < ylim[0]) | (values[:, 1] > ylim[1])
            )
            ax.scatter(values[selected, 0], values[selected, 1], s=3, alpha=0.14, color=color, rasterized=True)
            ax.set(title=title, xlabel=labels[0], ylabel=labels[1], xlim=xlim, ylim=ylim)
            ax.set_aspect("equal", adjustable="box")
            ax.text(0.02, 0.02, f"{100 * outside:.2f}% outside", transform=ax.transAxes, fontsize=7, bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"})
        for column, ax in enumerate(axes[4:]):
            limits = xlim if column == 0 else ylim
            bins = np.linspace(*limits, 60)
            ax.hist(reference[:, column], bins=bins, density=True, histtype="step", lw=2, color="black", label="reference")
            for method in METHODS:
                ax.hist(candidate_arrays[method][:, column], bins=bins, density=True, histtype="step", lw=2, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
            ax.set(title=f"marginal {column + 1}", xlabel=labels[column], ylabel="density", xlim=limits)
            ax.legend(fontsize=6)
    else:
        n_show = min(6, reference.shape[1])
        fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), constrained_layout=True)
        for column, ax in enumerate(axes.flat):
            if column >= n_show:
                ax.axis("off")
                continue
            limits = robust_pooled_limits(pooled, column)
            bins = np.linspace(*limits, 55)
            ax.hist(reference[:, column], bins=bins, density=True, histtype="step", lw=2, color="black", label="reference")
            for method in METHODS:
                ax.hist(candidate_arrays[method][:, column], bins=bins, density=True, histtype="step", lw=1.8, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
            ax.set(title=f"marginal {column + 1}", xlabel=labels[column], ylabel="density", xlim=limits)
            ax.legend(fontsize=6)
    fig.suptitle(f"{task_name}, observation {observation_number}: common pooled view", fontsize=10)
    _export_figure(fig, output_dir, f"posterior_observation_{observation_number}")
    plt.show()


def plot_posterior_weights(
    diagnostic_frame: pd.DataFrame,
    raw_weights: Mapping[str, np.ndarray],
    *,
    observation_number: int,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7), constrained_layout=True)
    for method in (METHOD_MULTICLASS, METHOD_BINARY):
        values = np.log(
            np.maximum(
                np.asarray(raw_weights[method]) * len(raw_weights[method]),
                np.finfo(np.float64).tiny,
            )
        )
        axes[0].hist(values, bins=70, density=True, histtype="step", lw=2, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
    ratio_frame = diagnostic_frame.set_index("method")
    axes[1].bar(
        ["multiclass", "binary"],
        [ratio_frame.loc[METHOD_MULTICLASS, "ESS_fraction"], ratio_frame.loc[METHOD_BINARY, "ESS_fraction"]],
        color=[METHOD_COLORS[METHOD_MULTICLASS], METHOD_COLORS[METHOD_BINARY]],
    )
    axes[2].bar(
        ["multiclass", "binary"],
        [ratio_frame.loc[METHOD_MULTICLASS, "max_weight"], ratio_frame.loc[METHOD_BINARY, "max_weight"]],
        color=[METHOD_COLORS[METHOD_MULTICLASS], METHOD_COLORS[METHOD_BINARY]],
    )
    axes[0].set(title="posterior weight spectrum", xlabel=r"$\log(Nw)$", ylabel="density")
    axes[1].set(title="importance efficiency", ylabel="ESS / proposal rows", ylim=(0, 1.02))
    axes[2].set(title="largest normalized weight", ylabel="max weight", yscale="log")
    axes[0].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.25)
    _export_figure(fig, output_dir, f"posterior_weights_observation_{observation_number}")
    plt.show()


def plot_predictive_comparison(
    reference_theta: np.ndarray,
    reference_x: np.ndarray,
    candidates: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    observation_number: int,
    seed: int,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.2), constrained_layout=True)
    n_marginals = min(3, reference_x.shape[1])
    for column in range(3):
        ax = axes.flat[column]
        if column >= n_marginals:
            ax.axis("off")
            continue
        arrays = [reference_x, *(candidates[method][1] for method in METHODS)]
        limits = robust_pooled_limits(arrays, column)
        bins = np.linspace(*limits, 55)
        ax.hist(reference_x[:, column], bins=bins, density=True, histtype="step", lw=2, color="black", label="official predictive")
        for method in METHODS:
            ax.hist(candidates[method][1][:, column], bins=bins, density=True, histtype="step", lw=1.8, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        ax.set(title=f"predictive marginal {column + 1}", xlabel=fr"$x_{{{column + 1}}}$", ylabel="density")
        ax.legend(fontsize=6)
    ax = axes.flat[3]
    joint = [(reference_theta, reference_x, "official", "black")]
    joint.extend((candidates[m][0], candidates[m][1], METHOD_LABELS[m], METHOD_COLORS[m]) for m in METHODS)
    theta_limits = robust_pooled_limits([row[0] for row in joint], 0)
    x_limits = robust_pooled_limits([row[1] for row in joint], 0)
    rng = np.random.default_rng(int(seed))
    for theta, x, label, color in joint:
        selected = rng.choice(len(x), min(len(x), 2_000), replace=False)
        ax.scatter(theta[selected, 0], x[selected, 0], s=5, alpha=0.17, color=color, label=label, rasterized=True)
    ax.set(title="posterior-predictive joint slice", xlabel=r"$\theta_1$", ylabel=r"$x_1$", xlim=theta_limits, ylim=x_limits)
    ax.legend(fontsize=6)
    _export_figure(fig, output_dir, f"predictive_observation_{observation_number}")
    plt.show()


def _prepare_bank(
    task: Any,
    role: str,
    n_rows: int,
    cache_path: Path,
    seed: int,
    transform: Any,
    widths: np.ndarray,
    *,
    dequantization_seed: int,
) -> PreparedBank:
    raw = load_or_simulate_bank(
        task,
        int(n_rows),
        cache_path=cache_path,
        seed=int(seed),
        chunk_size=20_000,
    )
    if int(raw.seed) != int(seed):
        raise RuntimeError(
            f"Cached {role} bank seed {raw.seed} does not match required seed {seed}. "
            "Use the exact role-specific path or regenerate explicitly."
        )
    theta = require_finite_rows(str(task.name), f"{role} theta", raw.theta).astype(np.float32)
    z = require_finite_rows(
        str(task.name), f"{role} transformed theta", transform.forward(theta)
    ).astype(np.float32)
    x = require_finite_rows(
        str(task.name),
        f"{role} observations",
        dequantize(raw.x, widths, int(dequantization_seed)),
    ).astype(np.float32)
    return PreparedBank(
        role=role,
        theta=theta,
        z=z,
        x=x,
        seed=int(seed),
        cache_path=cache_path,
    )


def _install_requested_backend(sbibm_module: Any, task_name: str) -> tuple[str, Any]:
    diagnostics = None
    backend = "sbibm_default"
    if task_name == "sir" and os.environ.get("EX9C_SIR_BACKEND", "").lower() == "python_rk4":
        from utils_sir_backend import install_sir_python_backend

        diagnostics = install_sir_python_backend(sbibm_module)
        backend = str(diagnostics["backend"])
    elif (
        task_name == "lotka_volterra"
        and os.environ.get("EX9C_LOTKA_VOLTERRA_BACKEND", "").lower()
        == "python_logrk4"
    ):
        from utils_lotka_volterra_backend import install_lotka_volterra_python_backend

        diagnostics = install_lotka_volterra_python_backend(sbibm_module)
        backend = str(diagnostics["backend"])
    return backend, diagnostics


def _save_sample_archives(
    result_root: Path,
    task_name: str,
    run_tag: str,
    observation_number: int,
    posterior_reference: np.ndarray,
    posterior: Mapping[str, np.ndarray],
    predictive_reference_theta: np.ndarray,
    predictive_reference_x: np.ndarray,
    predictive: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    campaign_signature_value: str,
    campaign_seed: int,
    simulator_backend: str,
    widths: np.ndarray,
) -> None:
    common = {
        "method_keys": np.asarray(METHODS),
        "observation_number": np.asarray(observation_number),
        "campaign_signature": np.asarray(campaign_signature_value),
        "campaign_seed": np.asarray(campaign_seed),
        "simulator_backend": np.asarray(simulator_backend),
        "metric_schema": np.asarray(METRIC_SCHEMA),
    }
    posterior_path = result_root / (
        f"{task_name}__{run_tag}__observation{observation_number}__posterior_samples.npz"
    )
    np.savez_compressed(
        posterior_path,
        reference_theta=posterior_reference,
        simple_flow_theta=posterior[METHOD_FLOW],
        hybrid_multiclass_theta=posterior[METHOD_MULTICLASS],
        hybrid_binary_theta=posterior[METHOD_BINARY],
        **common,
    )
    predictive_path = result_root / (
        f"{task_name}__{run_tag}__observation{observation_number}__predictive_samples.npz"
    )
    np.savez_compressed(
        predictive_path,
        reference_theta=predictive_reference_theta,
        reference_x=predictive_reference_x,
        simple_flow_theta=predictive[METHOD_FLOW][0],
        simple_flow_x=predictive[METHOD_FLOW][1],
        hybrid_multiclass_theta=predictive[METHOD_MULTICLASS][0],
        hybrid_multiclass_x=predictive[METHOD_MULTICLASS][1],
        hybrid_binary_theta=predictive[METHOD_BINARY][0],
        hybrid_binary_x=predictive[METHOD_BINARY][1],
        dequantization_widths=widths,
        **common,
    )
    print("Saved", posterior_path)
    print("Saved", predictive_path)


def plot_metric_summary(metrics: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), constrained_layout=True)
    for method in METHODS:
        selected = metrics.loc[metrics["method"] == method].sort_values("num_observation")
        axes[0].plot(selected["num_observation"], selected["posterior_C2ST"], "o-", color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        axes[1].plot(selected["num_observation"], selected["predictive_x_C2ST"], "o-", color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        axes[2].plot(selected["num_observation"], selected["predictive_joint_C2ST"], "o-", color=METHOD_COLORS[method], label=METHOD_LABELS[method])
    for ax, title in zip(axes, ("posterior", "predictive x", "predictive joint")):
        ax.axhline(0.5, color="black", ls="--", lw=1)
        ax.set(title=f"{title} C2ST", xlabel="observation", ylabel="C2ST", ylim=(0.45, 1.0))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6)
    _export_figure(fig, output_dir, "metric_summary_by_observation")
    plt.show()


def run_task(
    task_name: str,
    *,
    profile: str = "TUTORIAL",
    seed: int,
    artifact_root: str | Path,
    load_if_available: bool = True,
    c2st_max_samples: int | None = None,
) -> pd.DataFrame:
    """Run one complete 9c task and return its long-form metric table."""

    import sbibm

    task_name = str(task_name).lower()
    if task_name not in ALL_TASKS:
        raise ValueError(f"task_name must be one of {ALL_TASKS}")
    profile = normalize_profile(profile)
    seed = validate_seed(seed)
    campaign = PROFILES[profile]
    run_tag = campaign_run_tag(profile, seed)
    signature = campaign_signature(profile)
    artifact_root = Path(artifact_root).expanduser().resolve()
    model_root = artifact_root / "models"
    cache_root = artifact_root / "simulation_banks"
    result_root = artifact_root / "results"
    figure_root = artifact_root / "figures_scripts"
    for directory in (model_root, cache_root, result_root, figure_root):
        directory.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed)
    install_nflows_rqs_float64_retry()
    simulator_backend, backend_diagnostics = _install_requested_backend(
        sbibm, task_name
    )
    task = sbibm.get_task(task_name)
    transform = infer_parameter_transform(task)
    task_index = ALL_TASKS.index(task_name)
    task_seed = int(seed) + 100_000 * task_index
    backend_suffix = "" if simulator_backend == "sbibm_default" else f"__{simulator_backend}"

    role_specs = {
        "flow": (int(campaign["flow_simulations"]), task_seed + 101),
        "ratio_train": (int(campaign["ratio_train_simulations"]), task_seed + 211),
        "ratio_validation": (int(campaign["ratio_validation_simulations"]), task_seed + 307),
        "audit": (int(campaign["audit_simulations"]), task_seed + 401),
    }
    if len({seed_value for _, seed_value in role_specs.values()}) != len(role_specs):
        raise RuntimeError("Role-specific simulator seeds unexpectedly collide")
    cache_paths = {
        role: cache_root
        / f"{task_name}__{role}__n{n_rows}__seed{role_seed}{backend_suffix}.npz"
        for role, (n_rows, role_seed) in role_specs.items()
    }
    if len(set(cache_paths.values())) != len(cache_paths):
        raise RuntimeError("Role-specific simulator cache paths unexpectedly collide")

    # Widths are a deterministic density convention fixed by the flow bank.
    flow_n, flow_seed = role_specs["flow"]
    flow_raw = load_or_simulate_bank(
        task,
        flow_n,
        cache_path=cache_paths["flow"],
        seed=flow_seed,
        chunk_size=20_000,
    )
    if int(flow_raw.seed) != flow_seed:
        raise RuntimeError("Flow cache has the wrong seed")
    widths = dequantization_widths(task_name, flow_raw.x)
    flow_bank = PreparedBank(
        role="flow",
        theta=require_finite_rows(task_name, "flow theta", flow_raw.theta).astype(np.float32),
        z=require_finite_rows(task_name, "flow z", transform.forward(flow_raw.theta)).astype(np.float32),
        x=require_finite_rows(task_name, "flow x", dequantize(flow_raw.x, widths, task_seed + 501)).astype(np.float32),
        seed=flow_seed,
        cache_path=cache_paths["flow"],
    )
    manifest = {
        "task": task_name,
        "profile": profile,
        "campaign_seed": seed,
        "task_seed": task_seed,
        "run_tag": run_tag,
        "campaign_schema": CAMPAIGN_SCHEMA,
        "campaign_signature": signature,
        "simulator_backend": simulator_backend,
        "backend_diagnostics": backend_diagnostics,
        "banks": {
            role: {"rows": n_rows, "seed": role_seed, "cache_path": str(cache_paths[role])}
            for role, (n_rows, role_seed) in role_specs.items()
        },
        "bank_contract": "fresh_disjoint_by_role_specific_rng_seed_and_persistent_path",
        "flow_model": flow_model_config(),
        "flow_training": flow_training_config(campaign),
        "classifier": {
            "members": campaign["classifier_members"],
            "width": campaign["classifier_width"],
            "layers": campaign["classifier_layers"],
            "steps": campaign["classifier_steps"],
            "objective": "CE only",
        },
        "dequantization_widths": widths.tolist(),
    }
    run_result_dir = result_root / task_name / run_tag
    run_figure_dir = figure_root / task_name / run_tag
    run_model_dir = model_root / task_name / run_tag
    for directory in (run_result_dir, run_figure_dir, run_model_dir):
        directory.mkdir(parents=True, exist_ok=True)
    _json_dump(run_result_dir / "campaign_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, default=str))
    print("Device:", device)

    q_phi = train_spline_flow_ensemble(
        flow_bank.z,
        context=flow_bank.x,
        checkpoint=run_model_dir / "q_phi.pt",
        ensemble_size=int(campaign["flow_members"]),
        model_config=flow_model_config(),
        training_config=flow_training_config(campaign),
        device=device,
        seed=task_seed + 1_001,
        load_if_available=bool(load_if_available),
        verify_checkpoint_data=True,
    )
    q_eta = train_spline_flow_ensemble(
        flow_bank.x,
        context=flow_bank.z,
        checkpoint=run_model_dir / "q_eta.pt",
        ensemble_size=int(campaign["flow_members"]),
        model_config=flow_model_config(),
        training_config=flow_training_config(campaign),
        device=device,
        seed=task_seed + 2_001,
        load_if_available=bool(load_if_available),
        verify_checkpoint_data=True,
    )

    classifier_paths = classifier_checkpoint_paths(run_model_dir, campaign)
    classifier_complete = all(path.is_file() for path in classifier_paths)
    classifier_partial = any(path.is_file() for path in classifier_paths) and not classifier_complete
    if load_if_available and classifier_partial:
        raise RuntimeError(
            "Partial 9c classifier campaign found. Refusing to mix members or "
            "factorizations from different runs."
        )
    classifiers: dict[str, list[dict[str, Any]]] = {}
    if load_if_available and classifier_complete:
        input_dim = flow_bank.z.shape[1] + flow_bank.x.shape[1]
        for family in ("multiclass", "posterior_binary", "likelihood_binary"):
            classifiers[family] = _load_classifier_family(
                family,
                run_model_dir / family,
                campaign,
                signature,
                input_dim,
                device,
            )
    else:
        ratio_banks = {}
        for role in ("ratio_train", "ratio_validation"):
            n_rows, role_seed = role_specs[role]
            ratio_banks[role] = _prepare_bank(
                task,
                role,
                n_rows,
                cache_paths[role],
                role_seed,
                transform,
                widths,
                dequantization_seed=task_seed + (601 if role == "ratio_train" else 701),
            )
        train_classes = build_class_bank(
            ratio_banks["ratio_train"], q_phi, q_eta, seed=task_seed + 3_001
        )
        validation_classes = build_class_bank(
            ratio_banks["ratio_validation"], q_phi, q_eta, seed=task_seed + 4_001
        )
        del ratio_banks
        gc.collect()
        center, scale = fit_common_transform(train_classes)
        for family_index, family in enumerate(("multiclass", "posterior_binary", "likelihood_binary")):
            classifiers[family] = train_classifier_family(
                family,
                train_classes,
                validation_classes,
                run_model_dir / family,
                campaign,
                signature,
                center,
                scale,
                seed=task_seed + 6_001 + 10_000 * family_index,
                device=device,
                load_if_available=False,
            )
        del train_classes, validation_classes
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    reference_center = classifiers["multiclass"][0]["center"]
    reference_scale = classifiers["multiclass"][0]["scale"]
    for family, packs in classifiers.items():
        for pack in packs:
            if not np.array_equal(pack["center"], reference_center) or not np.array_equal(pack["scale"], reference_scale):
                raise RuntimeError(
                    f"{family} does not share the common classifier transform"
                )

    audit_n, audit_seed = role_specs["audit"]
    audit_bank = _prepare_bank(
        task,
        "audit",
        audit_n,
        cache_paths["audit"],
        audit_seed,
        transform,
        widths,
        dequantization_seed=task_seed + 801,
    )
    flow_audit_frame = flow_audit(q_phi, q_eta, audit_bank)
    flow_audit_frame.to_csv(run_result_dir / "flow_fresh_audit.csv", index=False)
    plot_flow_training_and_audit(q_phi, q_eta, flow_audit_frame, run_figure_dir)
    plot_classifier_histories(classifiers, run_figure_dir)
    audit_classes = build_class_bank(
        audit_bank, q_phi, q_eta, seed=task_seed + 5_001
    )
    classifier_audit_frame, ratio_audit_frame = audit_ratio_models(
        audit_classes,
        classifiers,
        seed=task_seed + 30_001,
        max_metric_samples=int(
            c2st_max_samples or campaign["metric_max_samples"]
        ),
        output_dir=run_figure_dir,
    )
    classifier_audit_frame.to_csv(
        run_result_dir / "classifier_fresh_audit.csv", index=False
    )
    ratio_audit_frame.to_csv(run_result_dir / "ratio_agreement_and_closure.csv", index=False)
    del audit_classes
    gc.collect()

    metric_rows = []
    posterior_diagnostic_rows = []
    predictive_diagnostic_rows = []
    contrast_rows = []
    labels = task.get_labels_parameters()
    metric_max = int(c2st_max_samples or campaign["metric_max_samples"])
    observations = {
        int(number): require_finite_rows(
            task_name,
            f"official observation {number}",
            task.get_observation(num_observation=int(number)).detach().cpu().numpy().reshape(1, -1),
        ).astype(np.float32)
        for number in campaign["observations"]
    }
    for observation_number, observation in observations.items():
        print("\n" + "=" * 88)
        print(f"{task_name}: observation {observation_number}")
        observation_seed = task_seed + 50_000 + 1_000 * observation_number
        posterior, posterior_diagnostics, raw_weights = draw_posterior_comparison(
            task,
            transform,
            observation,
            widths,
            q_phi,
            classifiers,
            n_proposal=int(campaign["n_proposal"]),
            n_posterior=int(campaign["n_posterior"]),
            n_jitters=int(campaign["observation_jitters"]),
            seed=observation_seed,
        )
        posterior_diagnostics.insert(0, "num_observation", observation_number)
        posterior_diagnostic_rows.extend(posterior_diagnostics.to_dict("records"))
        reference = require_finite_rows(
            task_name,
            f"reference posterior {observation_number}",
            task.get_reference_posterior_samples(
                num_observation=int(observation_number)
            ).detach().cpu().numpy(),
        ).astype(np.float32)
        posterior_metrics = paired_distribution_metrics(
            reference,
            posterior,
            seed=observation_seed + 100,
            max_samples=metric_max,
        )
        plot_posterior_comparison(
            reference,
            posterior,
            labels,
            observation_number=observation_number,
            task_name=task_name,
            seed=observation_seed + 200,
            output_dir=run_figure_dir,
        )
        plot_posterior_weights(
            posterior_diagnostics,
            raw_weights,
            observation_number=observation_number,
            output_dir=run_figure_dir,
        )

        predictive, predictive_diagnostics = draw_predictive_comparison(
            transform,
            posterior,
            q_eta,
            classifiers,
            n_outputs=int(campaign["predictive_samples"]),
            n_candidates=int(campaign["predictive_candidates"]),
            seed=observation_seed + 300,
        )
        predictive_diagnostics.insert(0, "num_observation", observation_number)
        predictive_diagnostic_rows.extend(predictive_diagnostics.to_dict("records"))
        n_reference = min(
            int(campaign["predictive_reference_calls"]), len(reference)
        )
        reference_rng = np.random.default_rng(observation_seed + 400)
        reference_theta = reference[
            reference_rng.choice(len(reference), n_reference, replace=False)
        ]
        seed_everything(observation_seed + 401)
        simulator = task.get_simulator(max_calls=n_reference)
        reference_raw_x = simulator(
            torch.as_tensor(reference_theta, dtype=torch.float32)
        )
        reference_x = task.flatten_data(reference_raw_x).detach().cpu().numpy().astype(np.float32)
        reference_x = require_finite_rows(
            task_name,
            f"official predictive x {observation_number}",
            dequantize(reference_x, widths, observation_seed + 402),
        ).astype(np.float32)
        predictive_x_metrics = paired_distribution_metrics(
            reference_x,
            {method: predictive[method][1] for method in METHODS},
            seed=observation_seed + 500,
            max_samples=metric_max,
        )
        reference_joint = np.column_stack([reference_theta, reference_x])
        predictive_joint = {
            method: np.column_stack(predictive[method]) for method in METHODS
        }
        predictive_joint_metrics = paired_distribution_metrics(
            reference_joint,
            predictive_joint,
            seed=observation_seed + 600,
            max_samples=metric_max,
        )
        plot_predictive_comparison(
            reference_theta,
            reference_x,
            predictive,
            observation_number=observation_number,
            seed=observation_seed + 700,
            output_dir=run_figure_dir,
        )
        _save_sample_archives(
            run_result_dir,
            task_name,
            run_tag,
            observation_number,
            reference,
            posterior,
            reference_theta,
            reference_x,
            predictive,
            campaign_signature_value=signature,
            campaign_seed=seed,
            simulator_backend=simulator_backend,
            widths=widths,
        )
        for space, arrays in (
            ("posterior", posterior),
            ("predictive_x", {method: predictive[method][1] for method in METHODS}),
            ("predictive_joint", predictive_joint),
        ):
            for left_index, left in enumerate(METHODS):
                for right in METHODS[left_index + 1 :]:
                    contrast_rows.append(
                        {
                            "task": task_name,
                            "num_observation": observation_number,
                            "space": space,
                            "left": left,
                            "right": right,
                            "C2ST": _c2st_equal(
                                arrays[left],
                                arrays[right],
                                observation_seed + 800 + len(contrast_rows),
                                metric_max,
                            ),
                        }
                    )
        posterior_diag = posterior_diagnostics.set_index("method")
        predictive_diag = predictive_diagnostics.set_index("method")
        for method in METHODS:
            metric_rows.append(
                {
                    "task": task_name,
                    "profile": profile,
                    "run_tag": run_tag,
                    "seed": seed,
                    "campaign_schema": CAMPAIGN_SCHEMA,
                    "campaign_signature": signature,
                    "metric_schema": METRIC_SCHEMA,
                    "jana_paper_commit": JANA_PAPER_COMMIT,
                    "simulator_backend": simulator_backend,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "num_observation": observation_number,
                    "flow_simulations": campaign["flow_simulations"],
                    "ratio_train_simulations": campaign["ratio_train_simulations"],
                    "ratio_validation_simulations": campaign["ratio_validation_simulations"],
                    "audit_simulations": campaign["audit_simulations"],
                    "flow_members": campaign["flow_members"],
                    "classifier_members": 0 if method == METHOD_FLOW else campaign["classifier_members"],
                    "classifier_factorization": (
                        "none" if method == METHOD_FLOW else "one_multiclass" if method == METHOD_MULTICLASS else "two_separate_binary"
                    ),
                    "posterior_C2ST": posterior_metrics[method]["C2ST"],
                    "posterior_MMD2": posterior_metrics[method]["MMD2"],
                    "posterior_MMD": posterior_metrics[method]["MMD"],
                    "predictive_x_C2ST": predictive_x_metrics[method]["C2ST"],
                    "predictive_x_MMD2": predictive_x_metrics[method]["MMD2"],
                    "predictive_x_MMD": predictive_x_metrics[method]["MMD"],
                    "predictive_joint_C2ST": predictive_joint_metrics[method]["C2ST"],
                    "predictive_joint_MMD2": predictive_joint_metrics[method]["MMD2"],
                    "predictive_joint_MMD": predictive_joint_metrics[method]["MMD"],
                    "posterior_ESS_fraction": posterior_diag.loc[method, "ESS_fraction"],
                    "posterior_max_weight": posterior_diag.loc[method, "max_weight"],
                    "predictive_candidate_ESS_fraction_mean": predictive_diag.loc[method, "candidate_ESS_fraction_mean"],
                    "predictive_candidate_ESS_fraction_min": predictive_diag.loc[method, "candidate_ESS_fraction_min"],
                    "predictive_candidate_max_weight_worst": predictive_diag.loc[method, "candidate_max_weight_worst"],
                    "classifier_objective": "none_flow_NLL" if method == METHOD_FLOW else "equal_prior_CE_only",
                    "ratio_arithmetic": "none" if method == METHOD_FLOW else "arithmetic_mean_memberwise_direct_float64_softmax_quotient",
                }
            )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(run_result_dir / "metrics.csv", index=False)
    pd.DataFrame(posterior_diagnostic_rows).to_csv(
        run_result_dir / "posterior_weight_diagnostics.csv", index=False
    )
    pd.DataFrame(predictive_diagnostic_rows).to_csv(
        run_result_dir / "predictive_weight_diagnostics.csv", index=False
    )
    pd.DataFrame(contrast_rows).to_csv(
        run_result_dir / "direct_method_contrasts.csv", index=False
    )
    plot_metric_summary(metrics, run_figure_dir)
    status = {
        **manifest,
        "status": "complete",
        "metric_rows": len(metrics),
        "result_directory": str(run_result_dir),
        "figure_directory": str(run_figure_dir),
    }
    _json_dump(run_result_dir / "status.json", status)
    print("Completed", task_name, run_tag)
    return metrics


def run_from_environment() -> pd.DataFrame:
    task_name = os.environ.get("EX9C_TASK", "two_moons")
    profile = os.environ.get("EX9C_PROFILE", "TUTORIAL")
    seed = int(os.environ.get("EX9C_SEED", "31082026"))
    artifact_root = os.environ.get(
        "EX9C_ARTIFACT_ROOT", "./exercise_9c_SBIBM_hybrid_artifacts"
    )
    load_if_available = os.environ.get("EX9C_LOAD_IF_AVAILABLE", "1") != "0"
    max_samples = os.environ.get("EX9C_C2ST_MAX_SAMPLES")
    return run_task(
        task_name,
        profile=profile,
        seed=seed,
        artifact_root=artifact_root,
        load_if_available=load_if_available,
        c2st_max_samples=None if max_samples is None else int(max_samples),
    )
