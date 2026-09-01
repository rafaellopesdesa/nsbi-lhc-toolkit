"""Self-contained ML and statistical runtime for the SLCP paper study.

The low-level conditional-flow implementation is a frozen, local snapshot of
the audited Exercise-9 implementation.  The paper-specific bank preparation,
capacity selection, JANA baselines, broadened proposals, ratio corrections,
and diagnostics live at the end of this same module.  Nothing here imports a
utility from the parent workshop directory.

The functions intentionally accept and return NumPy arrays.  Keeping the
PyTorch and ``nflows`` details here makes the statistical identities in the
notebooks easier to read without hiding any step.  Scalar flows also expose
their analytic CDF and inverse-CDF through the standard-normal base.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import subprocess
import tempfile
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

# cuBLAS reads this at CUDA initialization, so set it before importing torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import logsumexp, ndtr, ndtri
from scipy.spatial.distance import cdist, pdist
from torch.utils.data import DataLoader, TensorDataset


PAPER_RUNTIME_SCHEMA = "slcp_paper_summary_v2"


def install_nflows_rqs_float64_retry(
    *, required_version: str = "0.14",
):
    """Install the audited finite-tensor inverse-RQS precision retry.

    ``nflows==0.14`` can evaluate a nearly double inverse-spline root with a
    tiny negative discriminant in float32.  The wrapped kernel retries only
    that failed inverse call, with the same finite tensors in float64.  It
    never clips, drops, or resamples a row, and a float64 failure remains
    fatal.  Repeated installation is idempotent across the Exercise 9 guards.
    """

    import functools
    import importlib
    import inspect
    from importlib.metadata import version
    import warnings

    installed_version = version("nflows")
    if installed_version != str(required_version):
        raise RuntimeError(
            "The audited inverse-RQS guard requires "
            f"nflows=={required_version}; found {installed_version}."
        )
    module = importlib.import_module(
        "nflows.transforms.splines.rational_quadratic"
    )
    original = module.rational_quadratic_spline
    existing_flags = (
        "_hybrid_float64_retry",
        "_exercise9_float64_retry",
        "_ex9b_float64_retry",
    )
    if any(getattr(original, flag, False) for flag in existing_flags):
        return original
    signature = inspect.signature(original)

    @functools.wraps(original)
    def guarded(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except AssertionError as error32:
            bound = signature.bind(*args, **kwargs)
            inputs = bound.arguments["inputs"]
            inverse = bound.arguments.get(
                "inverse", signature.parameters["inverse"].default
            )
            if not inverse or inputs.dtype != torch.float32:
                raise
            floating = [
                value
                for value in (*args, *kwargs.values())
                if torch.is_tensor(value) and value.is_floating_point()
            ]
            if any(not bool(torch.isfinite(value).all()) for value in floating):
                raise FloatingPointError(
                    "Non-finite tensor reached the inverse RQS; refusing the "
                    "precision retry."
                ) from error32

            def to_float64(value):
                if torch.is_tensor(value) and value.is_floating_point():
                    return value.to(dtype=torch.float64)
                return value

            try:
                outputs64, logdet64 = original(
                    *(to_float64(value) for value in args),
                    **{
                        name: to_float64(value)
                        for name, value in kwargs.items()
                    },
                )
            except AssertionError as error64:
                raise RuntimeError(
                    "The inverse-RQS discriminant also failed in float64; "
                    "retrain this flow."
                ) from error64
            if not (
                bool(torch.isfinite(outputs64).all())
                and bool(torch.isfinite(logdet64).all())
            ):
                raise FloatingPointError(
                    "The float64 inverse-RQS retry returned non-finite values."
                ) from error32
            guarded.retry_count += 1
            if guarded.retry_count == 1:
                warnings.warn(
                    "nflows float32 inverse-RQS cancellation: retrying the "
                    "same spline call in float64.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            outputs = outputs64.to(dtype=inputs.dtype)
            logdet = logdet64.to(dtype=inputs.dtype)
            if not (
                bool(torch.isfinite(outputs).all())
                and bool(torch.isfinite(logdet).all())
            ):
                raise FloatingPointError(
                    "Casting the inverse-RQS retry back to float32 produced "
                    "non-finite values."
                ) from error32
            return outputs, logdet

    guarded._hybrid_float64_retry = True
    guarded.retry_count = 0
    guarded._float32_original = original
    module.rational_quadratic_spline = guarded
    importlib.import_module(
        "nflows.transforms.splines"
    ).rational_quadratic_spline = guarded
    importlib.import_module(
        "nflows.transforms.autoregressive"
    ).rational_quadratic_spline = guarded
    linear_tail = importlib.import_module(
        "nflows.transforms.splines"
    ).unconstrained_rational_quadratic_spline
    if linear_tail.__globals__.get("rational_quadratic_spline") is not guarded:
        raise RuntimeError(
            "The nflows linear-tail inverse did not bind to the audited guard."
        )
    return guarded


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


def _as_2d_floating(
    values: np.ndarray, name: str, dtype: np.dtype,
) -> np.ndarray:
    """Validate a two-dimensional floating array without forcing float32."""

    values = np.asarray(values, dtype=dtype)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")
    if len(values) and not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values.")
    return values


def _flow_evaluation_dtypes(flow: nn.Module) -> tuple[torch.dtype, np.dtype]:
    """Return matching Torch/NumPy dtypes for scalar-flow evaluation."""

    torch_dtype = next(flow.parameters()).dtype
    if torch_dtype == torch.float32:
        return torch_dtype, np.dtype(np.float32)
    if torch_dtype == torch.float64:
        return torch_dtype, np.dtype(np.float64)
    raise TypeError(
        "Scalar spline CDF evaluation supports float32 or float64 flows; "
        f"received {torch_dtype}."
    )


def _standardizer_transform(
    standardizer: ArrayStandardizer,
    values: np.ndarray,
    dtype: np.dtype,
) -> np.ndarray:
    """Apply a stored affine standardizer in a requested inference dtype."""

    values = np.asarray(values, dtype=dtype)
    mean = np.asarray(standardizer.mean, dtype=dtype)
    std = np.asarray(standardizer.std, dtype=dtype)
    return (values - mean) / std


def _standardizer_inverse(
    standardizer: ArrayStandardizer,
    values: np.ndarray,
    dtype: np.dtype,
) -> np.ndarray:
    """Invert a stored affine standardizer in a requested inference dtype."""

    values = np.asarray(values, dtype=dtype)
    mean = np.asarray(standardizer.mean, dtype=dtype)
    std = np.asarray(standardizer.std, dtype=dtype)
    return values * std + mean


def _array_fingerprint(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for values in arrays:
        values = np.ascontiguousarray(np.asarray(values))
        digest.update(str(values.dtype).encode("utf-8"))
        digest.update(str(values.shape).encode("utf-8"))
        digest.update(values.view(np.uint8))
    return digest.hexdigest()


def _build_quadratic_spline(
    *,
    n_features: int,
    context_features: int,
    n_unbounded_affine_layers: int,
    affine_hidden_features: int,
    affine_hidden_layers: int,
    use_layer_permutations: bool,
    n_coupling_layers: int,
    hidden_features: int,
    hidden_layers: int,
    spline_num_bins: int,
    spline_tail_bound: float,
    dropout_probability: float,
    activation: str = "relu",
    identity_initialization: bool = False,
    linear_mixing: str = "none",
) -> nn.Module:
    """Build the workshop rational-quadratic spline coupling flow.

    Exactly one mechanism exchanges the transformed and conditioning
    coordinates.  The corrected Exercise-9 topology uses alternating masks
    with no fixed reversal.  ``use_layer_permutations=True`` is rejected:
    combining the historical reversal with the alternating masks cancels the
    role swap for even-dimensional targets and leaves half of the original
    coordinates untransformed.

    ``linear_mixing="lu"`` inserts a learned invertible LU map after every
    vector coupling.  It remains an opt-in upgrade over the Exercise-9
    alternating-mask topology.
    Scalar flows retain their monotone autoregressive construction: an LU map
    in one dimension adds no useful coordinate mixing.
    """
    try:
        from nflows.distributions.normal import StandardNormal
        from nflows.flows.base import Flow
        from nflows.nn.nets import ResidualNet
        from nflows.transforms.base import CompositeTransform
        from nflows.transforms.coupling import (
            AffineCouplingTransform,
            PiecewiseRationalQuadraticCouplingTransform,
        )
        from nflows.transforms.autoregressive import (
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
        )
        from nflows.transforms.lu import LULinear
        from nflows.utils.torchutils import create_alternating_binary_mask
    except ImportError as exc:
        raise ImportError(
            "The conditional-flow exercises require nflows. Install it with "
            "`pip install nflows`."
        ) from exc

    activation_name = str(activation).lower()
    activations = {"relu": F.relu, "silu": F.silu}
    if activation_name not in activations:
        raise ValueError(
            "activation must be one of "
            f"{sorted(activations)}; received {activation!r}."
        )
    activation_function = activations[activation_name]
    linear_mixing = str(linear_mixing).lower()
    if linear_mixing not in {"none", "lu"}:
        raise ValueError(
            "linear_mixing must be 'none' or 'lu'; "
            f"received {linear_mixing!r}."
        )
    if use_layer_permutations:
        raise ValueError(
            "The Exercise-9 coupling stack must not combine alternating "
            "masks with fixed reversals. Set use_layer_permutations=False."
        )

    def make_net(in_features: int, out_features: int) -> nn.Module:
        return ResidualNet(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            context_features=(context_features or None),
            num_blocks=hidden_layers,
            activation=activation_function,
            dropout_probability=dropout_probability,
            use_batch_norm=False,
        )

    def make_affine_net(in_features: int, out_features: int) -> nn.Module:
        return ResidualNet(
            in_features=in_features,
            out_features=out_features,
            hidden_features=affine_hidden_features,
            context_features=(context_features or None),
            num_blocks=affine_hidden_layers,
            activation=activation_function,
            dropout_probability=dropout_probability,
            use_batch_norm=False,
        )

    def initialize_affine_identity(transform: nn.Module) -> None:
        """Initialize an nflows affine coupling close to the identity."""

        final_layer = transform.transform_net.final_layer
        n_transformed = int(final_layer.bias.numel() // 2)
        # nflows uses scale=sigmoid(u+2)+1e-3.  This value gives scale=1.
        scale_logit = math.log(0.999 / 0.001) - 2.0
        with torch.no_grad():
            final_layer.weight.zero_()
            final_layer.bias.zero_()
            final_layer.bias[n_transformed:].fill_(scale_logit)

    def initialize_rqs_identity(transform: nn.Module) -> None:
        """Initialize every coupling spline with equal bins and slope one."""

        final_layer = transform.transform_net.final_layer
        multiplier = 3 * spline_num_bins - 1
        if final_layer.bias.numel() % multiplier:
            raise RuntimeError("Unexpected nflows RQS coupling output shape.")
        derivative_logit = math.log(
            math.expm1(1.0 - transform.min_derivative)
        )
        with torch.no_grad():
            final_layer.weight.zero_()
            bias = final_layer.bias.view(-1, multiplier)
            bias.zero_()
            bias[:, 2 * spline_num_bins :].fill_(derivative_logit)

    transforms = []
    if int(n_unbounded_affine_layers) and int(n_features) == 1:
        raise ValueError(
            "Unbounded affine coupling layers require at least two target "
            "features. Keep n_unbounded_affine_layers=0 for scalar flows."
        )
    for layer_index in range(int(n_unbounded_affine_layers)):
        mask = create_alternating_binary_mask(
            features=n_features,
            even=(layer_index % 2 == 0),
        )
        transform = AffineCouplingTransform(
            mask=mask,
            transform_net_create_fn=make_affine_net,
        )
        if identity_initialization:
            initialize_affine_identity(transform)
        transforms.append(transform)
        if linear_mixing == "lu":
            transforms.append(LULinear(int(n_features), identity_init=True))

    for layer_index in range(n_coupling_layers):
        if n_features == 1:
            # A coupling transform needs at least two target coordinates: one
            # coordinate is held fixed while another is transformed.  With a
            # scalar target, alternating coupling masks otherwise create
            # identity layers.  The one-dimensional autoregressive transform
            # is the exact analogue in the same rational-quadratic-spline
            # family and can be conditioned on the supplied context.
            transform = MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=1,
                hidden_features=hidden_features,
                context_features=(context_features or None),
                num_bins=spline_num_bins,
                tails="linear",
                tail_bound=spline_tail_bound,
                num_blocks=hidden_layers,
                activation=activation_function,
                dropout_probability=dropout_probability,
                use_batch_norm=False,
                use_residual_blocks=True,
                random_mask=False,
            )
            if identity_initialization:
                # Equal bin widths/heights and unit derivatives make the RQS
                # exactly the identity.  nflows pads the two linear-tail
                # endpoint derivatives internally, so only K-1 interior
                # derivative logits occur in the MADE output.
                final_layer = transform.autoregressive_net.final_layer
                derivative_logit = math.log(
                    math.expm1(1.0 - transform.min_derivative)
                )
                with torch.no_grad():
                    final_layer.weight.zero_()
                    final_layer.bias.zero_()
                    final_layer.bias[2 * spline_num_bins :].fill_(
                        derivative_logit
                    )
            transforms.append(transform)
        else:
            mask = create_alternating_binary_mask(
                features=n_features,
                even=(layer_index % 2 == 0),
            )
            transform = PiecewiseRationalQuadraticCouplingTransform(
                mask=mask,
                transform_net_create_fn=make_net,
                num_bins=spline_num_bins,
                tails="linear",
                tail_bound=spline_tail_bound,
                apply_unconditional_transform=False,
            )
            if identity_initialization:
                initialize_rqs_identity(transform)
            transforms.append(transform)
            if linear_mixing == "lu":
                transforms.append(
                    LULinear(int(n_features), identity_init=True)
                )

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
        n_unbounded_affine_layers=int(
            config.get("n_unbounded_affine_layers", 0)
        ),
        affine_hidden_features=int(
            config.get("affine_hidden_features", config["hidden_features"])
        ),
        affine_hidden_layers=int(
            config.get("affine_hidden_layers", config["hidden_layers"])
        ),
        # The corrected topology uses alternating masks alone. Historical
        # fixed reversals are rejected inside the builder.
        use_layer_permutations=bool(
            config.get("use_layer_permutations", False)
        ),
        n_coupling_layers=int(config["n_coupling_layers"]),
        hidden_features=int(config["hidden_features"]),
        hidden_layers=int(config["hidden_layers"]),
        spline_num_bins=int(config["spline_num_bins"]),
        spline_tail_bound=float(config["spline_tail_bound"]),
        dropout_probability=float(config.get("dropout_probability", 0.0)),
        activation=str(config.get("activation", "relu")),
        identity_initialization=bool(
            config.get("identity_initialization", False)
        ),
        linear_mixing=str(config.get("linear_mixing", "none")),
    )
    return flow.to(device)


def load_spline_flow(
    checkpoint: str | Path, device: torch.device
) -> dict[str, Any]:
    """Load a flow trained by :func:`train_spline_flow`."""
    checkpoint = Path(checkpoint)
    saved = _torch_load(checkpoint, device)
    config = dict(saved["config"])
    if "use_layer_permutations" not in config:
        legacy_permutations = any(
            key.endswith("._permutation")
            for key in saved["state_dict"]
        )
        if legacy_permutations:
            raise RuntimeError(
                f"Checkpoint {checkpoint} uses the retired Exercise-9 "
                "double-alternation topology (alternating masks plus "
                "reversals). Retrain it with "
                "use_layer_permutations=False and a new run tag."
            )
        config["use_layer_permutations"] = False
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
        "training_fingerprint": saved.get("training_fingerprint"),
    }


def train_spline_flow(
    target: np.ndarray,
    *,
    context: np.ndarray | None,
    validation_target: np.ndarray | None = None,
    validation_context: np.ndarray | None = None,
    checkpoint: str | Path,
    model_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    device: torch.device,
    seed: int,
    load_if_available: bool = True,
    verify_checkpoint_data: bool = False,
) -> dict[str, Any]:
    """Train an unconditional or conditional quadratic-spline flow.

    With ``verify_checkpoint_data=True``, a reusable checkpoint is bound to
    the exact target/context rows, seed, architecture, and training settings.
    ``standardize_target=False`` keeps an already natural target coordinate
    unchanged.  For a scalar linear-tail RQS, ``identity_initialization=True``
    makes the initial transport exact identity; paired with the training flag
    ``retain_initial_model=True``, that model remains a validation fallback.
    """
    checkpoint = _flow_checkpoint(checkpoint)
    if (
        load_if_available
        and checkpoint.exists()
        and not verify_checkpoint_data
    ):
        print(f"Loading spline flow from {checkpoint}")
        return load_spline_flow(checkpoint, device)

    target = _as_2d_float32(target, "target")
    if len(target) < 4:
        raise ValueError("At least four target rows are required.")
    if context is not None:
        context = _as_2d_float32(context, "context")
        if len(context) != len(target):
            raise ValueError("context must contain one row per target row.")
    explicit_validation = validation_target is not None
    if explicit_validation:
        validation_target = _as_2d_float32(
            validation_target, "validation_target"
        )
        if validation_target.shape[1] != target.shape[1]:
            raise ValueError("Training and validation targets have different widths.")
        if context is None:
            if validation_context is not None:
                raise ValueError(
                    "validation_context must be None for an unconditional flow."
                )
        else:
            if validation_context is None:
                raise ValueError("A conditional flow requires validation_context.")
            validation_context = _as_2d_float32(
                validation_context, "validation_context"
            )
            if (
                len(validation_context) != len(validation_target)
                or validation_context.shape[1] != context.shape[1]
            ):
                raise ValueError("Invalid explicit validation context shape.")
    elif validation_context is not None:
        raise ValueError("validation_context requires validation_target.")

    config = dict(model_config)
    # Persist the topology choice so corrected checkpoints are never confused
    # with historical files that silently combined two role exchanges.
    config.setdefault("use_layer_permutations", False)
    config["n_features"] = int(target.shape[1])
    config["context_features"] = 0 if context is None else int(context.shape[1])
    training_fingerprint = None
    if verify_checkpoint_data:
        fingerprint_context = (
            np.asarray(context)
            if context is not None
            else np.empty((len(target), 0), dtype=np.float32)
        )
        fingerprint_validation_target = (
            np.asarray(validation_target)
            if explicit_validation
            else np.empty((0, target.shape[1]), dtype=np.float32)
        )
        fingerprint_validation_context = (
            np.asarray(validation_context)
            if explicit_validation and validation_context is not None
            else np.empty(
                (0, 0 if context is None else context.shape[1]),
                dtype=np.float32,
            )
        )
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
            target,
            fingerprint_context,
            fingerprint_validation_target,
            fingerprint_validation_context,
            np.asarray([seed], dtype=np.int64),
            np.asarray([fingerprint_configuration]),
        )
        if load_if_available and checkpoint.exists():
            loaded = load_spline_flow(checkpoint, device)
            if loaded.get("training_fingerprint") != training_fingerprint:
                raise RuntimeError(
                    f"Checkpoint {checkpoint} was trained on different flow "
                    "rows. Bump the run tag or remove that checkpoint "
                    "explicitly."
                )
            if loaded["config"] != config:
                raise RuntimeError(
                    f"Checkpoint {checkpoint} has a different flow "
                    "architecture. Bump the run tag."
                )
            print(
                "Loaded spline flow with matching data fingerprint from "
                f"{checkpoint}"
            )
            return loaded
    if bool(config.get("standardize_target", True)):
        target_scaler = ArrayStandardizer.fit(target)
    else:
        target_scaler = ArrayStandardizer(
            mean=np.zeros(target.shape[1], dtype=np.float32),
            std=np.ones(target.shape[1], dtype=np.float32),
        )
    target_scaled = target_scaler.transform(target)
    context_scaler = None
    if context is not None:
        context_scaler = ArrayStandardizer.fit(context)
        context_scaled = context_scaler.transform(context)
    generator = torch.Generator().manual_seed(int(seed))
    target_tensor = torch.tensor(target_scaled, dtype=torch.float32)
    if explicit_validation:
        validation_target_scaled = target_scaler.transform(validation_target)
        validation_target_tensor = torch.tensor(
            validation_target_scaled, dtype=torch.float32
        )
        if context is None:
            training_dataset = TensorDataset(target_tensor)
            validation_dataset = TensorDataset(validation_target_tensor)
        else:
            context_tensor = torch.tensor(context_scaled, dtype=torch.float32)
            validation_context_tensor = torch.tensor(
                context_scaler.transform(validation_context), dtype=torch.float32
            )
            training_dataset = TensorDataset(target_tensor, context_tensor)
            validation_dataset = TensorDataset(
                validation_target_tensor, validation_context_tensor
            )
    else:
        validation_fraction = float(
            training_config.get("validation_fraction", 0.2)
        )
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between zero and one.")
        n_validation = min(
            len(target_scaled) - 1,
            max(1, int(round(validation_fraction * len(target_scaled)))),
        )
        order = torch.randperm(len(target_scaled), generator=generator).numpy()
        validation_indices = order[:n_validation]
        training_indices = order[n_validation:]
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

    # The DataLoader generator alone does not control parameter
    # initialization.  Seed all model RNGs immediately before construction so
    # a checkpoint is reproducible independently of notebook execution order.
    seed_everything(int(seed))
    flow = _build_flow_from_config(config, device)
    learning_rate = float(training_config["learning_rate"])
    weight_decay = float(training_config.get("weight_decay", 0.0))
    if weight_decay == 0.0:
        optimizer = torch.optim.Adam(flow.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.AdamW(
            flow.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    scheduler_name = str(training_config.get("lr_scheduler", "plateau")).lower()
    minimum_learning_rate = float(
        training_config.get("min_learning_rate", 1.0e-6)
    )
    scheduler_factor = float(training_config.get("lr_scheduler_factor", 0.3))
    scheduler_patience = int(training_config.get("lr_scheduler_patience", 2))
    if scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=minimum_learning_rate,
        )
        scheduler_uses_validation = True
    elif scheduler_name == "step":
        if scheduler_patience <= 0:
            raise ValueError("Step LR interval must be strictly positive.")
        if not 0.0 < scheduler_factor < 1.0:
            raise ValueError("Step LR factor must lie strictly between zero and one.")
        if not 0.0 < minimum_learning_rate <= learning_rate:
            raise ValueError(
                "Minimum learning rate must be positive and no larger than the initial rate."
            )
        floor_factor = minimum_learning_rate / learning_rate

        def step_multiplier(completed_epochs):
            return max(
                floor_factor,
                scheduler_factor ** (int(completed_epochs) // scheduler_patience),
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=step_multiplier
        )
        scheduler_uses_validation = False
    elif scheduler_name in {"progressive", "six_equal_log10_plateaus"}:
        progressive_epochs = int(training_config["n_epochs"])
        if progressive_epochs < 6:
            # Smoke runs still exercise the scheduler without zero-length bins.
            progressive_levels = progressive_epochs
        else:
            progressive_levels = 6
        floor_factor = minimum_learning_rate / learning_rate

        def progressive_multiplier(completed_epochs):
            level = min(
                progressive_levels - 1,
                int(completed_epochs * progressive_levels / progressive_epochs),
            )
            return max(floor_factor, 0.1 ** level)

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=progressive_multiplier
        )
        scheduler_uses_validation = False
    else:
        raise ValueError(
            f"Unknown lr_scheduler={scheduler_name!r}; expected 'plateau', "
            "'step', or 'six_equal_log10_plateaus'."
        )
    n_epochs = int(training_config["n_epochs"])
    patience = int(training_config.get("patience", n_epochs))
    print_every = max(1, int(training_config.get("print_every", 1)))
    gradient_clip = float(training_config.get("gradient_clip", 5.0))
    retain_initial_model = bool(
        training_config.get("retain_initial_model", False)
    )
    minimum_validation_improvement = float(
        training_config.get("minimum_validation_improvement", 1.0e-4)
    )
    checkpoint_selection = str(
        training_config.get("checkpoint_selection", "best_validation")
    ).lower()
    if checkpoint_selection not in {"best_validation", "final_epoch"}:
        raise ValueError(
            "checkpoint_selection must be 'best_validation' or 'final_epoch'."
        )
    if minimum_validation_improvement < 0.0:
        raise ValueError("minimum_validation_improvement must be non-negative.")
    best_validation = math.inf
    best_state = None
    stale_epochs = 0
    history = {"train": [], "validation": [], "learning_rate": []}
    best_epoch = 0

    def validation_nll() -> float:
        flow.eval()
        loss_sum = 0.0
        row_count = 0
        with torch.no_grad():
            for batch in validation_loader:
                target_batch = batch[0].to(device)
                context_batch = batch[1].to(device) if len(batch) == 2 else None
                losses = -flow.log_prob(
                    target_batch, context=context_batch
                )
                loss_sum += float(losses.sum().detach().cpu())
                row_count += int(len(target_batch))
        return loss_sum / max(1, row_count)

    if retain_initial_model:
        initial_validation = validation_nll()
        best_validation = initial_validation
        best_state = copy.deepcopy(flow.state_dict())
        best_epoch = 0
        history["initial_validation"] = [initial_validation]
        print(
            "  retained initial flow as validation baseline: "
            f"{initial_validation:.6f}"
        )

    print(
        f"Training {'conditional' if context is not None else 'unconditional'} "
        f"quadratic-spline flow on {len(training_dataset):,} rows"
    )
    for epoch in range(1, n_epochs + 1):
        flow.train()
        train_loss_sum = 0.0
        train_row_count = 0
        for batch in training_loader:
            target_batch = batch[0].to(device)
            context_batch = batch[1].to(device) if len(batch) == 2 else None
            loss = -flow.log_prob(target_batch, context=context_batch).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), gradient_clip)
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu()) * len(target_batch)
            train_row_count += int(len(target_batch))

        train_loss = train_loss_sum / max(1, train_row_count)
        validation_loss = validation_nll()
        history["train"].append(train_loss)
        history["validation"].append(validation_loss)
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])
        if scheduler_uses_validation:
            scheduler.step(validation_loss)
        else:
            scheduler.step()

        if validation_loss < best_validation - minimum_validation_improvement:
            best_validation = validation_loss
            best_state = copy.deepcopy(flow.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % print_every == 0 or epoch == n_epochs:
            print(
                f"  epoch {epoch:02d}/{n_epochs}: train={train_loss:.4f}, "
                f"validation={validation_loss:.4f}, "
                f"lr={history['learning_rate'][-1]:.1e}"
            )
        # A progressive run must execute at least one optimizer epoch at every
        # declared LR plateau.  Early stopping is therefore inactive until the
        # final sixth has actually been used.
        final_plateau_used = (
            scheduler_name not in {"progressive", "six_equal_log10_plateaus"}
            or epoch >= (5 * n_epochs) // 6 + 1
        )
        if stale_epochs >= patience and final_plateau_used:
            print(f"  early stopping after {epoch} epochs")
            break

    if checkpoint_selection == "best_validation" and best_state is not None:
        flow.load_state_dict(best_state)
    flow.eval()
    history["best_validation"] = [best_validation]
    history["best_epoch"] = [best_epoch]
    if checkpoint_selection == "final_epoch":
        history["selected_validation"] = [float(history["validation"][-1])]
        history["selected_epoch"] = [int(len(history["validation"]))]
    else:
        history["selected_validation"] = [best_validation]
        history["selected_epoch"] = [best_epoch]
    history["checkpoint_selection"] = [checkpoint_selection]
    torch.save(
        {
            "state_dict": flow.state_dict(),
            "config": config,
            "target_mean": target_scaler.mean,
            "target_std": target_scaler.std,
            "context_mean": None if context_scaler is None else context_scaler.mean,
            "context_std": None if context_scaler is None else context_scaler.std,
            "history": history,
            "training_fingerprint": training_fingerprint,
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
def spline_flow_log_prob_with_base_scale(
    flow_pack: Mapping[str, Any],
    target: np.ndarray,
    *,
    context: np.ndarray | None = None,
    base_scale: float,
    batch_size: int = 65_536,
) -> np.ndarray:
    """Evaluate a fitted transport with an isotropic broadened Gaussian base.

    The learned target-to-base transformation and all stored standardizers are
    unchanged.  Only the base density is replaced by ``N(0, base_scale**2 I)``.
    Setting ``base_scale=1`` is numerically equivalent to
    :func:`spline_flow_log_prob`.
    """

    base_scale = float(base_scale)
    if not np.isfinite(base_scale) or base_scale <= 0.0:
        raise ValueError("base_scale must be finite and strictly positive.")
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
    log_scale = math.log(base_scale)
    log_two_pi = math.log(2.0 * math.pi)
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
        embedded_context = flow._embedding_net(context_tensor)
        base, log_abs_det = flow._transform(
            target_tensor, context=embedded_context
        )
        base_log_probability = -0.5 * torch.sum(
            torch.square(base / base_scale) + log_two_pi + 2.0 * log_scale,
            dim=1,
        )
        log_probability = (
            base_log_probability + log_abs_det
        ).detach().cpu().numpy()
        chunks.append(log_probability + target_scaler.log_det_to_standard)
    if not chunks:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(chunks)


@torch.no_grad()
def scalar_spline_flow_cdf(
    flow_pack: Mapping[str, Any],
    target: np.ndarray,
    *,
    context: np.ndarray | None = None,
    batch_size: int = 65_536,
) -> np.ndarray:
    """Evaluate the analytic CDF of a scalar standard-normal-base flow.

    A one-dimensional rational-quadratic-spline flow is a monotone map from
    its target coordinate to the standard-normal base coordinate.  If that
    full data-to-base map is ``z = f(y; context)``, its conditional CDF is
    simply ``Phi(z)``.  This helper includes the affine target and context
    standardizers stored by :func:`train_spline_flow`.  Evaluation follows
    the flow parameter dtype, so promoting a fitted scalar flow to float64
    also promotes its complete CDF path without refitting its standardizers.

    The function is intentionally restricted to scalar targets.  In more
    than one target dimension a normalizing flow still supplies a density,
    but applying a component-wise base CDF would not be the joint CDF.
    """

    flow = flow_pack["flow"]
    torch_dtype, numpy_dtype = _flow_evaluation_dtypes(flow)
    target = _as_2d_floating(target, "target", numpy_dtype)
    if target.shape[1] != 1:
        raise ValueError("scalar_spline_flow_cdf requires one target column.")
    conditional = flow_pack["context_scaler"] is not None
    if conditional:
        if context is None:
            raise ValueError("This flow requires context.")
        context = _as_2d_floating(context, "context", numpy_dtype)
        if len(context) != len(target):
            raise ValueError("context must contain one row per target row.")
    elif context is not None:
        raise ValueError("This flow is unconditional; context must be None.")

    device = next(flow.parameters()).device
    target_scaler = flow_pack["target_scaler"]
    context_scaler = flow_pack["context_scaler"]
    flow.eval()
    chunks = []
    for start in range(0, len(target), int(batch_size)):
        stop = start + int(batch_size)
        target_tensor = torch.tensor(
            _standardizer_transform(
                target_scaler, target[start:stop], numpy_dtype
            ),
            dtype=torch_dtype,
            device=device,
        )
        context_tensor = None
        if conditional:
            context_tensor = torch.tensor(
                _standardizer_transform(
                    context_scaler, context[start:stop], numpy_dtype
                ),
                dtype=torch_dtype,
                device=device,
            )
        base_coordinate = flow.transform_to_noise(
            target_tensor, context=context_tensor
        )
        probability = ndtr(
            base_coordinate[:, 0].detach().cpu().numpy().astype(np.float64)
        )
        chunks.append(probability)
    if not chunks:
        return np.empty(0, dtype=numpy_dtype)
    return np.concatenate(chunks)


@torch.no_grad()
def scalar_spline_flow_icdf(
    flow_pack: Mapping[str, Any],
    probability: np.ndarray,
    *,
    context: np.ndarray | None = None,
    batch_size: int = 65_536,
) -> np.ndarray:
    """Analytically invert a scalar standard-normal-base flow CDF.

    Probabilities are first mapped through ``Phi^{-1}``, then through the
    inverse spline transport, and finally through the stored inverse target
    standardization.  One context row is required per probability for a
    conditional flow.  As for :func:`scalar_spline_flow_cdf`, all arithmetic
    follows the fitted flow's current float32 or float64 parameter dtype.
    """

    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    if not np.isfinite(probability).all() or np.any(
        (probability <= 0.0) | (probability >= 1.0)
    ):
        raise ValueError("probability must be finite and strictly in (0, 1).")
    flow = flow_pack["flow"]
    torch_dtype, numpy_dtype = _flow_evaluation_dtypes(flow)
    conditional = flow_pack["context_scaler"] is not None
    if conditional:
        if context is None:
            raise ValueError("This flow requires context.")
        context = _as_2d_floating(context, "context", numpy_dtype)
        if len(context) != len(probability):
            raise ValueError(
                "context must contain one row per requested probability."
            )
    elif context is not None:
        raise ValueError("This flow is unconditional; context must be None.")

    device = next(flow.parameters()).device
    target_scaler = flow_pack["target_scaler"]
    context_scaler = flow_pack["context_scaler"]
    flow.eval()
    chunks = []
    for start in range(0, len(probability), int(batch_size)):
        stop = start + int(batch_size)
        base_coordinate = torch.tensor(
            ndtri(probability[start:stop])[:, None],
            dtype=torch_dtype,
            device=device,
        )
        context_tensor = None
        if conditional:
            context_tensor = torch.tensor(
                _standardizer_transform(
                    context_scaler, context[start:stop], numpy_dtype
                ),
                dtype=torch_dtype,
                device=device,
            )
        embedded_context = flow._embedding_net(context_tensor)
        target_scaled, _ = flow._transform.inverse(
            base_coordinate, context=embedded_context
        )
        chunks.append(
            _standardizer_inverse(
                target_scaler,
                target_scaled.detach().cpu().numpy(),
                numpy_dtype,
            )[:, 0]
        )
    if not chunks:
        return np.empty(0, dtype=numpy_dtype)
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


@torch.no_grad()
def sample_spline_flow_with_base_scale(
    flow_pack: Mapping[str, Any],
    n_samples: int,
    *,
    context: np.ndarray | None = None,
    base_scale: float,
    batch_size: int = 16_384,
) -> np.ndarray:
    """Sample a fitted transport from ``N(0, base_scale**2 I)`` latent draws.

    Return shapes follow :func:`sample_spline_flow`.  This is a proposal-only
    operation: the fitted transformation is not retrained or modified.
    """

    n_samples = int(n_samples)
    base_scale = float(base_scale)
    if n_samples < 1:
        raise ValueError("n_samples must be positive.")
    if not np.isfinite(base_scale) or base_scale <= 0.0:
        raise ValueError("base_scale must be finite and strictly positive.")
    if base_scale == 1.0:
        return sample_spline_flow(
            flow_pack, n_samples, context=context, batch_size=batch_size
        )

    flow = flow_pack["flow"]
    device = next(flow.parameters()).device
    target_scaler = flow_pack["target_scaler"]
    context_scaler = flow_pack["context_scaler"]
    n_features = int(np.asarray(target_scaler.mean).size)
    flow.eval()

    if context_scaler is None:
        if context is not None:
            raise ValueError("This flow is unconditional; context must be None.")
        chunks = []
        remaining = n_samples
        while remaining:
            current = min(int(batch_size), remaining)
            base = base_scale * torch.randn(
                current, n_features, dtype=torch.float32, device=device
            )
            values, _ = flow._transform.inverse(base, context=None)
            chunks.append(target_scaler.inverse(values.detach().cpu().numpy()))
            remaining -= current
        return np.concatenate(chunks, axis=0)

    if context is None:
        raise ValueError("This flow requires context.")
    context = _as_2d_float32(context, "context")
    if len(context) == 1:
        context_tensor = torch.tensor(
            context_scaler.transform(context),
            dtype=torch.float32,
            device=device,
        )
        embedded_context = flow._embedding_net(context_tensor)
        chunks = []
        remaining = n_samples
        while remaining:
            current = min(int(batch_size), remaining)
            base = base_scale * torch.randn(
                current, n_features, dtype=torch.float32, device=device
            )
            repeated_context = torch.repeat_interleave(
                embedded_context, current, dim=0
            )
            values, _ = flow._transform.inverse(
                base, context=repeated_context
            )
            chunks.append(target_scaler.inverse(values.detach().cpu().numpy()))
            remaining -= current
        return np.concatenate(chunks, axis=0)
    context_batch_size = max(1, int(batch_size) // n_samples)
    chunks = []
    for start in range(0, len(context), context_batch_size):
        local_context = context[start : start + context_batch_size]
        context_tensor = torch.tensor(
            context_scaler.transform(local_context),
            dtype=torch.float32,
            device=device,
        )
        embedded_context = flow._embedding_net(context_tensor)
        base = base_scale * torch.randn(
            len(local_context),
            n_samples,
            n_features,
            dtype=torch.float32,
            device=device,
        )
        flat_base = base.reshape(-1, n_features)
        flat_context = torch.repeat_interleave(
            embedded_context, n_samples, dim=0
        )
        values, _ = flow._transform.inverse(
            flat_base, context=flat_context
        )
        original = target_scaler.inverse(
            values.detach().cpu().numpy()
        ).reshape(len(local_context), n_samples, n_features)
        chunks.append(original)
    output = np.concatenate(chunks, axis=0)
    if len(context) == 1:
        return output[0]
    return output


def train_spline_flow_ensemble(
    target: np.ndarray,
    *,
    context: np.ndarray | None,
    validation_target: np.ndarray | None = None,
    validation_context: np.ndarray | None = None,
    checkpoint: str | Path,
    ensemble_size: int,
    model_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    device: torch.device,
    seed: int,
    load_if_available: bool = True,
    verify_checkpoint_data: bool = False,
) -> list[dict[str, Any]]:
    """Train independent flow members representing an equal-weight mixture.

    A member index is inserted before the checkpoint suffix.  Every member
    retains :func:`train_spline_flow`'s exact data/config fingerprint while
    receiving an independent initialization, data split, and shuffle seed.
    The returned object is intentionally a plain list: mixture density and
    sampling semantics are supplied by the explicit helpers below.
    """

    ensemble_size = int(ensemble_size)
    if ensemble_size < 1:
        raise ValueError("ensemble_size must be positive.")
    checkpoint = Path(checkpoint)
    suffix = checkpoint.suffix or ".pt"
    stem = checkpoint.stem if checkpoint.suffix else checkpoint.name
    parent = checkpoint.parent
    members = []
    for member_index in range(ensemble_size):
        member_seed = int(seed) + 10_007 * member_index
        torch.manual_seed(member_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(member_seed)
        member_checkpoint = parent / (
            f"{stem}.member_{member_index:02d}{suffix}"
        )
        print(
            f"Flow-mixture member {member_index + 1}/{ensemble_size}: "
            f"{member_checkpoint}"
        )
        members.append(
            train_spline_flow(
                target,
                context=context,
                validation_target=validation_target,
                validation_context=validation_context,
                checkpoint=member_checkpoint,
                model_config=model_config,
                training_config=training_config,
                device=device,
                seed=member_seed,
                load_if_available=load_if_available,
                verify_checkpoint_data=verify_checkpoint_data,
            )
        )
    return members


def _validate_flow_ensemble(
    flow_packs: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    members = list(flow_packs)
    if not members:
        raise ValueError("A flow ensemble must contain at least one member.")
    target_dims = {
        int(np.asarray(member["target_scaler"].mean).size)
        for member in members
    }
    conditional = {
        member["context_scaler"] is not None for member in members
    }
    context_dims = {
        0
        if member["context_scaler"] is None
        else int(np.asarray(member["context_scaler"].mean).size)
        for member in members
    }
    configurations = {
        json.dumps(dict(member.get("config", {})), sort_keys=True, default=str)
        for member in members
    }
    if (
        len(target_dims) != 1
        or len(conditional) != 1
        or len(context_dims) != 1
        or len(configurations) != 1
    ):
        raise ValueError(
            "Every flow-mixture member must have matching target/context "
            "dimensions and architecture configuration."
        )
    return members


def spline_flow_ensemble_log_prob(
    flow_packs: Sequence[Mapping[str, Any]],
    target: np.ndarray,
    *,
    context: np.ndarray | None = None,
    batch_size: int = 65_536,
) -> np.ndarray:
    """Evaluate the equal-weight mixture density in log space."""

    members = _validate_flow_ensemble(flow_packs)
    member_log_probabilities = np.stack(
        [
            spline_flow_log_prob(
                member,
                target,
                context=context,
                batch_size=batch_size,
            )
            for member in members
        ],
        axis=0,
    )
    return logsumexp(member_log_probabilities, axis=0) - math.log(len(members))


def spline_flow_defensive_ensemble_log_prob(
    flow_packs: Sequence[Mapping[str, Any]],
    target: np.ndarray,
    *,
    context: np.ndarray | None = None,
    broad_fraction: float,
    broad_base_scale: float,
    batch_size: int = 65_536,
) -> np.ndarray:
    """Evaluate an exact nominal-plus-broadened equal-weight flow mixture."""

    members = _validate_flow_ensemble(flow_packs)
    broad_fraction = float(broad_fraction)
    broad_base_scale = float(broad_base_scale)
    if not 0.0 <= broad_fraction < 1.0:
        raise ValueError("broad_fraction must lie in [0, 1).")
    if not np.isfinite(broad_base_scale) or broad_base_scale <= 0.0:
        raise ValueError("broad_base_scale must be finite and positive.")
    nominal = spline_flow_ensemble_log_prob(
        members, target, context=context, batch_size=batch_size
    )
    if broad_fraction == 0.0:
        return nominal
    broad_members = np.stack(
        [
            spline_flow_log_prob_with_base_scale(
                member,
                target,
                context=context,
                base_scale=broad_base_scale,
                batch_size=batch_size,
            )
            for member in members
        ],
        axis=0,
    )
    broad = logsumexp(broad_members, axis=0) - math.log(len(members))
    return np.logaddexp(
        math.log1p(-broad_fraction) + nominal,
        math.log(broad_fraction) + broad,
    )


def scalar_spline_flow_ensemble_cdf(
    flow_packs: Sequence[Mapping[str, Any]],
    target: np.ndarray,
    *,
    context: np.ndarray | None = None,
    batch_size: int = 65_536,
) -> np.ndarray:
    """Evaluate the CDF of an equal-weight scalar-flow mixture."""

    members = _validate_flow_ensemble(flow_packs)
    probabilities = np.stack(
        [
            scalar_spline_flow_cdf(
                member,
                target,
                context=context,
                batch_size=batch_size,
            )
            for member in members
        ],
        axis=0,
    )
    return probabilities.mean(axis=0)


def scalar_spline_flow_ensemble_icdf(
    flow_packs: Sequence[Mapping[str, Any]],
    probability: np.ndarray,
    *,
    context: np.ndarray | None = None,
    batch_size: int = 65_536,
    tolerance: float = 2.0e-7,
    max_iterations: int = 80,
) -> np.ndarray:
    """Numerically invert an equal-weight scalar-flow mixture CDF.

    A mixture of monotone flows has an analytic CDF but generally no closed
    inverse.  Brackets are expanded using only the fitted flows, followed by
    vectorized bisection.  This routine is intended for audits and quantiles;
    ordinary draws should use :func:`sample_spline_flow_ensemble`.
    """

    members = _validate_flow_ensemble(flow_packs)
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    if not np.isfinite(probability).all() or np.any(
        (probability <= 0.0) | (probability >= 1.0)
    ):
        raise ValueError("probability must be finite and strictly in (0, 1).")
    conditional = members[0]["context_scaler"] is not None
    if conditional:
        if context is None:
            raise ValueError("This flow mixture requires context.")
        context = _as_2d_float32(context, "context")
        if len(context) != len(probability):
            raise ValueError(
                "context must contain one row per requested probability."
            )
    elif context is not None:
        raise ValueError("This flow mixture is unconditional; context must be None.")

    means = np.asarray(
        [float(member["target_scaler"].mean[0]) for member in members]
    )
    scales = np.asarray(
        [float(member["target_scaler"].std[0]) for member in members]
    )
    center = float(np.median(means))
    radius = max(8.0, float(np.max(np.abs(means - center) + 12.0 * scales)))
    lower = np.full(len(probability), center - radius, dtype=np.float64)
    upper = np.full(len(probability), center + radius, dtype=np.float64)

    def mixture_cdf(values: np.ndarray) -> np.ndarray:
        return scalar_spline_flow_ensemble_cdf(
            members,
            np.asarray(values, dtype=np.float32)[:, None],
            context=context,
            batch_size=batch_size,
        ).astype(np.float64)

    for _ in range(16):
        low_bad = mixture_cdf(lower) > probability
        high_bad = mixture_cdf(upper) < probability
        if not (np.any(low_bad) or np.any(high_bad)):
            break
        width = upper - lower
        lower[low_bad] -= width[low_bad]
        upper[high_bad] += width[high_bad]
    else:
        raise RuntimeError("Could not bracket every requested mixture quantile.")

    for _ in range(int(max_iterations)):
        midpoint = 0.5 * (lower + upper)
        below = mixture_cdf(midpoint) < probability
        lower[below] = midpoint[below]
        upper[~below] = midpoint[~below]
        if float(np.max(upper - lower, initial=0.0)) <= float(tolerance):
            break
    return 0.5 * (lower + upper)


def sample_spline_flow_ensemble(
    flow_packs: Sequence[Mapping[str, Any]],
    n_samples: int,
    *,
    context: np.ndarray | None = None,
    seed: int,
    batch_size: int = 16_384,
    allocation: str = "balanced",
) -> np.ndarray:
    """Draw from an equal-weight flow mixture.

    ``allocation="balanced"`` randomizes member labels while making their
    per-context counts differ by at most one.  Every individual draw has the
    correct mixture marginal and conditional-moment banks get lower component
    allocation noise, but draws within one context are not IID.

    ``allocation="iid"`` uses independent uniform categorical labels.  Use
    this mode whenever an ordinary IID sample-variance formula is required,
    notably for the raw-evidence bridge loss.  Shapes match
    :func:`sample_spline_flow`.  Torch RNG state is restored on return.
    """

    members = _validate_flow_ensemble(flow_packs)
    n_samples = int(n_samples)
    if n_samples < 1:
        raise ValueError("n_samples must be positive.")
    allocation = str(allocation).lower()
    if allocation not in {"balanced", "iid"}:
        raise ValueError("allocation must be 'balanced' or 'iid'.")
    conditional = members[0]["context_scaler"] is not None
    if conditional:
        if context is None:
            raise ValueError("This flow mixture requires context.")
        context_rows = _as_2d_float32(context, "context")
    else:
        if context is not None:
            raise ValueError(
                "This flow mixture is unconditional; context must be None."
            )
        context_rows = np.empty((1, 0), dtype=np.float32)

    rng = np.random.default_rng(int(seed))
    n_context = len(context_rows)
    n_members = len(members)
    if allocation == "iid":
        assignments = rng.integers(
            0, n_members, size=(n_context, n_samples), dtype=np.int64
        )
    elif n_samples == 1:
        # The one-draw class-building path can contain O(10^6) contexts.
        assignments = rng.integers(
            0, n_members, size=(n_context, 1), dtype=np.int64
        )
    else:
        assignments = np.empty((n_context, n_samples), dtype=np.int64)
        base_count, remainder = divmod(n_samples, n_members)
        for row in range(n_context):
            labels = np.repeat(np.arange(n_members), base_count)
            if remainder:
                labels = np.concatenate(
                    [labels, rng.permutation(n_members)[:remainder]]
                )
            rng.shuffle(labels)
            assignments[row] = labels

    n_features = int(np.asarray(members[0]["target_scaler"].mean).size)
    output = np.empty((n_context, n_samples, n_features), dtype=np.float32)
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    try:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        for member_index, member in enumerate(members):
            counts = np.sum(assignments == member_index, axis=1)
            for count in np.unique(counts):
                count = int(count)
                if count == 0:
                    continue
                rows = np.flatnonzero(counts == count)
                member_context = context_rows[rows] if conditional else None
                values = sample_spline_flow(
                    member,
                    count,
                    context=member_context,
                    batch_size=batch_size,
                )
                if len(rows) == 1:
                    values = np.asarray(values)[None, :, :]
                positions = np.nonzero(
                    assignments[rows] == member_index
                )[1].reshape(len(rows), count)
                output[rows[:, None], positions, :] = values
    finally:
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)

    if not conditional or n_context == 1:
        return output[0]
    return output


def sample_spline_flow_defensive_ensemble(
    flow_packs: Sequence[Mapping[str, Any]],
    n_samples: int,
    *,
    context: np.ndarray | None = None,
    seed: int,
    broad_fraction: float,
    broad_base_scale: float,
    batch_size: int = 16_384,
) -> np.ndarray:
    """Draw IID samples from the exact nominal-plus-broadened flow mixture.

    Each draw first selects a flow member uniformly and then selects the
    nominal or broadened Gaussian base with probabilities ``1-epsilon`` and
    ``epsilon``.  The Torch RNG state is restored on return.
    """

    members = _validate_flow_ensemble(flow_packs)
    n_samples = int(n_samples)
    broad_fraction = float(broad_fraction)
    broad_base_scale = float(broad_base_scale)
    if n_samples < 1:
        raise ValueError("n_samples must be positive.")
    if not 0.0 <= broad_fraction < 1.0:
        raise ValueError("broad_fraction must lie in [0, 1).")
    if not np.isfinite(broad_base_scale) or broad_base_scale <= 0.0:
        raise ValueError("broad_base_scale must be finite and positive.")
    if broad_fraction == 0.0:
        return sample_spline_flow_ensemble(
            members,
            n_samples,
            context=context,
            seed=seed,
            batch_size=batch_size,
            allocation="iid",
        )

    conditional = members[0]["context_scaler"] is not None
    if conditional:
        if context is None:
            raise ValueError("This flow mixture requires context.")
        context_rows = _as_2d_float32(context, "context")
    else:
        if context is not None:
            raise ValueError(
                "This flow mixture is unconditional; context must be None."
            )
        context_rows = np.empty((1, 0), dtype=np.float32)

    rng = np.random.default_rng(int(seed))
    n_context = len(context_rows)
    n_members = len(members)
    member_assignments = rng.integers(
        0, n_members, size=(n_context, n_samples), dtype=np.int64
    )
    broad_assignments = rng.random((n_context, n_samples)) < broad_fraction
    n_features = int(np.asarray(members[0]["target_scaler"].mean).size)
    output = np.empty((n_context, n_samples, n_features), dtype=np.float32)

    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    try:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        for member_index, member in enumerate(members):
            for broad in (False, True):
                component = (member_assignments == member_index) & (
                    broad_assignments == broad
                )
                counts = np.sum(component, axis=1)
                for count_value in np.unique(counts):
                    count_value = int(count_value)
                    if count_value == 0:
                        continue
                    rows = np.flatnonzero(counts == count_value)
                    member_context = context_rows[rows] if conditional else None
                    values = sample_spline_flow_with_base_scale(
                        member,
                        count_value,
                        context=member_context,
                        base_scale=(broad_base_scale if broad else 1.0),
                        batch_size=batch_size,
                    )
                    if len(rows) == 1:
                        values = np.asarray(values)[None, :, :]
                    positions = np.nonzero(component[rows])[1].reshape(
                        len(rows), count_value
                    )
                    output[rows[:, None], positions, :] = values
    finally:
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)

    if not conditional or n_context == 1:
        return output[0]
    return output


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
        "training_fingerprint": saved.get("training_fingerprint"),
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
            "training_fingerprint": pack.get("training_fingerprint"),
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
    positive_weights: np.ndarray | None = None,
    negative_weights: np.ndarray | None = None,
    verify_checkpoint_data: bool = False,
) -> dict[str, Any]:
    """Train a balanced neural classifier whose logit estimates log p/q.

    ``paired_group_ids`` is used by the dual hNPE--hNDE exercises.  Each
    positive/negative row pair is generated from one simulator index and must
    remain on the same side of the train/validation boundary.  Splitting the
    concatenated class rows independently leaks the shared parameter or
    observation into validation and can make a memorizing network appear to
    generalize.  Optional class weights implement importance-weighted BCE.
    They are normalized separately within each class and split so the
    effective classifier prior remains exactly balanced.  At least one row in
    each class must retain positive weight in both training and validation.
    ``verify_checkpoint_data`` makes a reusable checkpoint conditional on the
    exact class arrays, pair ids, weights, seed, model architecture, and
    training configuration.  It is intended for calibration stages whose
    inputs can change while a checkpoint path remains the same; legacy
    workshop checkpoints keep the faster path-only behavior by default.
    """
    checkpoint = _flow_checkpoint(checkpoint)
    if (
        load_if_available
        and checkpoint.exists()
        and not verify_checkpoint_data
    ):
        print(f"Loading ratio classifier from {checkpoint}")
        return load_ratio_classifier(checkpoint, device)
    importance_weights_supplied = (
        positive_weights is not None or negative_weights is not None
    )
    positive = _as_2d_float32(positive, "positive")
    negative = _as_2d_float32(negative, "negative")
    if positive.shape[1] != negative.shape[1]:
        raise ValueError("positive and negative must have the same columns.")

    def validate_class_weights(
        values: np.ndarray | None,
        n_rows: int,
        name: str,
    ) -> np.ndarray:
        if values is None:
            return np.ones(n_rows, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if len(values) != n_rows:
            raise ValueError(f"{name} must contain one value per class row.")
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError(f"{name} must be finite and non-negative.")
        if not float(values.sum()) > 0.0:
            raise ValueError(f"{name} must have positive total weight.")
        return values

    positive_weights = validate_class_weights(
        positive_weights, len(positive), "positive_weights"
    )
    negative_weights = validate_class_weights(
        negative_weights, len(negative), "negative_weights"
    )

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

    training_fingerprint = None
    if verify_checkpoint_data:
        fingerprint_group_ids = (
            np.asarray(paired_group_ids)
            if paired_group_ids is not None
            else np.empty(0, dtype=np.int64)
        )
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
            positive,
            negative,
            positive_weights,
            negative_weights,
            fingerprint_group_ids,
            np.asarray([seed], dtype=np.int64),
            np.asarray([fingerprint_configuration]),
        )
        if load_if_available and checkpoint.exists():
            loaded = load_ratio_classifier(checkpoint, device)
            expected_config = dict(model_config)
            expected_config["n_features"] = int(positive.shape[1])
            if loaded.get("training_fingerprint") != training_fingerprint:
                raise RuntimeError(
                    f"Checkpoint {checkpoint} was trained on different "
                    "classifier rows. Bump the calibration run tag or "
                    "remove that checkpoint explicitly."
                )
            if loaded["config"] != expected_config:
                raise RuntimeError(
                    f"Checkpoint {checkpoint} has a different ratio-model "
                    "architecture. Bump the calibration run tag or remove "
                    "that checkpoint explicitly."
                )
            print(
                "Loaded ratio classifier with matching data fingerprint "
                f"from {checkpoint}"
            )
            return loaded

    rng = np.random.default_rng(seed)
    if paired_group_ids is None:
        positive_indices = rng.choice(
            len(positive), n_per_class, replace=False
        )
        negative_indices = rng.choice(
            len(negative), n_per_class, replace=False
        )
        positive = positive[positive_indices]
        negative = negative[negative_indices]
        positive_weights = positive_weights[positive_indices]
        negative_weights = negative_weights[negative_indices]
    values = np.concatenate([positive, negative], axis=0)
    labels = np.concatenate(
        [np.ones(n_per_class), np.zeros(n_per_class)]
    ).astype(np.float32)
    raw_weights = np.concatenate(
        [positive_weights, negative_weights]
    ).astype(np.float64)

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

    def balanced_split_weights(
        indices: np.ndarray, split_name: str
    ) -> np.ndarray:
        split_labels = labels[indices]
        split_weights = raw_weights[indices].copy()
        for class_label in (0.0, 1.0):
            class_mask = split_labels == class_label
            class_sum = float(split_weights[class_mask].sum())
            if not class_sum > 0.0:
                raise ValueError(
                    f"{split_name} has no positive weight for class "
                    f"{int(class_label)}."
                )
            # Each class contributes half of the split loss.  The overall
            # mean weight is one, keeping loss scales comparable to ordinary
            # balanced BCE.
            split_weights[class_mask] *= (
                0.5 * len(indices) / class_sum
            )
        return split_weights.astype(np.float32)

    training_weights = balanced_split_weights(
        training_indices, "training split"
    )
    validation_weights = balanced_split_weights(
        validation_indices, "validation split"
    )
    training_weights_tensor = torch.tensor(
        training_weights, dtype=torch.float32
    )
    validation_weights_tensor = torch.tensor(
        validation_weights, dtype=torch.float32
    )
    training_dataset = TensorDataset(
        values_tensor[training_indices],
        labels_tensor[training_indices],
        training_weights_tensor,
    )
    validation_dataset = TensorDataset(
        values_tensor[validation_indices],
        labels_tensor[validation_indices],
        validation_weights_tensor,
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
        "weighted_bce": importance_weights_supplied,
        "nonuniform_weights": bool(
            not np.allclose(positive_weights, 1.0)
            or not np.allclose(negative_weights, 1.0)
        ),
    }
    print(
        f"Training balanced ratio classifier on {n_per_class:,} rows per class "
        f"with {split_strategy} validation"
    )

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_losses = []
        for values_batch, labels_batch, weights_batch in training_loader:
            values_batch = values_batch.to(device)
            labels_batch = labels_batch.to(device)
            weights_batch = weights_batch.to(device)
            event_loss = F.binary_cross_entropy_with_logits(
                model(values_batch), labels_batch, reduction="none"
            )
            loss = torch.mean(weights_batch * event_loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        validation_loss_numerator = 0.0
        validation_weight_sum = 0.0
        with torch.no_grad():
            for values_batch, labels_batch, weights_batch in validation_loader:
                values_batch = values_batch.to(device)
                labels_batch = labels_batch.to(device)
                weights_batch = weights_batch.to(device)
                event_loss = F.binary_cross_entropy_with_logits(
                    model(values_batch), labels_batch, reduction="none"
                )
                validation_loss_numerator += float(
                    torch.sum(weights_batch * event_loss).detach().cpu()
                )
                validation_weight_sum += float(
                    torch.sum(weights_batch).detach().cpu()
                )
        train_loss = float(np.mean(train_losses))
        validation_loss = validation_loss_numerator / validation_weight_sum
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
        "training_fingerprint": training_fingerprint,
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


# ---------------------------------------------------------------------------
# Paper-study data contract
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Seed all modern-runtime RNGs and enforce the frozen strict policy."""

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
    if (
        hasattr(torch.backends, "cuda")
        and hasattr(torch.backends.cuda, "matmul")
        and hasattr(torch.backends.cuda.matmul, "allow_tf32")
    ):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _write_csv(path: str | Path, frame: pd.DataFrame) -> None:
    """Atomically publish a CSV so shard aggregation is crash-safe."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _implementation_manifest() -> dict[str, Any]:
    """Fingerprint the checked-in runtime that produced an artifact."""

    source_root = Path(__file__).resolve().parent
    filenames = (
        "config.py",
        "utils.py",
        "utils_ratio.py",
        "utils_jana.py",
        "utils_plotting.py",
        "generate_notebooks.py",
        "requirements_jana.txt",
    )
    hashes = {}
    for filename in filenames:
        path = source_root / filename
        if path.is_file():
            hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    git_commit = None
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source_root,
            check=True,
            text=True,
            capture_output=True,
        )
        git_commit = completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    package_versions = {"numpy": np.__version__, "torch": torch.__version__}
    try:
        from importlib.metadata import PackageNotFoundError, version

        for package in ("nflows", "sbibm", "scipy", "pandas"):
            try:
                package_versions[package] = version(package)
            except PackageNotFoundError:
                package_versions[package] = None
    except ImportError:  # pragma: no cover
        pass
    return {
        "git_commit": git_commit,
        "source_sha256": hashes,
        "package_versions": package_versions,
    }


def _array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("utf-8"))
        digest.update(json.dumps(contiguous.shape).encode("utf-8"))
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _mapping_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class BoxUniformLogitTransform:
    """Logit transform for the bounded, factorized SLCP prior."""

    low: np.ndarray
    high: np.ndarray
    clip: float = 1.0e-6

    def __post_init__(self) -> None:
        low = np.asarray(self.low, dtype=np.float64).reshape(-1)
        high = np.asarray(self.high, dtype=np.float64).reshape(-1)
        if low.shape != high.shape or np.any(high <= low):
            raise ValueError("Invalid bounded-prior limits.")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    @property
    def width(self) -> np.ndarray:
        return self.high - self.low

    def forward(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float64)
        unit = (theta - self.low) / self.width
        tolerance = 32.0 * np.finfo(np.float32).eps
        if np.any(unit < -tolerance) or np.any(unit > 1.0 + tolerance):
            raise ValueError("Parameter value lies outside the SLCP prior box.")
        unit = np.clip(unit, self.clip, 1.0 - self.clip)
        return (np.log(unit) - np.log1p(-unit)).astype(np.float32)

    def inverse(self, latent: np.ndarray) -> np.ndarray:
        latent = np.asarray(latent, dtype=np.float64)
        positive = latent >= 0.0
        sigmoid = np.empty_like(latent)
        sigmoid[positive] = 1.0 / (1.0 + np.exp(-latent[positive]))
        exponential = np.exp(latent[~positive])
        sigmoid[~positive] = exponential / (1.0 + exponential)
        return (self.low + self.width * sigmoid).astype(np.float32)

    def log_abs_det_inverse(self, latent: np.ndarray) -> np.ndarray:
        latent = np.asarray(latent, dtype=np.float64)
        # log(sigmoid(z)) + log(1 - sigmoid(z)), evaluated stably.
        log_sigmoid = -np.logaddexp(0.0, -latent)
        log_one_minus = -np.logaddexp(0.0, latent)
        return np.sum(
            np.log(self.width) + log_sigmoid + log_one_minus, axis=-1
        )

    def prior_log_prob_latent(self, latent: np.ndarray) -> np.ndarray:
        latent = np.asarray(latent, dtype=np.float64)
        log_prior_theta = -float(np.log(self.width).sum())
        return log_prior_theta + self.log_abs_det_inverse(latent)


@dataclass(frozen=True)
class SLCPBank:
    role: str
    theta: np.ndarray
    x: np.ndarray
    seed: int
    fingerprint: str
    path: Path

    @property
    def latent(self) -> np.ndarray:
        return slcp_parameter_transform().forward(self.theta)


def slcp_parameter_transform() -> BoxUniformLogitTransform:
    return BoxUniformLogitTransform(
        low=np.full(5, -3.0, dtype=np.float64),
        high=np.full(5, 3.0, dtype=np.float64),
    )


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".npz", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _simulate_slcp_bank(
    *, size: int, seed: int, role: str, path: Path, chunk_size: int
) -> SLCPBank:
    import sbibm

    size = int(size)
    if size < 1:
        raise ValueError("Bank size must be positive.")
    seed_everything(seed)
    task = sbibm.get_task("slcp")
    prior = task.get_prior()
    simulator = task.get_simulator(max_calls=size)
    theta_chunks: list[np.ndarray] = []
    x_chunks: list[np.ndarray] = []
    for start in range(0, size, int(chunk_size)):
        count = min(int(chunk_size), size - start)
        theta_tensor = prior(num_samples=count)
        x_tensor = simulator(theta_tensor)
        theta_chunks.append(
            theta_tensor.detach().cpu().numpy().reshape(count, -1).astype(np.float32)
        )
        x_chunks.append(
            x_tensor.detach().cpu().numpy().reshape(count, -1).astype(np.float32)
        )
        print(f"{role}: generated {start + count:,}/{size:,} SLCP pairs")
    theta = np.concatenate(theta_chunks)
    x = np.concatenate(x_chunks)
    if theta.shape != (size, 5) or x.shape != (size, 8):
        raise RuntimeError(
            f"Unexpected SLCP shapes theta={theta.shape}, x={x.shape}."
        )
    if not (np.isfinite(theta).all() and np.isfinite(x).all()):
        raise FloatingPointError("The generated SLCP bank contains non-finite rows.")
    fingerprint = _array_sha256(theta, x)
    _atomic_savez(
        path,
        theta=theta,
        x=x,
        seed=np.asarray(seed, dtype=np.int64),
        role=np.asarray(role),
        fingerprint=np.asarray(fingerprint),
        schema=np.asarray(PAPER_RUNTIME_SCHEMA),
    )
    return SLCPBank(role, theta, x, int(seed), fingerprint, path)


def _load_slcp_bank(path: str | Path, *, expected_role: str) -> SLCPBank:
    path = Path(path)
    with np.load(path, allow_pickle=False) as saved:
        theta = np.asarray(saved["theta"], dtype=np.float32)
        x = np.asarray(saved["x"], dtype=np.float32)
        seed = int(saved["seed"])
        role = str(saved["role"])
        stored_fingerprint = str(saved["fingerprint"])
        schema = str(saved["schema"])
    if role != expected_role or schema != PAPER_RUNTIME_SCHEMA:
        raise RuntimeError(f"Bank contract mismatch in {path}.")
    fingerprint = _array_sha256(theta, x)
    if fingerprint != stored_fingerprint:
        raise RuntimeError(f"Bank fingerprint mismatch in {path}.")
    return SLCPBank(role, theta, x, seed, fingerprint, path)


def prepare_slcp_banks(
    artifact_root: str | Path,
    *,
    campaign: Mapping[str, Any] | None = None,
    budgets: Sequence[int] | None = None,
    master_seed: int | None = None,
    audit_seed: int | None = None,
    jana_shape_seed: int | None = None,
    jana_pilot_seed: int | None = None,
    jana_validation_seed: int | None = None,
    split_seed: int | None = None,
    audit_size: int | None = None,
    jana_shape_size: int = 2,
    jana_pilot_size: int = 2,
    jana_validation_size: int = 300,
    validation_fraction: float = 0.10,
    chunk_size: int = 20_000,
    load_if_available: bool = True,
) -> pd.DataFrame:
    """Create the nested master, exact-JANA auxiliary, and audit banks."""

    if campaign is not None:
        budgets = campaign["budgets"] if budgets is None else budgets
        master_seed = (
            campaign["master_simulation_seed"]
            if master_seed is None else master_seed
        )
        audit_seed = (
            campaign["audit_simulation_seed"] if audit_seed is None else audit_seed
        )
        jana_shape_seed = (
            campaign.get("jana_shape_simulation_seed", int(audit_seed) + 3)
            if jana_shape_seed is None else jana_shape_seed
        )
        jana_shape_size = int(
            campaign.get("jana_shape_simulations", jana_shape_size)
        )
        jana_pilot_seed = (
            campaign.get("jana_pilot_simulation_seed", int(audit_seed) + 2)
            if jana_pilot_seed is None else jana_pilot_seed
        )
        jana_pilot_size = int(
            campaign.get("jana_pilot_simulations", jana_pilot_size)
        )
        jana_validation_seed = (
            campaign.get("jana_validation_simulation_seed", int(audit_seed) + 1)
            if jana_validation_seed is None else jana_validation_seed
        )
        jana_validation_size = int(
            campaign.get("jana_validation_simulations", jana_validation_size)
        )
        split_seed = campaign["split_seed"] if split_seed is None else split_seed
        audit_size = (
            campaign["audit_simulations"] if audit_size is None else audit_size
        )
        validation_fraction = float(
            campaign.get("validation_fraction", validation_fraction)
        )
    budgets = (10_000, 100_000, 1_000_000) if budgets is None else budgets
    master_seed = 12092026 if master_seed is None else int(master_seed)
    audit_seed = 12092027 if audit_seed is None else int(audit_seed)
    jana_shape_seed = (
        12092031 if jana_shape_seed is None else int(jana_shape_seed)
    )
    jana_pilot_seed = (
        12092030 if jana_pilot_seed is None else int(jana_pilot_seed)
    )
    jana_validation_seed = (
        12092028 if jana_validation_seed is None else int(jana_validation_seed)
    )
    split_seed = 12092029 if split_seed is None else int(split_seed)
    audit_size = 100_000 if audit_size is None else int(audit_size)
    budgets = tuple(sorted({int(value) for value in budgets}))
    if not budgets or budgets[0] < 64:
        raise ValueError("Budgets must contain at least 64 simulations.")
    if not 0.0 < float(validation_fraction) < 0.5:
        raise ValueError("validation_fraction must lie strictly between 0 and 0.5.")
    artifact_root = Path(artifact_root).expanduser().resolve()
    bank_root = artifact_root / "simulation_banks"
    manifest_root = artifact_root / "manifests"
    bank_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)
    master_path = bank_root / f"slcp_master_n{budgets[-1]}_seed{int(master_seed)}.npz"
    audit_path = bank_root / f"slcp_audit_n{int(audit_size)}_seed{int(audit_seed)}.npz"
    jana_validation_path = bank_root / (
        "slcp_jana_validation_"
        f"n{int(jana_validation_size)}_seed{int(jana_validation_seed)}.npz"
    )
    jana_pilot_path = bank_root / (
        "slcp_jana_pilot_"
        f"n{int(jana_pilot_size)}_seed{int(jana_pilot_seed)}.npz"
    )
    jana_shape_path = bank_root / (
        "slcp_jana_shape_"
        f"n{int(jana_shape_size)}_seed{int(jana_shape_seed)}.npz"
    )

    if load_if_available and master_path.exists():
        master = _load_slcp_bank(master_path, expected_role="master")
        if len(master.theta) != budgets[-1] or master.seed != int(master_seed):
            raise RuntimeError("Existing master bank has the wrong size or seed.")
    else:
        master = _simulate_slcp_bank(
            size=budgets[-1], seed=master_seed, role="master",
            path=master_path, chunk_size=chunk_size,
        )
    if load_if_available and audit_path.exists():
        audit = _load_slcp_bank(audit_path, expected_role="audit")
        if len(audit.theta) != int(audit_size) or audit.seed != int(audit_seed):
            raise RuntimeError("Existing audit bank has the wrong size or seed.")
    else:
        audit = _simulate_slcp_bank(
            size=audit_size, seed=audit_seed, role="audit",
            path=audit_path, chunk_size=chunk_size,
        )
    if load_if_available and jana_validation_path.exists():
        jana_validation = _load_slcp_bank(
            jana_validation_path, expected_role="jana_validation"
        )
        if (
            len(jana_validation.theta) != int(jana_validation_size)
            or jana_validation.seed != int(jana_validation_seed)
        ):
            raise RuntimeError("Existing JANA validation bank has the wrong contract.")
    else:
        jana_validation = _simulate_slcp_bank(
            size=jana_validation_size,
            seed=jana_validation_seed,
            role="jana_validation",
            path=jana_validation_path,
            chunk_size=chunk_size,
        )
    if load_if_available and jana_pilot_path.exists():
        jana_pilot = _load_slcp_bank(jana_pilot_path, expected_role="jana_pilot")
        if (
            len(jana_pilot.theta) != int(jana_pilot_size)
            or jana_pilot.seed != int(jana_pilot_seed)
        ):
            raise RuntimeError("Existing JANA pilot bank has the wrong contract.")
    else:
        jana_pilot = _simulate_slcp_bank(
            size=jana_pilot_size,
            seed=jana_pilot_seed,
            role="jana_pilot",
            path=jana_pilot_path,
            chunk_size=chunk_size,
        )
    if load_if_available and jana_shape_path.exists():
        jana_shape = _load_slcp_bank(jana_shape_path, expected_role="jana_shape")
        if (
            len(jana_shape.theta) != int(jana_shape_size)
            or jana_shape.seed != int(jana_shape_seed)
        ):
            raise RuntimeError("Existing JANA shape bank has the wrong contract.")
    else:
        jana_shape = _simulate_slcp_bank(
            size=jana_shape_size,
            seed=jana_shape_seed,
            role="jana_shape",
            path=jana_shape_path,
            chunk_size=chunk_size,
        )
    fingerprints = {
        master.fingerprint,
        audit.fingerprint,
        jana_shape.fingerprint,
        jana_pilot.fingerprint,
        jana_validation.fingerprint,
    }
    seeds = {
        master.seed,
        audit.seed,
        jana_shape.seed,
        jana_pilot.seed,
        jana_validation.seed,
    }
    if len(fingerprints) != 5 or len(seeds) != 5:
        raise RuntimeError(
            "Master, JANA-shape, JANA-pilot, JANA-validation, and audit banks are not independent."
        )

    rows: list[dict[str, Any]] = []
    previous_ids: set[int] = set()
    # One row-level assignment is generated for the largest prefix.  Every
    # smaller budget inherits the corresponding prefix, so both training and
    # validation partitions are nested rather than being reshuffled per N.
    split_rng = np.random.default_rng(int(split_seed))
    validation_assignment = (
        split_rng.random(budgets[-1]) < float(validation_fraction)
    )
    for budget in budgets:
        ids = np.arange(budget, dtype=np.int64)
        if previous_ids and not previous_ids.issubset(set(ids.tolist())):
            raise RuntimeError("Simulation budgets are not nested prefixes.")
        previous_ids = set(ids.tolist())
        validation_ids = ids[validation_assignment[:budget]]
        training_ids = ids[~validation_assignment[:budget]]
        if len(validation_ids) < 8 or len(training_ids) < 32:
            raise RuntimeError(
                "Fixed split produced too few rows; use a larger smoke budget."
            )
        if np.intersect1d(training_ids, validation_ids).size:
            raise RuntimeError("Training/validation split overlap.")
        split_fingerprint = _array_sha256(ids, training_ids, validation_ids)
        split_path = bank_root / (
            f"slcp_nested_budget_n{budget}_masterseed{int(master_seed)}.npz"
        )
        _atomic_savez(
            split_path,
            ids=ids,
            training_ids=training_ids,
            validation_ids=validation_ids,
            master_fingerprint=np.asarray(master.fingerprint),
            split_fingerprint=np.asarray(split_fingerprint),
            schema=np.asarray(PAPER_RUNTIME_SCHEMA),
        )
        row = {
            "budget": budget,
            "training_rows": len(training_ids),
            "validation_rows": len(validation_ids),
            "master_seed": int(master_seed),
            "split_seed": int(split_seed),
            "master_fingerprint": master.fingerprint,
            "split_fingerprint": split_fingerprint,
            "split_path": str(split_path),
        }
        rows.append(row)
        _write_json(
            manifest_root / f"slcp_budget_{budget}.json",
            {
                **row,
                "schema": PAPER_RUNTIME_SCHEMA,
                "task": "slcp",
                "subset_semantics": "prefix_of_one_fixed_shuffled_master_bank",
                "created_utc": _utc_now(),
            },
        )
    _write_json(
        manifest_root / "slcp_banks.json",
        {
            "schema": PAPER_RUNTIME_SCHEMA,
            "task": "slcp",
            "budgets": list(budgets),
            "master": {
                "path": str(master.path), "size": len(master.theta),
                "seed": master.seed, "fingerprint": master.fingerprint,
            },
            "audit": {
                "path": str(audit.path), "size": len(audit.theta),
                "seed": audit.seed, "fingerprint": audit.fingerprint,
            },
            "jana_validation": {
                "path": str(jana_validation.path),
                "size": len(jana_validation.theta),
                "seed": jana_validation.seed,
                "fingerprint": jana_validation.fingerprint,
                "budget_accounting": "external_300_calls_required_by_exact_JANA_protocol",
            },
            "jana_pilot": {
                "path": str(jana_pilot.path),
                "size": len(jana_pilot.theta),
                "seed": jana_pilot.seed,
                "fingerprint": jana_pilot.fingerprint,
                "budget_accounting": "external_2_calls_required_by_JANA_Trainer_initialization",
            },
            "jana_shape": {
                "path": str(jana_shape.path),
                "size": len(jana_shape.theta),
                "seed": jana_shape.seed,
                "fingerprint": jana_shape.fingerprint,
                "budget_accounting": "external_2_calls_required_by_JANA_Benchmark_shape_inference",
            },
            "validation_fraction": float(validation_fraction),
            "generation_chunk_size": int(chunk_size),
            "split_seed": int(split_seed),
            "split_semantics": "one_fixed_master_row_assignment_inherited_by_all_prefixes",
            "implementation": _implementation_manifest(),
            "created_utc": _utc_now(),
        },
    )
    return pd.DataFrame(rows)


def load_slcp_budget(
    artifact_root: str | Path, budget: int
) -> dict[str, Any]:
    """Load and verify one nested budget and its shared split."""

    artifact_root = Path(artifact_root).expanduser().resolve()
    manifest_path = artifact_root / "manifests" / "slcp_banks.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != PAPER_RUNTIME_SCHEMA:
        raise RuntimeError("SLCP bank manifest schema mismatch.")
    master = _load_slcp_bank(manifest["master"]["path"], expected_role="master")
    split_path = artifact_root / "simulation_banks" / (
        f"slcp_nested_budget_n{int(budget)}_masterseed{int(master.seed)}.npz"
    )
    with np.load(split_path, allow_pickle=False) as saved:
        ids = np.asarray(saved["ids"], dtype=np.int64)
        training_ids = np.asarray(saved["training_ids"], dtype=np.int64)
        validation_ids = np.asarray(saved["validation_ids"], dtype=np.int64)
        master_fingerprint = str(saved["master_fingerprint"])
        stored_split_fingerprint = str(saved["split_fingerprint"])
        schema = str(saved["schema"])
    if schema != PAPER_RUNTIME_SCHEMA or master_fingerprint != master.fingerprint:
        raise RuntimeError("Nested budget does not match the master bank.")
    split_fingerprint = _array_sha256(ids, training_ids, validation_ids)
    if split_fingerprint != stored_split_fingerprint:
        raise RuntimeError("Nested split fingerprint mismatch.")
    if len(ids) != int(budget) or not np.array_equal(ids, np.arange(int(budget))):
        raise RuntimeError("Nested budget is not the expected master-bank prefix.")
    transform = slcp_parameter_transform()
    theta = master.theta[ids]
    x = master.x[ids]
    return {
        "theta": theta,
        "z": transform.forward(theta),
        "x": x,
        "training_ids": training_ids,
        "validation_ids": validation_ids,
        "master_fingerprint": master.fingerprint,
        "split_fingerprint": split_fingerprint,
        "transform": transform,
        "manifest": manifest,
    }


def load_slcp_audit(artifact_root: str | Path) -> dict[str, Any]:
    artifact_root = Path(artifact_root).expanduser().resolve()
    manifest = json.loads(
        (artifact_root / "manifests" / "slcp_banks.json").read_text()
    )
    bank = _load_slcp_bank(manifest["audit"]["path"], expected_role="audit")
    return {
        "theta": bank.theta,
        "z": slcp_parameter_transform().forward(bank.theta),
        "x": bank.x,
        "fingerprint": bank.fingerprint,
        "seed": bank.seed,
    }


def exact_slcp_log_likelihood(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate the analytic SLCP likelihood for paired or broadcast rows."""

    theta = np.asarray(theta, dtype=np.float64).reshape(-1, 5)
    x = np.asarray(x, dtype=np.float64).reshape(-1, 8)
    if len(theta) == 1 and len(x) > 1:
        theta = np.repeat(theta, len(x), axis=0)
    elif len(x) == 1 and len(theta) > 1:
        x = np.repeat(x, len(theta), axis=0)
    elif len(theta) != len(x):
        raise ValueError("theta and x must have equal rows or one broadcast row.")
    mean = theta[:, None, :2]
    observations = x.reshape(-1, 4, 2)
    difference = observations - mean
    scale1 = np.square(theta[:, 2])
    scale2 = np.square(theta[:, 3])
    correlation = np.tanh(theta[:, 4])
    variance1 = np.square(scale1)
    variance2 = np.square(scale2)
    covariance = correlation * scale1 * scale2
    determinant = variance1 * variance2 - np.square(covariance)
    values = np.full(len(theta), -np.inf, dtype=np.float64)
    valid = (determinant > 0.0) & np.isfinite(determinant)
    if np.any(valid):
        quadratic = (
            variance2[valid, None] * np.square(difference[valid, :, 0])
            + variance1[valid, None] * np.square(difference[valid, :, 1])
            - 2.0
            * covariance[valid, None]
            * difference[valid, :, 0]
            * difference[valid, :, 1]
        ) / determinant[valid, None]
        per_draw = (
            math.log(2.0 * math.pi)
            + 0.5 * np.log(determinant[valid])[:, None]
            + 0.5 * quadratic
        )
        values[valid] = -np.sum(per_draw, axis=1)
    if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
        raise FloatingPointError("Exact SLCP likelihood returned invalid values.")
    return values


# ---------------------------------------------------------------------------
# Broad flow ensembles and the parameter-prior defensive component
# ---------------------------------------------------------------------------


def spline_flow_broad_ensemble_log_prob(
    flow_packs: Sequence[Mapping[str, Any]],
    target: np.ndarray,
    *,
    context: np.ndarray | None,
    base_scale: float,
    batch_size: int = 65_536,
) -> np.ndarray:
    """Density of an equal-weight ensemble using only a broadened base."""

    members = _validate_flow_ensemble(flow_packs)
    base_scale = float(base_scale)
    if not np.isfinite(base_scale) or base_scale <= 0.0:
        raise ValueError("base_scale must be finite and positive.")
    member_log_probabilities = np.stack(
        [
            spline_flow_log_prob_with_base_scale(
                member,
                target,
                context=context,
                base_scale=base_scale,
                batch_size=batch_size,
            )
            for member in members
        ],
        axis=0,
    )
    return logsumexp(member_log_probabilities, axis=0) - math.log(len(members))


def sample_spline_flow_broad_ensemble(
    flow_packs: Sequence[Mapping[str, Any]],
    n_samples: int,
    *,
    context: np.ndarray | None,
    seed: int,
    base_scale: float,
    batch_size: int = 16_384,
) -> np.ndarray:
    """Draw IID samples from the broad-only equal-weight flow ensemble."""

    members = _validate_flow_ensemble(flow_packs)
    n_samples = int(n_samples)
    base_scale = float(base_scale)
    if n_samples < 1 or not np.isfinite(base_scale) or base_scale <= 0.0:
        raise ValueError("Invalid broad-ensemble sampling request.")
    conditional = members[0]["context_scaler"] is not None
    if conditional:
        if context is None:
            raise ValueError("This broad flow ensemble requires context.")
        context_rows = _as_2d_float32(context, "context")
    else:
        if context is not None:
            raise ValueError("This broad flow ensemble is unconditional.")
        context_rows = np.empty((1, 0), dtype=np.float32)
    rng = np.random.default_rng(int(seed))
    n_context = len(context_rows)
    assignments = rng.integers(
        0, len(members), size=(n_context, n_samples), dtype=np.int64
    )
    n_features = int(np.asarray(members[0]["target_scaler"].mean).size)
    output = np.empty((n_context, n_samples, n_features), dtype=np.float32)
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        for member_index, member in enumerate(members):
            counts = np.sum(assignments == member_index, axis=1)
            for count in np.unique(counts):
                count = int(count)
                if count == 0:
                    continue
                rows = np.flatnonzero(counts == count)
                local_context = context_rows[rows] if conditional else None
                values = sample_spline_flow_with_base_scale(
                    member,
                    count,
                    context=local_context,
                    base_scale=base_scale,
                    batch_size=batch_size,
                )
                if len(rows) == 1:
                    values = np.asarray(values)[None, :, :]
                positions = np.nonzero(
                    assignments[rows] == member_index
                )[1].reshape(len(rows), count)
                output[rows[:, None], positions, :] = values
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
    if not conditional or n_context == 1:
        return output[0]
    return output


def posterior_proposal_log_prob(
    q_phi: Sequence[Mapping[str, Any]],
    latent: np.ndarray,
    *,
    context: np.ndarray,
    base_scale: float,
    prior_fraction: float,
    transform: BoxUniformLogitTransform | None = None,
    batch_size: int = 65_536,
) -> np.ndarray:
    """Evaluate broad-flow plus transformed-prior proposal density in z."""

    prior_fraction = float(prior_fraction)
    if not 0.0 <= prior_fraction < 1.0:
        raise ValueError("prior_fraction must lie in [0, 1).")
    transform = transform or slcp_parameter_transform()
    broad = spline_flow_broad_ensemble_log_prob(
        q_phi,
        latent,
        context=context,
        base_scale=base_scale,
        batch_size=batch_size,
    )
    prior = transform.prior_log_prob_latent(latent)
    if prior_fraction == 0.0:
        return broad
    return np.logaddexp(
        math.log1p(-prior_fraction) + broad,
        math.log(prior_fraction) + prior,
    )


def sample_posterior_proposal(
    q_phi: Sequence[Mapping[str, Any]],
    n_samples: int,
    *,
    context: np.ndarray,
    seed: int,
    base_scale: float,
    prior_fraction: float,
    transform: BoxUniformLogitTransform | None = None,
    batch_size: int = 16_384,
) -> np.ndarray:
    """Draw IID samples from broad q_phi plus a transformed-prior defense."""

    context = _as_2d_float32(context, "context")
    transform = transform or slcp_parameter_transform()
    prior_fraction = float(prior_fraction)
    if not 0.0 <= prior_fraction < 1.0:
        raise ValueError("prior_fraction must lie in [0, 1).")
    values = sample_spline_flow_broad_ensemble(
        q_phi,
        int(n_samples),
        context=context,
        seed=int(seed) + 1,
        base_scale=base_scale,
        batch_size=batch_size,
    )
    if len(context) == 1:
        values = np.asarray(values, dtype=np.float32)[None, :, :]
    rng = np.random.default_rng(int(seed))
    use_prior = rng.random((len(context), int(n_samples))) < prior_fraction
    n_prior = int(use_prior.sum())
    if n_prior:
        theta = rng.uniform(
            transform.low,
            transform.high,
            size=(n_prior, len(transform.low)),
        ).astype(np.float32)
        values[use_prior] = transform.forward(theta)
    if not np.isfinite(values).all():
        raise FloatingPointError("Posterior proposal sampling returned non-finite values.")
    return values[0] if len(context) == 1 else values


def likelihood_proposal_log_prob(
    q_eta: Sequence[Mapping[str, Any]],
    x: np.ndarray,
    *,
    latent: np.ndarray,
    base_scale: float,
    batch_size: int = 65_536,
) -> np.ndarray:
    """Evaluate the broad-only likelihood proposal g_eta(x|z)."""

    return spline_flow_broad_ensemble_log_prob(
        q_eta,
        x,
        context=latent,
        base_scale=base_scale,
        batch_size=batch_size,
    )


def sample_likelihood_proposal(
    q_eta: Sequence[Mapping[str, Any]],
    n_samples: int,
    *,
    latent: np.ndarray,
    seed: int,
    base_scale: float,
    batch_size: int = 16_384,
) -> np.ndarray:
    """Draw IID samples from the broad-only likelihood proposal."""

    return sample_spline_flow_broad_ensemble(
        q_eta,
        int(n_samples),
        context=latent,
        seed=seed,
        base_scale=base_scale,
        batch_size=batch_size,
    )


def flow_ensemble_checkpoint_paths(
    checkpoint: str | Path, ensemble_size: int
) -> list[Path]:
    checkpoint = Path(checkpoint)
    suffix = checkpoint.suffix or ".pt"
    stem = checkpoint.stem if checkpoint.suffix else checkpoint.name
    return [
        checkpoint.parent / f"{stem}.member_{index:02d}{suffix}"
        for index in range(int(ensemble_size))
    ]


def load_spline_flow_ensemble(
    checkpoint: str | Path,
    ensemble_size: int,
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
    paths = flow_ensemble_checkpoint_paths(checkpoint, ensemble_size)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete flow ensemble: {missing}")
    return [load_spline_flow(path, device) for path in paths]


def flow_parameter_count(pack: Mapping[str, Any]) -> int:
    return int(sum(parameter.numel() for parameter in pack["flow"].parameters()))


# ---------------------------------------------------------------------------
# Capacity selection and the shared matched-flow checkpoints
# ---------------------------------------------------------------------------


def _config_module():
    try:
        from . import config as paper_config
    except ImportError:
        import config as paper_config
    return paper_config


def _execution_subset(
    configured: Sequence[int], requested: Sequence[int] | None, name: str
) -> tuple[int, ...]:
    """Validate a run shard without changing the campaign identity."""

    configured_values = tuple(int(value) for value in configured)
    if requested is None:
        return configured_values
    selected = tuple(dict.fromkeys(int(value) for value in requested))
    if not selected:
        raise ValueError(f"{name} execution subset cannot be empty.")
    unknown = set(selected) - set(configured_values)
    if unknown:
        raise ValueError(
            f"{name} execution subset contains values outside the campaign: "
            f"{sorted(unknown)}"
        )
    return selected


def _flow_model_config(
    campaign: Mapping[str, Any], architecture: Mapping[str, Any]
) -> dict[str, Any]:
    result = dict(campaign["flow_fixed_config"])
    result.update(
        {
            "n_coupling_layers": int(architecture["n_coupling_layers"]),
            "hidden_features": int(architecture["hidden_features"]),
        }
    )
    return result


def _flow_training_config(
    campaign: Mapping[str, Any], *, capacity_screen: bool = False
) -> dict[str, Any]:
    policy = campaign["matched_flow_training"]
    epochs = policy.get("maximum_epochs_override")
    if epochs is None:
        epochs = (
            policy["capacity_screen_epochs"]
            if capacity_screen
            else policy["maximum_epochs"]
        )
    return {
        "batch_size": int(policy["batch_size"]),
        "n_epochs": int(epochs),
        "learning_rate": float(policy["initial_learning_rate"]),
        "min_learning_rate": float(policy["minimum_learning_rate"]),
        "lr_scheduler": "six_equal_log10_plateaus",
        "lr_scheduler_factor": 0.1,
        "lr_scheduler_patience": 50,
        "validation_fraction": float(campaign["validation_fraction"]),
        "patience": (
            min(int(policy["early_stopping_patience"]), int(epochs))
            if bool(policy.get("early_stopping", False))
            else int(epochs) + 1
        ),
        "print_every": max(1, min(50, int(epochs))),
        "gradient_clip": float(policy["gradient_clip"]),
        "weight_decay": 0.0,
        "checkpoint_selection": str(policy["checkpoint_selection"]),
    }


def _architecture_key(architecture: Mapping[str, Any]) -> str:
    return (
        f"blocks{int(architecture['n_coupling_layers'])}_"
        f"width{int(architecture['hidden_features'])}"
    )


def _capacity_paths(
    artifact_root: Path, campaign: Mapping[str, Any]
) -> tuple[Path, Path]:
    paper_config = _config_module()
    signature = paper_config.campaign_signature(campaign)
    result_root = artifact_root / "results" / "capacity"
    return (
        result_root / f"capacity_scan_{signature}.csv",
        result_root / f"selected_architectures_{signature}.json",
    )


def run_capacity_scan(
    artifact_root: str | Path,
    campaign: Mapping[str, Any],
    *,
    budgets_to_run: Sequence[int] | None = None,
    ml_seeds_to_run: Sequence[int] | None = None,
    load_if_available: bool = True,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Select the smallest flow within one SE of the best validation NLL."""

    paper_config = _config_module()
    paper_config.validate_campaign_config(campaign)
    artifact_root = Path(artifact_root).expanduser().resolve()
    result_path, selection_path = _capacity_paths(artifact_root, campaign)
    run_budgets = _execution_subset(
        campaign["budgets"], budgets_to_run, "budget"
    )
    run_seeds = _execution_subset(
        campaign["ml_seeds"], ml_seeds_to_run, "ML-seed"
    )
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    install_nflows_rqs_float64_retry()
    rows: list[dict[str, Any]] = []
    training_config = _flow_training_config(campaign, capacity_screen=True)
    signature = paper_config.campaign_signature(campaign)
    for budget in run_budgets:
        bank = load_slcp_budget(artifact_root, int(budget))
        train_ids = bank["training_ids"]
        validation_ids = bank["validation_ids"]
        routes = {
            "posterior": (
                bank["z"][train_ids], bank["x"][train_ids],
                bank["z"][validation_ids], bank["x"][validation_ids],
            ),
            "likelihood": (
                bank["x"][train_ids], bank["z"][train_ids],
                bank["x"][validation_ids], bank["z"][validation_ids],
            ),
        }
        for route, (target, context, val_target, val_context) in routes.items():
            for architecture_index, architecture in enumerate(
                campaign["flow_architecture_grid"]
            ):
                model_config = _flow_model_config(campaign, architecture)
                architecture_key = _architecture_key(architecture)
                for seed_index, ml_seed in enumerate(run_seeds):
                    checkpoint = (
                        artifact_root / "models" / "capacity" / signature
                        / f"budget_{int(budget)}" / route / architecture_key
                        / f"seed_{int(ml_seed)}" / "screen.pt"
                    )
                    row_path = checkpoint.parent / "screen_result.json"
                    if load_if_available and row_path.exists():
                        continue
                    pack = train_spline_flow(
                        target,
                        context=context,
                        validation_target=val_target,
                        validation_context=val_context,
                        checkpoint=checkpoint,
                        model_config=model_config,
                        training_config=training_config,
                        device=device,
                        seed=int(ml_seed) + 10_000 * architecture_index,
                        load_if_available=load_if_available,
                        verify_checkpoint_data=True,
                    )
                    validation_log_prob = spline_flow_log_prob(
                        pack, val_target, context=val_context
                    )
                    losses = -np.asarray(validation_log_prob, dtype=np.float64)
                    row = {
                            "schema": PAPER_RUNTIME_SCHEMA,
                            "campaign_signature": signature,
                            "budget": int(budget),
                            "route": route,
                            "architecture": architecture_key,
                            "n_coupling_layers": int(
                                architecture["n_coupling_layers"]
                            ),
                            "hidden_features": int(
                                architecture["hidden_features"]
                            ),
                            "ml_seed": int(ml_seed),
                            "validation_nll": float(losses.mean()),
                            "validation_nll_row_sem": float(
                                losses.std(ddof=1) / math.sqrt(len(losses))
                            ),
                            "validation_rows": int(len(losses)),
                            "parameter_count": flow_parameter_count(pack),
                            "selected_epoch": int(
                                pack["history"].get("selected_epoch", [0])[0]
                            ),
                            "split_fingerprint": bank["split_fingerprint"],
                            "checkpoint": str(checkpoint),
                        }
                    rows.append(row)
                    _write_json(row_path, row)
    result_files = sorted(
        (
            artifact_root / "models" / "capacity" / signature
        ).glob("**/screen_result.json")
    )
    collected = [json.loads(path.read_text()) for path in result_files]
    scan = pd.DataFrame(collected)
    if scan.empty:
        raise RuntimeError("The capacity scan produced no result records.")
    scan = scan.drop_duplicates(
        ["budget", "route", "architecture", "ml_seed"], keep="last"
    ).sort_values(["budget", "route", "parameter_count", "ml_seed"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(result_path, scan)
    expected_keys = {
        (int(budget), route, _architecture_key(architecture), int(seed))
        for budget in campaign["budgets"]
        for route in ("posterior", "likelihood")
        for architecture in campaign["flow_architecture_grid"]
        for seed in campaign["ml_seeds"]
    }
    actual_keys = {
        (int(row.budget), str(row.route), str(row.architecture), int(row.ml_seed))
        for row in scan.itertuples()
    }
    missing_keys = expected_keys - actual_keys
    if missing_keys:
        if selection_path.exists():
            raise RuntimeError(
                "A capacity selection exists although its screen grid is incomplete."
            )
        return {
            "scan": scan,
            "selection": None,
            "complete": False,
            "missing_screen_runs": len(missing_keys),
        }
    selected: dict[str, Any] = {
        "schema": PAPER_RUNTIME_SCHEMA,
        "campaign_signature": signature,
        "selection_rule": "smallest_parameter_count_within_one_SE_of_best_mean_validation_NLL",
        "routes": {},
    }
    for (budget, route), group in scan.groupby(["budget", "route"], sort=True):
        aggregate = (
            group.groupby(
                [
                    "architecture", "n_coupling_layers", "hidden_features",
                    "parameter_count",
                ],
                as_index=False,
            )
            .agg(
                validation_nll=("validation_nll", "mean"),
                seed_std=("validation_nll", "std"),
                n_seeds=("ml_seed", "nunique"),
                row_sem=("validation_nll_row_sem", "mean"),
            )
        )
        aggregate["seed_sem"] = aggregate["seed_std"] / np.sqrt(
            aggregate["n_seeds"].clip(lower=1)
        )
        best_index = aggregate["validation_nll"].idxmin()
        best = aggregate.loc[best_index]
        best_sem = float(best["seed_sem"])
        if not np.isfinite(best_sem) or best_sem <= 0.0:
            best_sem = float(best["row_sem"])
        threshold = float(best["validation_nll"] + best_sem)
        eligible = aggregate[aggregate["validation_nll"] <= threshold].copy()
        choice = eligible.sort_values(
            ["parameter_count", "validation_nll", "architecture"]
        ).iloc[0]
        selected["routes"].setdefault(str(int(budget)), {})[route] = {
            "architecture": str(choice["architecture"]),
            "n_coupling_layers": int(choice["n_coupling_layers"]),
            "hidden_features": int(choice["hidden_features"]),
            "parameter_count": int(choice["parameter_count"]),
            "validation_nll": float(choice["validation_nll"]),
            "best_validation_nll": float(best["validation_nll"]),
            "one_se_threshold": threshold,
            "n_seeds": int(choice["n_seeds"]),
        }
    _write_json(selection_path, selected)
    return {
        "scan": scan,
        "selection": selected,
        "complete": True,
        "missing_screen_runs": 0,
    }


def _load_selected_architectures(
    artifact_root: Path, campaign: Mapping[str, Any]
) -> dict[str, Any]:
    _, selection_path = _capacity_paths(artifact_root, campaign)
    if not selection_path.exists():
        raise FileNotFoundError(
            "Run 01_SLCP_flow_capacity.ipynb before training matched flows."
        )
    selected = json.loads(selection_path.read_text())
    signature = _config_module().campaign_signature(campaign)
    if selected.get("campaign_signature") != signature:
        raise RuntimeError("Selected architecture belongs to another campaign.")
    return selected


def train_matched_flow_pair(
    artifact_root: str | Path,
    campaign: Mapping[str, Any],
    *,
    budget: int,
    ml_seed: int,
    load_if_available: bool = True,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Train q_phi and q_eta once; baseline and corrections share them."""

    paper_config = _config_module()
    paper_config.validate_campaign_config(campaign)
    artifact_root = Path(artifact_root).expanduser().resolve()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    install_nflows_rqs_float64_retry()
    selected = _load_selected_architectures(artifact_root, campaign)
    route_selection = selected["routes"][str(int(budget))]
    bank = load_slcp_budget(artifact_root, int(budget))
    train_ids = bank["training_ids"]
    validation_ids = bank["validation_ids"]
    signature = paper_config.campaign_signature(campaign)
    model_root = (
        artifact_root / "models" / "matched" / signature
        / f"budget_{int(budget)}" / f"seed_{int(ml_seed)}"
    )
    training_config = _flow_training_config(campaign)
    ensemble_size = int(campaign["matched_flow_training"]["ensemble_members"])
    packs: dict[str, Any] = {}
    route_arrays = {
        "posterior": (bank["z"], bank["x"], "q_phi"),
        "likelihood": (bank["x"], bank["z"], "q_eta"),
    }
    for route, (target, context, stem) in route_arrays.items():
        architecture = route_selection[route]
        checkpoint = model_root / f"{stem}.pt"
        packs[stem] = train_spline_flow_ensemble(
            target[train_ids],
            context=context[train_ids],
            validation_target=target[validation_ids],
            validation_context=context[validation_ids],
            checkpoint=checkpoint,
            ensemble_size=ensemble_size,
            model_config=_flow_model_config(campaign, architecture),
            training_config=training_config,
            device=device,
            seed=int(ml_seed) + (1_000 if route == "posterior" else 2_000),
            load_if_available=load_if_available,
            verify_checkpoint_data=True,
        )
    manifest = {
        "schema": PAPER_RUNTIME_SCHEMA,
        "campaign_signature": signature,
        "budget": int(budget),
        "ml_seed": int(ml_seed),
        "training_pair_fingerprint": bank["master_fingerprint"],
        "split_fingerprint": bank["split_fingerprint"],
        "training_rows": int(len(train_ids)),
        "validation_rows": int(len(validation_ids)),
        "posterior_architecture": route_selection["posterior"],
        "likelihood_architecture": route_selection["likelihood"],
        "ensemble_members": ensemble_size,
        "checkpoint_root": str(model_root),
        "shared_by": [
            "separate_flows",
            "separate_flows_corrected_multiclass",
            "separate_flows_corrected_binary",
        ],
        "implementation": _implementation_manifest(),
        "created_utc": _utc_now(),
    }
    _write_json(model_root / "manifest.json", manifest)
    return {**packs, "manifest": manifest, "bank": bank, "device": device}


# ---------------------------------------------------------------------------
# Common inference and metrics
# ---------------------------------------------------------------------------


def normalized_log_weights(log_weights: np.ndarray) -> np.ndarray:
    log_weights = np.asarray(log_weights, dtype=np.float64).reshape(-1)
    if np.any(np.isnan(log_weights)) or np.any(np.isposinf(log_weights)):
        raise FloatingPointError("Importance log weights contain NaN or +inf.")
    finite = np.isfinite(log_weights)
    if not np.any(finite):
        raise FloatingPointError("Every importance log weight is -inf.")
    shifted = log_weights - np.max(log_weights[finite])
    weights = np.exp(shifted)
    total = weights.sum(dtype=np.float64)
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError("Importance weights cannot be normalized.")
    return weights / total


def slcp_prior_support(theta: np.ndarray) -> np.ndarray:
    """Return the exact support mask for the SBIBM SLCP box prior."""

    theta = np.asarray(theta)
    if theta.ndim != 2 or theta.shape[1] != 5:
        raise ValueError(f"Expected SLCP theta with shape (n, 5), got {theta.shape}.")
    return np.all((theta >= -3.0) & (theta <= 3.0), axis=1)


def weight_diagnostics(weights: np.ndarray) -> dict[str, float]:
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    total = weights.sum(dtype=np.float64)
    if not np.isfinite(total) or total <= 0.0 or np.any(weights < 0.0):
        raise FloatingPointError("Invalid importance weights.")
    weights = weights / total
    ess = 1.0 / np.square(weights).sum()
    return {
        "ess": float(ess),
        "ess_fraction": float(ess / len(weights)),
        "max_weight": float(weights.max()),
        "entropy": float(-np.sum(weights * np.log(np.clip(weights, 1.0e-300, None)))),
    }


def systematic_resample(
    values: np.ndarray, weights: np.ndarray, n_samples: int, seed: int
) -> np.ndarray:
    values = np.asarray(values)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    weights = weights / weights.sum(dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    positions = (rng.random() + np.arange(int(n_samples))) / int(n_samples)
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    indices = np.searchsorted(cumulative, positions, side="right")
    return np.asarray(values[indices])


def c2st_metric(
    left: np.ndarray,
    right: np.ndarray,
    *,
    seed: int,
    max_samples: int,
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


def _mmd_metric(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    max_samples: int,
    seed: int,
) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    n = min(len(reference), len(candidate), int(max_samples))
    if n < 32:
        return float("nan")
    rng = np.random.default_rng(int(seed))
    reference = reference[rng.choice(len(reference), n, replace=False)]
    candidate = candidate[rng.choice(len(candidate), n, replace=False)]
    mean = reference.mean(axis=0)
    scale = reference.std(axis=0)
    scale = np.where(scale > 1.0e-8, scale, 1.0)
    reference = (reference - mean) / scale
    candidate = (candidate - mean) / scale
    pilot = reference[: min(1_000, len(reference))]
    distances = pdist(pilot, metric="sqeuclidean")
    distances = distances[distances > 0.0]
    bandwidth2 = max(float(np.median(distances)) if len(distances) else 1.0, 1e-8)

    def kernel_mean(first: np.ndarray, second: np.ndarray) -> float:
        total = 0.0
        count = 0
        for start in range(0, len(first), 256):
            distance2 = cdist(
                first[start : start + 256], second, metric="sqeuclidean"
            )
            total += float(np.exp(-distance2 / (2.0 * bandwidth2)).sum())
            count += int(distance2.size)
        return total / count

    mmd2 = max(
        0.0,
        kernel_mean(reference, reference)
        + kernel_mean(candidate, candidate)
        - 2.0 * kernel_mean(reference, candidate),
    )
    return math.sqrt(mmd2)


def distribution_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    seed: int,
    max_samples: int,
) -> dict[str, Any]:
    return {
        "C2ST": c2st_metric(
            reference, candidate, seed=seed, max_samples=max_samples
        ),
        "MMD": _mmd_metric(
            reference, candidate, seed=seed + 1, max_samples=max_samples
        ),
    }


def _cycle_diagnostics(
    posterior_log_ratio: np.ndarray, likelihood_log_ratio: np.ndarray
) -> dict[str, float]:
    # Copy because callers also persist the uncentered log densities.
    posterior = np.asarray(posterior_log_ratio, dtype=np.float64).reshape(-1).copy()
    likelihood = np.asarray(likelihood_log_ratio, dtype=np.float64).reshape(-1).copy()
    posterior -= np.median(posterior)
    likelihood -= np.median(likelihood)
    if len(posterior) < 3:
        return {"pearson": float("nan"), "slope": float("nan"), "rms": float("nan")}
    pearson = float(np.corrcoef(posterior, likelihood)[0, 1])
    denominator = float(np.dot(posterior, posterior))
    slope = float(np.dot(posterior, likelihood) / denominator) if denominator else float("nan")
    residual = likelihood - posterior
    return {
        "pearson": pearson,
        "slope": slope,
        "rms": float(np.sqrt(np.mean(np.square(residual)))),
    }


def _task_observation(task: Any, number: int) -> np.ndarray:
    return (
        task.get_observation(num_observation=int(number))
        .detach().cpu().numpy().reshape(1, -1).astype(np.float32)
    )


def _reference_posterior(task: Any, number: int) -> np.ndarray:
    return (
        task.get_reference_posterior_samples(num_observation=int(number))
        .detach().cpu().numpy().reshape(-1, 5).astype(np.float32)
    )


def _finite_error_summary(
    learned: np.ndarray,
    exact: np.ndarray,
    *,
    require_all_exact_finite_rows: bool = False,
) -> dict[str, float]:
    """RMSE summaries on the non-singular analytic SLCP rows."""

    learned = np.asarray(learned, dtype=np.float64).reshape(-1)
    exact = np.asarray(exact, dtype=np.float64).reshape(-1)
    exact_finite = np.isfinite(exact)
    if require_all_exact_finite_rows and np.any(
        exact_finite & ~np.isfinite(learned)
    ):
        missing = int(np.sum(exact_finite & ~np.isfinite(learned)))
        raise FloatingPointError(
            f"Learned likelihood is non-finite on {missing} finite exact-audit rows."
        )
    valid = np.isfinite(learned) & exact_finite
    if not np.any(valid):
        return {"rmse": float("nan"), "centered_rmse": float("nan"), "rows": 0}
    difference = learned[valid] - exact[valid]
    return {
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "centered_rmse": float(
            np.sqrt(np.mean(np.square(difference - np.mean(difference))))
        ),
        "rows": int(valid.sum()),
    }


def _audit_joint_diagnostics(
    *,
    artifact_root: str | Path,
    campaign: Mapping[str, Any],
    q_phi: Sequence[Mapping[str, Any]],
    q_eta: Sequence[Mapping[str, Any]],
    seed: int,
    base_scale: float = 1.0,
    prior_fraction: float = 0.0,
    posterior_log_ratio: Any | None = None,
    likelihood_log_ratio: Any | None = None,
) -> dict[str, float]:
    """Fresh-bank closure for the two learned joint factorizations.

    The first half of the independent audit bank supplies simulator-reference
    rows.  The disjoint second half supplies only the p(x) marginal for P and
    the p(theta) marginal for L, avoiding paired-row leakage in C2ST.
    """

    audit = load_slcp_audit(artifact_root)
    requested = int(campaign.get("audit_joint_samples", campaign["metric_max_samples"]))
    n = min(requested, len(audit["theta"]) // 2)
    if n < 32:
        return {
            "posterior_joint_C2ST": float("nan"),
            "posterior_joint_MMD": float("nan"),
            "predictive_x_C2ST": float("nan"),
            "predictive_x_MMD": float("nan"),
            "predictive_joint_C2ST": float("nan"),
            "predictive_joint_MMD": float("nan"),
            "posterior_joint_ESS_fraction": float("nan"),
            "likelihood_joint_ESS_fraction": float("nan"),
        }
    reference_theta = audit["theta"][:n]
    reference_x = audit["x"][:n]
    reference_joint = np.column_stack([reference_theta, reference_x])
    marginal_theta = audit["theta"][-n:]
    marginal_z = audit["z"][-n:]
    marginal_x = audit["x"][-n:]
    transform = slcp_parameter_transform()

    if float(base_scale) == 1.0 and float(prior_fraction) == 0.0:
        posterior_z = sample_spline_flow_ensemble(
            q_phi, 1, context=marginal_x, seed=int(seed) + 1, allocation="iid"
        )[:, 0, :]
    else:
        posterior_z = sample_posterior_proposal(
            q_phi,
            1,
            context=marginal_x,
            seed=int(seed) + 1,
            base_scale=float(base_scale),
            prior_fraction=float(prior_fraction),
        )[:, 0, :]
    posterior_points = np.column_stack([posterior_z, marginal_x]).astype(np.float32)
    posterior_weights = normalized_log_weights(
        np.zeros(n, dtype=np.float64)
        if posterior_log_ratio is None
        else np.asarray(posterior_log_ratio(posterior_points), dtype=np.float64)
    )
    posterior_joint = np.column_stack(
        [transform.inverse(posterior_z), marginal_x]
    )
    posterior_joint = systematic_resample(
        posterior_joint, posterior_weights, n, int(seed) + 2
    )

    if float(base_scale) == 1.0:
        likelihood_x = sample_spline_flow_ensemble(
            q_eta, 1, context=marginal_z, seed=int(seed) + 3, allocation="iid"
        )[:, 0, :]
    else:
        likelihood_x = sample_likelihood_proposal(
            q_eta,
            1,
            latent=marginal_z,
            seed=int(seed) + 3,
            base_scale=float(base_scale),
        )[:, 0, :]
    likelihood_points = np.column_stack([marginal_z, likelihood_x]).astype(np.float32)
    likelihood_weights = normalized_log_weights(
        np.zeros(n, dtype=np.float64)
        if likelihood_log_ratio is None
        else np.asarray(likelihood_log_ratio(likelihood_points), dtype=np.float64)
    )
    likelihood_joint = np.column_stack([marginal_theta, likelihood_x])
    likelihood_joint = systematic_resample(
        likelihood_joint, likelihood_weights, n, int(seed) + 4
    )
    likelihood_x_resampled = likelihood_joint[:, 5:]

    max_samples = int(campaign["metric_max_samples"])
    posterior_metrics = distribution_metrics(
        reference_joint,
        posterior_joint,
        seed=int(seed) + 5,
        max_samples=max_samples,
    )
    predictive_x_metrics = distribution_metrics(
        reference_x,
        likelihood_x_resampled,
        seed=int(seed) + 6,
        max_samples=max_samples,
    )
    predictive_joint_metrics = distribution_metrics(
        reference_joint,
        likelihood_joint,
        seed=int(seed) + 7,
        max_samples=max_samples,
    )
    return {
        "posterior_joint_C2ST": posterior_metrics["C2ST"],
        "posterior_joint_MMD": posterior_metrics["MMD"],
        "predictive_x_C2ST": predictive_x_metrics["C2ST"],
        "predictive_x_MMD": predictive_x_metrics["MMD"],
        "predictive_joint_C2ST": predictive_joint_metrics["C2ST"],
        "predictive_joint_MMD": predictive_joint_metrics["MMD"],
        "posterior_joint_ESS_fraction": weight_diagnostics(posterior_weights)[
            "ess_fraction"
        ],
        "likelihood_joint_ESS_fraction": weight_diagnostics(likelihood_weights)[
            "ess_fraction"
        ],
    }


def _corrected_likelihood_normalization(
    *,
    artifact_root: str | Path,
    campaign: Mapping[str, Any],
    q_eta: Sequence[Mapping[str, Any]],
    seed: int,
    base_scale: float,
    likelihood_log_ratio: Any | None,
) -> dict[str, float]:
    """Estimate integral g_eta(x|z) r_L(z,x) dx on audit parameters."""

    if likelihood_log_ratio is None:
        return {"rms": 0.0, "mean": 0.0, "max_abs": 0.0}
    audit = load_slcp_audit(artifact_root)
    n_theta = min(int(campaign["normalization_theta"]), len(audit["z"]))
    n_x = int(campaign["normalization_x_per_theta"])
    latent = audit["z"][:n_theta]
    generated = sample_likelihood_proposal(
        q_eta,
        n_x,
        latent=latent,
        seed=int(seed) + 1,
        base_scale=float(base_scale),
    )
    flat_latent = np.repeat(latent, n_x, axis=0)
    flat_x = np.asarray(generated, dtype=np.float32).reshape(-1, 8)
    points = np.column_stack([flat_latent, flat_x]).astype(np.float32)
    log_ratio = np.asarray(likelihood_log_ratio(points), dtype=np.float64).reshape(
        n_theta, n_x
    )
    log_z = logsumexp(log_ratio, axis=1) - math.log(n_x)
    return {
        "rms": float(np.sqrt(np.mean(np.square(log_z)))),
        "mean": float(np.mean(log_z)),
        "max_abs": float(np.max(np.abs(log_z))),
    }


def _separate_flow_common_density_audit(
    *,
    artifact_root: str | Path,
    campaign: Mapping[str, Any],
    q_eta: Sequence[Mapping[str, Any]],
    base_scale: float = 1.0,
    likelihood_log_ratio: Any | None = None,
) -> dict[str, float | int | str]:
    """Evaluate likelihood-density errors on one fixed audit table.

    Every separate-flow arm uses the identical first half of the independent
    audit bank.  This keeps paired method contrasts from being confounded by
    each inference method's proposal distribution.
    """

    audit = load_slcp_audit(artifact_root)
    requested = int(campaign.get("audit_joint_samples", campaign["metric_max_samples"]))
    n = min(requested, len(audit["theta"]) // 2)
    if n < 32:
        raise RuntimeError("The common likelihood audit needs at least 32 rows.")
    theta = np.asarray(audit["theta"][:n], dtype=np.float32)
    latent = np.asarray(audit["z"][:n], dtype=np.float32)
    x = np.asarray(audit["x"][:n], dtype=np.float32)
    points = np.column_stack([latent, x]).astype(np.float32)
    log_q_eta = likelihood_proposal_log_prob(
        q_eta,
        x,
        latent=latent,
        base_scale=float(base_scale),
    )
    log_r_l = (
        np.zeros(n, dtype=np.float64)
        if likelihood_log_ratio is None
        else np.asarray(likelihood_log_ratio(points), dtype=np.float64)
    )
    learned_log_likelihood = log_q_eta + log_r_l
    exact_log_likelihood = exact_slcp_log_likelihood(theta, x)
    error = _finite_error_summary(
        learned_log_likelihood,
        exact_log_likelihood,
        require_all_exact_finite_rows=True,
    )
    return {
        "exact_likelihood_log_error": error["rmse"],
        "exact_likelihood_centered_log_error": error["centered_rmse"],
        "exact_likelihood_rows": error["rows"],
        "likelihood_audit_rows": int(n),
        "likelihood_audit_fingerprint": _array_sha256(theta, x),
        "audit_bank_fingerprint": str(audit["fingerprint"]),
    }


def _separate_flow_common_cycle_audit(
    *,
    artifact_root: str | Path,
    campaign: Mapping[str, Any],
    q_phi: Sequence[Mapping[str, Any]],
    q_eta: Sequence[Mapping[str, Any]],
    observation: np.ndarray,
    base_scale: float = 1.0,
    prior_fraction: float = 0.0,
    posterior_log_ratio: Any | None = None,
    likelihood_log_ratio: Any | None = None,
) -> dict[str, float | int | str]:
    """Check Bayes' identity on a fixed theta grid at one fixed observation."""

    audit = load_slcp_audit(artifact_root)
    requested = int(campaign.get("audit_joint_samples", campaign["metric_max_samples"]))
    n = min(requested, len(audit["theta"]) // 2)
    theta = np.asarray(audit["theta"][:n], dtype=np.float32)
    latent = np.asarray(audit["z"][:n], dtype=np.float32)
    observation = np.asarray(observation, dtype=np.float32).reshape(1, -1)
    repeated_observation = np.repeat(observation, n, axis=0)
    points = np.column_stack([latent, repeated_observation]).astype(np.float32)
    transform = slcp_parameter_transform()
    log_q_phi = posterior_proposal_log_prob(
        q_phi,
        latent,
        context=repeated_observation,
        base_scale=float(base_scale),
        prior_fraction=float(prior_fraction),
        transform=transform,
    )
    log_q_eta = likelihood_proposal_log_prob(
        q_eta,
        repeated_observation,
        latent=latent,
        base_scale=float(base_scale),
    )
    log_r_p = (
        np.zeros(n, dtype=np.float64)
        if posterior_log_ratio is None
        else np.asarray(posterior_log_ratio(points), dtype=np.float64)
    )
    log_r_l = (
        np.zeros(n, dtype=np.float64)
        if likelihood_log_ratio is None
        else np.asarray(likelihood_log_ratio(points), dtype=np.float64)
    )
    log_prior = transform.prior_log_prob_latent(latent)
    valid = (
        np.isfinite(log_q_phi)
        & np.isfinite(log_r_p)
        & np.isfinite(log_prior)
        & np.isfinite(log_q_eta)
        & np.isfinite(log_r_l)
    )
    cycle = _cycle_diagnostics(
        log_q_phi[valid] + log_r_p[valid] - log_prior[valid],
        log_q_eta[valid] + log_r_l[valid],
    )
    return {
        "bayes_cycle_pearson": cycle["pearson"],
        "bayes_cycle_slope": cycle["slope"],
        "bayes_cycle_residual_rms": cycle["rms"],
        "bayes_cycle_rows": int(valid.sum()),
        "bayes_cycle_theta_fingerprint": _array_sha256(theta),
    }


def evaluate_matched_jana(
    *,
    artifact_root: str | Path,
    campaign: Mapping[str, Any],
    budget: int,
    ml_seed: int,
    q_phi: Sequence[Mapping[str, Any]],
    q_eta: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Evaluate nominal matched flows through posterior and likelihood routes."""

    import sbibm

    artifact_root = Path(artifact_root).expanduser().resolve()
    task = sbibm.get_task("slcp")
    transform = slcp_parameter_transform()
    n_posterior = int(campaign["posterior_samples"])
    n_proposal = int(campaign["proposal_candidates"])
    max_samples = int(campaign["metric_max_samples"])
    rows: list[dict[str, Any]] = []
    signature = _config_module().campaign_signature(campaign)
    sample_root = (
        artifact_root / "results" / "separate_flows" / signature
        / f"budget_{int(budget)}" / f"seed_{int(ml_seed)}" / "samples"
    )
    sample_root.mkdir(parents=True, exist_ok=True)
    bank = load_slcp_budget(artifact_root, int(budget))
    audit_metrics = _audit_joint_diagnostics(
        artifact_root=artifact_root,
        campaign=campaign,
        q_phi=q_phi,
        q_eta=q_eta,
        seed=int(ml_seed) + 70_000,
    )
    density_audit = _separate_flow_common_density_audit(
        artifact_root=artifact_root,
        campaign=campaign,
        q_eta=q_eta,
    )
    for observation_number in campaign["observations"]:
        observation_number = int(observation_number)
        observation = _task_observation(task, observation_number)
        cycle_audit = _separate_flow_common_cycle_audit(
            artifact_root=artifact_root,
            campaign=campaign,
            q_phi=q_phi,
            q_eta=q_eta,
            observation=observation,
        )
        nominal_z = sample_spline_flow_ensemble(
            q_phi,
            n_posterior,
            context=observation,
            seed=int(ml_seed) + 10_000 + observation_number,
            allocation="iid",
        ).reshape(n_posterior, -1)
        nominal_theta = transform.inverse(nominal_z)

        candidate_z = sample_spline_flow_ensemble(
            q_phi,
            n_proposal,
            context=observation,
            seed=int(ml_seed) + 20_000 + observation_number,
            allocation="iid",
        ).reshape(n_proposal, -1)
        repeated_observation = np.repeat(observation, n_proposal, axis=0)
        log_q_phi = spline_flow_ensemble_log_prob(
            q_phi, candidate_z, context=repeated_observation
        )
        log_q_eta = spline_flow_ensemble_log_prob(
            q_eta, repeated_observation, context=candidate_z
        )
        log_prior = transform.prior_log_prob_latent(candidate_z)
        weights = normalized_log_weights(log_prior + log_q_eta - log_q_phi)
        likelihood_z = systematic_resample(
            candidate_z,
            weights,
            n_posterior,
            int(ml_seed) + 30_000 + observation_number,
        )
        likelihood_theta = transform.inverse(likelihood_z)
        reference = _reference_posterior(task, observation_number)
        posterior_metrics = distribution_metrics(
            reference,
            nominal_theta,
            seed=int(ml_seed) + 40_000 + observation_number,
            max_samples=max_samples,
        )
        likelihood_metrics = distribution_metrics(
            reference,
            likelihood_theta,
            seed=int(ml_seed) + 50_000 + observation_number,
            max_samples=max_samples,
        )
        route_metrics = distribution_metrics(
            nominal_theta,
            likelihood_theta,
            seed=int(ml_seed) + 60_000 + observation_number,
            max_samples=max_samples,
        )
        exact_log_likelihood = exact_slcp_log_likelihood(
            transform.inverse(candidate_z), repeated_observation
        )
        likelihood_error = log_q_eta - exact_log_likelihood
        likelihood_error_summary = _finite_error_summary(
            log_q_eta, exact_log_likelihood
        )
        cycle = _cycle_diagnostics(log_q_phi - log_prior, log_q_eta)
        diagnostics = weight_diagnostics(weights)
        np.savez_compressed(
            sample_root / f"observation_{observation_number:02d}.npz",
            reference_theta=reference,
            posterior_theta=nominal_theta,
            likelihood_theta=likelihood_theta,
            likelihood_candidate_z=candidate_z,
            likelihood_weights=weights,
            exact_log_likelihood=exact_log_likelihood,
            learned_log_likelihood=log_q_eta,
        )
        rows.append(
            {
                "schema": PAPER_RUNTIME_SCHEMA,
                "campaign_signature": signature,
                "method": "separate_flows",
                "factorization": "none",
                "budget": int(budget),
                "simulator_calls": int(budget),
                "training_rows": int(len(bank["training_ids"])),
                "validation_rows": int(len(bank["validation_ids"])),
                "ml_seed": int(ml_seed),
                "observation": observation_number,
                "posterior_C2ST": posterior_metrics["C2ST"],
                "posterior_MMD": posterior_metrics["MMD"],
                "likelihood_posterior_C2ST": likelihood_metrics["C2ST"],
                "likelihood_posterior_MMD": likelihood_metrics["MMD"],
                "posterior_likelihood_route_C2ST": route_metrics["C2ST"],
                "posterior_likelihood_route_MMD": route_metrics["MMD"],
                "posterior_ESS_fraction": 1.0,
                "posterior_max_weight": float(1.0 / n_posterior),
                "likelihood_posterior_ESS_fraction": diagnostics["ess_fraction"],
                "likelihood_posterior_max_weight": diagnostics["max_weight"],
                "bayes_cycle_pearson": cycle_audit["bayes_cycle_pearson"],
                "bayes_cycle_slope": cycle_audit["bayes_cycle_slope"],
                "bayes_cycle_residual_rms": cycle_audit[
                    "bayes_cycle_residual_rms"
                ],
                "bayes_cycle_rows": cycle_audit["bayes_cycle_rows"],
                "bayes_cycle_theta_fingerprint": cycle_audit[
                    "bayes_cycle_theta_fingerprint"
                ],
                "likelihood_log_Z_rms": 0.0,
                "likelihood_log_Z_mean": 0.0,
                "likelihood_log_Z_max_abs": 0.0,
                "exact_likelihood_log_error": density_audit[
                    "exact_likelihood_log_error"
                ],
                "exact_likelihood_centered_log_error": density_audit[
                    "exact_likelihood_centered_log_error"
                ],
                "exact_likelihood_rows": density_audit["exact_likelihood_rows"],
                "likelihood_audit_rows": density_audit["likelihood_audit_rows"],
                "likelihood_audit_fingerprint": density_audit[
                    "likelihood_audit_fingerprint"
                ],
                "audit_bank_fingerprint": density_audit[
                    "audit_bank_fingerprint"
                ],
                "deployed_proposal_exact_likelihood_log_error": likelihood_error_summary[
                    "rmse"
                ],
                "deployed_proposal_exact_likelihood_centered_log_error": likelihood_error_summary[
                    "centered_rmse"
                ],
                "deployed_proposal_exact_likelihood_rows": likelihood_error_summary[
                    "rows"
                ],
                "deployed_proposal_bayes_cycle_pearson": cycle["pearson"],
                "deployed_proposal_bayes_cycle_slope": cycle["slope"],
                "deployed_proposal_bayes_cycle_residual_rms": cycle["rms"],
                "deployed_proposal_bayes_cycle_rows": int(len(candidate_z)),
                "proposal_base_scale": 1.0,
                "proposal_prior_fraction": 0.0,
                **audit_metrics,
            }
        )
    return pd.DataFrame(rows)


def run_jana_campaign(
    artifact_root: str | Path,
    campaign: Mapping[str, Any],
    *,
    run_exact_paper: bool = True,
    run_matched: bool = True,
    budgets_to_run: Sequence[int] | None = None,
    ml_seeds_to_run: Sequence[int] | None = None,
    load_if_available: bool = True,
) -> dict[str, Any]:
    """Run the external exact JANA baseline and the shared matched baseline."""

    paper_config = _config_module()
    paper_config.validate_campaign_config(campaign)
    artifact_root = Path(artifact_root).expanduser().resolve()
    signature = paper_config.campaign_signature(campaign)
    result_root = artifact_root / "results" / "jana"
    result_root.mkdir(parents=True, exist_ok=True)
    output: dict[str, Any] = {}
    run_budgets = _execution_subset(
        campaign["budgets"], budgets_to_run, "budget"
    )
    run_seeds = _execution_subset(
        campaign["ml_seeds"], ml_seeds_to_run, "ML-seed"
    )
    if run_matched:
        matched_path = result_root / f"separate_flows_{signature}.csv"
        for budget in run_budgets:
            for ml_seed in run_seeds:
                per_run_path = (
                    artifact_root
                    / "results"
                    / "separate_flows"
                    / signature
                    / f"budget_{int(budget)}"
                    / f"seed_{int(ml_seed)}"
                    / "metrics.csv"
                )
                if not (load_if_available and per_run_path.exists()):
                    trained = train_matched_flow_pair(
                        artifact_root,
                        campaign,
                        budget=int(budget),
                        ml_seed=int(ml_seed),
                        load_if_available=load_if_available,
                    )
                    per_run = evaluate_matched_jana(
                            artifact_root=artifact_root,
                            campaign=campaign,
                            budget=int(budget),
                            ml_seed=int(ml_seed),
                            q_phi=trained["q_phi"],
                            q_eta=trained["q_eta"],
                        )
                    per_run_path.parent.mkdir(parents=True, exist_ok=True)
                    _write_csv(per_run_path, per_run)
        run_files = sorted(
            (artifact_root / "results" / "separate_flows" / signature).glob(
                "budget_*/seed_*/metrics.csv"
            )
        )
        if not run_files:
            raise RuntimeError("No separate-flow metric shards were produced.")
        matched = pd.concat(
            [pd.read_csv(path) for path in run_files], ignore_index=True
        ).drop_duplicates(
            ["method", "budget", "ml_seed", "observation"], keep="last"
        )
        _write_csv(matched_path, matched)
        output["separate_flows"] = matched
    if run_exact_paper:
        try:
            from .utils_jana import run_exact_jana_campaign
        except ImportError:
            from utils_jana import run_exact_jana_campaign
        exact = run_exact_jana_campaign(
            artifact_root=artifact_root,
            campaign=campaign,
            budgets_to_run=run_budgets,
            ml_seeds_to_run=run_seeds,
            load_if_available=load_if_available,
        )
        if not isinstance(exact, pd.DataFrame):
            exact = pd.DataFrame(exact)
        output["jana_paper"] = exact
    return output


# ---------------------------------------------------------------------------
# Hybrid class banks, proposal selection, and corrected inference
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HybridClassBank:
    simulator: np.ndarray
    posterior_reference: np.ndarray
    likelihood_reference: np.ndarray
    source_indices: np.ndarray
    proposal_scale: float
    prior_fraction: float
    fingerprint: str

    @property
    def groups(self) -> np.ndarray:
        return np.stack(
            [self.simulator, self.posterior_reference, self.likelihood_reference],
            axis=1,
        ).astype(np.float32)


def build_hybrid_class_bank(
    *,
    latent: np.ndarray,
    x: np.ndarray,
    source_indices: np.ndarray,
    q_phi: Sequence[Mapping[str, Any]],
    q_eta: Sequence[Mapping[str, Any]],
    base_scale: float,
    prior_fraction: float,
    seed: int,
) -> HybridClassBank:
    """Build paired S/P/L rows from one shared set of genuine simulations."""

    latent = _as_2d_float32(latent, "latent")
    x = _as_2d_float32(x, "x")
    source_indices = np.asarray(source_indices, dtype=np.int64).reshape(-1)
    if len(latent) != len(x) or len(latent) != len(source_indices):
        raise ValueError("Hybrid class-bank arrays must have equal row counts.")
    posterior_latent = sample_posterior_proposal(
        q_phi,
        1,
        context=x,
        seed=int(seed) + 1,
        base_scale=base_scale,
        prior_fraction=prior_fraction,
    )[:, 0, :]
    likelihood_x = sample_likelihood_proposal(
        q_eta,
        1,
        latent=latent,
        seed=int(seed) + 2,
        base_scale=base_scale,
    )[:, 0, :]
    simulator = np.column_stack([latent, x]).astype(np.float32)
    posterior_reference = np.column_stack([posterior_latent, x]).astype(np.float32)
    likelihood_reference = np.column_stack([latent, likelihood_x]).astype(np.float32)
    fingerprint = _array_sha256(
        simulator,
        posterior_reference,
        likelihood_reference,
        source_indices,
        np.asarray([base_scale, prior_fraction], dtype=np.float64),
    )
    return HybridClassBank(
        simulator=simulator,
        posterior_reference=posterior_reference,
        likelihood_reference=likelihood_reference,
        source_indices=source_indices,
        proposal_scale=float(base_scale),
        prior_fraction=float(prior_fraction),
        fingerprint=fingerprint,
    )


def _load_ratio_api():
    try:
        from . import utils_ratio
    except ImportError:
        import utils_ratio
    return utils_ratio


def _proposal_selection_path(
    artifact_root: Path, campaign: Mapping[str, Any]
) -> Path:
    signature = _config_module().campaign_signature(campaign)
    return (
        artifact_root / "results" / "hybrid"
        / f"selected_proposal_{signature}.json"
    )


def _proposal_grid(campaign: Mapping[str, Any]) -> list[tuple[float, float]]:
    return [
        (float(scale), float(fraction))
        for scale in campaign["proposal"]["broad_base_scales"]
        for fraction in campaign["proposal"]["prior_fractions"]
    ]


def _proposal_tag(base_scale: float, prior_fraction: float) -> str:
    scale = str(float(base_scale)).replace(".", "p")
    fraction = str(float(prior_fraction)).replace(".", "p")
    return f"tau_{scale}_epsilon_{fraction}"


def _hybrid_ratio_bank(
    values: HybridClassBank,
    *,
    role: str,
    campaign_signature: str,
    budget: int,
    ml_seed: int,
    split_fingerprint: str,
    source_bank_fingerprint: str,
) -> Any:
    ratio_api = _load_ratio_api()
    return ratio_api.RatioClassBank(
        simulator=values.simulator,
        posterior_reference=values.posterior_reference,
        likelihood_reference=values.likelihood_reference,
        source_bank_fingerprint=str(source_bank_fingerprint),
        row_ids=values.source_indices,
        provenance={
            "schema": PAPER_RUNTIME_SCHEMA,
            "role": str(role),
            "campaign_signature": str(campaign_signature),
            "budget": int(budget),
            "ml_seed": int(ml_seed),
            "split_fingerprint": str(split_fingerprint),
            "source_indices_sha256": _array_sha256(values.source_indices),
            "hybrid_bank_fingerprint": values.fingerprint,
            "proposal_base_scale": float(values.proposal_scale),
            "proposal_prior_fraction": float(values.prior_fraction),
        },
    )


def select_hybrid_proposal(
    artifact_root: str | Path,
    campaign: Mapping[str, Any],
    *,
    include_ablation: bool = True,
    load_if_available: bool = True,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Select (tau, epsilon) using only split validation rows.

    A small multiclass pilot is fitted on one half of the validation
    partition and evaluated on the other.  The official SBIBM posterior and
    the independent audit bank are never touched.  Candidate corrections are
    ranked by worst-route joint C2ST deviation from 0.5; statistically tied
    candidates are resolved by the smaller worst-route weight ESS.
    """

    paper_config = _config_module()
    paper_config.validate_campaign_config(campaign)
    artifact_root = Path(artifact_root).expanduser().resolve()
    signature = paper_config.campaign_signature(campaign)
    selection_path = _proposal_selection_path(artifact_root, campaign)
    scan_path = selection_path.with_name(
        selection_path.name.replace("selected_proposal_", "proposal_scan_")
    ).with_suffix(".csv")
    if load_if_available and selection_path.exists() and scan_path.exists():
        selected = json.loads(selection_path.read_text())
        if selected.get("campaign_signature") != signature:
            raise RuntimeError("Proposal selection belongs to another campaign.")
        return {"selection": selected, "scan": pd.read_csv(scan_path)}

    requested_budget = int(campaign["proposal"]["selection_budget"])
    if requested_budget in set(int(value) for value in campaign["budgets"]):
        selection_budget = requested_budget
    else:
        # The SMOKE profile intentionally has no 100k point.  Its largest
        # nested prefix exercises the same code path and is explicitly marked
        # as a non-paper substitute in the manifest.
        selection_budget = max(int(value) for value in campaign["budgets"])
    headline_candidates = _proposal_grid(campaign)
    candidates = list(headline_candidates)
    if include_ablation:
        candidates.extend([(1.0, 0.0)])
        candidates.extend(
            (float(scale), 0.0)
            for scale in campaign["proposal"]["broad_base_scales"]
            if float(scale) != 1.0
        )
    candidates = list(dict.fromkeys(candidates))
    ratio_api = _load_ratio_api()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, Any]] = []
    for ml_seed in campaign["ml_seeds"]:
        ml_seed = int(ml_seed)
        trained = train_matched_flow_pair(
            artifact_root,
            campaign,
            budget=selection_budget,
            ml_seed=ml_seed,
            load_if_available=load_if_available,
            device=device,
        )
        bank = trained["bank"]
        validation_ids = np.asarray(bank["validation_ids"], dtype=np.int64)
        if len(validation_ids) < 96:
            raise RuntimeError(
                "Proposal selection requires at least 96 validation rows."
            )
        rng = np.random.default_rng(
            int(campaign["split_seed"]) + ml_seed + selection_budget
        )
        order = validation_ids[rng.permutation(len(validation_ids))]
        split_fractions = tuple(
            float(value)
            for value in campaign["proposal"]["selection_pilot_split_fractions"]
        )
        if len(split_fractions) != 3 or not math.isclose(
            sum(split_fractions), 1.0, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError(
                "selection_pilot_split_fractions must contain three values summing to one."
            )
        n_train = max(32, int(round(split_fractions[0] * len(order))))
        n_checkpoint = max(32, int(round(split_fractions[1] * len(order))))
        if n_train + n_checkpoint > len(order) - 32:
            n_train = len(order) - 64
            n_checkpoint = 32
        pilot_train_ids = order[:n_train]
        checkpoint_ids = order[n_train : n_train + n_checkpoint]
        closure_ids = order[n_train + n_checkpoint :]
        if min(len(pilot_train_ids), len(checkpoint_ids), len(closure_ids)) < 32:
            raise RuntimeError("Every proposal-selection split needs at least 32 rows.")
        pilot_config = dict(campaign["ratio_training"])
        pilot_config["ensemble_members_override"] = min(
            int(campaign["proposal"].get("selection_pilot_members", 3)),
            int(
                campaign["ratio_training"].get("ensemble_members_override")
                or campaign["ratio_training"]["ensemble_members"]
            ),
        )
        pilot_config["training_steps_override"] = min(
            int(campaign["proposal"].get("selection_pilot_steps", 2_000)),
            int(
                campaign["ratio_training"].get("training_steps_override")
                or campaign["ratio_training"]["training_steps"]
            ),
        )
        for candidate_index, (base_scale, prior_fraction) in enumerate(candidates):
            pilot_train = build_hybrid_class_bank(
                latent=bank["z"][pilot_train_ids],
                x=bank["x"][pilot_train_ids],
                source_indices=pilot_train_ids,
                q_phi=trained["q_phi"],
                q_eta=trained["q_eta"],
                base_scale=base_scale,
                prior_fraction=prior_fraction,
                seed=ml_seed + 100_000 + 100 * candidate_index,
            )
            pilot_checkpoint = build_hybrid_class_bank(
                latent=bank["z"][checkpoint_ids],
                x=bank["x"][checkpoint_ids],
                source_indices=checkpoint_ids,
                q_phi=trained["q_phi"],
                q_eta=trained["q_eta"],
                base_scale=base_scale,
                prior_fraction=prior_fraction,
                seed=ml_seed + 200_000 + 100 * candidate_index,
            )
            pilot_closure = build_hybrid_class_bank(
                latent=bank["z"][closure_ids],
                x=bank["x"][closure_ids],
                source_indices=closure_ids,
                q_phi=trained["q_phi"],
                q_eta=trained["q_eta"],
                base_scale=base_scale,
                prior_fraction=prior_fraction,
                seed=ml_seed + 250_000 + 100 * candidate_index,
            )
            train_ratio_bank = _hybrid_ratio_bank(
                pilot_train,
                role="proposal_selection_train",
                campaign_signature=signature,
                budget=selection_budget,
                ml_seed=ml_seed,
                split_fingerprint=bank["split_fingerprint"],
                source_bank_fingerprint=bank["master_fingerprint"],
            )
            checkpoint_ratio_bank = _hybrid_ratio_bank(
                pilot_checkpoint,
                role="proposal_selection_checkpoint_validation",
                campaign_signature=signature,
                budget=selection_budget,
                ml_seed=ml_seed,
                split_fingerprint=bank["split_fingerprint"],
                source_bank_fingerprint=bank["master_fingerprint"],
            )
            trained_families: dict[str, Sequence[Mapping[str, Any]]] = {}
            family_order = (
                ratio_api.FAMILY_MULTICLASS,
                ratio_api.FAMILY_POSTERIOR_BINARY,
                ratio_api.FAMILY_LIKELIHOOD_BINARY,
            )
            for family_index, family in enumerate(family_order):
                checkpoint_root = (
                    artifact_root
                    / "models"
                    / "proposal_selection"
                    / signature
                    / f"budget_{selection_budget}"
                    / f"seed_{ml_seed}"
                    / _proposal_tag(base_scale, prior_fraction)
                    / family
                )
                trained_families[family] = ratio_api.train_classifier_family(
                    family,
                    train_ratio_bank,
                    checkpoint_ratio_bank,
                    checkpoint_root,
                    pilot_config,
                    seed=(
                        ml_seed
                        + 300_000
                        + 1_000 * candidate_index
                        + 100_000 * family_index
                    ),
                    validation_seed=(
                        ml_seed
                        + 700_000
                        + 1_000 * candidate_index
                        + 100_000 * family_index
                    ),
                    device=device,
                    study_metadata={
                        "stage": "proposal_selection",
                        "campaign_signature": signature,
                        "selection_budget": selection_budget,
                        "proposal_base_scale": base_scale,
                        "proposal_prior_fraction": prior_fraction,
                        "closure_rows_sha256": _array_sha256(closure_ids),
                        "closure_rows_never_used_for_checkpointing": True,
                    },
                    load_if_available=load_if_available,
                )
            for factorization_index, factorization in enumerate(
                campaign["proposal"]["selection_factorizations"]
            ):
                factorization = str(factorization)
                log_r_p = ratio_api.predict_posterior_log_ratio(
                    trained_families,
                    pilot_closure.posterior_reference,
                    factorization=factorization,
                )
                log_r_l = ratio_api.predict_likelihood_log_ratio(
                    trained_families,
                    pilot_closure.likelihood_reference,
                    factorization=factorization,
                )
                weights_p = normalized_log_weights(log_r_p)
                weights_l = normalized_log_weights(log_r_l)
                metric_seed = (
                    ml_seed
                    + 1_100_000
                    + 10_000 * factorization_index
                    + candidate_index
                )
                corrected_p = systematic_resample(
                    pilot_closure.posterior_reference,
                    weights_p,
                    len(pilot_closure.source_indices),
                    metric_seed + 1_000,
                )
                corrected_l = systematic_resample(
                    pilot_closure.likelihood_reference,
                    weights_l,
                    len(pilot_closure.source_indices),
                    metric_seed + 2_000,
                )
                c2st_p = c2st_metric(
                    pilot_closure.simulator,
                    corrected_p,
                    seed=metric_seed + 3_000,
                    max_samples=int(campaign["metric_max_samples"]),
                )
                c2st_l = c2st_metric(
                    pilot_closure.simulator,
                    corrected_l,
                    seed=metric_seed + 4_000,
                    max_samples=int(campaign["metric_max_samples"]),
                )
                diagnostics_p = weight_diagnostics(weights_p)
                diagnostics_l = weight_diagnostics(weights_l)
                used_families = (
                    (ratio_api.FAMILY_MULTICLASS,)
                    if factorization == "multiclass"
                    else (
                        ratio_api.FAMILY_POSTERIOR_BINARY,
                        ratio_api.FAMILY_LIKELIHOOD_BINARY,
                    )
                )
                rows.append(
                    {
                        "schema": PAPER_RUNTIME_SCHEMA,
                        "campaign_signature": signature,
                        "selection_budget_requested": requested_budget,
                        "selection_budget": selection_budget,
                        "selection_budget_is_smoke_substitute": bool(
                            selection_budget != requested_budget
                        ),
                        "ml_seed": ml_seed,
                        "factorization": factorization,
                        "proposal_base_scale": base_scale,
                        "proposal_prior_fraction": prior_fraction,
                        "headline_candidate": (base_scale, prior_fraction)
                        in headline_candidates,
                        "pilot_training_rows": int(len(pilot_train_ids)),
                        "pilot_checkpoint_validation_rows": int(
                            len(checkpoint_ids)
                        ),
                        "pilot_closure_rows": int(len(closure_ids)),
                        "pilot_closure_rows_sha256": _array_sha256(closure_ids),
                        "posterior_joint_C2ST": c2st_p,
                        "likelihood_joint_C2ST": c2st_l,
                        "worst_C2ST_deviation": float(
                            max(abs(c2st_p - 0.5), abs(c2st_l - 0.5))
                        ),
                        "posterior_ESS_fraction": diagnostics_p["ess_fraction"],
                        "likelihood_ESS_fraction": diagnostics_l["ess_fraction"],
                        "minimum_ESS_fraction": float(
                            min(
                                diagnostics_p["ess_fraction"],
                                diagnostics_l["ess_fraction"],
                            )
                        ),
                        "pilot_selected_validation_CE": float(
                            np.mean(
                                [
                                    pack["history"]["selected_validation_ce"]
                                    for family in used_families
                                    for pack in trained_families[family]
                                ]
                            )
                        ),
                    }
                )
    scan = pd.DataFrame(rows)
    headline = scan[scan["headline_candidate"]].copy()
    expected_factorizations = {
        str(value) for value in campaign["proposal"]["selection_factorizations"]
    }
    observed_factorizations = set(headline["factorization"].astype(str))
    if observed_factorizations != expected_factorizations:
        raise RuntimeError(
            "Proposal scan is missing a preregistered correction factorization."
        )
    per_seed = (
        headline.groupby(
            ["proposal_base_scale", "proposal_prior_fraction", "ml_seed"],
            as_index=False,
        )
        .agg(
            worst_factorization_closure=("worst_C2ST_deviation", "max"),
            worst_factorization_minimum_ESS=("minimum_ESS_fraction", "min"),
            closure_rows=("pilot_closure_rows", "min"),
            n_factorizations=("factorization", "nunique"),
        )
    )
    if not np.all(
        per_seed["n_factorizations"].to_numpy()
        == len(expected_factorizations)
    ):
        raise RuntimeError("Proposal scan has incomplete per-seed factorizations.")
    aggregate = (
        per_seed.groupby(
            ["proposal_base_scale", "proposal_prior_fraction"], as_index=False
        )
        .agg(
            closure_mean=("worst_factorization_closure", "mean"),
            closure_std=("worst_factorization_closure", "std"),
            minimum_ESS_mean=("worst_factorization_minimum_ESS", "mean"),
            evaluation_rows=("closure_rows", "min"),
            n_seeds=("ml_seed", "nunique"),
        )
    )
    aggregate["closure_sem"] = aggregate["closure_std"] / np.sqrt(
        aggregate["n_seeds"].clip(lower=1)
    )
    best = aggregate.loc[aggregate["closure_mean"].idxmin()]
    statistical_floor = 0.5 / math.sqrt(max(1, int(best["evaluation_rows"])))
    best_sem = float(best["closure_sem"])
    if not np.isfinite(best_sem):
        best_sem = statistical_floor
    threshold = float(best["closure_mean"] + max(best_sem, statistical_floor))
    eligible = aggregate[aggregate["closure_mean"] <= threshold]
    choice = eligible.sort_values(
        [
            "minimum_ESS_mean",
            "proposal_base_scale",
            "proposal_prior_fraction",
        ],
        ascending=[False, True, True],
    ).iloc[0]
    selected = {
        "schema": PAPER_RUNTIME_SCHEMA,
        "campaign_signature": signature,
        "selection_budget_requested": requested_budget,
        "selection_budget": selection_budget,
        "selection_budget_is_smoke_substitute": bool(
            selection_budget != requested_budget
        ),
        "uses_reference_posterior": False,
        "uses_audit_bank": False,
        "pilot_split": (
            "validation_partition_split_once_into_disjoint_train_"
            "checkpoint_validation_and_untouched_closure"
        ),
        "selection_factorizations": sorted(expected_factorizations),
        "ranking": (
            "worst_route_and_factorization_within_one_uncertainty_of_best_"
            "joint_closure_then_highest_worst_factorization_minimum_ESS"
        ),
        "closure_threshold": threshold,
        "proposal_base_scale": float(choice["proposal_base_scale"]),
        "proposal_prior_fraction": float(choice["proposal_prior_fraction"]),
        "closure_mean": float(choice["closure_mean"]),
        "minimum_ESS_mean": float(choice["minimum_ESS_mean"]),
        "frozen_across_budgets": True,
        "created_utc": _utc_now(),
    }
    scan_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(scan_path, scan)
    _write_json(selection_path, selected)
    return {"selection": selected, "scan": scan}


def train_hybrid_ratio_models(
    artifact_root: str | Path,
    campaign: Mapping[str, Any],
    *,
    budget: int,
    ml_seed: int,
    q_phi: Sequence[Mapping[str, Any]],
    q_eta: Sequence[Mapping[str, Any]],
    base_scale: float,
    prior_fraction: float,
    load_if_available: bool = True,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Fit all three CE families on the complete shared training partition."""

    artifact_root = Path(artifact_root).expanduser().resolve()
    signature = _config_module().campaign_signature(campaign)
    bank = load_slcp_budget(artifact_root, int(budget))
    train_ids = np.asarray(bank["training_ids"], dtype=np.int64)
    validation_ids = np.asarray(bank["validation_ids"], dtype=np.int64)
    training = build_hybrid_class_bank(
        latent=bank["z"][train_ids],
        x=bank["x"][train_ids],
        source_indices=train_ids,
        q_phi=q_phi,
        q_eta=q_eta,
        base_scale=base_scale,
        prior_fraction=prior_fraction,
        seed=int(ml_seed) + 910_000,
    )
    validation = build_hybrid_class_bank(
        latent=bank["z"][validation_ids],
        x=bank["x"][validation_ids],
        source_indices=validation_ids,
        q_phi=q_phi,
        q_eta=q_eta,
        base_scale=base_scale,
        prior_fraction=prior_fraction,
        seed=int(ml_seed) + 920_000,
    )
    train_ratio_bank = _hybrid_ratio_bank(
        training,
        role="ratio_training",
        campaign_signature=signature,
        budget=int(budget),
        ml_seed=int(ml_seed),
        split_fingerprint=bank["split_fingerprint"],
        source_bank_fingerprint=bank["master_fingerprint"],
    )
    validation_ratio_bank = _hybrid_ratio_bank(
        validation,
        role="ratio_validation",
        campaign_signature=signature,
        budget=int(budget),
        ml_seed=int(ml_seed),
        split_fingerprint=bank["split_fingerprint"],
        source_bank_fingerprint=bank["master_fingerprint"],
    )
    ratio_api = _load_ratio_api()
    model_root = (
        artifact_root
        / "models"
        / "hybrid"
        / signature
        / f"budget_{int(budget)}"
        / f"seed_{int(ml_seed)}"
        / _proposal_tag(base_scale, prior_fraction)
    )
    classifiers = ratio_api.train_ratio_ensembles(
        train_ratio_bank,
        validation_ratio_bank,
        model_root,
        campaign["ratio_training"],
        seed=int(ml_seed) + 930_000,
        device=device,
        study_metadata={
            "stage": "headline_hybrid",
            "campaign_signature": signature,
            "budget": int(budget),
            "ml_seed": int(ml_seed),
            "proposal_base_scale": float(base_scale),
            "proposal_prior_fraction": float(prior_fraction),
            "shared_flow_checkpoints": True,
        },
        load_if_available=load_if_available,
    )
    manifest = {
        "schema": PAPER_RUNTIME_SCHEMA,
        "campaign_signature": signature,
        "budget": int(budget),
        "ml_seed": int(ml_seed),
        "proposal_base_scale": float(base_scale),
        "proposal_prior_fraction": float(prior_fraction),
        "training_rows": int(len(train_ids)),
        "validation_rows": int(len(validation_ids)),
        "split_fingerprint": bank["split_fingerprint"],
        "training_bank_fingerprint": training.fingerprint,
        "validation_bank_fingerprint": validation.fingerprint,
        "ratio_members": ratio_api.ratio_ensemble_summary(classifiers),
        "implementation": _implementation_manifest(),
        "created_utc": _utc_now(),
    }
    _write_json(model_root / "manifest.json", manifest)
    return {
        "classifiers": classifiers,
        "manifest": manifest,
        "training_bank": training,
        "validation_bank": validation,
    }


def evaluate_hybrid(
    *,
    artifact_root: str | Path,
    campaign: Mapping[str, Any],
    budget: int,
    ml_seed: int,
    q_phi: Sequence[Mapping[str, Any]],
    q_eta: Sequence[Mapping[str, Any]],
    classifiers: Mapping[str, Sequence[Mapping[str, Any]]],
    factorization: str,
    base_scale: float,
    prior_fraction: float,
) -> pd.DataFrame:
    """Evaluate one corrected factorization on both inference routes."""

    import sbibm

    factorization = str(factorization).lower()
    if factorization not in {"multiclass", "binary"}:
        raise ValueError("factorization must be 'multiclass' or 'binary'.")
    artifact_root = Path(artifact_root).expanduser().resolve()
    signature = _config_module().campaign_signature(campaign)
    ratio_api = _load_ratio_api()
    task = sbibm.get_task("slcp")
    transform = slcp_parameter_transform()
    n_posterior = int(campaign["posterior_samples"])
    n_proposal = int(campaign["proposal_candidates"])
    max_samples = int(campaign["metric_max_samples"])
    method = f"separate_flows_corrected_{factorization}"
    bank = load_slcp_budget(artifact_root, int(budget))

    def posterior_ratio(points: np.ndarray, *, return_members: bool = False):
        return ratio_api.predict_posterior_log_ratio(
            classifiers,
            points,
            factorization=factorization,
            return_members=return_members,
        )

    def likelihood_ratio(points: np.ndarray, *, return_members: bool = False):
        return ratio_api.predict_likelihood_log_ratio(
            classifiers,
            points,
            factorization=factorization,
            return_members=return_members,
        )

    audit_metrics = _audit_joint_diagnostics(
        artifact_root=artifact_root,
        campaign=campaign,
        q_phi=q_phi,
        q_eta=q_eta,
        seed=int(ml_seed) + 70_000,
        base_scale=base_scale,
        prior_fraction=prior_fraction,
        posterior_log_ratio=posterior_ratio,
        likelihood_log_ratio=likelihood_ratio,
    )
    normalization = _corrected_likelihood_normalization(
        artifact_root=artifact_root,
        campaign=campaign,
        q_eta=q_eta,
        seed=int(ml_seed) + 80_000,
        base_scale=base_scale,
        likelihood_log_ratio=likelihood_ratio,
    )
    density_audit = _separate_flow_common_density_audit(
        artifact_root=artifact_root,
        campaign=campaign,
        q_eta=q_eta,
        base_scale=base_scale,
        likelihood_log_ratio=likelihood_ratio,
    )
    ratio_families = (
        [ratio_api.FAMILY_MULTICLASS]
        if factorization == "multiclass"
        else [
            ratio_api.FAMILY_POSTERIOR_BINARY,
            ratio_api.FAMILY_LIKELIHOOD_BINARY,
        ]
    )
    selected_ce = [
        float(pack["history"]["selected_validation_ce"])
        for family in ratio_families
        for pack in classifiers[family]
    ]
    sample_root = (
        artifact_root
        / "results"
        / "hybrid"
        / signature
        / f"budget_{int(budget)}"
        / f"seed_{int(ml_seed)}"
        / factorization
        / "samples"
    )
    sample_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for observation_number in campaign["observations"]:
        observation_number = int(observation_number)
        observation = _task_observation(task, observation_number)
        cycle_audit = _separate_flow_common_cycle_audit(
            artifact_root=artifact_root,
            campaign=campaign,
            q_phi=q_phi,
            q_eta=q_eta,
            observation=observation,
            base_scale=base_scale,
            prior_fraction=prior_fraction,
            posterior_log_ratio=posterior_ratio,
            likelihood_log_ratio=likelihood_ratio,
        )
        candidate_z = sample_posterior_proposal(
            q_phi,
            n_proposal,
            context=observation,
            seed=int(ml_seed) + 1_500_000 + observation_number,
            base_scale=base_scale,
            prior_fraction=prior_fraction,
        ).reshape(n_proposal, -1)
        repeated_observation = np.repeat(observation, n_proposal, axis=0)
        points = np.column_stack([candidate_z, repeated_observation]).astype(np.float32)
        log_r_p, member_log_r_p = posterior_ratio(points, return_members=True)
        log_r_l, member_log_r_l = likelihood_ratio(points, return_members=True)
        log_g_phi = posterior_proposal_log_prob(
            q_phi,
            candidate_z,
            context=repeated_observation,
            base_scale=base_scale,
            prior_fraction=prior_fraction,
            transform=transform,
        )
        log_g_eta = likelihood_proposal_log_prob(
            q_eta,
            repeated_observation,
            latent=candidate_z,
            base_scale=base_scale,
        )
        log_prior = transform.prior_log_prob_latent(candidate_z)
        posterior_weights = normalized_log_weights(log_r_p)
        likelihood_weights = normalized_log_weights(
            log_prior + log_g_eta + log_r_l - log_g_phi
        )
        posterior_z = systematic_resample(
            candidate_z,
            posterior_weights,
            n_posterior,
            int(ml_seed) + 1_600_000 + observation_number,
        )
        likelihood_z = systematic_resample(
            candidate_z,
            likelihood_weights,
            n_posterior,
            int(ml_seed) + 1_700_000 + observation_number,
        )
        posterior_theta = transform.inverse(posterior_z)
        likelihood_theta = transform.inverse(likelihood_z)
        reference = _reference_posterior(task, observation_number)
        posterior_metrics = distribution_metrics(
            reference,
            posterior_theta,
            seed=int(ml_seed) + 40_000 + observation_number,
            max_samples=max_samples,
        )
        likelihood_metrics = distribution_metrics(
            reference,
            likelihood_theta,
            seed=int(ml_seed) + 50_000 + observation_number,
            max_samples=max_samples,
        )
        route_metrics = distribution_metrics(
            posterior_theta,
            likelihood_theta,
            seed=int(ml_seed) + 60_000 + observation_number,
            max_samples=max_samples,
        )
        posterior_diagnostics = weight_diagnostics(posterior_weights)
        likelihood_diagnostics = weight_diagnostics(likelihood_weights)
        exact_log_likelihood = exact_slcp_log_likelihood(
            transform.inverse(candidate_z), repeated_observation
        )
        corrected_log_likelihood = log_g_eta + log_r_l
        exact_error = _finite_error_summary(
            corrected_log_likelihood, exact_log_likelihood
        )
        cycle = _cycle_diagnostics(
            log_g_phi + log_r_p - log_prior,
            corrected_log_likelihood,
        )
        np.savez_compressed(
            sample_root / f"observation_{observation_number:02d}.npz",
            reference_theta=reference,
            candidate_z=candidate_z,
            posterior_theta=posterior_theta,
            likelihood_theta=likelihood_theta,
            posterior_weights=posterior_weights,
            likelihood_weights=likelihood_weights,
            posterior_log_ratio=log_r_p,
            likelihood_log_ratio=log_r_l,
            exact_log_likelihood=exact_log_likelihood,
            corrected_log_likelihood=corrected_log_likelihood,
        )
        rows.append(
            {
                "schema": PAPER_RUNTIME_SCHEMA,
                "campaign_signature": signature,
                "method": method,
                "factorization": factorization,
                "budget": int(budget),
                "simulator_calls": int(budget),
                "training_rows": int(len(bank["training_ids"])),
                "validation_rows": int(len(bank["validation_ids"])),
                "ml_seed": int(ml_seed),
                "observation": observation_number,
                "posterior_C2ST": posterior_metrics["C2ST"],
                "posterior_MMD": posterior_metrics["MMD"],
                "likelihood_posterior_C2ST": likelihood_metrics["C2ST"],
                "likelihood_posterior_MMD": likelihood_metrics["MMD"],
                "posterior_likelihood_route_C2ST": route_metrics["C2ST"],
                "posterior_likelihood_route_MMD": route_metrics["MMD"],
                "posterior_ESS_fraction": posterior_diagnostics["ess_fraction"],
                "posterior_max_weight": posterior_diagnostics["max_weight"],
                "likelihood_posterior_ESS_fraction": likelihood_diagnostics[
                    "ess_fraction"
                ],
                "likelihood_posterior_max_weight": likelihood_diagnostics[
                    "max_weight"
                ],
                "bayes_cycle_pearson": cycle_audit["bayes_cycle_pearson"],
                "bayes_cycle_slope": cycle_audit["bayes_cycle_slope"],
                "bayes_cycle_residual_rms": cycle_audit[
                    "bayes_cycle_residual_rms"
                ],
                "bayes_cycle_rows": cycle_audit["bayes_cycle_rows"],
                "bayes_cycle_theta_fingerprint": cycle_audit[
                    "bayes_cycle_theta_fingerprint"
                ],
                "likelihood_log_Z_rms": normalization["rms"],
                "likelihood_log_Z_mean": normalization["mean"],
                "likelihood_log_Z_max_abs": normalization["max_abs"],
                "exact_likelihood_log_error": density_audit[
                    "exact_likelihood_log_error"
                ],
                "exact_likelihood_centered_log_error": density_audit[
                    "exact_likelihood_centered_log_error"
                ],
                "exact_likelihood_rows": density_audit["exact_likelihood_rows"],
                "likelihood_audit_rows": density_audit["likelihood_audit_rows"],
                "likelihood_audit_fingerprint": density_audit[
                    "likelihood_audit_fingerprint"
                ],
                "audit_bank_fingerprint": density_audit[
                    "audit_bank_fingerprint"
                ],
                "deployed_proposal_exact_likelihood_log_error": exact_error["rmse"],
                "deployed_proposal_exact_likelihood_centered_log_error": exact_error[
                    "centered_rmse"
                ],
                "deployed_proposal_exact_likelihood_rows": exact_error["rows"],
                "deployed_proposal_bayes_cycle_pearson": cycle["pearson"],
                "deployed_proposal_bayes_cycle_slope": cycle["slope"],
                "deployed_proposal_bayes_cycle_residual_rms": cycle["rms"],
                "deployed_proposal_bayes_cycle_rows": int(len(candidate_z)),
                "proposal_base_scale": float(base_scale),
                "proposal_prior_fraction": float(prior_fraction),
                "posterior_ratio_member_log_std": float(
                    np.mean(np.std(member_log_r_p, axis=0))
                ),
                "likelihood_ratio_member_log_std": float(
                    np.mean(np.std(member_log_r_l, axis=0))
                ),
                "ratio_selected_validation_CE_mean": float(np.mean(selected_ce)),
                "ratio_selected_validation_CE_std": float(np.std(selected_ce)),
                **audit_metrics,
            }
        )
    return pd.DataFrame(rows)


def train_exact_jana_ratio_models(
    artifact_root: str | Path,
    campaign: Mapping[str, Any],
    *,
    budget: int,
    ml_seed: int,
    export_manifest: Mapping[str, Any],
    load_if_available: bool = True,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Train CE corrections on nominal S/P/L banks exported by exact JANA."""

    artifact_root = Path(artifact_root).expanduser().resolve()
    signature = _config_module().campaign_signature(campaign)
    arrays_path = Path(export_manifest["arrays_path"]).expanduser().resolve()
    if not arrays_path.is_file():
        raise FileNotFoundError(f"Missing exact-JANA ratio bank: {arrays_path}")
    with np.load(arrays_path, allow_pickle=False) as saved:
        train_s = np.asarray(saved["train_S"], dtype=np.float32)
        train_p = np.asarray(saved["train_P"], dtype=np.float32)
        train_l = np.asarray(saved["train_L"], dtype=np.float32)
        validation_s = np.asarray(saved["validation_S"], dtype=np.float32)
        validation_p = np.asarray(saved["validation_P"], dtype=np.float32)
        validation_l = np.asarray(saved["validation_L"], dtype=np.float32)
        train_ids = np.asarray(saved["training_source_ids"], dtype=np.int64)
        validation_ids = np.asarray(saved["validation_source_ids"], dtype=np.int64)
        stored_fingerprint = str(saved["content_fingerprint"])
    obtained_fingerprint = _array_sha256(
        train_ids,
        validation_ids,
        train_s,
        train_p,
        train_l,
        validation_s,
        validation_p,
        validation_l,
    )
    if obtained_fingerprint != stored_fingerprint or obtained_fingerprint != str(
        export_manifest["content_fingerprint"]
    ):
        raise RuntimeError("Exact-JANA ratio-bank content fingerprint mismatch.")
    bank_manifest = json.loads(
        (artifact_root / "manifests" / "slcp_banks.json").read_text()
    )
    ratio_api = _load_ratio_api()
    train_bank = ratio_api.RatioClassBank(
        simulator=train_s,
        posterior_reference=train_p,
        likelihood_reference=train_l,
        source_bank_fingerprint=str(bank_manifest["master"]["fingerprint"]),
        row_ids=train_ids,
        provenance={
            "schema": PAPER_RUNTIME_SCHEMA,
            "role": "exact_jana_ratio_training",
            "campaign_signature": signature,
            "budget": int(budget),
            "ml_seed": int(ml_seed),
            "source_bank": "master",
            "export_contract_sha256": export_manifest["contract_sha256"],
            "content_fingerprint": obtained_fingerprint,
        },
    )
    validation_bank = ratio_api.RatioClassBank(
        simulator=validation_s,
        posterior_reference=validation_p,
        likelihood_reference=validation_l,
        source_bank_fingerprint=str(
            bank_manifest["jana_validation"]["fingerprint"]
        ),
        row_ids=validation_ids,
        provenance={
            "schema": PAPER_RUNTIME_SCHEMA,
            "role": "exact_jana_ratio_validation",
            "campaign_signature": signature,
            "budget": int(budget),
            "ml_seed": int(ml_seed),
            "source_bank": "jana_validation",
            "export_contract_sha256": export_manifest["contract_sha256"],
            "content_fingerprint": obtained_fingerprint,
        },
    )
    model_root = (
        artifact_root
        / "models"
        / "jana_paper_correction"
        / signature
        / f"budget_{int(budget)}"
        / f"seed_{int(ml_seed)}"
    )
    classifiers = ratio_api.train_ratio_ensembles(
        train_bank,
        validation_bank,
        model_root,
        campaign["ratio_training"],
        seed=int(ml_seed) + 2_100_000,
        device=device,
        study_metadata={
            "stage": "exact_jana_correction",
            "campaign_signature": signature,
            "budget": int(budget),
            "ml_seed": int(ml_seed),
            "proposal": "nominal_exact_jana_no_broadening_no_prior_defense",
            "export_contract_sha256": export_manifest["contract_sha256"],
        },
        load_if_available=load_if_available,
    )
    manifest = {
        "schema": PAPER_RUNTIME_SCHEMA,
        "campaign_signature": signature,
        "budget": int(budget),
        "ml_seed": int(ml_seed),
        "proposal": "nominal_exact_jana_no_broadening_no_prior_defense",
        "training_rows": int(len(train_ids)),
        "validation_rows": int(len(validation_ids)),
        "ratio_bank": str(arrays_path),
        "ratio_bank_fingerprint": obtained_fingerprint,
        "ratio_members": ratio_api.ratio_ensemble_summary(classifiers),
        "implementation": _implementation_manifest(),
        "created_utc": _utc_now(),
    }
    _write_json(model_root / "manifest.json", manifest)
    return {"classifiers": classifiers, "manifest": manifest}


def _evaluate_exact_jana_corrected_audit(
    *,
    audit_path: str | Path,
    normalization_path: str | Path,
    campaign: Mapping[str, Any],
    classifiers: Mapping[str, Sequence[Mapping[str, Any]]],
    factorization: str,
    ml_seed: int,
) -> dict[str, Any]:
    """Apply a modern ratio ensemble to exact-JANA saved audit proposals."""

    ratio_api = _load_ratio_api()
    with np.load(audit_path, allow_pickle=False) as saved:
        reference_theta = np.asarray(saved["audit_reference_theta"], dtype=np.float32)
        reference_x = np.asarray(saved["audit_reference_x"], dtype=np.float32)
        marginal_theta = np.asarray(saved["audit_marginal_theta"], dtype=np.float32)
        marginal_x = np.asarray(saved["audit_marginal_x"], dtype=np.float32)
        posterior_theta = np.asarray(
            saved["jana_posterior_joint_theta"], dtype=np.float32
        )
        predictive_x = np.asarray(saved["jana_predictive_x"], dtype=np.float32)
        jana_audit_log_likelihood = np.asarray(
            saved["jana_log_likelihood_physical"], dtype=np.float64
        )
        exact_audit_log_likelihood = np.asarray(
            saved["exact_log_likelihood_physical"], dtype=np.float64
        )
        audit_bank_fingerprint = str(saved["audit_bank_fingerprint"])
    n = len(reference_theta)
    if not (
        len(reference_x)
        == len(marginal_theta)
        == len(marginal_x)
        == len(posterior_theta)
        == len(predictive_x)
        == n
    ):
        raise RuntimeError("Exact-JANA audit arrays have inconsistent row counts.")
    reference_joint = np.column_stack([reference_theta, reference_x])
    p_points = np.column_stack([posterior_theta, marginal_x]).astype(np.float32)
    l_points = np.column_stack([marginal_theta, predictive_x]).astype(np.float32)
    log_r_p = ratio_api.predict_posterior_log_ratio(
        classifiers, p_points, factorization=factorization
    )
    log_r_l = ratio_api.predict_likelihood_log_ratio(
        classifiers, l_points, factorization=factorization
    )
    audit_log_r_l = ratio_api.predict_likelihood_log_ratio(
        classifiers, reference_joint.astype(np.float32), factorization=factorization
    )
    likelihood_error = _finite_error_summary(
        jana_audit_log_likelihood + audit_log_r_l,
        exact_audit_log_likelihood,
        require_all_exact_finite_rows=True,
    )
    posterior_support = slcp_prior_support(posterior_theta)
    weights_p = normalized_log_weights(
        np.where(posterior_support, log_r_p, -np.inf)
    )
    weights_l = normalized_log_weights(log_r_l)
    corrected_p = systematic_resample(
        p_points, weights_p, n, int(ml_seed) + 70_001
    )
    corrected_l = systematic_resample(
        l_points, weights_l, n, int(ml_seed) + 70_002
    )
    max_samples = int(campaign["metric_max_samples"])
    posterior_metrics = distribution_metrics(
        reference_joint,
        corrected_p,
        seed=int(ml_seed) + 70_005,
        max_samples=max_samples,
    )
    predictive_x_metrics = distribution_metrics(
        reference_x,
        corrected_l[:, 5:],
        seed=int(ml_seed) + 70_006,
        max_samples=max_samples,
    )
    predictive_joint_metrics = distribution_metrics(
        reference_joint,
        corrected_l,
        seed=int(ml_seed) + 70_007,
        max_samples=max_samples,
    )
    with np.load(normalization_path, allow_pickle=False) as saved:
        normalization_points = np.asarray(
            saved["normalization_points_raw_theta_x"], dtype=np.float32
        )
        group_ids = np.asarray(saved["normalization_group_ids"], dtype=np.int64)
    normalization_log_ratio = ratio_api.predict_likelihood_log_ratio(
        classifiers, normalization_points, factorization=factorization
    )
    log_z_values = []
    for group_id in np.unique(group_ids):
        values = normalization_log_ratio[group_ids == group_id]
        log_z_values.append(float(logsumexp(values) - math.log(len(values))))
    log_z = np.asarray(log_z_values, dtype=np.float64)
    return {
        "posterior_joint_C2ST": posterior_metrics["C2ST"],
        "posterior_joint_MMD": posterior_metrics["MMD"],
        "predictive_x_C2ST": predictive_x_metrics["C2ST"],
        "predictive_x_MMD": predictive_x_metrics["MMD"],
        "predictive_joint_C2ST": predictive_joint_metrics["C2ST"],
        "predictive_joint_MMD": predictive_joint_metrics["MMD"],
        "posterior_joint_ESS_fraction": weight_diagnostics(weights_p)[
            "ess_fraction"
        ],
        "likelihood_joint_ESS_fraction": weight_diagnostics(weights_l)[
            "ess_fraction"
        ],
        "likelihood_log_Z_rms": float(np.sqrt(np.mean(np.square(log_z)))),
        "likelihood_log_Z_mean": float(np.mean(log_z)),
        "likelihood_log_Z_max_abs": float(np.max(np.abs(log_z))),
        "exact_likelihood_log_error": likelihood_error["rmse"],
        "exact_likelihood_centered_log_error": likelihood_error["centered_rmse"],
        "exact_likelihood_rows": likelihood_error["rows"],
        "likelihood_audit_rows": int(n),
        "likelihood_audit_fingerprint": _array_sha256(
            reference_theta, reference_x
        ),
        "audit_bank_fingerprint": audit_bank_fingerprint,
    }


def evaluate_exact_jana_correction(
    *,
    artifact_root: str | Path,
    campaign: Mapping[str, Any],
    budget: int,
    ml_seed: int,
    classifiers: Mapping[str, Sequence[Mapping[str, Any]]],
    factorization: str,
    exact_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate one ratio factorization over saved exact-JANA proposals."""

    factorization = str(factorization).lower()
    if factorization not in {"multiclass", "binary"}:
        raise ValueError("factorization must be 'multiclass' or 'binary'.")
    artifact_root = Path(artifact_root).expanduser().resolve()
    signature = _config_module().campaign_signature(campaign)
    ratio_api = _load_ratio_api()
    subset = exact_rows[
        (exact_rows["method"].astype(str) == "jana_paper")
        & (exact_rows["budget"].astype(int) == int(budget))
        & (exact_rows["ml_seed"].astype(int) == int(ml_seed))
    ].copy()
    if set(subset["observation"].astype(int)) != set(
        int(value) for value in campaign["observations"]
    ):
        raise RuntimeError("Exact-JANA baseline rows are incomplete for correction.")
    audit_paths = set(subset["likelihood_audit_arrays"].astype(str))
    normalization_paths = set(subset["likelihood_normalization_arrays"].astype(str))
    if len(audit_paths) != 1 or len(normalization_paths) != 1:
        raise RuntimeError("Exact-JANA run has inconsistent audit artifact paths.")
    audit_path = next(iter(audit_paths))
    audit_metrics = _evaluate_exact_jana_corrected_audit(
        audit_path=audit_path,
        normalization_path=next(iter(normalization_paths)),
        campaign=campaign,
        classifiers=classifiers,
        factorization=factorization,
        ml_seed=int(ml_seed),
    )
    with np.load(audit_path, allow_pickle=False) as saved:
        cycle_observation_ids = np.asarray(
            saved["bayes_cycle_observation_ids"], dtype=np.int64
        )
        cycle_theta = np.asarray(
            saved["bayes_cycle_theta_grid"], dtype=np.float32
        )
        cycle_log_q_phi = np.asarray(
            saved["bayes_cycle_log_q_phi_theta_given_x"], dtype=np.float64
        )
        cycle_log_q_eta = np.asarray(
            saved["bayes_cycle_log_q_eta_x_given_theta"], dtype=np.float64
        )
        cycle_log_prior = np.asarray(
            saved["bayes_cycle_log_prior_theta"], dtype=np.float64
        )
        cycle_theta_fingerprint = str(
            saved["bayes_cycle_theta_grid_fingerprint"]
        )
    if set(cycle_observation_ids.tolist()) != {
        int(value) for value in campaign["observations"]
    }:
        raise RuntimeError("Exact-JANA Bayes-cycle observations are incomplete.")
    expected_cycle_shape = (len(cycle_observation_ids), len(cycle_theta))
    if any(
        values.shape != expected_cycle_shape
        for values in (cycle_log_q_phi, cycle_log_q_eta, cycle_log_prior)
    ):
        raise RuntimeError("Exact-JANA Bayes-cycle arrays have inconsistent shapes.")
    if _array_sha256(cycle_theta) != cycle_theta_fingerprint:
        raise RuntimeError("Exact-JANA Bayes-cycle theta fingerprint mismatch.")
    ratio_families = (
        [ratio_api.FAMILY_MULTICLASS]
        if factorization == "multiclass"
        else [ratio_api.FAMILY_POSTERIOR_BINARY, ratio_api.FAMILY_LIKELIHOOD_BINARY]
    )
    selected_ce = [
        float(pack["history"]["selected_validation_ce"])
        for family in ratio_families
        for pack in classifiers[family]
    ]
    sample_root = (
        artifact_root
        / "results"
        / "jana_paper_correction"
        / signature
        / f"budget_{int(budget)}"
        / f"seed_{int(ml_seed)}"
        / factorization
        / "samples"
    )
    sample_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for record in subset.sort_values("observation").itertuples(index=False):
        observation_number = int(record.observation)
        cycle_index = int(
            np.flatnonzero(cycle_observation_ids == observation_number)[0]
        )
        route_path = Path(record.route_arrays)
        with np.load(route_path, allow_pickle=False) as saved:
            reference = np.asarray(saved["reference_theta"], dtype=np.float32)
            observation = np.asarray(saved["observation_x"], dtype=np.float32).reshape(1, -1)
            candidate_theta = np.asarray(
                saved["likelihood_proposal_theta"], dtype=np.float32
            )
            log_q_phi = np.asarray(
                saved["likelihood_proposal_log_density_theta"], dtype=np.float64
            )
            log_q_eta = np.asarray(
                saved["jana_log_likelihood_physical"], dtype=np.float64
            )
            log_prior = np.asarray(
                saved["prior_log_density_theta"], dtype=np.float64
            )
            exact_log_likelihood = np.asarray(
                saved["exact_log_likelihood_physical"], dtype=np.float64
            )
        repeated_observation = np.repeat(observation, len(candidate_theta), axis=0)
        points = np.column_stack([candidate_theta, repeated_observation]).astype(
            np.float32
        )
        log_r_p, member_log_r_p = ratio_api.predict_posterior_log_ratio(
            classifiers,
            points,
            factorization=factorization,
            return_members=True,
        )
        log_r_l, member_log_r_l = ratio_api.predict_likelihood_log_ratio(
            classifiers,
            points,
            factorization=factorization,
            return_members=True,
        )
        cycle_observation = np.repeat(observation, len(cycle_theta), axis=0)
        cycle_points = np.column_stack(
            [cycle_theta, cycle_observation]
        ).astype(np.float32)
        cycle_log_r_p = ratio_api.predict_posterior_log_ratio(
            classifiers, cycle_points, factorization=factorization
        )
        cycle_log_r_l = ratio_api.predict_likelihood_log_ratio(
            classifiers, cycle_points, factorization=factorization
        )
        cycle_audit_valid = (
            np.isfinite(cycle_log_q_phi[cycle_index])
            & np.isfinite(cycle_log_r_p)
            & np.isfinite(cycle_log_prior[cycle_index])
            & np.isfinite(cycle_log_q_eta[cycle_index])
            & np.isfinite(cycle_log_r_l)
        )
        cycle_audit = _cycle_diagnostics(
            cycle_log_q_phi[cycle_index, cycle_audit_valid]
            + cycle_log_r_p[cycle_audit_valid]
            - cycle_log_prior[cycle_index, cycle_audit_valid],
            cycle_log_q_eta[cycle_index, cycle_audit_valid]
            + cycle_log_r_l[cycle_audit_valid],
        )
        posterior_weights = normalized_log_weights(
            np.where(np.isfinite(log_prior), log_r_p, -np.inf)
        )
        likelihood_weights = normalized_log_weights(
            log_prior + log_q_eta + log_r_l - log_q_phi
        )
        n_posterior = int(campaign["posterior_samples"])
        posterior_theta = systematic_resample(
            candidate_theta,
            posterior_weights,
            n_posterior,
            int(ml_seed) + 1_600_000 + observation_number,
        )
        likelihood_theta = systematic_resample(
            candidate_theta,
            likelihood_weights,
            n_posterior,
            int(ml_seed) + 1_700_000 + observation_number,
        )
        max_samples = int(campaign["metric_max_samples"])
        posterior_metrics = distribution_metrics(
            reference,
            posterior_theta,
            seed=int(ml_seed) + 40_000 + observation_number,
            max_samples=max_samples,
        )
        likelihood_metrics = distribution_metrics(
            reference,
            likelihood_theta,
            seed=int(ml_seed) + 50_000 + observation_number,
            max_samples=max_samples,
        )
        route_metrics = distribution_metrics(
            posterior_theta,
            likelihood_theta,
            seed=int(ml_seed) + 60_000 + observation_number,
            max_samples=max_samples,
        )
        posterior_diagnostics = weight_diagnostics(posterior_weights)
        likelihood_diagnostics = weight_diagnostics(likelihood_weights)
        corrected_log_likelihood = log_q_eta + log_r_l
        exact_error = _finite_error_summary(
            corrected_log_likelihood, exact_log_likelihood
        )
        cycle_valid = (
            np.isfinite(log_q_phi)
            & np.isfinite(log_r_p)
            & np.isfinite(log_prior)
            & np.isfinite(corrected_log_likelihood)
        )
        cycle = _cycle_diagnostics(
            log_q_phi[cycle_valid]
            + log_r_p[cycle_valid]
            - log_prior[cycle_valid],
            corrected_log_likelihood[cycle_valid],
        )
        np.savez_compressed(
            sample_root / f"observation_{observation_number:02d}.npz",
            reference_theta=reference,
            candidate_theta=candidate_theta,
            posterior_theta=posterior_theta,
            likelihood_theta=likelihood_theta,
            posterior_weights=posterior_weights,
            likelihood_weights=likelihood_weights,
            posterior_log_ratio=log_r_p,
            likelihood_log_ratio=log_r_l,
            exact_log_likelihood=exact_log_likelihood,
            corrected_log_likelihood=corrected_log_likelihood,
        )
        rows.append(
            {
                "schema": PAPER_RUNTIME_SCHEMA,
                "campaign_signature": signature,
                "method": f"jana_paper_corrected_{factorization}",
                "factorization": factorization,
                "budget": int(budget),
                "simulator_calls": int(budget) + 304,
                "training_rows": int(budget),
                "validation_rows": 300,
                "ml_seed": int(ml_seed),
                "observation": observation_number,
                "posterior_C2ST": posterior_metrics["C2ST"],
                "posterior_MMD": posterior_metrics["MMD"],
                "likelihood_posterior_C2ST": likelihood_metrics["C2ST"],
                "likelihood_posterior_MMD": likelihood_metrics["MMD"],
                "posterior_likelihood_route_C2ST": route_metrics["C2ST"],
                "posterior_likelihood_route_MMD": route_metrics["MMD"],
                "posterior_ESS_fraction": posterior_diagnostics["ess_fraction"],
                "posterior_max_weight": posterior_diagnostics["max_weight"],
                "likelihood_posterior_ESS_fraction": likelihood_diagnostics[
                    "ess_fraction"
                ],
                "likelihood_posterior_max_weight": likelihood_diagnostics[
                    "max_weight"
                ],
                "bayes_cycle_pearson": cycle_audit["pearson"],
                "bayes_cycle_slope": cycle_audit["slope"],
                "bayes_cycle_residual_rms": cycle_audit["rms"],
                "bayes_cycle_rows": int(cycle_audit_valid.sum()),
                "bayes_cycle_theta_fingerprint": cycle_theta_fingerprint,
                "likelihood_log_Z_rms": audit_metrics["likelihood_log_Z_rms"],
                "likelihood_log_Z_mean": audit_metrics["likelihood_log_Z_mean"],
                "likelihood_log_Z_max_abs": audit_metrics[
                    "likelihood_log_Z_max_abs"
                ],
                "exact_likelihood_log_error": audit_metrics[
                    "exact_likelihood_log_error"
                ],
                "exact_likelihood_centered_log_error": audit_metrics[
                    "exact_likelihood_centered_log_error"
                ],
                "exact_likelihood_rows": audit_metrics["exact_likelihood_rows"],
                "likelihood_audit_rows": audit_metrics["likelihood_audit_rows"],
                "likelihood_audit_fingerprint": audit_metrics[
                    "likelihood_audit_fingerprint"
                ],
                "audit_bank_fingerprint": audit_metrics[
                    "audit_bank_fingerprint"
                ],
                "deployed_proposal_exact_likelihood_log_error": exact_error[
                    "rmse"
                ],
                "deployed_proposal_exact_likelihood_centered_log_error": exact_error[
                    "centered_rmse"
                ],
                "deployed_proposal_exact_likelihood_rows": exact_error["rows"],
                "deployed_proposal_bayes_cycle_pearson": cycle["pearson"],
                "deployed_proposal_bayes_cycle_slope": cycle["slope"],
                "deployed_proposal_bayes_cycle_residual_rms": cycle["rms"],
                "deployed_proposal_bayes_cycle_rows": int(cycle_valid.sum()),
                "proposal_base_scale": 1.0,
                "proposal_prior_fraction": 0.0,
                "posterior_ratio_member_log_std": float(
                    np.mean(np.std(member_log_r_p, axis=0))
                ),
                "likelihood_ratio_member_log_std": float(
                    np.mean(np.std(member_log_r_l, axis=0))
                ),
                "ratio_selected_validation_CE_mean": float(np.mean(selected_ce)),
                "ratio_selected_validation_CE_std": float(np.std(selected_ce)),
                **{
                    key: value
                    for key, value in audit_metrics.items()
                    if not key.startswith("likelihood_log_Z_")
                },
            }
        )
    return pd.DataFrame(rows)


def run_exact_jana_corrections(
    artifact_root: str | Path,
    campaign: Mapping[str, Any],
    *,
    factorizations: Sequence[str],
    budgets_to_run: Sequence[int],
    ml_seeds_to_run: Sequence[int],
    load_if_available: bool,
) -> pd.DataFrame:
    """Export nominal JANA banks, train ratios, and merge corrected shards."""

    try:
        from .utils_jana import (
            export_exact_jana_ratio_banks,
            run_exact_jana_campaign,
        )
    except ImportError:
        from utils_jana import export_exact_jana_ratio_banks, run_exact_jana_campaign
    artifact_root = Path(artifact_root).expanduser().resolve()
    signature = _config_module().campaign_signature(campaign)
    exact_rows = run_exact_jana_campaign(
        artifact_root,
        campaign,
        load_if_available=load_if_available,
        budgets_to_run=budgets_to_run,
        ml_seeds_to_run=ml_seeds_to_run,
    )
    exports = export_exact_jana_ratio_banks(
        artifact_root,
        campaign,
        load_if_available=load_if_available,
        budgets_to_run=budgets_to_run,
        ml_seeds_to_run=ml_seeds_to_run,
    )
    export_by_key = {
        (int(item["budget"]), int(item["seed"])): item
        for item in exports["runs"]
    }
    expected_methods = {f"jana_paper_corrected_{value}" for value in factorizations}
    expected_observations = {
        int(value) for value in campaign["observations"]
    }
    for budget in budgets_to_run:
        for ml_seed in ml_seeds_to_run:
            per_run_path = (
                artifact_root
                / "results"
                / "jana_paper_correction"
                / signature
                / f"budget_{int(budget)}"
                / f"seed_{int(ml_seed)}"
                / "metrics.csv"
            )
            existing = (
                pd.read_csv(per_run_path)
                if load_if_available and per_run_path.exists()
                else pd.DataFrame()
            )
            present = set()
            if not existing.empty and {
                "method",
                "observation",
            }.issubset(existing.columns):
                for method_name, group in existing.groupby("method"):
                    if set(group["observation"].astype(int)) == expected_observations:
                        present.add(str(method_name))
            missing = tuple(
                value
                for value in factorizations
                if f"jana_paper_corrected_{value}" not in present
            )
            if not missing:
                continue
            export = export_by_key.get((int(budget), int(ml_seed)))
            if export is None:
                raise RuntimeError("Missing exact-JANA ratio-bank export shard.")
            ratio_models = train_exact_jana_ratio_models(
                artifact_root,
                campaign,
                budget=int(budget),
                ml_seed=int(ml_seed),
                export_manifest=export,
                load_if_available=load_if_available,
            )
            new_frames = [
                evaluate_exact_jana_correction(
                    artifact_root=artifact_root,
                    campaign=campaign,
                    budget=int(budget),
                    ml_seed=int(ml_seed),
                    classifiers=ratio_models["classifiers"],
                    factorization=factorization,
                    exact_rows=exact_rows,
                )
                for factorization in missing
            ]
            combined = pd.concat([existing, *new_frames], ignore_index=True)
            combined = combined.drop_duplicates(
                ["method", "budget", "ml_seed", "observation"], keep="last"
            )
            _write_csv(per_run_path, combined)
    shard_files = sorted(
        (artifact_root / "results" / "jana_paper_correction" / signature).glob(
            "budget_*/seed_*/metrics.csv"
        )
    )
    if not shard_files:
        raise RuntimeError("No exact-JANA correction metric shards were produced.")
    merged = pd.concat(
        [pd.read_csv(path) for path in shard_files], ignore_index=True
    ).drop_duplicates(
        ["method", "budget", "ml_seed", "observation"], keep="last"
    )
    result_path = (
        artifact_root
        / "results"
        / "hybrid"
        / f"jana_paper_corrected_{signature}.csv"
    )
    _write_csv(result_path, merged)
    requested_keys = {
        (method, int(budget), int(seed), int(observation))
        for method in expected_methods
        for budget in budgets_to_run
        for seed in ml_seeds_to_run
        for observation in expected_observations
    }
    obtained_keys = {
        (str(row.method), int(row.budget), int(row.ml_seed), int(row.observation))
        for row in merged.itertuples()
    }
    missing_keys = requested_keys - obtained_keys
    if missing_keys:
        raise RuntimeError(
            "Merged exact-JANA correction table is incomplete; first missing "
            f"keys: {sorted(missing_keys)[:10]}"
        )
    return merged


def run_hybrid_campaign(
    artifact_root: str | Path,
    campaign: Mapping[str, Any],
    *,
    factorizations: Sequence[str] = ("multiclass", "binary"),
    run_proposal_ablation: bool = True,
    budgets_to_run: Sequence[int] | None = None,
    ml_seeds_to_run: Sequence[int] | None = None,
    load_if_available: bool = True,
) -> dict[str, Any]:
    """Select the proposal, train both ratio constructions, and evaluate."""

    paper_config = _config_module()
    paper_config.validate_campaign_config(campaign)
    artifact_root = Path(artifact_root).expanduser().resolve()
    signature = paper_config.campaign_signature(campaign)
    normalized_factorizations = tuple(
        dict.fromkeys(str(value).lower() for value in factorizations)
    )
    invalid = set(normalized_factorizations) - {"multiclass", "binary"}
    if invalid:
        raise ValueError(f"Unknown hybrid factorizations: {sorted(invalid)}")
    run_budgets = _execution_subset(
        campaign["budgets"], budgets_to_run, "budget"
    )
    run_seeds = _execution_subset(
        campaign["ml_seeds"], ml_seeds_to_run, "ML-seed"
    )
    exact_metrics = run_exact_jana_corrections(
        artifact_root,
        campaign,
        factorizations=normalized_factorizations,
        budgets_to_run=run_budgets,
        ml_seeds_to_run=run_seeds,
        load_if_available=load_if_available,
    )
    proposal_result = select_hybrid_proposal(
        artifact_root,
        campaign,
        include_ablation=run_proposal_ablation,
        load_if_available=load_if_available,
    )
    selection = proposal_result["selection"]
    base_scale = float(selection["proposal_base_scale"])
    prior_fraction = float(selection["proposal_prior_fraction"])
    result_path = (
        artifact_root / "results" / "hybrid" / f"hybrid_{signature}.csv"
    )
    expected = {
        f"separate_flows_corrected_{value}"
        for value in normalized_factorizations
    }
    for budget in run_budgets:
        for ml_seed in run_seeds:
            per_run_path = (
                artifact_root
                / "results"
                / "hybrid"
                / signature
                / f"budget_{int(budget)}"
                / f"seed_{int(ml_seed)}"
                / "metrics.csv"
            )
            existing = (
                pd.read_csv(per_run_path)
                if load_if_available and per_run_path.exists()
                else pd.DataFrame()
            )
            present = set()
            if not existing.empty and {
                "method",
                "observation",
            }.issubset(existing.columns):
                expected_observations = {
                    int(value) for value in campaign["observations"]
                }
                for method_name, group in existing.groupby("method"):
                    if set(group["observation"].astype(int)) == expected_observations:
                        present.add(str(method_name))
            missing_factorizations = tuple(
                value
                for value in normalized_factorizations
                if f"separate_flows_corrected_{value}" not in present
            )
            if missing_factorizations:
                trained = train_matched_flow_pair(
                    artifact_root,
                    campaign,
                    budget=int(budget),
                    ml_seed=int(ml_seed),
                    load_if_available=load_if_available,
                )
                ratio_models = train_hybrid_ratio_models(
                    artifact_root,
                    campaign,
                    budget=int(budget),
                    ml_seed=int(ml_seed),
                    q_phi=trained["q_phi"],
                    q_eta=trained["q_eta"],
                    base_scale=base_scale,
                    prior_fraction=prior_fraction,
                    load_if_available=load_if_available,
                    device=trained["device"],
                )
                new_frames = []
                for factorization in missing_factorizations:
                    new_frames.append(
                        evaluate_hybrid(
                            artifact_root=artifact_root,
                            campaign=campaign,
                            budget=int(budget),
                            ml_seed=int(ml_seed),
                            q_phi=trained["q_phi"],
                            q_eta=trained["q_eta"],
                            classifiers=ratio_models["classifiers"],
                            factorization=factorization,
                            base_scale=base_scale,
                            prior_fraction=prior_fraction,
                        )
                    )
                combined = pd.concat([existing, *new_frames], ignore_index=True)
                combined = combined.drop_duplicates(
                    ["method", "budget", "ml_seed", "observation"], keep="last"
                )
                per_run_path.parent.mkdir(parents=True, exist_ok=True)
                _write_csv(per_run_path, combined)
    run_files = sorted(
        (artifact_root / "results" / "hybrid" / signature).glob(
            "budget_*/seed_*/metrics.csv"
        )
    )
    if not run_files:
        raise RuntimeError("No separate-flow correction metric shards were produced.")
    metrics = pd.concat(
        [pd.read_csv(path) for path in run_files], ignore_index=True
    ).drop_duplicates(
        ["method", "budget", "ml_seed", "observation"], keep="last"
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(result_path, metrics)
    requested_keys = {
        (method, int(budget), int(seed), int(observation))
        for method in expected
        for budget in run_budgets
        for seed in run_seeds
        for observation in campaign["observations"]
    }
    obtained_keys = {
        (str(row.method), int(row.budget), int(row.ml_seed), int(row.observation))
        for row in metrics.itertuples()
    }
    missing_keys = requested_keys - obtained_keys
    if missing_keys:
        raise RuntimeError(
            "Merged separate-flow correction table is incomplete; first missing "
            f"keys: {sorted(missing_keys)[:10]}"
        )
    combined_metrics = pd.concat([exact_metrics, metrics], ignore_index=True, sort=False)
    return {
        "selection": selection,
        "proposal_scan": proposal_result["scan"],
        "jana_paper_corrections": exact_metrics,
        "separate_flow_corrections": metrics,
        "metrics": combined_metrics,
    }


def _load_completed_metric_tables(
    artifact_root: Path, campaign: Mapping[str, Any]
) -> dict[str, pd.DataFrame]:
    signature = _config_module().campaign_signature(campaign)
    paths = {
        "jana_paper": artifact_root / "results" / "jana" / f"jana_paper_{signature}.csv",
        "jana_paper_corrections": artifact_root / "results" / "hybrid" / f"jana_paper_corrected_{signature}.csv",
        "separate_flows": artifact_root / "results" / "jana" / f"separate_flows_{signature}.csv",
        "separate_flow_corrections": artifact_root / "results" / "hybrid" / f"hybrid_{signature}.csv",
    }
    tables: dict[str, pd.DataFrame] = {}
    for name, path in paths.items():
        if path.exists():
            table = pd.read_csv(path)
            if "campaign_signature" not in table or set(
                table["campaign_signature"].astype(str)
            ) != {signature}:
                raise RuntimeError(f"Metric table {path} has the wrong campaign.")
            tables[name] = table
    return tables


def _metric_quality(values: np.ndarray, metric: str) -> tuple[np.ndarray, str, float | None]:
    """Orient a diagnostic so larger transformed values always mean better."""

    values = np.asarray(values, dtype=np.float64)
    if str(metric).endswith("C2ST"):
        return -np.abs(values - 0.5), "closest_to_0.5", 0.5
    if metric in {"bayes_cycle_pearson", "bayes_cycle_slope"}:
        return -np.abs(values - 1.0), "closest_to_1", 1.0
    if metric == "likelihood_log_Z_mean":
        return -np.abs(values), "closest_to_0", 0.0
    if metric.endswith("ESS_fraction"):
        return values, "higher", None
    return -values, "lower", 0.0


def build_paper_comparison(
    artifact_root: str | Path,
    campaign: Mapping[str, Any],
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Validate completed stages and emit long, summary, paired, and figures."""

    paper_config = _config_module()
    paper_config.validate_campaign_config(campaign)
    artifact_root = Path(artifact_root).expanduser().resolve()
    signature = paper_config.campaign_signature(campaign)
    tables = _load_completed_metric_tables(artifact_root, campaign)
    required_tables = {
        "jana_paper",
        "jana_paper_corrections",
        "separate_flows",
        "separate_flow_corrections",
    }
    missing_tables = required_tables - set(tables)
    if require_complete and missing_tables:
        raise FileNotFoundError(
            "Missing completed paper stages: " + ", ".join(sorted(missing_tables))
        )
    if not tables:
        raise FileNotFoundError("No completed metric tables were found.")
    long = pd.concat(tables.values(), ignore_index=True, sort=False)
    expected_methods = set(campaign["methods"])
    present_methods = set(long["method"].astype(str))
    if require_complete and present_methods != expected_methods:
        raise RuntimeError(
            "Method set mismatch; missing="
            f"{sorted(expected_methods - present_methods)}, unexpected="
            f"{sorted(present_methods - expected_methods)}"
        )
    key_columns = ["method", "budget", "ml_seed", "observation"]
    if long.duplicated(key_columns).any():
        duplicates = long.loc[long.duplicated(key_columns, keep=False), key_columns]
        raise RuntimeError(f"Duplicate metric rows detected:\n{duplicates}")
    expected_keys = {
        (method, int(budget), int(seed), int(observation))
        for method in expected_methods
        for budget in campaign["budgets"]
        for seed in campaign["ml_seeds"]
        for observation in campaign["observations"]
    }
    actual_keys = {
        (str(row.method), int(row.budget), int(row.ml_seed), int(row.observation))
        for row in long.itertuples()
    }
    missing_keys = expected_keys - actual_keys
    if require_complete and missing_keys:
        preview = sorted(missing_keys)[:10]
        raise RuntimeError(
            f"Metric campaign is incomplete ({len(missing_keys)} missing rows); "
            f"first missing keys: {preview}"
        )
    required_metadata = {
        "schema",
        "campaign_signature",
        "method",
        "factorization",
        "budget",
        "simulator_calls",
        "training_rows",
        "validation_rows",
        "ml_seed",
        "observation",
        "audit_bank_fingerprint",
        "likelihood_audit_fingerprint",
        "likelihood_audit_rows",
        "bayes_cycle_rows",
        "bayes_cycle_theta_fingerprint",
    }
    missing_metadata = required_metadata - set(long.columns)
    missing_diagnostics = set(campaign["diagnostics"]) - set(long.columns)
    if require_complete and (missing_metadata or missing_diagnostics):
        raise RuntimeError(
            "Incomplete standardized result schema; missing metadata="
            f"{sorted(missing_metadata)}, diagnostics={sorted(missing_diagnostics)}"
        )
    if require_complete:
        metadata_columns = sorted(required_metadata)
        null_counts = {
            name: int(long[name].isna().sum()) for name in metadata_columns
        }
        null_counts = {name: count for name, count in null_counts.items() if count}
        if null_counts:
            raise RuntimeError(f"Null required metadata values: {null_counts}")
        if set(long["schema"].astype(str)) != {PAPER_RUNTIME_SCHEMA}:
            raise RuntimeError("Metric rows mix incompatible runtime schemas.")
        if "unknown" in set(long["audit_bank_fingerprint"].astype(str)):
            raise RuntimeError("Metric rows do not identify the independent audit bank.")
        if long["audit_bank_fingerprint"].astype(str).nunique() != 1:
            raise RuntimeError("Methods do not share one independent audit bank.")
        if long["likelihood_audit_fingerprint"].astype(str).nunique() != 1:
            raise RuntimeError(
                "Methods do not share one fixed likelihood-density audit table."
            )
        if long["likelihood_audit_rows"].astype(int).nunique() != 1:
            raise RuntimeError("Methods use different likelihood-audit row counts.")
        if long["exact_likelihood_rows"].astype(int).nunique() != 1:
            raise RuntimeError(
                "Methods did not score the same finite exact-likelihood audit rows."
            )
        if long["bayes_cycle_theta_fingerprint"].astype(str).nunique() != 1:
            raise RuntimeError("Methods do not share one fixed Bayes-cycle theta grid.")
        jana = long[long["method"].astype(str).str.startswith("jana_paper")]
        separate = long[
            long["method"].astype(str).str.startswith("separate_flows")
        ]
        if not np.all(
            jana["simulator_calls"].astype(int).to_numpy()
            == jana["budget"].astype(int).to_numpy() + 304
        ):
            raise RuntimeError("Exact-JANA rows violate N+304 simulation accounting.")
        if not np.all(
            separate["simulator_calls"].astype(int).to_numpy()
            == separate["budget"].astype(int).to_numpy()
        ):
            raise RuntimeError("Separate-flow rows violate N simulation accounting.")
    metric_columns = [
        name for name in campaign["diagnostics"] if name in long.columns
    ]
    if require_complete:
        nonfinite = {
            metric: int((~np.isfinite(pd.to_numeric(long[metric], errors="coerce"))).sum())
            for metric in metric_columns
        }
        nonfinite = {key: value for key, value in nonfinite.items() if value}
        if nonfinite:
            raise RuntimeError(f"Non-finite required diagnostic values: {nonfinite}")
    observation_summary = (
        long.groupby(["method", "budget", "observation"], as_index=False)[
            metric_columns
        ]
        .agg(["mean", "std", "sem"])
    )
    observation_summary.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in observation_summary.columns
    ]
    seed_summary = (
        long.groupby(["method", "budget", "ml_seed"], as_index=False)[
            metric_columns
        ]
        .mean()
    )
    overall_summary = (
        seed_summary.groupby(["method", "budget"], as_index=False)[metric_columns]
        .agg(["mean", "std", "sem"])
    )
    overall_summary.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in overall_summary.columns
    ]
    paired_rows: list[dict[str, Any]] = []
    contrasts = (
        ("jana_paper_corrected_multiclass", "jana_paper"),
        ("jana_paper_corrected_binary", "jana_paper"),
        (
            "jana_paper_corrected_binary",
            "jana_paper_corrected_multiclass",
        ),
        ("separate_flows_corrected_multiclass", "separate_flows"),
        ("separate_flows_corrected_binary", "separate_flows"),
        (
            "separate_flows_corrected_binary",
            "separate_flows_corrected_multiclass",
        ),
    )
    pair_index = ["budget", "ml_seed", "observation"]
    for treatment_name, control_name in contrasts:
        treatment = long[long["method"] == treatment_name].set_index(pair_index)
        control = long[long["method"] == control_name].set_index(pair_index)
        common = treatment.index.intersection(control.index)
        differences = treatment.loc[common, metric_columns] - control.loc[
            common, metric_columns
        ]
        differences = differences.reset_index()
        oriented = pd.DataFrame(index=common)
        metric_goals: dict[str, tuple[str, float | None]] = {}
        for metric in metric_columns:
            treatment_quality, goal, ideal = _metric_quality(
                treatment.loc[common, metric].to_numpy(), metric
            )
            control_quality, _, _ = _metric_quality(
                control.loc[common, metric].to_numpy(), metric
            )
            oriented[metric] = treatment_quality - control_quality
            metric_goals[metric] = (goal, ideal)
        oriented = oriented.reset_index()
        for (budget, observation), group in differences.groupby(
            ["budget", "observation"], sort=True
        ):
            oriented_group = oriented[
                (oriented["budget"] == budget)
                & (oriented["observation"] == observation)
            ]
            for metric in metric_columns:
                values = np.asarray(group[metric], dtype=np.float64)
                values = values[np.isfinite(values)]
                improvement = np.asarray(
                    oriented_group[metric], dtype=np.float64
                )
                improvement = improvement[np.isfinite(improvement)]
                goal, ideal = metric_goals[metric]
                paired_rows.append(
                    {
                        "contrast": f"{treatment_name}-{control_name}",
                        "summary_level": "observation",
                        "budget": int(budget),
                        "observation": int(observation),
                        "metric": metric,
                        "metric_goal": goal,
                        "ideal_value": ideal,
                        "n_seed_pairs": int(len(values)),
                        "mean_difference": float(np.mean(values)) if len(values) else float("nan"),
                        "sem_difference": float(np.std(values, ddof=1) / math.sqrt(len(values)))
                        if len(values) > 1
                        else float("nan"),
                        "mean_oriented_improvement": (
                            float(np.mean(improvement))
                            if len(improvement)
                            else float("nan")
                        ),
                        "sem_oriented_improvement": (
                            float(
                                np.std(improvement, ddof=1)
                                / math.sqrt(len(improvement))
                            )
                            if len(improvement) > 1
                            else float("nan")
                        ),
                    }
                )
        seed_differences = differences.groupby(
            ["budget", "ml_seed"], as_index=False
        )[metric_columns].mean()
        seed_oriented = oriented.groupby(
            ["budget", "ml_seed"], as_index=False
        )[metric_columns].mean()
        for budget, group in seed_differences.groupby("budget", sort=True):
            oriented_group = seed_oriented[seed_oriented["budget"] == budget]
            for metric in metric_columns:
                values = np.asarray(group[metric], dtype=np.float64)
                values = values[np.isfinite(values)]
                improvement = np.asarray(
                    oriented_group[metric], dtype=np.float64
                )
                improvement = improvement[np.isfinite(improvement)]
                goal, ideal = metric_goals[metric]
                paired_rows.append(
                    {
                        "contrast": f"{treatment_name}-{control_name}",
                        "summary_level": "observation_averaged_within_seed",
                        "budget": int(budget),
                        "observation": "all",
                        "metric": metric,
                        "metric_goal": goal,
                        "ideal_value": ideal,
                        "n_seed_pairs": int(len(values)),
                        "mean_difference": float(np.mean(values)) if len(values) else float("nan"),
                        "sem_difference": float(np.std(values, ddof=1) / math.sqrt(len(values)))
                        if len(values) > 1
                        else float("nan"),
                        "mean_oriented_improvement": (
                            float(np.mean(improvement))
                            if len(improvement)
                            else float("nan")
                        ),
                        "sem_oriented_improvement": (
                            float(
                                np.std(improvement, ddof=1)
                                / math.sqrt(len(improvement))
                            )
                            if len(improvement) > 1
                            else float("nan")
                        ),
                    }
                )
    paired = pd.DataFrame(paired_rows)
    output_root = artifact_root / "paper_outputs" / signature
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "metrics_long.csv", long)
    _write_csv(output_root / "metrics_by_observation.csv", observation_summary)
    _write_csv(output_root / "metrics_overall.csv", overall_summary)
    _write_csv(output_root / "paired_contrasts.csv", paired)
    try:
        from . import utils_plotting
    except ImportError:
        import utils_plotting
    figure_root = output_root / "figures"
    figure_paths: list[Path] = []
    figure_paths.extend(
        utils_plotting.plot_metric_summary(long, figure_root).values()
    )
    figure_paths.extend(utils_plotting.plot_ess_summary(long, figure_root).values())
    capacity_path, _ = _capacity_paths(artifact_root, campaign)
    if capacity_path.exists():
        selection = _load_selected_architectures(artifact_root, campaign)
        figure_paths.extend(
            utils_plotting.plot_capacity_scan(
                pd.read_csv(capacity_path), figure_root, selection=selection
            ).values()
        )
    manifest = {
        "schema": PAPER_RUNTIME_SCHEMA,
        "campaign_signature": signature,
        "complete": not missing_tables and not missing_keys,
        "present_methods": sorted(present_methods),
        "missing_tables": sorted(missing_tables),
        "missing_rows": len(missing_keys),
        "metric_columns": metric_columns,
        "paired_contrast_convention": (
            "raw_difference_is_treatment_minus_control; "
            "positive_oriented_improvement_always_means_closer_to_metric_goal"
        ),
        "figure_paths": [str(path) for path in figure_paths],
        "implementation": _implementation_manifest(),
        "created_utc": _utc_now(),
    }
    _write_json(output_root / "campaign_manifest.json", manifest)
    return {
        "metrics": long,
        "by_observation": observation_summary,
        "overall": overall_summary,
        "paired_contrasts": paired,
        "manifest": manifest,
    }
