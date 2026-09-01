"""Plain cross-entropy density-ratio ensembles for the SLCP paper study.

This module is deliberately self-contained.  It implements the classifier
part of the hybrid construction without importing any of the historical
workshop utilities.  The three equal-prior class distributions are

``S = p(z, x)``, ``P = g_phi(z | x) p(x)``, and
``L = p(z) g_eta(x | z)``.

A multiclass classifier estimates both ``S/P`` and ``S/L`` probability
quotients, while two independent binary classifiers provide the comparison
factorization.  Models are plain ReLU MLPs trained with cross entropy and Adam:
there is no dropout, weight decay, layer normalization, calibration loss, or
auxiliary normalization objective.

Inference is log-space first.  Every member contributes a float64
``log_softmax`` difference, and the ensemble log ratio is the logarithm of the
arithmetic mean, evaluated with ``logsumexp``.  Ratio-returning compatibility
wrappers exponentiate only the final result and fail loudly if float64 cannot
represent it.

Training-group sampling is also explicit: each member walks deterministic
shuffled permutations of the complete eligible partition and reshuffles only
after every group has been consumed.  A model is checkpoint-eligible only
after completing at least one full pass.

Checkpoint reuse is intentionally strict.  Every checkpoint is bound to the
exact training and validation arrays, their provenance, the common input
transform, the family configuration, member index, and member seed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import random
import sys
import tempfile
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # Package import from ``paper_summary``.
    from . import config as paper_config
except ImportError:  # Direct notebook import with paper_summary on sys.path.
    import config as paper_config  # type: ignore[no-redef]


RATIO_CHECKPOINT_SCHEMA = "paper_summary_plain_ce_ratio_v1"
FROZEN_LEARNING_RATE_LEVELS = (
    1.0e-4,
    1.0e-5,
    1.0e-6,
    1.0e-7,
    1.0e-8,
    1.0e-9,
)
FROZEN_LEARNING_RATE_DROP_FRACTIONS = (
    1 / 6,
    2 / 6,
    3 / 6,
    4 / 6,
    5 / 6,
)

CLASS_S = 0
CLASS_P = 1
CLASS_L = 2

FAMILY_MULTICLASS = "multiclass"
FAMILY_POSTERIOR_BINARY = "posterior_binary"
FAMILY_LIKELIHOOD_BINARY = "likelihood_binary"
RATIO_FAMILIES = (
    FAMILY_MULTICLASS,
    FAMILY_POSTERIOR_BINARY,
    FAMILY_LIKELIHOOD_BINARY,
)


def _as_2d_float32(values: Any, name: str) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")
    if len(array) == 0:
        raise ValueError(f"{name} must contain at least one row.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": _array_sha256(value),
        }
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} in ratio metadata.")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    )


def _array_sha256(values: np.ndarray, *, chunk_bytes: int = 8 << 20) -> str:
    """Hash an array's dtype, shape, and exact contiguous byte contents."""

    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("utf-8"))
    digest.update(_canonical_json(list(array.shape)).encode("utf-8"))
    raw = memoryview(array).cast("B")
    for start in range(0, raw.nbytes, int(chunk_bytes)):
        digest.update(raw[start : start + int(chunk_bytes)])
    return digest.hexdigest()


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_deterministic_runtime(
    device: str | torch.device,
) -> dict[str, Any]:
    """Enable the strict deterministic settings used by ratio classifiers.

    PyTorch determinism is hardware- and release-specific.  The returned
    record therefore captures both the enforced backend flags and the runtime
    identity needed to interpret the reproducibility guarantee.
    """

    device = torch.device(device)
    cuda_initialized_before = bool(
        torch.cuda.is_available()
        and hasattr(torch.cuda, "is_initialized")
        and torch.cuda.is_initialized()
    )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
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

    device_record: dict[str, Any] = {"device": str(device)}
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA ratio device was requested but is unavailable.")
        device_index = torch.cuda.current_device() if device.index is None else device.index
        properties = torch.cuda.get_device_properties(device_index)
        device_record.update(
            {
                "cuda_device_index": int(device_index),
                "cuda_device_name": str(properties.name),
                "cuda_compute_capability": [int(properties.major), int(properties.minor)],
            }
        )
    limitations = [
        "bitwise reproducibility is guaranteed only for the same PyTorch, CUDA, "
        "driver, device architecture, and thread configuration",
        "checkpoint loading on a different runtime preserves scientific metadata "
        "but is not claimed to reproduce training bitwise",
    ]
    if cuda_initialized_before:
        limitations.append(
            "CUDA was initialized before CUBLAS_WORKSPACE_CONFIG was enforced; "
            "strict PyTorch deterministic mode remains active and will raise if "
            "an operation cannot honor it"
        )
    cudnn_backend = getattr(torch.backends, "cudnn", None)
    cuda_backend = getattr(torch.backends, "cuda", None)
    cuda_matmul_backend = getattr(cuda_backend, "matmul", None)
    return {
        "deterministic_policy": "strict_torch_deterministic_algorithms",
        "torch_deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "torch_deterministic_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
            if hasattr(torch, "is_deterministic_algorithms_warn_only_enabled")
            else False
        ),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_initialized_before_configuration": cuda_initialized_before,
        "cudnn_deterministic": bool(
            getattr(cudnn_backend, "deterministic", False)
        ),
        "cudnn_benchmark": bool(getattr(cudnn_backend, "benchmark", False)),
        "cudnn_allow_tf32": bool(
            getattr(cudnn_backend, "allow_tf32", False)
        ),
        "cuda_matmul_allow_tf32": bool(
            getattr(cuda_matmul_backend, "allow_tf32", False)
        ),
        "float32_matmul_precision": (
            torch.get_float32_matmul_precision()
            if hasattr(torch, "get_float32_matmul_precision")
            else "unavailable"
        ),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "cudnn_version": (
            None if cudnn_backend is None else cudnn_backend.version()
        ),
        "numpy_version": str(np.__version__),
        "python_version": str(sys.version),
        "platform": str(platform.platform()),
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        **device_record,
        "reproducibility_limits": limitations,
    }


def _validate_recorded_deterministic_runtime(runtime: Mapping[str, Any]) -> None:
    required = {
        "deterministic_policy": "strict_torch_deterministic_algorithms",
        "torch_deterministic_algorithms_enabled": True,
        "torch_deterministic_warn_only": False,
        "cublas_workspace_config": ":4096:8",
        "cudnn_benchmark": False,
        "cudnn_allow_tf32": False,
        "cuda_matmul_allow_tf32": False,
        "float32_matmul_precision": "highest",
    }
    for key, expected in required.items():
        if runtime.get(key) != expected:
            raise RuntimeError(
                f"Recorded ratio runtime violates deterministic setting {key!r}."
            )
    if runtime.get("cudnn_version") is not None and runtime.get(
        "cudnn_deterministic"
    ) is not True:
        raise RuntimeError("Recorded cuDNN runtime was not deterministic.")
    for key in (
        "torch_version",
        "numpy_version",
        "python_version",
        "platform",
        "device",
        "reproducibility_limits",
    ):
        if key not in runtime:
            raise RuntimeError(f"Recorded ratio runtime is missing {key!r}.")


@dataclass(frozen=True)
class RatioClassBank:
    """Matched equal-prior rows for the simulator and two flow references."""

    simulator: np.ndarray
    posterior_reference: np.ndarray
    likelihood_reference: np.ndarray
    provenance: Mapping[str, Any]
    source_bank_fingerprint: str | None = None
    row_ids: np.ndarray | None = None
    _fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        arrays = {
            "simulator": _as_2d_float32(self.simulator, "simulator"),
            "posterior_reference": _as_2d_float32(
                self.posterior_reference, "posterior_reference"
            ),
            "likelihood_reference": _as_2d_float32(
                self.likelihood_reference, "likelihood_reference"
            ),
        }
        shapes = {array.shape for array in arrays.values()}
        if len(shapes) != 1:
            raise ValueError(
                "S, P, and L class arrays must have the same row and feature shape."
            )
        object.__setattr__(self, "simulator", arrays["simulator"])
        object.__setattr__(
            self, "posterior_reference", arrays["posterior_reference"]
        )
        object.__setattr__(
            self, "likelihood_reference", arrays["likelihood_reference"]
        )
        for array in arrays.values():
            array.setflags(write=False)
        provenance = json.loads(_canonical_json(dict(self.provenance)))
        # Validate now so a mutable or unserializable value cannot corrupt a
        # checkpoint contract after training has started.
        _canonical_json(provenance)
        object.__setattr__(self, "provenance", provenance)
        if (self.source_bank_fingerprint is None) != (self.row_ids is None):
            raise ValueError(
                "source_bank_fingerprint and row_ids must be provided together."
            )
        source_bank_fingerprint = self.source_bank_fingerprint
        row_ids = self.row_ids
        if source_bank_fingerprint is not None:
            source_bank_fingerprint = str(source_bank_fingerprint).strip()
            if not source_bank_fingerprint:
                raise ValueError("source_bank_fingerprint must be non-empty.")
            row_ids = np.ascontiguousarray(np.asarray(row_ids, dtype=np.int64))
            if row_ids.ndim != 1 or len(row_ids) != len(arrays["simulator"]):
                raise ValueError(
                    "row_ids must be one-dimensional with one ID per class group."
                )
            if len(np.unique(row_ids)) != len(row_ids):
                raise ValueError("row_ids must be unique within a RatioClassBank.")
            row_ids.setflags(write=False)
        object.__setattr__(
            self, "source_bank_fingerprint", source_bank_fingerprint
        )
        object.__setattr__(self, "row_ids", row_ids)
        payload = {
            "schema": RATIO_CHECKPOINT_SCHEMA,
            "rows": len(arrays["simulator"]),
            "input_dim": int(arrays["simulator"].shape[1]),
            "simulator_sha256": _array_sha256(arrays["simulator"]),
            "posterior_reference_sha256": _array_sha256(
                arrays["posterior_reference"]
            ),
            "likelihood_reference_sha256": _array_sha256(
                arrays["likelihood_reference"]
            ),
            "provenance": provenance,
            "source_bank_fingerprint": source_bank_fingerprint,
            "row_ids_sha256": (
                None if row_ids is None else _array_sha256(row_ids)
            ),
        }
        object.__setattr__(self, "_fingerprint", _payload_sha256(payload))

    def __len__(self) -> int:
        return len(self.simulator)

    @property
    def input_dim(self) -> int:
        return int(self.simulator.shape[1])

    def classes(self, family: str) -> tuple[np.ndarray, ...]:
        family = normalize_family(family)
        if family == FAMILY_MULTICLASS:
            return (
                self.simulator,
                self.posterior_reference,
                self.likelihood_reference,
            )
        if family == FAMILY_POSTERIOR_BINARY:
            return self.simulator, self.posterior_reference
        return self.simulator, self.likelihood_reference

    def fingerprint(self) -> str:
        return self._fingerprint


def ratio_bank_fingerprint(bank: RatioClassBank) -> str:
    return bank.fingerprint()


def validate_source_bank_separation(
    train_bank: RatioClassBank, validation_bank: RatioClassBank
) -> dict[str, Any]:
    """Prove shared-source disjointness or identify genuinely separate banks.

    Numeric row IDs are comparable only when both class banks name the same
    immutable source-bank fingerprint.  Different source fingerprints denote
    separate banks, so overlapping local IDs are deliberately permitted (the
    exact-JANA train and validation banks use this case).
    """

    for role, bank in (("training", train_bank), ("validation", validation_bank)):
        if bank.source_bank_fingerprint is None or bank.row_ids is None:
            raise ValueError(
                f"{role} RatioClassBank lacks source_bank_fingerprint/row_ids; "
                "held-out disjointness cannot be proven from provenance text."
            )
    train_source = str(train_bank.source_bank_fingerprint)
    validation_source = str(validation_bank.source_bank_fingerprint)
    same_source = train_source == validation_source
    if same_source:
        overlap = np.intersect1d(
            train_bank.row_ids, validation_bank.row_ids, assume_unique=True
        )
        if len(overlap):
            preview = overlap[: min(8, len(overlap))].tolist()
            raise ValueError(
                f"Training and validation share {len(overlap):,} row IDs from "
                f"source bank {train_source}; examples: {preview}."
            )
        relationship = "same_source_bank_disjoint_row_ids"
    else:
        relationship = "different_source_banks_local_row_ids_not_compared"
    return {
        "relationship": relationship,
        "same_source_bank": same_source,
        "train_source_bank_fingerprint": train_source,
        "validation_source_bank_fingerprint": validation_source,
        "train_row_ids_sha256": _array_sha256(train_bank.row_ids),
        "validation_row_ids_sha256": _array_sha256(validation_bank.row_ids),
        "overlap_count": 0,
        "numeric_row_id_overlap_checked": same_source,
    }


def normalize_family(family: str) -> str:
    family = str(family).lower()
    aliases = {
        "multi": FAMILY_MULTICLASS,
        "three_class": FAMILY_MULTICLASS,
        "posterior": FAMILY_POSTERIOR_BINARY,
        "binary_posterior": FAMILY_POSTERIOR_BINARY,
        "likelihood": FAMILY_LIKELIHOOD_BINARY,
        "binary_likelihood": FAMILY_LIKELIHOOD_BINARY,
    }
    family = aliases.get(family, family)
    if family not in RATIO_FAMILIES:
        raise ValueError(f"Unknown ratio family {family!r}; expected {RATIO_FAMILIES}.")
    return family


class PlainCEMLP(nn.Module):
    """A logits-only ReLU MLP with no implicit regularization."""

    def __init__(
        self,
        input_dim: int,
        width: int,
        hidden_layers: int,
        n_classes: int,
    ) -> None:
        super().__init__()
        if min(int(input_dim), int(width), int(hidden_layers), int(n_classes)) < 1:
            raise ValueError("All PlainCEMLP dimensions must be positive.")
        layers: list[nn.Module] = []
        current = int(input_dim)
        for _ in range(int(hidden_layers)):
            layers.extend((nn.Linear(current, int(width)), nn.ReLU()))
            current = int(width)
        layers.append(nn.Linear(current, int(n_classes)))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def _mapping_from_config(config: Any) -> dict[str, Any]:
    if config is None:
        candidates = (
            "RATIO_CONFIG",
            "RATIO_TRAINING_CONFIG",
            "RATIO_TRAINING",
            "CLASSIFIER_CONFIG",
        )
        for name in candidates:
            if hasattr(paper_config, name):
                config = getattr(paper_config, name)
                break
        else:
            raise ValueError(
                "Pass ratio_config explicitly or define RATIO_TRAINING in "
                "paper_summary/config.py."
            )
    if is_dataclass(config):
        config = asdict(config)
    if not isinstance(config, Mapping):
        raise TypeError("ratio_config must be a mapping or dataclass instance.")
    mapping = dict(config)
    for nested_key in ("ratio", "ratio_training", "classifier", "training"):
        nested = mapping.get(nested_key)
        if isinstance(nested, Mapping) and any(
            name in nested
            for name in (
                "members",
                "classifier_members",
                "ensemble_members",
                "width",
                "hidden_width",
                "steps",
                "training_steps",
            )
        ):
            mapping = {**mapping, **dict(nested)}
    return mapping


def _config_value(
    mapping: Mapping[str, Any], names: Sequence[str], default: Any = None
) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def normalize_ratio_config(config: Any = None) -> dict[str, Any]:
    """Return the canonical, fully explicit classifier training contract."""

    raw = _mapping_from_config(config)
    normalized = {
        "schema": RATIO_CHECKPOINT_SCHEMA,
        "members": int(
            _config_value(
                raw,
                ("members", "classifier_members", "ensemble_size", "ensemble_members"),
                10,
            )
        ),
        "width": int(
            _config_value(raw, ("width", "classifier_width", "hidden_width"), 1024)
        ),
        "hidden_layers": int(
            _config_value(raw, ("hidden_layers", "classifier_layers", "layers"), 4)
        ),
        "row_batch_budget": int(
            _config_value(
                raw,
                ("row_batch_budget", "classifier_row_batch_budget", "batch_size"),
                1024,
            )
        ),
        "steps": int(
            _config_value(raw, ("steps", "classifier_steps", "training_steps"), 5000)
        ),
        "validation_interval": int(
            _config_value(
                raw,
                (
                    "validation_interval",
                    "classifier_validation_interval",
                    "validation_interval_steps",
                ),
                200,
            )
        ),
        "max_validation_groups": int(
            _config_value(raw, ("max_validation_groups",), 20_000)
        ),
        "validation_chunk_groups": int(
            _config_value(raw, ("validation_chunk_groups",), 1024)
        ),
        "initial_learning_rate": float(
            _config_value(raw, ("initial_learning_rate", "learning_rate"), 1.0e-4)
        ),
        "minimum_learning_rate": float(
            _config_value(raw, ("minimum_learning_rate", "min_learning_rate"), 1.0e-9)
        ),
        "learning_rate_schedule": str(
            _config_value(
                raw, ("learning_rate_schedule",), "six_equal_log10_plateaus"
            )
        ),
        "learning_rate_levels": tuple(
            float(value)
            for value in _config_value(
                raw,
                ("learning_rate_levels",),
                FROZEN_LEARNING_RATE_LEVELS,
            )
        ),
        "learning_rate_drop_fractions": tuple(
            float(value)
            for value in _config_value(
                raw,
                ("learning_rate_drop_fractions", "lr_drop_fractions"),
                FROZEN_LEARNING_RATE_DROP_FRACTIONS,
            )
        ),
        "gradient_clip": float(_config_value(raw, ("gradient_clip",), 5.0)),
        "minimum_validation_improvement": float(
            _config_value(raw, ("minimum_validation_improvement",), 1.0e-6)
        ),
        "member_seed_stride": int(
            _config_value(raw, ("member_seed_stride",), 10_007)
        ),
        "validation_seed_offset": int(
            _config_value(raw, ("validation_seed_offset",), 13)
        ),
        "objective": "equal_prior_cross_entropy_only",
        "optimizer": "adam",
        "activation": "relu",
        "dropout_probability": 0.0,
        "weight_decay": 0.0,
        "layer_normalization": False,
        "calibration_loss": False,
        "bridge_loss": False,
        "normalization_penalty": False,
        "explicit_exponentials_in_training": False,
        "genuine_training_rows": str(
            _config_value(
                raw,
                ("genuine_training_rows",),
                "same_complete_training_partition_as_flows",
            )
        ),
        "training_sampler": (
            "deterministic_shuffled_cycles_without_replacement_within_pass"
        ),
        "optimizer_steps_policy": (
            "run_all_configured_steps_select_best_heldout_checkpoint"
        ),
        "early_stopping_applied": False,
        "held_out_checkpoint_selection": (
            "best_validation_ce_after_first_complete_training_pass"
        ),
        "deterministic_backend_policy": (
            "strict_torch_deterministic_algorithms_cublas_4096_8_tf32_disabled"
        ),
        "ratio_aggregation": (
            "arithmetic_mean_memberwise_float64_softmax_probability_quotients"
        ),
        "inference_implementation": (
            "float64_log_softmax_difference_then_logsumexp_minus_log_members"
        ),
        "declared_ratio_representation": str(
            _config_value(
                raw, ("ratio_representation",), "log_softmax_difference"
            )
        ),
        "declared_ensemble_combination": str(
            _config_value(
                raw,
                ("ensemble_combination",),
                "logsumexp_arithmetic_mean",
            )
        ),
    }
    member_override = _config_value(raw, ("ensemble_members_override",), None)
    step_override = _config_value(raw, ("training_steps_override",), None)
    if member_override is not None:
        normalized["members"] = int(member_override)
    if step_override is not None:
        normalized["steps"] = int(step_override)
        normalized["validation_interval"] = min(
            int(normalized["validation_interval"]), int(normalized["steps"])
        )
    normalized["learning_rate_boundaries_zero_based"] = tuple(
        max(1, int(round(int(normalized["steps"]) * fraction)))
        for fraction in normalized["learning_rate_drop_fractions"]
    )
    schedule_edges = (
        0,
        *normalized["learning_rate_boundaries_zero_based"],
        int(normalized["steps"]),
    )
    normalized["learning_rate_plateau_step_counts"] = tuple(
        right - left for left, right in zip(schedule_edges[:-1], schedule_edges[1:])
    )
    regularization_guards = {
        "dropout": ("dropout", "dropout_probability"),
        "weight_decay": ("weight_decay",),
        "layer_normalization": ("layer_normalization", "layer_norm", "use_layer_norm"),
        "calibration_loss": ("calibration_loss", "use_calibration_loss"),
        "bridge_loss": ("bridge_loss", "use_bridge_loss"),
        "normalization_penalty": (
            "normalization_penalty",
            "use_normalization_penalty",
        ),
    }
    for label, names in regularization_guards.items():
        value = _config_value(raw, names, 0.0 if label != "layer_normalization" else False)
        if bool(value):
            raise ValueError(f"{label} must remain disabled for the paper ratio study.")

    required_declarations = {
        "optimizer": "adam",
        "activation": "relu",
        "objective": "equal_prior_cross_entropy_only",
        "explicit_exponentials_in_training": False,
        "early_stopping": False,
    }
    for name, expected in required_declarations.items():
        if name in raw and raw[name] != expected:
            raise ValueError(
                f"ratio_config[{name!r}] must be {expected!r}; got {raw[name]!r}."
            )
    if normalized["learning_rate_schedule"] != "six_equal_log10_plateaus":
        raise ValueError(
            "Only the frozen six_equal_log10_plateaus schedule is supported."
        )
    if normalized["declared_ratio_representation"] != "log_softmax_difference":
        raise ValueError("The frozen ratio representation is log_softmax_difference.")
    if normalized["declared_ensemble_combination"] != (
        "logsumexp_arithmetic_mean"
    ):
        raise ValueError(
            "The frozen ensemble combination is logsumexp_arithmetic_mean."
        )
    if normalized["genuine_training_rows"] != (
        "same_complete_training_partition_as_flows"
    ):
        raise ValueError(
            "Ratio classifiers must use the same complete training partition "
            "as the flows."
        )

    positive_integer_fields = (
        "members",
        "width",
        "hidden_layers",
        "row_batch_budget",
        "steps",
        "validation_interval",
        "max_validation_groups",
        "validation_chunk_groups",
        "member_seed_stride",
    )
    for name in positive_integer_fields:
        if int(normalized[name]) < 1:
            raise ValueError(f"ratio_config[{name!r}] must be positive.")
    if normalized["validation_interval"] > normalized["steps"]:
        raise ValueError("validation_interval cannot exceed the number of steps.")
    if not (
        0.0 < normalized["minimum_learning_rate"]
        <= normalized["initial_learning_rate"]
    ):
        raise ValueError("Learning-rate bounds are inconsistent.")
    fractions = normalized["learning_rate_drop_fractions"]
    if fractions != FROZEN_LEARNING_RATE_DROP_FRACTIONS:
        raise ValueError(
            "learning_rate_drop_fractions must be the frozen equal-duration "
            "six-plateau boundaries."
        )
    if tuple(sorted(set(fractions))) != fractions or any(
        not 0.0 < value < 1.0 for value in fractions
    ):
        raise ValueError(
            "learning_rate_drop_fractions must be unique, increasing, and in (0, 1)."
        )
    levels = normalized["learning_rate_levels"]
    if levels != FROZEN_LEARNING_RATE_LEVELS:
        raise ValueError(
            "learning_rate_levels must be exactly "
            "(1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9)."
        )
    if len(levels) != len(fractions) + 1:
        raise ValueError(
            "learning_rate_levels must have exactly one more entry than the "
            "drop fractions."
        )
    if any(not np.isfinite(value) or value <= 0.0 for value in levels):
        raise ValueError("Every learning-rate level must be finite and positive.")
    if any(right >= left for left, right in zip(levels[:-1], levels[1:])):
        raise ValueError("learning_rate_levels must be strictly decreasing.")
    if not math.isclose(
        levels[0], normalized["initial_learning_rate"], rel_tol=0.0, abs_tol=0.0
    ) or not math.isclose(
        levels[-1], normalized["minimum_learning_rate"], rel_tol=0.0, abs_tol=0.0
    ):
        raise ValueError(
            "The first and last learning-rate levels must equal the declared "
            "initial and minimum rates."
        )
    boundaries = normalized["learning_rate_boundaries_zero_based"]
    if tuple(sorted(set(boundaries))) != boundaries:
        raise ValueError(
            "The requested training_steps are too small to realize every "
            "learning-rate plateau."
        )
    plateau_counts = normalized["learning_rate_plateau_step_counts"]
    if len(plateau_counts) != 6 or sum(plateau_counts) != normalized["steps"]:
        raise ValueError("The six learning-rate plateaus do not cover all steps.")
    if max(plateau_counts) - min(plateau_counts) > 1:
        raise ValueError("Learning-rate plateau durations differ by more than one step.")
    if normalized["gradient_clip"] <= 0.0:
        raise ValueError("gradient_clip must be positive.")
    if normalized["minimum_validation_improvement"] < 0.0:
        raise ValueError("minimum_validation_improvement must be non-negative.")
    return normalized


def fit_common_transform(bank: RatioClassBank) -> tuple[np.ndarray, np.ndarray]:
    """Fit one class-symmetric column transform shared by all families."""

    count = 3 * len(bank)
    total = np.zeros(bank.input_dim, dtype=np.float64)
    total2 = np.zeros(bank.input_dim, dtype=np.float64)
    for values in bank.classes(FAMILY_MULTICLASS):
        for start in range(0, len(values), 65_536):
            chunk = np.asarray(values[start : start + 65_536], dtype=np.float64)
            total += chunk.sum(axis=0)
            total2 += np.square(chunk).sum(axis=0)
    center = total / count
    variance = np.maximum(total2 / count - np.square(center), 0.0)
    scale = np.sqrt(variance)
    scale = np.where(scale > 1.0e-6, scale, 1.0)
    return center.astype(np.float32), scale.astype(np.float32)


def _validate_transform(
    center: np.ndarray, scale: np.ndarray, input_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    center = np.ascontiguousarray(np.asarray(center, dtype=np.float32))
    scale = np.ascontiguousarray(np.asarray(scale, dtype=np.float32))
    if center.shape != (int(input_dim),) or scale.shape != (int(input_dim),):
        raise ValueError("Classifier center and scale have the wrong shape.")
    if not np.isfinite(center).all() or not np.isfinite(scale).all():
        raise ValueError("Classifier transform contains non-finite values.")
    if np.any(scale <= 0.0):
        raise ValueError("Every classifier scale must be positive.")
    return center, scale


def _family_contract(
    family: str,
    input_dim: int,
    config: Mapping[str, Any],
    *,
    study_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    family = normalize_family(family)
    contract = {
        "checkpoint_schema": RATIO_CHECKPOINT_SCHEMA,
        "family": family,
        "input_dim": int(input_dim),
        "n_classes": 3 if family == FAMILY_MULTICLASS else 2,
        "config": dict(config),
        "study_metadata": dict(study_metadata or {}),
    }
    # Store the exact canonical JSON data model, not arbitrary Python objects
    # whose equality semantics could make checkpoint validation ambiguous.
    return json.loads(_canonical_json(contract))


def _training_fingerprint(
    family: str,
    train_bank: RatioClassBank,
    validation_bank: RatioClassBank,
    validation_indices: np.ndarray,
    validation_selection_seed: int,
    source_separation: Mapping[str, Any],
    center: np.ndarray,
    scale: np.ndarray,
    family_contract: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    family = normalize_family(family)
    payload = {
        "family_contract": dict(family_contract),
        "train_bank_fingerprint": train_bank.fingerprint(),
        "validation_bank_fingerprint": validation_bank.fingerprint(),
        "validation_indices_sha256": _array_sha256(validation_indices),
        "validation_selection_seed": int(validation_selection_seed),
        "source_separation": dict(source_separation),
        "center_sha256": _array_sha256(center),
        "scale_sha256": _array_sha256(scale),
    }
    return _payload_sha256(payload), payload


def _learning_rate_for_step(
    step: int, total_steps: int, config: Mapping[str, Any]
) -> float:
    if int(total_steps) != int(config["steps"]):
        raise ValueError("Learning-rate schedule received a different step budget.")
    level_index = sum(
        int(step) >= int(boundary)
        for boundary in config["learning_rate_boundaries_zero_based"]
    )
    return float(config["learning_rate_levels"][level_index])


def _set_learning_rate(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
    config: Mapping[str, Any],
) -> float:
    value = _learning_rate_for_step(step, total_steps, config)
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


class DeterministicCyclingGroupSampler:
    """Visit every group once per shuffled pass, then reshuffle deterministically.

    A batch may straddle a pass boundary.  In that case its first part finishes
    the current permutation and its remainder starts the next independently
    shuffled permutation.  Consequently no group repeats within a pass, while
    the configured optimizer batch size and step count remain unchanged.
    """

    def __init__(self, n_groups: int, batch_groups: int, *, seed: int) -> None:
        self.n_groups = int(n_groups)
        self.batch_groups = int(batch_groups)
        self.seed = int(seed)
        if self.n_groups < 1 or self.batch_groups < 1:
            raise ValueError("Cycling sampler sizes must be positive.")
        self._rng = np.random.default_rng(self.seed)
        self._order = self._rng.permutation(self.n_groups).astype(
            np.int64, copy=False
        )
        self._cursor = 0
        self.group_draws = 0
        self.completed_full_passes = 0

    @property
    def groups_into_current_pass(self) -> int:
        return int(self._cursor)

    def next_indices(self) -> np.ndarray:
        remaining = self.batch_groups
        pieces: list[np.ndarray] = []
        while remaining:
            available = self.n_groups - self._cursor
            take = min(remaining, available)
            pieces.append(self._order[self._cursor : self._cursor + take])
            self._cursor += take
            self.group_draws += take
            remaining -= take
            if self._cursor == self.n_groups:
                self.completed_full_passes += 1
                self._order = self._rng.permutation(self.n_groups).astype(
                    np.int64, copy=False
                )
                self._cursor = 0
        output = np.ascontiguousarray(np.concatenate(pieces), dtype=np.int64)
        if output.shape != (self.batch_groups,):
            raise RuntimeError("Cycling sampler returned the wrong batch shape.")
        return output

    def exposure(self, *, n_classes: int) -> dict[str, Any]:
        return _training_exposure(
            self.n_groups,
            int(n_classes),
            self.group_draws,
            batch_groups=self.batch_groups,
            sampler_seed=self.seed,
        )


def _training_exposure(
    eligible_groups: int,
    n_classes: int,
    group_draws: int,
    *,
    batch_groups: int,
    sampler_seed: int,
) -> dict[str, Any]:
    eligible_groups = int(eligible_groups)
    n_classes = int(n_classes)
    group_draws = int(group_draws)
    if min(eligible_groups, n_classes, batch_groups) < 1 or group_draws < 0:
        raise ValueError("Invalid grouped-training exposure inputs.")
    full_passes, groups_into_current_pass = divmod(
        group_draws, eligible_groups
    )
    return {
        "training_sampler": (
            "deterministic_shuffled_cycles_without_replacement_within_pass"
        ),
        "sampler_seed": int(sampler_seed),
        "eligible_training_groups": eligible_groups,
        "groups_per_optimizer_step": int(batch_groups),
        "class_rows_per_group": n_classes,
        "group_draws": group_draws,
        "class_rows_seen": group_draws * n_classes,
        "completed_full_passes": full_passes,
        "groups_into_current_pass": groups_into_current_pass,
        "unique_groups_seen_at_least_once": min(eligible_groups, group_draws),
        "all_training_groups_seen": group_draws >= eligible_groups,
    }


def _group_cross_entropy(model: nn.Module, groups: torch.Tensor) -> torch.Tensor:
    n_groups, n_classes, n_features = groups.shape
    logits = model(groups.reshape(-1, n_features)).reshape(
        n_groups, n_classes, n_classes
    )
    labels = torch.arange(n_classes, device=groups.device).repeat(n_groups)
    return F.cross_entropy(logits.reshape(-1, n_classes), labels)


@torch.no_grad()
def _validation_cross_entropy(
    model: nn.Module,
    arrays: Sequence[np.ndarray],
    indices: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
    *,
    chunk_groups: int,
) -> float:
    model.eval()
    total = 0.0
    rows = 0
    for start in range(0, len(indices), int(chunk_groups)):
        local = indices[start : start + int(chunk_groups)]
        groups = _group_batch(arrays, local, center, scale, device)
        loss = _group_cross_entropy(model, groups)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite held-out ratio validation CE.")
        total += float(loss.detach().cpu()) * len(local)
        rows += len(local)
    return total / max(1, rows)


def classifier_checkpoint_paths(
    checkpoint_dir: str | Path, members: int
) -> list[Path]:
    directory = Path(checkpoint_dir)
    return [directory / f"member_{member:02d}.pt" for member in range(int(members))]


def _safe_torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        saved = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        saved = torch.load(path, map_location=device)
    if not isinstance(saved, dict):
        raise RuntimeError(f"Ratio checkpoint {path} is not a dictionary.")
    return saved


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as temporary:
            temporary_name = temporary.name
        torch.save(dict(payload), temporary_name)
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _validate_training_exposure_metadata(
    history: Mapping[str, Any],
    exposure_payload: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    eligible_groups: int,
    n_classes: int,
    member_seed: int,
) -> None:
    group_batch = max(1, int(config["row_batch_budget"]) // int(n_classes))
    sampler_seed = int(member_seed) + 1
    if history.get("training_sampler") != config["training_sampler"]:
        raise RuntimeError("Ratio checkpoint has the wrong training sampler.")
    if history.get("optimizer_steps_policy") != config["optimizer_steps_policy"]:
        raise RuntimeError("Ratio checkpoint has the wrong optimizer-step policy.")
    if history.get("learning_rate_levels") != list(
        config["learning_rate_levels"]
    ):
        raise RuntimeError("Ratio checkpoint has the wrong learning-rate levels.")
    if history.get("learning_rate_boundaries_zero_based") != list(
        config["learning_rate_boundaries_zero_based"]
    ):
        raise RuntimeError("Ratio checkpoint has the wrong learning-rate boundaries.")
    if history.get("learning_rate_plateau_step_counts") != list(
        config["learning_rate_plateau_step_counts"]
    ):
        raise RuntimeError("Ratio checkpoint has the wrong plateau durations.")
    if int(history.get("eligible_training_groups", -1)) != int(eligible_groups):
        raise RuntimeError("Ratio checkpoint has the wrong eligible-group count.")
    if int(history.get("groups_per_optimizer_step", -1)) != group_batch:
        raise RuntimeError("Ratio checkpoint has the wrong grouped batch size.")
    if int(history.get("class_rows_per_group", -1)) != int(n_classes):
        raise RuntimeError("Ratio checkpoint has the wrong class-row multiplicity.")

    completed_steps = int(history.get("optimizer_steps_completed", -1))
    selected_step = int(history.get("selected_step", -1))
    if completed_steps != int(config["steps"]):
        raise RuntimeError(
            "Ratio checkpoint did not complete every configured optimizer step."
        )
    validation_steps = [int(value) for value in history.get("step", ())]
    expected_validation_steps = list(
        range(
            int(config["validation_interval"]),
            int(config["steps"]) + 1,
            int(config["validation_interval"]),
        )
    )
    if not expected_validation_steps or expected_validation_steps[-1] != int(
        config["steps"]
    ):
        expected_validation_steps.append(int(config["steps"]))
    if validation_steps != expected_validation_steps:
        raise RuntimeError(
            "Ratio checkpoint does not contain the frozen held-out validation schedule."
        )
    if selected_step not in validation_steps:
        raise RuntimeError("Selected ratio checkpoint step was never validated.")
    expected_completed = _training_exposure(
        eligible_groups,
        n_classes,
        completed_steps * group_batch,
        batch_groups=group_batch,
        sampler_seed=sampler_seed,
    )
    expected_selected = _training_exposure(
        eligible_groups,
        n_classes,
        selected_step * group_batch,
        batch_groups=group_batch,
        sampler_seed=sampler_seed,
    )
    expected_payload = {
        "completed": expected_completed,
        "selected_checkpoint": expected_selected,
    }
    if dict(exposure_payload) != expected_payload:
        raise RuntimeError("Ratio checkpoint training-exposure payload is inconsistent.")
    if history.get("completed_training_exposure") != expected_completed:
        raise RuntimeError("Completed training exposure is inconsistent with history.")
    if history.get("selected_checkpoint_exposure") != expected_selected:
        raise RuntimeError("Selected-checkpoint exposure is inconsistent with history.")
    if not bool(expected_selected["all_training_groups_seen"]):
        raise RuntimeError(
            "Selected ratio checkpoint did not see every eligible training group."
        )

    fields = (
        "train_ce",
        "validation_ce",
        "learning_rate",
        "validation_group_draws",
        "validation_class_rows_seen",
        "validation_completed_full_passes",
        "validation_groups_into_current_pass",
        "validation_checkpoint_eligible",
    )
    if any(len(history.get(name, ())) != len(validation_steps) for name in fields):
        raise RuntimeError("Ratio validation exposure history has inconsistent lengths.")
    validation_values = [float(value) for value in history["validation_ce"]]
    training_values = [float(value) for value in history["train_ce"]]
    learning_rates = [float(value) for value in history["learning_rate"]]
    if not np.isfinite(validation_values).all() or not np.isfinite(
        training_values
    ).all():
        raise RuntimeError("Ratio checkpoint contains non-finite CE history.")
    expected_learning_rates = [
        _learning_rate_for_step(step - 1, int(config["steps"]), config)
        for step in validation_steps
    ]
    if learning_rates != expected_learning_rates:
        raise RuntimeError("Ratio checkpoint learning-rate history is inconsistent.")

    replay_best_value = math.inf
    replay_best_step = -1
    for index, step in enumerate(validation_steps):
        expected = _training_exposure(
            eligible_groups,
            n_classes,
            step * group_batch,
            batch_groups=group_batch,
            sampler_seed=sampler_seed,
        )
        observed = (
            int(history["validation_group_draws"][index]),
            int(history["validation_class_rows_seen"][index]),
            int(history["validation_completed_full_passes"][index]),
            int(history["validation_groups_into_current_pass"][index]),
            bool(history["validation_checkpoint_eligible"][index]),
        )
        expected_values = (
            expected["group_draws"],
            expected["class_rows_seen"],
            expected["completed_full_passes"],
            expected["groups_into_current_pass"],
            expected["all_training_groups_seen"],
        )
        if observed != expected_values:
            raise RuntimeError(
                f"Ratio validation exposure is inconsistent at step {step}."
            )
        if expected["all_training_groups_seen"] and validation_values[
            index
        ] < replay_best_value - float(config["minimum_validation_improvement"]):
            replay_best_value = validation_values[index]
            replay_best_step = step
    if replay_best_step != selected_step or not math.isclose(
        float(history.get("selected_validation_ce", math.nan)),
        replay_best_value,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise RuntimeError(
            "Ratio checkpoint is not the replayed best eligible held-out-CE state."
        )


def _load_classifier_family(
    family: str,
    checkpoint_dir: Path,
    config: Mapping[str, Any],
    family_contract: Mapping[str, Any],
    training_fingerprint: str,
    training_fingerprint_payload: Mapping[str, Any],
    center: np.ndarray,
    scale: np.ndarray,
    *,
    base_seed: int,
    expected_training_groups: int,
    load_runtime: Mapping[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    family = normalize_family(family)
    n_classes = int(family_contract["n_classes"])
    paths = classifier_checkpoint_paths(checkpoint_dir, int(config["members"]))
    packs = []
    for member, checkpoint in enumerate(paths):
        saved = _safe_torch_load(checkpoint, device)
        expected_seed = int(base_seed) + int(config["member_seed_stride"]) * member
        expected = {
            "checkpoint_schema": RATIO_CHECKPOINT_SCHEMA,
            "family_contract": dict(family_contract),
            "family_contract_fingerprint": _payload_sha256(family_contract),
            "training_fingerprint": training_fingerprint,
            "training_fingerprint_payload": dict(training_fingerprint_payload),
            "member": member,
            "member_seed": expected_seed,
        }
        for key, value in expected.items():
            if saved.get(key) != value:
                raise RuntimeError(
                    f"Ratio checkpoint contract mismatch for {key!r}: {checkpoint}. "
                    "Use a new run tag instead of mixing campaigns."
                )
        _validate_training_exposure_metadata(
            saved.get("history", {}),
            saved.get("training_exposure", {}),
            config,
            eligible_groups=int(expected_training_groups),
            n_classes=n_classes,
            member_seed=expected_seed,
        )
        _validate_recorded_deterministic_runtime(
            saved.get("training_runtime", {})
        )
        saved_runtime = saved.get("training_runtime", {})
        if int(saved_runtime.get("member_seed", -1)) != expected_seed or int(
            saved_runtime.get("sampler_seed", -1)
        ) != expected_seed + 1:
            raise RuntimeError(
                f"Ratio checkpoint runtime seed metadata mismatch: {checkpoint}."
            )
        saved_center, saved_scale = _validate_transform(
            saved.get("center"), saved.get("scale"), int(family_contract["input_dim"])
        )
        if not np.array_equal(saved_center, center) or not np.array_equal(
            saved_scale, scale
        ):
            raise RuntimeError(f"Classifier transform mismatch in {checkpoint}.")
        model = PlainCEMLP(
            int(family_contract["input_dim"]),
            int(config["width"]),
            int(config["hidden_layers"]),
            n_classes,
        ).to(device)
        model.load_state_dict(saved["state_dict"], strict=True)
        model.eval()
        packs.append(
            {
                "model": model,
                "center": center.copy(),
                "scale": scale.copy(),
                "history": saved["history"],
                "family": family,
                "member": member,
                "member_seed": expected_seed,
                "checkpoint": checkpoint,
                "family_contract": dict(family_contract),
                "training_fingerprint": training_fingerprint,
                "data_provenance": saved["data_provenance"],
                "training_runtime": saved["training_runtime"],
                "load_runtime": dict(load_runtime),
            }
        )
    validate_classifier_ensemble(packs, expected_family=family)
    return packs


def train_classifier_family(
    family: str,
    train_bank: RatioClassBank,
    validation_bank: RatioClassBank,
    checkpoint_dir: str | Path,
    ratio_config: Any,
    *,
    seed: int,
    validation_seed: int | None = None,
    device: str | torch.device | None = None,
    center: np.ndarray | None = None,
    scale: np.ndarray | None = None,
    study_metadata: Mapping[str, Any] | None = None,
    load_if_available: bool = True,
) -> list[dict[str, Any]]:
    """Train or exactly validate and load one CE classifier ensemble."""

    family = normalize_family(family)
    config = normalize_ratio_config(ratio_config)
    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    runtime_settings = configure_deterministic_runtime(device)
    _validate_recorded_deterministic_runtime(runtime_settings)
    if train_bank.input_dim != validation_bank.input_dim:
        raise ValueError("Training and validation banks have different dimensions.")
    train_bank_fingerprint = train_bank.fingerprint()
    validation_bank_fingerprint = validation_bank.fingerprint()
    if train_bank_fingerprint == validation_bank_fingerprint:
        raise ValueError("Ratio validation must use a genuinely held-out bank.")
    source_separation = validate_source_bank_separation(
        train_bank, validation_bank
    )
    if center is None or scale is None:
        if center is not None or scale is not None:
            raise ValueError("Provide both center and scale, or neither.")
        center, scale = fit_common_transform(train_bank)
    center, scale = _validate_transform(center, scale, train_bank.input_dim)

    family_contract = _family_contract(
        family,
        train_bank.input_dim,
        config,
        study_metadata=study_metadata,
    )
    validation_seed = int(seed) if validation_seed is None else int(validation_seed)
    validation_selection_seed = validation_seed + int(
        config["validation_seed_offset"]
    )
    validation_rng = np.random.default_rng(
        validation_selection_seed
    )
    n_validation = min(len(validation_bank), int(config["max_validation_groups"]))
    validation_indices = np.ascontiguousarray(
        validation_rng.choice(len(validation_bank), n_validation, replace=False),
        dtype=np.int64,
    )
    training_fingerprint, fingerprint_payload = _training_fingerprint(
        family,
        train_bank,
        validation_bank,
        validation_indices,
        validation_selection_seed,
        source_separation,
        center,
        scale,
        family_contract,
    )
    checkpoint_dir = Path(checkpoint_dir)
    paths = classifier_checkpoint_paths(checkpoint_dir, int(config["members"]))
    expected_paths = {path.resolve() for path in paths}
    unexpected_paths = {
        path.resolve() for path in checkpoint_dir.glob("member_*.pt")
    } - expected_paths
    if unexpected_paths:
        raise RuntimeError(
            f"Unexpected ratio members in {checkpoint_dir}: "
            f"{sorted(path.name for path in unexpected_paths)}. Use a new run tag."
        )
    present = [path.is_file() for path in paths]
    if load_if_available and all(present):
        return _load_classifier_family(
            family,
            checkpoint_dir,
            config,
            family_contract,
            training_fingerprint,
            fingerprint_payload,
            center,
            scale,
            base_seed=int(seed),
            expected_training_groups=len(train_bank),
            load_runtime=runtime_settings,
            device=device,
        )
    if load_if_available and any(present):
        raise RuntimeError(
            f"Partial ratio ensemble in {checkpoint_dir}; refusing to mix runs."
        )

    arrays = train_bank.classes(family)
    validation_arrays = validation_bank.classes(family)
    n_classes = len(arrays)
    group_batch = max(1, int(config["row_batch_budget"]) // n_classes)
    total_steps = int(config["steps"])
    if total_steps * group_batch < len(train_bank):
        raise ValueError(
            f"The configured {total_steps:,} steps × {group_batch:,} groups/step "
            f"cannot expose all {len(train_bank):,} eligible training groups."
        )
    validation_interval = int(config["validation_interval"])
    packs: list[dict[str, Any]] = []
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for member, checkpoint in enumerate(paths):
        member_seed = int(seed) + int(config["member_seed_stride"]) * member
        _seed_everything(member_seed)
        model = PlainCEMLP(
            train_bank.input_dim,
            int(config["width"]),
            int(config["hidden_layers"]),
            n_classes,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(config["initial_learning_rate"])
        )
        sampler = DeterministicCyclingGroupSampler(
            len(train_bank), group_batch, seed=member_seed + 1
        )
        history: dict[str, Any] = {
            "step": [],
            "train_ce": [],
            "validation_ce": [],
            "learning_rate": [],
            "selected_step": None,
            "selected_validation_ce": None,
            "optimizer_steps_completed": 0,
            "learning_rate_levels": list(config["learning_rate_levels"]),
            "learning_rate_boundaries_zero_based": list(
                config["learning_rate_boundaries_zero_based"]
            ),
            "learning_rate_plateau_step_counts": list(
                config["learning_rate_plateau_step_counts"]
            ),
            "optimizer_steps_policy": config["optimizer_steps_policy"],
            "training_sampler": config["training_sampler"],
            "eligible_training_groups": len(train_bank),
            "groups_per_optimizer_step": group_batch,
            "class_rows_per_group": n_classes,
            "validation_group_draws": [],
            "validation_class_rows_seen": [],
            "validation_completed_full_passes": [],
            "validation_groups_into_current_pass": [],
            "validation_checkpoint_eligible": [],
        }
        best_state: dict[str, torch.Tensor] | None = None
        best_value = math.inf
        best_step = 0
        running_losses: list[float] = []
        print(
            f"{family} member {member + 1}/{config['members']}: "
            f"{total_steps:,} CE steps, {group_batch:,} matched groups/step"
        )
        for step in range(1, total_steps + 1):
            learning_rate = _set_learning_rate(
                optimizer, step - 1, total_steps, config
            )
            indices = sampler.next_indices()
            groups = _group_batch(arrays, indices, center, scale, device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = _group_cross_entropy(model, groups)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite {family} training CE.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["gradient_clip"])
            )
            optimizer.step()
            history["optimizer_steps_completed"] = step
            running_losses.append(float(loss.detach().cpu()))
            if step % validation_interval == 0 or step == total_steps:
                value = _validation_cross_entropy(
                    model,
                    validation_arrays,
                    validation_indices,
                    center,
                    scale,
                    device,
                    chunk_groups=int(config["validation_chunk_groups"]),
                )
                history["step"].append(step)
                history["train_ce"].append(float(np.mean(running_losses)))
                history["validation_ce"].append(value)
                history["learning_rate"].append(learning_rate)
                exposure = sampler.exposure(n_classes=n_classes)
                history["validation_group_draws"].append(
                    exposure["group_draws"]
                )
                history["validation_class_rows_seen"].append(
                    exposure["class_rows_seen"]
                )
                history["validation_completed_full_passes"].append(
                    exposure["completed_full_passes"]
                )
                history["validation_groups_into_current_pass"].append(
                    exposure["groups_into_current_pass"]
                )
                checkpoint_eligible = bool(
                    exposure["all_training_groups_seen"]
                )
                history["validation_checkpoint_eligible"].append(
                    checkpoint_eligible
                )
                running_losses.clear()
                print(
                    f"  step {step:6d}/{total_steps}: "
                    f"train={history['train_ce'][-1]:.6f}, "
                    f"held-out={value:.6f}, lr={learning_rate:.1e}, "
                    f"passes={exposure['completed_full_passes']}"
                )
                if checkpoint_eligible:
                    if value < best_value - float(
                        config["minimum_validation_improvement"]
                    ):
                        best_value = value
                        best_state = copy.deepcopy(model.state_dict())
                        best_step = step
        if best_state is None:
            raise RuntimeError(f"{family} never produced a finite checkpoint.")
        model.load_state_dict(best_state)
        model.eval()
        history["selected_step"] = best_step
        history["selected_validation_ce"] = best_value
        completed_exposure = sampler.exposure(n_classes=n_classes)
        selected_exposure = _training_exposure(
            len(train_bank),
            n_classes,
            best_step * group_batch,
            batch_groups=group_batch,
            sampler_seed=member_seed + 1,
        )
        if not bool(selected_exposure["all_training_groups_seen"]):
            raise RuntimeError(
                "Selected ratio checkpoint did not consume the full training "
                "partition."
            )
        history["completed_training_exposure"] = completed_exposure
        history["selected_checkpoint_exposure"] = selected_exposure
        checkpoint_payload = {
            "checkpoint_schema": RATIO_CHECKPOINT_SCHEMA,
            "state_dict": best_state,
            "family_contract": dict(family_contract),
            "family_contract_fingerprint": _payload_sha256(family_contract),
            "training_fingerprint": training_fingerprint,
            "training_fingerprint_payload": fingerprint_payload,
            "center": center,
            "scale": scale,
            "history": history,
            "training_exposure": {
                "completed": completed_exposure,
                "selected_checkpoint": selected_exposure,
            },
            "member": member,
            "member_seed": member_seed,
            "training_runtime": {
                **runtime_settings,
                "member_seed": member_seed,
                "sampler_seed": member_seed + 1,
            },
            "data_provenance": {
                "training": dict(train_bank.provenance),
                "validation": dict(validation_bank.provenance),
                "train_bank_fingerprint": train_bank_fingerprint,
                "validation_bank_fingerprint": validation_bank_fingerprint,
                "source_separation": source_separation,
            },
        }
        _validate_training_exposure_metadata(
            history,
            checkpoint_payload["training_exposure"],
            config,
            eligible_groups=len(train_bank),
            n_classes=n_classes,
            member_seed=member_seed,
        )
        _validate_recorded_deterministic_runtime(
            checkpoint_payload["training_runtime"]
        )
        _atomic_torch_save(checkpoint_payload, checkpoint)
        packs.append(
            {
                "model": model,
                "center": center.copy(),
                "scale": scale.copy(),
                "history": history,
                "family": family,
                "member": member,
                "member_seed": member_seed,
                "checkpoint": checkpoint,
                "family_contract": dict(family_contract),
                "training_fingerprint": training_fingerprint,
                "data_provenance": checkpoint_payload["data_provenance"],
                "training_runtime": checkpoint_payload["training_runtime"],
                "load_runtime": runtime_settings,
            }
        )
    validate_classifier_ensemble(packs, expected_family=family)
    return packs


def train_ratio_ensembles(
    train_bank: RatioClassBank,
    validation_bank: RatioClassBank,
    checkpoint_root: str | Path,
    ratio_config: Any,
    *,
    seed: int,
    device: str | torch.device | None = None,
    study_metadata: Mapping[str, Any] | None = None,
    load_if_available: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Train/load multiclass and both binary ensembles with one transform."""

    center, scale = fit_common_transform(train_bank)
    checkpoint_root = Path(checkpoint_root)
    outputs = {}
    for family_index, family in enumerate(RATIO_FAMILIES):
        outputs[family] = train_classifier_family(
            family,
            train_bank,
            validation_bank,
            checkpoint_root / family,
            ratio_config,
            seed=int(seed) + 100_000 * family_index,
            validation_seed=int(seed),
            device=device,
            center=center,
            scale=scale,
            study_metadata=study_metadata,
            load_if_available=load_if_available,
        )
    return outputs


def validate_classifier_ensemble(
    packs: Sequence[Mapping[str, Any]], *, expected_family: str | None = None
) -> None:
    if not packs:
        raise ValueError("A classifier ensemble must contain at least one member.")
    families = {str(pack.get("family")) for pack in packs}
    if len(families) != 1:
        raise ValueError("Classifier ensemble members have different families.")
    family = normalize_family(next(iter(families)))
    if expected_family is not None and family != normalize_family(expected_family):
        raise ValueError("Classifier ensemble has the wrong family.")
    members = [int(pack.get("member", -1)) for pack in packs]
    if members != list(range(len(packs))):
        raise ValueError("Classifier members must be ordered and contiguous from zero.")
    fingerprints = {str(pack.get("training_fingerprint")) for pack in packs}
    contracts = {
        _canonical_json(pack.get("family_contract", {})) for pack in packs
    }
    if len(fingerprints) != 1 or len(contracts) != 1:
        raise ValueError("Classifier members belong to different training contracts.")
    for pack in packs:
        contract = pack.get("family_contract")
        if not isinstance(contract, Mapping):
            raise ValueError("Classifier member lacks a family contract mapping.")
        if normalize_family(str(contract.get("family"))) != family:
            raise ValueError(
                "Classifier member family label disagrees with its frozen contract."
            )
        expected_classes = 3 if family == FAMILY_MULTICLASS else 2
        if int(contract.get("n_classes", -1)) != expected_classes:
            raise ValueError("Classifier contract has the wrong output class count.")
    centers = [np.asarray(pack.get("center"), dtype=np.float32) for pack in packs]
    scales = [np.asarray(pack.get("scale"), dtype=np.float32) for pack in packs]
    if any(
        not np.array_equal(centers[0], value) for value in centers[1:]
    ) or any(not np.array_equal(scales[0], value) for value in scales[1:]):
        raise ValueError("Classifier ensemble members use different transforms.")


def _validated_prediction_points(
    packs: Sequence[Mapping[str, Any]], points: np.ndarray, batch_size: int
) -> np.ndarray:
    validate_classifier_ensemble(packs)
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2:
        raise ValueError("points must be a two-dimensional array.")
    if not np.isfinite(points).all():
        raise ValueError("points contains non-finite values.")
    input_dim = int(packs[0]["family_contract"]["input_dim"])
    if points.shape[1] != input_dim:
        raise ValueError(
            f"points has {points.shape[1]} columns; expected {input_dim}."
        )
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive.")
    return points


@torch.no_grad()
def predict_probabilities(
    packs: Sequence[Mapping[str, Any]],
    points: np.ndarray,
    *,
    batch_size: int = 16_384,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ensemble-mean and member-wise float64 softmax probabilities."""

    points = _validated_prediction_points(packs, points, batch_size)
    member_probabilities = []
    for pack in packs:
        model = pack["model"]
        model.eval()
        model_device = next(model.parameters()).device
        chunks = []
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
        n_classes = int(pack["family_contract"]["n_classes"])
        member_probabilities.append(
            np.concatenate(chunks, axis=0)
            if chunks
            else np.empty((0, n_classes), dtype=np.float64)
        )
    members = np.stack(member_probabilities, axis=0)
    return members.mean(axis=0), members


@torch.no_grad()
def predict_log_probability_ratio(
    packs: Sequence[Mapping[str, Any]],
    points: np.ndarray,
    *,
    numerator: int = CLASS_S,
    denominator: int = CLASS_P,
    batch_size: int = 16_384,
    return_members: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Return the stable log arithmetic mean of member-wise density ratios.

    For member ``m`` this computes

    ``ell_m = log_softmax(logits_m)[numerator] -
    log_softmax(logits_m)[denominator]``

    in float64.  The deployed ensemble is the arithmetic, not geometric,
    ratio mean, so its logarithm is ``logsumexp(ell_m) - log(M)``.  No
    probability or raw quotient is formed by this function.
    """

    points = _validated_prediction_points(packs, points, batch_size)
    n_classes = int(packs[0]["family_contract"]["n_classes"])
    numerator = int(numerator)
    denominator = int(denominator)
    if not 0 <= numerator < n_classes or not 0 <= denominator < n_classes:
        raise ValueError("Ratio class indices lie outside the classifier output.")
    if numerator == denominator:
        raise ValueError("Ratio numerator and denominator must differ.")

    devices = {str(next(pack["model"].parameters()).device) for pack in packs}
    if len(devices) != 1:
        raise ValueError("Every classifier ensemble member must use the same device.")
    model_device = next(packs[0]["model"].parameters()).device
    ensemble_chunks: list[np.ndarray] = []
    member_chunks: list[np.ndarray] = []
    for start in range(0, len(points), int(batch_size)):
        member_log_ratios = []
        for pack in packs:
            model = pack["model"]
            model.eval()
            values = (
                points[start : start + int(batch_size)] - pack["center"]
            ) / pack["scale"]
            tensor = torch.as_tensor(
                values, dtype=torch.float32, device=model_device
            )
            logits = model(tensor).to(torch.float64)
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError(
                    "Non-finite classifier logits reached ratio inference."
                )
            log_probability = torch.log_softmax(logits, dim=1)
            member_log_ratios.append(
                log_probability[:, numerator] - log_probability[:, denominator]
            )
        stacked = torch.stack(member_log_ratios, dim=0)
        ensemble = torch.logsumexp(stacked, dim=0) - math.log(len(packs))
        if not bool(torch.isfinite(stacked).all()) or not bool(
            torch.isfinite(ensemble).all()
        ):
            raise FloatingPointError("Non-finite log density ratio at inference.")
        ensemble_chunks.append(ensemble.detach().cpu().numpy())
        if return_members:
            member_chunks.append(stacked.detach().cpu().numpy())

    ensemble_output = (
        np.concatenate(ensemble_chunks, axis=0)
        if ensemble_chunks
        else np.empty(0, dtype=np.float64)
    )
    if not return_members:
        return ensemble_output
    member_output = (
        np.concatenate(member_chunks, axis=1)
        if member_chunks
        else np.empty((len(packs), 0), dtype=np.float64)
    )
    return ensemble_output, member_output


def _exponentiate_log_ratio(
    log_ratio: np.ndarray, *, name: str
) -> np.ndarray:
    """Exponentiate only a final log ratio, with representability guards."""

    log_ratio = np.asarray(log_ratio, dtype=np.float64)
    maximum_log = math.log(np.finfo(np.float64).max)
    minimum_log = math.log(np.nextafter(np.float64(0.0), np.float64(1.0)))
    if np.any(log_ratio > maximum_log):
        raise OverflowError(
            f"{name} exceeds float64 ratio range; use the log-ratio API."
        )
    if np.any(log_ratio < minimum_log):
        raise FloatingPointError(
            f"{name} underflows float64 ratio range; use the log-ratio API."
        )
    ratio = np.exp(log_ratio)
    if not np.isfinite(ratio).all() or np.any(ratio <= 0.0):
        raise FloatingPointError(
            f"{name} is not representable as a positive finite float64 ratio; "
            "use the log-ratio API."
        )
    return ratio


def predict_probability_ratio(
    packs: Sequence[Mapping[str, Any]],
    points: np.ndarray,
    *,
    numerator: int = CLASS_S,
    denominator: int = CLASS_P,
    batch_size: int = 16_384,
    return_members: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper returning the final float64 density ratio.

    All model evaluation and ensemble arithmetic occur in log space through
    :func:`predict_log_probability_ratio`.  Exponentiation happens only on
    the final returned ensemble/member log ratios and is guarded against
    float64 overflow and underflow.
    """

    output = predict_log_probability_ratio(
        packs,
        points,
        numerator=numerator,
        denominator=denominator,
        batch_size=batch_size,
        return_members=return_members,
    )
    if not return_members:
        return _exponentiate_log_ratio(output, name="ensemble log ratio")
    ensemble_log_ratio, member_log_ratios = output
    return (
        _exponentiate_log_ratio(
            ensemble_log_ratio, name="ensemble log ratio"
        ),
        _exponentiate_log_ratio(member_log_ratios, name="member log ratio"),
    )


def predict_multiclass_ratios(
    packs: Sequence[Mapping[str, Any]],
    points: np.ndarray,
    *,
    batch_size: int = 16_384,
    return_members: bool = False,
) -> tuple[Any, Any]:
    """Compatibility API returning both representable multiclass ratios."""

    validate_classifier_ensemble(packs, expected_family=FAMILY_MULTICLASS)
    posterior = predict_probability_ratio(
        packs,
        points,
        numerator=CLASS_S,
        denominator=CLASS_P,
        batch_size=batch_size,
        return_members=return_members,
    )
    likelihood = predict_probability_ratio(
        packs,
        points,
        numerator=CLASS_S,
        denominator=CLASS_L,
        batch_size=batch_size,
        return_members=return_members,
    )
    return posterior, likelihood


def predict_multiclass_log_ratios(
    packs: Sequence[Mapping[str, Any]],
    points: np.ndarray,
    *,
    batch_size: int = 16_384,
    return_members: bool = False,
) -> tuple[Any, Any]:
    """Return stable multiclass posterior and likelihood log ratios."""

    validate_classifier_ensemble(packs, expected_family=FAMILY_MULTICLASS)
    posterior = predict_log_probability_ratio(
        packs,
        points,
        numerator=CLASS_S,
        denominator=CLASS_P,
        batch_size=batch_size,
        return_members=return_members,
    )
    likelihood = predict_log_probability_ratio(
        packs,
        points,
        numerator=CLASS_S,
        denominator=CLASS_L,
        batch_size=batch_size,
        return_members=return_members,
    )
    return posterior, likelihood


def _required_classifier_family(
    classifiers: Mapping[str, Sequence[Mapping[str, Any]]], family: str
) -> Sequence[Mapping[str, Any]]:
    family = normalize_family(family)
    if not isinstance(classifiers, Mapping):
        raise TypeError("classifiers must map explicit family names to ensembles.")
    if family not in classifiers:
        raise KeyError(f"classifiers is missing required family {family!r}.")
    packs = classifiers[family]
    validate_classifier_ensemble(packs, expected_family=family)
    return packs


def predict_posterior_log_ratio(
    classifiers: Mapping[str, Sequence[Mapping[str, Any]]],
    points: np.ndarray,
    *,
    factorization: str,
    batch_size: int = 16_384,
    return_members: bool = False,
) -> Any:
    """Evaluate the posterior correction as a stable log density ratio."""

    factorization = str(factorization).lower()
    if factorization in {"multiclass", "multi", "three_class"}:
        return predict_log_probability_ratio(
            _required_classifier_family(classifiers, FAMILY_MULTICLASS),
            points,
            numerator=CLASS_S,
            denominator=CLASS_P,
            batch_size=batch_size,
            return_members=return_members,
        )
    if factorization in {"binary", "separate_binary", "posterior_binary"}:
        return predict_log_probability_ratio(
            _required_classifier_family(classifiers, FAMILY_POSTERIOR_BINARY),
            points,
            numerator=CLASS_S,
            denominator=CLASS_P,
            batch_size=batch_size,
            return_members=return_members,
        )
    raise ValueError("factorization must be 'multiclass' or 'binary'.")


def predict_likelihood_log_ratio(
    classifiers: Mapping[str, Sequence[Mapping[str, Any]]],
    points: np.ndarray,
    *,
    factorization: str,
    batch_size: int = 16_384,
    return_members: bool = False,
) -> Any:
    """Evaluate the likelihood correction as a stable log density ratio."""

    factorization = str(factorization).lower()
    if factorization in {"multiclass", "multi", "three_class"}:
        return predict_log_probability_ratio(
            _required_classifier_family(classifiers, FAMILY_MULTICLASS),
            points,
            numerator=CLASS_S,
            denominator=CLASS_L,
            batch_size=batch_size,
            return_members=return_members,
        )
    if factorization in {"binary", "separate_binary", "likelihood_binary"}:
        # In a binary S/L classifier the likelihood-reference class occupies
        # output index one (the same numeric index as CLASS_P).
        return predict_log_probability_ratio(
            _required_classifier_family(classifiers, FAMILY_LIKELIHOOD_BINARY),
            points,
            numerator=CLASS_S,
            denominator=CLASS_P,
            batch_size=batch_size,
            return_members=return_members,
        )
    raise ValueError("factorization must be 'multiclass' or 'binary'.")


def predict_posterior_ratio(
    classifiers: Mapping[str, Sequence[Mapping[str, Any]]],
    points: np.ndarray,
    *,
    factorization: str,
    batch_size: int = 16_384,
    return_members: bool = False,
) -> Any:
    """Compatibility API returning a representable posterior density ratio."""

    output = predict_posterior_log_ratio(
        classifiers,
        points,
        factorization=factorization,
        batch_size=batch_size,
        return_members=return_members,
    )
    if not return_members:
        return _exponentiate_log_ratio(output, name="posterior ensemble log ratio")
    ensemble, members = output
    return (
        _exponentiate_log_ratio(ensemble, name="posterior ensemble log ratio"),
        _exponentiate_log_ratio(members, name="posterior member log ratio"),
    )


def predict_likelihood_ratio(
    classifiers: Mapping[str, Sequence[Mapping[str, Any]]],
    points: np.ndarray,
    *,
    factorization: str,
    batch_size: int = 16_384,
    return_members: bool = False,
) -> Any:
    """Compatibility API returning a representable likelihood density ratio."""

    output = predict_likelihood_log_ratio(
        classifiers,
        points,
        factorization=factorization,
        batch_size=batch_size,
        return_members=return_members,
    )
    if not return_members:
        return _exponentiate_log_ratio(output, name="likelihood ensemble log ratio")
    ensemble, members = output
    return (
        _exponentiate_log_ratio(ensemble, name="likelihood ensemble log ratio"),
        _exponentiate_log_ratio(members, name="likelihood member log ratio"),
    )


def ratio_ensemble_summary(
    classifiers: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    """Return compact checkpoint/selection metadata for manifests or tables."""

    rows = []
    for family in RATIO_FAMILIES:
        packs = classifiers[family]
        validate_classifier_ensemble(packs, expected_family=family)
        for pack in packs:
            history = pack["history"]
            runtime = pack["training_runtime"]
            selected_exposure = history["selected_checkpoint_exposure"]
            completed_exposure = history["completed_training_exposure"]
            rows.append(
                {
                    "family": family,
                    "member": int(pack["member"]),
                    "member_seed": int(pack["member_seed"]),
                    "selected_step": int(history["selected_step"]),
                    "selected_validation_ce": float(
                        history["selected_validation_ce"]
                    ),
                    "selected_completed_full_passes": int(
                        selected_exposure["completed_full_passes"]
                    ),
                    "selected_group_draws": int(selected_exposure["group_draws"]),
                    "completed_full_passes": int(
                        completed_exposure["completed_full_passes"]
                    ),
                    "completed_group_draws": int(
                        completed_exposure["group_draws"]
                    ),
                    "training_device": str(runtime["device"]),
                    "torch_version": str(runtime["torch_version"]),
                    "deterministic_algorithms": bool(
                        runtime["torch_deterministic_algorithms_enabled"]
                    ),
                    "checkpoint": str(pack["checkpoint"]),
                    "training_fingerprint": str(pack["training_fingerprint"]),
                }
            )
    return rows


def validate_probability_quotient_arithmetic() -> None:
    """Check stable log arithmetic averaging against its raw-ratio identity."""

    probabilities = np.asarray(
        [
            [[0.8, 0.2], [0.3, 0.7]],
            [[0.6, 0.4], [0.9, 0.1]],
        ],
        dtype=np.float64,
    )
    member_log_ratios = (
        np.log(probabilities[:, :, 0]) - np.log(probabilities[:, :, 1])
    )
    maximum = member_log_ratios.max(axis=0)
    ensemble_log_ratio = maximum + np.log(
        np.exp(member_log_ratios - maximum).mean(axis=0)
    )
    obtained = _exponentiate_log_ratio(
        ensemble_log_ratio, name="validation ensemble log ratio"
    )
    ratio_of_means = probabilities[:, :, 0].mean(axis=0) / probabilities[
        :, :, 1
    ].mean(axis=0)
    expected = np.asarray([2.75, 4.714285714285714], dtype=np.float64)
    if not np.allclose(obtained, expected, rtol=1.0e-14, atol=0.0):
        raise AssertionError("Member-wise probability quotient arithmetic changed.")
    if np.allclose(obtained, ratio_of_means, rtol=1.0e-12, atol=0.0):
        raise AssertionError("Ratio of mean probabilities was used accidentally.")


def validate_cycling_group_sampler() -> None:
    """Cheap deterministic check of complete shuffled training passes."""

    first = DeterministicCyclingGroupSampler(7, 3, seed=123)
    second = DeterministicCyclingGroupSampler(7, 3, seed=123)
    first_draws = np.concatenate([first.next_indices() for _ in range(5)])
    second_draws = np.concatenate([second.next_indices() for _ in range(5)])
    if not np.array_equal(first_draws, second_draws):
        raise AssertionError("Cycling group sampler is not deterministic.")
    expected_group_set = np.arange(7)
    for start in (0, 7):
        if not np.array_equal(
            np.sort(first_draws[start : start + 7]), expected_group_set
        ):
            raise AssertionError(
                "A shuffled training pass omitted or repeated an eligible group."
            )
    exposure = first.exposure(n_classes=3)
    if (
        exposure["group_draws"] != 15
        or exposure["completed_full_passes"] != 2
        or exposure["groups_into_current_pass"] != 1
        or exposure["class_rows_seen"] != 45
        or not exposure["all_training_groups_seen"]
    ):
        raise AssertionError("Cycling group exposure accounting is inconsistent.")


__all__ = [
    "CLASS_L",
    "CLASS_P",
    "CLASS_S",
    "DeterministicCyclingGroupSampler",
    "FAMILY_LIKELIHOOD_BINARY",
    "FAMILY_MULTICLASS",
    "FAMILY_POSTERIOR_BINARY",
    "FROZEN_LEARNING_RATE_DROP_FRACTIONS",
    "FROZEN_LEARNING_RATE_LEVELS",
    "PlainCEMLP",
    "RATIO_CHECKPOINT_SCHEMA",
    "RATIO_FAMILIES",
    "RatioClassBank",
    "classifier_checkpoint_paths",
    "configure_deterministic_runtime",
    "fit_common_transform",
    "normalize_family",
    "normalize_ratio_config",
    "predict_likelihood_log_ratio",
    "predict_likelihood_ratio",
    "predict_log_probability_ratio",
    "predict_multiclass_log_ratios",
    "predict_multiclass_ratios",
    "predict_posterior_log_ratio",
    "predict_posterior_ratio",
    "predict_probabilities",
    "predict_probability_ratio",
    "ratio_bank_fingerprint",
    "ratio_ensemble_summary",
    "train_classifier_family",
    "train_ratio_ensembles",
    "validate_classifier_ensemble",
    "validate_cycling_group_sampler",
    "validate_probability_quotient_arithmetic",
    "validate_source_bank_separation",
]
