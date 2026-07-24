"""Factorizable normalizing-flow helpers for the Exercise 8 FNF variant.

This module is deliberately independent of ``utils_systematics.py``.  It
implements the input-systematic construction of

    Valsecchi, Donegà and Wallny, arXiv:2602.13184,

adapted to the five-dimensional ML4HEP toy.  The frozen nominal density is the
Exercise 5 hNDE,

    p_0(x) = p_ref(x) r_0(x) / Z_0,

with the same reference flow and four-member process/reference ratio ensemble
used by the nominal Exercise 8.  It is composed with an invertible,
nuisance-dependent residual map ``T_alpha``:

    p(x | alpha) = p_0(T_alpha(x)) |det dT_alpha / dx|.

The one fixed nominal constant ``Z_0`` is inherited from the hNDE
normalization.  The residual is exactly the identity at ``alpha = 0``.  Its
autoregressive scale and shift are linear plus quadratic polynomials in
``alpha``.  The change-of-variables formula then makes every nuisance value a
normalized density; no alpha-dependent partition-function correction is used
anywhere here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def _as_alpha_tensor(
    alpha: float | np.ndarray | torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return one nuisance value per event with shape ``(batch, 1)``."""
    if isinstance(alpha, torch.Tensor):
        value = alpha.to(device=device, dtype=dtype)
    else:
        value = torch.as_tensor(alpha, device=device, dtype=dtype)

    if value.ndim == 0:
        return value.reshape(1, 1).expand(batch_size, 1)
    if value.ndim == 1:
        if value.numel() == 1:
            return value.reshape(1, 1).expand(batch_size, 1)
        if value.numel() == batch_size:
            return value.reshape(batch_size, 1)
    if value.ndim == 2 and value.shape == (batch_size, 1):
        return value
    raise ValueError(
        "alpha must be scalar, length batch_size, or shape (batch_size, 1)."
    )


class _ConstantCoefficientNet(nn.Module):
    """Four trainable coefficients for the first autoregressive dimension."""

    def __init__(self, n_outputs: int = 4) -> None:
        super().__init__()
        self.coefficients = nn.Parameter(torch.zeros(n_outputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.coefficients.unsqueeze(0).expand(x.shape[0], -1)


class _CoefficientMLP(nn.Module):
    """Small MLP initialized so the residual starts at the identity."""

    def __init__(
        self,
        n_inputs: int,
        hidden_features: int,
        hidden_layers: int,
        n_outputs: int = 4,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = int(n_inputs)
        for _ in range(int(hidden_layers)):
            layers.extend(
                [
                    nn.Linear(previous, int(hidden_features)),
                    nn.ELU(),
                ]
            )
            previous = int(hidden_features)
        layers.append(nn.Linear(previous, int(n_outputs)))
        self.network = nn.Sequential(*layers)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PolynomialAutoregressiveResidual(nn.Module):
    """One affine autoregressive FNF residual layer.

    For dimension ``j`` the correction from the varied frame to the nominal
    frame is

    ``x_nom,j = x_j * exp(s_j) + t_j``,

    where ``s_j`` and ``t_j`` depend only on ``x_<j`` and have the polynomial
    nuisance dependence

    ``a_j * alpha + b_j * alpha**2``.

    This is invertible by sequential substitution.  Both the map and its
    log-Jacobian vanish continuously to the identity at ``alpha = 0``.
    """

    def __init__(
        self,
        n_features: int,
        *,
        hidden_features: int = 128,
        hidden_layers: int = 2,
        quadratic_damping: float = 1.0,
        log_scale_clip: float = 1.5,
        shift_clip: float = 5.0,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.quadratic_damping = float(quadratic_damping)
        self.log_scale_clip = float(log_scale_clip)
        self.shift_clip = float(shift_clip)

        if self.n_features < 1:
            raise ValueError("n_features must be positive.")
        if self.log_scale_clip <= 0.0 or self.shift_clip <= 0.0:
            raise ValueError("The residual clipping scales must be positive.")

        networks: list[nn.Module] = []
        for dimension in range(self.n_features):
            if dimension == 0:
                networks.append(_ConstantCoefficientNet())
            else:
                networks.append(
                    _CoefficientMLP(
                        dimension,
                        hidden_features=hidden_features,
                        hidden_layers=hidden_layers,
                    )
                )
        self.coefficient_networks = nn.ModuleList(networks)

    def _scale_and_shift(
        self,
        dimension: int,
        x_previous: torch.Tensor,
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = self.coefficient_networks[dimension](x_previous)
        alpha_linear = alpha[:, 0]
        alpha_quadratic = (
            self.quadratic_damping * alpha_linear**2.0
        )

        raw_log_scale = (
            coefficients[:, 0] * alpha_linear
            + coefficients[:, 1] * alpha_quadratic
        )
        raw_shift = (
            coefficients[:, 2] * alpha_linear
            + coefficients[:, 3] * alpha_quadratic
        )
        log_scale = self.log_scale_clip * torch.tanh(
            raw_log_scale / self.log_scale_clip
        )
        shift = self.shift_clip * torch.tanh(raw_shift / self.shift_clip)
        return log_scale, shift

    def forward(
        self,
        x: torch.Tensor,
        alpha: float | np.ndarray | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map varied observations to the nominal frame."""
        alpha_tensor = _as_alpha_tensor(
            alpha,
            batch_size=x.shape[0],
            device=x.device,
            dtype=x.dtype,
        )
        transformed = []
        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for dimension in range(self.n_features):
            previous = x[:, :dimension]
            log_scale, shift = self._scale_and_shift(
                dimension, previous, alpha_tensor
            )
            transformed.append(
                x[:, dimension] * torch.exp(log_scale) + shift
            )
            log_det = log_det + log_scale
        return torch.stack(transformed, dim=1), log_det

    def inverse(
        self,
        y: torch.Tensor,
        alpha: float | np.ndarray | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map nominal-frame points to the varied observation frame."""
        alpha_tensor = _as_alpha_tensor(
            alpha,
            batch_size=y.shape[0],
            device=y.device,
            dtype=y.dtype,
        )
        reconstructed: list[torch.Tensor] = []
        inverse_log_det = torch.zeros(
            y.shape[0], device=y.device, dtype=y.dtype
        )
        for dimension in range(self.n_features):
            if reconstructed:
                previous = torch.stack(reconstructed, dim=1)
            else:
                previous = y[:, :0]
            log_scale, shift = self._scale_and_shift(
                dimension, previous, alpha_tensor
            )
            value = (y[:, dimension] - shift) * torch.exp(-log_scale)
            reconstructed.append(value)
            inverse_log_det = inverse_log_det - log_scale
        return torch.stack(reconstructed, dim=1), inverse_log_det


class _ReversePermutation(nn.Module):
    """Self-inverse feature reversal with zero log-Jacobian."""

    def forward(
        self,
        x: torch.Tensor,
        alpha: float | np.ndarray | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del alpha
        return x.flip(dims=(1,)), torch.zeros(
            x.shape[0], device=x.device, dtype=x.dtype
        )

    def inverse(
        self,
        y: torch.Tensor,
        alpha: float | np.ndarray | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(y, alpha)


class FactorizableResidualStack(nn.Module):
    """A mixing stack whose net map is exactly identity at ``alpha = 0``."""

    def __init__(
        self,
        n_features: int,
        *,
        n_residual_layers: int = 2,
        hidden_features: int = 128,
        hidden_layers: int = 2,
        quadratic_damping: float = 1.0,
        log_scale_clip: float = 1.5,
        shift_clip: float = 5.0,
    ) -> None:
        super().__init__()
        n_residual_layers = int(n_residual_layers)
        if n_residual_layers < 1:
            raise ValueError("n_residual_layers must be positive.")

        transforms: list[nn.Module] = []
        for layer_index in range(n_residual_layers):
            if layer_index > 0:
                transforms.append(_ReversePermutation())
            transforms.append(
                PolynomialAutoregressiveResidual(
                    n_features,
                    hidden_features=hidden_features,
                    hidden_layers=hidden_layers,
                    quadratic_damping=quadratic_damping,
                    log_scale_clip=log_scale_clip,
                    shift_clip=shift_clip,
                )
            )
        # Repeated reversals must cancel at alpha=0.  For an even number of
        # residual layers the list above contains an odd number of reversals.
        if n_residual_layers % 2 == 0:
            transforms.append(_ReversePermutation())
        self.transforms = nn.ModuleList(transforms)

    def forward(
        self,
        x: torch.Tensor,
        alpha: float | np.ndarray | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value = x
        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for transform in self.transforms:
            value, increment = transform(value, alpha)
            log_det = log_det + increment
        return value, log_det

    def inverse(
        self,
        y: torch.Tensor,
        alpha: float | np.ndarray | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value = y
        log_det = torch.zeros(y.shape[0], device=y.device, dtype=y.dtype)
        for transform in reversed(self.transforms):
            value, increment = transform.inverse(value, alpha)
            log_det = log_det + increment
        return value, log_det


class _FrozenRatioMLP(nn.Module):
    """Differentiable reconstruction of one saved Exercise 5 ONNX MLP."""

    def __init__(self, linear_layers: Sequence[nn.Linear]) -> None:
        super().__init__()
        if len(linear_layers) < 2:
            raise ValueError("A ratio MLP must contain at least two linear layers.")
        self.linear_layers = nn.ModuleList(linear_layers)
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = values
        for layer in self.linear_layers[:-1]:
            hidden = F.silu(layer(hidden))
        return torch.sigmoid(self.linear_layers[-1](hidden)).reshape(-1)


def _onnx_attribute(node: Any, name: str, default: Any) -> Any:
    """Read one ONNX node attribute without importing ONNX at module import."""
    from onnx.helper import get_attribute_value

    for attribute in node.attribute:
        if attribute.name == name:
            return get_attribute_value(attribute)
    return default


def _torch_ratio_mlp_from_onnx(model_proto: Any) -> _FrozenRatioMLP:
    """Reconstruct the workshop's SiLU MLP directly from its ONNX weights.

    Exercise 5 exports ``DensityRatioLightning`` with one ONNX ``Gemm`` node
    per linear layer.  Reconstructing those frozen layers in PyTorch preserves
    gradients with respect to the input coordinates, which are required when
    the FNF residual is trained.  No ratio network is retrained or distilled.
    """
    from onnx import numpy_helper

    initializers = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model_proto.graph.initializer
    }
    linear_layers: list[nn.Linear] = []
    gemm_nodes = [
        node for node in model_proto.graph.node if node.op_type == "Gemm"
    ]
    for node in gemm_nodes:
        if len(node.input) < 3:
            raise ValueError("An Exercise 5 Gemm node is missing its bias.")
        if node.input[1] not in initializers or node.input[2] not in initializers:
            raise ValueError("An Exercise 5 Gemm parameter is not an initializer.")
        if int(_onnx_attribute(node, "transA", 0)) != 0:
            raise ValueError("Unsupported transposed data input in ratio ONNX graph.")

        stored_weight = np.asarray(
            initializers[node.input[1]], dtype=np.float32
        )
        stored_bias = np.asarray(
            initializers[node.input[2]], dtype=np.float32
        ).reshape(-1)
        transposed_weight = int(_onnx_attribute(node, "transB", 0))
        alpha = float(_onnx_attribute(node, "alpha", 1.0))
        beta = float(_onnx_attribute(node, "beta", 1.0))
        effective_weight = (
            stored_weight if transposed_weight else stored_weight.T
        )
        effective_weight = alpha * effective_weight
        effective_bias = beta * stored_bias

        if effective_weight.ndim != 2:
            raise ValueError("A ratio ONNX weight is not two-dimensional.")
        if effective_weight.shape[0] != len(effective_bias):
            raise ValueError("A ratio ONNX weight and bias are incompatible.")
        layer = nn.Linear(
            int(effective_weight.shape[1]),
            int(effective_weight.shape[0]),
        )
        with torch.no_grad():
            layer.weight.copy_(torch.from_numpy(effective_weight.copy()))
            layer.bias.copy_(torch.from_numpy(effective_bias.copy()))
        linear_layers.append(layer)

    # Newer torch exporters may express Linear as MatMul followed by Add.
    if not linear_layers:
        consumers: dict[str, list[Any]] = {}
        for candidate in model_proto.graph.node:
            for input_name in candidate.input:
                consumers.setdefault(input_name, []).append(candidate)
        for node in model_proto.graph.node:
            if node.op_type != "MatMul" or len(node.input) != 2:
                continue
            weight_name = next(
                (name for name in node.input if name in initializers),
                None,
            )
            if weight_name is None:
                continue
            add_candidates = [
                candidate
                for candidate in consumers.get(node.output[0], [])
                if candidate.op_type == "Add"
            ]
            if len(add_candidates) != 1:
                raise ValueError(
                    "Could not identify the bias Add for a ratio MatMul node."
                )
            add_node = add_candidates[0]
            bias_name = next(
                (
                    name
                    for name in add_node.input
                    if name in initializers and name != weight_name
                ),
                None,
            )
            if bias_name is None:
                raise ValueError("A ratio MatMul layer is missing its bias.")
            stored_weight = np.asarray(
                initializers[weight_name], dtype=np.float32
            )
            effective_weight = stored_weight.T
            effective_bias = np.asarray(
                initializers[bias_name], dtype=np.float32
            ).reshape(-1)
            if effective_weight.shape[0] != len(effective_bias):
                raise ValueError("A ratio MatMul weight and bias are incompatible.")
            layer = nn.Linear(
                int(effective_weight.shape[1]),
                int(effective_weight.shape[0]),
            )
            with torch.no_grad():
                layer.weight.copy_(torch.from_numpy(effective_weight.copy()))
                layer.bias.copy_(torch.from_numpy(effective_bias.copy()))
            linear_layers.append(layer)

    if not linear_layers:
        raise ValueError(
            "No Gemm or MatMul-plus-Add layers were found in the ratio ONNX graph."
        )
    if not any(node.op_type == "Sigmoid" for node in model_proto.graph.node):
        raise ValueError("The saved Exercise 5 ratio graph has no sigmoid output.")
    return _FrozenRatioMLP(linear_layers)


class _DifferentiableAffineScaler(nn.Module):
    """Torch equivalent of the fitted Exercise 5 feature scaler."""

    def __init__(self, scaler: Any, features: Sequence[str]) -> None:
        super().__init__()
        expected_features = list(features)
        fitted = scaler
        if hasattr(scaler, "named_transformers_"):
            if "scaler" not in scaler.named_transformers_:
                raise ValueError("The ratio ColumnTransformer has no 'scaler' step.")
            fitted = scaler.named_transformers_["scaler"]
            scaled_columns = None
            for name, _, columns in scaler.transformers_:
                if name == "scaler":
                    scaled_columns = list(columns)
                    break
            if scaled_columns != expected_features:
                raise ValueError(
                    "The saved ratio scaler feature order differs from the "
                    "Exercise 5 feature order."
                )

        if hasattr(fitted, "min_") and hasattr(fitted, "scale_"):
            multiplier = np.asarray(fitted.scale_, dtype=np.float32)
            offset = np.asarray(fitted.min_, dtype=np.float32)
        elif hasattr(fitted, "mean_") and hasattr(fitted, "scale_"):
            standard_deviation = np.asarray(fitted.scale_, dtype=np.float32)
            multiplier = 1.0 / standard_deviation
            offset = -np.asarray(fitted.mean_, dtype=np.float32) / standard_deviation
        else:
            raise TypeError(
                "Only the affine MinMax/Standard scalers used by Exercise 5 "
                "can be reconstructed differentiably."
            )
        if multiplier.shape != (len(expected_features),):
            raise ValueError("The ratio scaler dimension does not match FEATURES.")
        self.register_buffer("multiplier", torch.from_numpy(multiplier.copy()))
        self.register_buffer("offset", torch.from_numpy(offset.copy()))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.multiplier + self.offset


class _FrozenRatioMember(nn.Module):
    """One frozen scaler-plus-classifier member in density-ratio space."""

    def __init__(
        self,
        scaler: Any,
        model_proto: Any,
        features: Sequence[str],
        *,
        score_epsilon: float = 1.0e-9,
        ratio_floor: float = 1.0e-12,
    ) -> None:
        super().__init__()
        self.scaler = _DifferentiableAffineScaler(scaler, features)
        self.network = _torch_ratio_mlp_from_onnx(model_proto)
        self.score_epsilon = float(score_epsilon)
        self.ratio_floor = float(ratio_floor)

    def log_ratio(self, values: torch.Tensor) -> torch.Tensor:
        score = self.network(self.scaler(values))
        score = torch.clamp(score, 0.0, 1.0 - self.score_epsilon)
        ratio = torch.clamp(
            score / torch.clamp(1.0 - score, min=self.score_epsilon),
            min=self.ratio_floor,
        )
        return torch.log(ratio)


class HybridNominalDensity(nn.Module):
    """Frozen Exercise 5 hNDE nominal density for one selected process."""

    def __init__(
        self,
        reference_flow_pack: Mapping[str, Any],
        ratio_packs: Sequence[Mapping[str, Any]],
        *,
        features: Sequence[str],
    ) -> None:
        super().__init__()
        self.features = list(features)
        if self.features != list(reference_flow_pack["features"]):
            raise ValueError("The hNDE and reference-flow feature orders differ.")
        if not ratio_packs:
            raise ValueError("At least one Exercise 5 ratio member is required.")

        self.reference_flow = reference_flow_pack["flow"]
        for parameter in self.reference_flow.parameters():
            parameter.requires_grad = False
        self.reference_flow.eval()

        reference_scaler = reference_flow_pack["scaler"]
        self.register_buffer(
            "reference_mean",
            torch.as_tensor(reference_scaler.mean, dtype=torch.float32),
        )
        self.register_buffer(
            "reference_std",
            torch.as_tensor(reference_scaler.std, dtype=torch.float32),
        )
        self.register_buffer(
            "reference_log_det",
            torch.tensor(
                reference_scaler.log_det_x_to_z_standardization,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "log_ratio_normalization",
            torch.tensor(0.0, dtype=torch.float32),
        )
        self.ratio_members = nn.ModuleList(
            [
                _FrozenRatioMember(
                    pack["scaler"],
                    pack["model_proto"],
                    self.features,
                )
                for pack in ratio_packs
            ]
        )
        for parameter in self.parameters():
            parameter.requires_grad = False

        self._sampling_values_cpu: torch.Tensor | None = None
        self._sampling_probabilities_cpu: torch.Tensor | None = None

    @property
    def device(self) -> torch.device:
        return self.reference_mean.device

    def member_scores_tensor(self, values: torch.Tensor) -> torch.Tensor:
        """Return the native classifier score from every frozen member."""
        return torch.stack(
            [
                member.network(member.scaler(values))
                for member in self.ratio_members
            ],
            dim=0,
        )

    def log_ratio_tensor(self, values: torch.Tensor) -> torch.Tensor:
        member_log_ratios = torch.stack(
            [member.log_ratio(values) for member in self.ratio_members],
            dim=0,
        )
        return torch.logsumexp(member_log_ratios, dim=0) - math.log(
            len(self.ratio_members)
        )

    def configure_reference_support(
        self,
        reference_sample: np.ndarray,
        raw_ratio: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Normalize and retain one self-consistent reference support.

        ``raw_ratio`` must be evaluated with this differentiable ensemble.
        Using the same numerical implementation for the normalization,
        importance-resampling weights, and density evaluation avoids an
        otherwise unnecessary ONNX/PyTorch backend mismatch in saturated
        classifier tails.
        """
        values = np.asarray(reference_sample, dtype=np.float32)
        ratios = np.asarray(raw_ratio, dtype=np.float64).reshape(-1)
        if values.ndim != 2 or values.shape[1] != len(self.features):
            raise ValueError("reference_sample has the wrong shape.")
        if len(values) != len(ratios):
            raise ValueError("reference sample values and ratios are misaligned.")
        if not np.isfinite(ratios).all() or np.any(ratios < 0.0):
            raise ValueError("raw_ratio must be finite and nonnegative.")

        normalization = float(ratios.mean())
        if not np.isfinite(normalization) or normalization <= 0.0:
            raise ValueError("The ratio normalization must be finite and positive.")
        normalized_ratio = ratios / normalization
        ratio_sum = float(normalized_ratio.sum())
        if ratio_sum <= 0.0:
            raise ValueError("reference_sample_ratio has zero total weight.")

        with torch.no_grad():
            self.log_ratio_normalization.fill_(math.log(normalization))
        self._sampling_values_cpu = torch.from_numpy(values.copy())
        self._sampling_probabilities_cpu = torch.from_numpy(
            (normalized_ratio / ratio_sum).copy()
        )
        return normalization, normalized_ratio

    def log_prob_tensor(self, values: torch.Tensor) -> torch.Tensor:
        standardized = (values - self.reference_mean) / self.reference_std
        reference_log_probability = (
            self.reference_flow.log_prob(standardized)
            + self.reference_log_det
        )
        return (
            reference_log_probability
            + self.log_ratio_tensor(values)
            - self.log_ratio_normalization
        )

    @torch.no_grad()
    def sample_tensor(self, n: int) -> torch.Tensor:
        """Importance-resample the persisted Exercise 5 reference support."""
        if (
            self._sampling_values_cpu is None
            or self._sampling_probabilities_cpu is None
        ):
            raise RuntimeError(
                "This hNDE pack has no reference support for importance resampling."
            )
        indices = torch.multinomial(
            self._sampling_probabilities_cpu,
            int(n),
            replacement=True,
        )
        return self._sampling_values_cpu[indices].to(self.device)


def build_hybrid_nominal_density(
    process_name: str,
    *,
    reference_flow_pack: Mapping[str, Any],
    ratio_packs: Sequence[Mapping[str, Any]],
    features: Sequence[str],
    reference_sample: np.ndarray,
    device: torch.device,
    batch_size: int = 65_536,
) -> dict[str, Any]:
    """Build and self-normalize the frozen Exercise 5 hNDE for FNF."""
    density = HybridNominalDensity(
        reference_flow_pack,
        ratio_packs,
        features=features,
    ).to(device)
    density.eval()
    density_pack = {
        "density": density,
        "features": list(features),
        "coordinate_scaler": reference_flow_pack["scaler"],
        "identifier": (
            f"exercise5_hnde_{process_name}_"
            f"{len(ratio_packs)}members"
        ),
        "reference_flow_path": reference_flow_pack.get("path", ""),
    }
    raw_ratio = hybrid_nominal_ratio_x(
        density_pack,
        reference_sample,
        batch_size=batch_size,
    ).astype(np.float64)
    normalization, normalized_ratio = density.configure_reference_support(
        reference_sample,
        raw_ratio,
    )
    density_pack["ratio_normalization"] = normalization
    density_pack["reference_sample_ratio"] = normalized_ratio
    return density_pack


@torch.no_grad()
def hybrid_nominal_log_prob_x(
    density_pack: Mapping[str, Any],
    x: pd.DataFrame | np.ndarray,
    *,
    batch_size: int = 65_536,
) -> np.ndarray:
    """Evaluate a frozen Exercise 5 hNDE in original feature coordinates."""
    density: HybridNominalDensity = density_pack["density"]
    features = list(density_pack["features"])
    if isinstance(x, pd.DataFrame):
        values = x[features].to_numpy(dtype=np.float32)
    else:
        values = np.asarray(x, dtype=np.float32)
    density.eval()
    chunks = []
    for start in range(0, len(values), int(batch_size)):
        stop = min(start + int(batch_size), len(values))
        values_tensor = torch.as_tensor(
            values[start:stop],
            dtype=torch.float32,
            device=density.device,
        )
        chunks.append(
            density.log_prob_tensor(values_tensor).detach().cpu().numpy()
        )
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)


@torch.no_grad()
def hybrid_nominal_ratio_x(
    density_pack: Mapping[str, Any],
    x: pd.DataFrame | np.ndarray,
    *,
    batch_size: int = 65_536,
) -> np.ndarray:
    """Evaluate the differentiable copy of the Exercise 5 ratio ensemble."""
    density: HybridNominalDensity = density_pack["density"]
    features = list(density_pack["features"])
    if isinstance(x, pd.DataFrame):
        values = x[features].to_numpy(dtype=np.float32)
    else:
        values = np.asarray(x, dtype=np.float32)
    density.eval()
    chunks = []
    for start in range(0, len(values), int(batch_size)):
        stop = min(start + int(batch_size), len(values))
        values_tensor = torch.as_tensor(
            values[start:stop],
            dtype=torch.float32,
            device=density.device,
        )
        chunks.append(
            torch.exp(density.log_ratio_tensor(values_tensor))
            .detach()
            .cpu()
            .numpy()
        )
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)


@torch.no_grad()
def hybrid_nominal_member_scores_x(
    density_pack: Mapping[str, Any],
    x: pd.DataFrame | np.ndarray,
    *,
    batch_size: int = 65_536,
) -> np.ndarray:
    """Evaluate every reconstructed member in native classifier-score space."""
    density: HybridNominalDensity = density_pack["density"]
    features = list(density_pack["features"])
    if isinstance(x, pd.DataFrame):
        values = x[features].to_numpy(dtype=np.float32)
    else:
        values = np.asarray(x, dtype=np.float32)
    density.eval()
    chunks = []
    for start in range(0, len(values), int(batch_size)):
        stop = min(start + int(batch_size), len(values))
        values_tensor = torch.as_tensor(
            values[start:stop],
            dtype=torch.float32,
            device=density.device,
        )
        chunks.append(
            density.member_scores_tensor(values_tensor)
            .detach()
            .cpu()
            .numpy()
        )
    if not chunks:
        return np.empty((len(density.ratio_members), 0), dtype=np.float32)
    return np.concatenate(chunks, axis=1)


class FactorizableSystematicFlow(nn.Module):
    """Frozen nominal hNDE plus a trainable FNF systematic residual."""

    def __init__(
        self,
        base_density_pack: Mapping[str, Any],
        *,
        residual_config: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.base_density: HybridNominalDensity = base_density_pack["density"]
        for parameter in self.base_density.parameters():
            parameter.requires_grad = False
        self.base_density.eval()

        features = list(base_density_pack["features"])
        self.features = features
        scaler = base_density_pack["coordinate_scaler"]
        self.register_buffer(
            "scaler_mean",
            torch.as_tensor(scaler.mean, dtype=torch.float32),
        )
        self.register_buffer(
            "scaler_std",
            torch.as_tensor(scaler.std, dtype=torch.float32),
        )
        config = dict(residual_config)
        self.residual_config = config
        self.residual = FactorizableResidualStack(
            len(features),
            n_residual_layers=int(config.get("n_residual_layers", 2)),
            hidden_features=int(config.get("hidden_features", 128)),
            hidden_layers=int(config.get("hidden_layers", 2)),
            quadratic_damping=float(config.get("quadratic_damping", 1.0)),
            log_scale_clip=float(config.get("log_scale_clip", 1.5)),
            shift_clip=float(config.get("shift_clip", 5.0)),
        )

    @property
    def device(self) -> torch.device:
        return self.scaler_mean.device

    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.scaler_mean) / self.scaler_std

    def _destandardize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scaler_std + self.scaler_mean

    def log_prob_tensor(
        self,
        x: torch.Tensor,
        alpha: float | np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the normalized density in original feature coordinates."""
        standardized = self._standardize(x)
        nominal_frame, residual_log_det = self.residual(
            standardized, alpha
        )
        nominal_values = self._destandardize(nominal_frame)
        return (
            self.base_density.log_prob_tensor(nominal_values)
            + residual_log_det
        )

    @torch.no_grad()
    def sample_tensor(
        self,
        n: int,
        alpha: float | np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """Draw from ``p(x | alpha)`` in original feature coordinates."""
        nominal_values = self.base_density.sample_tensor(int(n))
        nominal_frame = self._standardize(nominal_values)
        observed_frame, _ = self.residual.inverse(nominal_frame, alpha)
        return self._destandardize(observed_frame)


def fnf_checkpoint_path(process_name: str, model_dir: str | Path) -> Path:
    return Path(model_dir) / f"fnf_{process_name}.pt"


def _torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_factorizable_systematic_flow(
    process_name: str,
    *,
    base_density_pack: Mapping[str, Any],
    model_dir: str | Path,
    device: torch.device,
    expected_features: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Load one residual checkpoint on top of its supplied nominal hNDE."""
    path = fnf_checkpoint_path(process_name, model_dir)
    checkpoint = _torch_load(path, device)
    features = list(checkpoint["features"])
    if expected_features is not None and features != list(expected_features):
        raise ValueError(
            f"Checkpoint features {features} do not match "
            f"{list(expected_features)}."
        )
    if features != list(base_density_pack["features"]):
        raise ValueError("The FNF and nominal-hNDE feature orders differ.")
    checkpoint_identifier = checkpoint.get("base_density_identifier")
    supplied_identifier = base_density_pack.get("identifier")
    if (
        checkpoint_identifier is not None
        and supplied_identifier is not None
        and checkpoint_identifier != supplied_identifier
    ):
        raise ValueError(
            "The FNF checkpoint was trained with a different nominal density."
        )

    model = FactorizableSystematicFlow(
        base_density_pack,
        residual_config=checkpoint["residual_config"],
    ).to(device)
    model.residual.load_state_dict(checkpoint["residual_state_dict"])
    model.eval()
    model.base_density.eval()
    return {
        "model": model,
        "features": features,
        "residual_config": dict(checkpoint["residual_config"]),
        "training_config": dict(checkpoint.get("training_config", {})),
        "history": checkpoint.get("history", {}),
        "path": path,
    }


@dataclass
class _AnchorSplit:
    x_train: np.ndarray
    alpha_train: np.ndarray
    weight_train: np.ndarray
    x_validation: np.ndarray
    alpha_validation: np.ndarray
    weight_validation: np.ndarray


def _prepare_anchor_split(
    varied_samples: Mapping[float, pd.DataFrame],
    *,
    validation_samples: Mapping[float, pd.DataFrame] | None,
    features: Sequence[str],
    max_events_per_anchor: int | None,
    max_validation_events_per_anchor: int | None,
    validation_fraction: float,
    seed: int,
) -> _AnchorSplit:
    """Prepare disjoint training and validation samples at every anchor."""
    if not varied_samples:
        raise ValueError("At least one non-nominal nuisance anchor is required.")
    if 0.0 in {float(value) for value in varied_samples}:
        raise ValueError(
            "The residual is identity at alpha=0; train it on nonzero anchors."
        )
    if validation_samples is None and not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one.")
    if validation_samples is not None:
        training_anchors = {float(value) for value in varied_samples}
        validation_anchors = {float(value) for value in validation_samples}
        if validation_anchors != training_anchors:
            raise ValueError(
                "Training and validation samples must have the same anchors."
            )

    rng = np.random.default_rng(seed)
    train_parts = []
    validation_parts = []
    n_anchors = len(varied_samples)

    def select_dataframe(
        dataframe: pd.DataFrame,
        maximum: int | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(dataframe) < 1:
            raise ValueError("Every nuisance anchor must contain events.")
        n_events = len(dataframe)
        if maximum is not None:
            n_events = min(n_events, int(maximum))
        selected = rng.choice(len(dataframe), size=n_events, replace=False)
        values = dataframe.iloc[selected][list(features)].to_numpy(
            dtype=np.float32
        )
        if "weight" in dataframe:
            weights = dataframe.iloc[selected]["weight"].to_numpy(
                dtype=np.float64
            )
        else:
            weights = np.ones(n_events, dtype=np.float64)
        if (
            not np.isfinite(weights).all()
            or np.any(weights < 0.0)
            or not weights.sum() > 0.0
        ):
            raise ValueError(
                "FNF shape training requires finite, non-negative weights "
                "with positive sum."
            )
        return values, weights

    def append_part(
        target: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        *,
        values: np.ndarray,
        weights: np.ndarray,
        indices: np.ndarray,
        alpha: float,
    ) -> None:
        local_weights = weights[indices].copy()
        local_weights /= local_weights.sum()
        local_weights /= n_anchors
        target.append(
            (
                values[indices],
                np.full(len(indices), float(alpha), dtype=np.float32),
                local_weights,
            )
        )

    for alpha, dataframe in sorted(
        varied_samples.items(), key=lambda item: float(item[0])
    ):
        if validation_samples is None and len(dataframe) < 2:
            raise ValueError(f"Anchor alpha={alpha} has fewer than two events.")
        values, weights = select_dataframe(
            dataframe, max_events_per_anchor
        )
        if validation_samples is None:
            order = rng.permutation(len(values))
            n_validation = max(
                1, int(round(validation_fraction * len(values)))
            )
            n_validation = min(n_validation, len(values) - 1)
            validation_indices = order[:n_validation]
            train_indices = order[n_validation:]
            append_part(
                train_parts,
                values=values,
                weights=weights,
                indices=train_indices,
                alpha=float(alpha),
            )
            append_part(
                validation_parts,
                values=values,
                weights=weights,
                indices=validation_indices,
                alpha=float(alpha),
            )
            continue

        append_part(
            train_parts,
            values=values,
            weights=weights,
            indices=np.arange(len(values)),
            alpha=float(alpha),
        )
        validation_dataframe = validation_samples[alpha]
        validation_values, validation_weights = select_dataframe(
            validation_dataframe,
            max_validation_events_per_anchor,
        )
        append_part(
            validation_parts,
            values=validation_values,
            weights=validation_weights,
            indices=np.arange(len(validation_values)),
            alpha=float(alpha),
        )

    def concatenate(parts):
        return tuple(
            np.concatenate([part[index] for part in parts])
            for index in range(3)
        )

    x_train, alpha_train, weight_train = concatenate(train_parts)
    x_validation, alpha_validation, weight_validation = concatenate(
        validation_parts
    )
    return _AnchorSplit(
        x_train=x_train,
        alpha_train=alpha_train,
        weight_train=weight_train,
        x_validation=x_validation,
        alpha_validation=alpha_validation,
        weight_validation=weight_validation,
    )


def _make_weighted_loader(
    x: np.ndarray,
    alpha: np.ndarray,
    weight: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    dataset = TensorDataset(
        torch.as_tensor(x, dtype=torch.float32),
        torch.as_tensor(alpha, dtype=torch.float32),
        torch.as_tensor(weight, dtype=torch.float64),
    )
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def train_factorizable_systematic_flow(
    process_name: str,
    *,
    base_density_pack: Mapping[str, Any],
    varied_samples: Mapping[float, pd.DataFrame],
    validation_samples: Mapping[float, pd.DataFrame] | None = None,
    features: Sequence[str],
    model_dir: str | Path,
    residual_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    device: torch.device,
    load_if_available: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    """Train the residual by conditional maximum likelihood at the anchors."""
    path = fnf_checkpoint_path(process_name, model_dir)
    if load_if_available and path.exists():
        print(f"Loading existing {process_name} FNF from {path}")
        return load_factorizable_systematic_flow(
            process_name,
            base_density_pack=base_density_pack,
            model_dir=model_dir,
            device=device,
            expected_features=features,
        )

    residual_config = dict(residual_config)
    training_config = dict(training_config)
    split = _prepare_anchor_split(
        varied_samples,
        validation_samples=validation_samples,
        features=features,
        max_events_per_anchor=training_config.get(
            "max_events_per_anchor"
        ),
        max_validation_events_per_anchor=training_config.get(
            "max_validation_events_per_anchor"
        ),
        validation_fraction=float(
            training_config.get("validation_fraction", 0.2)
        ),
        seed=seed,
    )
    train_loader = _make_weighted_loader(
        split.x_train,
        split.alpha_train,
        split.weight_train,
        batch_size=int(training_config.get("batch_size", 4096)),
        shuffle=True,
        seed=seed,
    )
    validation_loader = _make_weighted_loader(
        split.x_validation,
        split.alpha_validation,
        split.weight_validation,
        batch_size=int(training_config.get("batch_size", 4096)),
        shuffle=False,
        seed=seed + 1,
    )

    model = FactorizableSystematicFlow(
        base_density_pack,
        residual_config=residual_config,
    ).to(device)
    model.base_density.eval()
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training_config.get("learning_rate", 1.0e-4)),
        weight_decay=float(training_config.get("weight_decay", 1.0e-6)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training_config.get("lr_scheduler_factor", 0.2)),
        patience=int(training_config.get("lr_scheduler_patience", 3)),
        min_lr=float(training_config.get("min_learning_rate", 1.0e-7)),
    )

    n_epochs = int(training_config.get("n_epochs", 60))
    patience = int(training_config.get("patience", 12))
    gradient_clip = float(training_config.get("gradient_clip", 5.0))
    best_validation = np.inf
    best_state = None
    stale_epochs = 0
    history = {"train_nll": [], "validation_nll": [], "learning_rate": []}

    print(
        f"Training {process_name} FNF residual on "
        f"{len(split.x_train):,} anchor events; "
        f"validating on {len(split.x_validation):,}"
    )
    for epoch in range(1, n_epochs + 1):
        model.train()
        model.base_density.eval()
        train_numerator = 0.0
        train_denominator = 0.0
        for x_batch, alpha_batch, weight_batch in train_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            alpha_batch = alpha_batch.to(device, non_blocking=True)
            weight_batch = weight_batch.to(
                device, dtype=x_batch.dtype, non_blocking=True
            )
            event_nll = -model.log_prob_tensor(x_batch, alpha_batch)
            loss = torch.sum(weight_batch * event_nll) / torch.sum(weight_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, gradient_clip)
            optimizer.step()
            train_numerator += float(
                torch.sum(weight_batch * event_nll).detach().cpu()
            )
            train_denominator += float(weight_batch.sum().detach().cpu())

        model.eval()
        model.base_density.eval()
        validation_numerator = 0.0
        validation_denominator = 0.0
        with torch.no_grad():
            for x_batch, alpha_batch, weight_batch in validation_loader:
                x_batch = x_batch.to(device, non_blocking=True)
                alpha_batch = alpha_batch.to(device, non_blocking=True)
                weight_batch = weight_batch.to(
                    device, dtype=x_batch.dtype, non_blocking=True
                )
                event_nll = -model.log_prob_tensor(x_batch, alpha_batch)
                validation_numerator += float(
                    torch.sum(weight_batch * event_nll).cpu()
                )
                validation_denominator += float(weight_batch.sum().cpu())

        train_nll = train_numerator / train_denominator
        validation_nll = validation_numerator / validation_denominator
        scheduler.step(validation_nll)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history["train_nll"].append(train_nll)
        history["validation_nll"].append(validation_nll)
        history["learning_rate"].append(learning_rate)
        print(
            f"  epoch {epoch:03d}: train NLL = {train_nll:.5f}, "
            f"validation NLL = {validation_nll:.5f}, "
            f"lr = {learning_rate:.3e}"
        )

        if validation_nll < best_validation - 1.0e-5:
            best_validation = validation_nll
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.residual.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"  early stopping after {epoch} epochs")
                break

    if best_state is not None:
        model.residual.load_state_dict(best_state)
    model.eval()
    model.base_density.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "residual_state_dict": model.residual.state_dict(),
            "features": list(features),
            "residual_config": residual_config,
            "training_config": training_config,
            "history": history,
            "base_density_identifier": str(
                base_density_pack.get("identifier", "")
            ),
            "reference_flow_path": str(
                base_density_pack.get("reference_flow_path", "")
            ),
        },
        path,
    )
    print(f"Saved {process_name} FNF to {path}")
    return {
        "model": model,
        "features": list(features),
        "residual_config": residual_config,
        "training_config": training_config,
        "history": history,
        "path": path,
    }


@torch.no_grad()
def fnf_log_prob_x(
    fnf_pack: Mapping[str, Any],
    x: pd.DataFrame | np.ndarray,
    alpha: float | np.ndarray,
    *,
    batch_size: int = 65_536,
) -> np.ndarray:
    """Evaluate one FNF density in original feature coordinates."""
    model: FactorizableSystematicFlow = fnf_pack["model"]
    features = list(fnf_pack["features"])
    if isinstance(x, pd.DataFrame):
        values = x[features].to_numpy(dtype=np.float32)
    else:
        values = np.asarray(x, dtype=np.float32)
    alpha_values = np.asarray(alpha)
    alpha_is_scalar = alpha_values.ndim == 0 or alpha_values.size == 1
    if not alpha_is_scalar and len(alpha_values.reshape(-1)) != len(values):
        raise ValueError("A non-scalar alpha must match the number of events.")

    model.eval()
    chunks = []
    for start in range(0, len(values), int(batch_size)):
        stop = min(start + int(batch_size), len(values))
        x_tensor = torch.as_tensor(
            values[start:stop], dtype=torch.float32, device=model.device
        )
        if alpha_is_scalar:
            batch_alpha: float | np.ndarray = float(alpha_values.reshape(-1)[0])
        else:
            batch_alpha = alpha_values.reshape(-1)[start:stop]
        log_probability = model.log_prob_tensor(x_tensor, batch_alpha)
        chunks.append(log_probability.detach().cpu().numpy())
    return (
        np.concatenate(chunks)
        if chunks
        else np.empty(0, dtype=np.float32)
    )


@torch.no_grad()
def fnf_sample_x(
    fnf_pack: Mapping[str, Any],
    n: int,
    alpha: float,
    *,
    batch_size: int = 65_536,
) -> np.ndarray:
    """Draw FNF samples without retaining a full GPU-sized tensor."""
    model: FactorizableSystematicFlow = fnf_pack["model"]
    model.eval()
    chunks = []
    remaining = int(n)
    while remaining > 0:
        current = min(int(batch_size), remaining)
        chunks.append(
            model.sample_tensor(current, float(alpha)).detach().cpu().numpy()
        )
        remaining -= current
    if not chunks:
        return np.empty((0, len(fnf_pack["features"])), dtype=np.float32)
    return np.concatenate(chunks)


def analytic_reconstructed_mixture_density(
    x: np.ndarray,
    components: Sequence[tuple[float, np.ndarray, np.ndarray]],
    *,
    response_scale: Sequence[float],
    resolution: Sequence[float],
    alpha: float,
    fractional_scale_per_sigma: float = 0.1,
) -> np.ndarray:
    """Analytic reconstructed density for the Gaussian-mixture toy."""
    from scipy.stats import multivariate_normal

    values = np.asarray(x, dtype=np.float64)
    response_scale = np.asarray(response_scale, dtype=np.float64)
    resolution = np.asarray(resolution, dtype=np.float64)
    multiplier = 1.0 + fractional_scale_per_sigma * float(alpha)
    if multiplier <= 0.0:
        raise ValueError("The detector-response multiplier must be positive.")
    response = np.diag(multiplier * response_scale)
    resolution_covariance = np.diag(resolution**2.0)

    fractions = np.asarray(
        [component[0] for component in components], dtype=np.float64
    )
    fractions /= fractions.sum()
    density = np.zeros(len(values), dtype=np.float64)
    for fraction, (_, mean, covariance) in zip(fractions, components):
        reco_mean = response @ np.asarray(mean, dtype=np.float64)
        reco_covariance = (
            response
            @ np.asarray(covariance, dtype=np.float64)
            @ response.T
            + resolution_covariance
        )
        density += fraction * multivariate_normal(
            mean=reco_mean,
            cov=reco_covariance,
        ).pdf(values)
    return density


def analytic_selected_log_density(
    x: np.ndarray,
    components: Sequence[tuple[float, np.ndarray, np.ndarray]],
    *,
    response_scale: Sequence[float],
    resolution: Sequence[float],
    alpha: float,
    selection_efficiency: float,
    fractional_scale_per_sigma: float = 0.1,
    floor: float = 1.0e-300,
) -> np.ndarray:
    """Truth log density conditional on being in the fixed selected region."""
    if not 0.0 < float(selection_efficiency) <= 1.0:
        raise ValueError("selection_efficiency must be in (0, 1].")
    density = analytic_reconstructed_mixture_density(
        x,
        components,
        response_scale=response_scale,
        resolution=resolution,
        alpha=alpha,
        fractional_scale_per_sigma=fractional_scale_per_sigma,
    )
    return np.log(np.maximum(density, floor)) - math.log(
        float(selection_efficiency)
    )


def fnf_normalization_diagnostics(
    fnf_pack: Mapping[str, Any],
    alpha_values: Sequence[float],
    *,
    n_reference: int = 200_000,
    batch_size: int = 65_536,
) -> pd.DataFrame:
    """Check ``E_{p(x|0)}[p(x|alpha)/p(x|0)] = 1`` without rescaling."""
    from scipy.special import logsumexp

    if int(n_reference) < 2:
        raise ValueError("n_reference must be at least two.")
    nominal_sample = fnf_sample_x(
        fnf_pack, int(n_reference), 0.0, batch_size=batch_size
    )
    nominal_log_probability = fnf_log_prob_x(
        fnf_pack, nominal_sample, 0.0, batch_size=batch_size
    )
    rows = []
    for alpha in alpha_values:
        varied_log_probability = fnf_log_prob_x(
            fnf_pack,
            nominal_sample,
            float(alpha),
            batch_size=batch_size,
        )
        log_ratio = varied_log_probability - nominal_log_probability
        n_events = len(log_ratio)
        log_sum = float(logsumexp(log_ratio))
        log_sum_squared = float(logsumexp(2.0 * log_ratio))
        mean = math.exp(log_sum - math.log(n_events))
        second_moment = math.exp(
            log_sum_squared - math.log(n_events)
        )
        standard_error = math.sqrt(
            max(second_moment - mean**2, 0.0) / max(n_events - 1, 1)
        )
        effective_sample_size = math.exp(
            min(2.0 * log_sum - log_sum_squared, math.log(n_events))
        )
        rows.append(
            {
                "alpha": float(alpha),
                "mean density ratio": mean,
                "MC standard error": standard_error,
                "(mean - 1) / SE": (
                    (mean - 1.0) / standard_error
                    if standard_error > 0.0
                    else 0.0
                ),
                "ESS": effective_sample_size,
                "reference events": n_events,
            }
        )
    return pd.DataFrame(rows)


def log_quadratic_yield(
    alpha: float | np.ndarray,
    *,
    nominal: float,
    down: float,
    up: float,
) -> np.ndarray:
    """Positive smooth rate morph through the ``-1, 0, +1`` anchors."""
    nominal = float(nominal)
    down = float(down)
    up = float(up)
    if min(nominal, down, up) <= 0.0:
        raise ValueError("All yield anchors must be positive.")
    alpha_array = np.asarray(alpha, dtype=np.float64)
    log_up = math.log(up / nominal)
    log_down = math.log(down / nominal)
    linear = 0.5 * (log_up - log_down)
    quadratic = 0.5 * (log_up + log_down)
    return nominal * np.exp(
        linear * alpha_array + quadratic * alpha_array**2.0
    )


class FNFExtendedLikelihood:
    """Extended unbinned likelihood evaluated directly with two FNF densities.

    The shape terms are normalized by their invertible flow definitions.  The
    independently measured selected yields are interpolated through
    :func:`log_quadratic_yield`.  No HistFactory point-wise shape interpolation
    enters this likelihood.

    The numerical Asimov integral is evaluated on a finite sample from the
    reference density.  Therefore the mixture shape ratio is self-normalized
    on that integration support,

    ``R(x; mu, alpha) / mean_ref[R(x; mu, alpha)]``.

    This is the finite-sample implementation of the exact identity
    ``E_ref[R] = 1``.  It is one global expectation for each parameter point,
    not a fitted process-density correction or a pointwise normalization.
    """

    def __init__(
        self,
        *,
        signal_fnf: Mapping[str, Any],
        background_fnf: Mapping[str, Any],
        events: np.ndarray,
        weights: np.ndarray,
        signal_reference_ratio: np.ndarray,
        background_reference_ratio: np.ndarray,
        signal_yields: Mapping[str, float],
        background_yields: Mapping[str, float],
        alpha_constraint_sigma: float = 1.0,
        batch_size: int = 65_536,
    ) -> None:
        self.signal_model: FactorizableSystematicFlow = signal_fnf["model"]
        self.background_model: FactorizableSystematicFlow = background_fnf[
            "model"
        ]
        if self.signal_model.device != self.background_model.device:
            raise ValueError("Signal and background FNF models use different devices.")
        self.device = self.signal_model.device

        values = np.asarray(events, dtype=np.float32)
        event_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if values.ndim != 2 or len(values) != len(event_weights):
            raise ValueError("events and weights have incompatible shapes.")
        if not np.isfinite(event_weights).all() or np.any(event_weights < 0.0):
            raise ValueError("Asimov weights must be finite and non-negative.")
        self.events = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        self.weights = torch.as_tensor(
            event_weights, dtype=torch.float64, device=self.device
        )
        self.total_weight = torch.sum(self.weights)

        signal_ratio = np.asarray(
            signal_reference_ratio, dtype=np.float64
        ).reshape(-1)
        background_ratio = np.asarray(
            background_reference_ratio, dtype=np.float64
        ).reshape(-1)
        for name, ratio in [
            ("signal_reference_ratio", signal_ratio),
            ("background_reference_ratio", background_ratio),
        ]:
            if len(ratio) != len(values):
                raise ValueError(f"{name} is misaligned with events.")
            if not np.isfinite(ratio).all() or np.any(ratio <= 0.0):
                raise ValueError(f"{name} must be finite and strictly positive.")
        self.signal_reference_log_ratio = torch.as_tensor(
            np.log(signal_ratio),
            dtype=torch.float64,
            device=self.device,
        )
        self.background_reference_log_ratio = torch.as_tensor(
            np.log(background_ratio),
            dtype=torch.float64,
            device=self.device,
        )
        self.signal_yields = {
            key: float(signal_yields[key]) for key in ["down", "nominal", "up"]
        }
        self.background_yields = {
            key: float(background_yields[key])
            for key in ["down", "nominal", "up"]
        }
        self.alpha_constraint_sigma = float(alpha_constraint_sigma)
        self.batch_size = int(batch_size)
        if self.alpha_constraint_sigma <= 0.0:
            raise ValueError("alpha_constraint_sigma must be positive.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")

        self.signal_model.eval()
        self.background_model.eval()
        self.signal_model.base_density.eval()
        self.background_model.base_density.eval()
        with torch.no_grad():
            signal_nominal = []
            background_nominal = []
            for start in range(0, len(self.events), self.batch_size):
                stop = min(start + self.batch_size, len(self.events))
                event_batch = self.events[start:stop]
                signal_nominal.append(
                    self.signal_model.log_prob_tensor(event_batch, 0.0)
                    .detach()
                    .to(torch.float64)
                )
                background_nominal.append(
                    self.background_model.log_prob_tensor(event_batch, 0.0)
                    .detach()
                    .to(torch.float64)
                )
            self.signal_nominal_log_density = torch.cat(signal_nominal)
            self.background_nominal_log_density = torch.cat(
                background_nominal
            )

    @staticmethod
    def _yield_tensor(
        alpha: torch.Tensor,
        anchors: Mapping[str, float],
    ) -> torch.Tensor:
        nominal = torch.as_tensor(
            anchors["nominal"], dtype=alpha.dtype, device=alpha.device
        )
        log_up = torch.as_tensor(
            math.log(anchors["up"] / anchors["nominal"]),
            dtype=alpha.dtype,
            device=alpha.device,
        )
        log_down = torch.as_tensor(
            math.log(anchors["down"] / anchors["nominal"]),
            dtype=alpha.dtype,
            device=alpha.device,
        )
        linear = 0.5 * (log_up - log_down)
        quadratic = 0.5 * (log_up + log_down)
        return nominal * torch.exp(linear * alpha + quadratic * alpha**2.0)

    def _batch_log_intensity_ratio(
        self,
        start: int,
        stop: int,
        *,
        mu: torch.Tensor,
        alpha: torch.Tensor,
        signal_yield: torch.Tensor,
        background_yield: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``log(lambda(x; mu, alpha) / p_ref(x))`` on one batch."""
        event_batch = self.events[start:stop]
        signal_log_density = self.signal_model.log_prob_tensor(
            event_batch, alpha
        ).to(torch.float64)
        background_log_density = self.background_model.log_prob_tensor(
            event_batch, alpha
        ).to(torch.float64)
        signal_log_ratio = (
            self.signal_reference_log_ratio[start:stop]
            + signal_log_density
            - self.signal_nominal_log_density[start:stop]
        )
        background_log_ratio = (
            self.background_reference_log_ratio[start:stop]
            + background_log_density
            - self.background_nominal_log_density[start:stop]
        )

        signal_coefficient = torch.clamp(
            mu * signal_yield, min=torch.finfo(mu.dtype).tiny
        )
        background_coefficient = torch.clamp(
            background_yield, min=torch.finfo(mu.dtype).tiny
        )
        return torch.logaddexp(
            torch.log(signal_coefficient) + signal_log_ratio,
            torch.log(background_coefficient) + background_log_ratio,
        )

    @torch.no_grad()
    def _log_mixture_ratio_normalization(
        self, parameters: torch.Tensor
    ) -> torch.Tensor:
        """Estimate ``log E_ref[R(x; mu, alpha)]`` on this support."""
        mu, alpha = parameters[0], parameters[1]
        signal_yield = self._yield_tensor(alpha, self.signal_yields)
        background_yield = self._yield_tensor(alpha, self.background_yields)
        expected_yield = mu * signal_yield + background_yield
        log_expected_yield = torch.log(
            torch.clamp(
                expected_yield,
                min=torch.finfo(parameters.dtype).tiny,
            )
        )
        log_ratio_sum = torch.tensor(
            -torch.inf, dtype=torch.float64, device=self.device
        )
        for start in range(0, len(self.events), self.batch_size):
            stop = min(start + self.batch_size, len(self.events))
            log_intensity_ratio = self._batch_log_intensity_ratio(
                start,
                stop,
                mu=mu,
                alpha=alpha,
                signal_yield=signal_yield,
                background_yield=background_yield,
            )
            batch_log_sum = torch.logsumexp(
                log_intensity_ratio - log_expected_yield,
                dim=0,
            )
            log_ratio_sum = torch.logaddexp(log_ratio_sum, batch_log_sum)
        return log_ratio_sum - math.log(len(self.events))

    def _nll_tensor(self, parameters: torch.Tensor) -> torch.Tensor:
        if parameters.shape != (2,):
            raise ValueError("Expected parameters [mu, alpha].")
        mu, alpha = parameters[0], parameters[1]
        signal_yield = self._yield_tensor(alpha, self.signal_yields)
        background_yield = self._yield_tensor(alpha, self.background_yields)
        expected_yield = mu * signal_yield + background_yield
        log_expected_yield = torch.log(
            torch.clamp(
                expected_yield,
                min=torch.finfo(parameters.dtype).tiny,
            )
        )
        event_term = torch.zeros((), dtype=torch.float64, device=self.device)
        log_ratio_sum = torch.tensor(
            -torch.inf, dtype=torch.float64, device=self.device
        )
        for start in range(0, len(self.events), self.batch_size):
            stop = min(start + self.batch_size, len(self.events))
            weight_batch = self.weights[start:stop]
            log_intensity_ratio = self._batch_log_intensity_ratio(
                start,
                stop,
                mu=mu,
                alpha=alpha,
                signal_yield=signal_yield,
                background_yield=background_yield,
            )
            event_term = event_term + torch.sum(
                weight_batch * log_intensity_ratio
            )
            batch_log_sum = torch.logsumexp(
                log_intensity_ratio - log_expected_yield,
                dim=0,
            )
            log_ratio_sum = torch.logaddexp(log_ratio_sum, batch_log_sum)

        log_mixture_normalization = (
            log_ratio_sum - math.log(len(self.events))
        )
        constraint = (alpha / self.alpha_constraint_sigma) ** 2.0
        return (
            2.0
            * (
                expected_yield.to(torch.float64)
                - event_term
                + self.total_weight * log_mixture_normalization
            )
            + constraint
        )

    def model(self, parameters: Sequence[float] | np.ndarray) -> float:
        """Minuit-compatible array-call objective."""
        values = torch.as_tensor(
            np.asarray(parameters, dtype=np.float64),
            dtype=torch.float64,
            device=self.device,
        )
        with torch.no_grad():
            return float(self._nll_tensor(values).detach().cpu())

    def mixture_ratio_normalization(
        self, parameters: Sequence[float] | np.ndarray
    ) -> float:
        """Return the finite-support estimate of ``E_ref[R]``."""
        values = torch.as_tensor(
            np.asarray(parameters, dtype=np.float64),
            dtype=torch.float64,
            device=self.device,
        )
        return float(
            torch.exp(self._log_mixture_ratio_normalization(values))
            .detach()
            .cpu()
        )

    def model_grad(
        self, parameters: Sequence[float] | np.ndarray
    ) -> np.ndarray:
        """Memory-bounded automatic-differentiation gradient for Minuit.

        The Exercise 5 hNDE contains four large ratio members per process.
        Accumulating every event batch into one autograd graph would retain all
        intermediate activations.  Differentiate the rate/constraint term once
        and each event batch separately instead.
        """
        values = torch.tensor(
            np.asarray(parameters, dtype=np.float64),
            dtype=torch.float64,
            device=self.device,
            requires_grad=True,
        )
        mu, alpha = values[0], values[1]
        signal_yield = self._yield_tensor(alpha, self.signal_yields)
        background_yield = self._yield_tensor(alpha, self.background_yields)
        rate_and_constraint = 2.0 * (
            mu * signal_yield + background_yield
        ) + (alpha / self.alpha_constraint_sigma) ** 2.0
        gradient = torch.autograd.grad(rate_and_constraint, values)[0]
        with torch.no_grad():
            log_mixture_normalization = (
                self._log_mixture_ratio_normalization(values)
            )
            log_ratio_sum = (
                log_mixture_normalization + math.log(len(self.events))
            )

        for start in range(0, len(self.events), self.batch_size):
            stop = min(start + self.batch_size, len(self.events))
            weight_batch = self.weights[start:stop]

            batch_mu, batch_alpha = values[0], values[1]
            batch_signal_yield = self._yield_tensor(
                batch_alpha, self.signal_yields
            )
            batch_background_yield = self._yield_tensor(
                batch_alpha, self.background_yields
            )
            batch_expected_yield = (
                batch_mu * batch_signal_yield + batch_background_yield
            )
            batch_log_expected_yield = torch.log(
                torch.clamp(
                    batch_expected_yield,
                    min=torch.finfo(values.dtype).tiny,
                )
            )
            log_intensity_ratio = self._batch_log_intensity_ratio(
                start,
                stop,
                mu=batch_mu,
                alpha=batch_alpha,
                signal_yield=batch_signal_yield,
                background_yield=batch_background_yield,
            )
            event_objective = -2.0 * torch.sum(
                weight_batch * log_intensity_ratio
            )

            # Differentiate log(mean R) without retaining all event batches:
            # the detached coefficients are the global softmax weights of
            # log R, so this surrogate has exactly the required gradient.
            log_shape_ratio = (
                log_intensity_ratio - batch_log_expected_yield
            )
            normalization_weight = torch.exp(
                log_shape_ratio.detach() - log_ratio_sum
            )
            normalization_objective = (
                2.0
                * self.total_weight
                * torch.sum(normalization_weight * log_shape_ratio)
            )
            gradient = gradient + torch.autograd.grad(
                event_objective + normalization_objective,
                values,
            )[0]

        return gradient.detach().cpu().numpy().astype(np.float64)
