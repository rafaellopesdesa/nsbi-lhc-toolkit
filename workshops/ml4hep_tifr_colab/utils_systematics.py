"""Reference-normalized systematic morphs for Exercise 8.

The generic ``normplusshape`` implementation interpolates the total yield and
the event-wise shape ratio independently.  Even when the up/down shape anchors
are normalized, a nonlinear point-wise interpolation need not remain
normalized between the anchors.  This module supplies the Exercise 8
reference-sampled Asimov model, for which the interpolated shape is divided by
its partition function on the reference support at every nuisance value.

This helper is deliberately separate from the common tutorial model so that
the algorithms used by the other exercises are unchanged.
"""

from __future__ import annotations

from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np

from nsbi_common_utils.models.sbi_parametric_model import (
    _calculate_combined_var,
    sbi_parametric_model,
)


class ReferenceNormalizedSystematicsModel:
    """Wrap a one-channel reference-sampled unbinned systematic model.

    The nominal process ratios ``r_c`` must have unit arithmetic mean on the
    reference support.  For each process and nuisance point, the raw
    interpolated conditional-shape morph ``g_c`` is replaced by

    ``g_c / mean_reference(r_c * g_c)``.

    The resulting conditional shape integrates to one for every nuisance
    value, while the independently interpolated ``hi_data``/``lo_data``
    factors retain sole control of the selected yield.
    """

    def __init__(self, workspace, measurement_to_fit):
        self.raw_model = sbi_parametric_model(
            workspace=workspace,
            measurement_to_fit=measurement_to_fit,
        )
        if self.raw_model.channels_binned:
            raise ValueError(
                "ReferenceNormalizedSystematicsModel supports no binned "
                "channels."
            )
        if len(self.raw_model.channels_unbinned) != 1:
            raise ValueError(
                "ReferenceNormalizedSystematicsModel requires exactly one "
                "unbinned channel."
            )

        self.list_parameters = list(self.raw_model.list_parameters)
        self.initial_parameter_values = (
            self.raw_model.initial_parameter_values
        )
        self.num_unconstrained_param = (
            self.raw_model.num_unconstrained_param
        )
        self._model_data = self.raw_model._model_data
        self._jit_nll, self._jit_value_and_grad, self._jit_many = (
            self._build_jit_functions()
        )

        nominal_integrals = np.asarray(
            jnp.mean(self._model_data["ratios"], axis=1),
            dtype=float,
        )
        if not np.allclose(
            nominal_integrals, 1.0, rtol=0.0, atol=1.0e-10
        ):
            raise ValueError(
                "Nominal process/reference ratios must have unit mean on "
                "the Asimov reference support. Got "
                f"{nominal_integrals.tolist()}."
            )

    def get_model_parameters(self):
        """Return parameter names and initial values in fit order."""
        return self.list_parameters, self.initial_parameter_values

    def _build_jit_functions(self):
        num_unconstrained = self.num_unconstrained_param
        has_systematics = self.raw_model.has_normplusshape
        batched_variation = jax.vmap(
            _calculate_combined_var, in_axes=(None, 0, 0)
        )

        def nll(param_vec, data):
            nuisance_parameters = param_vec[num_unconstrained:]
            norm_modifiers = jnp.prod(
                jnp.where(
                    data["norm_matrix"],
                    param_vec[None, :],
                    1.0,
                ),
                axis=1,
            )

            if has_systematics:
                yield_variations = batched_variation(
                    nuisance_parameters,
                    data["tot_up_unbinned"],
                    data["tot_dn_unbinned"],
                )
                raw_shape_variations = batched_variation(
                    nuisance_parameters,
                    data["var_up_unbinned"],
                    data["var_dn_unbinned"],
                )
            else:
                yield_variations = jnp.ones_like(
                    data["unbinned_total"]
                )
                raw_shape_variations = jnp.ones_like(data["ratios"])

            # The support points are iid from the learned reference density.
            # Their arithmetic mean is therefore the empirical reference
            # integral used to normalize each process's conditional shape.
            shape_partition = jnp.mean(
                data["ratios"] * raw_shape_variations,
                axis=1,
            )
            shape_variations = (
                raw_shape_variations / shape_partition[:, None]
            )

            expected_rate = jnp.sum(
                norm_modifiers[:, None]
                * data["unbinned_total"]
                * yield_variations,
                axis=0,
            )
            differential_rate_over_reference = jnp.sum(
                norm_modifiers[:, None]
                * data["unbinned_total"]
                * yield_variations
                * data["ratios"]
                * shape_variations,
                axis=0,
            )

            rate_term = -2.0 * jnp.sum(
                data["expected_rate"] * jnp.log(expected_rate)
                - expected_rate
            )
            event_term = -2.0 * jnp.sum(
                data["weights"]
                * (
                    jnp.log(differential_rate_over_reference)
                    - jnp.log(expected_rate)
                )
            )
            constraint_term = jnp.sum(nuisance_parameters**2)
            return rate_term + event_term + constraint_term

        return (
            jax.jit(nll),
            jax.jit(jax.value_and_grad(nll, argnums=0)),
            jax.jit(jax.vmap(nll, in_axes=(0, None))),
        )

    def model(self, param_array):
        """Evaluate the reference-normalized ``-2 log L``."""
        parameters = jnp.asarray(param_array)
        return self._jit_nll(parameters, self._model_data)

    def model_grad(self, param_array):
        """Evaluate the gradient of the reference-normalized likelihood."""
        parameters = jnp.asarray(param_array)
        _, gradient = self._jit_value_and_grad(
            parameters, self._model_data
        )
        return np.asarray(gradient)

    def model_many(
        self,
        parameter_points: Iterable[Iterable[float]],
        batch_size: int = 4,
    ):
        """Evaluate many parameter points in bounded-memory JAX batches."""
        points = np.asarray(parameter_points, dtype=float)
        if points.ndim != 2 or points.shape[1] != len(self.list_parameters):
            raise ValueError(
                "parameter_points must have shape "
                f"(n_points, {len(self.list_parameters)})."
            )
        if len(points) == 0:
            return np.empty(0, dtype=float)

        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")

        values = []
        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            valid_size = len(batch)
            if valid_size < batch_size:
                padding = np.repeat(
                    batch[-1:, :],
                    batch_size - valid_size,
                    axis=0,
                )
                batch = np.concatenate([batch, padding], axis=0)
            batch_values = np.asarray(
                self._jit_many(jnp.asarray(batch), self._model_data),
                dtype=float,
            )
            values.append(batch_values[:valid_size])
        return np.concatenate(values)
