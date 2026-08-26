"""Audited Python compatibility backend for the ``sbibm`` Lotka--Volterra task.

The public ``sbibm==1.1.0`` task delegates ODE integration to the legacy
``diffeqtorch``/PyJulia stack.  This module replaces only that numerical
solver.  It retains the official equations, initial populations, four-
parameter LogNormal prior, ten summary times, and LogNormal observation law.

Integration is performed in log-population coordinates, which is the exact
positive-state change of variables for the same Lotka--Volterra equations.
Before use, the vectorized fourth-order Runge--Kutta result is compared with a
high-accuracy DOP853 integration throughout the crossed four-sigma prior box.
Official observations and reference-posterior samples remain untouched.
"""

from __future__ import annotations

import os
import types
from typing import Any

import numpy as np
import torch
from scipy.integrate import solve_ivp
from sbibm.tasks.simulator import Simulator


BACKEND_ID = "sbibm_lotka_volterra_python_logrk4_dt0p0025_v1"
INITIAL_POPULATIONS = np.asarray([30.0, 1.0], dtype=np.float64)
OBSERVATION_TIMES = np.arange(0.0, 19.0, 2.1, dtype=np.float64)
RK4_STEP = 0.0025
LOGNORMAL_SCALE = 0.1
STATE_CLIP_MIN = 1.0e-10
STATE_CLIP_MAX = 1.0e4
AUDIT_TOLERANCE = 5.0e-5


def _as_parameter_array(parameters: Any) -> np.ndarray:
    if torch.is_tensor(parameters):
        parameters = parameters.detach().cpu().numpy()
    array = np.asarray(parameters, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError(
            "Lotka--Volterra parameters must have shape "
            "(num_simulations, 4) for (alpha, beta, gamma, delta)."
        )
    if not np.isfinite(array).all() or np.any(array <= 0.0):
        raise ValueError(
            "Lotka--Volterra parameters must be finite and strictly positive."
        )
    return array


def _log_state_rhs(log_states: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    alpha, beta, gamma, delta = parameters.T
    prey = np.exp(log_states[:, 0])
    predator = np.exp(log_states[:, 1])
    return np.column_stack((
        alpha - beta * predator,
        -gamma + delta * prey,
    ))


def _log_rk4_chunk(parameters: np.ndarray) -> np.ndarray:
    num_simulations = len(parameters)
    log_states = np.tile(np.log(INITIAL_POPULATIONS), (num_simulations, 1))
    summaries = np.empty(
        (num_simulations, 2 * len(OBSERVATION_TIMES)), dtype=np.float64
    )
    summaries[:, 0] = INITIAL_POPULATIONS[0]
    summaries[:, len(OBSERVATION_TIMES)] = INITIAL_POPULATIONS[1]
    save_steps = np.rint(OBSERVATION_TIMES / RK4_STEP).astype(np.int64)
    next_save = 1

    for step in range(1, int(save_steps[-1]) + 1):
        k1 = _log_state_rhs(log_states, parameters)
        k2 = _log_state_rhs(log_states + 0.5 * RK4_STEP * k1, parameters)
        k3 = _log_state_rhs(log_states + 0.5 * RK4_STEP * k2, parameters)
        k4 = _log_state_rhs(log_states + RK4_STEP * k3, parameters)
        log_states += (RK4_STEP / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.isfinite(log_states).all():
            raise FloatingPointError(
                "The Python Lotka--Volterra integrator produced non-finite values."
            )
        if next_save < len(save_steps) and step == int(save_steps[next_save]):
            summaries[:, next_save] = np.exp(log_states[:, 0])
            summaries[:, len(save_steps) + next_save] = np.exp(log_states[:, 1])
            next_save += 1

    if next_save != len(save_steps):
        raise RuntimeError(
            "The Lotka--Volterra integrator did not visit every observation time."
        )
    if not np.isfinite(summaries).all() or np.any(summaries <= 0.0):
        raise FloatingPointError(
            "The Python Lotka--Volterra integrator returned invalid populations."
        )
    return summaries


def lotka_volterra_populations(
    parameters: Any,
    *,
    chunk_size: int = 10_000,
) -> np.ndarray:
    """Return prey then predator populations at days 0, 2.1, ..., 18.9."""

    array = _as_parameter_array(parameters)
    outputs = []
    for start in range(0, len(array), int(chunk_size)):
        outputs.append(_log_rk4_chunk(array[start : start + int(chunk_size)]))
    return np.concatenate(outputs, axis=0)


def _dop853_populations(parameters: np.ndarray) -> np.ndarray:
    outputs = []
    initial_log_state = np.log(INITIAL_POPULATIONS)
    for alpha, beta, gamma, delta in parameters:
        def rhs(_time, log_state):
            prey, predator = np.exp(log_state)
            return (
                alpha - beta * predator,
                -gamma + delta * prey,
            )

        solution = solve_ivp(
            rhs,
            (float(OBSERVATION_TIMES[0]), float(OBSERVATION_TIMES[-1])),
            initial_log_state,
            method="DOP853",
            t_eval=OBSERVATION_TIMES,
            rtol=1.0e-11,
            atol=1.0e-12,
            max_step=0.005,
        )
        if not solution.success or solution.y.shape != (2, len(OBSERVATION_TIMES)):
            raise RuntimeError(
                f"High-accuracy Lotka--Volterra audit solve failed: {solution.message}"
            )
        populations = np.exp(solution.y)
        outputs.append(np.concatenate((populations[0], populations[1])))
    return np.asarray(outputs, dtype=np.float64)


def _observation_locations(populations: np.ndarray) -> np.ndarray:
    return np.log(np.clip(populations, STATE_CLIP_MIN, STATE_CLIP_MAX))


def audit_lotka_volterra_backend() -> dict[str, Any]:
    """Audit the observation locations on center and four-sigma prior corners."""

    prior_location = np.asarray([-0.125, -3.0, -0.125, -3.0], dtype=np.float64)
    prior_scale = 0.5
    corner_signs = np.asarray(
        np.meshgrid(*[[-1.0, 1.0]] * 4), dtype=np.float64
    ).T.reshape(-1, 4)
    audit_parameters = np.vstack((
        np.exp(prior_location),
        np.exp(prior_location + 4.0 * prior_scale * corner_signs),
    ))
    rk4 = lotka_volterra_populations(audit_parameters)
    reference = _dop853_populations(audit_parameters)
    max_abs_log_location_error = float(np.max(np.abs(
        _observation_locations(rk4) - _observation_locations(reference)
    )))
    if max_abs_log_location_error > AUDIT_TOLERANCE:
        raise RuntimeError(
            "Python Lotka--Volterra log-RK4 audit failed: "
            f"max |delta log population|={max_abs_log_location_error:.3e} "
            f"> {AUDIT_TOLERANCE:.3e}."
        )
    return {
        "backend": BACKEND_ID,
        "rk4_step_days": RK4_STEP,
        "audit_points": int(len(audit_parameters)),
        "audit_prior_extent_sigma": 4.0,
        "audit_max_abs_log_location_error": max_abs_log_location_error,
        "audit_tolerance": AUDIT_TOLERANCE,
        "largest_fraction_of_observation_sigma": (
            max_abs_log_location_error / LOGNORMAL_SCALE
        ),
    }


def _python_get_simulator(self, max_calls=None):
    def simulator(parameters):
        populations = lotka_volterra_populations(parameters)
        locations = torch.log(torch.as_tensor(
            np.clip(populations, STATE_CLIP_MIN, STATE_CLIP_MAX),
            dtype=torch.float32,
        ))
        return torch.distributions.LogNormal(
            loc=locations,
            scale=torch.full_like(locations, LOGNORMAL_SCALE),
        ).sample().to(torch.float32)

    return Simulator(task=self, simulator=simulator, max_calls=max_calls)


def install_lotka_volterra_python_backend(sbibm_module) -> dict[str, Any]:
    """Patch only newly constructed default Lotka--Volterra tasks."""

    current_get_task = sbibm_module.get_task
    if getattr(current_get_task, "_ex9b_lotka_volterra_python_backend", False):
        diagnostics = audit_lotka_volterra_backend()
        os.environ["EX9B_SIMULATOR_BACKEND"] = BACKEND_ID
        return diagnostics

    def get_task(task_name, *args, **kwargs):
        task = current_get_task(task_name, *args, **kwargs)
        if str(task_name).lower() == "lotka_volterra":
            if (
                task.summary != "subsample"
                or task.dim_data != 20
                or not np.isclose(float(task.days), 20.0)
                or not np.isclose(float(task.saveat), 0.1)
            ):
                raise ValueError(
                    "The Exercise-9b Python Lotka--Volterra backend supports only "
                    "the official default sbibm task configuration."
                )
            task.get_simulator = types.MethodType(_python_get_simulator, task)
            task.ex9b_simulator_backend = BACKEND_ID
        return task

    get_task._ex9b_lotka_volterra_python_backend = True
    get_task._ex9b_original_get_task = current_get_task
    sbibm_module.get_task = get_task

    diagnostics = audit_lotka_volterra_backend()
    smoke_task = sbibm_module.get_task("lotka_volterra")
    smoke_parameters = torch.tensor(
        [[np.exp(-0.125), np.exp(-3.0), np.exp(-0.125), np.exp(-3.0)]],
        dtype=torch.float32,
    )
    smoke_data = smoke_task.get_simulator(max_calls=1)(smoke_parameters)
    if smoke_data.shape != torch.Size([1, 20]):
        raise RuntimeError(
            f"Unexpected Python Lotka--Volterra smoke shape: {tuple(smoke_data.shape)}"
        )
    if not bool(torch.isfinite(smoke_data).all() and (smoke_data > 0.0).all()):
        raise FloatingPointError(
            "Python Lotka--Volterra smoke simulation returned invalid data."
        )
    diagnostics["smoke_shape"] = list(smoke_data.shape)
    diagnostics["smoke_positive"] = True
    os.environ["EX9B_SIMULATOR_BACKEND"] = BACKEND_ID
    return diagnostics
