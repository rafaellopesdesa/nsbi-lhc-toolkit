"""Audited Python compatibility backend for the ``sbibm`` SIR task.

``sbibm==1.1.0`` delegates SIR integration to the unmaintained
``diffeqtorch``/PyJulia stack.  Current Colab runtimes no longer provide a
compatible Julia/SciML environment.  This module changes only the numerical
ODE implementation: it retains the official SIR equations, initial state,
parameter prior, ten observation times, and Binomial observation model.

The vectorized fourth-order Runge--Kutta implementation is checked against
SciPy's high-accuracy DOP853 solver before it is installed.  Official sbibm
observations and official reference-posterior samples continue to be loaded
from the package without modification.
"""

from __future__ import annotations

import os
import types
from typing import Any

import numpy as np
import torch
from scipy.integrate import solve_ivp
from sbibm.tasks.simulator import Simulator


BACKEND_ID = "sbibm_sir_python_rk4_dt0p1_v1"
POPULATION = 1_000_000.0
INITIAL_INFECTED = 1.0
INITIAL_RECOVERED = 0.0
OBSERVATION_TIMES = np.arange(0.0, 154.0, 17.0, dtype=np.float64)
RK4_STEP = 0.1
AUDIT_TOLERANCE = 5.0e-7


def _as_parameter_array(parameters: Any) -> np.ndarray:
    if torch.is_tensor(parameters):
        parameters = parameters.detach().cpu().numpy()
    array = np.asarray(parameters, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(
            "SIR parameters must have shape (num_simulations, 2) for (beta, gamma)."
        )
    if not np.isfinite(array).all() or np.any(array <= 0.0):
        raise ValueError("SIR beta and gamma must be finite and strictly positive.")
    return array


def _sir_rhs(states: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    susceptible = states[:, 0]
    infected = states[:, 1]
    infection = beta * susceptible * infected / POPULATION
    return np.column_stack((
        -infection,
        infection - gamma * infected,
        gamma * infected,
    ))


def _rk4_chunk(parameters: np.ndarray) -> np.ndarray:
    num_simulations = len(parameters)
    beta = parameters[:, 0]
    gamma = parameters[:, 1]
    states = np.empty((num_simulations, 3), dtype=np.float64)
    states[:, 0] = POPULATION - INITIAL_INFECTED - INITIAL_RECOVERED
    states[:, 1] = INITIAL_INFECTED
    states[:, 2] = INITIAL_RECOVERED

    probabilities = np.empty(
        (num_simulations, len(OBSERVATION_TIMES)), dtype=np.float64
    )
    probabilities[:, 0] = INITIAL_INFECTED / POPULATION
    save_steps = np.rint(OBSERVATION_TIMES / RK4_STEP).astype(np.int64)
    next_save = 1

    for step in range(1, int(save_steps[-1]) + 1):
        k1 = _sir_rhs(states, beta, gamma)
        k2 = _sir_rhs(states + 0.5 * RK4_STEP * k1, beta, gamma)
        k3 = _sir_rhs(states + 0.5 * RK4_STEP * k2, beta, gamma)
        k4 = _sir_rhs(states + RK4_STEP * k3, beta, gamma)
        states += (RK4_STEP / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if next_save < len(save_steps) and step == int(save_steps[next_save]):
            probabilities[:, next_save] = states[:, 1] / POPULATION
            next_save += 1

    if next_save != len(save_steps):
        raise RuntimeError("The SIR integrator did not visit every observation time.")
    if not np.isfinite(states).all() or not np.isfinite(probabilities).all():
        raise FloatingPointError("The Python SIR integrator produced non-finite values.")
    mass_error = float(np.max(np.abs(states.sum(axis=1) - POPULATION)))
    if mass_error > 1.0e-6 * POPULATION:
        raise RuntimeError(f"The SIR integrator failed population conservation: {mass_error}")
    if np.min(probabilities) < -1.0e-8 or np.max(probabilities) > 1.0 + 1.0e-8:
        raise RuntimeError("The SIR integrator produced invalid infection probabilities.")
    return np.clip(probabilities, 0.0, 1.0)


def sir_infected_probabilities(
    parameters: Any,
    *,
    chunk_size: int = 10_000,
) -> np.ndarray:
    """Return infected-population fractions at days 0, 17, ..., 153."""

    array = _as_parameter_array(parameters)
    outputs = []
    for start in range(0, len(array), int(chunk_size)):
        outputs.append(_rk4_chunk(array[start : start + int(chunk_size)]))
    return np.concatenate(outputs, axis=0)


def _dop853_probabilities(parameters: np.ndarray) -> np.ndarray:
    outputs = []
    initial_state = np.asarray(
        [POPULATION - INITIAL_INFECTED, INITIAL_INFECTED, INITIAL_RECOVERED],
        dtype=np.float64,
    )
    for beta, gamma in parameters:
        def rhs(_time, state):
            susceptible, infected, _recovered = state
            infection = beta * susceptible * infected / POPULATION
            return (-infection, infection - gamma * infected, gamma * infected)

        solution = solve_ivp(
            rhs,
            (float(OBSERVATION_TIMES[0]), float(OBSERVATION_TIMES[-1])),
            initial_state,
            method="DOP853",
            t_eval=OBSERVATION_TIMES,
            rtol=1.0e-11,
            atol=1.0e-12,
        )
        if not solution.success or solution.y.shape != (3, len(OBSERVATION_TIMES)):
            raise RuntimeError(f"High-accuracy SIR audit solve failed: {solution.message}")
        outputs.append(solution.y[1] / POPULATION)
    return np.asarray(outputs, dtype=np.float64)


def audit_sir_backend() -> dict[str, Any]:
    """Validate RK4 accuracy on central and crossed two-sigma prior points."""

    beta_center, gamma_center = 0.4, 0.125
    beta_low, beta_high = (
        np.exp(np.log(beta_center) - 2.0 * 0.5),
        np.exp(np.log(beta_center) + 2.0 * 0.5),
    )
    gamma_low, gamma_high = (
        np.exp(np.log(gamma_center) - 2.0 * 0.2),
        np.exp(np.log(gamma_center) + 2.0 * 0.2),
    )
    audit_parameters = np.asarray([
        [beta_center, gamma_center],
        [beta_low, gamma_low],
        [beta_high, gamma_high],
        [beta_high, gamma_low],
        [beta_low, gamma_high],
    ], dtype=np.float64)
    rk4 = sir_infected_probabilities(audit_parameters)
    reference = _dop853_probabilities(audit_parameters)
    max_abs_error = float(np.max(np.abs(rk4 - reference)))
    if max_abs_error > AUDIT_TOLERANCE:
        raise RuntimeError(
            "Python SIR RK4 audit failed: "
            f"max |delta p|={max_abs_error:.3e} > {AUDIT_TOLERANCE:.3e}."
        )
    return {
        "backend": BACKEND_ID,
        "rk4_step_days": RK4_STEP,
        "audit_points": int(len(audit_parameters)),
        "audit_max_abs_probability_error": max_abs_error,
        "audit_tolerance": AUDIT_TOLERANCE,
        "largest_expected_count_shift": 1000.0 * max_abs_error,
    }


def _python_get_simulator(self, max_calls=None):
    def simulator(parameters):
        probabilities = sir_infected_probabilities(parameters)
        probability_tensor = torch.as_tensor(probabilities, dtype=torch.float32)
        return torch.distributions.Binomial(
            total_count=float(self.total_count),
            probs=probability_tensor.clamp(0.0, 1.0),
        ).sample().to(torch.float32)

    return Simulator(task=self, simulator=simulator, max_calls=max_calls)


def install_sir_python_backend(sbibm_module) -> dict[str, Any]:
    """Patch only newly constructed ``sir`` tasks and run a strict preflight."""

    current_get_task = sbibm_module.get_task
    if getattr(current_get_task, "_ex9b_sir_python_backend", False):
        diagnostics = audit_sir_backend()
        os.environ["EX9B_SIMULATOR_BACKEND"] = BACKEND_ID
        return diagnostics

    def get_task(task_name, *args, **kwargs):
        task = current_get_task(task_name, *args, **kwargs)
        if str(task_name).lower() == "sir":
            task.get_simulator = types.MethodType(_python_get_simulator, task)
            task.ex9b_simulator_backend = BACKEND_ID
        return task

    get_task._ex9b_sir_python_backend = True
    get_task._ex9b_original_get_task = current_get_task
    sbibm_module.get_task = get_task

    diagnostics = audit_sir_backend()
    smoke_task = sbibm_module.get_task("sir")
    smoke_parameters = torch.tensor([[0.4, 0.125]], dtype=torch.float32)
    smoke_data = smoke_task.get_simulator(max_calls=1)(smoke_parameters)
    if smoke_data.shape != torch.Size([1, 10]):
        raise RuntimeError(f"Unexpected Python SIR smoke shape: {tuple(smoke_data.shape)}")
    if not bool(torch.isfinite(smoke_data).all()):
        raise FloatingPointError("Python SIR smoke simulation returned non-finite data.")
    diagnostics["smoke_shape"] = list(smoke_data.shape)
    diagnostics["smoke_integer_counts"] = bool(
        torch.equal(smoke_data, torch.round(smoke_data))
    )
    os.environ["EX9B_SIMULATOR_BACKEND"] = BACKEND_ID
    return diagnostics
