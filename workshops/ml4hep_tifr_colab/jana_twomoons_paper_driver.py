"""Pinned legacy-BayesFlow training driver for the JANA Two Moons study.

This module deliberately mirrors
``bayesflow-org/JANA-Paper/experiments/two_moons/jana.py`` at commit
6cbbc94faf0aa85147986f7f9516d13a52551bd4.  It is launched by
Exercise_9a_JANA_TwoMoons.ipynb in an isolated Python 3.11 environment with
the paper's TensorFlow and BayesFlow versions.

The driver trains two systems on one fixed simulation bank:

1. the original jointly optimized JANA posterior/likelihood amortizer; and
2. topology-matched posterior and likelihood amortizers trained separately,
   which provide the frozen proposals for the hybrid CE correction.

The second system changes only the optimizer partition.  It retains the
paper's simulator, prior, flow definitions, batch size, epoch count, learning
rate, cosine schedule, gradient clipping, and BayesFlow implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
from pathlib import Path

# The published script explicitly disabled GPU visibility.  Keeping this line
# before importing TensorFlow makes the JANA part reproduce the CPU campaign.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

from bayesflow.amortizers import (
    AmortizedLikelihood,
    AmortizedPosterior,
    AmortizedPosteriorLikelihood,
)
from bayesflow.networks import InvertibleNetwork
from bayesflow.simulation import GenerativeModel, Prior, Simulator
from bayesflow.trainers import Trainer


JANA_PAPER_COMMIT = "6cbbc94faf0aa85147986f7f9516d13a52551bd4"
BAYESFLOW_COMMIT = "153dfefadd347717b7aeb9c4872a4b51ac04e83c"
OBSERVATION = np.array([[0.0, 0.0]], dtype=np.float32)
PRIOR_LOW = np.array([-2.0, -2.0], dtype=np.float32)
PRIOR_HIGH = np.array([2.0, 2.0], dtype=np.float32)
MEAN_RADIUS = 1.0
SD_RADIUS = 0.1
BASE_OFFSET = 1.0


def seed_everything(seed: int) -> None:
    """Add deterministic seeds without changing the published objective."""

    random.seed(int(seed))
    np.random.seed(int(seed))
    tf.random.set_seed(int(seed))


def prior_fun() -> np.ndarray:
    """Exact prior from the JANA paper's Wiqvist Two Moons script."""

    return np.random.uniform(low=PRIOR_LOW, high=PRIOR_HIGH)


def simulator_numpy(theta: np.ndarray) -> np.ndarray:
    """Exact Wiqvist simulator used in the main-paper comparison."""

    theta = np.asarray(theta).reshape(-1)
    angle = np.array(np.pi * (np.random.random(1) - 0.5))
    radius = MEAN_RADIUS + np.random.normal(loc=0.0, scale=1.0, size=1) * SD_RADIUS
    point = np.array(
        [radius * np.cos(angle) + BASE_OFFSET, radius * np.sin(angle)]
    )
    rotation = np.array([-np.pi / 4.0])
    cosine = np.cos(rotation)
    sine = np.sin(rotation)
    z0 = cosine * theta[0] - sine * theta[1]
    z1 = sine * theta[0] + cosine * theta[1]
    return point + np.array([-np.abs(z0), z1])


def simulator_numpy_batched(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta)
    samples = np.zeros((len(theta), theta.shape[1]), dtype=np.float64)
    for index in range(len(theta)):
        samples[index] = simulator_numpy(theta[index]).reshape(-1)
    return samples


def analytic_posterior_numpy(
    observation: np.ndarray, n_samples: int
) -> np.ndarray:
    """Analytical sampler copied from the paper repository."""

    observation = np.asarray(observation).reshape(-1)
    angle = -np.pi / 4.0
    cosine = np.cos(-angle)
    sine = np.sin(-angle)
    theta = np.zeros((int(n_samples), 2), dtype=np.float64)
    for index in range(int(n_samples)):
        point = simulator_numpy(np.zeros(2))
        q = np.zeros(2)
        q[0] = point[0] - observation[0]
        q[1] = observation[1] - point[1]
        if np.random.rand() < 0.5:
            q[0] = -q[0]
        theta[index, 0] = cosine * q[0] - sine * q[1]
        theta[index, 1] = sine * q[0] + cosine * q[1]
    return theta


def configure_joint(forward_dict: dict) -> dict:
    """Exact configurator from ``experiments/two_moons/jana.py``."""

    return {
        "posterior_inputs": {
            "direct_conditions": forward_dict["sim_data"].astype(np.float32),
            "parameters": forward_dict["prior_draws"].astype(np.float32),
        },
        "likelihood_inputs": {
            "observables": forward_dict["sim_data"].astype(np.float32),
            "conditions": forward_dict["prior_draws"].astype(np.float32),
        },
    }


def configure_posterior(forward_dict: dict) -> dict:
    values = configure_joint(forward_dict)
    return values["posterior_inputs"]


def configure_likelihood(forward_dict: dict) -> dict:
    values = configure_joint(forward_dict)
    return values["likelihood_inputs"]


def make_posterior_network() -> InvertibleNetwork:
    # Exact main-paper topology: four all-spline coupling layers.
    return InvertibleNetwork(
        num_params=2,
        num_coupling_layers=4,
        coupling_design="spline",
        permutation="learnable",
    )


def make_likelihood_network() -> InvertibleNetwork:
    # Exact main-paper topology: affine/spline/affine/spline/affine.
    return InvertibleNetwork(
        num_params=2,
        num_coupling_layers=5,
        coupling_design="interleaved",
        permutation="learnable",
    )


def make_generator() -> GenerativeModel:
    return GenerativeModel(
        prior=Prior(prior_fun=prior_fun),
        simulator=Simulator(batch_simulator_fun=simulator_numpy_batched),
    )


def save_history(history, path: Path) -> None:
    if isinstance(history, pd.DataFrame):
        history.to_csv(path, index=False)
        return
    try:
        pd.DataFrame(history).to_csv(path, index=False)
    except Exception:
        path.with_suffix(".json").write_text(
            json.dumps(history, default=lambda value: np.asarray(value).tolist(), indent=2)
        )


def sample_posterior(amortizer, contexts: np.ndarray, n_samples: int) -> np.ndarray:
    values = amortizer.sample(
        {"direct_conditions": np.asarray(contexts, dtype=np.float32)},
        int(n_samples),
    )
    values = np.asarray(values, dtype=np.float32)
    if len(np.atleast_2d(contexts)) == 1:
        return values.reshape(int(n_samples), 2)
    return values.reshape(len(contexts), int(n_samples), 2)


def sample_likelihood(amortizer, conditions: np.ndarray, n_samples: int) -> np.ndarray:
    values = amortizer.sample(
        {"conditions": np.asarray(conditions, dtype=np.float32)}, int(n_samples)
    )
    values = np.asarray(values, dtype=np.float32)
    if len(np.atleast_2d(conditions)) == 1:
        return values.reshape(int(n_samples), 2)
    return values.reshape(len(conditions), int(n_samples), 2)


def chunked_log_likelihood(
    amortizer, theta: np.ndarray, observation: np.ndarray, chunk_size: int = 8192
) -> np.ndarray:
    chunks = []
    observation = np.asarray(observation, dtype=np.float32).reshape(1, 2)
    for start in range(0, len(theta), int(chunk_size)):
        stop = min(len(theta), start + int(chunk_size))
        observables = np.repeat(observation, stop - start, axis=0)
        values = amortizer.log_likelihood(
            {
                "observables": observables,
                "conditions": np.asarray(theta[start:stop], dtype=np.float32),
            }
        )
        chunks.append(np.asarray(values, dtype=np.float64).reshape(-1))
    return np.concatenate(chunks)


def chunked_log_posterior(
    amortizer, theta: np.ndarray, observation: np.ndarray, chunk_size: int = 8192
) -> np.ndarray:
    chunks = []
    observation = np.asarray(observation, dtype=np.float32).reshape(1, 2)
    for start in range(0, len(theta), int(chunk_size)):
        stop = min(len(theta), start + int(chunk_size))
        contexts = np.repeat(observation, stop - start, axis=0)
        values = amortizer.log_posterior(
            {
                "parameters": np.asarray(theta[start:stop], dtype=np.float32),
                "direct_conditions": contexts,
            }
        )
        chunks.append(np.asarray(values, dtype=np.float64).reshape(-1))
    return np.concatenate(chunks)


def assert_finite(name: str, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise FloatingPointError(f"Non-finite values in {name}: shape={values.shape}")
    return values


def make_audit_points(
    posterior,
    likelihood,
    theta: np.ndarray,
    x: np.ndarray,
    seed: int,
    n_anchors: int = 256,
    n_inner: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    n_anchors = min(int(n_anchors), len(x), len(theta))
    if n_anchors < 1:
        raise ValueError("At least one simulation is required for the audits")
    indices_x = rng.choice(len(x), size=n_anchors, replace=False)
    indices_theta = rng.choice(len(theta), size=n_anchors, replace=False)
    x_anchor = np.asarray(x[indices_x], dtype=np.float32)
    theta_anchor = np.asarray(theta[indices_theta], dtype=np.float32)
    theta_draw = sample_posterior(posterior, x_anchor, int(n_inner))
    x_draw = sample_likelihood(likelihood, theta_anchor, int(n_inner))
    posterior_points = np.concatenate(
        [
            theta_draw.reshape(-1, 2),
            np.repeat(x_anchor[:, None, :], int(n_inner), axis=1).reshape(-1, 2),
        ],
        axis=1,
    )
    likelihood_points = np.concatenate(
        [
            np.repeat(theta_anchor[:, None, :], int(n_inner), axis=1).reshape(-1, 2),
            x_draw.reshape(-1, 2),
        ],
        axis=1,
    )
    group_ids = np.repeat(np.arange(n_anchors), int(n_inner))
    return posterior_points, likelihood_points, group_ids, group_ids.copy()


def train_campaign(args: argparse.Namespace) -> None:
    artifact_root = Path(args.artifact_root).expanduser().resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    arrays_path = artifact_root / "legacy_flow_outputs.npz"
    metadata_path = artifact_root / "legacy_flow_metadata.json"
    if arrays_path.is_file() and metadata_path.is_file() and not args.force:
        saved = json.loads(metadata_path.read_text())
        expected = {
            "seed": int(args.seed),
            "simulation_budget": int(args.simulation_budget),
            "posterior_proposals": int(args.posterior_proposals),
            "prior_proposals": int(args.prior_proposals),
            "posterior_samples": int(args.posterior_samples),
            "classifier_schema": "S_P_L_equal_prior_raw_theta_x_v1",
        }
        if all(saved.get(key) == value for key, value in expected.items()):
            print("Reusing compatible legacy-flow artifacts:", arrays_path)
            return
        raise RuntimeError(
            "Existing legacy-flow artifacts do not match this campaign. "
            "Use --force or select a different artifact root."
        )

    seed_everything(args.seed)
    generator = make_generator()

    # The paper script creates the two networks in this order and optimizes the
    # two NLLs jointly through AmortizedPosteriorLikelihood.
    joint_posterior = AmortizedPosterior(make_posterior_network())
    joint_likelihood = AmortizedLikelihood(make_likelihood_network())
    joint = AmortizedPosteriorLikelihood(joint_posterior, joint_likelihood)
    joint_trainer = Trainer(
        amortizer=joint,
        default_lr=0.0005,
        generative_model=generator,
        configurator=configure_joint,
        memory=False,
    )

    simulations = generator(int(args.simulation_budget))
    theta_train = assert_finite(
        "theta_train", np.asarray(simulations["prior_draws"], dtype=np.float32)
    )
    x_train = assert_finite(
        "x_train", np.asarray(simulations["sim_data"], dtype=np.float32)
    )
    rng_state_before_joint_training = np.random.get_state()
    joint_history = joint_trainer.train_offline(
        simulations,
        epochs=64,
        batch_size=32,
        validation_sims=300,
        save_checkpoint=False,
    )
    save_history(joint_history, artifact_root / "joint_jana_history.csv")
    tf.train.Checkpoint(amortizer=joint).write(
        str(artifact_root / "joint_jana_checkpoint")
    )

    # Build the matched hybrid reference flows with exactly the same two
    # architecture constructors.  Resetting seeds makes the comparison paired;
    # separate optimization is the only intentional flow-training difference.
    seed_everything(args.seed)
    separate_posterior = AmortizedPosterior(make_posterior_network())
    separate_likelihood = AmortizedLikelihood(make_likelihood_network())

    posterior_trainer = Trainer(
        amortizer=separate_posterior,
        default_lr=0.0005,
        generative_model=generator,
        configurator=configure_posterior,
        memory=False,
    )
    likelihood_trainer = Trainer(
        amortizer=separate_likelihood,
        default_lr=0.0005,
        generative_model=generator,
        configurator=configure_likelihood,
        memory=False,
    )

    np.random.set_state(rng_state_before_joint_training)
    posterior_history = posterior_trainer.train_offline(
        simulations,
        epochs=64,
        batch_size=32,
        validation_sims=300,
        save_checkpoint=False,
    )
    np.random.set_state(rng_state_before_joint_training)
    likelihood_history = likelihood_trainer.train_offline(
        simulations,
        epochs=64,
        batch_size=32,
        validation_sims=300,
        save_checkpoint=False,
    )
    save_history(posterior_history, artifact_root / "separate_posterior_history.csv")
    save_history(likelihood_history, artifact_root / "separate_likelihood_history.csv")
    tf.train.Checkpoint(amortizer=separate_posterior).write(
        str(artifact_root / "separate_posterior_checkpoint")
    )
    tf.train.Checkpoint(amortizer=separate_likelihood).write(
        str(artifact_root / "separate_likelihood_checkpoint")
    )

    # S/P/L groups for the equal-prior three-class CE correction.  No new
    # simulator calls enter classifier training: all S rows are from the same
    # fixed simulation bank used by the flows.
    seed_everything(args.seed + 101)
    theta_p = sample_posterior(separate_posterior, x_train, 1).reshape(-1, 2)
    x_l = sample_likelihood(separate_likelihood, theta_train, 1).reshape(-1, 2)

    seed_everything(args.seed + 201)
    joint_posterior_samples = sample_posterior(
        joint_posterior, OBSERVATION, int(args.posterior_samples)
    )
    separate_posterior_proposals = sample_posterior(
        separate_posterior, OBSERVATION, int(args.posterior_proposals)
    )

    rng = np.random.default_rng(int(args.seed + 301))
    prior_theta = rng.uniform(
        low=PRIOR_LOW,
        high=PRIOR_HIGH,
        size=(int(args.prior_proposals), 2),
    ).astype(np.float32)
    joint_log_likelihood = chunked_log_likelihood(
        joint_likelihood, prior_theta, OBSERVATION
    )
    separate_log_likelihood = chunked_log_likelihood(
        separate_likelihood, prior_theta, OBSERVATION
    )
    joint_log_posterior = chunked_log_posterior(
        joint_posterior, prior_theta, OBSERVATION
    )
    separate_log_posterior = chunked_log_posterior(
        separate_posterior, prior_theta, OBSERVATION
    )

    seed_everything(args.seed + 401)
    analytic_reference = analytic_posterior_numpy(
        OBSERVATION.reshape(-1), int(args.analytic_samples)
    ).astype(np.float32)

    seed_everything(args.seed + 501)
    (
        audit_posterior_points,
        audit_likelihood_points,
        audit_posterior_group,
        audit_likelihood_group,
    ) = make_audit_points(
        separate_posterior,
        separate_likelihood,
        theta_train,
        x_train,
        seed=args.seed + 502,
    )

    arrays = {
        "theta_train": theta_train,
        "x_train": x_train,
        "theta_p": assert_finite("theta_p", theta_p),
        "x_l": assert_finite("x_l", x_l),
        "joint_posterior_samples": assert_finite(
            "joint_posterior_samples", joint_posterior_samples
        ),
        "separate_posterior_proposals": assert_finite(
            "separate_posterior_proposals", separate_posterior_proposals
        ),
        "prior_theta": prior_theta,
        "joint_log_likelihood": assert_finite(
            "joint_log_likelihood", joint_log_likelihood
        ),
        "separate_log_likelihood": assert_finite(
            "separate_log_likelihood", separate_log_likelihood
        ),
        "joint_log_posterior": assert_finite(
            "joint_log_posterior", joint_log_posterior
        ),
        "separate_log_posterior": assert_finite(
            "separate_log_posterior", separate_log_posterior
        ),
        "analytic_reference": assert_finite(
            "analytic_reference", analytic_reference
        ),
        "audit_posterior_points": assert_finite(
            "audit_posterior_points", audit_posterior_points
        ),
        "audit_likelihood_points": assert_finite(
            "audit_likelihood_points", audit_likelihood_points
        ),
        "audit_posterior_group": audit_posterior_group,
        "audit_likelihood_group": audit_likelihood_group,
    }
    np.savez_compressed(arrays_path, **arrays)

    from bayesflow.version import __version__ as bayesflow_version

    metadata = {
        "seed": int(args.seed),
        "simulation_budget": int(args.simulation_budget),
        "posterior_proposals": int(args.posterior_proposals),
        "prior_proposals": int(args.prior_proposals),
        "posterior_samples": int(args.posterior_samples),
        "analytic_samples": int(args.analytic_samples),
        "observation": OBSERVATION.reshape(-1).tolist(),
        "prior": "Uniform([-2,-2],[2,2])",
        "simulator": "Wiqvist_et_al_main_paper_two_moons",
        "jana_paper_commit": JANA_PAPER_COMMIT,
        "bayesflow_commit": BAYESFLOW_COMMIT,
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "bayesflow": bayesflow_version,
        "numpy": np.__version__,
        "joint_training": {
            "epochs": 64,
            "batch_size": 32,
            "default_learning_rate": 0.0005,
            "optimizer": "Adam with BayesFlow default cosine decay",
            "global_clipnorm": 1.0,
            "validation_simulations": 300,
            "early_stopping": False,
        },
        "posterior_topology": {
            "coupling_layers": 4,
            "design": "spline",
            "permutation": "learnable",
            "act_norm": True,
            "dense_layers_per_coupling": 2,
            "dense_units": 128,
            "activation": "relu",
            "spline_bins": 16,
            "spline_domain": [-5.0, 5.0],
            "dropout": 0.05,
        },
        "likelihood_topology": {
            "coupling_layers": 5,
            "design": ["affine", "spline", "affine", "spline", "affine"],
            "permutation": "learnable",
            "act_norm": True,
            "dense_layers_per_coupling": 2,
            "dense_units": 128,
            "activation": "relu",
            "affine_dropout": 0.01,
            "spline_dropout": 0.05,
        },
        "classifier_schema": "S_P_L_equal_prior_raw_theta_x_v1",
        "separate_flow_difference": (
            "same topology/data/epochs/batch/LR; posterior and likelihood "
            "optimized by separate Trainer instances"
        ),
        "artifacts": {key: list(np.asarray(value).shape) for key, value in arrays.items()},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print("Saved:", arrays_path)
    print("Saved:", metadata_path)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--simulation-budget", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--posterior-proposals", type=int, default=150_000)
    parser.add_argument("--prior-proposals", type=int, default=150_000)
    parser.add_argument("--posterior-samples", type=int, default=10_000)
    parser.add_argument("--analytic-samples", type=int, default=10_000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    train_campaign(parse_args())
