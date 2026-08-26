"""Pinned legacy-BayesFlow driver for the JANA Figure 3 benchmarks.

This module mirrors the Gaussian Mixture and SIR notebooks in
``bayesflow-org/JANA-Paper`` at commit
``6cbbc94faf0aa85147986f7f9516d13a52551bd4``.  It is launched by
``Exercise_9a_JANA.ipynb`` in an isolated Python 3.11 environment containing
TensorFlow 2.12 and BayesFlow commit
``153dfefadd347717b7aeb9c4872a4b51ac04e83c``.

For each task, the driver trains:

1. the original jointly optimized JANA posterior/likelihood amortizer; and
2. topology-matched posterior and likelihood amortizers trained separately.

The second pair supplies the frozen proposals used by the notebook's
post-training equal-prior multiclass CE correction.  It changes only the
optimizer partition: simulator bank, task-specific preprocessing, flow
definitions, latent bases, epoch count, batch size, validation budget,
learning-rate schedule, clipping, and BayesFlow implementation remain those
of the corresponding JANA Figure 3 notebook.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from bayesflow import benchmarks
from bayesflow.amortizers import (
    AmortizedLikelihood,
    AmortizedPosterior,
    AmortizedPosteriorLikelihood,
)
from bayesflow.networks import InvertibleNetwork
from bayesflow.trainers import Trainer


JANA_PAPER_COMMIT = "6cbbc94faf0aa85147986f7f9516d13a52551bd4"
BAYESFLOW_COMMIT = "153dfefadd347717b7aeb9c4872a4b51ac04e83c"
DRIVER_SCHEMA = "jana_fig3_exact_task_specific_v1"
TASKS = ("gaussian_mixture", "sir")
SIMULATION_BUDGET = 10_000
VALIDATION_SIMULATIONS = 300
TEST_DATASETS = 1_000
POSTERIOR_DRAWS = 250

TASK_CAMPAIGNS = {
    "gaussian_mixture": {
        "epochs": 150,
        "batch_size": 64,
        "parameter_names": [r"$\theta_1$", r"$\theta_2$"],
    },
    "sir": {
        "epochs": 250,
        "batch_size": 32,
        "parameter_names": [r"$\beta$", r"$\gamma$"],
    },
}


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    tf.random.set_seed(int(seed))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_finite(name: str, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise FloatingPointError(f"Non-finite values in {name}: shape={values.shape}")
    return values


def save_history(history, path: Path) -> None:
    if isinstance(history, pd.DataFrame):
        history.to_csv(path, index=False)
        return
    try:
        pd.DataFrame(history).to_csv(path, index=False)
    except Exception:
        path.with_suffix(".json").write_text(
            json.dumps(
                history,
                default=lambda value: np.asarray(value).tolist(),
                indent=2,
            )
        )


def make_components(task: str, seed: int):
    """Build the exact task-specific JANA objects and configurator."""

    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}; got {task!r}")
    benchmark = benchmarks.Benchmark(task, seed=int(seed))

    if task == "gaussian_mixture":
        settings = {
            "dense_args": dict(
                units=64,
                activation="swish",
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            ),
            "dropout_prob": 0.05,
            "num_dense": 1,
        }

        def make_posterior():
            return AmortizedPosterior(
                InvertibleNetwork(
                    num_params=2,
                    num_coupling_layers=6,
                    permutation="learnable",
                    coupling_design="spline",
                    coupling_settings=settings,
                )
            )

        def make_likelihood():
            return AmortizedLikelihood(
                InvertibleNetwork(
                    num_params=2,
                    num_coupling_layers=6,
                    coupling_design="spline",
                    coupling_settings=settings,
                )
            )

        def configure(forward_dict, preconfigured=False):
            if not preconfigured:
                output = benchmark.configurator(forward_dict)
            else:
                output = copy.deepcopy(forward_dict)
            output["posterior_inputs"]["parameters"] /= 10.0
            output["likelihood_inputs"]["conditions"] /= 10.0
            return output

        topology = {
            "posterior": {
                "coupling_layers": 6,
                "design": "spline",
                "permutation": "learnable",
                "dense_layers_per_coupling": 1,
                "dense_units": 64,
                "activation": "swish",
                "l2": 1e-4,
                "dropout": 0.05,
            },
            "likelihood": {
                "coupling_layers": 6,
                "design": "spline",
                "permutation": "fixed (BayesFlow default)",
                "dense_layers_per_coupling": 1,
                "dense_units": 64,
                "activation": "swish",
                "l2": 1e-4,
                "dropout": 0.05,
            },
            "latent_bases": "unit multivariate Gaussian",
            "preprocessing": "BayesFlow x/12 plus JANA theta/10",
        }
    else:
        likelihood_settings = {
            "dense_args": dict(
                units=64,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            ),
            "num_dense": 1,
            "dropout_prob": 0.05,
        }
        posterior_settings = {
            "dense_args": dict(
                units=64,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            ),
            "num_dense": 1,
            "dropout_prob": 0.05,
        }
        latent_posterior = tfd.MultivariateStudentTLinearOperator(
            df=50,
            loc=[0.0] * 2,
            scale=tf.linalg.LinearOperatorDiag([1.0] * 2),
        )
        latent_likelihood = tfd.MultivariateStudentTLinearOperator(
            df=50,
            loc=[0.0] * 10,
            scale=tf.linalg.LinearOperatorDiag([1.0] * 10),
        )
        summary_net = lambda x, **kwargs: x[:, :, 0]

        def make_posterior():
            return AmortizedPosterior(
                InvertibleNetwork(
                    num_params=2,
                    coupling_settings=posterior_settings,
                    num_coupling_layers=6,
                    coupling_design="spline",
                ),
                summary_net=summary_net,
                latent_dist=latent_posterior,
            )

        def make_likelihood():
            return AmortizedLikelihood(
                InvertibleNetwork(
                    num_params=10,
                    coupling_settings=likelihood_settings,
                    num_coupling_layers=8,
                ),
                latent_dist=latent_likelihood,
            )

        def configure(forward_dict, preconfigured=False):
            if not preconfigured:
                output = benchmark.configurator(
                    forward_dict, as_summary_condition=True
                )
            else:
                output = copy.deepcopy(forward_dict)
            observation_shape = output["likelihood_inputs"]["observables"].shape
            condition_shape = output["posterior_inputs"]["summary_conditions"].shape
            output["likelihood_inputs"]["observables"] += (
                1e-3 * np.random.normal(size=observation_shape)
            )
            output["posterior_inputs"]["summary_conditions"] += (
                1e-3 * np.random.normal(size=condition_shape)
            )
            return output

        topology = {
            "posterior": {
                "coupling_layers": 6,
                "design": "spline",
                "permutation": "fixed (BayesFlow default)",
                "dense_layers_per_coupling": 1,
                "dense_units": 64,
                "activation": "relu",
                "l2": 1e-4,
                "dropout": 0.05,
            },
            "likelihood": {
                "coupling_layers": 8,
                "design": "affine (BayesFlow default)",
                "permutation": "fixed (BayesFlow default)",
                "dense_layers_per_coupling": 1,
                "dense_units": 64,
                "activation": "relu",
                "l2": 1e-4,
                "dropout": 0.05,
            },
            "latent_bases": "multivariate Student-t, df=50, dimensions 2 and 10",
            "preprocessing": "mock summary x[:,:,0] plus independent N(0,1e-6) jitter",
        }

    def configure_posterior(forward_dict):
        return configure(forward_dict)["posterior_inputs"]

    def configure_likelihood(forward_dict):
        return configure(forward_dict)["likelihood_inputs"]

    return {
        "benchmark": benchmark,
        "make_posterior": make_posterior,
        "make_likelihood": make_likelihood,
        "configure": configure,
        "configure_posterior": configure_posterior,
        "configure_likelihood": configure_likelihood,
        "topology": topology,
    }


def flatten_posterior_context(task: str, posterior_inputs: dict) -> np.ndarray:
    if task == "gaussian_mixture":
        values = posterior_inputs["direct_conditions"]
    else:
        values = posterior_inputs["summary_conditions"][:, :, 0]
    return assert_finite("posterior context", np.asarray(values, dtype=np.float32))


def posterior_input(task: str, contexts: np.ndarray) -> dict:
    contexts = np.asarray(contexts, dtype=np.float32)
    if task == "gaussian_mixture":
        return {"direct_conditions": contexts}
    return {"summary_conditions": contexts[:, :, np.newaxis]}


def sample_posterior(
    task: str,
    amortizer,
    contexts: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    contexts = np.asarray(contexts, dtype=np.float32)
    values = amortizer.sample(posterior_input(task, contexts), int(n_samples))
    values = np.asarray(values, dtype=np.float32)
    return assert_finite(
        "posterior samples",
        values.reshape(len(contexts), int(n_samples), 2),
    )


def sample_likelihood(
    amortizer,
    conditions: np.ndarray,
    n_samples: int,
    data_dim: int,
) -> np.ndarray:
    conditions = np.asarray(conditions, dtype=np.float32)
    values = amortizer.sample(
        {"conditions": conditions}, int(n_samples)
    )
    values = np.asarray(values, dtype=np.float32)
    return assert_finite(
        "likelihood samples",
        values.reshape(len(conditions), int(n_samples), int(data_dim)),
    )


def validate_test_bank(path: Path, task: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing pinned JANA Figure 3 test bank: {path}. "
            "Point --jana-root at JANA-Paper commit " + JANA_PAPER_COMMIT
        )
    expected_name = f"{task}_test.pkl"
    if path.name != expected_name:
        raise RuntimeError(f"Expected {expected_name}, got {path.name}")


def load_test_bank(path: Path) -> dict:
    import pickle

    with path.open("rb") as stream:
        return pickle.load(stream)


def train_task(args: argparse.Namespace) -> None:
    task = args.task
    campaign = TASK_CAMPAIGNS[task]
    training_epochs = 2 if args.smoke else int(campaign["epochs"])
    validation_simulations = 64 if args.smoke else VALIDATION_SIMULATIONS
    artifact_root = Path(args.artifact_root).expanduser().resolve() / task
    artifact_root.mkdir(parents=True, exist_ok=True)
    arrays_path = artifact_root / "legacy_flow_outputs.npz"
    metadata_path = artifact_root / "legacy_flow_metadata.json"
    expected_reuse = {
        "driver_schema": DRIVER_SCHEMA,
        "task": task,
        "seed": int(args.seed),
        "simulation_budget": int(args.simulation_budget),
        "posterior_candidates": int(args.posterior_candidates),
        "likelihood_candidates": int(args.likelihood_candidates),
        "posterior_draws": int(args.posterior_draws),
        "smoke": bool(args.smoke),
        "classifier_schema": "S_P_L_equal_prior_jana_model_coordinates_v1",
    }
    if arrays_path.is_file() and metadata_path.is_file() and not args.force:
        saved = json.loads(metadata_path.read_text())
        if all(saved.get(key) == value for key, value in expected_reuse.items()):
            print("Reusing compatible Figure 3 flow artifacts:", arrays_path)
            return
        raise RuntimeError(
            "Existing Figure 3 flow artifacts do not match this campaign. "
            "Use --force or select a different artifact root."
        )

    if args.smoke:
        if int(args.simulation_budget) > 1_000:
            raise ValueError("Smoke mode is limited to at most 1,000 simulations.")
    elif int(args.simulation_budget) != SIMULATION_BUDGET:
        raise ValueError(
            "The paper Figure 3 campaign requires exactly 10,000 simulations."
        )

    jana_root = Path(args.jana_root).expanduser().resolve()
    test_path = (
        jana_root
        / "experiments"
        / "benchmarks"
        / "test_data"
        / f"{task}_test.pkl"
    )
    validate_test_bank(test_path, task)

    seed_everything(args.seed)
    joint_components = make_components(task, args.seed)
    joint_posterior = joint_components["make_posterior"]()
    joint_likelihood = joint_components["make_likelihood"]()
    joint = AmortizedPosteriorLikelihood(joint_posterior, joint_likelihood)
    joint_trainer = Trainer(
        amortizer=joint,
        generative_model=joint_components["benchmark"].generative_model,
        configurator=joint_components["configure"],
        memory=False,
    )

    # In the paper notebooks, the Trainer pilot check is performed before this
    # fixed 10,000-simulation offline bank is generated.
    simulations = joint_components["benchmark"].generative_model(
        int(args.simulation_budget)
    )
    training_rng_state = np.random.get_state()
    joint_history = joint_trainer.train_offline(
        simulations,
        epochs=training_epochs,
        batch_size=int(campaign["batch_size"]),
        validation_sims=validation_simulations,
        save_checkpoint=False,
    )
    save_history(joint_history, artifact_root / "joint_jana_history.csv")
    joint_checkpoint = tf.train.Checkpoint(amortizer=joint).write(
        str(artifact_root / "joint_jana_checkpoint")
    )

    # Build the hybrid proposal pair from the same constructors and bank.
    # Resetting seeds pairs initialization and the offline shuffle as closely as
    # the legacy implementation permits; only the Trainer partition differs.
    seed_everything(args.seed)
    separate_components = make_components(task, args.seed)
    separate_posterior = separate_components["make_posterior"]()
    separate_likelihood = separate_components["make_likelihood"]()
    posterior_trainer = Trainer(
        amortizer=separate_posterior,
        generative_model=separate_components["benchmark"].generative_model,
        configurator=separate_components["configure_posterior"],
        memory=False,
    )
    likelihood_trainer = Trainer(
        amortizer=separate_likelihood,
        generative_model=separate_components["benchmark"].generative_model,
        configurator=separate_components["configure_likelihood"],
        memory=False,
    )
    np.random.set_state(training_rng_state)
    posterior_history = posterior_trainer.train_offline(
        simulations,
        epochs=training_epochs,
        batch_size=int(campaign["batch_size"]),
        validation_sims=validation_simulations,
        save_checkpoint=False,
    )
    np.random.set_state(training_rng_state)
    likelihood_history = likelihood_trainer.train_offline(
        simulations,
        epochs=training_epochs,
        batch_size=int(campaign["batch_size"]),
        validation_sims=validation_simulations,
        save_checkpoint=False,
    )
    save_history(
        posterior_history, artifact_root / "separate_posterior_history.csv"
    )
    save_history(
        likelihood_history, artifact_root / "separate_likelihood_history.csv"
    )
    posterior_checkpoint = tf.train.Checkpoint(
        amortizer=separate_posterior
    ).write(str(artifact_root / "separate_posterior_checkpoint"))
    likelihood_checkpoint = tf.train.Checkpoint(
        amortizer=separate_likelihood
    ).write(str(artifact_root / "separate_likelihood_checkpoint"))

    # Construct the CE S/P/L bank only after both separate flows are frozen.
    seed_everything(args.seed + 101)
    configured_training = separate_components["configure"](
        copy.deepcopy(simulations)
    )
    theta_train = np.asarray(
        configured_training["posterior_inputs"]["parameters"],
        dtype=np.float32,
    )
    x_train = flatten_posterior_context(
        task, configured_training["posterior_inputs"]
    )
    data_dim = x_train.shape[1]
    theta_p = sample_posterior(
        task, separate_posterior, x_train, 1
    )[:, 0, :]
    x_l = sample_likelihood(
        separate_likelihood, theta_train, 1, data_dim
    )[:, 0, :]

    # Figure 3 evaluates the shipped 1,000-dataset test bank with 250 draws.
    seed_everything(args.seed + 201)
    configured_test = separate_components["configure"](
        load_test_bank(test_path), preconfigured=True
    )
    theta_test = np.asarray(
        configured_test["posterior_inputs"]["parameters"], dtype=np.float32
    )
    x_test = flatten_posterior_context(
        task, configured_test["posterior_inputs"]
    )
    if len(theta_test) != TEST_DATASETS:
        raise RuntimeError(
            f"Pinned {task} test bank has {len(theta_test)} rows, expected 1000"
        )

    seed_everything(args.seed + 301)
    joint_posterior_true = sample_posterior(
        task, joint_posterior, x_test, int(args.posterior_draws)
    )
    joint_surrogate_x = sample_likelihood(
        joint_likelihood, theta_test, 1, data_dim
    )[:, 0, :]
    joint_posterior_surrogate = sample_posterior(
        task, joint_posterior, joint_surrogate_x, int(args.posterior_draws)
    )

    seed_everything(args.seed + 401)
    separate_posterior_true = sample_posterior(
        task, separate_posterior, x_test, int(args.posterior_draws)
    )
    separate_surrogate_x = sample_likelihood(
        separate_likelihood, theta_test, 1, data_dim
    )[:, 0, :]
    separate_posterior_surrogate = sample_posterior(
        task,
        separate_posterior,
        separate_surrogate_x,
        int(args.posterior_draws),
    )

    seed_everything(args.seed + 501)
    posterior_true_candidates = sample_posterior(
        task,
        separate_posterior,
        x_test,
        int(args.posterior_candidates),
    )
    likelihood_candidates = sample_likelihood(
        separate_likelihood,
        theta_test,
        int(args.likelihood_candidates),
        data_dim,
    )

    arrays = {
        "theta_train": assert_finite("theta_train", theta_train),
        "x_train": assert_finite("x_train", x_train),
        "theta_p": assert_finite("theta_p", theta_p),
        "x_l": assert_finite("x_l", x_l),
        "theta_test": assert_finite("theta_test", theta_test),
        "x_test": assert_finite("x_test", x_test),
        "joint_posterior_true": joint_posterior_true,
        "joint_surrogate_x": joint_surrogate_x,
        "joint_posterior_surrogate": joint_posterior_surrogate,
        "separate_posterior_true": separate_posterior_true,
        "separate_surrogate_x": separate_surrogate_x,
        "separate_posterior_surrogate": separate_posterior_surrogate,
        "posterior_true_candidates": posterior_true_candidates,
        "likelihood_candidates": likelihood_candidates,
    }
    np.savez_compressed(arrays_path, **arrays)

    from bayesflow.version import __version__ as bayesflow_version

    metadata = {
        **expected_reuse,
        "jana_paper_commit": JANA_PAPER_COMMIT,
        "bayesflow_commit": BAYESFLOW_COMMIT,
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "bayesflow": bayesflow_version,
        "numpy": np.__version__,
        "test_data_path": str(test_path),
        "test_data_sha256": sha256_file(test_path),
        "test_datasets": int(len(theta_test)),
        "training": {
            "epochs": training_epochs,
            "batch_size": int(campaign["batch_size"]),
            "default_learning_rate": 0.0005,
            "optimizer": "Adam with BayesFlow default cosine decay",
            "global_clipnorm": 1.0,
            "validation_simulations": validation_simulations,
            "early_stopping": False,
        },
        "topology": joint_components["topology"],
        "joint_checkpoint": str(joint_checkpoint),
        "separate_posterior_checkpoint": str(posterior_checkpoint),
        "separate_likelihood_checkpoint": str(likelihood_checkpoint),
        "separate_flow_difference": (
            "same topology, model coordinates, simulator bank, epochs, batch, "
            "optimizer, schedule, clipping, and validation budget; posterior and "
            "likelihood use separate Trainer instances"
        ),
        "arrays": {
            key: list(np.asarray(value).shape) for key, value in arrays.items()
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print("Saved:", arrays_path)
    print("Saved:", metadata_path)


def sample_selected_contexts(args: argparse.Namespace) -> None:
    task = args.task
    artifact_root = Path(args.artifact_root).expanduser().resolve() / task
    metadata_path = artifact_root / "legacy_flow_metadata.json"
    selected_path = artifact_root / "selected_hybrid_contexts.npz"
    output_path = artifact_root / "selected_context_posterior_candidates.npz"
    if not metadata_path.is_file() or not selected_path.is_file():
        raise FileNotFoundError(
            "Train the legacy flows and save selected_hybrid_contexts.npz first."
        )
    metadata = json.loads(metadata_path.read_text())
    expected = {
        "driver_schema": DRIVER_SCHEMA,
        "task": task,
        "seed": int(args.seed),
        "posterior_candidates": int(args.posterior_candidates),
        "jana_paper_commit": JANA_PAPER_COMMIT,
        "bayesflow_commit": BAYESFLOW_COMMIT,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Incompatible selected-context campaign: {mismatches}")

    selected = dict(np.load(selected_path, allow_pickle=False))
    contexts = assert_finite(
        "selected hybrid contexts",
        np.asarray(selected["contexts"], dtype=np.float32),
    )
    selection_digest = hashlib.sha256(
        np.ascontiguousarray(contexts).view(np.uint8)
    ).hexdigest()
    if output_path.is_file() and not args.force:
        existing = dict(np.load(output_path, allow_pickle=False))
        stored_digest = str(np.asarray(existing["selection_sha256"]).reshape(()))
        candidates = np.asarray(existing["posterior_candidates"])
        if (
            stored_digest == selection_digest
            and candidates.shape
            == (len(contexts), int(args.posterior_candidates), 2)
        ):
            print("Reusing selected-context proposals:", output_path)
            return
        raise RuntimeError(
            "Stale selected-context proposals. Use --force to replace them."
        )

    seed_everything(args.seed + 701)
    components = make_components(task, args.seed)
    posterior = components["make_posterior"]()
    checkpoint_prefix = str(artifact_root / "separate_posterior_checkpoint")
    # ``write`` checkpoints contain model variables but deliberately omit the
    # Checkpoint object's save_counter.  ``read`` is the matching restore API.
    status = tf.train.Checkpoint(amortizer=posterior).read(checkpoint_prefix)
    candidates = sample_posterior(
        task, posterior, contexts, int(args.posterior_candidates)
    )
    status.assert_existing_objects_matched()
    np.savez_compressed(
        output_path,
        posterior_candidates=candidates,
        contexts=contexts,
        selection_sha256=np.asarray(selection_digest),
    )
    print("Saved:", output_path)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "selected-contexts"), default="train")
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--jana-root", required=True)
    parser.add_argument("--simulation-budget", type=int, default=SIMULATION_BUDGET)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--posterior-candidates", type=int, default=4096)
    parser.add_argument("--likelihood-candidates", type=int, default=1024)
    parser.add_argument("--posterior-draws", type=int, default=POSTERIOR_DRAWS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "selected-contexts" and args.simulation_budget < 1:
        parser.error("--simulation-budget must be positive")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.mode == "train":
        train_task(parsed)
    else:
        sample_selected_contexts(parsed)
