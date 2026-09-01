"""Exact, isolated BayesFlow/JANA support for the SLCP paper campaign.

This module is intentionally separate from :mod:`utils`.  The exact JANA row
uses TensorFlow 2.12 and the historical BayesFlow revision pinned in
``requirements_jana.txt``; the matched-flow and hybrid rows use the modern
PyTorch runtime.  Keeping the two environments separate prevents an
installation needed for one row from silently changing another row.

Only NumPy and the Python standard library are imported at module import time.
TensorFlow, pandas, and BayesFlow are loaded lazily by training or inference
functions, so modern-runtime code may safely import the artifact helpers.

The primary protocol reproduces the SLCP notebook in the pinned JANA paper:

* a joint ``AmortizedPosteriorLikelihood`` objective;
* a six-coupling interleaved posterior flow;
* a four-coupling affine likelihood flow;
* learnable permutations and BayesFlow's default ActNorm/coupling settings;
* 100 epochs, batch size 32, initial learning rate 5e-4, the BayesFlow cosine
  schedule and global gradient clip norm 1;
* observations divided by 30 and parameters kept in physical coordinates.

The sole data adapter replaces online simulations by nested prefixes of the
shared fixed SLCP bank.  As in the upstream notebook, Benchmark construction
uses two rows for shape inference, the Trainer uses another two-row
consistency/ActNorm pilot, and validation uses 300 additional rows.  All fixed
external banks are independent of training and audit data, and the primary row
is reported as ``N + 2 + 2 + 300``.  An inside-budget mode exists only as an
explicitly non-exact sensitivity check.

No ratio correction is implemented here.  The likelihood route is ordinary
importance sampling with the learned JANA likelihood, the exact box prior,
and JANA's own nominal posterior ``q_phi`` as the method-native proposal.  The
lower-level constructor still requires explicit physical-theta arrays and
their physical-space log density, so the proposal measure is never implicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from config import (
        JANA_BAYESFLOW_COMMIT,
        JANA_PAPER_COMMIT,
        JANA_PAPER_SETTINGS,
        PAPER_BUDGETS,
    )
except ImportError:  # pragma: no cover - supports direct copying of this file
    JANA_PAPER_COMMIT = "6cbbc94faf0aa85147986f7f9516d13a52551bd4"
    JANA_BAYESFLOW_COMMIT = "153dfefadd347717b7aeb9c4872a4b51ac04e83c"
    PAPER_BUDGETS = (10_000, 100_000, 1_000_000)
    JANA_PAPER_SETTINGS = {
        "epochs": 100,
        "batch_size": 32,
        "validation_simulations": 300,
        "initial_learning_rate": 5.0e-4,
    }


JANA_RUNTIME_SCHEMA = "slcp_jana_paper_bayesflow_v1"
JANA_EVALUATION_SCHEMA = "slcp_jana_routes_v1"
POSTERIOR_DIMENSION = 5
OBSERVATION_DIMENSION = 8
OBSERVATION_SCALE = 30.0
PRIOR_LOW = -3.0
PRIOR_HIGH = 3.0
UPSTREAM_VALIDATION_ROWS = 300
UPSTREAM_PILOT_ROWS = 2
UPSTREAM_SHAPE_ROWS = 2
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 5.0e-4
EXPECTED_TENSORFLOW = "2.12.0"
EXPECTED_NUMPY = "1.23.5"
EXPECTED_TENSORFLOW_PROBABILITY = "0.20.1"
EXPECTED_KERAS = "2.12.0"


@dataclass(frozen=True)
class SLCPBank:
    """Verified arrays and provenance read from one fixed SLCP bank."""

    path: Path
    theta: np.ndarray
    x: np.ndarray
    role: str | None
    seed: int | None
    content_fingerprint: str
    file_sha256: str


@dataclass(frozen=True)
class TrainingSlice:
    """Concrete train/validation arrays and their simulation accounting."""

    theta_train: np.ndarray
    x_train: np.ndarray
    theta_validation: np.ndarray
    x_validation: np.ndarray
    theta_pilot: np.ndarray
    x_pilot: np.ndarray
    theta_shape: np.ndarray
    x_shape: np.ndarray
    training_indices: np.ndarray
    validation_indices: np.ndarray
    validation_mode: str
    paper_exact_validation_protocol: bool
    requested_budget: int
    training_simulator_calls: int
    shape_simulator_calls: int
    pilot_simulator_calls: int
    validation_simulator_calls: int
    total_simulator_calls: int
    master: SLCPBank
    shape_bank: SLCPBank | None
    pilot_bank: SLCPBank | None
    validation_bank: SLCPBank | None
    split_path: Path | None
    split_fingerprint: str | None


@dataclass(frozen=True)
class LoadedJANA:
    """Restored legacy objects.  Instances live only in the legacy runtime."""

    joint: Any
    posterior: Any
    likelihood: Any
    manifest: Mapping[str, Any]
    run_directory: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(*arrays: np.ndarray) -> str:
    """Match the content hash used by the sibling paper-summary runtime."""

    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("utf-8"))
        digest.update(json.dumps(contiguous.shape).encode("utf-8"))
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _mapping_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_savez(path: str | Path, **arrays: np.ndarray) -> None:
    path = Path(path)
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


def _atomic_write_dataframe_csv(path: str | Path, frame: Any) -> None:
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


def _as_scalar(saved: Mapping[str, np.ndarray], key: str) -> Any | None:
    if key not in saved:
        return None
    value = np.asarray(saved[key])
    if value.size != 1:
        raise RuntimeError(f"Bank metadata {key!r} is not scalar.")
    return value.reshape(()).item()


def _first_present(
    saved: Mapping[str, np.ndarray], keys: Sequence[str], label: str
) -> np.ndarray:
    for key in keys:
        if key in saved:
            return np.asarray(saved[key])
    raise KeyError(f"Could not find {label}; tried keys {tuple(keys)}")


def load_slcp_bank(path: str | Path, *, expected_role: str | None = None) -> SLCPBank:
    """Load and verify a paper-summary or generic ``theta``/``x`` NPZ bank."""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing SLCP bank: {path}")
    with np.load(path, allow_pickle=False) as saved_file:
        saved = {key: saved_file[key] for key in saved_file.files}
    theta = np.asarray(
        _first_present(saved, ("theta", "theta_train", "prior_draws"), "theta"),
        dtype=np.float32,
    ).reshape(-1, POSTERIOR_DIMENSION)
    x = np.asarray(
        _first_present(saved, ("x", "x_train", "sim_data"), "x"),
        dtype=np.float32,
    ).reshape(-1, OBSERVATION_DIMENSION)
    if len(theta) != len(x) or len(theta) < 1:
        raise RuntimeError(
            f"SLCP bank row mismatch: theta={theta.shape}, x={x.shape}."
        )
    if not (np.isfinite(theta).all() and np.isfinite(x).all()):
        raise FloatingPointError(f"Non-finite values in SLCP bank {path}.")
    role_value = _as_scalar(saved, "role")
    role = None if role_value is None else str(role_value)
    seed_value = _as_scalar(saved, "seed")
    seed = None if seed_value is None else int(seed_value)
    if expected_role is not None and role != str(expected_role):
        raise RuntimeError(
            f"Expected bank role {expected_role!r}, found {role!r} in {path}."
        )
    fingerprint = _array_sha256(theta, x)
    stored_fingerprint = _as_scalar(saved, "fingerprint")
    if stored_fingerprint is not None and str(stored_fingerprint) != fingerprint:
        raise RuntimeError(f"Content fingerprint mismatch in {path}.")
    return SLCPBank(
        path=path,
        theta=theta,
        x=x,
        role=role,
        seed=seed,
        content_fingerprint=fingerprint,
        file_sha256=_sha256_file(path),
    )


def _load_split_indices(
    split_path: str | Path, master: SLCPBank, budget: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    path = Path(split_path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as saved:
        ids = np.asarray(saved["ids"], dtype=np.int64).reshape(-1)
        training_ids = np.asarray(saved["training_ids"], dtype=np.int64).reshape(-1)
        validation_ids = np.asarray(saved["validation_ids"], dtype=np.int64).reshape(-1)
        stored_master = str(saved["master_fingerprint"])
        stored_split = str(saved["split_fingerprint"])
    if stored_master != master.content_fingerprint:
        raise RuntimeError(f"Split {path} does not belong to master bank {master.path}.")
    expected = np.arange(int(budget), dtype=np.int64)
    if not np.array_equal(ids, expected):
        raise RuntimeError(f"Split {path} is not the nested prefix for N={budget}.")
    if np.intersect1d(training_ids, validation_ids).size:
        raise RuntimeError(f"Training/validation overlap in split {path}.")
    if not np.array_equal(np.sort(np.r_[training_ids, validation_ids]), ids):
        raise RuntimeError(f"Split {path} does not partition its prefix.")
    fingerprint = _array_sha256(ids, training_ids, validation_ids)
    if fingerprint != stored_split:
        raise RuntimeError(f"Split fingerprint mismatch in {path}.")
    return ids, training_ids, validation_ids, fingerprint


def prepare_training_slice(
    master_bank_path: str | Path,
    *,
    budget: int,
    validation_mode: str = "external_paper",
    shape_bank_path: str | Path | None = None,
    pilot_bank_path: str | Path | None = None,
    validation_bank_path: str | Path | None = None,
    split_path: str | Path | None = None,
    training_indices: np.ndarray | None = None,
    validation_indices: np.ndarray | None = None,
) -> TrainingSlice:
    """Resolve fixed arrays for the exact or sensitivity validation protocol.

    ``external_paper`` (the default) trains on all ``N`` master-prefix rows,
    consumes an independent two-row Benchmark shape bank, initializes/checks
    the networks on another independent two-row Trainer pilot, and validates
    on the first 300 rows of a fourth bank.  Passing ``training_indices`` makes
    this a non-exact variant unless they equal the ordered full prefix.

    ``inside_budget`` consumes no extra simulations.  It uses explicit caller
    indices, the shared split file, or (last resort) the last 300 prefix rows.
    It is always marked non-exact because the paper used 300 external rows.
    """

    budget = int(budget)
    if budget < 1:
        raise ValueError("budget must be positive.")
    master = load_slcp_bank(master_bank_path, expected_role="master")
    if budget > len(master.theta):
        raise ValueError(
            f"Requested N={budget:,}, but master bank has {len(master.theta):,} rows."
        )
    prefix = np.arange(budget, dtype=np.int64)
    split_fingerprint = None
    resolved_split_path = None
    split_training = None
    split_validation = None
    if split_path is not None:
        resolved_split_path = Path(split_path).expanduser().resolve()
        _, split_training, split_validation, split_fingerprint = _load_split_indices(
            resolved_split_path, master, budget
        )

    mode = str(validation_mode).strip().lower()
    if mode not in {"external_paper", "inside_budget"}:
        raise ValueError(
            "validation_mode must be 'external_paper' or 'inside_budget'."
        )

    if training_indices is not None:
        train_ids = np.asarray(training_indices, dtype=np.int64).reshape(-1)
    elif mode == "inside_budget" and split_training is not None:
        train_ids = split_training
    else:
        train_ids = prefix
    if len(train_ids) < 2 or np.any(train_ids < 0) or np.any(train_ids >= budget):
        raise ValueError("training_indices must contain at least two rows in [0, N).")
    if len(np.unique(train_ids)) != len(train_ids):
        raise ValueError("training_indices contains duplicates.")

    pilot_bank: SLCPBank | None = None
    shape_bank: SLCPBank | None = None
    validation_bank: SLCPBank | None = None
    if mode == "external_paper":
        if shape_bank_path is None:
            raise ValueError(
                "The exact protocol requires shape_bank_path with two "
                "independent benchmark shape-inference rows."
            )
        if pilot_bank_path is None:
            raise ValueError(
                "The exact protocol requires pilot_bank_path with two "
                "independent pre-generated rows."
            )
        if validation_bank_path is None:
            raise ValueError(
                "The exact protocol requires validation_bank_path with 300 "
                "independent pre-generated rows."
            )
        shape_bank = load_slcp_bank(shape_bank_path, expected_role="jana_shape")
        pilot_bank = load_slcp_bank(pilot_bank_path, expected_role="jana_pilot")
        validation_bank = load_slcp_bank(
            validation_bank_path, expected_role="jana_validation"
        )
        if len(shape_bank.theta) < UPSTREAM_SHAPE_ROWS:
            raise ValueError(
                f"External shape bank needs {UPSTREAM_SHAPE_ROWS} rows; "
                f"found {len(shape_bank.theta)}."
            )
        if len(pilot_bank.theta) < UPSTREAM_PILOT_ROWS:
            raise ValueError(
                f"External pilot bank needs {UPSTREAM_PILOT_ROWS} rows; "
                f"found {len(pilot_bank.theta)}."
            )
        if len(validation_bank.theta) < UPSTREAM_VALIDATION_ROWS:
            raise ValueError(
                f"External validation bank needs {UPSTREAM_VALIDATION_ROWS} rows; "
                f"found {len(validation_bank.theta)}."
            )
        independent_banks = (master, shape_bank, pilot_bank, validation_bank)
        if len({bank.path for bank in independent_banks}) != len(independent_banks):
            raise RuntimeError(
                "Training, shape, pilot, and validation bank paths must differ."
            )
        if len({bank.content_fingerprint for bank in independent_banks}) != len(
            independent_banks
        ):
            raise RuntimeError(
                "Training, shape, pilot, and validation banks are not independent."
            )
        known_seeds = [bank.seed for bank in independent_banks if bank.seed is not None]
        if len(set(known_seeds)) != len(known_seeds):
            raise RuntimeError(
                "Training, shape, pilot, and validation bank seeds must differ."
            )
        theta_shape = shape_bank.theta[:UPSTREAM_SHAPE_ROWS]
        x_shape = shape_bank.x[:UPSTREAM_SHAPE_ROWS]
        theta_pilot = pilot_bank.theta[:UPSTREAM_PILOT_ROWS]
        x_pilot = pilot_bank.x[:UPSTREAM_PILOT_ROWS]
        val_ids = np.arange(UPSTREAM_VALIDATION_ROWS, dtype=np.int64)
        theta_validation = validation_bank.theta[val_ids]
        x_validation = validation_bank.x[val_ids]
        validation_calls = UPSTREAM_VALIDATION_ROWS
        shape_calls = UPSTREAM_SHAPE_ROWS
        pilot_calls = UPSTREAM_PILOT_ROWS
        paper_exact = np.array_equal(train_ids, prefix)
    else:
        if validation_indices is not None:
            val_ids = np.asarray(validation_indices, dtype=np.int64).reshape(-1)
        elif split_validation is not None:
            val_ids = split_validation
        else:
            if budget <= UPSTREAM_VALIDATION_ROWS:
                raise ValueError(
                    "inside_budget without an explicit split requires N > 300."
                )
            val_ids = prefix[-UPSTREAM_VALIDATION_ROWS:]
            if training_indices is None:
                train_ids = prefix[:-UPSTREAM_VALIDATION_ROWS]
        if len(val_ids) < 1 or np.any(val_ids < 0) or np.any(val_ids >= budget):
            raise ValueError("validation_indices must be a non-empty subset of [0, N).")
        if len(np.unique(val_ids)) != len(val_ids):
            raise ValueError("validation_indices contains duplicates.")
        if np.intersect1d(train_ids, val_ids).size:
            raise ValueError("Inside-budget training and validation indices overlap.")
        if len(train_ids) < max(UPSTREAM_SHAPE_ROWS, UPSTREAM_PILOT_ROWS):
            raise ValueError(
                "Inside-budget sensitivity mode needs at least two training rows "
                "after validation is removed."
            )
        theta_validation = master.theta[val_ids]
        x_validation = master.x[val_ids]
        theta_pilot = master.theta[train_ids[:UPSTREAM_PILOT_ROWS]]
        x_pilot = master.x[train_ids[:UPSTREAM_PILOT_ROWS]]
        theta_shape = master.theta[train_ids[:UPSTREAM_SHAPE_ROWS]]
        x_shape = master.x[train_ids[:UPSTREAM_SHAPE_ROWS]]
        validation_calls = 0
        shape_calls = 0
        pilot_calls = 0
        paper_exact = False

    theta_train = master.theta[train_ids]
    x_train = master.x[train_ids]
    if not (
        np.isfinite(theta_train).all()
        and np.isfinite(x_train).all()
        and np.isfinite(theta_validation).all()
        and np.isfinite(x_validation).all()
        and np.isfinite(theta_pilot).all()
        and np.isfinite(x_pilot).all()
        and np.isfinite(theta_shape).all()
        and np.isfinite(x_shape).all()
    ):
        raise FloatingPointError("Prepared JANA arrays contain non-finite values.")
    return TrainingSlice(
        theta_train=theta_train,
        x_train=x_train,
        theta_validation=theta_validation,
        x_validation=x_validation,
        theta_pilot=theta_pilot,
        x_pilot=x_pilot,
        theta_shape=theta_shape,
        x_shape=x_shape,
        training_indices=train_ids,
        validation_indices=val_ids,
        validation_mode=mode,
        paper_exact_validation_protocol=bool(paper_exact),
        requested_budget=budget,
        training_simulator_calls=budget,
        shape_simulator_calls=shape_calls,
        pilot_simulator_calls=pilot_calls,
        validation_simulator_calls=validation_calls,
        total_simulator_calls=budget + shape_calls + pilot_calls + validation_calls,
        master=master,
        shape_bank=shape_bank,
        pilot_bank=pilot_bank,
        validation_bank=validation_bank,
        split_path=resolved_split_path,
        split_fingerprint=split_fingerprint,
    )


def configure_joint(forward_dict: Mapping[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """Exact SLCP configurator from the pinned JANA notebook."""

    theta = np.asarray(forward_dict["prior_draws"], dtype=np.float32)
    x_scaled = np.asarray(forward_dict["sim_data"], dtype=np.float32) / OBSERVATION_SCALE
    return {
        "posterior_inputs": {
            "direct_conditions": x_scaled,
            "parameters": theta,
        },
        "likelihood_inputs": {
            "observables": x_scaled,
            "conditions": theta,
        },
    }


def _legacy_imports() -> dict[str, Any]:
    """Load the pinned legacy stack only when a JANA operation is requested."""

    try:
        import tensorflow as tf
        from bayesflow.amortizers import (
            AmortizedLikelihood,
            AmortizedPosterior,
            AmortizedPosteriorLikelihood,
        )
        from bayesflow.networks import InvertibleNetwork
        from bayesflow.trainers import Trainer
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Exact JANA requires a separate legacy environment. Install "
            "requirements_jana.txt there and run this module with that Python."
        ) from error
    return {
        "tf": tf,
        "AmortizedLikelihood": AmortizedLikelihood,
        "AmortizedPosterior": AmortizedPosterior,
        "AmortizedPosteriorLikelihood": AmortizedPosteriorLikelihood,
        "InvertibleNetwork": InvertibleNetwork,
        "Trainer": Trainer,
    }


def _distribution_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _bayesflow_direct_url() -> Mapping[str, Any] | None:
    try:
        distribution = importlib_metadata.distribution("bayesflow")
    except importlib_metadata.PackageNotFoundError:
        return None
    text = distribution.read_text("direct_url.json")
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def validate_legacy_runtime(*, strict: bool = True) -> dict[str, Any]:
    """Record, and in exact mode enforce, the historical runtime provenance."""

    imported = _legacy_imports()
    tf = imported["tf"]
    direct_url = _bayesflow_direct_url()
    installed_commit = None
    if direct_url is not None:
        installed_commit = direct_url.get("vcs_info", {}).get("commit_id")
    runtime = {
        "python": platform.python_version(),
        "tensorflow": str(tf.__version__),
        "numpy": str(np.__version__),
        "tensorflow_probability": _distribution_version(
            "tensorflow-probability"
        ),
        "keras": _distribution_version("keras"),
        "pandas": _distribution_version("pandas"),
        "bayesflow": _distribution_version("bayesflow"),
        "bayesflow_direct_url": direct_url,
        "bayesflow_installed_commit": installed_commit,
        "bayesflow_expected_commit": JANA_BAYESFLOW_COMMIT,
        "jana_paper_expected_commit": JANA_PAPER_COMMIT,
    }
    issues = []
    if platform.python_version_tuple()[:2] != ("3", "11"):
        issues.append(
            "Python 3.11 required, found " + platform.python_version()
        )
    if runtime["tensorflow"] != EXPECTED_TENSORFLOW:
        issues.append(
            f"tensorflow=={EXPECTED_TENSORFLOW} required, found {runtime['tensorflow']}"
        )
    if runtime["numpy"] != EXPECTED_NUMPY:
        issues.append(f"numpy=={EXPECTED_NUMPY} required, found {runtime['numpy']}")
    if runtime["tensorflow_probability"] != EXPECTED_TENSORFLOW_PROBABILITY:
        issues.append(
            "tensorflow-probability=="
            f"{EXPECTED_TENSORFLOW_PROBABILITY} required, found "
            f"{runtime['tensorflow_probability']}"
        )
    if runtime["keras"] != EXPECTED_KERAS:
        issues.append(
            f"keras=={EXPECTED_KERAS} required, found {runtime['keras']}"
        )
    if installed_commit != JANA_BAYESFLOW_COMMIT:
        issues.append(
            "BayesFlow direct-url metadata does not prove commit "
            f"{JANA_BAYESFLOW_COMMIT}; found {installed_commit!r}"
        )
    runtime["strict"] = bool(strict)
    runtime["issues"] = issues
    runtime["paper_exact_runtime"] = not issues
    if strict and issues:
        raise RuntimeError("Legacy JANA runtime mismatch: " + "; ".join(issues))
    return runtime


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    imported = _legacy_imports()
    imported["tf"].random.set_seed(int(seed))


def build_exact_jana() -> LoadedJANA:
    """Construct the exact pinned SLCP JANA amortizer without restoring it."""

    imported = _legacy_imports()
    network = imported["InvertibleNetwork"]
    posterior = imported["AmortizedPosterior"](
        network(
            num_params=POSTERIOR_DIMENSION,
            num_coupling_layers=6,
            coupling_design="interleaved",
            permutation="learnable",
        )
    )
    likelihood = imported["AmortizedLikelihood"](
        network(
            num_params=OBSERVATION_DIMENSION,
            num_coupling_layers=4,
            permutation="learnable",
        )
    )
    joint = imported["AmortizedPosteriorLikelihood"](posterior, likelihood)
    return LoadedJANA(
        joint=joint,
        posterior=posterior,
        likelihood=likelihood,
        manifest={},
        run_directory=Path("."),
    )


def _materialize_variables(model: LoadedJANA) -> None:
    """Build both Keras branches with non-degenerate values after restore."""

    theta = np.asarray(
        [
            [-0.5, 0.2, 0.8, -1.1, 0.3],
            [0.7, -0.4, -1.2, 0.9, -0.2],
        ],
        dtype=np.float32,
    )
    x = np.asarray(
        [
            [-1.0, 0.5, 0.2, -0.7, 1.1, -0.4, 0.8, -1.3],
            [0.6, -1.2, -0.5, 0.9, -0.8, 1.4, -0.1, 0.3],
        ],
        dtype=np.float32,
    )
    model.joint.compute_loss(
        configure_joint({"prior_draws": theta, "sim_data": x})
    )


def _run_trainer_pilot(
    model: LoadedJANA, theta_pilot: np.ndarray, x_pilot: np.ndarray
) -> None:
    """Mirror Trainer's two-simulation consistency/ActNorm initialization."""

    configured = configure_joint(
        {
            "prior_draws": np.asarray(theta_pilot, dtype=np.float32),
            "sim_data": np.asarray(x_pilot, dtype=np.float32),
        }
    )
    losses = model.joint.compute_loss(configured)
    for name, value in losses.items():
        numeric = np.asarray(value, dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise FloatingPointError(
                f"Non-finite JANA consistency-pilot loss {name!r}."
            )


def _consume_benchmark_shape_rows(
    theta_shape: np.ndarray, x_shape: np.ndarray
) -> None:
    """Mirror Benchmark's two simulator calls used only for shape inference."""

    configured = configure_joint(
        {
            "prior_draws": np.asarray(theta_shape, dtype=np.float32),
            "sim_data": np.asarray(x_shape, dtype=np.float32),
        }
    )
    posterior = configured["posterior_inputs"]
    likelihood = configured["likelihood_inputs"]
    expected_shapes = {
        "parameters": (UPSTREAM_SHAPE_ROWS, POSTERIOR_DIMENSION),
        "direct_conditions": (UPSTREAM_SHAPE_ROWS, OBSERVATION_DIMENSION),
        "conditions": (UPSTREAM_SHAPE_ROWS, POSTERIOR_DIMENSION),
        "observables": (UPSTREAM_SHAPE_ROWS, OBSERVATION_DIMENSION),
    }
    observed = {
        "parameters": posterior["parameters"].shape,
        "direct_conditions": posterior["direct_conditions"].shape,
        "conditions": likelihood["conditions"].shape,
        "observables": likelihood["observables"].shape,
    }
    if observed != expected_shapes:
        raise RuntimeError(
            f"JANA benchmark shape-inference contract failed: {observed}."
        )


def default_run_directory(
    artifact_root: str | Path, *, budget: int, seed: int
) -> Path:
    return (
        Path(artifact_root).expanduser().resolve()
        / "jana_paper"
        / f"budget_n{int(budget):07d}"
        / f"seed_{int(seed)}"
    )


def _save_history(history: Any, path: Path) -> list[Path]:
    try:
        import pandas as pd

        if isinstance(history, pd.DataFrame):
            history.to_csv(path, index=False)
            return [path]
        if isinstance(history, Mapping):
            paths: list[Path] = []
            for name, values in history.items():
                frame = values if isinstance(values, pd.DataFrame) else pd.DataFrame(values)
                component_path = path.with_name(
                    f"{path.stem}_{str(name).lower()}{path.suffix}"
                )
                frame.to_csv(component_path, index=False)
                paths.append(component_path)
            if paths:
                return paths
        pd.DataFrame(history).to_csv(path, index=False)
        return [path]
    except Exception:
        json_path = path.with_suffix(".json")
        _atomic_write_json(
            json_path,
            {
                "history": json.loads(
                    json.dumps(
                        history,
                        default=lambda value: np.asarray(value).tolist(),
                    )
                )
            },
        )
        return [json_path]


def _checkpoint_files(prefix: Path) -> list[Path]:
    return sorted(path for path in prefix.parent.glob(prefix.name + "*") if path.is_file())


def _manifest_checkpoint_valid(run_directory: Path, contract_hash: str) -> bool:
    manifest_path = run_directory / "checkpoint_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != JANA_RUNTIME_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("training_contract_sha256") != contract_hash
    ):
        return False
    for entry in manifest.get("checkpoint_files", []):
        path = run_directory / entry["relative_path"]
        if not path.is_file() or _sha256_file(path) != entry["sha256"]:
            return False
    return bool(manifest.get("checkpoint_files")) and manifest.get(
        "checkpoint_artifact_sha256"
    ) == _checkpoint_artifact_sha256(manifest)


def _checkpoint_artifact_sha256(manifest: Mapping[str, Any]) -> str:
    return _mapping_sha256(
        {
            "checkpoint_prefix": manifest.get("checkpoint_prefix"),
            "checkpoint_files": [
                {
                    "relative_path": entry.get("relative_path"),
                    "bytes": entry.get("bytes"),
                    "sha256": entry.get("sha256"),
                }
                for entry in manifest.get("checkpoint_files", [])
            ],
        }
    )


def _evaluation_output_manifest_valid(manifest_path: str | Path) -> bool:
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if (
        manifest.get("schema") != JANA_EVALUATION_SCHEMA
        or manifest.get("status") != "complete"
    ):
        return False
    output_directory = Path(manifest.get("output_directory", manifest_path.parent))
    outputs = manifest.get("output_files", [])
    if not outputs:
        return False
    return all(
        (output_directory / entry["relative_path"]).is_file()
        and _sha256_file(output_directory / entry["relative_path"])
        == entry["sha256"]
        for entry in outputs
    )


def train_exact_jana(
    master_bank_path: str | Path,
    *,
    artifact_root: str | Path,
    budget: int,
    seed: int,
    shape_bank_path: str | Path | None,
    pilot_bank_path: str | Path | None,
    validation_bank_path: str | Path | None,
    validation_mode: str = "external_paper",
    split_path: str | Path | None = None,
    training_indices: np.ndarray | None = None,
    validation_indices: np.ndarray | None = None,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    load_if_available: bool = True,
    force: bool = False,
    strict_runtime: bool = True,
    run_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Train one exact JANA model and write a fingerprinted TF checkpoint."""

    if force and load_if_available:
        load_if_available = False
    epochs = int(epochs)
    batch_size = int(batch_size)
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")
    data = prepare_training_slice(
        master_bank_path,
        budget=budget,
        validation_mode=validation_mode,
        shape_bank_path=shape_bank_path,
        pilot_bank_path=pilot_bank_path,
        validation_bank_path=validation_bank_path,
        split_path=split_path,
        training_indices=training_indices,
        validation_indices=validation_indices,
    )
    runtime = validate_legacy_runtime(strict=strict_runtime)
    run_directory = (
        default_run_directory(artifact_root, budget=budget, seed=seed)
        if run_directory is None
        else Path(run_directory).expanduser().resolve()
    )
    run_directory.mkdir(parents=True, exist_ok=True)

    topology = {
        "joint_amortizer": "AmortizedPosteriorLikelihood",
        "posterior": {
            "dimension": 5,
            "coupling_layers": 6,
            "coupling_design": "interleaved",
            "permutation": "learnable",
            "act_norm": True,
        },
        "likelihood": {
            "dimension": 8,
            "coupling_layers": 4,
            "coupling_design": "affine_default",
            "permutation": "learnable",
            "act_norm": True,
        },
        "coupling_defaults_from_pinned_bayesflow": {
            "dense_layers": 2,
            "dense_units": 128,
            "activation": "relu",
            "affine_l2": 5.0e-4,
            "affine_dropout": 0.01,
            "soft_clamp": 1.9,
            "spline_l2": 5.0e-3,
            "spline_dropout": 0.05,
            "spline_bins": 16,
            "spline_domain": [-5.0, 5.0],
        },
    }
    training_contract = {
        "schema": JANA_RUNTIME_SCHEMA,
        "task": "slcp",
        "seed": int(seed),
        "requested_budget": int(budget),
        "training_rows": int(len(data.theta_train)),
        "validation_rows": int(len(data.theta_validation)),
        "validation_mode": data.validation_mode,
        "paper_exact_validation_protocol": data.paper_exact_validation_protocol,
        "simulation_accounting": {
            "training_calls_N": data.training_simulator_calls,
            "external_benchmark_shape_inference_calls": data.shape_simulator_calls,
            "external_consistency_actnorm_pilot_calls": data.pilot_simulator_calls,
            "external_validation_calls": data.validation_simulator_calls,
            "total_calls": data.total_simulator_calls,
            "display": (
                f"{data.training_simulator_calls} + {data.shape_simulator_calls} + "
                f"{data.pilot_simulator_calls} + "
                f"{data.validation_simulator_calls}"
            ),
        },
        "training_index_sha256": _array_sha256(data.training_indices),
        "validation_index_sha256": _array_sha256(data.validation_indices),
        "training_array_sha256": _array_sha256(data.theta_train, data.x_train),
        "validation_array_sha256": _array_sha256(
            data.theta_validation, data.x_validation
        ),
        "pilot_array_sha256": _array_sha256(data.theta_pilot, data.x_pilot),
        "shape_array_sha256": _array_sha256(data.theta_shape, data.x_shape),
        "master_bank": {
            "path": str(data.master.path),
            "role": data.master.role,
            "seed": data.master.seed,
            "rows": len(data.master.theta),
            "content_sha256": data.master.content_fingerprint,
            "file_sha256": data.master.file_sha256,
        },
        "validation_bank": None
        if data.validation_bank is None
        else {
            "path": str(data.validation_bank.path),
            "role": data.validation_bank.role,
            "seed": data.validation_bank.seed,
            "rows": len(data.validation_bank.theta),
            "content_sha256": data.validation_bank.content_fingerprint,
            "file_sha256": data.validation_bank.file_sha256,
        },
        "pilot_bank": None
        if data.pilot_bank is None
        else {
            "path": str(data.pilot_bank.path),
            "role": data.pilot_bank.role,
            "seed": data.pilot_bank.seed,
            "rows": len(data.pilot_bank.theta),
            "used_rows": UPSTREAM_PILOT_ROWS,
            "content_sha256": data.pilot_bank.content_fingerprint,
            "file_sha256": data.pilot_bank.file_sha256,
        },
        "shape_bank": None
        if data.shape_bank is None
        else {
            "path": str(data.shape_bank.path),
            "role": data.shape_bank.role,
            "seed": data.shape_bank.seed,
            "rows": len(data.shape_bank.theta),
            "used_rows": UPSTREAM_SHAPE_ROWS,
            "content_sha256": data.shape_bank.content_fingerprint,
            "file_sha256": data.shape_bank.file_sha256,
        },
        "shared_split_path": None if data.split_path is None else str(data.split_path),
        "shared_split_sha256": data.split_fingerprint,
        "preprocessing": {
            "theta": "physical coordinates; no transformation",
            "x": "divide every component by 30",
        },
        "topology": topology,
        "optimization": {
            "objective": "sum of posterior and likelihood NLLs",
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": "BayesFlow Trainer Adam",
            "initial_learning_rate": DEFAULT_LEARNING_RATE,
            "schedule": "BayesFlow cosine decay to zero",
            "global_clipnorm": 1.0,
            "early_stopping": False,
            "consistency_actnorm_pilot": {
                "rows": UPSTREAM_PILOT_ROWS,
                "source": (
                    "independent_fixed_jana_pilot_bank"
                    if data.paper_exact_validation_protocol
                    else "inside_budget_training_rows_sensitivity_only"
                ),
                "purpose": "mirror Trainer generative-model consistency check",
            },
            "benchmark_shape_inference": {
                "rows": UPSTREAM_SHAPE_ROWS,
                "source": (
                    "independent_fixed_jana_shape_bank"
                    if data.paper_exact_validation_protocol
                    else "inside_budget_training_rows_sensitivity_only"
                ),
                "purpose": "mirror Benchmark generative-model shape inference",
                "network_effect": "none",
            },
        },
        "jana_paper_commit": JANA_PAPER_COMMIT,
        "bayesflow_commit": JANA_BAYESFLOW_COMMIT,
        "driver_source_sha256": _sha256_file(Path(__file__).resolve()),
        "requirements_sha256": _sha256_file(
            Path(__file__).resolve().parent / "requirements_jana.txt"
        ),
        "runtime": runtime,
        "ratio_correction": False,
    }
    contract_hash = _mapping_sha256(training_contract)
    if load_if_available and _manifest_checkpoint_valid(run_directory, contract_hash):
        return json.loads((run_directory / "checkpoint_manifest.json").read_text())
    existing_manifest = run_directory / "checkpoint_manifest.json"
    if existing_manifest.exists() and not force:
        raise RuntimeError(
            f"Existing checkpoint in {run_directory} is incompatible or incomplete. "
            "Use force=True or choose a new artifact root."
        )

    _consume_benchmark_shape_rows(data.theta_shape, data.x_shape)
    seed_everything(seed)
    model = build_exact_jana()
    _run_trainer_pilot(model, data.theta_pilot, data.x_pilot)
    imported = _legacy_imports()
    trainer = imported["Trainer"](
        amortizer=model.joint,
        default_lr=DEFAULT_LEARNING_RATE,
        configurator=configure_joint,
        memory=False,
    )
    training_dict = {
        "prior_draws": data.theta_train,
        "sim_data": data.x_train,
    }
    validation_dict = {
        "prior_draws": data.theta_validation,
        "sim_data": data.x_validation,
    }
    started = _utc_now()
    history = trainer.train_offline(
        training_dict,
        epochs=epochs,
        batch_size=batch_size,
        validation_sims=validation_dict,
        save_checkpoint=False,
        early_stopping=False,
    )
    history_paths = _save_history(history, run_directory / "training_history.csv")
    checkpoint_prefix = run_directory / "joint_jana_checkpoint"
    written_prefix = imported["tf"].train.Checkpoint(amortizer=model.joint).write(
        str(checkpoint_prefix)
    )
    if str(written_prefix) != str(checkpoint_prefix):
        checkpoint_prefix = Path(str(written_prefix))
    files = _checkpoint_files(checkpoint_prefix)
    if not files:
        raise RuntimeError("TensorFlow did not write any checkpoint files.")
    manifest = {
        **training_contract,
        "status": "complete",
        "started_utc": started,
        "completed_utc": _utc_now(),
        "run_directory": str(run_directory),
        "training_contract_sha256": contract_hash,
        "checkpoint_prefix": checkpoint_prefix.name,
        "checkpoint_files": [
            {
                "relative_path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in files
        ],
        "history_files": [
            {
                "relative_path": history_path.name,
                "sha256": _sha256_file(history_path),
            }
            for history_path in history_paths
        ],
    }
    manifest["checkpoint_artifact_sha256"] = _checkpoint_artifact_sha256(manifest)
    _atomic_write_json(existing_manifest, manifest)
    return manifest


def load_exact_jana(
    run_directory: str | Path, *, strict_runtime: bool = True
) -> LoadedJANA:
    """Restore and fingerprint-check one jointly trained exact JANA model."""

    run_directory = Path(run_directory).expanduser().resolve()
    manifest_path = run_directory / "checkpoint_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing JANA checkpoint manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != JANA_RUNTIME_SCHEMA or manifest.get("status") != "complete":
        raise RuntimeError(f"Invalid or incomplete JANA manifest: {manifest_path}")
    validate_legacy_runtime(strict=strict_runtime)
    for entry in manifest.get("checkpoint_files", []):
        path = run_directory / entry["relative_path"]
        if not path.is_file() or _sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"Checkpoint fingerprint mismatch: {path}")
    if manifest.get("checkpoint_artifact_sha256") != _checkpoint_artifact_sha256(
        manifest
    ):
        raise RuntimeError("Checkpoint artifact fingerprint does not match its manifest.")
    model = build_exact_jana()
    prefix = run_directory / manifest["checkpoint_prefix"]
    imported = _legacy_imports()
    status = imported["tf"].train.Checkpoint(amortizer=model.joint).read(str(prefix))
    _materialize_variables(model)
    status.assert_consumed()
    return LoadedJANA(
        joint=model.joint,
        posterior=model.posterior,
        likelihood=model.likelihood,
        manifest=manifest,
        run_directory=run_directory,
    )


def _resolve_loaded(
    model_or_directory: LoadedJANA | str | Path, *, strict_runtime: bool = True
) -> LoadedJANA:
    if isinstance(model_or_directory, LoadedJANA):
        return model_or_directory
    return load_exact_jana(model_or_directory, strict_runtime=strict_runtime)


def _as_rows(values: np.ndarray, dimension: int, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size % int(dimension):
        raise ValueError(f"{name} cannot be reshaped to (*, {dimension}).")
    values = values.reshape(-1, int(dimension))
    if len(values) < 1 or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite rows.")
    return values


def sample_nominal_posterior(
    model_or_directory: LoadedJANA | str | Path,
    observations: np.ndarray,
    *,
    n_samples: int,
    seed: int,
    context_batch_size: int = 8192,
    sample_batch_size: int = 8192,
    strict_runtime: bool = True,
) -> np.ndarray:
    """Sample ``q_phi(theta | x)`` in physical parameter coordinates."""

    model = _resolve_loaded(model_or_directory, strict_runtime=strict_runtime)
    observations = _as_rows(observations, OBSERVATION_DIMENSION, "observations")
    n_samples = int(n_samples)
    context_batch_size = int(context_batch_size)
    sample_batch_size = int(sample_batch_size)
    if n_samples < 1 or context_batch_size < 1 or sample_batch_size < 1:
        raise ValueError("Sample counts and batch sizes must be positive.")
    seed_everything(seed)
    chunks = []
    for start in range(0, len(observations), int(context_batch_size)):
        context = observations[start : start + int(context_batch_size)]
        sample_chunks = []
        for sample_start in range(0, n_samples, sample_batch_size):
            local_samples = min(sample_batch_size, n_samples - sample_start)
            values = model.joint.sample_parameters(
                {"direct_conditions": context / OBSERVATION_SCALE}, local_samples
            )
            values = np.asarray(values, dtype=np.float32)
            expected = len(context) * local_samples * POSTERIOR_DIMENSION
            if values.size != expected:
                raise RuntimeError(f"Unexpected posterior sample shape {values.shape}.")
            sample_chunks.append(
                values.reshape(len(context), local_samples, POSTERIOR_DIMENSION)
            )
        chunks.append(np.concatenate(sample_chunks, axis=1))
    values = np.concatenate(chunks, axis=0)
    if not np.isfinite(values).all():
        raise FloatingPointError("JANA posterior sampling returned non-finite values.")
    return values[0] if len(observations) == 1 else values


def evaluate_nominal_log_posterior(
    model_or_directory: LoadedJANA | str | Path,
    theta: np.ndarray,
    observations: np.ndarray,
    *,
    chunk_size: int = 8192,
    strict_runtime: bool = True,
) -> np.ndarray:
    """Evaluate paired/broadcast ``log q_phi(theta | x)`` in physical theta."""

    model = _resolve_loaded(model_or_directory, strict_runtime=strict_runtime)
    theta, observations = _broadcast_pairs(theta, observations)
    chunks = []
    for start in range(0, len(theta), int(chunk_size)):
        stop = min(len(theta), start + int(chunk_size))
        values = model.joint.log_posterior(
            {
                "parameters": theta[start:stop],
                "direct_conditions": observations[start:stop] / OBSERVATION_SCALE,
            }
        )
        chunks.append(np.asarray(values, dtype=np.float64).reshape(-1))
    output = np.concatenate(chunks)
    if not np.isfinite(output).all():
        raise FloatingPointError("JANA log posterior returned non-finite values.")
    return output


def _broadcast_pairs(
    theta: np.ndarray, observations: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    theta = _as_rows(theta, POSTERIOR_DIMENSION, "theta")
    observations = _as_rows(observations, OBSERVATION_DIMENSION, "observations")
    if len(theta) == 1 and len(observations) > 1:
        theta = np.repeat(theta, len(observations), axis=0)
    elif len(observations) == 1 and len(theta) > 1:
        observations = np.repeat(observations, len(theta), axis=0)
    elif len(theta) != len(observations):
        raise ValueError("theta and observations need equal rows or one broadcast row.")
    return theta, observations


def evaluate_nominal_log_likelihood(
    model_or_directory: LoadedJANA | str | Path,
    theta: np.ndarray,
    observations: np.ndarray,
    *,
    physical_density: bool = True,
    chunk_size: int = 8192,
    strict_runtime: bool = True,
) -> np.ndarray:
    """Evaluate the learned likelihood in physical-x or scaled-x measure.

    BayesFlow learns the density of ``y=x/30``.  Therefore the physical-space
    density is ``log q_y(x/30|theta) - 8 log(30)``.  The Jacobian is retained
    for analytic-likelihood comparisons even though it cancels normalized
    importance weights at a fixed observation.
    """

    model = _resolve_loaded(model_or_directory, strict_runtime=strict_runtime)
    theta, observations = _broadcast_pairs(theta, observations)
    chunks = []
    for start in range(0, len(theta), int(chunk_size)):
        stop = min(len(theta), start + int(chunk_size))
        values = model.joint.log_likelihood(
            {
                "observables": observations[start:stop] / OBSERVATION_SCALE,
                "conditions": theta[start:stop],
            }
        )
        chunks.append(np.asarray(values, dtype=np.float64).reshape(-1))
    output = np.concatenate(chunks)
    if physical_density:
        output = output - OBSERVATION_DIMENSION * math.log(OBSERVATION_SCALE)
    if not np.isfinite(output).all():
        raise FloatingPointError("JANA log likelihood returned non-finite values.")
    return output


def sample_nominal_likelihood(
    model_or_directory: LoadedJANA | str | Path,
    theta: np.ndarray,
    *,
    n_samples: int,
    seed: int,
    physical_coordinates: bool = True,
    context_batch_size: int = 8192,
    sample_batch_size: int = 8192,
    strict_runtime: bool = True,
) -> np.ndarray:
    """Sample ``q_eta(x | theta)``; physical output multiplies samples by 30."""

    model = _resolve_loaded(model_or_directory, strict_runtime=strict_runtime)
    theta = _as_rows(theta, POSTERIOR_DIMENSION, "theta")
    n_samples = int(n_samples)
    context_batch_size = int(context_batch_size)
    sample_batch_size = int(sample_batch_size)
    if n_samples < 1 or context_batch_size < 1 or sample_batch_size < 1:
        raise ValueError("Sample counts and batch sizes must be positive.")
    seed_everything(seed)
    chunks = []
    for start in range(0, len(theta), int(context_batch_size)):
        conditions = theta[start : start + int(context_batch_size)]
        sample_chunks = []
        for sample_start in range(0, n_samples, sample_batch_size):
            local_samples = min(sample_batch_size, n_samples - sample_start)
            values = model.joint.sample_data(
                {"conditions": conditions}, local_samples
            )
            values = np.asarray(values, dtype=np.float32)
            expected = len(conditions) * local_samples * OBSERVATION_DIMENSION
            if values.size != expected:
                raise RuntimeError(f"Unexpected likelihood sample shape {values.shape}.")
            sample_chunks.append(
                values.reshape(len(conditions), local_samples, OBSERVATION_DIMENSION)
            )
        chunks.append(np.concatenate(sample_chunks, axis=1))
    values = np.concatenate(chunks, axis=0)
    if physical_coordinates:
        values = values * OBSERVATION_SCALE
    if not np.isfinite(values).all():
        raise FloatingPointError("JANA likelihood sampling returned non-finite values.")
    return values[0] if len(theta) == 1 else values


def slcp_prior_log_density(theta: np.ndarray) -> np.ndarray:
    """Physical-space log density of Uniform([-3, 3]^5)."""

    theta = np.asarray(theta, dtype=np.float64).reshape(-1, POSTERIOR_DIMENSION)
    inside = np.all((theta >= PRIOR_LOW) & (theta <= PRIOR_HIGH), axis=1)
    output = np.full(len(theta), -np.inf, dtype=np.float64)
    output[inside] = -POSTERIOR_DIMENSION * math.log(PRIOR_HIGH - PRIOR_LOW)
    return output


def slcp_theta_to_latent(
    theta: np.ndarray, *, outside: str = "raise"
) -> np.ndarray:
    """Map physical box parameters to common logit coordinates ``z``.

    Exact JANA models raw physical parameters and can sample outside the prior
    box.  ``outside='nan'`` preserves those proposal rows while making the
    undefined logit coordinate explicit.
    """

    theta = np.asarray(theta, dtype=np.float64).reshape(-1, POSTERIOR_DIMENSION)
    unit = (theta - PRIOR_LOW) / (PRIOR_HIGH - PRIOR_LOW)
    outside_mask = np.any((unit < 0.0) | (unit > 1.0), axis=1)
    if outside not in {"raise", "nan"}:
        raise ValueError("outside must be 'raise' or 'nan'.")
    if outside == "raise" and outside_mask.any():
        raise ValueError("theta lies outside the SLCP prior box.")
    unit = np.clip(unit, 1.0e-12, 1.0 - 1.0e-12)
    latent = (np.log(unit) - np.log1p(-unit)).astype(np.float32)
    if outside == "nan":
        latent[outside_mask] = np.nan
    return latent


def exact_slcp_log_likelihood(
    theta: np.ndarray, observations: np.ndarray
) -> np.ndarray:
    """Analytic physical-x SLCP likelihood for paired or broadcast rows.

    Each eight-vector contains four IID bivariate-normal observations.  The
    covariance uses scales ``theta_3**2`` and ``theta_4**2`` and correlation
    ``tanh(theta_5)`` exactly as in the benchmark.  Singular parameter rows
    receive ``-inf`` and should be masked in learned-vs-exact error summaries.
    """

    theta, observations = _broadcast_pairs(theta, observations)
    theta64 = theta.astype(np.float64)
    x64 = observations.astype(np.float64).reshape(-1, 4, 2)
    difference = x64 - theta64[:, None, :2]
    scale1 = np.square(theta64[:, 2])
    scale2 = np.square(theta64[:, 3])
    variance1 = np.square(scale1)
    variance2 = np.square(scale2)
    covariance = np.tanh(theta64[:, 4]) * scale1 * scale2
    determinant = variance1 * variance2 - np.square(covariance)
    valid = np.isfinite(determinant) & (determinant > 0.0)
    output = np.full(len(theta64), -np.inf, dtype=np.float64)
    if np.any(valid):
        d = difference[valid]
        v1 = variance1[valid]
        v2 = variance2[valid]
        cov = covariance[valid]
        det = determinant[valid]
        quadratic = (
            v2[:, None] * np.square(d[:, :, 0])
            + v1[:, None] * np.square(d[:, :, 1])
            - 2.0 * cov[:, None] * d[:, :, 0] * d[:, :, 1]
        ) / det[:, None]
        output[valid] = (
            -4.0 * math.log(2.0 * math.pi)
            - 2.0 * np.log(det)
            - 0.5 * np.sum(quadratic, axis=1)
        )
    return output


def construct_likelihood_route_posterior(
    model_or_directory: LoadedJANA | str | Path,
    observation: np.ndarray,
    proposal_theta: np.ndarray,
    proposal_log_density_theta: np.ndarray,
    *,
    n_samples: int,
    seed: int,
    strict_runtime: bool = True,
) -> dict[str, Any]:
    """Construct the likelihood-route posterior from an explicit proposal.

    ``proposal_theta`` must be in physical SLCP coordinates and
    ``proposal_log_density_theta`` must be the corresponding physical-space
    log density.  This explicit interface supports method-native proposals
    (JANA ``q_phi``, matched ``q_phi``, or deployed hybrid ``g_phi``) without
    silently mixing coordinate measures.
    """

    observation = _as_rows(observation, OBSERVATION_DIMENSION, "observation")
    if len(observation) != 1:
        raise ValueError("construct_likelihood_route_posterior expects one observation.")
    proposal_theta = _as_rows(proposal_theta, POSTERIOR_DIMENSION, "proposal_theta")
    proposal_log_density_theta = np.asarray(
        proposal_log_density_theta, dtype=np.float64
    ).reshape(-1)
    if len(proposal_log_density_theta) != len(proposal_theta):
        raise ValueError("proposal samples and log density must have equal rows.")
    if not np.isfinite(proposal_log_density_theta).all():
        raise ValueError("proposal_log_density_theta must be finite for every row.")
    n_samples = int(n_samples)
    if n_samples < 1:
        raise ValueError("n_samples must be positive.")
    learned_log_likelihood = evaluate_nominal_log_likelihood(
        model_or_directory,
        proposal_theta,
        observation,
        physical_density=True,
        strict_runtime=strict_runtime,
    )
    prior_log_density = slcp_prior_log_density(proposal_theta)
    log_weights = (
        prior_log_density
        + learned_log_likelihood
        - proposal_log_density_theta
    )
    finite = np.isfinite(log_weights)
    if not finite.any():
        raise FloatingPointError("All likelihood-route importance weights are zero.")
    maximum = float(np.max(log_weights[finite]))
    unnormalized = np.zeros(len(log_weights), dtype=np.float64)
    unnormalized[finite] = np.exp(log_weights[finite] - maximum)
    total = float(unnormalized.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError("Likelihood-route importance weights failed to normalize.")
    weights = unnormalized / total
    ess = float(1.0 / np.sum(np.square(weights)))
    log_evidence = maximum + math.log(total) - math.log(len(log_weights))
    rng = np.random.default_rng(int(seed))
    positions = (rng.random() + np.arange(n_samples)) / n_samples
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    indices = np.searchsorted(cumulative, positions, side="right")
    samples = proposal_theta[indices].astype(np.float32, copy=True)
    return {
        "samples": samples,
        "resampled_indices": indices.astype(np.int64),
        "normalized_weights": weights,
        "log_weights": log_weights,
        "learned_log_likelihood": learned_log_likelihood,
        "prior_log_density": prior_log_density,
        "proposal_log_density_theta": proposal_log_density_theta,
        "ess": ess,
        "ess_fraction": ess / len(weights),
        "max_weight": float(weights.max()),
        "finite_weight_fraction": float(finite.mean()),
        "log_evidence_importance_estimate": float(log_evidence),
    }


def exact_likelihood_comparison(
    model_or_directory: LoadedJANA | str | Path,
    theta: np.ndarray,
    observations: np.ndarray,
    *,
    strict_runtime: bool = True,
) -> dict[str, np.ndarray]:
    """Return raw arrays for learned-vs-analytic physical SLCP likelihood."""

    theta, observations = _broadcast_pairs(theta, observations)
    learned = evaluate_nominal_log_likelihood(
        model_or_directory,
        theta,
        observations,
        physical_density=True,
        strict_runtime=strict_runtime,
    )
    exact = exact_slcp_log_likelihood(theta, observations)
    valid = np.isfinite(exact) & np.isfinite(learned)
    residual = np.full(len(exact), np.nan, dtype=np.float64)
    residual[valid] = learned[valid] - exact[valid]
    return {
        "theta": theta,
        "x": observations,
        "jana_log_likelihood_physical": learned,
        "exact_log_likelihood_physical": exact,
        "log_likelihood_residual": residual,
        "valid": valid,
    }


def _route_moment_agreement(
    posterior_route: np.ndarray, likelihood_route: np.ndarray
) -> dict[str, float]:
    posterior_route = np.asarray(posterior_route, dtype=np.float64)
    likelihood_route = np.asarray(likelihood_route, dtype=np.float64)
    mean_a = posterior_route.mean(axis=0)
    mean_b = likelihood_route.mean(axis=0)
    covariance_a = np.cov(posterior_route, rowvar=False)
    covariance_b = np.cov(likelihood_route, rowvar=False)
    pooled_scale = np.sqrt(
        np.maximum(0.5 * (np.diag(covariance_a) + np.diag(covariance_b)), 1.0e-12)
    )
    return {
        "route_mean_l2": float(np.linalg.norm(mean_a - mean_b)),
        "route_mean_standardized_rms": float(
            np.sqrt(np.mean(np.square((mean_a - mean_b) / pooled_scale)))
        ),
        "route_covariance_frobenius": float(
            np.linalg.norm(covariance_a - covariance_b, ord="fro")
        ),
    }


def _normalize_observation_inputs(
    observations: np.ndarray,
    reference_posterior_samples: np.ndarray,
    proposal_theta: np.ndarray,
    proposal_log_density_theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    observations = _as_rows(observations, OBSERVATION_DIMENSION, "observations")
    n_observations = len(observations)
    reference = np.asarray(reference_posterior_samples, dtype=np.float32)
    if reference.ndim == 2 and n_observations == 1:
        reference = reference[None, :, :]
    if reference.ndim != 3 or reference.shape[0] != n_observations or reference.shape[2] != 5:
        raise ValueError(
            "reference_posterior_samples must have shape (O, R, 5), or (R, 5) for O=1."
        )
    proposals = np.asarray(proposal_theta, dtype=np.float32)
    log_q = np.asarray(proposal_log_density_theta, dtype=np.float64)
    shared = proposals.ndim == 2
    if shared:
        if proposals.shape[1] != 5 or log_q.shape != (len(proposals),):
            raise ValueError("Shared proposal needs shapes (K,5) and (K,).")
        proposals = np.repeat(proposals[None, :, :], n_observations, axis=0)
        log_q = np.repeat(log_q[None, :], n_observations, axis=0)
    elif (
        proposals.ndim != 3
        or proposals.shape[0] != n_observations
        or proposals.shape[2] != 5
        or log_q.shape != proposals.shape[:2]
    ):
        raise ValueError("Per-observation proposal needs shapes (O,K,5) and (O,K).")
    if not (
        np.isfinite(reference).all()
        and np.isfinite(proposals).all()
        and np.isfinite(log_q).all()
    ):
        raise ValueError("Evaluation inputs must be finite.")
    return observations, reference, proposals, log_q, shared


def run_standardized_evaluation(
    model_or_directory: LoadedJANA | str | Path,
    *,
    observations: np.ndarray,
    reference_posterior_samples: np.ndarray,
    proposal_theta: np.ndarray | None = None,
    proposal_log_density_theta: np.ndarray | None = None,
    proposal_candidates: int = 150_000,
    observation_ids: Sequence[int] | None = None,
    posterior_samples: int = 10_000,
    likelihood_route_samples: int | None = None,
    seed: int = 31082026,
    output_directory: str | Path | None = None,
    audit_reference_theta: np.ndarray | None = None,
    audit_reference_x: np.ndarray | None = None,
    audit_reference_ids: np.ndarray | None = None,
    audit_marginal_theta: np.ndarray | None = None,
    audit_marginal_x: np.ndarray | None = None,
    audit_marginal_ids: np.ndarray | None = None,
    audit_bank_fingerprint: str | None = None,
    normalization_theta: np.ndarray | None = None,
    normalization_theta_ids: np.ndarray | None = None,
    normalization_x_per_theta: int = 0,
    load_if_available: bool = True,
    strict_runtime: bool = True,
) -> dict[str, Any]:
    """Write standardized arrays for modern-runtime metrics and comparisons.

    The output NPZ for every observation contains the three C2ST pairs:
    reference/direct posterior, reference/likelihood-route posterior, and
    direct/likelihood-route agreement.  C2ST itself is intentionally computed
    later by the common modern runtime so every paper row uses the same metric
    implementation and split seed.
    """

    model = _resolve_loaded(model_or_directory, strict_runtime=strict_runtime)
    observations = _as_rows(observations, OBSERVATION_DIMENSION, "observations")
    n_observations = len(observations)
    reference = np.asarray(reference_posterior_samples, dtype=np.float32)
    if reference.ndim == 2 and n_observations == 1:
        reference = reference[None, :, :]
    if (
        reference.ndim != 3
        or reference.shape[0] != n_observations
        or reference.shape[2] != POSTERIOR_DIMENSION
        or not np.isfinite(reference).all()
    ):
        raise ValueError(
            "reference_posterior_samples must be finite with shape (O,R,5), "
            "or (R,5) for one observation."
        )
    if (proposal_theta is None) != (proposal_log_density_theta is None):
        raise ValueError(
            "proposal_theta and proposal_log_density_theta must be supplied together."
        )
    native_proposal = proposal_theta is None
    proposals = None
    log_q = None
    proposal_shared = False
    if not native_proposal:
        observations, reference, proposals, log_q, proposal_shared = (
            _normalize_observation_inputs(
                observations,
                reference,
                proposal_theta,
                proposal_log_density_theta,
            )
        )
    proposal_candidates = int(proposal_candidates)
    if proposal_candidates < 1:
        raise ValueError("proposal_candidates must be positive.")
    ids = (
        np.arange(1, n_observations + 1, dtype=np.int64)
        if observation_ids is None
        else np.asarray(observation_ids, dtype=np.int64).reshape(-1)
    )
    if len(ids) != n_observations or len(np.unique(ids)) != len(ids):
        raise ValueError("observation_ids must be unique and match observations.")
    posterior_samples = int(posterior_samples)
    likelihood_route_samples = int(
        posterior_samples if likelihood_route_samples is None else likelihood_route_samples
    )
    output_directory = (
        model.run_directory / "standardized_results"
        if output_directory is None
        else Path(output_directory).expanduser().resolve()
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    audit_inputs = (
        audit_reference_theta,
        audit_reference_x,
        audit_reference_ids,
        audit_marginal_theta,
        audit_marginal_x,
        audit_marginal_ids,
    )
    audit_supplied = all(value is not None for value in audit_inputs)
    if any(value is None for value in audit_inputs) and not all(
        value is None for value in audit_inputs
    ):
        raise ValueError(
            "Provide all disjoint audit reference/marginal arrays and IDs, or none."
        )
    if (normalization_theta is None) != (normalization_theta_ids is None):
        raise ValueError(
            "normalization_theta and normalization_theta_ids must be supplied together."
        )
    normalization_supplied = normalization_theta is not None
    normalization_x_per_theta = int(normalization_x_per_theta)
    if normalization_supplied and normalization_x_per_theta < 1:
        raise ValueError("normalization_x_per_theta must be positive.")

    input_contract = {
        "schema": JANA_EVALUATION_SCHEMA,
        "checkpoint_contract_sha256": model.manifest.get("training_contract_sha256"),
        "checkpoint_artifact_sha256": _checkpoint_artifact_sha256(model.manifest),
        "driver_source_sha256": _sha256_file(Path(__file__).resolve()),
        "requirements_sha256": _sha256_file(
            Path(__file__).resolve().parent / "requirements_jana.txt"
        ),
        "observation_ids": ids.tolist(),
        "observations_sha256": _array_sha256(observations),
        "reference_sha256": _array_sha256(reference),
        "proposal_theta_sha256": (
            None if native_proposal else _array_sha256(proposals)
        ),
        "proposal_log_density_theta_sha256": (
            None if native_proposal else _array_sha256(log_q)
        ),
        "proposal_source": (
            "method_native_nominal_q_phi"
            if native_proposal
            else "explicit_caller_arrays"
        ),
        "proposal_candidates": (
            proposal_candidates if native_proposal else int(proposals.shape[1])
        ),
        "proposal_shared_across_observations": proposal_shared,
        "proposal_coordinate_measure": "physical_theta",
        "posterior_samples": posterior_samples,
        "likelihood_route_samples": likelihood_route_samples,
        "seed": int(seed),
        "audit_input_sha256": (
            None
            if not audit_supplied
            else _array_sha256(
                np.asarray(audit_reference_ids, dtype=np.int64),
                np.asarray(audit_marginal_ids, dtype=np.int64),
                np.asarray(audit_reference_theta, dtype=np.float32),
                np.asarray(audit_reference_x, dtype=np.float32),
                np.asarray(audit_marginal_theta, dtype=np.float32),
                np.asarray(audit_marginal_x, dtype=np.float32),
            )
        ),
        "audit_bank_fingerprint": audit_bank_fingerprint,
        "normalization_input_sha256": (
            None
            if not normalization_supplied
            else _array_sha256(
                np.asarray(normalization_theta_ids, dtype=np.int64),
                np.asarray(normalization_theta, dtype=np.float32),
                np.asarray([normalization_x_per_theta], dtype=np.int64),
            )
        ),
        "ratio_correction": False,
    }
    evaluation_hash = _mapping_sha256(input_contract)
    manifest_path = output_directory / "evaluation_manifest.json"
    if load_if_available and manifest_path.is_file():
        saved_manifest = json.loads(manifest_path.read_text())
        if (
            saved_manifest.get("status") == "complete"
            and saved_manifest.get("evaluation_contract_sha256") == evaluation_hash
        ):
            valid_outputs = all(
                (output_directory / entry["relative_path"]).is_file()
                and _sha256_file(output_directory / entry["relative_path"])
                == entry["sha256"]
                for entry in saved_manifest.get("output_files", [])
            )
            if valid_outputs:
                return saved_manifest
        raise RuntimeError(
            f"Existing evaluation in {output_directory} is incompatible or incomplete."
        )

    direct = sample_nominal_posterior(
        model,
        observations,
        n_samples=posterior_samples,
        seed=seed,
        strict_runtime=strict_runtime,
    )
    if n_observations == 1:
        direct = direct[None, :, :]
    if native_proposal:
        proposals = sample_nominal_posterior(
            model,
            observations,
            n_samples=proposal_candidates,
            seed=int(seed) + 200_003,
            strict_runtime=strict_runtime,
        )
        if n_observations == 1:
            proposals = proposals[None, :, :]
        log_q = np.stack(
            [
                evaluate_nominal_log_posterior(
                    model,
                    proposals[index],
                    observations[index],
                    strict_runtime=strict_runtime,
                )
                for index in range(n_observations)
            ]
        )
    assert proposals is not None and log_q is not None
    rows: list[dict[str, Any]] = []
    output_files: list[Path] = []
    for index, observation_id in enumerate(ids):
        route = construct_likelihood_route_posterior(
            model,
            observations[index],
            proposals[index],
            log_q[index],
            n_samples=likelihood_route_samples,
            seed=int(seed) + 1009 * (index + 1),
            strict_runtime=strict_runtime,
        )
        comparison = exact_likelihood_comparison(
            model,
            proposals[index],
            observations[index],
            strict_runtime=strict_runtime,
        )
        learned_log_posterior = (
            log_q[index]
            if native_proposal
            else evaluate_nominal_log_posterior(
                model,
                proposals[index],
                observations[index],
                strict_runtime=strict_runtime,
            )
        )
        proposal_prior_log_density = slcp_prior_log_density(proposals[index])
        proposal_z = slcp_theta_to_latent(proposals[index], outside="nan")
        valid_residual = comparison["log_likelihood_residual"]
        valid_residual = valid_residual[np.isfinite(valid_residual)]
        agreement = _route_moment_agreement(direct[index], route["samples"])
        result_path = output_directory / f"observation_{int(observation_id):02d}_routes.npz"
        _atomic_savez(
            result_path,
            observation_id=np.asarray(observation_id, dtype=np.int64),
            observation_x=observations[index],
            reference_theta=reference[index],
            posterior_route_theta=direct[index],
            likelihood_route_theta=route["samples"],
            likelihood_proposal_theta=proposals[index],
            likelihood_proposal_z=proposal_z,
            likelihood_proposal_inside_prior=np.isfinite(proposal_z).all(axis=1),
            likelihood_proposal_log_density_theta=log_q[index],
            candidate_theta=proposals[index],
            candidate_log_q_phi_theta=learned_log_posterior,
            jana_log_posterior_theta=learned_log_posterior,
            prior_log_density_theta=proposal_prior_log_density,
            likelihood_route_log_weights=route["log_weights"],
            likelihood_route_normalized_weights=route["normalized_weights"],
            likelihood_route_resampled_indices=route["resampled_indices"],
            jana_log_likelihood_physical=comparison[
                "jana_log_likelihood_physical"
            ],
            candidate_log_q_eta_x_given_theta=comparison[
                "jana_log_likelihood_physical"
            ],
            exact_log_likelihood_physical=comparison[
                "exact_log_likelihood_physical"
            ],
            candidate_exact_log_likelihood_physical=comparison[
                "exact_log_likelihood_physical"
            ],
            log_likelihood_residual=comparison["log_likelihood_residual"],
            exact_likelihood_valid=comparison["valid"],
        )
        output_files.append(result_path)
        rows.append(
            {
                "observation_id": int(observation_id),
                "likelihood_route_proposal": input_contract["proposal_source"],
                "posterior_route_c2st_input": "reference_theta vs posterior_route_theta",
                "likelihood_route_c2st_input": "reference_theta vs likelihood_route_theta",
                "route_agreement_c2st_input": (
                    "posterior_route_theta vs likelihood_route_theta"
                ),
                "c2st_status": "pending_common_modern_runtime",
                "likelihood_route_ess": route["ess"],
                "likelihood_route_ess_fraction": route["ess_fraction"],
                "likelihood_route_max_weight": route["max_weight"],
                "likelihood_route_finite_weight_fraction": route[
                    "finite_weight_fraction"
                ],
                "likelihood_route_log_evidence": route[
                    "log_evidence_importance_estimate"
                ],
                "direct_posterior_outside_prior_fraction": float(
                    np.mean(np.any((direct[index] < PRIOR_LOW) | (direct[index] > PRIOR_HIGH), axis=1))
                ),
                "exact_likelihood_valid_fraction": float(comparison["valid"].mean()),
                "exact_likelihood_log_error_mean": (
                    float(valid_residual.mean()) if len(valid_residual) else float("nan")
                ),
                "exact_likelihood_log_error_rms": (
                    float(np.sqrt(np.mean(np.square(valid_residual))))
                    if len(valid_residual)
                    else float("nan")
                ),
                **agreement,
            }
        )

    diagnostics_path = output_directory / "route_diagnostics.csv"
    try:
        import pandas as pd

        pd.DataFrame(rows).to_csv(diagnostics_path, index=False)
    except ImportError:  # pragma: no cover - requirements always include pandas
        import csv

        with diagnostics_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    output_files.append(diagnostics_path)

    normalization_metadata = None
    if normalization_supplied:
        normalization_theta_rows = _as_rows(
            normalization_theta, POSTERIOR_DIMENSION, "normalization_theta"
        )
        normalization_ids = np.asarray(
            normalization_theta_ids, dtype=np.int64
        ).reshape(-1)
        if len(normalization_ids) != len(normalization_theta_rows):
            raise ValueError("Normalization theta rows and IDs must match.")
        normalization_x = sample_nominal_likelihood(
            model,
            normalization_theta_rows,
            n_samples=normalization_x_per_theta,
            seed=int(seed) + 910_001,
            physical_coordinates=True,
            strict_runtime=strict_runtime,
        )
        if len(normalization_theta_rows) == 1:
            normalization_x = normalization_x[None, :, :]
        repeated_theta = np.repeat(
            normalization_theta_rows, normalization_x_per_theta, axis=0
        )
        group_ids = np.repeat(
            normalization_ids, normalization_x_per_theta
        ).astype(np.int64)
        normalization_points = np.column_stack(
            [repeated_theta, normalization_x.reshape(-1, OBSERVATION_DIMENSION)]
        ).astype(np.float32)
        normalization_path = output_directory / "likelihood_normalization_inputs.npz"
        normalization_fingerprint = _array_sha256(
            normalization_ids,
            normalization_theta_rows,
            normalization_x,
            repeated_theta,
            group_ids,
            normalization_points,
        )
        _atomic_savez(
            normalization_path,
            normalization_theta_ids=normalization_ids,
            normalization_theta=normalization_theta_rows,
            normalization_x=normalization_x,
            normalization_repeated_theta=repeated_theta,
            normalization_group_ids=group_ids,
            normalization_points_raw_theta_x=normalization_points,
            normalization_fingerprint=np.asarray(normalization_fingerprint),
        )
        output_files.append(normalization_path)
        normalization_metadata = {
            "theta_rows": len(normalization_theta_rows),
            "x_per_theta": normalization_x_per_theta,
            "flat_rows": len(repeated_theta),
            "fingerprint": normalization_fingerprint,
        }

    audit_metadata = None
    if audit_supplied:
        reference_theta = _as_rows(
            audit_reference_theta, POSTERIOR_DIMENSION, "audit_reference_theta"
        )
        reference_x = _as_rows(
            audit_reference_x, OBSERVATION_DIMENSION, "audit_reference_x"
        )
        marginal_theta = _as_rows(
            audit_marginal_theta, POSTERIOR_DIMENSION, "audit_marginal_theta"
        )
        marginal_x = _as_rows(
            audit_marginal_x, OBSERVATION_DIMENSION, "audit_marginal_x"
        )
        reference_ids = np.asarray(audit_reference_ids, dtype=np.int64).reshape(-1)
        marginal_ids = np.asarray(audit_marginal_ids, dtype=np.int64).reshape(-1)
        lengths = {
            len(reference_theta),
            len(reference_x),
            len(reference_ids),
            len(marginal_theta),
            len(marginal_x),
            len(marginal_ids),
        }
        if len(lengths) != 1 or not lengths.pop():
            raise ValueError("All audit reference/marginal arrays need one common size.")
        if np.intersect1d(reference_ids, marginal_ids).size:
            raise ValueError("Audit reference and marginal row IDs must be disjoint.")
        audit = exact_likelihood_comparison(
            model, reference_theta, reference_x, strict_runtime=strict_runtime
        )
        # Bayes' rule gives
        #   log q_phi(theta|x) - log p(theta)
        #     = log q_eta(x|theta) - log p(x).
        # The additive log p(x) is constant only when x is fixed.  Therefore
        # evaluate each official observation over one common independent theta
        # grid; pairing each audit theta with its own audit x would invalidate
        # the slope and residual diagnostics.
        cycle_theta_fingerprint = _array_sha256(reference_theta)
        cycle_log_q_phi = np.stack(
            [
                evaluate_nominal_log_posterior(
                    model,
                    reference_theta,
                    observations[index],
                    strict_runtime=strict_runtime,
                )
                for index in range(n_observations)
            ]
        )
        cycle_log_q_eta = np.stack(
            [
                evaluate_nominal_log_likelihood(
                    model,
                    reference_theta,
                    observations[index],
                    physical_density=True,
                    strict_runtime=strict_runtime,
                )
                for index in range(n_observations)
            ]
        )
        cycle_log_prior = np.broadcast_to(
            slcp_prior_log_density(reference_theta),
            (n_observations, len(reference_theta)),
        ).copy()
        audit_path = output_directory / "independent_likelihood_audit.npz"
        predictive_x = sample_nominal_likelihood(
            model,
            marginal_theta,
            n_samples=1,
            seed=int(seed) + 900_001,
            physical_coordinates=True,
            strict_runtime=strict_runtime,
        ).reshape(-1, OBSERVATION_DIMENSION)
        posterior_joint_theta = sample_nominal_posterior(
            model,
            marginal_x,
            n_samples=1,
            seed=int(seed) + 900_002,
            strict_runtime=strict_runtime,
        ).reshape(-1, POSTERIOR_DIMENSION)
        row_fingerprint = _array_sha256(
            reference_ids,
            marginal_ids,
            reference_theta,
            reference_x,
            marginal_theta,
            marginal_x,
        )
        _atomic_savez(
            audit_path,
            **audit,
            audit_reference_ids=reference_ids,
            audit_reference_theta=reference_theta,
            audit_reference_x=reference_x,
            bayes_cycle_observation_ids=ids,
            bayes_cycle_observations_x=observations,
            bayes_cycle_theta_grid=reference_theta,
            bayes_cycle_log_q_phi_theta_given_x=cycle_log_q_phi,
            bayes_cycle_log_q_eta_x_given_theta=cycle_log_q_eta,
            bayes_cycle_log_prior_theta=cycle_log_prior,
            bayes_cycle_theta_grid_fingerprint=np.asarray(
                cycle_theta_fingerprint
            ),
            bayes_cycle_theta_fingerprint=np.asarray(cycle_theta_fingerprint),
            audit_marginal_ids=marginal_ids,
            audit_marginal_theta=marginal_theta,
            audit_marginal_x=marginal_x,
            jana_posterior_joint_theta=posterior_joint_theta,
            jana_predictive_x=predictive_x,
            audit_row_fingerprint=np.asarray(row_fingerprint),
            audit_bank_fingerprint=np.asarray(audit_bank_fingerprint or "unknown"),
        )
        output_files.append(audit_path)
        audit_metadata = {
            "reference_rows": len(reference_theta),
            "marginal_rows": len(marginal_theta),
            "reference_id_range": [int(reference_ids.min()), int(reference_ids.max())],
            "marginal_id_range": [int(marginal_ids.min()), int(marginal_ids.max())],
            "row_fingerprint": row_fingerprint,
            "bank_fingerprint": audit_bank_fingerprint,
            "theta_x_sha256": _array_sha256(reference_theta, reference_x),
            "valid_fraction": float(audit["valid"].mean()),
            "exact_finite_rows": int(
                np.isfinite(audit["exact_log_likelihood_physical"]).sum()
            ),
            "bayes_cycle_observations": ids.tolist(),
            "bayes_cycle_theta_rows": len(reference_theta),
            "bayes_cycle_theta_grid_fingerprint": cycle_theta_fingerprint,
            "bayes_cycle_log_q_phi_sha256": _array_sha256(cycle_log_q_phi),
            "bayes_cycle_log_q_eta_sha256": _array_sha256(cycle_log_q_eta),
            "bayes_cycle_log_prior_sha256": _array_sha256(cycle_log_prior),
            "posterior_joint_theta_sha256": _array_sha256(posterior_joint_theta),
            "predictive_x_sha256": _array_sha256(predictive_x),
        }

    manifest = {
        **input_contract,
        "status": "complete",
        "created_utc": _utc_now(),
        "run_directory": str(model.run_directory),
        "output_directory": str(output_directory),
        "evaluation_contract_sha256": evaluation_hash,
        "independent_likelihood_audit": audit_metadata,
        "likelihood_normalization_inputs": normalization_metadata,
        "route_diagnostics": rows,
        "output_files": [
            {
                "relative_path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in output_files
        ],
    }
    _atomic_write_json(manifest_path, manifest)
    return manifest


def export_nominal_ratio_class_bank(
    model_or_directory: LoadedJANA | str | Path,
    *,
    master_bank_path: str | Path,
    validation_bank_path: str | Path,
    budget: int,
    seed: int,
    output_directory: str | Path | None = None,
    context_batch_size: int = 8192,
    load_if_available: bool = True,
    strict_runtime: bool = True,
) -> dict[str, Any]:
    """Export nominal exact-JANA S/P/L banks for modern ratio training.

    All features are ``[theta, x]`` in raw physical coordinates.  ``S`` uses
    the genuine fixed pairs, ``P`` replaces theta by one nominal JANA
    posterior draw at the same x, and ``L`` replaces x by one nominal JANA
    likelihood draw at the same theta.  The dedicated 300-row JANA validation
    bank is transformed analogously and never enters the N-row training bank.
    This function exports data only; it does not fit or apply a ratio model.
    """

    model = _resolve_loaded(model_or_directory, strict_runtime=strict_runtime)
    budget = int(budget)
    context_batch_size = int(context_batch_size)
    if budget < 1 or context_batch_size < 1:
        raise ValueError("budget and context_batch_size must be positive.")
    master = load_slcp_bank(master_bank_path, expected_role="master")
    validation = load_slcp_bank(
        validation_bank_path, expected_role="jana_validation"
    )
    if budget > len(master.theta):
        raise ValueError(f"N={budget} exceeds the master-bank size.")
    if len(validation.theta) < UPSTREAM_VALIDATION_ROWS:
        raise ValueError("The JANA validation bank has fewer than 300 rows.")
    checkpoint_master = model.manifest.get("master_bank", {}).get("content_sha256")
    checkpoint_validation = model.manifest.get("validation_bank", {}).get(
        "content_sha256"
    )
    if checkpoint_master != master.content_fingerprint:
        raise RuntimeError("Ratio export master bank differs from JANA training.")
    if checkpoint_validation != validation.content_fingerprint:
        raise RuntimeError("Ratio export validation bank differs from JANA training.")
    if int(model.manifest.get("requested_budget", -1)) != budget:
        raise RuntimeError("Ratio export budget differs from the JANA checkpoint.")
    if int(model.manifest.get("seed", -1)) != int(seed):
        raise RuntimeError("Ratio export seed differs from the JANA checkpoint.")
    expected_ids = np.arange(budget, dtype=np.int64)
    if (
        model.manifest.get("paper_exact_validation_protocol") is not True
        or int(model.manifest.get("training_rows", -1)) != budget
        or model.manifest.get("training_index_sha256")
        != _array_sha256(expected_ids)
        or model.manifest.get("training_array_sha256")
        != _array_sha256(master.theta[expected_ids], master.x[expected_ids])
    ):
        raise RuntimeError(
            "Ratio export requires the exact ordered full-prefix JANA checkpoint."
        )

    output_directory = (
        model.run_directory / "ratio_class_bank"
        if output_directory is None
        else Path(output_directory).expanduser().resolve()
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    arrays_path = output_directory / "nominal_spl_class_bank.npz"
    manifest_path = output_directory / "manifest.json"
    contract = {
        "schema": JANA_RUNTIME_SCHEMA,
        "purpose": "nominal_exact_jana_ratio_training_bank",
        "checkpoint_contract_sha256": model.manifest.get(
            "training_contract_sha256"
        ),
        "checkpoint_manifest_path": str(
            model.run_directory / "checkpoint_manifest.json"
        ),
        "checkpoint_artifact_sha256": _checkpoint_artifact_sha256(model.manifest),
        "driver_source_sha256": _sha256_file(Path(__file__).resolve()),
        "requirements_sha256": _sha256_file(
            Path(__file__).resolve().parent / "requirements_jana.txt"
        ),
        "budget": budget,
        "seed": int(seed),
        "training_rows": budget,
        "validation_rows": UPSTREAM_VALIDATION_ROWS,
        "master_content_sha256": master.content_fingerprint,
        "validation_content_sha256": validation.content_fingerprint,
        "feature_coordinates": "raw_physical_theta_plus_raw_physical_x",
        "class_order": ["S", "P", "L"],
        "proposal": "nominal_exact_jana_no_broadening_no_prior_defense",
        "ratio_training": False,
        "context_batch_size": context_batch_size,
        "sample_batch_size": 8192,
        "sampling_seeds": {
            "training_P": int(seed) + 110_001,
            "training_L": int(seed) + 120_001,
            "validation_P": int(seed) + 130_001,
            "validation_L": int(seed) + 140_001,
        },
    }
    contract_hash = _mapping_sha256(contract)
    if load_if_available and arrays_path.is_file() and manifest_path.is_file():
        saved = json.loads(manifest_path.read_text())
        if (
            saved.get("status") == "complete"
            and saved.get("contract_sha256") == contract_hash
            and saved.get("arrays_sha256") == _sha256_file(arrays_path)
        ):
            return saved
        raise RuntimeError(f"Incompatible cached exact-JANA ratio bank: {arrays_path}")

    training_ids = expected_ids
    validation_ids = np.arange(UPSTREAM_VALIDATION_ROWS, dtype=np.int64)
    theta_train = master.theta[training_ids]
    x_train = master.x[training_ids]
    theta_validation = validation.theta[validation_ids]
    x_validation = validation.x[validation_ids]
    theta_p_train = sample_nominal_posterior(
        model,
        x_train,
        n_samples=1,
        seed=contract["sampling_seeds"]["training_P"],
        context_batch_size=context_batch_size,
        strict_runtime=strict_runtime,
    ).reshape(budget, POSTERIOR_DIMENSION)
    x_l_train = sample_nominal_likelihood(
        model,
        theta_train,
        n_samples=1,
        seed=contract["sampling_seeds"]["training_L"],
        physical_coordinates=True,
        context_batch_size=context_batch_size,
        strict_runtime=strict_runtime,
    ).reshape(budget, OBSERVATION_DIMENSION)
    theta_p_validation = sample_nominal_posterior(
        model,
        x_validation,
        n_samples=1,
        seed=contract["sampling_seeds"]["validation_P"],
        context_batch_size=context_batch_size,
        strict_runtime=strict_runtime,
    ).reshape(UPSTREAM_VALIDATION_ROWS, POSTERIOR_DIMENSION)
    x_l_validation = sample_nominal_likelihood(
        model,
        theta_validation,
        n_samples=1,
        seed=contract["sampling_seeds"]["validation_L"],
        physical_coordinates=True,
        context_batch_size=context_batch_size,
        strict_runtime=strict_runtime,
    ).reshape(UPSTREAM_VALIDATION_ROWS, OBSERVATION_DIMENSION)

    train_s = np.column_stack([theta_train, x_train]).astype(np.float32)
    train_p = np.column_stack([theta_p_train, x_train]).astype(np.float32)
    train_l = np.column_stack([theta_train, x_l_train]).astype(np.float32)
    validation_s = np.column_stack([theta_validation, x_validation]).astype(
        np.float32
    )
    validation_p = np.column_stack(
        [theta_p_validation, x_validation]
    ).astype(np.float32)
    validation_l = np.column_stack(
        [theta_validation, x_l_validation]
    ).astype(np.float32)
    content_fingerprint = _array_sha256(
        training_ids,
        validation_ids,
        train_s,
        train_p,
        train_l,
        validation_s,
        validation_p,
        validation_l,
    )
    _atomic_savez(
        arrays_path,
        train_S=train_s,
        train_P=train_p,
        train_L=train_l,
        validation_S=validation_s,
        validation_P=validation_p,
        validation_L=validation_l,
        training_source_ids=training_ids,
        validation_source_ids=validation_ids,
        feature_order=np.asarray(
            [
                "theta_1",
                "theta_2",
                "theta_3",
                "theta_4",
                "theta_5",
                "x_1",
                "x_2",
                "x_3",
                "x_4",
                "x_5",
                "x_6",
                "x_7",
                "x_8",
            ]
        ),
        content_fingerprint=np.asarray(content_fingerprint),
    )
    manifest = {
        **contract,
        "status": "complete",
        "contract_sha256": contract_hash,
        "content_fingerprint": content_fingerprint,
        "arrays_path": str(arrays_path),
        "arrays_sha256": _sha256_file(arrays_path),
        "arrays_bytes": arrays_path.stat().st_size,
        "created_utc": _utc_now(),
    }
    _atomic_write_json(manifest_path, manifest)
    return manifest


def run_jana_campaign(
    *,
    artifact_root: str | Path,
    master_bank_path: str | Path,
    shape_bank_path: str | Path,
    pilot_bank_path: str | Path,
    validation_bank_path: str | Path,
    budgets: Sequence[int] = PAPER_BUDGETS,
    seeds: Sequence[int] = (31082026,),
    split_paths: Mapping[int, str | Path] | None = None,
    validation_mode: str = "external_paper",
    profile: str = "PAPER",
    epochs: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    load_if_available: bool = True,
    force: bool = False,
    strict_runtime: bool = True,
) -> dict[str, Any]:
    """Train/reuse the fixed-bank JANA grid inside the legacy runtime."""

    budgets = tuple(int(value) for value in budgets)
    seeds = tuple(int(value) for value in seeds)
    if not budgets or not seeds:
        raise ValueError("budgets and seeds must be non-empty.")
    profile = str(profile).upper()
    training_epochs = int(
        (2 if profile == "SMOKE" else DEFAULT_EPOCHS) if epochs is None else epochs
    )
    manifests = []
    for budget in budgets:
        for seed in seeds:
            manifests.append(
                train_exact_jana(
                    master_bank_path,
                    artifact_root=artifact_root,
                    budget=budget,
                    seed=seed,
                    shape_bank_path=shape_bank_path,
                    pilot_bank_path=pilot_bank_path,
                    validation_bank_path=validation_bank_path,
                    validation_mode=validation_mode,
                    split_path=None if split_paths is None else split_paths.get(budget),
                    epochs=training_epochs,
                    batch_size=batch_size,
                    load_if_available=load_if_available,
                    force=force,
                    strict_runtime=strict_runtime,
                )
            )
    selected_runs = [
        {
            "budget": manifest["requested_budget"],
            "seed": manifest["seed"],
            "run_directory": manifest["run_directory"],
            "training_contract_sha256": manifest[
                "training_contract_sha256"
            ],
            "paper_exact_validation_protocol": manifest[
                "paper_exact_validation_protocol"
            ],
            "simulation_accounting": manifest["simulation_accounting"],
        }
        for manifest in manifests
    ]
    summary = {
        "schema": JANA_RUNTIME_SCHEMA,
        "status": "complete",
        "profile": profile,
        "budgets": list(budgets),
        "seeds": list(seeds),
        "master_bank_path": str(Path(master_bank_path).expanduser().resolve()),
        "shape_bank_path": str(Path(shape_bank_path).expanduser().resolve()),
        "pilot_bank_path": str(Path(pilot_bank_path).expanduser().resolve()),
        "validation_bank_path": str(
            Path(validation_bank_path).expanduser().resolve()
        ),
        "validation_mode": validation_mode,
        "runs": selected_runs,
        "created_utc": _utc_now(),
    }
    summary_path = (
        Path(artifact_root).expanduser().resolve()
        / "jana_paper"
        / "campaign_manifest.json"
    )
    # Several literal N=1M runs may be sharded across jobs.  Preserve every
    # completed budget/seed entry instead of allowing the last subprocess to
    # overwrite the campaign-level provenance written by another shard.
    import fcntl

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.with_suffix(".lock").open("a+") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        keyed: dict[tuple[int, int], dict[str, Any]] = {}
        if summary_path.is_file():
            try:
                previous = json.loads(summary_path.read_text())
            except (OSError, json.JSONDecodeError):
                previous = {}
            if previous.get("schema") == JANA_RUNTIME_SCHEMA:
                for run in previous.get("runs", []):
                    try:
                        keyed[(int(run["budget"]), int(run["seed"]))] = run
                    except (KeyError, TypeError, ValueError):
                        continue
        for run in selected_runs:
            keyed[(int(run["budget"]), int(run["seed"]))] = run
        summary["runs"] = [keyed[key] for key in sorted(keyed)]
        summary["completed_pairs"] = [
            f"{budget}:{seed}" for budget, seed in sorted(keyed)
        ]
        summary["updated_utc"] = _utc_now()
        _atomic_write_json(summary_path, summary)
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
    return summary


def launch_isolated_campaign(
    python_executable: str | Path,
    *,
    artifact_root: str | Path,
    master_bank_path: str | Path,
    shape_bank_path: str | Path,
    pilot_bank_path: str | Path,
    validation_bank_path: str | Path,
    budgets: Sequence[int],
    seeds: Sequence[int],
    profile: str = "PAPER",
    load_if_available: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """Launch this file with the explicitly selected legacy Python runtime."""

    command = [
        str(Path(python_executable).expanduser()),
        str(Path(__file__).resolve()),
        "campaign",
        "--artifact-root",
        str(Path(artifact_root).expanduser().resolve()),
        "--master-bank",
        str(Path(master_bank_path).expanduser().resolve()),
        "--shape-bank",
        str(Path(shape_bank_path).expanduser().resolve()),
        "--pilot-bank",
        str(Path(pilot_bank_path).expanduser().resolve()),
        "--validation-bank",
        str(Path(validation_bank_path).expanduser().resolve()),
        "--budgets",
        *[str(int(value)) for value in budgets],
        "--seeds",
        *[str(int(value)) for value in seeds],
        "--profile",
        str(profile),
    ]
    if not load_if_available:
        command.append("--no-load-if-available")
    if force:
        command.append("--force")
    subprocess.run(command, check=True)
    summary_path = (
        Path(artifact_root).expanduser().resolve()
        / "jana_paper"
        / "campaign_manifest.json"
    )
    if not summary_path.is_file():
        raise RuntimeError("The isolated JANA process returned no campaign manifest.")
    return json.loads(summary_path.read_text())


def resolve_jana_python(artifact_root: str | Path) -> Path:
    """Resolve and smoke-check the explicitly isolated Python 3.11 runtime."""

    artifact_root = Path(artifact_root).expanduser().resolve()
    configured = os.environ.get("PAPER_SUMMARY_JANA_PYTHON")
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    configured_environment = os.environ.get("PAPER_SUMMARY_JANA_ENV")
    if configured_environment:
        candidates.append(
            Path(configured_environment).expanduser() / "bin" / "python"
        )
    # A virtual environment stored on a mounted Google Drive is both very
    # slow and frequently non-portable across Colab runtimes.  Prefer a fresh
    # runtime-local environment there unless the caller explicitly selected a
    # location with PAPER_SUMMARY_JANA_ENV.
    if not configured_environment and Path("/content").is_dir():
        candidates.append(Path("/content") / "paper_summary_jana_env" / "bin" / "python")
    candidates.extend(
        [
            artifact_root / "envs" / "jana" / "bin" / "python",
            Path(__file__).resolve().parent / ".venv_jana" / "bin" / "python",
        ]
    )
    failures = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        check = subprocess.run(
            [
                str(candidate.resolve()),
                "-c",
                (
                    "import json,numpy,tensorflow,bayesflow,platform; "
                    "from importlib.metadata import distribution,version; "
                    "d=json.loads(distribution('bayesflow').read_text('direct_url.json')); "
                    "c=d.get('vcs_info',{}).get('commit_id'); "
                    f"assert platform.python_version_tuple()[:2]==('3','11'); "
                    f"assert numpy.__version__=='{EXPECTED_NUMPY}'; "
                    f"assert tensorflow.__version__=='{EXPECTED_TENSORFLOW}'; "
                    f"assert version('tensorflow-probability')=='{EXPECTED_TENSORFLOW_PROBABILITY}'; "
                    f"assert version('keras')=='{EXPECTED_KERAS}'; "
                    f"assert c=='{JANA_BAYESFLOW_COMMIT}'; "
                    "print(json.dumps({'python':platform.python_version(),"
                    "'numpy':numpy.__version__,'tensorflow':tensorflow.__version__,"
                    "'bayesflow_commit':c}))"
                ),
            ],
            text=True,
            capture_output=True,
        )
        if check.returncode == 0:
            return candidate.resolve()
        failures.append(f"{candidate}: {check.stderr.strip()[-500:]}")
    requirements = Path(__file__).resolve().parent / "requirements_jana.txt"
    detail = "\n".join(failures) if failures else "no candidate interpreter found"
    raise RuntimeError(
        "Exact JANA needs an isolated Python 3.11 environment. Create it with:\n"
        "  python3.11 -m venv /path/to/jana-env\n"
        f"  /path/to/jana-env/bin/python -m pip install -r {requirements}\n"
        "  export PAPER_SUMMARY_JANA_ENV=/path/to/jana-env\n"
        "  export PAPER_SUMMARY_JANA_PYTHON=/path/to/jana-env/bin/python\n"
        "The JANA code does not use JAX; if a pre-existing environment has an "
        "incompatible jax/jaxlib pair, uninstall both there.\n"
        f"Runtime checks: {detail}"
    )


def ensure_jana_environment(
    artifact_root: str | Path, install_if_missing: bool = True
) -> Path:
    """Resolve or create the isolated pinned Python 3.11 JANA environment.

    Installation is an explicit notebook action.  The helper prefers ``uv``
    because it can provision Python 3.11 in Colab without changing the modern
    kernel.  If ``uv`` is absent and installation is allowed, it is installed
    into the current (modern) environment first.  Colab uses the transient
    ``/content/paper_summary_jana_env`` by default so a Drive-backed artifact
    root never becomes a persistent virtual environment; callers can select
    another location explicitly with ``PAPER_SUMMARY_JANA_ENV``.
    """

    artifact_root = Path(artifact_root).expanduser().resolve()
    try:
        interpreter = resolve_jana_python(artifact_root)
        os.environ["PAPER_SUMMARY_JANA_PYTHON"] = str(interpreter)
        return interpreter
    except RuntimeError as initial_error:
        if not install_if_missing:
            raise
        initial_error_message = str(initial_error)

    requirements = Path(__file__).resolve().parent / "requirements_jana.txt"
    if not requirements.is_file():
        raise FileNotFoundError(f"Missing exact-JANA requirements: {requirements}")
    configured_environment = os.environ.get("PAPER_SUMMARY_JANA_ENV")
    if configured_environment:
        environment = Path(configured_environment).expanduser().resolve()
    elif Path("/content").is_dir():
        environment = Path("/content/paper_summary_jana_env")
    else:
        environment = artifact_root / "envs" / "jana"
    interpreter = environment / "bin" / "python"
    environment.parent.mkdir(parents=True, exist_ok=True)

    uv_executable = shutil.which("uv")
    if uv_executable is None:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "uv"], check=True
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                "Could not install uv while creating the isolated JANA runtime."
            ) from error
        uv_prefix = [sys.executable, "-m", "uv"]
    else:
        uv_prefix = [uv_executable]

    try:
        python311_usable = False
        if interpreter.is_file():
            probe = subprocess.run(
                [
                    str(interpreter),
                    "-c",
                    (
                        "import sys; "
                        "raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)"
                    ),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            python311_usable = probe.returncode == 0
        if not python311_usable:
            resolved_environment = environment.resolve()
            source_directory = Path(__file__).resolve().parent
            protected = {
                Path("/").resolve(),
                Path.home().resolve(),
                artifact_root,
                Path("/content").resolve(),
            }
            if (
                resolved_environment in protected
                or source_directory.is_relative_to(resolved_environment)
                or artifact_root.is_relative_to(resolved_environment)
            ):
                raise RuntimeError(
                    "Refusing to clear an unsafe PAPER_SUMMARY_JANA_ENV target: "
                    f"{resolved_environment}"
                )
            venv_command = [*uv_prefix, "venv", "--python", "3.11"]
            if environment.exists() or environment.is_symlink():
                # Recover safely from a stale/broken Colab venv without a broad
                # recursive deletion.  uv limits --clear to this validated path.
                venv_command.append("--clear")
            subprocess.run([*venv_command, str(environment)], check=True)
        subprocess.run(
            [
                *uv_prefix,
                "pip",
                "install",
                "--python",
                str(interpreter),
                "--requirement",
                str(requirements),
            ],
            check=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "Failed to create/install the pinned exact-JANA environment at "
            f"{environment}. The original resolution error was: "
            f"{initial_error_message}"
        ) from error

    os.environ["PAPER_SUMMARY_JANA_PYTHON"] = str(interpreter)
    validated = resolve_jana_python(artifact_root)
    _atomic_write_json(
        environment / "paper_summary_environment.json",
        {
            "schema": JANA_RUNTIME_SCHEMA,
            "python": str(validated),
            "requirements": str(requirements),
            "requirements_sha256": _sha256_file(requirements),
            "bayesflow_commit": JANA_BAYESFLOW_COMMIT,
            "tensorflow": EXPECTED_TENSORFLOW,
            "tensorflow_probability": EXPECTED_TENSORFLOW_PROBABILITY,
            "keras": EXPECTED_KERAS,
            "numpy": EXPECTED_NUMPY,
            "created_utc": _utc_now(),
        },
    )
    return validated


def _resolve_manifest_path(path: str | Path, artifact_root: Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = artifact_root / resolved
    return resolved.resolve()


def _load_campaign_banks(artifact_root: Path) -> dict[str, Any]:
    manifest_path = artifact_root / "manifests" / "slcp_banks.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}; run 00_prepare_SLCP_samples.ipynb first."
        )
    manifest = json.loads(manifest_path.read_text())
    required = ("master", "jana_shape", "jana_pilot", "jana_validation", "audit")
    missing = [key for key in required if key not in manifest]
    if missing:
        raise RuntimeError(
            f"SLCP bank manifest is missing {missing}; regenerate it with the "
            "current paper-summary bank preparation."
        )
    return {
        key: {
            **manifest[key],
            "path": str(_resolve_manifest_path(manifest[key]["path"], artifact_root)),
        }
        for key in required
    }


def _prepare_evaluation_input(
    artifact_root: Path,
    campaign: Mapping[str, Any],
    *,
    banks: Mapping[str, Any],
    load_if_available: bool,
) -> Path:
    """Create modern-runtime SBIBM references and independent audit inputs."""

    try:
        import sbibm
        try:
            from . import config as paper_config
        except ImportError:
            import config as paper_config
    except ImportError as error:
        raise RuntimeError(
            "The modern paper runtime needs sbibm==1.1.0 to prepare exact-JANA "
            "observations and reference posteriors."
        ) from error
    signature = paper_config.campaign_signature(campaign)
    input_root = artifact_root / "jana_paper" / "evaluation_inputs"
    input_root.mkdir(parents=True, exist_ok=True)
    input_path = input_root / f"slcp_routes_{signature}.npz"
    manifest_path = input_root / f"slcp_routes_{signature}.json"
    observation_ids = np.asarray(campaign["observations"], dtype=np.int64)
    requested_audit_rows = int(
        campaign.get("audit_joint_samples", campaign["metric_max_samples"])
    )
    audit_rows = min(requested_audit_rows, int(banks["audit"]["size"]) // 2)
    if audit_rows < 32:
        raise ValueError(
            "The independent audit bank needs at least twice 32 rows for "
            "disjoint reference/marginal closure diagnostics."
        )
    normalization_rows = min(
        int(campaign["normalization_theta"]), int(banks["audit"]["size"])
    )
    normalization_x_per_theta = int(campaign["normalization_x_per_theta"])
    contract = {
        "schema": JANA_EVALUATION_SCHEMA,
        "campaign_signature": signature,
        "observation_ids": observation_ids.tolist(),
        "proposal": "generated per method and observation in isolated evaluation",
        "proposal_candidates": int(campaign["proposal_candidates"]),
        "audit_reference_rows": audit_rows,
        "audit_marginal_rows": audit_rows,
        "audit_row_convention": "first_n_reference_last_n_marginals_disjoint",
        "audit_bank_fingerprint": banks["audit"].get("fingerprint"),
        "normalization_theta_rows": normalization_rows,
        "normalization_x_per_theta": normalization_x_per_theta,
    }
    contract_hash = _mapping_sha256(contract)
    if load_if_available and input_path.is_file() and manifest_path.is_file():
        saved = json.loads(manifest_path.read_text())
        if (
            saved.get("contract_sha256") == contract_hash
            and saved.get("file_sha256") == _sha256_file(input_path)
        ):
            return input_path
        raise RuntimeError(f"Incompatible cached JANA evaluation input: {input_path}")

    task = sbibm.get_task("slcp")
    observations = []
    references = []
    for observation_id in observation_ids:
        observations.append(
            task.get_observation(num_observation=int(observation_id))
            .detach()
            .cpu()
            .numpy()
            .reshape(OBSERVATION_DIMENSION)
            .astype(np.float32)
        )
        references.append(
            task.get_reference_posterior_samples(num_observation=int(observation_id))
            .detach()
            .cpu()
            .numpy()
            .reshape(-1, POSTERIOR_DIMENSION)
            .astype(np.float32)
        )
    reference_lengths = {len(values) for values in references}
    if len(reference_lengths) != 1:
        raise RuntimeError("SBIBM reference posterior banks have unequal lengths.")
    audit = load_slcp_bank(banks["audit"]["path"], expected_role="audit")
    reference_ids = np.arange(audit_rows, dtype=np.int64)
    marginal_ids = np.arange(len(audit.theta) - audit_rows, len(audit.theta), dtype=np.int64)
    audit_row_fingerprint = _array_sha256(
        reference_ids,
        marginal_ids,
        audit.theta[reference_ids],
        audit.x[reference_ids],
        audit.theta[marginal_ids],
        audit.x[marginal_ids],
    )
    _atomic_savez(
        input_path,
        observation_ids=observation_ids,
        observations=np.stack(observations),
        reference_posterior_samples=np.stack(references),
        audit_reference_ids=reference_ids,
        audit_reference_theta=audit.theta[reference_ids],
        audit_reference_x=audit.x[reference_ids],
        audit_marginal_ids=marginal_ids,
        audit_marginal_theta=audit.theta[marginal_ids],
        audit_marginal_x=audit.x[marginal_ids],
        audit_bank_fingerprint=np.asarray(audit.content_fingerprint),
        audit_row_fingerprint=np.asarray(audit_row_fingerprint),
        normalization_theta_ids=np.arange(normalization_rows, dtype=np.int64),
        normalization_theta=audit.theta[:normalization_rows],
        normalization_x_per_theta=np.asarray(
            normalization_x_per_theta, dtype=np.int64
        ),
    )
    _atomic_write_json(
        manifest_path,
        {
            **contract,
            "contract_sha256": contract_hash,
            "file_sha256": _sha256_file(input_path),
            "audit_row_fingerprint": audit_row_fingerprint,
            "created_utc": _utc_now(),
        },
    )
    return input_path


def _launch_isolated_evaluation(
    python_executable: Path,
    *,
    run_directory: Path,
    input_path: Path,
    output_directory: Path,
    posterior_samples: int,
    proposal_candidates: int,
    seed: int,
    load_if_available: bool,
) -> Mapping[str, Any]:
    command = [
        str(python_executable),
        str(Path(__file__).resolve()),
        "evaluate",
        "--run-directory",
        str(run_directory),
        "--input",
        str(input_path),
        "--output-directory",
        str(output_directory),
        "--posterior-samples",
        str(int(posterior_samples)),
        "--proposal-candidates",
        str(int(proposal_candidates)),
        "--likelihood-route-samples",
        str(int(posterior_samples)),
        "--seed",
        str(int(seed)),
    ]
    if not load_if_available:
        command.append("--no-load-if-available")
    subprocess.run(command, check=True)
    manifest_path = output_directory / "evaluation_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Isolated JANA evaluation did not write {manifest_path}.")
    return json.loads(manifest_path.read_text())


def _launch_isolated_ratio_export(
    python_executable: Path,
    *,
    run_directory: Path,
    master_bank_path: Path,
    validation_bank_path: Path,
    budget: int,
    seed: int,
    output_directory: Path,
    context_batch_size: int,
    load_if_available: bool,
) -> Mapping[str, Any]:
    command = [
        str(python_executable),
        str(Path(__file__).resolve()),
        "export-ratio-bank",
        "--run-directory",
        str(run_directory),
        "--master-bank",
        str(master_bank_path),
        "--validation-bank",
        str(validation_bank_path),
        "--budget",
        str(int(budget)),
        "--seed",
        str(int(seed)),
        "--output-directory",
        str(output_directory),
        "--context-batch-size",
        str(int(context_batch_size)),
    ]
    if not load_if_available:
        command.append("--no-load-if-available")
    subprocess.run(command, check=True)
    manifest_path = output_directory / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Isolated ratio export did not write {manifest_path}.")
    return json.loads(manifest_path.read_text())


def _campaign_subset(
    campaign: Mapping[str, Any],
    budgets_to_run: Sequence[int] | None,
    ml_seeds_to_run: Sequence[int] | None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    campaign_budgets = tuple(int(value) for value in campaign["budgets"])
    campaign_seeds = tuple(int(value) for value in campaign["ml_seeds"])
    budgets = (
        campaign_budgets
        if budgets_to_run is None
        else tuple(int(value) for value in budgets_to_run)
    )
    seeds = (
        campaign_seeds
        if ml_seeds_to_run is None
        else tuple(int(value) for value in ml_seeds_to_run)
    )
    if not budgets or not seeds:
        raise ValueError("Requested JANA budget and seed subsets must be non-empty.")
    unknown_budgets = sorted(set(budgets).difference(campaign_budgets))
    unknown_seeds = sorted(set(seeds).difference(campaign_seeds))
    if unknown_budgets or unknown_seeds:
        raise ValueError(
            f"JANA shard is outside the campaign: budgets={unknown_budgets}, "
            f"seeds={unknown_seeds}."
        )
    if len(set(budgets)) != len(budgets) or len(set(seeds)) != len(seeds):
        raise ValueError("JANA shard budgets/seeds must not contain duplicates.")
    return budgets, seeds


def _evaluate_saved_routes_modern(
    *,
    artifact_root: Path,
    campaign: Mapping[str, Any],
    budget: int,
    ml_seed: int,
    evaluation_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Apply the campaign's common PyTorch/SBIBM metrics to legacy arrays."""

    try:
        try:
            from . import config as paper_config
            from . import utils as paper_utils
        except ImportError:
            import config as paper_config
            import utils as paper_utils
    except ImportError as error:
        raise RuntimeError(
            "Exact-JANA arrays were produced, but the modern metric runtime "
            "could not import local config.py/utils.py."
        ) from error
    output_directory = Path(evaluation_manifest["output_directory"])
    normalization_path = output_directory / "likelihood_normalization_inputs.npz"
    if not normalization_path.is_file():
        raise FileNotFoundError(
            f"Missing exact-JANA normalization inputs: {normalization_path}"
        )
    max_samples = int(campaign["metric_max_samples"])
    signature = paper_config.campaign_signature(campaign)
    audit_path = output_directory / "independent_likelihood_audit.npz"
    with np.load(audit_path, allow_pickle=False) as audit:
        audit_reference_theta = np.asarray(
            audit["audit_reference_theta"], dtype=np.float32
        )
        audit_reference_x = np.asarray(
            audit["audit_reference_x"], dtype=np.float32
        )
        audit_marginal_theta = np.asarray(
            audit["audit_marginal_theta"], dtype=np.float32
        )
        audit_marginal_x = np.asarray(
            audit["audit_marginal_x"], dtype=np.float32
        )
        posterior_joint_theta = np.asarray(
            audit["jana_posterior_joint_theta"], dtype=np.float32
        )
        predictive_x = np.asarray(audit["jana_predictive_x"], dtype=np.float32)
        audit_residual = np.asarray(audit["log_likelihood_residual"], dtype=np.float64)
        audit_valid = np.asarray(audit["valid"], dtype=bool)
        audit_learned_log_likelihood = np.asarray(
            audit["jana_log_likelihood_physical"], dtype=np.float64
        )
        audit_exact_log_likelihood = np.asarray(
            audit["exact_log_likelihood_physical"], dtype=np.float64
        )
        cycle_observation_ids = np.asarray(
            audit["bayes_cycle_observation_ids"], dtype=np.int64
        )
        cycle_observations_x = np.asarray(
            audit["bayes_cycle_observations_x"], dtype=np.float32
        )
        cycle_theta_grid = np.asarray(
            audit["bayes_cycle_theta_grid"], dtype=np.float32
        )
        cycle_log_q_phi = np.asarray(
            audit["bayes_cycle_log_q_phi_theta_given_x"], dtype=np.float64
        )
        cycle_log_q_eta = np.asarray(
            audit["bayes_cycle_log_q_eta_x_given_theta"], dtype=np.float64
        )
        cycle_log_prior = np.asarray(
            audit["bayes_cycle_log_prior_theta"], dtype=np.float64
        )
        cycle_theta_fingerprint = str(
            audit["bayes_cycle_theta_grid_fingerprint"]
        )
        audit_row_fingerprint = str(audit["audit_row_fingerprint"])
        audit_bank_fingerprint = str(audit["audit_bank_fingerprint"])
    reference_joint = np.column_stack(
        [audit_reference_theta, audit_reference_x]
    )
    likelihood_audit_fingerprint = _array_sha256(
        audit_reference_theta, audit_reference_x
    )
    expected_cycle_ids = np.asarray(campaign["observations"], dtype=np.int64)
    expected_cycle_shape = (len(expected_cycle_ids), len(audit_reference_theta))
    if (
        not np.array_equal(cycle_observation_ids, expected_cycle_ids)
        or cycle_observations_x.shape
        != (len(expected_cycle_ids), OBSERVATION_DIMENSION)
        or cycle_theta_grid.shape
        != (len(audit_reference_theta), POSTERIOR_DIMENSION)
        or not np.array_equal(cycle_theta_grid, audit_reference_theta)
        or cycle_log_q_phi.shape != expected_cycle_shape
        or cycle_log_q_eta.shape != expected_cycle_shape
        or cycle_log_prior.shape != expected_cycle_shape
        or cycle_theta_fingerprint != _array_sha256(cycle_theta_grid)
    ):
        raise RuntimeError(
            "Exact-JANA fixed-observation Bayes-cycle arrays are inconsistent."
        )
    cycle_index_by_observation = {
        int(observation_id): index
        for index, observation_id in enumerate(cycle_observation_ids)
    }
    posterior_joint = np.column_stack([posterior_joint_theta, audit_marginal_x])
    likelihood_joint = np.column_stack([audit_marginal_theta, predictive_x])
    audit_seed = int(ml_seed) + 70_000
    posterior_joint_metrics = paper_utils.distribution_metrics(
        reference_joint,
        posterior_joint,
        seed=audit_seed + 5,
        max_samples=max_samples,
    )
    predictive_x_metrics = paper_utils.distribution_metrics(
        audit_reference_x,
        predictive_x,
        seed=audit_seed + 6,
        max_samples=max_samples,
    )
    predictive_joint_metrics = paper_utils.distribution_metrics(
        reference_joint,
        likelihood_joint,
        seed=audit_seed + 7,
        max_samples=max_samples,
    )
    exact_finite_audit = np.isfinite(audit_exact_log_likelihood)
    learned_failure = exact_finite_audit & ~np.isfinite(
        audit_learned_log_likelihood
    )
    residual_failure = exact_finite_audit & ~np.isfinite(audit_residual)
    if learned_failure.any() or residual_failure.any():
        raise FloatingPointError(
            "Exact-JANA learned likelihood is non-finite on finite common-audit "
            "rows; this is a model failure, not a row to exclude."
        )
    if not np.array_equal(
        audit_valid,
        exact_finite_audit & np.isfinite(audit_learned_log_likelihood),
    ):
        raise RuntimeError("Exact-JANA likelihood-audit validity mask is inconsistent.")
    finite_audit = exact_finite_audit
    audit_rms = (
        float(np.sqrt(np.mean(np.square(audit_residual[finite_audit]))))
        if finite_audit.any()
        else float("nan")
    )
    audit_centered_rms = (
        float(
            np.sqrt(
                np.mean(
                    np.square(
                        audit_residual[finite_audit]
                        - np.mean(audit_residual[finite_audit])
                    )
                )
            )
        )
        if finite_audit.any()
        else float("nan")
    )
    audit_finite_rows = int(exact_finite_audit.sum())
    rows = []
    for observation_id in campaign["observations"]:
        observation_id = int(observation_id)
        cycle_index = cycle_index_by_observation[observation_id]
        audit_cycle_valid = (
            np.isfinite(cycle_log_q_phi[cycle_index])
            & np.isfinite(cycle_log_q_eta[cycle_index])
            & np.isfinite(cycle_log_prior[cycle_index])
        )
        audit_cycle = paper_utils._cycle_diagnostics(
            cycle_log_q_phi[cycle_index, audit_cycle_valid]
            - cycle_log_prior[cycle_index, audit_cycle_valid],
            cycle_log_q_eta[cycle_index, audit_cycle_valid],
        )
        result_path = output_directory / f"observation_{observation_id:02d}_routes.npz"
        with np.load(result_path, allow_pickle=False) as saved:
            result_observation = np.asarray(
                saved["observation_x"], dtype=np.float32
            ).reshape(OBSERVATION_DIMENSION)
            reference = np.asarray(saved["reference_theta"], dtype=np.float32)
            posterior = np.asarray(saved["posterior_route_theta"], dtype=np.float32)
            likelihood = np.asarray(saved["likelihood_route_theta"], dtype=np.float32)
            weights = np.asarray(
                saved["likelihood_route_normalized_weights"], dtype=np.float64
            )
            log_q_phi = np.asarray(saved["jana_log_posterior_theta"], dtype=np.float64)
            log_q_eta = np.asarray(
                saved["jana_log_likelihood_physical"], dtype=np.float64
            )
            log_prior = np.asarray(saved["prior_log_density_theta"], dtype=np.float64)
            residual = np.asarray(saved["log_likelihood_residual"], dtype=np.float64)
            valid = np.asarray(saved["exact_likelihood_valid"], dtype=bool)
        if not np.array_equal(
            result_observation, cycle_observations_x[cycle_index]
        ):
            raise RuntimeError(
                f"Bayes-cycle observation {observation_id} differs from its route input."
            )
        posterior_metrics = paper_utils.distribution_metrics(
            reference,
            posterior,
            seed=int(ml_seed) + 40_000 + observation_id,
            max_samples=max_samples,
        )
        likelihood_metrics = paper_utils.distribution_metrics(
            reference,
            likelihood,
            seed=int(ml_seed) + 50_000 + observation_id,
            max_samples=max_samples,
        )
        route_metrics = paper_utils.distribution_metrics(
            posterior,
            likelihood,
            seed=int(ml_seed) + 60_000 + observation_id,
            max_samples=max_samples,
        )
        cycle_valid = np.isfinite(log_q_phi) & np.isfinite(log_q_eta) & np.isfinite(log_prior)
        cycle = paper_utils._cycle_diagnostics(
            log_q_phi[cycle_valid] - log_prior[cycle_valid],
            log_q_eta[cycle_valid],
        )
        weight_diagnostics = paper_utils.weight_diagnostics(weights)
        finite_residual = valid & np.isfinite(residual)
        observation_rms = (
            float(np.sqrt(np.mean(np.square(residual[finite_residual]))))
            if finite_residual.any()
            else float("nan")
        )
        centered_rms = (
            float(
                np.sqrt(
                    np.mean(
                        np.square(
                            residual[finite_residual]
                            - np.mean(residual[finite_residual])
                        )
                    )
                )
            )
            if finite_residual.any()
            else float("nan")
        )
        rows.append(
            {
                "schema": getattr(paper_utils, "PAPER_RUNTIME_SCHEMA", JANA_EVALUATION_SCHEMA),
                "campaign_signature": signature,
                "method": "jana_paper",
                "factorization": "none",
                "budget": int(budget),
                "simulator_calls": int(budget) + 2 + 2 + 300,
                "training_rows": int(budget),
                "validation_rows": 300,
                "ml_seed": int(ml_seed),
                "observation": observation_id,
                "posterior_C2ST": posterior_metrics["C2ST"],
                "posterior_MMD": posterior_metrics["MMD"],
                "likelihood_posterior_C2ST": likelihood_metrics["C2ST"],
                "likelihood_posterior_MMD": likelihood_metrics["MMD"],
                "posterior_likelihood_route_C2ST": route_metrics["C2ST"],
                "posterior_likelihood_route_MMD": route_metrics["MMD"],
                "posterior_joint_C2ST": posterior_joint_metrics["C2ST"],
                "posterior_joint_MMD": posterior_joint_metrics["MMD"],
                "predictive_x_C2ST": predictive_x_metrics["C2ST"],
                "predictive_x_MMD": predictive_x_metrics["MMD"],
                "predictive_joint_C2ST": predictive_joint_metrics["C2ST"],
                "predictive_joint_MMD": predictive_joint_metrics["MMD"],
                "posterior_joint_ESS_fraction": 1.0,
                "likelihood_joint_ESS_fraction": 1.0,
                "posterior_ESS_fraction": 1.0,
                "posterior_max_weight": 1.0 / len(posterior),
                "likelihood_posterior_ESS_fraction": weight_diagnostics["ess_fraction"],
                "likelihood_posterior_max_weight": weight_diagnostics["max_weight"],
                "bayes_cycle_pearson": audit_cycle["pearson"],
                "bayes_cycle_slope": audit_cycle["slope"],
                "bayes_cycle_residual_rms": audit_cycle["rms"],
                "bayes_cycle_rows": int(audit_cycle_valid.sum()),
                "deployed_proposal_bayes_cycle_pearson": cycle["pearson"],
                "deployed_proposal_bayes_cycle_slope": cycle["slope"],
                "deployed_proposal_bayes_cycle_residual_rms": cycle["rms"],
                "deployed_proposal_bayes_cycle_rows": int(cycle_valid.sum()),
                "likelihood_log_Z_rms": 0.0,
                "likelihood_log_Z_mean": 0.0,
                "likelihood_log_Z_max_abs": 0.0,
                # The standardized error is evaluated on the same independent
                # audit rows for every method.  Candidate-based errors diagnose
                # the deployed proposal but are not paired across methods.
                "exact_likelihood_log_error": audit_rms,
                "exact_likelihood_centered_log_error": audit_centered_rms,
                "exact_likelihood_rows": audit_finite_rows,
                "deployed_proposal_exact_likelihood_log_error": observation_rms,
                "deployed_proposal_exact_likelihood_centered_log_error": centered_rms,
                "deployed_proposal_exact_likelihood_rows": int(finite_residual.sum()),
                "independent_audit_exact_likelihood_log_error": audit_rms,
                "independent_audit_exact_likelihood_centered_log_error": audit_centered_rms,
                "independent_audit_exact_likelihood_rows": audit_finite_rows,
                "paper_simulator_calls": int(budget) + 2 + 2 + 300,
                "simulation_accounting": f"{int(budget)} + 2 + 2 + 300",
                "proposal": "method_native_nominal_q_phi",
                "proposal_base_scale": 1.0,
                "proposal_prior_fraction": 0.0,
                "route_arrays": str(result_path),
                "likelihood_audit_arrays": str(audit_path),
                "likelihood_normalization_arrays": str(normalization_path),
                "audit_row_fingerprint": audit_row_fingerprint,
                "audit_bank_fingerprint": audit_bank_fingerprint,
                "likelihood_audit_fingerprint": likelihood_audit_fingerprint,
                "likelihood_audit_rows": len(audit_reference_theta),
                "bayes_cycle_theta_fingerprint": cycle_theta_fingerprint,
                "bayes_cycle_theta_grid_fingerprint": cycle_theta_fingerprint,
            }
        )
    return rows


def run_exact_jana_campaign(
    artifact_root: str | Path,
    campaign: Mapping[str, Any],
    load_if_available: bool = True,
    budgets_to_run: Sequence[int] | None = None,
    ml_seeds_to_run: Sequence[int] | None = None,
):
    """Modern-runtime launcher returning fully evaluated exact-JANA rows.

    TensorFlow is never imported in this process.  Training and density
    evaluation run through ``PAPER_SUMMARY_JANA_PYTHON``; only saved NumPy
    arrays return to the modern runtime for the common C2ST/MMD implementation.
    """

    import pandas as pd

    try:
        try:
            from . import config as paper_config
        except ImportError:
            import config as paper_config
    except ImportError as error:
        raise RuntimeError("Cannot import the local paper-summary config.py.") from error
    paper_config.validate_campaign_config(campaign)
    selected_budgets, selected_seeds = _campaign_subset(
        campaign, budgets_to_run, ml_seeds_to_run
    )
    artifact_root = Path(artifact_root).expanduser().resolve()
    signature = paper_config.campaign_signature(campaign)
    cached_path = artifact_root / "results" / "jana" / f"jana_paper_{signature}.csv"
    result_manifest_path = cached_path.with_suffix(".manifest.json")
    result_root = artifact_root / "results" / "jana_paper" / signature
    expected_observations = {int(value) for value in campaign["observations"]}

    def pair_paths(budget: int, seed: int) -> tuple[Path, Path]:
        root = result_root / f"budget_{int(budget)}" / f"seed_{int(seed)}"
        return root / "metrics.csv", root / "metrics.manifest.json"

    def load_pair_shard(budget: int, seed: int):
        csv_path, manifest_path = pair_paths(budget, seed)
        if not (csv_path.is_file() and manifest_path.is_file()):
            return None
        try:
            shard_manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if (
            shard_manifest.get("campaign_signature") != signature
            or int(shard_manifest.get("budget", -1)) != int(budget)
            or int(shard_manifest.get("ml_seed", -1)) != int(seed)
            or shard_manifest.get("csv_sha256") != _sha256_file(csv_path)
            or not _evaluation_output_manifest_valid(
                shard_manifest.get("evaluation_manifest", "")
            )
        ):
            return None
        checkpoint_path = Path(shard_manifest.get("checkpoint_manifest", ""))
        if not checkpoint_path.is_file():
            return None
        try:
            checkpoint = json.loads(checkpoint_path.read_text())
            evaluation_manifest = json.loads(
                Path(shard_manifest["evaluation_manifest"]).read_text()
            )
        except (OSError, json.JSONDecodeError):
            return None
        if not _manifest_checkpoint_valid(
            checkpoint_path.parent,
            checkpoint.get("training_contract_sha256", ""),
        ):
            return None
        checkpoint_artifact = _checkpoint_artifact_sha256(checkpoint)
        current_driver = _sha256_file(Path(__file__).resolve())
        current_requirements = _sha256_file(
            Path(__file__).resolve().parent / "requirements_jana.txt"
        )
        if (
            evaluation_manifest.get("checkpoint_artifact_sha256")
            != checkpoint_artifact
            or shard_manifest.get("checkpoint_artifact_sha256")
            != checkpoint_artifact
            or checkpoint.get("driver_source_sha256") != current_driver
            or checkpoint.get("requirements_sha256") != current_requirements
            or evaluation_manifest.get("driver_source_sha256") != current_driver
            or evaluation_manifest.get("requirements_sha256")
            != current_requirements
        ):
            return None
        frame = pd.read_csv(csv_path)
        required_columns = {
            "budget",
            "ml_seed",
            "observation",
            "method",
            "route_arrays",
            *campaign["diagnostics"],
        }
        if not required_columns.issubset(frame.columns):
            return None
        selected = frame[
            (frame["budget"] == int(budget))
            & (frame["ml_seed"] == int(seed))
            & (frame["method"] == "jana_paper")
        ]
        if (
            len(selected) != len(expected_observations)
            or selected["observation"].nunique() != len(expected_observations)
            or set(selected["observation"].astype(int)) != expected_observations
            or not np.isfinite(
                selected[list(campaign["diagnostics"])].to_numpy(dtype=np.float64)
            ).all()
        ):
            return None
        return selected.copy()

    cached_pairs = {
        (budget, seed): load_pair_shard(budget, seed)
        for budget in selected_budgets
        for seed in selected_seeds
    }

    missing_pairs = [
        (budget, seed)
        for budget in selected_budgets
        for seed in selected_seeds
        if not (load_if_available and cached_pairs[(budget, seed)] is not None)
    ]
    banks = None
    legacy_python = None
    evaluation_input = None
    if missing_pairs:
        banks = _load_campaign_banks(artifact_root)
        legacy_python = resolve_jana_python(artifact_root)
        evaluation_input = _prepare_evaluation_input(
            artifact_root,
            campaign,
            banks=banks,
            load_if_available=load_if_available,
        )
        launch_isolated_campaign(
            legacy_python,
            artifact_root=artifact_root,
            master_bank_path=banks["master"]["path"],
            shape_bank_path=banks["jana_shape"]["path"],
            pilot_bank_path=banks["jana_pilot"]["path"],
            validation_bank_path=banks["jana_validation"]["path"],
            budgets=sorted({budget for budget, _ in missing_pairs}),
            seeds=sorted({seed for _, seed in missing_pairs}),
            profile=campaign["profile"],
            load_if_available=load_if_available,
        )
    for budget, ml_seed in missing_pairs:
        run_directory = default_run_directory(
            artifact_root, budget=int(budget), seed=int(ml_seed)
        )
        output_directory = (
            result_root
            / f"budget_{int(budget)}"
            / f"seed_{int(ml_seed)}"
            / "standardized"
        )
        evaluation = _launch_isolated_evaluation(
            legacy_python,
            run_directory=run_directory,
            input_path=evaluation_input,
            output_directory=output_directory,
            posterior_samples=int(campaign["posterior_samples"]),
            proposal_candidates=int(campaign["proposal_candidates"]),
            seed=int(ml_seed) + 500_000,
            load_if_available=load_if_available,
        )
        pair_rows = _evaluate_saved_routes_modern(
                artifact_root=artifact_root,
                campaign=campaign,
                budget=int(budget),
                ml_seed=int(ml_seed),
                evaluation_manifest=evaluation,
        )
        pair_frame = pd.DataFrame(pair_rows)
        missing_diagnostics = [
            name for name in campaign["diagnostics"] if name not in pair_frame.columns
        ]
        if missing_diagnostics:
            raise RuntimeError(
                "Exact-JANA output is missing configured diagnostics: "
                f"{missing_diagnostics}"
            )
        if (
            len(pair_frame) != len(expected_observations)
            or pair_frame["observation"].nunique() != len(expected_observations)
            or not np.isfinite(
                pair_frame[list(campaign["diagnostics"])].to_numpy(
                    dtype=np.float64
                )
            ).all()
        ):
            raise RuntimeError(
                "Exact-JANA metric shard is incomplete or contains non-finite "
                "configured diagnostics."
            )
        pair_csv, pair_manifest = pair_paths(int(budget), int(ml_seed))
        _atomic_write_dataframe_csv(pair_csv, pair_frame)
        checkpoint_path = run_directory / "checkpoint_manifest.json"
        checkpoint = json.loads(checkpoint_path.read_text())
        _atomic_write_json(
            pair_manifest,
            {
                "schema": JANA_EVALUATION_SCHEMA,
                "campaign_signature": signature,
                "budget": int(budget),
                "ml_seed": int(ml_seed),
                "rows": len(pair_frame),
                "csv_path": str(pair_csv),
                "csv_sha256": _sha256_file(pair_csv),
                "evaluation_manifest": str(
                    output_directory / "evaluation_manifest.json"
                ),
                "checkpoint_manifest": str(checkpoint_path),
                "checkpoint_artifact_sha256": _checkpoint_artifact_sha256(
                    checkpoint
                ),
                "updated_utc": _utc_now(),
            },
        )
    import fcntl

    cached_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cached_path.parent / f"jana_paper_{signature}.lock"
    with lock_path.open("a+") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        all_pair_frames = []
        pair_manifest_paths = []
        for budget in campaign["budgets"]:
            for ml_seed in campaign["ml_seeds"]:
                shard = load_pair_shard(int(budget), int(ml_seed))
                if shard is not None:
                    all_pair_frames.append(shard)
                    pair_manifest_paths.append(
                        str(pair_paths(int(budget), int(ml_seed))[1])
                    )
        if not all_pair_frames:
            raise RuntimeError("No valid exact-JANA metric shards are available.")
        merged = pd.concat(all_pair_frames, ignore_index=True)
        merged = (
            merged.drop_duplicates(
                subset=["method", "budget", "ml_seed", "observation"],
                keep="last",
            )
            .sort_values(["budget", "ml_seed", "observation"])
            .reset_index(drop=True)
        )
        _atomic_write_dataframe_csv(cached_path, merged)
        jana_rows = merged[merged["method"] == "jana_paper"]
        evaluation_manifests = sorted(
            {
                str(Path(path).parent / "evaluation_manifest.json")
                for path in jana_rows["route_arrays"]
            }
        )
        checkpoint_manifests = sorted(
            {
                str(
                    default_run_directory(
                        artifact_root,
                        budget=int(row.budget),
                        seed=int(row.ml_seed),
                    )
                    / "checkpoint_manifest.json"
                )
                for row in jana_rows[["budget", "ml_seed"]]
                .drop_duplicates()
                .itertuples(index=False)
            }
        )
        for evaluation_manifest in evaluation_manifests:
            if not _evaluation_output_manifest_valid(evaluation_manifest):
                raise RuntimeError(
                    "Exact-JANA evaluation manifest failed validation: "
                    f"{evaluation_manifest}"
                )
        _atomic_write_json(
            result_manifest_path,
            {
                "schema": JANA_EVALUATION_SCHEMA,
                "campaign_signature": signature,
                "csv_path": str(cached_path),
                "csv_sha256": _sha256_file(cached_path),
                "rows": len(merged),
                "completed_pairs": sorted(
                    {
                        f"{int(row.budget)}:{int(row.ml_seed)}"
                        for row in jana_rows[["budget", "ml_seed"]]
                        .drop_duplicates()
                        .itertuples(index=False)
                    }
                ),
                "pair_manifests": sorted(pair_manifest_paths),
                "evaluation_manifests": evaluation_manifests,
                "checkpoint_manifests": checkpoint_manifests,
                "updated_utc": _utc_now(),
            },
        )
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
    return merged


def export_exact_jana_ratio_banks(
    artifact_root: str | Path,
    campaign: Mapping[str, Any],
    load_if_available: bool = True,
    budgets_to_run: Sequence[int] | None = None,
    ml_seeds_to_run: Sequence[int] | None = None,
    context_batch_size: int = 8192,
) -> dict[str, Any]:
    """Launch nominal exact-JANA S/P/L exports for modern ratio training."""

    try:
        try:
            from . import config as paper_config
        except ImportError:
            import config as paper_config
    except ImportError as error:
        raise RuntimeError("Cannot import the local paper-summary config.py.") from error
    paper_config.validate_campaign_config(campaign)
    budgets, seeds = _campaign_subset(
        campaign, budgets_to_run, ml_seeds_to_run
    )
    context_batch_size = int(context_batch_size)
    if context_batch_size < 1:
        raise ValueError("context_batch_size must be positive.")
    artifact_root = Path(artifact_root).expanduser().resolve()
    signature = paper_config.campaign_signature(campaign)
    banks = _load_campaign_banks(artifact_root)
    legacy_python = resolve_jana_python(artifact_root)
    launch_isolated_campaign(
        legacy_python,
        artifact_root=artifact_root,
        master_bank_path=banks["master"]["path"],
        shape_bank_path=banks["jana_shape"]["path"],
        pilot_bank_path=banks["jana_pilot"]["path"],
        validation_bank_path=banks["jana_validation"]["path"],
        budgets=budgets,
        seeds=seeds,
        profile=campaign["profile"],
        load_if_available=load_if_available,
    )
    root = artifact_root / "ratio_banks" / "jana_paper" / signature
    for budget in budgets:
        for ml_seed in seeds:
            output_directory = (
                root / f"budget_{int(budget)}" / f"seed_{int(ml_seed)}"
            )
            _launch_isolated_ratio_export(
                legacy_python,
                run_directory=default_run_directory(
                    artifact_root, budget=int(budget), seed=int(ml_seed)
                ),
                master_bank_path=Path(banks["master"]["path"]),
                validation_bank_path=Path(banks["jana_validation"]["path"]),
                budget=int(budget),
                seed=int(ml_seed),
                output_directory=output_directory,
                context_batch_size=context_batch_size,
                load_if_available=load_if_available,
            )
    aggregate_path = root / "manifest.json"
    import fcntl

    root.mkdir(parents=True, exist_ok=True)
    with (root / "manifest.lock").open("a+") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        keyed = {}
        current_driver = _sha256_file(Path(__file__).resolve())
        current_requirements = _sha256_file(
            Path(__file__).resolve().parent / "requirements_jana.txt"
        )
        for budget in campaign["budgets"]:
            for ml_seed in campaign["ml_seeds"]:
                run_manifest_path = (
                    root
                    / f"budget_{int(budget)}"
                    / f"seed_{int(ml_seed)}"
                    / "manifest.json"
                )
                if not run_manifest_path.is_file():
                    continue
                try:
                    item = json.loads(run_manifest_path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                arrays_path = Path(item.get("arrays_path", ""))
                checkpoint_path = Path(item.get("checkpoint_manifest_path", ""))
                if not (arrays_path.is_file() and checkpoint_path.is_file()):
                    continue
                try:
                    checkpoint = json.loads(checkpoint_path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                if (
                    item.get("status") != "complete"
                    or int(item.get("budget", -1)) != int(budget)
                    or int(item.get("seed", -1)) != int(ml_seed)
                    or item.get("arrays_sha256") != _sha256_file(arrays_path)
                    or item.get("driver_source_sha256") != current_driver
                    or item.get("requirements_sha256") != current_requirements
                    or item.get("checkpoint_artifact_sha256")
                    != _checkpoint_artifact_sha256(checkpoint)
                    or not _manifest_checkpoint_valid(
                        checkpoint_path.parent,
                        checkpoint.get("training_contract_sha256", ""),
                    )
                ):
                    continue
                keyed[(int(budget), int(ml_seed))] = {
                    "budget": int(budget),
                    "seed": int(ml_seed),
                    "manifest_path": str(run_manifest_path),
                    "arrays_path": item["arrays_path"],
                    "arrays_sha256": item["arrays_sha256"],
                    "content_fingerprint": item["content_fingerprint"],
                    "contract_sha256": item["contract_sha256"],
                    "context_batch_size": int(item["context_batch_size"]),
                    "checkpoint_artifact_sha256": item[
                        "checkpoint_artifact_sha256"
                    ],
                }
        aggregate = {
            "schema": JANA_RUNTIME_SCHEMA,
            "purpose": "nominal_exact_jana_ratio_training_banks",
            "campaign_signature": signature,
            "feature_coordinates": "raw_physical_theta_plus_raw_physical_x",
            "proposal": "nominal_exact_jana_no_broadening_no_prior_defense",
            "simulation_accounting": "N + 2_shape + 2_trainer + 300_validation",
            "requested_context_batch_size": context_batch_size,
            "runs": [keyed[key] for key in sorted(keyed)],
            "updated_utc": _utc_now(),
        }
        _atomic_write_json(aggregate_path, aggregate)
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
    return aggregate


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="train one fixed-bank JANA model")
    train.add_argument("--artifact-root", required=True)
    train.add_argument("--master-bank", required=True)
    train.add_argument("--shape-bank", required=True)
    train.add_argument("--pilot-bank", required=True)
    train.add_argument("--validation-bank", required=True)
    train.add_argument("--budget", required=True, type=int)
    train.add_argument("--seed", required=True, type=int)
    train.add_argument("--split-path")
    train.add_argument(
        "--validation-mode",
        choices=("external_paper", "inside_budget"),
        default="external_paper",
    )
    train.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    train.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    train.add_argument("--force", action="store_true")
    train.add_argument("--no-load-if-available", action="store_true")
    train.add_argument("--allow-runtime-drift", action="store_true")

    campaign = subparsers.add_parser("campaign", help="train a budget/seed grid")
    campaign.add_argument("--artifact-root", required=True)
    campaign.add_argument("--master-bank", required=True)
    campaign.add_argument("--shape-bank", required=True)
    campaign.add_argument("--pilot-bank", required=True)
    campaign.add_argument("--validation-bank", required=True)
    campaign.add_argument("--budgets", nargs="+", required=True, type=int)
    campaign.add_argument("--seeds", nargs="+", required=True, type=int)
    campaign.add_argument("--profile", choices=("PAPER", "SMOKE"), default="PAPER")
    campaign.add_argument("--epochs", type=int)
    campaign.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    campaign.add_argument("--force", action="store_true")
    campaign.add_argument("--no-load-if-available", action="store_true")
    campaign.add_argument("--allow-runtime-drift", action="store_true")

    export_ratio = subparsers.add_parser(
        "export-ratio-bank",
        help="export nominal exact-JANA S/P/L arrays for modern ratio training",
    )
    export_ratio.add_argument("--run-directory", required=True)
    export_ratio.add_argument("--master-bank", required=True)
    export_ratio.add_argument("--validation-bank", required=True)
    export_ratio.add_argument("--budget", required=True, type=int)
    export_ratio.add_argument("--seed", required=True, type=int)
    export_ratio.add_argument("--output-directory", required=True)
    export_ratio.add_argument("--context-batch-size", type=int, default=8192)
    export_ratio.add_argument("--no-load-if-available", action="store_true")
    export_ratio.add_argument("--allow-runtime-drift", action="store_true")

    evaluate = subparsers.add_parser(
        "evaluate", help="write route/C2ST arrays from an evaluation-input NPZ"
    )
    evaluate.add_argument("--run-directory", required=True)
    evaluate.add_argument("--input", required=True)
    evaluate.add_argument("--output-directory")
    evaluate.add_argument("--posterior-samples", type=int, default=10_000)
    evaluate.add_argument("--proposal-candidates", type=int, default=150_000)
    evaluate.add_argument("--likelihood-route-samples", type=int)
    evaluate.add_argument("--seed", type=int, default=31082026)
    evaluate.add_argument("--no-load-if-available", action="store_true")
    evaluate.add_argument("--allow-runtime-drift", action="store_true")
    return parser


def _main(argv: Sequence[str] | None = None) -> int:
    args = _cli_parser().parse_args(argv)
    if args.command == "train":
        manifest = train_exact_jana(
            args.master_bank,
            artifact_root=args.artifact_root,
            budget=args.budget,
            seed=args.seed,
            shape_bank_path=args.shape_bank,
            pilot_bank_path=args.pilot_bank,
            validation_bank_path=args.validation_bank,
            validation_mode=args.validation_mode,
            split_path=args.split_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            load_if_available=not args.no_load_if_available,
            force=args.force,
            strict_runtime=not args.allow_runtime_drift,
        )
        print(Path(manifest["run_directory"]) / "checkpoint_manifest.json")
        return 0
    if args.command == "campaign":
        summary = run_jana_campaign(
            artifact_root=args.artifact_root,
            master_bank_path=args.master_bank,
            shape_bank_path=args.shape_bank,
            pilot_bank_path=args.pilot_bank,
            validation_bank_path=args.validation_bank,
            budgets=args.budgets,
            seeds=args.seeds,
            profile=args.profile,
            epochs=args.epochs,
            batch_size=args.batch_size,
            load_if_available=not args.no_load_if_available,
            force=args.force,
            strict_runtime=not args.allow_runtime_drift,
        )
        path = Path(args.artifact_root).expanduser().resolve() / "jana_paper" / "campaign_manifest.json"
        if summary.get("status") != "complete":
            raise RuntimeError("JANA campaign did not complete.")
        print(path)
        return 0
    if args.command == "export-ratio-bank":
        manifest = export_nominal_ratio_class_bank(
            args.run_directory,
            master_bank_path=args.master_bank,
            validation_bank_path=args.validation_bank,
            budget=args.budget,
            seed=args.seed,
            output_directory=args.output_directory,
            context_batch_size=args.context_batch_size,
            load_if_available=not args.no_load_if_available,
            strict_runtime=not args.allow_runtime_drift,
        )
        print(Path(manifest["arrays_path"]).parent / "manifest.json")
        return 0

    input_path = Path(args.input).expanduser().resolve()
    with np.load(input_path, allow_pickle=False) as saved:
        required = {"observations", "reference_posterior_samples"}
        missing = sorted(required.difference(saved.files))
        if missing:
            raise KeyError(f"Evaluation input {input_path} is missing {missing}.")
        kwargs: dict[str, Any] = {
            "observations": saved["observations"],
            "reference_posterior_samples": saved["reference_posterior_samples"],
        }
        if "proposal_theta" in saved.files or "proposal_log_density_theta" in saved.files:
            if not {
                "proposal_theta",
                "proposal_log_density_theta",
            }.issubset(saved.files):
                raise KeyError(
                    "Evaluation input must supply both proposal_theta and "
                    "proposal_log_density_theta, or neither."
                )
            kwargs["proposal_theta"] = saved["proposal_theta"]
            kwargs["proposal_log_density_theta"] = saved[
                "proposal_log_density_theta"
            ]
        for key in (
            "observation_ids",
            "audit_reference_ids",
            "audit_reference_theta",
            "audit_reference_x",
            "audit_marginal_ids",
            "audit_marginal_theta",
            "audit_marginal_x",
            "normalization_theta_ids",
            "normalization_theta",
        ):
            if key in saved.files:
                kwargs[key] = saved[key]
        if "audit_bank_fingerprint" in saved.files:
            kwargs["audit_bank_fingerprint"] = str(saved["audit_bank_fingerprint"])
        if "normalization_x_per_theta" in saved.files:
            kwargs["normalization_x_per_theta"] = int(
                saved["normalization_x_per_theta"]
            )
    manifest = run_standardized_evaluation(
        args.run_directory,
        **kwargs,
        posterior_samples=args.posterior_samples,
        proposal_candidates=args.proposal_candidates,
        likelihood_route_samples=args.likelihood_route_samples,
        seed=args.seed,
        output_directory=args.output_directory,
        load_if_available=not args.no_load_if_available,
        strict_runtime=not args.allow_runtime_drift,
    )
    print(Path(manifest["output_directory"]) / "evaluation_manifest.json")
    return 0


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_EPOCHS",
    "JANA_EVALUATION_SCHEMA",
    "JANA_RUNTIME_SCHEMA",
    "LoadedJANA",
    "OBSERVATION_SCALE",
    "PAPER_BUDGETS",
    "SLCPBank",
    "TrainingSlice",
    "build_exact_jana",
    "configure_joint",
    "construct_likelihood_route_posterior",
    "default_run_directory",
    "ensure_jana_environment",
    "evaluate_nominal_log_likelihood",
    "evaluate_nominal_log_posterior",
    "exact_likelihood_comparison",
    "exact_slcp_log_likelihood",
    "export_exact_jana_ratio_banks",
    "export_nominal_ratio_class_bank",
    "launch_isolated_campaign",
    "load_exact_jana",
    "load_slcp_bank",
    "prepare_training_slice",
    "resolve_jana_python",
    "run_exact_jana_campaign",
    "run_jana_campaign",
    "run_standardized_evaluation",
    "sample_nominal_likelihood",
    "sample_nominal_posterior",
    "slcp_prior_log_density",
    "slcp_theta_to_latent",
    "train_exact_jana",
    "validate_legacy_runtime",
]


if __name__ == "__main__":
    raise SystemExit(_main())
