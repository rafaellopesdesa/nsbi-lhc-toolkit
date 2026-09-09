"""Epoch-resumable execution of the pinned exact-JANA training protocol.

The model, configurator, Adam defaults, full-run cosine schedule, batch size,
validation and backpropagation primitive come from the original driver and
pinned BayesFlow. Only orchestration, progress reporting and persistence are
changed. No TensorFlow imports occur until this module runs in legacy Python.

Resume restores model and optimizer variables, optimizer iterations and the
completed epoch. Python/NumPy and the TF Generator states are also saved.
Legacy stateful dropout/tf.data streams are NOT fully checkpointable here:
resuming is a continuation of training, not a bitwise replay of an uninterrupted
run. An interruption within an epoch replays that epoch from its beginning.
"""

from __future__ import annotations

import functools
import inspect
import json
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
import time

import numpy as np

import utils_jana as jana


RESUME_SCHEMA = "exact_jana_epoch_resume_v1"


def _generations(root):
    return sorted(path for path in (Path(root) / "resume").glob("epoch_*/state.json")
                  if re.fullmatch(r"epoch_[0-9]{6}", path.parent.name))


def _file_record(path, root):
    return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
            "sha256": jana._sha256_file(path)}


def _record_valid(record, root):
    try:
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            return False
        return path.is_file() and path.stat().st_size == record["bytes"] and jana._sha256_file(path) == record["sha256"]
    except (KeyError, OSError, TypeError, ValueError):
        return False


def find_resume(run_directory, contract_hash):
    """Use the newest fully published generation; ignore interrupted writes."""
    root = Path(run_directory)
    candidates = list(reversed(_generations(root)))
    for path in candidates:
        try:
            state = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if state.get("schema") != RESUME_SCHEMA or state.get("contract_sha256") != contract_hash:
            raise RuntimeError(f"Incompatible resume checkpoint at {path}; refusing to mix training settings.")
        records = state.get("checkpoint_files", []) + state.get("history_files", [])
        if state.get("checkpoint_files") and all(_record_valid(record, root) for record in records):
            return state
        print(f"[exact JANA resume] Incomplete/damaged generation {path.parent.name}; checking the previous epoch.", flush=True)
    if candidates:
        raise RuntimeError("No intact resumable checkpoint remains; refusing to silently restart an existing training.")
    return None


def _python_tuple(value):
    return tuple(_python_tuple(x) for x in value) if isinstance(value, list) else value


def _rng_state():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]]}


def _restore_rng(state):
    random.setstate(_python_tuple(state["python"]))
    saved = state["numpy"]
    np.random.set_state((saved[0], np.asarray(saved[1], dtype=np.uint32), saved[2], saved[3], saved[4]))


def save_epoch(run_directory, checkpoint, *, epoch, contract, train_frame, val_frame, started_utc, previous):
    """Publish a complete generation before retiring older checkpoint weights."""
    root = Path(run_directory)
    resume_root = root / "resume"
    resume_root.mkdir(parents=True, exist_ok=True)
    history_root = root / "epoch_history"
    history_root.mkdir(exist_ok=True)
    history_records = list(previous.get("history_files", [])) if previous else []
    for label, frame in (("train", train_frame), ("validation", val_frame)):
        path = history_root / f"epoch_{epoch:06d}_{label}.csv"
        jana._atomic_write_dataframe_csv(path, frame)
        history_records.append(_file_record(path, root))
    pending = Path(tempfile.mkdtemp(prefix="pending-", dir=resume_root))
    # write/read, not save/restore: no untracked save_counter dependency.
    checkpoint.write(str(pending / "training"))
    files = jana._checkpoint_files(pending / "training")
    if not any(path.suffix == ".index" for path in files) or not any(".data-" in path.name for path in files):
        raise RuntimeError("TensorFlow did not write a complete epoch checkpoint.")
    destination = resume_root / f"epoch_{epoch:06d}"
    # A damaged/unpublished generation may occupy this name. Preserve it.
    if destination.exists():
        destination.rename(resume_root / (destination.name + f".replaced-{time.time_ns()}"))
    pending.rename(destination)
    state = {
        "schema": RESUME_SCHEMA, "contract": contract,
        "contract_sha256": jana._mapping_sha256(contract),
        "completed_epoch": epoch, "started_utc": started_utc,
        "saved_utc": jana._utc_now(),
        "optimizer_iterations": int(checkpoint.optimizer.iterations.numpy()),
        "checkpoint_prefix": str((destination / "training").relative_to(root)),
        "checkpoint_files": [_file_record(destination / path.name, root) for path in files],
        "history_files": history_records, "rng": _rng_state(),
        "resume_semantics": "completed_epoch; model+Adam+iterations; not bitwise dropout/shuffle replay",
    }
    # State is the commit marker. A crash before this write leaves the last
    # generation fully intact, even on a partially synced Drive mount.
    jana._atomic_write_json(destination / "state.json", state)
    jana._atomic_write_json(resume_root / "latest.json", state)
    # Retain current + previous checkpoint weights; retain ALL epoch histories.
    complete = _generations(root)
    for old in complete[:-2]:
        directory = old.parent
        if directory.parent == resume_root and directory.name == f"epoch_{int(directory.name.removeprefix('epoch_')):06d}":
            shutil.rmtree(directory)
    return state


def _loss_row(loss, epoch, batch=None):
    values = loss if isinstance(loss, dict) else {"Loss": loss}
    row = {str(key): float(np.asarray(value)) for key, value in values.items()}
    if not all(np.isfinite(value) for value in row.values()):
        raise FloatingPointError(f"Non-finite JANA loss in epoch {epoch}, batch {batch}; last saved epoch remains reusable.")
    row["Epoch"] = int(epoch)
    if batch is not None:
        row["Batch"] = int(batch)
    return row


def train_resumable(trainer, simulations_dict, *, epochs, batch_size, validation_sims, context, **kwargs):
    import pandas as pd
    import tensorflow as tf
    from bayesflow.helper_classes import SimulationDataset
    from bayesflow.helper_functions import backprop_step, extract_current_lr

    if kwargs.get("early_stopping") or kwargs.get("optimizer") is not None:
        raise ValueError("The resumable exact-JANA runner uses the pinned Adam and fixed epoch count.")
    root = context["run_directory"]
    dataset = SimulationDataset(simulations_dict, batch_size)
    # Crucial: initialize the schedule for ALL epochs, not just those remaining.
    trainer._setup_optimizer(None, epochs, dataset.num_batches)
    optimizer = trainer.optimizer
    optimizer.build(trainer.amortizer.trainable_variables)
    epoch_variable = tf.Variable(0, dtype=tf.int64, trainable=False)
    generator = tf.random.Generator.from_seed(context["seed"])
    tf.random.set_global_generator(generator)
    checkpoint = tf.train.Checkpoint(amortizer=trainer.amortizer, optimizer=optimizer,
                                    completed_epoch=epoch_variable, rng=generator)
    contract = {
        "schema": RESUME_SCHEMA, "seed": context["seed"], "budget": context["budget"],
        "epochs": epochs, "batch_size": batch_size, "batches_per_epoch": dataset.num_batches,
        "driver_sha256": jana._sha256_file(Path(jana.__file__)),
        "training_helper_sha256": jana._sha256_file(Path(__file__)),
        "requirements_sha256": jana._sha256_file(Path(__file__).with_name("requirements_jana.txt")),
        "bayesflow_commit": jana.JANA_BAYESFLOW_COMMIT,
        "training_sha256": jana._array_sha256(*[simulations_dict[key] for key in sorted(simulations_dict)]),
        "validation_sha256": jana._array_sha256(*[validation_sims[key] for key in sorted(validation_sims)]),
    }
    state = find_resume(root, jana._mapping_sha256(contract))
    completed = 0
    started_utc = jana._utc_now()
    if state:
        checkpoint.read(str(root / state["checkpoint_prefix"])).assert_consumed()
        completed = int(epoch_variable.numpy())
        steps = int(optimizer.iterations.numpy())
        if completed != state["completed_epoch"] or steps != completed * dataset.num_batches or steps != state["optimizer_iterations"]:
            raise RuntimeError("Checkpoint epoch and Adam iteration counts disagree.")
        _restore_rng(state["rng"])
        started_utc = state["started_utc"]
        print(f"[exact JANA resume] Restored epoch {completed}/{epochs}, Adam step {steps}; continuing at epoch {completed + 1}.", flush=True)
    else:
        print(f"[exact JANA train] Starting fresh: {epochs} epochs, {dataset.num_batches} batches/epoch, batch size {batch_size}.", flush=True)
    if context.get("require_gpu"):
        gpu_variables = sum("GPU:" in variable.device for variable in trainer.amortizer.trainable_variables)
        if not gpu_variables:
            raise RuntimeError("JANA model variables are not on the GPU; refusing CPU training.")
        print(f"[exact JANA GPU] {gpu_variables}/{len(trainer.amortizer.trainable_variables)} trainable variables on GPU.", flush=True)
    update = tf.function(backprop_step, reduce_retracing=True)
    validation = trainer.configurator(validation_sims)
    for epoch in range(completed + 1, epochs + 1):
        epoch_start = last_report = time.monotonic()
        rows = []
        print(f"[exact JANA train] Epoch {epoch}/{epochs} started (budget={context['budget']}, seed={context['seed']}).", flush=True)
        for batch, forward_dict in enumerate(dataset, 1):
            loss = trainer._train_step(batch_size, update, trainer.configurator(forward_dict))
            row = _loss_row(loss, epoch, batch)
            rows.append(row)
            now = time.monotonic()
            if batch == 1 or batch == dataset.num_batches or now - last_report >= 60:
                losses = ", ".join(f"{key}={value:.5g}" for key, value in row.items() if key not in ("Epoch", "Batch"))
                print(f"[exact JANA train] Epoch {epoch}/{epochs}, batch {batch}/{dataset.num_batches}, {now - epoch_start:.0f}s; {losses}", flush=True)
                last_report = now
        val_row = _loss_row(trainer.amortizer.compute_loss(validation), epoch)
        epoch_variable.assign(epoch)
        state = save_epoch(root, checkpoint, epoch=epoch, contract=contract,
                           train_frame=pd.DataFrame(rows), val_frame=pd.DataFrame([val_row]),
                           started_utc=started_utc, previous=state)
        elapsed = time.monotonic() - epoch_start
        lr = float(np.asarray(extract_current_lr(optimizer)))
        print(f"[exact JANA train] Epoch {epoch}/{epochs} saved; {elapsed:.1f}s, Adam step={int(optimizer.iterations.numpy())}, lr={lr:.3g}, validation={val_row}.", flush=True)
        print(f"[exact JANA checkpoint] {root / state['checkpoint_prefix']}", flush=True)
        jana._atomic_write_json(root / "training_progress.json", {
            "budget": context["budget"], "seed": context["seed"], "completed_epoch": epoch,
            "total_epochs": epochs, "epoch_seconds": elapsed, "updated_utc": jana._utc_now(),
            "checkpoint": state["checkpoint_prefix"], "validation": val_row,
        })
    # Keep the existing final checkpoint/history format for downstream 03/04.
    history = {}
    for label, suffix in (("train_losses", "_train.csv"), ("val_losses", "_validation.csv")):
        paths = [root / entry["path"] for entry in state["history_files"] if entry["path"].endswith(suffix)]
        history[label] = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    return history


def install_training_hook(*, gpu_info=None):
    """Wrap only this subprocess's train entry; retain the original contract."""
    from bayesflow.trainers import Trainer

    original_train = jana.train_exact_jana
    signature = inspect.signature(original_train)

    @functools.wraps(original_train)
    def training(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        options = bound.arguments
        root = options["run_directory"] or jana.default_run_directory(
            options["artifact_root"], budget=options["budget"], seed=options["seed"])
        root = Path(root).expanduser().resolve()
        if (options["force"] or not options["load_if_available"]) and (root / "resume").exists():
            # Explicit fresh training: preserve old partial attempts under a
            # non-active name, never mix their epochs into the new attempt.
            stamp = str(time.time_ns())
            for name in ("resume", "epoch_history"):
                path = root / name
                if path.exists():
                    path.rename(root / f"{name}.previous-{stamp}")
        context = {"run_directory": root, "seed": int(options["seed"]),
                   "budget": int(options["budget"]), "require_gpu": gpu_info is not None}
        old_offline = Trainer.train_offline
        executed = False
        try:
            def offline(trainer, simulations_dict, **settings):
                nonlocal executed
                executed = True
                return train_resumable(trainer, simulations_dict, context=context, **settings)
            Trainer.train_offline = offline
            result = original_train(*args, **kwargs)
        finally:
            Trainer.train_offline = old_offline
        # Sidecar execution provenance does not alter the original driver hash.
        if executed:
            jana._atomic_write_json(root / "training_execution.json", {
                "helper_sha256": jana._sha256_file(Path(__file__)), "gpu": gpu_info,
                "resume_schema": RESUME_SCHEMA, "completed_utc": jana._utc_now(),
            })
        return result

    jana.train_exact_jana = training


def main(argv=None):
    from utils_jana_gpu import check_tensorflow_gpu

    gpu_info = check_tensorflow_gpu()
    install_training_hook(gpu_info=gpu_info)
    return jana._main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
