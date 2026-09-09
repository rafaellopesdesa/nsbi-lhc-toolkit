"""Simulator-only NRE for Exercise 12; no dependency on a trained flow.

All new artifacts live below the caller's Exercise-12 artifact directory.
Completed banks and ensemble members are immutable, content-checked caches.
No function trains, replaces, or deletes the shared Exercise-5 PRESEL model.
Only NumPy is imported eagerly; PyTorch and ONNX are loaded when needed.
"""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np


FEATURES = ("x1", "x2", "x3", "x4", "x5")
SIMULATOR_VERSION = "exercise12-selected-simulator-v1"
TRAINING_VERSION = "exercise12-bce-swish-nadam-v2"
BANK_VERSION = "exercise12-immutable-bank-v1"
DEFAULT_TRAINING_CONFIG = {
    "ensemble_size": 4,
    "hidden_layers": 4,
    "hidden_features": 1024,
    "activation": "swish",
    "epochs": 140,
    "batch_size": 4096,
    "learning_rate": 1.0e-4,
    "scheduler_step": 20,
    "scheduler_gamma": 0.1,
    "minimum_learning_rate": 1.0e-10,
    "patience": None,
    "checkpoint_selection": "last",
    "device": "auto",
    "prediction_batch_size": 8192,
}


class CacheError(RuntimeError):
    """An existing artifact is incompatible, incomplete, or corrupted."""


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fingerprint(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_hash(array):
    """Hash numerical contents in bounded memory (also works on memmaps)."""
    array = np.asarray(array)
    digest = hashlib.sha256(_json({"shape": array.shape, "dtype": str(array.dtype)}).encode())
    for start in range(0, len(array), 100_000):
        digest.update(np.ascontiguousarray(array[start:start + 100_000]).tobytes())
    return digest.hexdigest()


def _positive_integer(value, name):
    try:
        valid = not isinstance(value, (bool, np.bool_)) and int(value) == value and int(value) > 0
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    return int(value)


def _finite_matrix(value, columns, name):
    value = np.asarray(value)
    if value.ndim != 2 or value.shape[1] != columns or not len(value):
        raise ValueError(f"{name} must have shape (N, {columns}) with N > 0.")
    for start in range(0, len(value), 100_000):
        if not np.isfinite(value[start:start + 100_000]).all():
            raise ValueError(f"{name} contains NaN or infinity.")
    return value


@contextmanager
def _new_artifact(path):
    """Publish a complete directory atomically; never overwrite old results.

    A killed runtime may leave a lock/staging directory.  A lock is not expired
    automatically: confirm no worker is using it before removing it manually.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + ".lock")
    try:
        descriptor = os.open(str(lock), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise CacheError(
            f"Artifact is locked: {lock}. If the previous Colab session ended, "
            "confirm that no worker is running before removing this lock. "
            "Completed artifacts and previous trainings must be preserved."
        ) from error
    temporary = None
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump({"pid": os.getpid(), "artifact": str(path)}, handle)
        if path.exists():
            raise CacheError(f"Artifact appeared while claiming the lock: {path}; rerun to load it.")
        temporary = Path(tempfile.mkdtemp(prefix=path.name + ".pending-", dir=path.parent))
        yield temporary
        os.rename(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            shutil.rmtree(temporary)
        lock.unlink(missing_ok=True)


def _write_manifest(directory, contract, files, **extra):
    payload = {
        "status": "complete", "contract": contract,
        "files": {name: _file_hash(directory / name) for name in files},
        **extra,
    }
    with (directory / "manifest.json").open("w") as handle:
        handle.write(_json(payload))
    return payload


def _verified_manifest(directory, contract):
    directory = Path(directory)
    try:
        with (directory / "manifest.json").open() as handle:
            payload = json.load(handle)
        if payload.get("status") != "complete" or payload.get("contract") != contract:
            raise ValueError("manifest contract/status does not match")
        files = payload.get("files", {})
        if not files:
            raise ValueError("manifest has no artifact hashes")
        for name, expected in files.items():
            if Path(name).name != name or _file_hash(directory / name) != expected:
                raise ValueError(f"hash mismatch for {name}")
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise CacheError(
            f"Refusing to overwrite incomplete/corrupt cache {directory}: {error}. "
            "Inspect or move only this Exercise-12 artifact aside before rerunning; "
            "no shared training has been changed."
        ) from error
    return payload


class Exercise5Selection:
    """The frozen ONNX PRESEL odds threshold and selected physical yields."""

    def __init__(self, session, scaler, ratio_cut, yields, fingerprint, sources):
        self.session = session
        self.scaler = scaler
        self.ratio_cut = float(ratio_cut)
        self.yields = np.asarray(yields, dtype=np.float64)
        self.fingerprint = str(fingerprint)
        self.sources = sources

    def __call__(self, features):
        import pandas as pd

        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != len(FEATURES):
            raise ValueError("PRESEL expects an (N, 5) feature matrix.")
        if not len(features):
            return np.zeros(0, dtype=bool)
        if not np.isfinite(features).all():
            raise ValueError("PRESEL input contains NaN or infinity.")
        result = np.empty(len(features), dtype=bool)
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        for start in range(0, len(features), 100_000):
            frame = pd.DataFrame(features[start:start + 100_000], columns=FEATURES)
            scaled = np.asarray(self.scaler.transform(frame), dtype=np.float32)
            scores = np.asarray(self.session.run([output_name], {input_name: scaled})[0]).reshape(-1)
            if len(scores) != len(frame) or not np.isfinite(scores).all():
                raise ValueError("Frozen PRESEL produced invalid scores.")
            # Match the workshop utils.predict_with_model wrapper, which
            # converts the upstream ONNX classifier score into density odds.
            scores = np.clip(scores.astype(np.float64), 0.0, 1.0 - 1e-9)
            ratios = scores / (1.0 - scores)
            result[start:start + len(frame)] = ratios >= self.ratio_cut
        return result


def load_exercise5_selection(work_dir, *, ratio_cut=None, yields=None):
    """Reuse PRESEL and infer its cut/yields from the saved simulator-bank metadata.

    Only three scalar NPZ entries are read, never the (potentially huge) ``q``
    array. All available metadata must agree. Explicit ``ratio_cut`` and
    ``yields=[lam_sig, lam_bkg]`` (or a signal/background mapping) override
    automatic inference, but must be supplied together when no banks exist.
    """
    work_dir = Path(work_dir)
    model = work_dir / "models_PRESEL" / "model0.onnx"
    scaler_path = work_dir / "models_PRESEL" / "model_scaler0.bin"
    missing = [str(path) for path in (model, scaler_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Exercise 12 reuses the frozen Exercise-5 PRESEL model. Missing: "
            + ", ".join(missing)
            + ". Point WORK_DIR to the same Drive directory used by Exercise 5. "
            "This notebook will not retrain or replace PRESEL."
        )
    bank_paths = sorted((work_dir / "simulator_toy_banks_hybrid").glob("*selected_q*.npz"))
    metadata = []
    for path in bank_paths:
        try:
            with np.load(path, allow_pickle=False) as payload:
                metadata.append(np.array([
                    float(payload["presel_ratio_cut"]),
                    float(payload["lam_sig"]), float(payload["lam_bkg"]),
                ]))
        except (OSError, ValueError, KeyError) as error:
            raise CacheError(f"Cannot read required Exercise-5 selection metadata in {path}: {error}") from error
    if metadata:
        baseline = metadata[0]
        if not np.isfinite(metadata).all() or not all(
            np.allclose(item, baseline, rtol=1e-12, atol=1e-15) for item in metadata
        ):
            raise CacheError(
                "Exercise-5 simulator banks contain inconsistent cut/yield metadata. "
                "Use the matching Exercise-5 artifact directory; do not guess a selection."
            )
        if ratio_cut is None:
            ratio_cut = float(baseline[0])
        if yields is None:
            yields = baseline[1:]
    if ratio_cut is None or yields is None:
        raise FileNotFoundError(
            f"No complete cut/yield metadata in {work_dir / 'simulator_toy_banks_hybrid'}. "
            "Run the Exercise-5 simulator-bank cell, or explicitly pass its saved "
            "PRESEL_RATIO_CUT and yields=[lam_sig, lam_bkg]."
        )
    if isinstance(yields, dict):
        yields = [yields["signal"], yields["background"]]
    yields = np.asarray(yields, dtype=float)
    if not np.isfinite(ratio_cut) or float(ratio_cut) < 0:
        raise ValueError("Exercise-5 PRESEL ratio cut must be finite and non-negative.")
    if yields.shape != (2,) or not np.isfinite(yields).all() or np.any(yields <= 0):
        raise ValueError("Selected yields must be finite positive [lam_sig, lam_bkg].")
    try:
        import joblib
        import onnxruntime as ort
    except ImportError as error:
        raise ImportError("Install joblib, scikit-learn, pandas and onnxruntime in the setup cell.") from error
    scaler = joblib.load(scaler_path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
    contract = {
        "version": "exercise5-presel-odds-v1", "features": list(FEATURES),
        "model_sha256": _file_hash(model), "scaler_sha256": _file_hash(scaler_path),
        "ratio_cut": float(ratio_cut), "yields": yields.tolist(),
    }
    return Exercise5Selection(session, scaler, ratio_cut, yields, _fingerprint(contract), {
        "model": str(model), "scaler": str(scaler_path),
        "metadata_banks": [str(path) for path in bank_paths], "contract": contract,
    })


def _simulator_contract():
    from utils_distributions import background_components, signal_components, smearing_parameters

    def pack(components):
        return [[float(frac), np.asarray(mean).tolist(), np.asarray(cov).tolist()]
                for frac, mean, cov in components]

    return {
        "version": SIMULATOR_VERSION, "features": list(FEATURES),
        "signal": pack(signal_components()), "background": pack(background_components()),
        "smearing": [np.asarray(item).tolist() for item in smearing_parameters()],
        "reference": "iid half selected-signal plus half selected-background",
    }


def _rng(seed, role, component):
    try:
        valid_seed = not isinstance(seed, (bool, np.bool_)) and int(seed) == seed and int(seed) >= 0
    except (TypeError, ValueError, OverflowError):
        valid_seed = False
    if not valid_seed:
        raise ValueError("seed must be a non-negative integer.")
    if not isinstance(role, str) or not role.strip():
        raise ValueError("A non-empty semantic bank role is required for independent RNG streams.")
    entropy = np.frombuffer(hashlib.sha256(f"{role}:{component}".encode()).digest(), dtype="<u4")
    return np.random.default_rng(np.random.SeedSequence([int(seed), *map(int, entropy)]))


def simulate_reconstructed(component, n, rng):
    """Same Gaussian-mixture + detector-smearing simulator as Exercise 5."""
    from utils_distributions import background_components, signal_components, smearing_parameters

    n = _positive_integer(n, "n")
    if component not in ("signal", "background"):
        raise ValueError("simulate_reconstructed component must be signal or background.")
    components = signal_components() if component == "signal" else background_components()
    fractions = np.asarray([item[0] for item in components], dtype=float)
    labels = rng.choice(len(components), size=n, p=fractions / fractions.sum())
    latent = np.empty((n, len(FEATURES)), dtype=np.float64)
    for index, (_, mean, covariance) in enumerate(components):
        mask = labels == index
        if mask.any():
            latent[mask] = rng.multivariate_normal(mean, covariance, int(mask.sum()))
    scale, resolution = smearing_parameters()
    return (latent * np.asarray(scale) + rng.normal(
        scale=np.asarray(resolution), size=latent.shape
    )).astype(np.float32)


def _selected_stream(selection, component, rng, batch_size):
    zero_batches = 0
    while True:
        features = simulate_reconstructed(component, batch_size, rng)
        keep = np.asarray(selection(features))
        if keep.dtype != np.bool_ or keep.shape != (len(features),):
            raise ValueError("selection must return one boolean per simulated event.")
        if keep.any():
            zero_batches = 0
            yield features[keep]
        else:
            zero_batches += 1
            if zero_batches >= 128:
                raise RuntimeError("No event passed PRESEL in 128 batches; check the cut and model.")


def selected_feature_chunks(selection, role, component, n, seed, batch_size=100_000):
    """Yield exactly ``n`` selected events, in bounded-memory chunks.

    REF mixes already-selected component laws with independent Bernoulli(1/2)
    assignments. Mixing inclusive samples before selection would be wrong.
    Roles/components domain-separate RNG streams even when integer seeds agree.
    """
    n = _positive_integer(n, "n")
    batch_size = _positive_integer(batch_size, "batch_size")
    if component not in ("signal", "background", "reference"):
        raise ValueError("component must be signal, background, or reference.")
    if component != "reference":
        stream = _selected_stream(selection, component, _rng(seed, role, component), batch_size)
        remaining = n
        while remaining:
            chunk = next(stream)[:remaining]
            remaining -= len(chunk)
            yield chunk
        return
    # Keep unused selected events between reference chunks. Both the labels
    # and component substreams remain independent and do not depend on n.
    label_rng = _rng(seed, role, "reference-labels")
    streams = {
        comp: _selected_stream(selection, comp, _rng(seed, role, "reference-" + comp), batch_size)
        for comp in ("signal", "background")
    }
    buffers = {comp: np.empty((0, len(FEATURES)), dtype=np.float32) for comp in streams}

    def take(comp, count):
        result = np.empty((count, len(FEATURES)), dtype=np.float32)
        cursor = 0
        while cursor < count:
            if not len(buffers[comp]):
                buffers[comp] = next(streams[comp])
            amount = min(count - cursor, len(buffers[comp]))
            result[cursor:cursor + amount] = buffers[comp][:amount]
            buffers[comp] = buffers[comp][amount:]
            cursor += amount
        return result

    remaining = n
    while remaining:
        labels = label_rng.random(batch_size) < 0.5
        chunk = np.empty((batch_size, len(FEATURES)), dtype=np.float32)
        chunk[labels] = take("signal", int(labels.sum()))
        chunk[~labels] = take("background", int((~labels).sum()))
        chunk = chunk[:remaining]
        remaining -= len(chunk)
        yield chunk


def _cached_bank(root, selection, predictor, role, component, n, seed, batch_size):
    n = _positive_integer(n, "n")
    batch_size = _positive_integer(batch_size, "batch_size")
    if component not in ("signal", "background", "reference"):
        raise ValueError("component must be signal, background, or reference.")
    _rng(seed, role, component)  # Validate even when the numerical bank already exists.
    kind = "features" if predictor is None else "ratios"
    columns = len(FEATURES) if predictor is None else 2
    dtype = np.float32 if predictor is None else np.float64
    contract = {
        "version": BANK_VERSION, "kind": kind, "selection": str(selection.fingerprint),
        "simulator": _simulator_contract(), "role": str(role), "component": component,
        "n": n, "seed": int(seed), "generation_batch_size": batch_size,
    }
    if predictor is not None:
        contract["predictor"] = str(predictor.fingerprint)
    directory = Path(root) / "banks" / kind / _fingerprint(contract)
    if directory.exists():
        _verified_manifest(directory, contract)
        result = np.load(directory / "values.npy", mmap_mode="r", allow_pickle=False)
        if result.shape != (n, columns) or result.dtype != dtype:
            raise CacheError(f"Wrong cached array shape/dtype in {directory}.")
        print(f"Reused {role}/{component}: {n:,} {kind}.")
        return result
    with _new_artifact(directory) as staging:
        result = np.lib.format.open_memmap(staging / "values.npy", mode="w+", dtype=dtype, shape=(n, columns))
        cursor = 0
        reported = 0
        print(f"Generating {role}/{component}: {n:,} {kind}.", flush=True)
        for features in selected_feature_chunks(selection, role, component, n, seed, batch_size):
            values = features if predictor is None else np.asarray(predictor(features), dtype=np.float64)
            if values.shape != (len(features), columns) or not np.isfinite(values).all():
                raise ValueError(f"Invalid {kind} during {role}/{component} generation.")
            if predictor is not None and np.any(values <= 0):
                raise ValueError("NRE ratios must be strictly positive; refusing to cache invalid values.")
            result[cursor:cursor + len(values)] = values
            cursor += len(values)
            if cursor == n or cursor - reported >= 1_000_000:
                print(f"  {role}/{component}: {cursor:,}/{n:,}", flush=True)
                reported = cursor
        result.flush()
        del result
        _write_manifest(staging, contract, ["values.npy"])
    print(f"Saved {role}/{component}: {n:,} {kind}.")
    return np.load(directory / "values.npy", mmap_mode="r", allow_pickle=False)


def cached_features(root, selection, role, component, n, seed, batch_size=100_000):
    """Generate/reuse a selected training or validation feature bank (read-only mmap)."""
    return _cached_bank(root, selection, None, role, component, n, seed, batch_size)


def cached_ratios(root, selection, predictor, role, component, n, seed, batch_size=100_000):
    """Generate selected events and store only their Nx2 NRE ratios, not features."""
    return _cached_bank(root, selection, predictor, role, component, n, seed, batch_size)


def _training_config(config):
    config = {} if config is None else dict(config)
    unknown = set(config) - set(DEFAULT_TRAINING_CONFIG)
    if unknown:
        raise ValueError(f"Unknown NRE training settings: {sorted(unknown)}")
    cfg = {**DEFAULT_TRAINING_CONFIG, **config}
    for key in ("ensemble_size", "hidden_layers", "hidden_features", "epochs", "batch_size",
                "scheduler_step", "prediction_batch_size"):
        cfg[key] = _positive_integer(cfg[key], key)
    if cfg["patience"] is not None:
        cfg["patience"] = _positive_integer(cfg["patience"], "patience")
    if cfg["checkpoint_selection"] not in ("last", "best_validation"):
        raise ValueError("checkpoint_selection must be last or best_validation.")
    for key in ("learning_rate", "scheduler_gamma", "minimum_learning_rate"):
        cfg[key] = float(cfg[key])
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be finite and positive.")
    if cfg["scheduler_gamma"] > 1:
        raise ValueError("scheduler_gamma must not exceed 1.")
    if cfg["minimum_learning_rate"] > cfg["learning_rate"]:
        raise ValueError("minimum_learning_rate must not exceed learning_rate.")
    if cfg["activation"] not in ("swish", "relu", "tanh"):
        raise ValueError("activation must be swish, relu or tanh.")
    if cfg["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda.")
    return cfg


def _epoch_learning_rate(config, epoch):
    """Actual rate for a zero-based epoch; the final decade has a hard floor."""
    return max(config["minimum_learning_rate"],
               config["learning_rate"] * config["scheduler_gamma"] ** (epoch // config["scheduler_step"]))


def _torch():
    try:
        import torch
    except ImportError as error:
        raise ImportError("Install PyTorch in the notebook setup cell to train/evaluate NRE.") from error
    return torch


def _network(config):
    torch = _torch()
    activation = {"swish": torch.nn.SiLU, "relu": torch.nn.ReLU, "tanh": torch.nn.Tanh}[config["activation"]]
    layers = []
    width = len(FEATURES)
    for _ in range(config["hidden_layers"]):
        layers.extend([torch.nn.Linear(width, config["hidden_features"]), activation()])
        width = config["hidden_features"]
    layers.append(torch.nn.Linear(width, 1))
    return torch.nn.Sequential(*layers)


def _device(config):
    torch = _torch()
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see a GPU; choose a Colab GPU runtime.")
    return torch.device("cuda" if config["device"] == "auto" and torch.cuda.is_available() else
                        "cpu" if config["device"] == "auto" else config["device"])


def _train_member(positive, reference, val_positive, val_reference, config, seed, device, scaler):
    torch = _torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = _network(config).to(device)
    optimizer = torch.optim.NAdam(model.parameters(), lr=config["learning_rate"], weight_decay=0.0)
    # Assign the rate explicitly at the start of each epoch: history and the
    # printed rate then describe actual updates, not the next scheduler step.
    rng = np.random.default_rng(seed)
    offset, scale = scaler
    # Equal class counts make sigmoid(logit)/(1-sigmoid(logit)) = exp(logit)
    # a direct p_component / p_reference estimate, with no prior correction.
    n_train, n_val = len(positive), len(val_positive)
    batch_size = config["batch_size"]

    def batch(arr_pos, arr_ref, indices, n_each):
        is_pos = indices < n_each
        features = np.empty((len(indices), len(FEATURES)), dtype=np.float32)
        features[is_pos] = arr_pos[indices[is_pos]]
        features[~is_pos] = arr_ref[indices[~is_pos] - n_each]
        features = np.asarray((features - offset) * scale - 1.5, dtype=np.float32)
        if not np.isfinite(features).all():
            raise FloatingPointError("Feature scaling produced non-finite training inputs.")
        return (torch.from_numpy(features).to(device),
                torch.from_numpy(is_pos.astype(np.float32)[:, None]).to(device))

    history = {"seed": seed, "train_loss": [], "validation_loss": [], "learning_rate": []}
    best_loss = float("inf")
    best_state = None
    stale = 0
    for epoch in range(config["epochs"]):
        epoch_started = time.monotonic()
        learning_rate = _epoch_learning_rate(config, epoch)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        print(f"  Starting epoch {epoch + 1:03d}/{config['epochs']}, lr={learning_rate:.3e}.", flush=True)
        model.train()
        permutation = rng.permutation(2 * n_train)
        train_sum = 0.0
        for start in range(0, 2 * n_train, batch_size):
            features, labels = batch(positive, reference, permutation[start:start + batch_size], n_train)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(features), labels)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f"Non-finite NRE train loss at epoch {epoch + 1}; no cache was replaced.")
            loss.backward()
            if not bool(torch.stack([torch.isfinite(param.grad).all()
                                     for param in model.parameters() if param.grad is not None]).all().item()):
                raise FloatingPointError("Non-finite NRE gradient; stopped without publishing this member.")
            optimizer.step()
            train_sum += float(loss.detach().item()) * len(features)
        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for start in range(0, 2 * n_val, batch_size):
                indices = np.arange(start, min(start + batch_size, 2 * n_val))
                features, labels = batch(val_positive, val_reference, indices, n_val)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(features), labels)
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError("Non-finite NRE validation loss; no checkpoint was published.")
                val_sum += float(loss.item()) * len(features)
        train_loss, val_loss = train_sum / (2 * n_train), val_sum / (2 * n_val)
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(val_loss)
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))
        print(f"  epoch {epoch + 1:03d}/{config['epochs']}: lr={learning_rate:.3e}, "
              f"train BCE={train_loss:.9f}, validation BCE={val_loss:.9f}, "
              f"elapsed={time.monotonic() - epoch_started:.1f}s", flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            history["best_epoch"] = epoch + 1
            stale = 0
        else:
            stale += 1
        if config["patience"] is not None and stale >= config["patience"]:
            break
    history["best_validation_loss"] = best_loss
    history["checkpoint_selection"] = config["checkpoint_selection"]
    history["selected_epoch"] = (history["best_epoch"] if config["checkpoint_selection"] == "best_validation"
                                 else len(history["train_loss"]))
    history["selected_learning_rate"] = history["learning_rate"][history["selected_epoch"] - 1]
    if config["checkpoint_selection"] == "best_validation":
        model.load_state_dict(best_state)
    print(f"  Selected {config['checkpoint_selection']} weights: epoch {history['selected_epoch']}, "
          f"lr={history['selected_learning_rate']:.3e}; "
          f"best validation epoch={history['best_epoch']}, BCE={best_loss:.9f}.", flush=True)
    model.eval()
    return model, history


class NREPredictor:
    """Two ensembles; average ratios via logsumexp, never average scores.

    ``histories`` is a dict mapping signal/background to a list (one per
    member) of dicts with train_loss, validation_loss, best_epoch and seed.
    Inputs have shape (N,5); columns of the float64 output are (r_S,r_B).
    """

    def __init__(self, models, scaler, config, fingerprint, histories, device):
        self.models = models
        self.offset, self.scale = scaler
        self.config = config
        self.fingerprint = fingerprint
        self.histories = histories
        self.device = device

    def log_ratios(self, features):
        torch = _torch()
        features = np.asarray(features)
        if features.shape == (0, len(FEATURES)):
            return np.empty((0, 2), dtype=np.float64)
        features = _finite_matrix(features, len(FEATURES), "features")
        result = np.empty((len(features), 2), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(features), self.config["prediction_batch_size"]):
                stop = min(start + self.config["prediction_batch_size"], len(features))
                scaled = np.asarray((features[start:stop] - self.offset) * self.scale - 1.5, dtype=np.float32)
                inputs = torch.from_numpy(scaled).to(self.device)
                for column, component in enumerate(("signal", "background")):
                    logits = np.stack([
                        model(inputs).detach().cpu().numpy().reshape(-1).astype(np.float64)
                        for model in self.models[component]
                    ])
                    if not np.isfinite(logits).all():
                        raise FloatingPointError("Non-finite trained NRE logits; inspect this ensemble before proceeding.")
                    maximum = logits.max(axis=0)
                    result[start:stop, column] = maximum + np.log(
                        np.exp(logits - maximum).mean(axis=0)
                    )
        return result

    def __call__(self, features):
        logarithms = self.log_ratios(features)
        with np.errstate(over="ignore", under="ignore"):
            ratios = np.exp(logarithms)
        if not np.isfinite(ratios).all() or np.any(ratios <= 0):
            raise FloatingPointError(
                "NRE ratios exceed float64 range. No arbitrary clipping is applied; inspect the model/inputs."
            )
        return ratios


def train_nre(root, train_signal, train_background, train_ref,
              validation_signal, validation_background, validation_ref, config=None, seed=120005):
    """Train/reuse independent S/REF and B/REF BCE-only classifier ensembles.

    Defaults match the Exercise-5 topology: four 4x1024 SiLU networks per
    component, MinMax[-1.5,1.5] inputs and batch 4096. NAdam runs 140 epochs:
    1e-4 initially, a factor 0.1 every 20 epochs, and a floor of 1e-10.
    All epochs run by default and the final weights are retained. Validation
    and its best epoch are diagnostics, not a default stopping/selection rule.
    There is no dropout, weight decay, calibration, flow,
    or density evaluation. Validation banks must be independent simulations.

    Each completed member is committed separately, so interruption can lose
    only the member being trained. Training settings, input content and model
    contents enter cache contracts; cache damage raises instead of retraining.
    """
    cfg = _training_config(config)
    _rng(seed, "training-validation", "seed-check")
    arrays = {
        "train_signal": train_signal, "train_background": train_background, "train_ref": train_ref,
        "validation_signal": validation_signal, "validation_background": validation_background,
        "validation_ref": validation_ref,
    }
    arrays = {key: _finite_matrix(value, len(FEATURES), key) for key, value in arrays.items()}
    if len({len(arrays[name]) for name in ("train_signal", "train_background", "train_ref")}) != 1:
        raise ValueError("Training S, B and REF banks must have equal counts for balanced BCE.")
    if len({len(arrays[name]) for name in ("validation_signal", "validation_background", "validation_ref")}) != 1:
        raise ValueError("Validation S, B and REF banks must have equal counts.")
    hashes = {name: _array_hash(value) for name, value in arrays.items()}
    for left, left_value in arrays.items():
        if not left.startswith("train_"):
            continue
        for right, right_value in arrays.items():
            if right.startswith("validation_") and (
                hashes[left] == hashes[right] or np.shares_memory(left_value, right_value)
            ):
                raise ValueError(f"{left} and {right} overlap or are identical; generate independent banks.")
    device = _device(cfg)
    torch = _torch()
    print(f"NRE training/inference device: {device}")
    # Device and inference batching do not alter the mathematical model/cache.
    training_cfg = {key: value for key, value in cfg.items() if key not in ("device", "prediction_batch_size")}
    contract = {
        "version": TRAINING_VERSION, "config": training_cfg, "seed": int(seed),
        "inputs": hashes, "features": list(FEATURES), "optimizer": "NAdam",
        "input_scaling": "train-only MinMax[-1.5,1.5] shared across all members",
        "ensemble": "arithmetic mean of exp(BCE logit)",
    }
    directory = Path(root) / "training" / _fingerprint(contract)
    # One frozen train-only scaler, shared by both ensembles.
    minima = np.min(np.stack([arrays[name].min(axis=0) for name in
                             ("train_signal", "train_background", "train_ref")]), axis=0).astype(np.float64)
    maxima = np.max(np.stack([arrays[name].max(axis=0) for name in
                             ("train_signal", "train_background", "train_ref")]), axis=0).astype(np.float64)
    scale = 3.0 / np.where(maxima > minima, maxima - minima, 1.0)
    scaler_directory = directory / "scaler"
    if scaler_directory.exists():
        _verified_manifest(scaler_directory, contract)
        with np.load(scaler_directory / "scaler.npz", allow_pickle=False) as payload:
            cached_minima, cached_scale = payload["offset"], payload["scale"]
        if not np.array_equal(minima, cached_minima) or not np.array_equal(scale, cached_scale):
            raise CacheError("Cached scaler disagrees with its recorded input banks.")
    else:
        with _new_artifact(scaler_directory) as staging:
            np.savez(staging / "scaler.npz", offset=minima, scale=scale)
            _write_manifest(staging, contract, ["scaler.npz"])
    models, histories, member_hashes = {}, {}, {}
    for component_index, component in enumerate(("signal", "background")):
        models[component], histories[component] = [], []
        for member in range(cfg["ensemble_size"]):
            member_seed = int(seed) + component_index * 1_000_000 + member * 10_007
            member_contract = {**contract, "component": component, "member": member, "member_seed": member_seed}
            member_directory = directory / component / f"member_{member:02d}"
            if member_directory.exists():
                manifest = _verified_manifest(member_directory, member_contract)
                with (member_directory / "history.json").open() as handle:
                    history = json.load(handle)
                # NPZ keeps loading independent of pickle or torch checkpoint
                # serialization changes; strict state_dict keys/shapes remain.
                model = _network(cfg).to(device)
                with np.load(member_directory / "weights.npz", allow_pickle=False) as payload:
                    weights = {key: torch.from_numpy(payload[key].copy()) for key in payload.files}
                if any(not bool(torch.isfinite(value).all().item()) for value in weights.values()):
                    raise CacheError(f"Non-finite saved network weights: {member_directory}")
                try:
                    model.load_state_dict(weights, strict=True)
                except RuntimeError as error:
                    raise CacheError(f"Invalid saved member structure in {member_directory}: {error}") from error
                model.eval()
                print(f"Reused {component}/REF member {member + 1}/{cfg['ensemble_size']}.")
            else:
                print(f"Training {component}/REF member {member + 1}/{cfg['ensemble_size']}.", flush=True)
                with _new_artifact(member_directory) as staging:
                    model, history = _train_member(
                        arrays["train_" + component], arrays["train_ref"],
                        arrays["validation_" + component], arrays["validation_ref"],
                        cfg, member_seed, device, (minima, scale),
                    )
                    history["member"] = member
                    history["component"] = component
                    np.savez(staging / "weights.npz", **{
                        key: value.detach().cpu().numpy() for key, value in model.state_dict().items()
                    })
                    with (staging / "history.json").open("w") as handle:
                        handle.write(_json(history))
                    manifest = _write_manifest(staging, member_contract, ["weights.npz", "history.json"])
            models[component].append(model)
            histories[component].append(history)
            member_hashes[f"{component}/{member}"] = manifest["files"]["weights.npz"]
    return NREPredictor(models, (minima, scale), cfg,
                        _fingerprint({"contract": contract, "weights": member_hashes}), histories, device)
