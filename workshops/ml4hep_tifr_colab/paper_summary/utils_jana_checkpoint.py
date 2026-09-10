"""Recover the pinned BayesFlow model without changing training fingerprints.

At the pinned revision, ``Orthogonal.W`` is a seeded *Tensor*, not a Variable.
It is neither optimized nor serialized by tf.train.Checkpoint. Reconstructing
the architecture with the training seed is therefore part of restoring a model.
Checkpoint integrity checks still apply to every serialized weight.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import utils_jana as jana


INFERENCE_REVISION = "seeded_orthogonal_restore_v1"


def inference_manifest_current(manifest):
    return isinstance(manifest, dict) and manifest.get("inference_revision") == INFERENCE_REVISION


def install_checkpoint_restore_hook():
    """Install once per driver import, including in either isolated worker."""
    original = jana.load_exact_jana
    if getattr(original, "_seeded_orthogonal_restore", False):
        return

    @functools.wraps(original)
    def load(run_directory, *, strict_runtime=True):
        manifest_path = Path(run_directory).expanduser().resolve() / "checkpoint_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if "seed" not in manifest:
            raise RuntimeError(f"Cannot reconstruct JANA rotations: missing training seed in {manifest_path}.")
        # Match train_exact_jana: seed immediately before build_exact_jana.
        # Reapply on EVERY load; sampling also resets the global RNG streams.
        jana.seed_everything(int(manifest["seed"]))
        tf = jana._legacy_imports()["tf"]
        tf.config.experimental.enable_tensor_float_32_execution(False)
        return original(run_directory, strict_runtime=strict_runtime)

    load._seeded_orthogonal_restore = True
    jana.load_exact_jana = load


def install_evaluation_cache_guard():
    """Reject results produced by the former, randomly reconstructed model."""
    original = jana._evaluation_output_manifest_valid
    if getattr(original, "_seeded_orthogonal_restore", False):
        return

    @functools.wraps(original)
    def valid(manifest_path):
        try:
            manifest = json.loads(Path(manifest_path).read_text())
        except (OSError, ValueError):
            return False
        return inference_manifest_current(manifest) and original(manifest_path)

    valid._seeded_orthogonal_restore = True
    jana._evaluation_output_manifest_valid = valid


def preserve_old_inference(directory, manifest_name):
    """Archive derived products only; callers own the relevant per-run claim."""
    directory = Path(directory)
    if not directory.is_dir() or not any(directory.iterdir()):
        return None
    try:
        manifest = json.loads((directory / manifest_name).read_text())
    except (OSError, ValueError):
        manifest = None
    if inference_manifest_current(manifest):
        return None
    archive = directory.with_name(directory.name + ".recovery-before-seeded-restore")
    number = 2
    while archive.exists():
        archive = directory.with_name(directory.name + f".recovery-before-seeded-restore-{number}")
        number += 1
    directory.rename(archive)
    print(f"[exact JANA recovery] Preserved old inference outputs in {archive}; reusing trained flows.", flush=True)
    return archive


def stamp_inference_manifest(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    manifest["inference_revision"] = INFERENCE_REVISION
    manifest["rotation_restore"] = "training seed before model construction; original checkpoint weights"
    jana._atomic_write_json(path, manifest)
