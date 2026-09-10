"""Reuse notebook 02's exact-JANA checkpoints in 03 without flow training.

Keep this orchestration outside utils_jana.py: changing that scientific
driver would invalidate every existing checkpoint's source fingerprint.
The legacy driver still validates the full training contract and file hashes.
"""

from __future__ import annotations

import functools
import inspect
import json
from pathlib import Path
import subprocess

import utils_jana as jana
from utils_jana_gpu import GPU_BATCH_SIZE_BY_BUDGET


def require_completed_checkpoints(artifact_root, pairs):
    """Fail before exporting large ratio banks if a prerequisite is missing."""
    missing = []
    for budget, seed in pairs:
        root = jana.default_run_directory(artifact_root, budget=budget, seed=seed)
        try:
            manifest = json.loads((root / "checkpoint_manifest.json").read_text())
            complete = manifest.get("status") == "complete"
        except (OSError, ValueError, AttributeError):
            complete = False
        if not complete:
            missing.append(f"budget={int(budget)}/seed={int(seed)}")
    if missing:
        raise RuntimeError(
            "Notebook 03 requires completed exact-JANA flow checkpoints. "
            "Missing or unfinished: " + ", ".join(missing) + ". "
            "Finish these runs in notebook 02 on a GPU with LOAD_IF_AVAILABLE=True, "
            "then rerun 03. Existing completed trainings and resume checkpoints "
            "are preserved; 03 does not start or restart flow training."
        )


def install_reuse_hook():
    """Validate using 02's batch sizes; never enter fresh/resumed training."""
    original = jana.train_exact_jana
    signature = inspect.signature(original)

    @functools.wraps(original)
    def reuse(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        options = bound.arguments
        root = options["run_directory"] or jana.default_run_directory(
            options["artifact_root"], budget=options["budget"], seed=options["seed"]
        )
        root = Path(root).expanduser().resolve()
        manifest_path = root / "checkpoint_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text())
            valid = jana._manifest_checkpoint_valid(
                root, manifest["training_contract_sha256"]
            )
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            valid = False
        if not valid:
            raise RuntimeError(
                f"No complete, intact exact-JANA checkpoint at {root}. "
                "Check or finish this run in notebook 02; 03 will not retrain it."
            )
        # The GPU optimization override is part of the training contract even
        # when the completed model is reused/evaluated on a CPU in notebook 03.
        options["batch_size"] = GPU_BATCH_SIZE_BY_BUDGET.get(
            int(options["budget"]), jana.DEFAULT_BATCH_SIZE
        )
        options["load_if_available"] = True
        options["force"] = False
        result = original(*bound.args, **bound.kwargs)
        print(
            f"[exact JANA reuse] {root.parent.name}/{root.name}: "
            f"completed checkpoint, batch size {options['batch_size']}; no flow training.",
            flush=True,
        )
        return result

    jana.train_exact_jana = reuse


def launch_saved_campaign(python_executable, **kwargs):
    """Match the campaign launcher interface while allowing checkpoint reuse only."""
    command = [str(python_executable), "-u", str(Path(__file__).resolve()), "campaign"]
    for key in ("artifact_root", "master_bank_path", "shape_bank_path", "pilot_bank_path", "validation_bank_path"):
        command += ["--" + key.removesuffix("_path").replace("_", "-"),
                    str(Path(kwargs[key]).expanduser().resolve())]
    for key in ("budgets", "seeds"):
        command += ["--" + key, *map(lambda value: str(int(value)), kwargs[key])]
    command += ["--profile", str(kwargs.get("profile", "PAPER"))]
    completed = subprocess.run(
        command, text=True, capture_output=True,
        env=jana._isolated_jana_subprocess_env(),
    )
    if completed.returncode:
        details = "\n".join(part.strip() for part in
                            (completed.stdout, completed.stderr) if part and part.strip())
        raise RuntimeError(
            "Exact-JANA checkpoint reuse failed; no flow training was requested. "
            "Resolve the prerequisite in notebook 02 rather than forcing a restart in 03. "
            "Isolated-process diagnostics:\n" + (details[-12000:] or "(no subprocess output)")
        )
    if completed.stdout.strip():
        print(completed.stdout.strip())
    summary = Path(kwargs["artifact_root"]).expanduser().resolve() / "jana_paper" / "campaign_manifest.json"
    return json.loads(summary.read_text())


if __name__ == "__main__":
    install_reuse_hook()
    raise SystemExit(jana._main())
