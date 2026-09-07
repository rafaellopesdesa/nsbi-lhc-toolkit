"""Resilient installation wrapper for the transient exact-JANA runtime.

The exact JANA checkpoints fingerprint ``utils_jana.py`` and
``requirements_jana.txt``.  Runtime-installation recovery therefore lives in
this separate modern-Python helper: package-download failures can be repaired
without changing the scientific driver or invalidating trained checkpoints.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import utils_jana


def _output_tail(
    completed: subprocess.CompletedProcess[str], limit: int = 4000
) -> str:
    output = "\n".join(
        part.strip()
        for part in (completed.stdout or "", completed.stderr or "")
        if part.strip()
    )
    return output[-limit:] or "(no installer output was captured)"


def _run_repair(
    label: str,
    command: Sequence[str],
    *,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    print(f"[exact JANA runtime] {label}")
    return subprocess.run(
        [str(value) for value in command],
        text=True,
        capture_output=True,
        env=environment,
    )


def ensure_jana_environment(
    artifact_root: str | Path,
    install_if_missing: bool = True,
) -> Path:
    """Resolve exact JANA, resuming an interrupted transient installation.

    The audited installer remains the primary path.  If its all-at-once pip
    command is interrupted, the Python 3.11 environment and any packages that
    were installed successfully are retained.  A retry then installs only the
    missing pins.  A final ``uv pip --reinstall`` pass is used only when that
    resumed installation still fails validation.
    """

    artifact_root = Path(artifact_root).expanduser().resolve()
    try:
        return utils_jana.ensure_jana_environment(
            artifact_root,
            install_if_missing=install_if_missing,
        )
    except RuntimeError as error:
        if not install_if_missing:
            raise
        initial_error = error

    configured_environment = os.environ.get("PAPER_SUMMARY_JANA_ENV")
    if configured_environment:
        runtime_directory = Path(configured_environment).expanduser().resolve()
    elif Path("/content").is_dir():
        runtime_directory = Path("/content/paper_summary_jana_env")
    else:
        runtime_directory = artifact_root / "envs" / "jana"
    interpreter = runtime_directory / "bin" / "python"
    requirements = Path(utils_jana.__file__).resolve().parent / "requirements_jana.txt"

    if not interpreter.is_file() or not requirements.is_file():
        raise RuntimeError(
            "The exact-JANA installer failed before leaving a repairable Python "
            f"3.11 environment at {runtime_directory}. Original error: {initial_error}"
        ) from initial_error

    subprocess_environment = utils_jana._isolated_jana_subprocess_env()
    attempts: list[tuple[str, list[str]]] = [
        (
            "Resuming the pinned pip installation (up to 8 download retries).",
            [
                str(interpreter),
                "-m",
                "pip",
                "--isolated",
                "install",
                "--no-input",
                "--disable-pip-version-check",
                "--no-user",
                "--no-deps",
                "--retries",
                "8",
                "--timeout",
                "120",
                "--requirement",
                str(requirements),
            ],
        )
    ]

    uv_executable = shutil.which("uv")
    uv_prefix = [uv_executable] if uv_executable else [sys.executable, "-m", "uv"]
    attempts.append(
        (
            "Retrying the exact pins with uv after pip did not validate.",
            [
                *uv_prefix,
                "pip",
                "install",
                "--python",
                str(interpreter),
                "--no-deps",
                "--reinstall",
                "--requirement",
                str(requirements),
            ],
        )
    )

    diagnostics = []
    for label, command in attempts:
        completed = _run_repair(
            label,
            command,
            environment=subprocess_environment,
        )
        diagnostics.append(
            f"{label} return code {completed.returncode}:\n{_output_tail(completed)}"
        )
        if completed.returncode != 0:
            continue
        os.environ["PAPER_SUMMARY_JANA_PYTHON"] = str(interpreter)
        try:
            validated = utils_jana.resolve_jana_python(artifact_root)
        except RuntimeError as validation_error:
            diagnostics.append(f"Validation after {label}:\n{validation_error}")
            os.environ.pop("PAPER_SUMMARY_JANA_PYTHON", None)
            continue
        print("[exact JANA runtime] Repaired and validated the Python 3.11 environment.")
        return validated

    os.environ.pop("PAPER_SUMMARY_JANA_PYTHON", None)
    raise RuntimeError(
        "Could not repair the transient exact-JANA environment. The trained "
        "Drive checkpoints were not modified.\n\n"
        f"Initial installer error:\n{initial_error}\n\n"
        + "\n\n".join(diagnostics)
    ) from initial_error


__all__ = ["ensure_jana_environment"]
