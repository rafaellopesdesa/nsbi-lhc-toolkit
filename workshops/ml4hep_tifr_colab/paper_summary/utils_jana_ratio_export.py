"""Checkpoint-compatible exact-JANA ratio-bank export entry point.

The trained checkpoints fingerprint :mod:`utils_jana`, so sampling-only
numerical handling lives outside that module.  Ratio-bank construction uses
the same stable float64 inverse-spline evaluation as the exact-JANA route
evaluation, then returns to the checkpoint's float32 model dtype.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import utils_jana as jana
from utils_jana_evaluation import _install_bayesflow_numerical_guards
from utils_jana_checkpoint import (
    install_checkpoint_restore_hook, preserve_old_inference, stamp_inference_manifest,
)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parsed = jana._cli_parser().parse_args(arguments)
    if parsed.command != "export-ratio-bank":
        raise ValueError(
            "utils_jana_ratio_export.py supports only 'export-ratio-bank'."
        )
    install_checkpoint_restore_hook()
    preserve_old_inference(parsed.output_directory, "manifest.json")
    _install_bayesflow_numerical_guards()
    result = jana._main(arguments)
    if result == 0:
        stamp_inference_manifest(Path(parsed.output_directory) / "manifest.json")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
