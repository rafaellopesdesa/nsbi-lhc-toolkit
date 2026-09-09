"""Checkpoint-compatible exact-JANA ratio-bank export entry point.

The trained checkpoints fingerprint :mod:`utils_jana`, so sampling-only
numerical handling lives outside that module.  Ratio-bank construction uses
the same stable float64 inverse-spline evaluation as the exact-JANA route
evaluation, then returns to the checkpoint's float32 model dtype.
"""

from __future__ import annotations

import sys
from typing import Sequence

import utils_jana as jana
from utils_jana_evaluation import _install_bayesflow_numerical_guards


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parsed = jana._cli_parser().parse_args(arguments)
    if parsed.command != "export-ratio-bank":
        raise ValueError(
            "utils_jana_ratio_export.py supports only 'export-ratio-bank'."
        )
    _install_bayesflow_numerical_guards()
    return jana._main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
