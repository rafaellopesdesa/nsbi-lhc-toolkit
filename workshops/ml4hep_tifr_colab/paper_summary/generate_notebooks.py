"""Generate the five thin, unexecuted SLCP paper-summary notebooks."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
BRANCH = "ml4hep_school_tutorial"
REPOSITORY = "rafaellopesdesa/nsbi-lhc-toolkit"


def markdown(cell_id: str, source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(cell_id: str, source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def badge(filename: str) -> str:
    url = (
        "https://colab.research.google.com/github/"
        f"{REPOSITORY}/blob/{BRANCH}/workshops/ml4hep_tifr_colab/"
        f"paper_summary/{filename}"
    )
    return (
        f'<a href="{url}" target="_parent"><img '
        'src="https://colab.research.google.com/assets/colab-badge.svg" '
        'alt="Open In Colab"/></a>\n'
    )


COMMON_SETUP = r'''# Google Colab setup -- safe to rerun and a no-op outside Colab.
import importlib.util
import json
import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path

# Must be set before the first CUDA/PyTorch initialization in this process.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REPO_URL = "https://github.com/rafaellopesdesa/nsbi-lhc-toolkit.git"
BRANCH = "ml4hep_school_tutorial"
USE_DRIVE = os.environ.get("PAPER_SUMMARY_USE_DRIVE", "1") != "0"

def run(*args, env=None):
    subprocess.run([str(arg) for arg in args], check=True, env=env)

def installed_version(distribution):
    try:
        return package_version(distribution)
    except PackageNotFoundError:
        return None

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    if USE_DRIVE:
        from google.colab import drive
        if not Path("/content/drive/MyDrive").exists():
            drive.mount("/content/drive")
        default_artifact_root = Path(
            "/content/drive/MyDrive/hybrid_nsbi_ml/paper_summary_SLCP"
        )
    else:
        default_artifact_root = Path("/content/paper_summary_SLCP_artifacts")

    repository = Path("/content/nsbi-lhc-toolkit")
    if not (repository / ".git").is_dir():
        clone_env = os.environ.copy()
        clone_env["GIT_LFS_SKIP_SMUDGE"] = "1"
        run(
            "git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
            "--branch", BRANCH, REPO_URL, repository, env=clone_env,
        )
    else:
        run("git", "-C", repository, "remote", "set-url", "origin", REPO_URL)
        run("git", "-C", repository, "fetch", "origin", BRANCH)
        run("git", "-C", repository, "checkout", BRANCH)
        run("git", "-C", repository, "pull", "--ff-only", "origin", BRANCH)
    run(
        "git", "-C", repository, "sparse-checkout", "set", "src",
        "workshops/ml4hep_tifr_colab/paper_summary",
    )
    SOURCE_DIR = repository / "workshops" / "ml4hep_tifr_colab" / "paper_summary"

    modern_requirements = []
    if installed_version("nflows") != "0.14":
        modern_requirements.append("nflows==0.14")
    if installed_version("sbibm") != "1.1.0":
        modern_requirements.append("sbibm==1.1.0")
    if importlib.util.find_spec("pyro") is None:
        modern_requirements.append("pyro-ppl")
    if modern_requirements:
        run(sys.executable, "-m", "pip", "install", "-q", *modern_requirements)
else:
    candidates = (
        Path.cwd(),
        Path.cwd() / "paper_summary",
        Path.cwd() / "workshops" / "ml4hep_tifr_colab" / "paper_summary",
    )
    SOURCE_DIR = next(
        (candidate.resolve() for candidate in candidates if (candidate / "config.py").is_file()),
        None,
    )
    if SOURCE_DIR is None:
        raise FileNotFoundError("Cannot locate the paper_summary source directory")
    default_artifact_root = SOURCE_DIR / "artifacts"

source_path = str(SOURCE_DIR)
if source_path not in sys.path:
    sys.path.insert(0, source_path)
os.chdir(SOURCE_DIR)

ARTIFACT_ROOT = Path(
    os.environ.get("PAPER_SUMMARY_ARTIFACT_ROOT", str(default_artifact_root))
).expanduser().resolve()
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
print("Paper-summary source:", SOURCE_DIR)
print("Persistent artifact root:", ARTIFACT_ROOT)
'''


COMMON_CONFIGURATION = r'''from config import (
    DEFAULT_ML_SEEDS,
    PAPER_BUDGETS,
    SMOKE_BUDGETS,
    SMOKE_ML_SEEDS,
    campaign_config,
    campaign_signature,
)

PROFILE = os.environ.get("PAPER_SUMMARY_PROFILE", "PAPER").upper()
CAMPAIGN_BUDGETS = list(PAPER_BUDGETS if PROFILE == "PAPER" else SMOKE_BUDGETS)
CAMPAIGN_ML_SEEDS = list(DEFAULT_ML_SEEDS if PROFILE == "PAPER" else SMOKE_ML_SEEDS)

def execution_subset(environment_name, configured):
    raw = os.environ.get(environment_name, "").strip()
    values = list(configured) if not raw else [int(value) for value in raw.split(",")]
    unknown = set(values) - set(configured)
    if not values or unknown:
        raise ValueError(f"Invalid {environment_name}: {values}; unknown={sorted(unknown)}")
    return values

BUDGETS_TO_RUN = execution_subset("PAPER_SUMMARY_RUN_BUDGETS", CAMPAIGN_BUDGETS)
ML_SEEDS_TO_RUN = execution_subset("PAPER_SUMMARY_RUN_SEEDS", CAMPAIGN_ML_SEEDS)
LOAD_IF_AVAILABLE = os.environ.get("PAPER_SUMMARY_LOAD_IF_AVAILABLE", "1") != "0"

CAMPAIGN = campaign_config(profile=PROFILE)
print(json.dumps({
    "profile": PROFILE,
    "campaign_budgets": CAMPAIGN_BUDGETS,
    "campaign_ml_seeds": CAMPAIGN_ML_SEEDS,
    "budgets_to_run": BUDGETS_TO_RUN,
    "ml_seeds_to_run": ML_SEEDS_TO_RUN,
    "load_if_available": LOAD_IF_AVAILABLE,
    "campaign_signature": campaign_signature(CAMPAIGN),
}, indent=2))
'''


DISPLAY_RESULT = r'''from IPython.display import display

def display_result(result):
    if hasattr(result, "style"):
        display(result.style.format(precision=4).hide(axis="index"))
    elif isinstance(result, dict):
        for name, value in result.items():
            print(f"\n{name}")
            if hasattr(value, "style"):
                display(value.style.format(precision=4).hide(axis="index"))
            else:
                display(value)
    else:
        display(result)
'''


NOTEBOOKS = {
    "00_prepare_SLCP_samples.ipynb": [
        markdown("paper-00-badge", badge("00_prepare_SLCP_samples.ipynb")),
        markdown(
            "paper-00-overview",
            r'''# 00 — Prepare the paired SLCP simulation banks

This stage is the only training-data simulator entry point.  It generates one
master prior-predictive SLCP bank and materializes deterministic nested budget
views.  A fixed master-row assignment supplies the common train/validation
split to every matched flow and classifier.  It also creates the separate,
fixed two-row shape-inference bank, two-row consistency/ActNorm pilot, and
300-row validation bank required by the exact upstream JANA process; its
reported simulator cost is therefore N training calls plus 304.  A fifth,
separately seeded audit bank is never exposed to model or proposal selection.

The manifest verifies row counts, nested splits, non-overlap of training and
validation indices, distinct five-bank provenance, and SHA256 array
fingerprints.  All later notebooks load these artifacts and must fail loudly
if any identity check changes.
''',
        ),
        code("paper-00-setup", COMMON_SETUP),
        code("paper-00-config", COMMON_CONFIGURATION),
        code("paper-00-display-helper", DISPLAY_RESULT),
        code(
            "paper-00-run",
            '''from utils import prepare_slcp_banks

BANK_RESULT = prepare_slcp_banks(
    artifact_root=ARTIFACT_ROOT,
    campaign=CAMPAIGN,
    load_if_available=LOAD_IF_AVAILABLE,
)
display_result(BANK_RESULT)
''',
        ),
    ],
    "01_SLCP_flow_capacity.ipynb": [
        markdown("paper-01-badge", badge("01_SLCP_flow_capacity.ipynb")),
        markdown(
            "paper-01-overview",
            r'''# 01 — Select matched posterior and likelihood flow capacity

For every simulation budget and ML seed, this notebook trains one RQS capacity
screen per preregistered architecture on the same complete training partition.
Posterior and likelihood architectures are selected separately by held-out NLL
using the one-standard-error rule across ML seeds.  The selected architecture
is then deployed as a four-member ensemble in the following notebooks.  Every
flow runs the full six-plateau schedule and deploys its final epoch.

No official posterior samples, audit-bank rows, C2ST values, or residual-ratio
diagnostics participate in selection.  The selected nominal checkpoints are
the separate-flow baseline and are loaded unchanged by its correction stage.
''',
        ),
        code("paper-01-setup", COMMON_SETUP),
        code("paper-01-config", COMMON_CONFIGURATION),
        code("paper-01-display-helper", DISPLAY_RESULT),
        code(
            "paper-01-run",
            '''from utils import run_capacity_scan

CAPACITY_RESULT = run_capacity_scan(
    artifact_root=ARTIFACT_ROOT,
    campaign=CAMPAIGN,
    budgets_to_run=BUDGETS_TO_RUN,
    ml_seeds_to_run=ML_SEEDS_TO_RUN,
    load_if_available=LOAD_IF_AVAILABLE,
)
display_result(CAPACITY_RESULT)
''',
        ),
    ],
    "02_SLCP_JANA.ipynb": [
        markdown("paper-02-badge", badge("02_SLCP_JANA.ipynb")),
        markdown(
            "paper-02-overview",
            r'''# 02 — JANA-paper and separate-flow baselines

This notebook produces two deliberately distinct baselines.  **JANA-paper**
runs the pinned algorithm and hyperparameters in its isolated legacy
TensorFlow/BayesFlow environment, with only the simulator input adapted to the
fixed nested banks.  In keeping with upstream, this row uses N training pairs,
the two-row shape bank, the fixed two-row Trainer pilot, and the fixed 300-pair
validation bank, and is labelled N+304 in resource tables.  **Separate flows** loads the nominal
posterior and likelihood ensembles selected in notebook 01.

Both baselines receive the full paired diagnostic suite: posterior-route and
likelihood-route C2ST/MMD, route agreement, predictive closure, importance
efficiency, Bayes-cycle and conditional-normalization checks, and comparison
with the analytic SLCP likelihood.  Each is the direct control for corrections
trained over that same flow base in notebook 03.
''',
        ),
        code("paper-02-setup", COMMON_SETUP),
        code("paper-02-config", COMMON_CONFIGURATION),
        code(
            "paper-02-options",
            '''RUN_EXACT_JANA_PAPER = True
RUN_NOMINAL_MATCHED = True
INSTALL_EXACT_JANA_ENV_IF_MISSING = (
    os.environ.get("PAPER_SUMMARY_INSTALL_JANA_ENV", "1") != "0"
)
''',
        ),
        code(
            "paper-02-jana-environment",
            '''if RUN_EXACT_JANA_PAPER:
    from utils_jana import ensure_jana_environment

    print("Preparing the isolated exact-JANA runtime (first install can take several minutes).")
    JANA_PYTHON = ensure_jana_environment(
        ARTIFACT_ROOT,
        install_if_missing=INSTALL_EXACT_JANA_ENV_IF_MISSING,
    )
    print("Exact-JANA Python:", JANA_PYTHON)
''',
        ),
        code("paper-02-display-helper", DISPLAY_RESULT),
        code(
            "paper-02-run",
            '''from utils import run_jana_campaign

JANA_RESULT = run_jana_campaign(
    artifact_root=ARTIFACT_ROOT,
    campaign=CAMPAIGN,
    run_exact_paper=RUN_EXACT_JANA_PAPER,
    run_matched=RUN_NOMINAL_MATCHED,
    budgets_to_run=BUDGETS_TO_RUN,
    ml_seeds_to_run=ML_SEEDS_TO_RUN,
    load_if_available=LOAD_IF_AVAILABLE,
)
display_result(JANA_RESULT)
''',
        ),
    ],
    "03_SLCP_hybrid.ipynb": [
        markdown("paper-03-badge", badge("03_SLCP_hybrid.ipynb")),
        markdown(
            "paper-03-overview",
            r'''# 03 — Ratio corrections over both flow bases

This stage fits both correction factorizations over both pretrained bases.  For
exact JANA it uses the nominal JANA posterior and likelihood proposals, as the
reproduction flow is already expected to be accurate.  For the separate flows
selected in notebook 01, the learned transports are driven by a single
broadened Gaussian latent base; the posterior proposal adds a defensive prior
component, while the observation-space likelihood proposal uses broadening
alone.  No posterior or likelihood flow is retrained here.

At 100k, a disjoint train/checkpoint-validation/untouched-closure split and
importance efficiency select one `(tau, epsilon)` pair without access to
reference posterior samples.  The rule takes the worst route and worst of the
multiclass/binary factorizations, so proposal tuning is symmetric.  That pair
is then frozen across budgets.  One equal-prior three-class CE ensemble is
compared with two independent equal-prior binary CE ensembles for each base.
All genuine classifier examples come from the same training rows used by the
corresponding flows.  The 100k proposal ablation applies to the separate-flow
base and is retained as a control, not as another headline pipeline.
''',
        ),
        code("paper-03-setup", COMMON_SETUP),
        code("paper-03-config", COMMON_CONFIGURATION),
        code(
            "paper-03-options",
            '''FACTORIZATIONS = ("multiclass", "binary")
RUN_PROPOSAL_ABLATION = True
INSTALL_EXACT_JANA_ENV_IF_MISSING = (
    os.environ.get("PAPER_SUMMARY_INSTALL_JANA_ENV", "1") != "0"
)
''',
        ),
        code(
            "paper-03-jana-environment",
            '''from utils_jana import ensure_jana_environment

print("Preparing the isolated exact-JANA runtime (first install can take several minutes).")
JANA_PYTHON = ensure_jana_environment(
    ARTIFACT_ROOT,
    install_if_missing=INSTALL_EXACT_JANA_ENV_IF_MISSING,
)
print("Exact-JANA Python:", JANA_PYTHON)
''',
        ),
        code("paper-03-display-helper", DISPLAY_RESULT),
        code(
            "paper-03-run",
            '''from utils import run_hybrid_campaign

HYBRID_RESULT = run_hybrid_campaign(
    artifact_root=ARTIFACT_ROOT,
    campaign=CAMPAIGN,
    factorizations=FACTORIZATIONS,
    run_proposal_ablation=RUN_PROPOSAL_ABLATION,
    budgets_to_run=BUDGETS_TO_RUN,
    ml_seeds_to_run=ML_SEEDS_TO_RUN,
    load_if_available=LOAD_IF_AVAILABLE,
)
display_result(HYBRID_RESULT)
''',
        ),
    ],
    "04_SLCP_comparison.ipynb": [
        markdown("paper-04-badge", badge("04_SLCP_comparison.ipynb")),
        markdown(
            "paper-04-overview",
            r'''# 04 — Paired SLCP paper comparison

This notebook performs no training and no simulation.  It validates campaign,
bank, checkpoint, budget, and ML-seed identities before combining completed
artifacts.  Tables and plots preserve observation-level results and report
paired seed uncertainties in addition to summaries across observations.

The primary contrasts are paired within each base: exact JANA versus its two
corrections, and separate flows versus their two corrections.  Posterior and
likelihood routes receive equal status, and accuracy is always shown together
with importance-weight efficiency and the exact-likelihood audit.
''',
        ),
        code("paper-04-setup", COMMON_SETUP),
        code("paper-04-config", COMMON_CONFIGURATION),
        code(
            "paper-04-options",
            '''REQUIRE_COMPLETE = PROFILE == "PAPER"
''',
        ),
        code("paper-04-display-helper", DISPLAY_RESULT),
        code(
            "paper-04-run",
            '''from utils import build_paper_comparison

COMPARISON_RESULT = build_paper_comparison(
    artifact_root=ARTIFACT_ROOT,
    campaign=CAMPAIGN,
    require_complete=REQUIRE_COMPLETE,
)
display_result(COMPARISON_RESULT)
''',
        ),
    ],
}


def write_notebooks() -> None:
    """Write all generated notebooks with deterministic formatting."""

    for filename, cells in NOTEBOOKS.items():
        path = HERE / filename
        path.write_text(json.dumps(notebook(cells), indent=1) + "\n")
        print("Wrote", path.name)


def main() -> None:
    write_notebooks()


if __name__ == "__main__":
    main()
