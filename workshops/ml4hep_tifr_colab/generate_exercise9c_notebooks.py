"""Generate the thin Exercise 9c task, core, and aggregate notebooks."""

from __future__ import annotations

import json
from pathlib import Path

from utils_exercise9c_contract import ALL_TASKS, TASK_NOTEBOOKS, TASK_TITLES


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


def badge(filename: str) -> str:
    url = (
        "https://colab.research.google.com/github/"
        f"{REPOSITORY}/blob/{BRANCH}/workshops/ml4hep_tifr_colab/{filename}"
    )
    return (
        f'<a href="{url}" target="_parent"><img '
        'src="https://colab.research.google.com/assets/colab-badge.svg" '
        'alt="Open In Colab"/></a>\n'
    )


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


COLAB_SETUP = r'''# Google Colab setup -- safe to rerun and a no-op outside Colab.
import importlib
import importlib.util
import os, sys, subprocess
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path

REPO_URL = "https://github.com/rafaellopesdesa/nsbi-lhc-toolkit.git"
BRANCH = "ml4hep_school_tutorial"
USE_DRIVE = os.environ.get("EX9C_USE_DRIVE", "1") != "0"

def run(*args, env=None):
    subprocess.run([str(arg) for arg in args], check=True, env=env)

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    if USE_DRIVE:
        from google.colab import drive
        if not Path("/content/drive/MyDrive").exists():
            drive.mount("/content/drive")
        SOURCE_ROOT = Path("/content")
        default_artifact_root = Path(
            "/content/drive/MyDrive/hybrid_nsbi_ml/exercise_9c_SBIBM_hybrid"
        )
    else:
        SOURCE_ROOT = Path("/content")
        default_artifact_root = Path("/content/exercise_9c_SBIBM_hybrid_artifacts")
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    REPO_DIR = SOURCE_ROOT / "nsbi-lhc-toolkit"
    TUTORIAL_DIR = REPO_DIR / "workshops" / "ml4hep_tifr_colab"
    if not (REPO_DIR / ".git").is_dir():
        clone_env = os.environ.copy()
        clone_env["GIT_LFS_SKIP_SMUDGE"] = "1"
        run(
            "git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
            "--branch", BRANCH, REPO_URL, REPO_DIR, env=clone_env,
        )
    else:
        run("git", "-C", REPO_DIR, "remote", "set-url", "origin", REPO_URL)
        run("git", "-C", REPO_DIR, "fetch", "origin", BRANCH)
        run("git", "-C", REPO_DIR, "checkout", BRANCH)
        run("git", "-C", REPO_DIR, "pull", "--ff-only", "origin", BRANCH)
    run(
        "git", "-C", REPO_DIR, "sparse-checkout", "set",
        "src", "workshops/ml4hep_tifr_colab",
    )
    for import_dir in (REPO_DIR / "src", TUTORIAL_DIR):
        path = str(import_dir.resolve())
        if path not in sys.path:
            sys.path.insert(0, path)

    def installed_version(distribution):
        try:
            return package_version(distribution)
        except PackageNotFoundError:
            return None

    if installed_version("nflows") != "0.14" or importlib.util.find_spec("pyro") is None:
        run(sys.executable, "-m", "pip", "install", "-q", "nflows==0.14", "pyro-ppl")
    if installed_version("sbibm") != "1.1.0":
        run(sys.executable, "-m", "pip", "install", "-q", "--no-deps", "sbibm==1.1.0")
    # SIR and Lotka--Volterra import the historical diffeqtorch layer. Their
    # launchers install audited Python solvers before any simulator call.
    if installed_version("julia") != "0.6.2" or importlib.util.find_spec("opt_einsum") is None:
        run(sys.executable, "-m", "pip", "install", "-q", "julia==0.6.2", "opt_einsum")
    if installed_version("diffeqtorch") != "1.0.0":
        run(sys.executable, "-m", "pip", "install", "-q", "--no-deps", "diffeqtorch==1.0.0")
    importlib.invalidate_caches()
    import nflows, pyro, sbibm
    assert installed_version("nflows") == "0.14"
    assert installed_version("sbibm") == "1.1.0"
    os.chdir(TUTORIAL_DIR)
else:
    default_artifact_root = Path.cwd() / "exercise_9c_SBIBM_hybrid_artifacts"
    for candidate in [Path.cwd(), Path.cwd() / "workshops" / "ml4hep_tifr_colab"]:
        if (candidate / "utils_exercise9c_hybrid.py").exists():
            sys.path.insert(0, str(candidate.resolve()))
            break

ARTIFACT_ROOT = Path(
    os.environ.get("EX9C_ARTIFACT_ROOT", str(default_artifact_root))
).expanduser().resolve()
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["EX9C_ARTIFACT_ROOT"] = str(ARTIFACT_ROOT)
print("Working directory:", Path.cwd())
print("Persistent Exercise-9c artifact root:", ARTIFACT_ROOT)
'''


def write_core() -> None:
    filename = "Exercise_9c_SBIBM_hybrid_Core.ipynb"
    cells = [
        markdown("ex9c-core-badge", badge(filename)),
        markdown(
            "ex9c-core-overview",
            r'''# Exercise 9c — the deliberately hybrid SBIBM campaign

This runner tests the original division of labor behind the hybrid model. The conditional flows are intentionally **modest normalized proposals**, not precision models: one member, four RQS coupling layers, width 64, two hidden layers, eight bins, linear tails, and no dropout. Precision is delegated to fresh-data density-ratio ensembles.

For a simulator joint (S(z,x)=p(z,x)), posterior reference (P(z,x)=q_\phi(z\mid x)p(x)), and likelihood reference (L(z,x)=p(z)q_\eta(x\mid z)), it compares:

1. direct samples from the modest flows;
2. one equal-prior three-class CE model, using (D_S/D_P) and (D_S/D_L);
3. two separate equal-prior binary CE models, (S\!:\!P) and (S\!:\!L).

Every deployed ratio is an arithmetic ensemble average of **member-wise direct float64 softmax probability quotients**. Logits are never exponentiated. All classifiers are plain ReLU MLPs trained with CE and Adam only—no dropout, weight decay, layer normalization, calibration loss, bridge loss, or normalization penalty.

The four simulator banks are role-separated and persistent: flow training, ratio training, ratio validation, and final audit. They use different seeds and cache paths. The audit bank is constructed only after checkpoint selection and never enters gradients or early stopping. Increasing the ratio bank therefore means genuinely fresh simulator information, not merely additional samples from the learned flow.
''',
        ),
        markdown(
            "ex9c-core-profiles",
            r'''## Compute profiles

`TUTORIAL` is the default Colab preview: 10k flow simulations, 100k fresh ratio-training pairs, 20k validation, 20k audit, and four members per ratio ensemble. `PAPER` is the intended scientific run: 1M fresh ratio-training pairs and ten 4×1024 ensembles. Because the separate-binary route contains two ensembles, PAPER trains 30 wide classifiers per task. `EXTREME` raises the ratio bank to 5M pairs to study saturation. Classifier compute is step-based, so bank size changes coverage without silently multiplying an epoch budget.

Run each task in its own Colab runtime. The source checkout is runtime-local and all caches/checkpoints/results use task- and run-specific paths on Drive.
''',
        ),
        code("ex9c-core-setup", COLAB_SETUP),
        code(
            "ex9c-core-contract",
            '''from utils_exercise9c_contract import PROFILES, campaign_run_tag, campaign_signature
PROFILE = os.environ.get("EX9C_PROFILE", "TUTORIAL").upper()
TASK_NAME = os.environ.get("EX9C_TASK", "two_moons")
BASE_SEED = int(os.environ.get("EX9C_SEED", "31082026"))
print(json.dumps({
    "task": TASK_NAME,
    "profile": PROFILE,
    "run_tag": campaign_run_tag(PROFILE, BASE_SEED),
    "campaign_signature": campaign_signature(PROFILE),
    "campaign": PROFILES[PROFILE],
    "four_bank_rule": ["flow", "ratio_train", "ratio_validation", "audit"],
    "comparison": ["modest flow", "one multiclass correction", "two binary corrections"],
}, indent=2))
''',
        ),
        markdown(
            "ex9c-core-diagnostics",
            r'''## Diagnostics produced by every task

The runner saves PNG, PDF, and standalone Python reproducer scripts for: flow training and fresh-bank NLL/tail checks; every classifier member's training CE, fresh validation CE, and learning rate; audit confusion matrices and calibration curves; multiclass-versus-binary log-ratio scatter plots and ensemble spread; fresh-bank reweighting closure and before/after C2ST; posterior comparisons on shared pooled ranges; log-weight spectra, ESS, and largest weights for every observation; posterior-predictive comparisons; and C2ST summaries across observations.

CSV/JSON/NPZ artifacts retain the bank provenance, flow and classifier audits, closure tests, posterior and predictive weights, direct method-versus-method C2ST, reference comparisons, and all sampled arrays. Treat low ESS, a dominant weight, weak closure, or large member disagreement as a failed hybrid approximation even if one marginal plot looks attractive.
''',
        ),
        code(
            "ex9c-core-run",
            '''from IPython.display import display
from utils_exercise9c_hybrid import run_from_environment

RESULT = run_from_environment()
display(RESULT.style.format(precision=4).hide(axis="index"))
''',
        ),
    ]
    (HERE / filename).write_text(json.dumps(notebook(cells), indent=1) + "\n")


def launcher_overview(task: str) -> str:
    title = TASK_TITLES[task]
    base = f'''# Exercise 9c — {title}: deliberately hybrid training

This notebook runs only the `{task}` SBIBM task and delegates to the shared Exercise-9c engine. It compares the same modest proposal flows against one multiclass ratio correction and two separate binary corrections, using fresh role-separated simulator banks and extensive audit-only diagnostics.

The default `TUTORIAL` profile is a substantial preview. Switch to `PAPER` for the intended 1M-pair, ten-member scientific campaign or `EXTREME` for the 5M-pair saturation test. Checkpoints and banks are persistent, fingerprinted by the campaign run tag, and safe to reuse with `LOAD_IF_AVAILABLE=True`.

After this finishes, run `Exercise_9c_SBIBM_hybrid.ipynb` to add the result to the cross-task comparison.
'''
    if task == "sir":
        base += r'''
## Audited SIR backend

Start from a fresh Colab runtime. The legacy Julia solver is replaced by the repository's audited vectorized RK4 backend; the official SIR equations, prior, Binomial observation law, observations, and reference posterior samples remain unchanged. The preflight comparison against DOP853 runs before the first simulation.
'''
    elif task == "lotka_volterra":
        base += r'''
## Audited Lotka–Volterra backend

Start from a fresh Colab runtime. The legacy Julia solver is replaced by the repository's audited positive log-space RK4 backend; the official model, prior, LogNormal observation law, observations, and reference posterior samples remain unchanged. Its numerical preflight runs before the first simulation.
'''
    return base


def write_launcher(task: str) -> None:
    filename = TASK_NOTEBOOKS[task]
    backend_lines = ""
    if task == "sir":
        backend_lines = 'os.environ["EX9C_SIR_BACKEND"] = "python_rk4"\n'
    elif task == "lotka_volterra":
        backend_lines = 'os.environ["EX9C_LOTKA_VOLTERRA_BACKEND"] = "python_logrk4"\n'
    config = f'''TASK_NAME = {task!r}
PROFILE = "TUTORIAL"  # SMOKE | TUTORIAL | PAPER | EXTREME
BASE_SEED = 31082026
LOAD_IF_AVAILABLE = True
# Set to an integer to override the profile's metric row cap, or leave None.
C2ST_MAX_SAMPLES = None

import os
os.environ["EX9C_TASK"] = TASK_NAME
os.environ["EX9C_PROFILE"] = PROFILE
os.environ["EX9C_SEED"] = str(BASE_SEED)
os.environ["EX9C_LOAD_IF_AVAILABLE"] = "1" if LOAD_IF_AVAILABLE else "0"
if C2ST_MAX_SAMPLES is not None:
    os.environ["EX9C_C2ST_MAX_SAMPLES"] = str(C2ST_MAX_SAMPLES)
else:
    os.environ.pop("EX9C_C2ST_MAX_SAMPLES", None)
{backend_lines}'''
    launch = '''import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/rafaellopesdesa/nsbi-lhc-toolkit.git"
BRANCH = "ml4hep_school_tutorial"
if "google.colab" in sys.modules:
    from google.colab import drive
    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive")
    repository = Path("/content/nsbi-lhc-toolkit")
    if not (repository / ".git").is_dir():
        clone_env = os.environ.copy()
        clone_env["GIT_LFS_SKIP_SMUDGE"] = "1"
        subprocess.run([
            "git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
            "--branch", BRANCH, REPO_URL, str(repository),
        ], check=True, env=clone_env)
    else:
        subprocess.run(["git", "-C", str(repository), "fetch", "origin", BRANCH], check=True)
        subprocess.run(["git", "-C", str(repository), "checkout", BRANCH], check=True)
        subprocess.run(["git", "-C", str(repository), "pull", "--ff-only", "origin", BRANCH], check=True)
    subprocess.run([
        "git", "-C", str(repository), "sparse-checkout", "set",
        "src", "workshops/ml4hep_tifr_colab",
    ], check=True)
    runner = repository / "workshops" / "ml4hep_tifr_colab" / "Exercise_9c_SBIBM_hybrid_Core.ipynb"
else:
    candidates = [
        Path.cwd() / "Exercise_9c_SBIBM_hybrid_Core.ipynb",
        Path.cwd() / "workshops" / "ml4hep_tifr_colab" / "Exercise_9c_SBIBM_hybrid_Core.ipynb",
    ]
    runner = next((path for path in candidates if path.exists()), None)
    if runner is None:
        raise FileNotFoundError("Cannot locate Exercise_9c_SBIBM_hybrid_Core.ipynb")

print("Running shared task engine:", runner)
get_ipython().run_line_magic("run", f'"{runner}"')
'''
    cells = [
        markdown(f"ex9c-{task}-badge", badge(filename)),
        markdown(f"ex9c-{task}-overview", launcher_overview(task)),
        code(f"ex9c-{task}-config", config),
        code(f"ex9c-{task}-launch", launch),
    ]
    (HERE / filename).write_text(json.dumps(notebook(cells), indent=1) + "\n")


def write_aggregate() -> None:
    filename = "Exercise_9c_SBIBM_hybrid.ipynb"
    setup = r'''import os, sys, subprocess
from pathlib import Path

PROFILE = "TUTORIAL"  # must match the task notebooks
SEEDS = [31082026]
USE_DRIVE = True

if "google.colab" in sys.modules:
    if USE_DRIVE:
        from google.colab import drive
        if not Path("/content/drive/MyDrive").exists():
            drive.mount("/content/drive")
        ARTIFACT_ROOT = Path("/content/drive/MyDrive/hybrid_nsbi_ml/exercise_9c_SBIBM_hybrid")
    else:
        ARTIFACT_ROOT = Path("/content/exercise_9c_SBIBM_hybrid_artifacts")
    repository = Path("/content/nsbi-lhc-toolkit")
    if not (repository / ".git").is_dir():
        clone_env = os.environ.copy()
        clone_env["GIT_LFS_SKIP_SMUDGE"] = "1"
        subprocess.run([
            "git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
            "--branch", "ml4hep_school_tutorial",
            "https://github.com/rafaellopesdesa/nsbi-lhc-toolkit.git", str(repository),
        ], check=True, env=clone_env)
    else:
        subprocess.run(["git", "-C", str(repository), "pull", "--ff-only", "origin", "ml4hep_school_tutorial"], check=True)
    subprocess.run(["git", "-C", str(repository), "sparse-checkout", "set", "src", "workshops/ml4hep_tifr_colab"], check=True)
    tutorial = repository / "workshops" / "ml4hep_tifr_colab"
    sys.path.insert(0, str(tutorial))
    os.chdir(tutorial)
else:
    ARTIFACT_ROOT = Path.cwd() / "exercise_9c_SBIBM_hybrid_artifacts"
    for candidate in [Path.cwd(), Path.cwd() / "workshops" / "ml4hep_tifr_colab"]:
        if (candidate / "utils_exercise9c_aggregate.py").exists():
            sys.path.insert(0, str(candidate.resolve()))
            break

print("Reading artifacts from:", ARTIFACT_ROOT)
'''
    cells = [
        markdown("ex9c-aggregate-badge", badge(filename)),
        markdown(
            "ex9c-aggregate-overview",
            '''# Exercise 9c — aggregate hybrid comparison

Run this after any subset of the ten task notebooks. It validates each task's profile, seed, and run-tag identity; partial campaigns are allowed and visibly marked. The figures compare the modest flow, one multiclass correction, and two separate binary corrections across posterior, predictive-data, and predictive-joint metrics. A dedicated delta heatmap answers the main question directly: where does multiclass factorization improve or degrade C2ST relative to separate binary estimators?

The collector also relates accuracy to posterior and predictive importance-weight efficiency, because a visually improved distribution with collapsing ESS is not a robust hybrid result.
''',
        ),
        code("ex9c-aggregate-setup", setup),
        code(
            "ex9c-aggregate-run",
            '''from IPython.display import display
from utils_exercise9c_aggregate import render_aggregate

COMBINED_METRICS, TASK_STATUS = render_aggregate(
    ARTIFACT_ROOT, profile=PROFILE, seeds=SEEDS
)
display(TASK_STATUS.style.hide(axis="index"))
if not COMBINED_METRICS.empty:
    display(COMBINED_METRICS.style.format(precision=4).hide(axis="index"))
''',
        ),
    ]
    (HERE / filename).write_text(json.dumps(notebook(cells), indent=1) + "\n")


def main() -> None:
    write_core()
    for task in ALL_TASKS:
        write_launcher(task)
    write_aggregate()
    print("Generated Exercise 9c core, ten task launchers, and aggregate notebook")


if __name__ == "__main__":
    main()
