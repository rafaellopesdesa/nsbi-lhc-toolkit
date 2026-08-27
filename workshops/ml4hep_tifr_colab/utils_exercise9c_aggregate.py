"""Aggregate collector and diagnostics for the Exercise 9c task family."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils_exercise9c_contract import (
    ALL_TASKS,
    METHOD_BINARY,
    METHOD_COLORS,
    METHOD_FLOW,
    METHOD_LABELS,
    METHOD_MULTICLASS,
    METHODS,
    TASK_TITLES,
    aggregate_run_tag,
    campaign_run_tag,
    normalize_profile,
    validate_seed,
)
from utils_plotting import export_standalone_figure_script


def _export(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    export_standalone_figure_script(
        fig, script_name=f"{stem}.py", output_dir=directory
    )
    fig.savefig(directory / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")


def collect_campaign(
    artifact_root: str | Path,
    *,
    profile: str,
    seeds: Iterable[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    artifact_root = Path(artifact_root).expanduser().resolve()
    profile = normalize_profile(profile)
    seeds = tuple(dict.fromkeys(validate_seed(seed) for seed in seeds))
    if not seeds:
        raise ValueError("At least one seed is required")
    frames = []
    status_rows = []
    for seed in seeds:
        run_tag = campaign_run_tag(profile, seed)
        for task in ALL_TASKS:
            directory = artifact_root / "results" / task / run_tag
            metrics_path = directory / "metrics.csv"
            status_path = directory / "status.json"
            row = {
                "task": task,
                "task_title": TASK_TITLES[task],
                "profile": profile,
                "seed": seed,
                "run_tag": run_tag,
                "metrics_path": str(metrics_path),
                "status_path": str(status_path),
                "complete": metrics_path.is_file() and status_path.is_file(),
            }
            if status_path.is_file():
                try:
                    status = json.loads(status_path.read_text())
                    row["reported_status"] = status.get("status")
                    row["campaign_signature"] = status.get("campaign_signature")
                except Exception as exc:
                    row["reported_status"] = f"invalid: {exc}"
            status_rows.append(row)
            if metrics_path.is_file():
                frame = pd.read_csv(metrics_path)
                required = {
                    "task",
                    "profile",
                    "seed",
                    "run_tag",
                    "method",
                    "num_observation",
                    "posterior_C2ST",
                    "predictive_x_C2ST",
                    "predictive_joint_C2ST",
                }
                missing = required - set(frame.columns)
                if missing:
                    raise RuntimeError(
                        f"{metrics_path} is missing columns {sorted(missing)}"
                    )
                if (
                    set(frame["task"]) != {task}
                    or set(frame["profile"]) != {profile}
                    or set(frame["seed"].astype(int)) != {seed}
                    or set(frame["run_tag"]) != {run_tag}
                ):
                    raise RuntimeError(f"Campaign identity mismatch in {metrics_path}")
                frames.append(frame)
    metrics = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return metrics, pd.DataFrame(status_rows)


def _mean_table(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "posterior_C2ST",
        "predictive_x_C2ST",
        "predictive_joint_C2ST",
        "posterior_MMD",
        "predictive_x_MMD",
        "predictive_joint_MMD",
        "posterior_ESS_fraction",
        "predictive_candidate_ESS_fraction_mean",
    ]
    return (
        metrics.groupby(["task", "method"], as_index=False)[columns]
        .mean()
        .sort_values(["task", "method"])
    )


def render_aggregate(
    artifact_root: str | Path,
    *,
    profile: str,
    seeds: Iterable[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    artifact_root = Path(artifact_root).expanduser().resolve()
    profile = normalize_profile(profile)
    seeds = tuple(dict.fromkeys(validate_seed(seed) for seed in seeds))
    metrics, status = collect_campaign(
        artifact_root, profile=profile, seeds=seeds
    )
    aggregate_tag = aggregate_run_tag(profile, seeds)
    output_dir = artifact_root / "aggregate" / aggregate_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    status.to_csv(output_dir / "task_status.csv", index=False)
    if metrics.empty:
        print("No completed 9c task metrics were found.")
        print(status[["task_title", "seed", "complete"]].to_string(index=False))
        return metrics, status
    metrics.to_csv(output_dir / "combined_metrics.csv", index=False)
    mean = _mean_table(metrics)
    mean.to_csv(output_dir / "task_method_means.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.8, 2.8), constrained_layout=True)
    matrix = status.pivot(index="seed", columns="task", values="complete").reindex(
        columns=ALL_TASKS
    )
    ax.imshow(matrix.to_numpy(dtype=float), vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set(
        title="9c artifact completeness (green = status + metrics)",
        xticks=np.arange(len(ALL_TASKS)),
        xticklabels=[TASK_TITLES[task] for task in ALL_TASKS],
        yticks=np.arange(len(matrix)),
        yticklabels=matrix.index,
        xlabel="SBIBM task",
        ylabel="campaign seed",
    )
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    _export(fig, output_dir, "campaign_completeness")
    plt.show()

    tasks_present = [task for task in ALL_TASKS if task in set(mean["task"])]
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.5), constrained_layout=True)
    for ax, metric, title in zip(
        axes,
        ("posterior_C2ST", "predictive_x_C2ST", "predictive_joint_C2ST"),
        ("posterior", "predictive x", "predictive joint"),
    ):
        x = np.arange(len(tasks_present))
        width = 0.24
        for index, method in enumerate(METHODS):
            values = (
                mean.loc[mean["method"] == method]
                .set_index("task")[metric]
                .reindex(tasks_present)
            )
            ax.bar(
                x + (index - 1) * width,
                values,
                width=width,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        ax.axhline(0.5, color="black", ls="--", lw=1)
        ax.set(
            title=f"mean {title} C2ST",
            xticks=x,
            xticklabels=[TASK_TITLES[task] for task in tasks_present],
            ylabel="C2ST",
            ylim=(0.45, 1.0),
        )
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=7)
    _export(fig, output_dir, "mean_c2st_all_tasks")
    plt.show()

    pivot = mean.pivot(index="task", columns="method")
    delta_columns = [
        "posterior_C2ST",
        "predictive_x_C2ST",
        "predictive_joint_C2ST",
    ]
    delta = np.column_stack(
        [
            (
                pivot[column][METHOD_MULTICLASS]
                - pivot[column][METHOD_BINARY]
            ).reindex(tasks_present)
            for column in delta_columns
        ]
    )
    fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    limit = max(0.01, float(np.nanmax(np.abs(delta))))
    image = ax.imshow(delta, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    for row in range(delta.shape[0]):
        for column in range(delta.shape[1]):
            ax.text(column, row, f"{delta[row, column]:+.3f}", ha="center", va="center", fontsize=8)
    ax.set(
        title="multiclass − separate-binary C2ST (negative favors multiclass)",
        xticks=range(3),
        xticklabels=["posterior", "predictive x", "predictive joint"],
        yticks=range(len(tasks_present)),
        yticklabels=[TASK_TITLES[task] for task in tasks_present],
    )
    fig.colorbar(image, ax=ax, label="C2ST difference")
    _export(fig, output_dir, "multiclass_minus_binary_c2st")
    plt.show()

    corrected = metrics.loc[metrics["method"].isin([METHOD_MULTICLASS, METHOD_BINARY])]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    for method in (METHOD_MULTICLASS, METHOD_BINARY):
        selected = corrected.loc[corrected["method"] == method]
        axes[0].scatter(
            selected["posterior_C2ST"],
            selected["posterior_ESS_fraction"],
            s=22,
            alpha=0.65,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
        axes[1].scatter(
            selected["predictive_joint_C2ST"],
            selected["predictive_candidate_ESS_fraction_mean"],
            s=22,
            alpha=0.65,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    axes[0].set(title="posterior accuracy vs weight efficiency", xlabel="posterior C2ST", ylabel="posterior ESS fraction")
    axes[1].set(title="predictive accuracy vs SIR efficiency", xlabel="predictive-joint C2ST", ylabel="candidate ESS fraction")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    _export(fig, output_dir, "accuracy_vs_weight_efficiency")
    plt.show()

    print(f"Collected {len(metrics):,} rows from {status['complete'].sum()}/{len(status)} task-seed runs")
    return metrics, status
