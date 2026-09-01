"""Publication-oriented plotting helpers for the SLCP paper campaign.

The functions in this module deliberately accept either pandas data frames,
records, or CSV paths.  They also write an informative placeholder instead of
failing when an upstream stage has not produced the requested columns yet.
Every figure is saved as both a high-resolution PNG and a vector PDF.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

# Notebooks run both locally and on headless Colab workers.  Selecting the
# renderer before importing pyplot makes saving deterministic in either case.
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


METHOD_ORDER = (
    "jana_paper",
    "jana_paper_corrected_multiclass",
    "jana_paper_corrected_binary",
    "separate_flows",
    "separate_flows_corrected_multiclass",
    "separate_flows_corrected_binary",
)
METHOD_LABELS = {
    "jana_paper": "JANA-paper",
    "jana_paper_corrected_multiclass": "JANA + ratio (multiclass)",
    "jana_paper_corrected_binary": "JANA + ratio (binary)",
    "separate_flows": "Separate flows",
    "separate_flows_corrected_multiclass": "Separate flows + ratio (multiclass)",
    "separate_flows_corrected_binary": "Separate flows + ratio (binary)",
}
METHOD_COLORS = {
    "jana_paper": "#4C78A8",
    "jana_paper_corrected_multiclass": "#72B7B2",
    "jana_paper_corrected_binary": "#F2CF5B",
    "separate_flows": "#7A5195",
    "separate_flows_corrected_multiclass": "#E45756",
    "separate_flows_corrected_binary": "#54A24B",
}
METHOD_MARKERS = {
    "jana_paper": "o",
    "jana_paper_corrected_multiclass": "v",
    "jana_paper_corrected_binary": "P",
    "separate_flows": "s",
    "separate_flows_corrected_multiclass": "^",
    "separate_flows_corrected_binary": "D",
}

_DEFAULT_METRIC_ORDER = (
    "posterior_C2ST",
    "likelihood_posterior_C2ST",
    "posterior_likelihood_route_C2ST",
    "posterior_MMD",
    "likelihood_posterior_MMD",
    "posterior_likelihood_route_MMD",
    "predictive_x_C2ST",
    "predictive_joint_C2ST",
    "posterior_ESS_fraction",
    "likelihood_posterior_ESS_fraction",
    "posterior_max_weight",
    "likelihood_posterior_max_weight",
    "bayes_cycle_pearson",
    "bayes_cycle_slope",
    "bayes_cycle_residual_rms",
    "likelihood_log_Z_rms",
    "exact_likelihood_log_error",
)


def set_publication_style() -> None:
    """Apply a compact journal-friendly style without requiring seaborn."""

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.linewidth": 1.6,
            "lines.markersize": 5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _coerce_frame(data: Any) -> pd.DataFrame:
    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, (str, Path)):
        path = Path(data).expanduser()
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    if isinstance(data, Mapping):
        try:
            return pd.DataFrame(data)
        except ValueError:
            return pd.DataFrame([data])
    try:
        return pd.DataFrame(list(data))
    except (TypeError, ValueError):
        return pd.DataFrame()


def _safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")
    return stem or "figure"


def _save_figure(
    figure: Figure,
    output_dir: str | Path,
    stem: str,
    *,
    dpi: int = 300,
) -> dict[str, Path]:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(stem)
    paths = {
        "png": destination / f"{stem}.png",
        "pdf": destination / f"{stem}.pdf",
    }
    figure.savefig(paths["png"], dpi=int(dpi), bbox_inches="tight")
    figure.savefig(paths["pdf"], bbox_inches="tight")
    plt.close(figure)
    return paths


def _placeholder(
    output_dir: str | Path,
    stem: str,
    title: str,
    message: str,
) -> dict[str, Path]:
    set_publication_style()
    figure, axis = plt.subplots(figsize=(7.0, 2.4), constrained_layout=True)
    axis.set_axis_off()
    axis.set_title(title, loc="left", fontweight="bold")
    axis.text(
        0.5,
        0.48,
        message,
        ha="center",
        va="center",
        transform=axis.transAxes,
        color="0.35",
        wrap=True,
    )
    return _save_figure(figure, output_dir, stem)


def _budget_label(value: Any) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return str(value)
    if number >= 1_000_000 and number % 1_000_000 == 0:
        return f"{number // 1_000_000}M"
    if number >= 1_000 and number % 1_000 == 0:
        return f"{number // 1_000}k"
    return f"{number:,}"


def _method_sort_key(method: Any) -> tuple[int, str]:
    text = str(method)
    try:
        return METHOD_ORDER.index(text), text
    except ValueError:
        return len(METHOD_ORDER), text


def _metric_label(metric: str) -> str:
    replacements = {
        "C2ST": "C2ST",
        "MMD": "MMD",
        "ESS": "ESS",
        "rms": "RMS",
        "pearson": "Pearson r",
    }
    label = metric.replace("_", " ")
    for source, target in replacements.items():
        label = re.sub(source, target, label, flags=re.IGNORECASE)
    return label


def _selected_architecture(
    selection: Mapping[str, Any] | None, budget: Any, route: Any
) -> str | None:
    if not selection:
        return None
    routes = selection.get("routes", selection)
    budget_block = routes.get(str(int(budget)), routes.get(int(budget), {}))
    route_block = budget_block.get(str(route), {}) if isinstance(budget_block, Mapping) else {}
    if isinstance(route_block, Mapping):
        architecture = route_block.get("architecture")
        return None if architecture is None else str(architecture)
    return str(route_block) if route_block else None


def plot_capacity_scan(
    scan: Any,
    output_dir: str | Path,
    *,
    selection: Mapping[str, Any] | None = None,
    filename_stem: str = "flow_capacity_scan",
) -> dict[str, Path]:
    """Plot held-out NLL by flow capacity, faceted by route and budget.

    Replicate error bars are standard errors across ML seeds.  A star marks
    the one-standard-error-rule choice when ``selection`` is supplied (or when
    a boolean ``selected`` column exists).
    """

    frame = _coerce_frame(scan)
    required = {"budget", "route", "validation_nll"}
    missing = sorted(required.difference(frame.columns))
    if frame.empty or missing:
        detail = "No capacity-scan rows are available." if frame.empty else (
            "Missing columns: " + ", ".join(missing)
        )
        return _placeholder(
            output_dir, filename_stem, "Flow capacity selection", detail
        )

    frame = frame.copy()
    frame["validation_nll"] = pd.to_numeric(
        frame["validation_nll"], errors="coerce"
    )
    frame = frame[np.isfinite(frame["validation_nll"])].copy()
    if frame.empty:
        return _placeholder(
            output_dir,
            filename_stem,
            "Flow capacity selection",
            "Validation NLL contains no finite values.",
        )
    if "architecture" not in frame:
        if {"n_coupling_layers", "hidden_features"}.issubset(frame.columns):
            frame["architecture"] = frame.apply(
                lambda row: (
                    f"{int(row['n_coupling_layers'])} blocks, "
                    f"{int(row['hidden_features'])} wide"
                ),
                axis=1,
            )
        else:
            frame["architecture"] = "candidate"

    grouping = ["budget", "route", "architecture"]
    for optional in ("n_coupling_layers", "hidden_features", "parameter_count"):
        if optional in frame:
            grouping.append(optional)
    aggregate = (
        frame.groupby(grouping, dropna=False, as_index=False)
        .agg(
            mean_nll=("validation_nll", "mean"),
            std_nll=("validation_nll", "std"),
            n=("validation_nll", "count"),
        )
    )
    aggregate["sem_nll"] = aggregate["std_nll"] / np.sqrt(
        aggregate["n"].clip(lower=1)
    )
    aggregate["sem_nll"] = aggregate["sem_nll"].fillna(0.0)

    budgets = sorted(aggregate["budget"].unique(), key=lambda value: float(value))
    routes = sorted(aggregate["route"].astype(str).unique())
    set_publication_style()
    nrows, ncols = len(routes), len(budgets)
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(4.0, 3.8 * ncols), max(3.2, 3.0 * nrows)),
        squeeze=False,
        constrained_layout=True,
        sharey="row",
    )
    for row_index, route in enumerate(routes):
        for column_index, budget in enumerate(budgets):
            axis = axes[row_index, column_index]
            subset = aggregate[
                (aggregate["route"].astype(str) == route)
                & (aggregate["budget"] == budget)
            ].copy()
            if "parameter_count" in subset and subset["parameter_count"].notna().any():
                subset = subset.sort_values(["parameter_count", "mean_nll"])
            elif {"n_coupling_layers", "hidden_features"}.issubset(subset.columns):
                subset = subset.sort_values(
                    ["n_coupling_layers", "hidden_features", "mean_nll"]
                )
            else:
                subset = subset.sort_values(["architecture"])
            positions = np.arange(len(subset), dtype=float)
            axis.errorbar(
                positions,
                subset["mean_nll"],
                yerr=subset["sem_nll"],
                fmt="o-",
                color="#4C78A8" if route == "posterior" else "#F58518",
                capsize=2.5,
            )
            chosen = _selected_architecture(selection, budget, route)
            if chosen is not None:
                mask = subset["architecture"].astype(str) == chosen
            elif "selected" in frame:
                selected_names = set(
                    frame.loc[
                        (frame["budget"] == budget)
                        & (frame["route"].astype(str) == route)
                        & frame["selected"].astype(bool),
                        "architecture",
                    ].astype(str)
                )
                mask = subset["architecture"].astype(str).isin(selected_names)
            else:
                mask = pd.Series(False, index=subset.index)
            if mask.any():
                axis.scatter(
                    positions[np.asarray(mask)],
                    subset.loc[mask, "mean_nll"],
                    marker="*",
                    s=110,
                    color="#D62728",
                    edgecolor="white",
                    linewidth=0.6,
                    zorder=5,
                    label="one-SE choice",
                )
                axis.legend(loc="best")
            labels = subset["architecture"].astype(str).tolist()
            axis.set_xticks(positions, labels, rotation=55, ha="right")
            axis.set_title(f"{route.capitalize()} · N={_budget_label(budget)}")
            if column_index == 0:
                axis.set_ylabel("Validation NLL")
            axis.set_xlabel("Architecture")
    figure.suptitle("Conditional-flow capacity scan", fontweight="bold")
    return _save_figure(figure, output_dir, filename_stem)


def _wide_metric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Accept either the campaign wide schema or a metric/value long table."""

    if {"metric", "value"}.issubset(frame.columns):
        identifiers = [
            column
            for column in ("method", "budget", "ml_seed", "observation")
            if column in frame
        ]
        if not identifiers:
            return frame
        return (
            frame.pivot_table(
                index=identifiers,
                columns="metric",
                values="value",
                aggfunc="mean",
            )
            .reset_index()
            .rename_axis(columns=None)
        )
    return frame


def _metric_columns(frame: pd.DataFrame, metrics: Sequence[str] | None) -> list[str]:
    if metrics is not None:
        return [metric for metric in metrics if metric in frame]
    ordered = [metric for metric in _DEFAULT_METRIC_ORDER if metric in frame]
    if ordered:
        return ordered
    metadata = {
        "budget",
        "ml_seed",
        "observation",
        "method",
        "schema",
        "campaign_signature",
    }
    return [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in metadata
    ]


def _seed_level_summary(
    frame: pd.DataFrame, metric: str
) -> pd.DataFrame:
    """Aggregate observations first, then use seed-to-seed variation."""

    columns = [
        column
        for column in ("method", "budget", "ml_seed", metric)
        if column in frame
    ]
    working = frame[columns].copy()
    working[metric] = pd.to_numeric(working[metric], errors="coerce")
    working = working[np.isfinite(working[metric])]
    if working.empty:
        return pd.DataFrame()
    if "method" not in working:
        working["method"] = "result"
    if "budget" not in working:
        working["budget"] = 1
    if "ml_seed" not in working:
        working["ml_seed"] = np.arange(len(working), dtype=int)
    seed_level = (
        working.groupby(["method", "budget", "ml_seed"], as_index=False)[metric]
        .mean()
    )
    summary = (
        seed_level.groupby(["method", "budget"], as_index=False)
        .agg(mean=(metric, "mean"), std=(metric, "std"), n=(metric, "count"))
    )
    summary["sem"] = summary["std"] / np.sqrt(summary["n"].clip(lower=1))
    summary["sem"] = summary["sem"].fillna(0.0)
    return summary


def _add_metric_reference(axis: Axes, metric: str) -> None:
    if "c2st" in metric.lower():
        axis.axhline(0.5, color="0.25", linewidth=0.9, linestyle="--", alpha=0.75)
    elif "ess_fraction" in metric.lower():
        axis.axhline(1.0, color="0.25", linewidth=0.9, linestyle="--", alpha=0.75)


def plot_metric_summary(
    results: Any,
    output_dir: str | Path,
    *,
    metrics: Sequence[str] | None = None,
    filename_stem: str = "paper_metric_summary",
    title: str = "SLCP benchmark summary",
) -> dict[str, Path]:
    """Plot route-aware diagnostics versus simulation budget.

    Each point averages observations within a seed and each error bar is the
    standard error across ML seeds.  Observation-level tables remain the
    canonical result; this plot is only the compact cross-budget summary.
    """

    frame = _wide_metric_frame(_coerce_frame(results))
    selected_metrics = _metric_columns(frame, metrics)
    if frame.empty or not selected_metrics:
        detail = "No result rows are available." if frame.empty else (
            "None of the requested metric columns are available."
        )
        return _placeholder(output_dir, filename_stem, title, detail)

    set_publication_style()
    ncols = 3 if len(selected_metrics) > 4 else min(2, len(selected_metrics))
    nrows = int(math.ceil(len(selected_metrics) / ncols))
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.25 * ncols, 3.05 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    legend_handles: dict[str, Any] = {}
    for metric_index, metric in enumerate(selected_metrics):
        axis = axes.flat[metric_index]
        summary = _seed_level_summary(frame, metric)
        if summary.empty:
            axis.set_axis_off()
            axis.text(0.5, 0.5, f"No finite values\n{metric}", ha="center", va="center")
            continue
        methods = sorted(summary["method"].astype(str).unique(), key=_method_sort_key)
        for method_index, method in enumerate(methods):
            subset = summary[summary["method"].astype(str) == method].copy()
            subset["budget_numeric"] = pd.to_numeric(subset["budget"], errors="coerce")
            subset = subset[np.isfinite(subset["budget_numeric"])].sort_values("budget_numeric")
            if subset.empty:
                continue
            color = METHOD_COLORS.get(method, plt.cm.tab10(method_index % 10))
            marker = METHOD_MARKERS.get(method, "o")
            artist = axis.errorbar(
                subset["budget_numeric"],
                subset["mean"],
                yerr=subset["sem"],
                color=color,
                marker=marker,
                capsize=2.5,
                label=METHOD_LABELS.get(method, method.replace("_", " ")),
            )
            legend_handles.setdefault(method, artist)
        positive_budgets = pd.to_numeric(summary["budget"], errors="coerce")
        if np.all(positive_budgets > 0) and positive_budgets.nunique() > 1:
            axis.set_xscale("log")
        axis.set_xlabel("Simulation budget")
        axis.set_ylabel(_metric_label(metric))
        axis.set_title(_metric_label(metric))
        _add_metric_reference(axis, metric)
    for unused in range(len(selected_metrics), nrows * ncols):
        axes.flat[unused].set_axis_off()
    if legend_handles:
        ordered = sorted(legend_handles, key=_method_sort_key)
        figure.legend(
            [legend_handles[key] for key in ordered],
            [METHOD_LABELS.get(key, key.replace("_", " ")) for key in ordered],
            loc="outside upper center",
            ncol=min(4, len(ordered)),
        )
    figure.suptitle(title, fontweight="bold")
    return _save_figure(figure, output_dir, filename_stem)


def plot_ess_summary(
    results: Any,
    output_dir: str | Path,
    *,
    metrics: Sequence[str] | None = None,
    filename_stem: str = "importance_weight_efficiency",
) -> dict[str, Path]:
    """Plot posterior- and likelihood-route ESS fractions across budgets."""

    frame = _wide_metric_frame(_coerce_frame(results))
    if metrics is None:
        metrics = (
            "posterior_ESS_fraction",
            "likelihood_posterior_ESS_fraction",
        )
    available = [metric for metric in metrics if metric in frame]
    if frame.empty or not available:
        return _placeholder(
            output_dir,
            filename_stem,
            "Importance-weight efficiency",
            "No ESS-fraction columns are available.",
        )

    set_publication_style()
    figure, axes = plt.subplots(
        1,
        len(available),
        figsize=(max(5.2, 4.5 * len(available)), 3.5),
        squeeze=False,
        constrained_layout=True,
        sharey=True,
    )
    legend_handles: dict[str, Any] = {}
    for metric_index, metric in enumerate(available):
        axis = axes[0, metric_index]
        summary = _seed_level_summary(frame, metric)
        methods = sorted(summary["method"].astype(str).unique(), key=_method_sort_key)
        for method_index, method in enumerate(methods):
            subset = summary[summary["method"].astype(str) == method].copy()
            subset["budget_numeric"] = pd.to_numeric(subset["budget"], errors="coerce")
            subset = subset[np.isfinite(subset["budget_numeric"])].sort_values("budget_numeric")
            if subset.empty:
                continue
            artist = axis.errorbar(
                subset["budget_numeric"],
                subset["mean"],
                yerr=subset["sem"],
                marker=METHOD_MARKERS.get(method, "o"),
                color=METHOD_COLORS.get(method, plt.cm.tab10(method_index % 10)),
                capsize=2.5,
                label=METHOD_LABELS.get(method, method.replace("_", " ")),
            )
            legend_handles.setdefault(method, artist)
        numeric_budget = pd.to_numeric(summary["budget"], errors="coerce")
        if np.all(numeric_budget > 0) and numeric_budget.nunique() > 1:
            axis.set_xscale("log")
        axis.axhline(1.0, color="0.25", linewidth=0.9, linestyle="--", alpha=0.75)
        axis.set_ylim(bottom=0.0)
        axis.set_xlabel("Simulation budget")
        axis.set_title(_metric_label(metric))
        if metric_index == 0:
            axis.set_ylabel("Effective sample-size fraction")
    if legend_handles:
        ordered = sorted(legend_handles, key=_method_sort_key)
        figure.legend(
            [legend_handles[key] for key in ordered],
            [METHOD_LABELS.get(key, key.replace("_", " ")) for key in ordered],
            loc="outside upper center",
            ncol=min(4, len(ordered)),
        )
    figure.suptitle("Importance-weight efficiency", fontweight="bold")
    return _save_figure(figure, output_dir, filename_stem)


def _coerce_samples(
    samples: Any, reference: np.ndarray | None
) -> dict[str, np.ndarray]:
    if isinstance(samples, Mapping):
        arrays = {str(label): np.asarray(value) for label, value in samples.items()}
    else:
        arrays = {"Posterior": np.asarray(samples)} if samples is not None else {}
    if reference is not None:
        arrays = {"Reference": np.asarray(reference), **arrays}
    cleaned: dict[str, np.ndarray] = {}
    for label, array in arrays.items():
        if array.ndim == 1:
            array = array[:, None]
        elif array.ndim > 2:
            array = array.reshape(array.shape[0], -1)
        if array.ndim == 2 and array.shape[0] > 0 and array.shape[1] > 0:
            cleaned[label] = np.asarray(array, dtype=np.float64)
    return cleaned


def plot_posterior_marginals(
    samples: Any,
    output_dir: str | Path,
    *,
    reference: np.ndarray | None = None,
    parameter_labels: Sequence[str] | None = None,
    weights: Mapping[str, np.ndarray] | None = None,
    bins: int = 45,
    filename_stem: str = "posterior_marginals",
    title: str = "Posterior marginals",
) -> dict[str, Path]:
    """Overlay one-dimensional posterior marginals for any number of methods.

    ``samples`` may be a single ``(N, D)`` array or a label-to-array mapping.
    Optional per-label importance weights are normalized by ``numpy.histogram``.
    Non-finite rows are removed separately for every dimension.
    """

    arrays = _coerce_samples(samples, reference)
    if not arrays:
        return _placeholder(
            output_dir, filename_stem, title, "No posterior samples are available."
        )
    dimensions = min(array.shape[1] for array in arrays.values())
    if dimensions <= 0:
        return _placeholder(
            output_dir, filename_stem, title, "Posterior arrays have no dimensions."
        )
    labels = list(parameter_labels or [rf"$\theta_{index + 1}$" for index in range(dimensions)])
    if len(labels) < dimensions:
        labels.extend(
            rf"$\theta_{index + 1}$" for index in range(len(labels), dimensions)
        )
    set_publication_style()
    ncols = min(3, dimensions)
    nrows = int(math.ceil(dimensions / ncols))
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.1 * ncols, 2.75 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    method_names = list(arrays)
    for dimension in range(dimensions):
        axis = axes.flat[dimension]
        finite_values = [
            array[:, dimension][np.isfinite(array[:, dimension])]
            for array in arrays.values()
        ]
        pooled = np.concatenate([value for value in finite_values if value.size])
        if pooled.size == 0:
            axis.set_axis_off()
            continue
        lower, upper = np.quantile(pooled, [0.0025, 0.9975])
        if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
            lower, upper = float(np.nanmin(pooled)), float(np.nanmax(pooled))
            padding = max(abs(lower) * 0.05, 0.5) if lower == upper else 0.0
            lower, upper = lower - padding, upper + padding
        edges = np.linspace(lower, upper, max(6, int(bins)) + 1)
        for method_index, method in enumerate(method_names):
            raw = arrays[method][:, dimension]
            mask = np.isfinite(raw)
            values = raw[mask]
            if values.size == 0:
                continue
            sample_weights = None
            if weights is not None and method in weights:
                candidate_weights = np.asarray(weights[method], dtype=np.float64).reshape(-1)
                if len(candidate_weights) == len(raw):
                    sample_weights = candidate_weights[mask]
                    sample_weights = np.where(np.isfinite(sample_weights), sample_weights, 0.0)
            density, _ = np.histogram(
                values,
                bins=edges,
                weights=sample_weights,
                density=True,
            )
            centers = 0.5 * (edges[:-1] + edges[1:])
            normalized_name = method.lower().replace(" ", "_")
            color = "#222222" if method.lower() == "reference" else METHOD_COLORS.get(
                normalized_name, plt.cm.tab10(method_index % 10)
            )
            linestyle = "--" if method.lower() == "reference" else "-"
            axis.plot(
                centers,
                density,
                color=color,
                linestyle=linestyle,
                label=METHOD_LABELS.get(normalized_name, method),
            )
        axis.set_xlabel(labels[dimension])
        axis.set_ylabel("Density" if dimension % ncols == 0 else "")
    for unused in range(dimensions, nrows * ncols):
        axes.flat[unused].set_axis_off()
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            legend_labels,
            loc="outside upper center",
            ncol=min(4, len(handles)),
        )
    figure.suptitle(title, fontweight="bold")
    return _save_figure(figure, output_dir, filename_stem)


__all__ = [
    "METHOD_COLORS",
    "METHOD_LABELS",
    "METHOD_ORDER",
    "plot_capacity_scan",
    "plot_ess_summary",
    "plot_metric_summary",
    "plot_posterior_marginals",
    "set_publication_style",
]
