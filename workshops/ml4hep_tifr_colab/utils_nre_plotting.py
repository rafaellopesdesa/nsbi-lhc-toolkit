"""Plots and standalone numerical figure exports for Exercise 12.

These helpers consume arrays, never neural networks.  Every public plotting
function saves a PDF and an editable Python reconstruction with its numerical
data embedded.  The reconstruction needs only NumPy and Matplotlib, not the
notebook, a trained classifier, or an external data file.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from utils_plotting import export_standalone_figure_script


__all__ = [
    "plot_training", "plot_ratio_validation", "plot_mle_convergence",
    "plot_asimov_scans", "plot_toy_comparison", "plot_compression_validation",
    "plot_simulator_score_closure",
]


_STYLE = {
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "mathtext.fontset": "dejavusans",
}


def _export(fig, output_dir, script_name):
    """Keep the shared exporter unchanged, adding PDF output to this script."""
    fig.tight_layout()
    script = export_standalone_figure_script(
        fig, script_name=script_name, output_dir=output_dir
    )
    fig.savefig(script.with_suffix(".pdf"), bbox_inches="tight")
    source = script.read_text(encoding="utf-8")
    # The existing workshop exporter reconstructs all artist data.  Preserve
    # its PNG output and also make the exported script reproduce the PDF.
    source = source.replace(
        "output_path = Path(__file__).with_suffix('.png')",
        "fig.savefig(Path(__file__).with_suffix('.pdf'), bbox_inches='tight')\n"
        "output_path = Path(__file__).with_suffix('.png')",
    )
    script.write_text(source, encoding="utf-8")
    return fig


def _array(values, name):
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a nonempty finite array.")
    return values


def _errorbar(ax, *args, label, **kwargs):
    # The shared exporter serializes artists, not ErrorbarContainer metadata.
    # Attach the public legend label to its marker line so it survives export.
    container = ax.errorbar(*args, **kwargs)
    container.lines[0].set_label(label)
    return container


def _ratios(values):
    if isinstance(values, dict):
        if "r_signal" in values:
            signal, background = values["r_signal"], values["r_background"]
        elif "rS" in values:
            signal, background = values["rS"], values["rB"]
        else:
            signal, background = values["signal"], values["background"]
    elif isinstance(values, tuple) and len(values) == 2:
        signal, background = values
    else:
        array = np.asarray(values)
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError("Ratios must be (rS, rB), a named dict, or an N-by-2 array.")
        signal, background = array[:, 0], array[:, 1]
    signal = _array(signal, "signal ratios")
    background = _array(background, "background ratios")
    if signal.shape != background.shape or np.any(signal <= 0) or np.any(background <= 0):
        raise ValueError("The two ratio arrays must match and be strictly positive.")
    return signal, background


def plot_training(histories, output_dir):
    """Plot every ensemble member's train/validation loss in two figures."""
    figures = {}
    with plt.rc_context(_STYLE):
        for component in ("signal", "background"):
            fig, ax = plt.subplots(figsize=(7, 5))
            members = histories[component]
            if not members:
                raise ValueError(f"No {component} training histories were supplied.")
            for index, history in enumerate(members):
                train = _array(history["train_loss"], "training loss")
                validation = _array(history["validation_loss"], "validation loss")
                ax.plot(
                    np.arange(1, len(train) + 1), train, color="C0", alpha=0.55,
                    label="Training (ensemble members)" if index == 0 else None,
                )
                ax.plot(
                    np.arange(1, len(validation) + 1), validation,
                    color="C3", alpha=0.65, linestyle="--",
                    label="Validation (ensemble members)" if index == 0 else None,
                )
            ax.set(xlabel="Epoch", ylabel="Binary cross-entropy", title=f"{component.capitalize()} / reference classifier")
            ax.grid(alpha=0.2)
            ax.legend(loc="upper right")
            figures[component] = _export(fig, output_dir, f"nre_training_{component}")
        # Old histories remain plottable; new runs also export the actual rate
        # used for every epoch and indicate the deployed checkpoint.
        if all("learning_rate" in history for members in histories.values() for history in members):
            fig, ax = plt.subplots(figsize=(7, 5))
            for component, color in (("signal", "C0"), ("background", "C1")):
                for index, history in enumerate(histories[component]):
                    rates = _array(history["learning_rate"], "learning rates")
                    if np.any(rates <= 0) or len(rates) != len(history["train_loss"]):
                        raise ValueError("Learning rates must be positive and match the training epochs.")
                    ax.plot(np.arange(1, len(rates) + 1), rates, color=color, alpha=0.6,
                            label=component.capitalize() if index == 0 else None)
                    selected = int(history.get("selected_epoch", history["best_epoch"]))
                    ax.plot(selected, rates[selected - 1], "o", color=color,
                            label="Selected weights" if component == "signal" and index == 0 else None)
            ax.set(yscale="log", xlabel="Epoch", ylabel="Learning rate")
            ax.grid(alpha=0.2)
            ax.legend(loc="upper right")
            figures["learning_rates"] = _export(fig, output_dir, "nre_training_learning_rates")
    return figures


def plot_simulator_score_closure(diagnostics, summary, output_dir):
    """MC uncertainty in the expected score, conditional on the frozen NRE."""
    scores = _array([item["score_at_truth"] for item in diagnostics], "scores")
    errors = _array([item["score_mc_se"] for item in diagnostics], "score errors")
    if np.any(errors < 0):
        raise ValueError("Score standard errors must be nonnegative.")
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        positions = np.arange(1, len(scores) + 1)
        _errorbar(ax, positions, scores, yerr=errors, fmt="o", color="C0", capsize=3,
                  label="Independent banks (MC error)")
        _errorbar(ax, [len(scores) + 1], [summary["score_mean"]],
                  yerr=[summary["score_mc_se"]], fmt="s", color="black", capsize=4,
                  label="Combined (propagated MC error)")
        ax.axhline(0, color="C3", linestyle="--", linewidth=1.4, label="Simulator score closure")
        ax.set_xticks(np.r_[positions, len(scores) + 1], [str(x) for x in positions] + ["Combined"])
        ax.set(xlabel="Independent integration bank",
               ylabel=rf"Expected likelihood score at $\mu={diagnostics[0]['mu_true']:g}$")
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
        return _export(fig, output_dir, "nre_simulator_score_closure")


def plot_ratio_validation(ref_ratios, signal_ratios, background_ratios, output_dir):
    """Compare held-out component samples with ratio-reweighted reference.

    Each pair of histograms uses identical bins and is normalized by its full
    sample/weight sum.  This is a diagnostic of the fitted ratios, not an
    assertion that the learned model exactly reproduces the simulator.
    """
    ref_s, ref_b = _ratios(ref_ratios)
    sig_s, sig_b = _ratios(signal_ratios)
    bkg_s, bkg_b = _ratios(background_ratios)
    ref_score = np.log(ref_s) - np.log(ref_b)
    sig_score = np.log(sig_s) - np.log(sig_b)
    bkg_score = np.log(bkg_s) - np.log(bkg_b)
    all_scores = np.concatenate((ref_score, sig_score, bkg_score))
    low, high = float(all_scores.min()), float(all_scores.max())
    padding = max(0.025 * (high - low), 0.05)
    bins = np.linspace(low - padding, high + padding, 51)
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
        for ax, label, target, weights in zip(
            axes, ("Signal", "Background"), (sig_score, bkg_score), (ref_s, ref_b)
        ):
            target_counts, _ = np.histogram(target, bins=bins)
            predicted, _ = np.histogram(ref_score, bins=bins, weights=weights / weights.sum())
            centers = 0.5 * (bins[:-1] + bins[1:])
            ax.stairs(predicted, bins, color="C0", linewidth=1.8, label="Ratio-reweighted reference")
            _errorbar(ax,
                centers, target_counts / len(target),
                yerr=np.sqrt(target_counts) / len(target), fmt="o", color="black",
                markersize=3, capsize=1.5, label="Held-out simulator",
            )
            ax.set(xlabel=r"$\log(\widehat r_S/\widehat r_B)$", ylabel="Probability / bin", title=label)
            ax.grid(alpha=0.2)
            ax.legend(loc="upper right")
        return _export(fig, output_dir, "nre_ratio_validation")


def plot_mle_convergence(study, output_dir):
    """Show individual repetitions and their mean +/- one repetition SD.

    The spread is not a standard error on the mean; finite Monte Carlo
    fluctuations need not approach the target monotonically.  Exact closure
    of the corrected finite-reference model is distinct from accuracy with
    respect to the simulator.
    """
    sizes = _array(study["sizes"], "reference sizes")
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        for key, color, label, marker in (
            ("raw", "C3", "Direct simulator Asimov", "o"),
            ("corrected", "C0", "Same-sample normalized Asimov", "s"),
        ):
            estimates = np.asarray([[r["mu_hat"] for r in repetition] for repetition in study[key]], dtype=float)
            if estimates.ndim != 2 or estimates.shape[1] != len(sizes) or estimates.shape[0] == 0:
                raise ValueError(f"{key} estimates must have shape (repetitions, sizes).")
            for repetition in estimates:
                ax.plot(sizes, repetition, color=color, alpha=0.12, linewidth=0.8)
            spread = estimates.std(axis=0, ddof=1) if len(estimates) > 1 else np.zeros(len(sizes))
            _errorbar(ax, sizes, estimates.mean(axis=0), yerr=spread, color=color,
                        fmt=f"{marker}-", linewidth=1.8, markersize=5, capsize=3,
                        label=label)
        ax.axhline(float(study.get("mu_true", 1.0)), color="black", linestyle=":", linewidth=1.2,
                   label=r"Generating value $\mu_A$")
        ax.set(xscale="log", xlabel="Events in each integration construction", ylabel=r"Asimov MLE $\widehat\mu_A$")
        ax.set_title("Lines: repetition means; error bars: repetition standard deviations")
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
        return _export(fig, output_dir, "nre_asimov_mle_convergence")


def plot_asimov_scans(results, sizes, label, output_dir, script_name):
    """Plot scans from one repetition, each referenced to its own minimum."""
    if len(results) != len(sizes) or not results:
        raise ValueError("One nonempty scan result is required per size.")
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        for index, (result, size) in enumerate(zip(results, sizes)):
            scan_mu = _array(result["scan_mu"], "scan mu")
            t_scan = _array(result["t_scan"], "scan statistic")
            if scan_mu.shape != t_scan.shape:
                raise ValueError("scan_mu and t_scan must have the same shape.")
            ax.plot(scan_mu, t_scan, color=f"C{index % 10}", linewidth=1.8,
                    label=fr"$M={int(size):,}$, $\widehat\mu_A={result['mu_hat']:.4f}$")
        ax.axhline(1.0, color="0.5", linestyle=":", linewidth=1)
        ax.set(xlabel=r"Signal strength $\mu$", ylabel=r"$t_{\mu,A}=-2\log[L_A(\mu)/L_A(\widehat\mu_A)]$", title=label)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
        return _export(fig, output_dir, script_name)


def _prediction_bin_probabilities(prediction, mu_edges, q_edges):
    """Gaussian/Wald and one-sided Cowan laws, including their zero atoms.

    The curvature Gaussian is centered on the actual Asimov MLE (not forced
    to one).  The q0 approximation uses sqrt(q0_Asimov) as its noncentral
    Gaussian shift.  Both are approximations, not finite-sample guarantees.
    """
    mean = float(prediction["mu_hat"])
    sigma = float(prediction["sigma_curvature"])
    q0_asimov = float(prediction["q0_asimov"])
    if not np.isfinite([mean, sigma, q0_asimov]).all() or sigma <= 0 or q0_asimov < 0:
        raise ValueError("Predictions require finite mu_hat, positive sigma_curvature and nonnegative q0_asimov.")
    # Physical fits have mu_hat >= 0.  CDF differences use F(0-)=0 so the
    # probability atom at the boundary belongs to the first histogram bin.
    mu_cdf = norm.cdf((mu_edges - mean) / sigma)
    mu_cdf[mu_edges < 0] = 0.0
    if mu_edges[0] == 0:
        mu_cdf[0] = 0.0
    q_cdf = norm.cdf(np.sqrt(np.maximum(q_edges, 0)) - np.sqrt(q0_asimov))
    q_cdf[q_edges < 0] = 0.0
    if q_edges[0] == 0:
        q_cdf[0] = 0.0
    return np.diff(mu_cdf), np.diff(q_cdf)


def _toy_histogram(ax, values, bins, label, color, markers=False):
    values = _array(values, label)
    if np.any(values < 0):
        raise ValueError("Physical toy estimates and q0 must be nonnegative.")
    counts, _ = np.histogram(values, bins=bins)
    probability = counts / values.size
    if markers:
        centers = 0.5 * (bins[:-1] + bins[1:])
        _errorbar(ax, centers, probability, yerr=np.sqrt(counts) / values.size,
                    fmt="o", color=color, markersize=3.0, capsize=1.5, label=label)
    else:
        ax.stairs(probability, bins, color=color, linewidth=1.8, label=label)


def plot_toy_comparison(toys_model, toys_simulator, predictions, labels, title,
                        output_dir, script_name):
    """Overlay two toy sources with binned Asimov asymptotic predictions.

    The right panel is logarithmic.  Counts are divided by the *total* toy
    count, including events beyond the displayed 99.9-percentile range; no
    truncated-distribution renormalization is performed.  Poisson counting
    errors are sqrt(n)/N.  Zero atoms enter the first bin for every curve.

    A raw simulator Asimov minimum at the physical boundary does not identify
    the unconstrained Gaussian center: it can be negative, not necessarily
    zero.  Its vanishing q0 also cannot determine the boundary probability.
    Both asymptotic overlays are therefore omitted for such raw constructions.
    This exclusion does not apply to a correctly normalized model generated
    at true mu=0, for which the usual null boundary law remains meaningful.
    """
    if len(predictions) != len(labels):
        raise ValueError("Each prediction needs a label.")
    model_mu, sim_mu = (_array(t["mu_hat"], "toy mu_hat") for t in (toys_model, toys_simulator))
    model_q, sim_q = (_array(t["q0"], "toy q0") for t in (toys_model, toys_simulator))
    mu_high = max(0.1, float(np.quantile(np.concatenate((model_mu, sim_mu)), 0.999)))
    q_high = max(1.0, float(np.quantile(np.concatenate((model_q, sim_q)), 0.999)))
    mu_edges = np.linspace(0.0, 1.04 * mu_high, 46)
    q_edges = np.linspace(0.0, 1.04 * q_high, 46)
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
        for ax, model, simulator, bins, xlabel in zip(
            axes, (model_mu, model_q), (sim_mu, sim_q), (mu_edges, q_edges),
            (r"Fitted signal strength $\widehat\mu$", r"Discovery statistic $q_0$"),
        ):
            _toy_histogram(ax, model, bins, "NRE-model toys", "0.45")
            _toy_histogram(ax, simulator, bins, "Simulator toys", "black", markers=True)
            ax.set(xlabel=xlabel, ylabel="Probability / bin", xlim=(bins[0], bins[-1]))
            ax.grid(alpha=0.2)
        omitted = []
        for index, (prediction, label) in enumerate(zip(predictions, labels)):
            if prediction.get("construction") == "raw_simulator" and (
                float(prediction["mu_hat"]) <= 1e-10
                or float(prediction["q0_asimov"]) <= 1e-12
            ):
                omitted.append(str(label))
                print(
                    f"Omitting both asymptotic overlays for {label}: the raw "
                    "Asimov fit is at/near the boundary; its unconstrained "
                    "Gaussian center and boundary mass are not determined."
                )
                continue
            mu_probabilities, q_probabilities = _prediction_bin_probabilities(prediction, mu_edges, q_edges)
            for ax, probabilities, bins in zip(axes, (mu_probabilities, q_probabilities), (mu_edges, q_edges)):
                ax.stairs(probabilities, bins, color=f"C{index % 10}", linewidth=1.8,
                          label=label)
        axes[1].set_yscale("log")
        axes[1].set_ylim(bottom=max(1e-7, 0.15 / max(len(model_q), len(sim_q))))
        axes[0].set_ylim(bottom=0)
        for ax in axes:
            ax.legend(loc="upper right")
        fig.suptitle(title)
        note = "Zero-boundary mass is included in the first bin; displayed tails are not renormalized."
        note += "".join(f"\nRaw boundary prediction omitted: {label}" for label in omitted)
        axes[0].text(0.0, -0.25, note,
                     transform=axes[0].transAxes, ha="left", fontsize=9)
        return _export(fig, output_dir, script_name)


def plot_compression_validation(validation, output_dir):
    """Show binned-score approximation errors on identical toy events.

    This numerical check isolates compression from statistical fluctuations:
    each plotted difference comes from fitting the same events twice.  The
    inference helper, not this visualization, applies the requested tolerance.
    """
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
        for ax, key, symbol in zip(axes, ("mu_hat", "q0"), (r"\widehat\mu", "q_0")):
            event = _array(validation[f"{key}_event"], f"event-level {key}")
            compressed = _array(validation[f"{key}_binned"], f"compressed {key}")
            if event.shape != compressed.shape:
                raise ValueError("Compression must be compared on the same toy events.")
            difference = compressed - event
            ax.plot(event, difference, "o", color="C0", markersize=4, alpha=0.7,
                    label="Identical events, two fits")
            ax.axhline(0.0, color="black", linestyle=":", linewidth=1)
            ax.set(xlabel=fr"Event-level ${symbol}$",
                   ylabel=fr"Compressed $-$ event-level ${symbol}$",
                   title=fr"Maximum $|\Delta|={np.max(np.abs(difference)):.2g}$; RMS $={np.sqrt(np.mean(difference**2)):.2g}$")
            ax.grid(alpha=0.2)
            ax.legend(loc="best")
        source = str(validation.get("source", "toys"))
        fig.suptitle(f"Score-compression validation: {source}")
        return _export(fig, output_dir, f"nre_compression_validation_{source}")
