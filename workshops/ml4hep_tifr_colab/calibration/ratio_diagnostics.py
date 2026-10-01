"""Read-only, held-out diagnostics for a density-ratio ensemble.

The two classes must be independently sampled, equally sized evaluation banks.
All members and the exact-ratio oracle are evaluated on these same events. No
member is selected, fitted, recalibrated, or assigned a new ensemble weight.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit, logsumexp


_Z95 = 1.959963984540054


def _validate_inputs(member_num, member_den, exact_num, exact_den):
    arrays = [np.asarray(a, dtype=np.float64) for a in
              (member_num, member_den, exact_num, exact_den)]
    mn, md, en, ed = arrays
    if mn.ndim != 2 or md.ndim != 2 or mn.shape != md.shape:
        raise ValueError("Member log ratios must have the same (members, events) shape.")
    if mn.shape[0] < 1 or mn.shape[1] < 2:
        raise ValueError("At least one member and two independent events per class are required.")
    if en.shape != (mn.shape[1],) or ed.shape != (md.shape[1],):
        raise ValueError("Exact log ratios must be one-dimensional, matching each balanced class.")
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("Every learned and exact log ratio must be finite; no clipping is applied.")
    return arrays


def _edges(values):
    edges = np.asarray(values, dtype=np.float64)
    if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
            or edges[0] != 0.0 or edges[-1] != 1.0 or np.any(np.diff(edges) <= 0)):
        raise ValueError("Score edges must increase strictly from 0 to 1, inclusive.")
    return edges


def _quantile_edges(scores, nbins=25):
    # Include both endpoints, including when expit rounds a very large logit to 1.
    return np.unique(np.r_[0.0, np.quantile(scores, np.arange(1, nbins) / nbins), 1.0])


def _groups(scores, edges):
    return np.searchsorted(edges[1:-1], scores, side="right")


def _divide(a, b):
    return np.divide(a, b, out=np.full(np.shape(a), np.nan, dtype=float), where=b > 0)


def _calibration_bins(sn, sd, edges):
    size = len(edges) - 1
    gn, gd = _groups(sn, edges), _groups(sd, edges)
    nn, nd = np.bincount(gn, minlength=size), np.bincount(gd, minlength=size)
    count = nn + nd
    mean_score = _divide(np.bincount(gn, weights=sn, minlength=size)
                         + np.bincount(gd, weights=sd, minlength=size), count)
    fraction = _divide(nn, count)
    inv_n = _divide(np.ones(size), count)
    denominator = 1 + _Z95 ** 2 * inv_n
    center = (fraction + 0.5 * _Z95 ** 2 * inv_n) / denominator
    half_width = _Z95 * np.sqrt(fraction * (1 - fraction) * inv_n
                              + 0.25 * _Z95 ** 2 * inv_n ** 2) / denominator
    residual = fraction - mean_score
    # Also export a class-stratified delta-method SE for the residual itself.
    # Its influence functions include fluctuations of the estimated mean score.
    # Unlike Wilson, this approximation can degenerate in extremely sparse bins.
    variance = np.zeros(size)
    for groups, score, target in ((gn, sn, 1.0), (gd, sd, 0.0)):
        influence = target - score - residual[groups]
        total = np.bincount(groups, weights=influence, minlength=size)
        total_sq = np.bincount(groups, weights=influence ** 2, minlength=size)
        variance += np.maximum(total_sq - total ** 2 / len(score), 0) / (len(score) * (len(score) - 1))
    residual_se = _divide(np.sqrt(variance), count / len(sn))
    return pd.DataFrame({
        "bin": np.arange(size), "score_left": edges[:-1], "score_right": edges[1:],
        "score_mean": mean_score, "n_num": nn, "n_den": nd, "n_total": count,
        "mass_num": nn / len(sn), "mass_den": nd / len(sd),
        "mass_balanced": count / (len(sn) + len(sd)),
        "fraction_num": fraction, "fraction_num_low95": center - half_width,
        "fraction_num_high95": center + half_width,
        "calibration_residual": residual, "residual_delta_se": residual_se,
        "residual_delta_low95": residual - _Z95 * residual_se,
        "residual_delta_high95": residual + _Z95 * residual_se,
        "residual_low95": center - half_width - mean_score,
        "residual_high95": center + half_width - mean_score,
    })


def _error_bins(exact_score, error, edges):
    groups, size = _groups(exact_score, edges), len(edges) - 1
    count = np.bincount(groups, minlength=size)
    mean = _divide(np.bincount(groups, weights=error, minlength=size), count)
    # A centered second pass avoids cancellation for almost constant residuals.
    centered_ss = np.bincount(groups, weights=(error - mean[groups]) ** 2, minlength=size)
    variance = _divide(centered_ss, count - 1)
    return pd.DataFrame({
        "bin": np.arange(size), "score_left": edges[:-1], "score_right": edges[1:],
        "score_mean": _divide(np.bincount(groups, weights=exact_score, minlength=size), count),
        "n_total": count, "mass_class": count / len(error),
        "mean_log_error": mean, "mean_log_error_se": np.sqrt(_divide(variance, count)),
        "rms_log_error": np.sqrt(_divide(np.bincount(groups, weights=error ** 2,
                                                    minlength=size), count)),
    })


def _mean_and_se(values):
    return float(np.mean(values)), float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _normalization(logr):
    scaled = np.exp(logr - np.max(logr))
    scaled_mean = np.mean(scaled)
    log_mean = float(logsumexp(logr) - np.log(len(logr)))
    # Keep the exact log mean even if its exponential cannot be represented.
    with np.errstate(over="ignore", under="ignore"):
        mean = float(np.exp(log_mean))
        relative_se = float(np.std(scaled, ddof=1) / scaled_mean / np.sqrt(len(logr)))
        se = mean * relative_se if relative_se else 0.0
    ess = float(np.sum(scaled) ** 2 / np.sum(scaled ** 2))
    return mean, se, log_mean, ess


def _summary(name, ln, ld, en, ed, members):
    loss_n, loss_d = np.logaddexp(0.0, -ln), np.logaddexp(0.0, ld)
    delta_n = loss_n - np.logaddexp(0.0, -en)
    delta_d = loss_d - np.logaddexp(0.0, ed)
    bce = 0.5 * (loss_n.mean() + loss_d.mean())
    bce_se = 0.5 * np.sqrt(np.var(loss_n, ddof=1) / len(ln)
                           + np.var(loss_d, ddof=1) / len(ld))
    excess = 0.5 * (delta_n.mean() + delta_d.mean())
    excess_se = 0.5 * np.sqrt(np.var(delta_n, ddof=1) / len(ln)
                              + np.var(delta_d, ddof=1) / len(ld))
    mean, se, log_mean, ess = _normalization(ld)
    row = dict(model=name, members=members, n_num=len(ln), n_den=len(ld),
               bce=float(bce), bce_se=float(bce_se), bce_excess_exact=float(excess),
               bce_excess_exact_se=float(excess_se),
               bce_excess_exact_low95=float(excess - _Z95 * excess_se),
               bce_excess_exact_high95=float(excess + _Z95 * excess_se),
               mean_ratio_den=mean, mean_ratio_den_se=se, log_mean_ratio_den=log_mean,
               ess_den=ess, ess_fraction_den=ess / len(ld))
    for label, learned, exact in (("num", ln, en), ("den", ld, ed)):
        error = learned - exact
        bias, bias_se = _mean_and_se(error)
        row.update({f"mean_log_error_{label}": bias, f"mean_log_error_se_{label}": bias_se,
                    f"rms_log_error_{label}": float(np.sqrt(np.mean(error ** 2))),
                    f"tail_score_gt_075_{label}": float(np.mean(expit(learned) > .75)),
                    f"tail_score_gt_090_{label}": float(np.mean(expit(learned) > .90))})
    return row


def diagnostic_report(member_logr_num, member_logr_den, exact_logr_num,
                      exact_logr_den, title="", score_edges=None):
    """Compare existing members, their arithmetic-ratio ensemble, and an oracle.

    Inputs are *log* target-to-reference ratios. Member arrays have shape
    ``(n_members, n_events)`` and exact arrays shape ``(n_events,)``. Numerator
    and denominator banks must be independent, balanced, and unused in fitting.
    Corresponding columns must refer to the same events for every model.

    Returns ``summary`` and ``bins`` DataFrames, ``figures`` as (name, Figure)
    pairs, and explanatory ``notes``. Empty bins remain in the numeric table.
    Wilson 95% reliability intervals are a binomial approximation for balanced
    class-stratified samples; they are not simultaneous confidence bands. Exact
    oracle curves use the same events and help expose finite-sample fluctuations.
    The paired BCE-excess SE uses within-class differences from the oracle, then
    combines the independent class means. It measures evaluation Monte Carlo
    uncertainty, not uncertainty from training or multiple-model comparisons.
    """
    mn, md, en, ed = _validate_inputs(member_logr_num, member_logr_den,
                                       exact_logr_num, exact_logr_den)
    members = len(mn)
    ensemble_n = logsumexp(mn, axis=0) - np.log(members)
    ensemble_d = logsumexp(md, axis=0) - np.log(members)
    models = [(f"Member {i + 1}", mn[i], md[i], 1) for i in range(members)]
    models += [("Ensemble", ensemble_n, ensemble_d, members), ("Exact ratio", en, ed, 0)]
    uniform_edges = _edges(np.linspace(0, 1, 26) if score_edges is None else score_edges)
    views = [("uniform" if score_edges is None else "custom", uniform_edges),
             ("equal_mass", _quantile_edges(np.r_[expit(ensemble_n), expit(ensemble_d)]))]
    exact_edges = _quantile_edges(np.r_[expit(en), expit(ed)])
    summary, tables, calibration, errors = [], [], {}, {}
    for name, ln, ld, count in models:
        summary.append(_summary(name, ln, ld, en, ed, count))
        for view, edges in views:
            table = _calibration_bins(expit(ln), expit(ld), edges)
            table.insert(0, "model", name)
            table.insert(0, "view", view)
            table.insert(0, "diagnostic", "calibration")
            calibration[view, name] = table
            tables.append(table)
        for label, learned, exact in (("num", ln, en), ("den", ld, ed)):
            table = _error_bins(expit(exact), learned - exact, exact_edges)
            table.insert(0, "class", label)
            table.insert(0, "model", name)
            table.insert(0, "view", "exact_equal_mass")
            table.insert(0, "diagnostic", "exact_log_ratio")
            errors[label, name] = table
            tables.append(table)

    figures = []
    # A notebook may have installed a global mplhep style with very large fonts.
    with plt.style.context("default"), plt.rc_context({"font.size": 10, "axes.titlesize": 11,
            "axes.labelsize": 10, "legend.fontsize": 8, "figure.titlesize": 13}):
        member_colors = (0, 2, 3, 4, 5, 6, 7, 8, 9)
        styles = {f"Member {i + 1}": dict(color=plt.get_cmap("tab10")(member_colors[i % len(member_colors)]),
                   alpha=.65, linewidth=1.0, linestyle="-") for i in range(members)}
        styles["Ensemble"] = dict(color="black", alpha=1.0, linewidth=2.0, linestyle="-")
        styles["Exact ratio"] = dict(color="darkorange", alpha=1.0, linewidth=2.0, linestyle="--")
        fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for column, (view, edges) in enumerate(views):
            for name, *_ in models:
                table = calibration[view, name]
                occupied = table.n_total > 0
                x = table.loc[occupied, "score_mean"].to_numpy()
                y = table.loc[occupied, "fraction_num"].to_numpy()
                residual = table.loc[occupied, "calibration_residual"].to_numpy()
                style = styles[name]
                ax[0, column].plot(x, y, marker=".", label=name, **style)
                ax[1, column].plot(x, residual, marker=".", **style)
                # Thin member intervals retain the actual precision without hiding
                # the ensemble/oracle comparison behind six opaque bands.
                for row, center in ((0, y), (1, residual)):
                    lower = table.loc[occupied, "fraction_num_low95"].to_numpy()
                    upper = table.loc[occupied, "fraction_num_high95"].to_numpy()
                    ax[row, column].errorbar(x, center,
                        yerr=np.array([np.maximum(y - lower, 0), np.maximum(upper - y, 0)]),
                        fmt="none", color=style["color"], alpha=.25 if name.startswith("Member") else .6,
                        elinewidth=.7, capsize=1)
            ax[0, column].plot([0, 1], [0, 1], color="gray", linewidth=1, zorder=0)
            ax[1, column].axhline(0, color="gray", linewidth=1)
            ax[0, column].set_title(f"{view.replace('_', ' ')} bins ({len(edges) - 1})")
            ax[0, column].set_ylim(-.03, 1.03)
            ax[0, column].set_ylabel("Observed numerator fraction")
            ax[1, column].set_ylabel("Observed fraction − mean score")
            for row in range(2):
                ax[row, column].set(xlim=(0, 1), xlabel="Mean predicted score in bin")
                ax[row, column].grid(alpha=.2)
        ax[0, 0].legend(loc="upper left")
        fig.suptitle(f"{title}\nReliability: approximate 95% intervals; same events for every curve")
        figures.append(("calibration", fig))

        fig, ax = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
        for row, (view, edges) in enumerate(views):
            centers = .5 * (edges[:-1] + edges[1:])
            for column, label in enumerate(("num", "den")):
                for name, *_ in models:
                    counts = calibration[view, name][f"n_{label}"].to_numpy()
                    ax[row, column].plot(centers, np.where(counts > 0, counts, np.nan),
                                         marker=".", label=name, **styles[name])
                ax[row, column].set(xlim=(0, 1), yscale="log", xlabel="Score-bin center",
                    ylabel="Events per bin", title=f"{view.replace('_', ' ')}: {'numerator' if label == 'num' else 'denominator'}")
                ax[row, column].grid(alpha=.2)
        ax[0, 0].legend()
        fig.suptitle(f"{title}\nClass occupancy (zero counts retained in the numeric table)")
        figures.append(("occupancy", fig))

        fig, ax = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
        for column, label in enumerate(("num", "den")):
            for name, *_ in models[:-1]:
                table = errors[label, name]
                occupied = table.n_total > 0
                x = table.loc[occupied, "score_mean"].to_numpy()
                mean = table.loc[occupied, "mean_log_error"].to_numpy()
                se = table.loc[occupied, "mean_log_error_se"].to_numpy()
                ax[0, column].errorbar(x, mean, yerr=se, marker=".", capsize=1,
                                       label=name, **styles[name])
                ax[1, column].plot(x, table.loc[occupied, "rms_log_error"], marker=".", **styles[name])
            ax[0, column].axhline(0, color="gray", linewidth=1)
            ax[0, column].set_title("Numerator events" if label == "num" else "Denominator events")
            ax[0, column].set_ylabel("Mean log-ratio error ± SE")
            ax[1, column].set_ylabel("RMS log-ratio error")
            ax[1, column].set_ylim(bottom=0)
            for row in range(2):
                ax[row, column].set(xlim=(0, 1), xlabel="Mean exact score in common bin")
                ax[row, column].grid(alpha=.2)
        ax[0, 0].legend()
        fig.suptitle(f"{title}\nLog-ratio error = log learned ratio − log exact ratio")
        figures.append(("exact_ratio", fig))

    return {"summary": pd.DataFrame(summary), "bins": pd.concat(tables, ignore_index=True),
            "figures": figures, "notes": [
                "Ensemble = arithmetic mean of ratios, evaluated stably in log space.",
                "Reliability bars are approximate pointwise Wilson 95% intervals for balanced samples.",
                "CSV residual_delta_se additionally includes fluctuations of the bin mean score via a class-stratified delta method; sparse-bin estimates may degenerate.",
                "Equal-mass reliability edges come from the pooled ensemble scores and are shared by all models.",
                "Exact-error edges come from the pooled exact scores and are shared by all models.",
                "BCE excess SE pairs each model with the exact oracle on the same events within each independent class.",
                "Finite-bank normalization and ESS estimates do not measure training uncertainty.",
                "No ranking, member selection, recalibration, or retraining is performed.",
            ]}
