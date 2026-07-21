"""Gaussian-mixture event generation for the ML4HEP-TIF NSBI tutorial.

Unlike the ``nsbi_atlas_workshop`` example -- where every feature is a single
unimodal Gaussian -- here each sample is a *mixture* of several correlated,
differently-oriented Gaussian components.  This gives

  * multi-modal marginals (bumps that a single Gaussian cannot capture), and
  * curved iso-density contours from mixing rotated components,

which make the joint density genuinely non-trivial to model -- a good stress
test for a Normalizing Flow density estimator (planned follow-up), while still
being cheap to sample and to reason about analytically.

We generate nominal ``background`` and ``signal`` samples plus, unless disabled,
an independent nominal ``data`` draw for visualisation.  Nominal samples and
detector-scale up/down variations are both written in bounded batches, so peak
memory is independent of the requested event count.  The variations keep the
latent density and Gaussian resolution residual fixed while multiplying the
response mean by 1.1 or 0.9.

Numerical-stability design
---------------------------
Density-ratio estimation is only well behaved when the reference (denominator)
has support everywhere the numerator does.  To guarantee bounded ratios when
we train one sample against another, **every** sample here contains a common,
broad ``BASE`` component (see ``utils.BASE_MEAN`` / ``utils.BASE_SIGMA``)
carrying a non-negligible mixing fraction.  Because that component alone already
covers the whole region of interest with p(x) > 0, no phase-space pocket exists
where one density vanishes while another does not -- so p_a(x)/p_b(x) never
blows up.
"""

import argparse
import gc
import os
from pathlib import Path

import numpy as np
import pandas as pd

from utils_distributions import (
    background_components,
    signal_components,
    smearing_parameters,
)

parser = argparse.ArgumentParser()
parser.add_argument("--n_bkg", type=int, default=1_000_000)
parser.add_argument("--n_sig", type=int, default=100_000)
parser.add_argument(
    "--with-systematics",
    action="store_true",
    help="Also create the scale-up/down detector-response parquets.",
)
parser.add_argument(
    "--systematics-only",
    action="store_true",
    help=(
        "Create the scale-up/down parquet files from existing nominal "
        "parquets without regenerating the latent events."
    ),
)
parser.add_argument(
    "--generation-batch-size",
    type=int,
    default=100_000,
    help="Maximum number of nominal events held in memory at once.",
)
parser.add_argument("--systematics-batch-size", type=int, default=100_000)
parser.add_argument(
    "--plot-sample-size",
    type=int,
    default=200_000,
    help="Maximum events retained per sample for diagnostic plots.",
)
parser.add_argument(
    "--skip-data",
    action="store_true",
    help="Do not generate the independent pseudo-data sample.",
)
parser.add_argument(
    "--skip-plots",
    action="store_true",
    help="Do not create the generator-level diagnostic plots.",
)
args = parser.parse_args()

n_bkg = args.n_bkg
n_sig = args.n_sig

features = ["z1", "z2", "z3", "z4", "z5"]
reco = ["x1", "x2", "x3", "x4", "x5"]

# Total expected yields (define the signal strength of the measurement).
LAM_BKG = 1_000_000.0
LAM_SIG = 1_100.0

SIGNAL_COLOR = "xkcd:hot pink"

SCALE_VARIATIONS = {
    "scale_up": 1.1,
    "scale_down": 0.9,
}


# The mixture definitions live in the lightweight ``utils_distributions``
# module and are re-exported by ``utils.py`` for the parameter-fitting notebooks.
def sample_mixture(components, n, rng):
    """Sample ``n`` events from a Gaussian mixture.

    ``components`` is a list of ``(fraction, mean, cov)``.  Fractions need not
    sum to 1 exactly; they are renormalised.  Component assignment is
    multinomial so the empirical fractions match the requested ones.
    """
    fracs = np.array([c[0] for c in components], dtype=float)
    fracs = fracs / fracs.sum()
    counts = rng.multinomial(n, fracs)
    n_dimensions = len(np.asarray(components[0][1]))
    sample = np.empty((n, n_dimensions), dtype=float)
    offset = 0
    for (_, mean, cov), c in zip(components, counts):
        if c > 0:
            sample[offset : offset + c] = rng.multivariate_normal(
                mean, cov, size=c
            )
            offset += c
    rng.shuffle(sample)  # avoid block-ordering by component
    return sample


def add_reco_smearing(df, rng):
    """Add x1,...,x5 as independently smeared versions of z1,...,z5."""
    scale, resolution = smearing_parameters()

    y = df[features].to_numpy(dtype=float, copy=False)
    x = rng.normal(loc=0.0, scale=resolution[None, :], size=y.shape)
    for index, response_scale in enumerate(scale):
        x[:, index] += response_scale * y[:, index]

    df[reco] = x
    return df


def scale_variation_dataframe(nominal_df, mean_multiplier):
    """Return a detector-scale variation of an existing nominal sample.

    The latent variables and the stochastic resolution residual are preserved
    event by event.  Only the centre of the Gaussian response is changed from

    ``z * scale`` to ``z * scale * mean_multiplier``.

    Thus ``mean_multiplier=1.1`` and ``0.9`` implement the requested
    ``alpha=+1`` and ``alpha=-1`` templates without changing ``p(z)`` or the
    detector resolution.
    """
    mean_multiplier = float(mean_multiplier)
    if mean_multiplier <= 0.0:
        raise ValueError("mean_multiplier must be positive.")

    missing = [column for column in [*features, *reco] if column not in nominal_df]
    if missing:
        raise KeyError(f"Nominal dataframe is missing columns: {missing}")

    scale, _ = smearing_parameters()
    scale = np.asarray(scale, dtype=float)
    z = nominal_df[features].to_numpy(dtype=float)
    x_nominal = nominal_df[reco].to_numpy(dtype=float)
    varied = nominal_df.copy()
    varied[reco] = x_nominal + (mean_multiplier - 1.0) * (
        z * scale[None, :]
    )
    return varied


def nominal_batch_dataframe(
    components,
    n_events,
    rng,
    *,
    label,
    expected_yield=None,
    total_events=None,
):
    """Generate one nominal batch with the established parquet schema."""
    n_events = int(n_events)
    batch = pd.DataFrame(
        sample_mixture(components, n_events, rng), columns=features
    )
    batch["fold"] = rng.integers(0, 2, size=n_events)
    batch["label"] = int(label)
    if expected_yield is not None:
        if total_events is None or int(total_events) < 1:
            raise ValueError("total_events must be positive for weighted samples.")
        batch["weight"] = float(expected_yield) / int(total_events)
    return add_reco_smearing(batch, rng)


def write_nominal_sample(
    output_path,
    components,
    n_events,
    rng,
    *,
    label,
    expected_yield=None,
    batch_size=100_000,
    plot_sample_size=200_000,
):
    """Generate and write one nominal sample with bounded peak memory.

    Only a small, bounded latent-feature sample is retained for diagnostic
    plots.  The complete event table is transferred directly from each batch
    to a Parquet row group and is never assembled as a single dataframe.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.unlink(missing_ok=True)

    n_events = int(n_events)
    batch_size = int(batch_size)
    plot_sample_size = max(0, int(plot_sample_size))
    if n_events < 1:
        raise ValueError("n_events must be positive.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    retained = []
    n_retained = 0
    writer = None
    sample_name = output_path.stem
    report_interval = max(
        batch_size,
        ((n_events + 10 * batch_size - 1) // (10 * batch_size)) * batch_size,
    )
    next_report = report_interval

    print(
        f"Generating {sample_name}: {n_events:,} events "
        f"in batches of at most {batch_size:,}..."
    )
    try:
        for start in range(0, n_events, batch_size):
            current_size = min(batch_size, n_events - start)
            batch = nominal_batch_dataframe(
                components,
                current_size,
                rng,
                label=label,
                expected_yield=expected_yield,
                total_events=n_events,
            )

            n_to_retain = min(plot_sample_size - n_retained, current_size)
            if n_to_retain > 0:
                retained.append(batch.iloc[:n_to_retain][features].copy())
                n_retained += n_to_retain

            table = pa.Table.from_pandas(batch, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary_path, table.schema, use_dictionary=False
                )
            writer.write_table(table)
            del table, batch

            batch_index = start // batch_size + 1
            if batch_index % 10 == 0:
                gc.collect()
                pa.default_memory_pool().release_unused()

            n_written = start + current_size
            if n_written >= next_report or n_written == n_events:
                print(
                    f"  {sample_name}: {n_written:,}/{n_events:,} "
                    f"({100.0 * n_written / n_events:.0f}%)"
                )
                while next_report <= n_written:
                    next_report += report_interval
    finally:
        if writer is not None:
            writer.close()

    os.replace(temporary_path, output_path)
    if retained:
        return pd.concat(retained, ignore_index=True)
    return pd.DataFrame(columns=features)


def write_scale_variations(
    nominal_path,
    *,
    output_dir=None,
    sample_name=None,
    batch_size=100_000,
):
    """Stream one nominal parquet and write its scale-up/down variations.

    Streaming avoids loading a complete generated sample merely to modify the
    detector response.  The output files are named
    ``<sample>_scale_up.parquet`` and ``<sample>_scale_down.parquet``.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    nominal_path = Path(nominal_path)
    if not nominal_path.exists():
        raise FileNotFoundError(nominal_path)
    output_dir = nominal_path.parent if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_name = nominal_path.stem if sample_name is None else str(sample_name)
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    targets = {
        variation: output_dir / f"{sample_name}_{variation}.parquet"
        for variation in SCALE_VARIATIONS
    }
    temporary_targets = {
        variation: target.with_suffix(target.suffix + ".tmp")
        for variation, target in targets.items()
    }
    for temporary_target in temporary_targets.values():
        temporary_target.unlink(missing_ok=True)
    writers = {variation: None for variation in SCALE_VARIATIONS}

    try:
        # ``pre_buffer`` defaults to True in current PyArrow releases.  Leaving
        # it enabled makes an otherwise batched pass retain the complete input
        # parquet in Arrow memory, which is fatal for the 100M-event sample.
        parquet = pq.ParquetFile(nominal_path, pre_buffer=False)
        n_events = parquet.metadata.num_rows
        report_interval = max(
            batch_size,
            ((n_events + 10 * batch_size - 1) // (10 * batch_size))
            * batch_size,
        )
        next_report = report_interval
        n_written = 0
        print(
            f"Writing {sample_name} scale variations for {n_events:,} events "
            f"in batches of at most {batch_size:,}..."
        )
        for batch_index, record_batch in enumerate(
            parquet.iter_batches(batch_size=batch_size, use_threads=False),
            start=1,
        ):
            current_size = record_batch.num_rows
            nominal_batch = record_batch.to_pandas(
                self_destruct=True, use_threads=False
            )
            for variation, multiplier in SCALE_VARIATIONS.items():
                varied_batch = scale_variation_dataframe(
                    nominal_batch, multiplier
                )
                table = pa.Table.from_pandas(varied_batch, preserve_index=False)
                if writers[variation] is None:
                    writers[variation] = pq.ParquetWriter(
                        temporary_targets[variation],
                        table.schema,
                        use_dictionary=False,
                    )
                writers[variation].write_table(table)
                del table, varied_batch
            del nominal_batch, record_batch

            if batch_index % 10 == 0:
                gc.collect()
                pa.default_memory_pool().release_unused()

            n_written += current_size
            if n_written >= next_report or n_written == n_events:
                print(
                    f"  {sample_name} variations: {n_written:,}/{n_events:,} "
                    f"({100.0 * n_written / n_events:.0f}%)"
                )
                while next_report <= n_written:
                    next_report += report_interval
    finally:
        for writer in writers.values():
            if writer is not None:
                writer.close()
        pa.default_memory_pool().release_unused()

    for variation, target in targets.items():
        os.replace(temporary_targets[variation], target)
    return targets


if args.systematics_only:
    for sample_name in ["background", "signal"]:
        written = write_scale_variations(
            Path("dataframes") / f"{sample_name}.parquet",
            sample_name=sample_name,
            batch_size=args.systematics_batch_size,
        )
        print(
            f"Created {sample_name} scale variations: "
            + ", ".join(str(path) for path in written.values())
        )
    raise SystemExit(0)

rng = np.random.default_rng(42)
generation_batch_size = int(args.generation_batch_size)
plot_sample_size = 0 if args.skip_plots else int(args.plot_sample_size)
if generation_batch_size < 1:
    raise ValueError("--generation-batch-size must be positive.")
if not args.skip_plots and plot_sample_size < 1:
    raise ValueError("--plot-sample-size must be positive unless plots are skipped.")

# Write each nominal sample before generating the next one.  At Exercise 8
# scale this replaces three simultaneously resident 100M/20M/100M-event
# dataframes with one bounded batch plus a small plotting sample.
background_plot = write_nominal_sample(
    "dataframes/background.parquet",
    background_components(),
    n_bkg,
    rng,
    label=0,
    expected_yield=LAM_BKG,
    batch_size=generation_batch_size,
    plot_sample_size=plot_sample_size,
)
signal_plot = write_nominal_sample(
    "dataframes/signal.parquet",
    signal_components(),
    n_sig,
    rng,
    label=1,
    expected_yield=LAM_SIG,
    batch_size=generation_batch_size,
    plot_sample_size=plot_sample_size,
)

data_plot = None
if not args.skip_data:
    data_plot = write_nominal_sample(
        "dataframes/data.parquet",
        background_components(),
        n_bkg,
        rng,
        label=0,
        batch_size=generation_batch_size,
        plot_sample_size=plot_sample_size,
    )

if args.with_systematics:
    # Build the detector-response variations from the nominal parquets in
    # bounded batches. This preserves z and the random resolution residual
    # event by event while avoiding four additional full-size dataframes in
    # memory. Other exercises retain the original nominal-only default.
    write_scale_variations(
        "dataframes/background.parquet",
        sample_name="background",
        batch_size=args.systematics_batch_size,
    )
    write_scale_variations(
        "dataframes/signal.parquet",
        sample_name="signal",
        batch_size=args.systematics_batch_size,
    )

if not args.skip_plots:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    os.makedirs("plots", exist_ok=True)

    def feature_bins(feat):
        lo = min(
            background_plot[feat].quantile(0.005),
            signal_plot[feat].quantile(0.005),
        )
        hi = max(
            background_plot[feat].quantile(0.995),
            signal_plot[feat].quantile(0.995),
        )
        return np.linspace(lo, hi, 61)

    # Plot bounded representative samples rather than reopening full parquets.
    for feat in features:
        fig, ax = plt.subplots(figsize=(7, 5))
        bins = feature_bins(feat)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        if data_plot is not None:
            ax.errorbar(
                bin_centers,
                np.histogram(data_plot[feat], bins=bins, density=True)[0],
                yerr=0,
                fmt="k.",
                capsize=3,
                ms=4,
                lw=1.2,
                label="Data",
                zorder=5,
            )
        ax.hist(
            background_plot[feat],
            bins=bins,
            density=True,
            histtype="stepfilled",
            lw=2,
            color="black",
            alpha=0.15,
            label="Background",
        )
        ax.hist(
            signal_plot[feat],
            bins=bins,
            density=True,
            histtype="step",
            lw=2,
            color=SIGNAL_COLOR,
            label="Signal",
        )
        ax.set_xlabel(feat, loc="right")
        ax.set_ylabel("Density", loc="top")
        ax.legend(fontsize=8)
        fig.savefig(f"plots/{feat}.pdf", bbox_inches="tight")
        plt.close(fig)

    # A log colour scale exposes the shared broad support component.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (fi, fj) in zip(axes, [(0, 1), (2, 3)]):
        ax.hist2d(
            background_plot[features[fi]],
            background_plot[features[fj]],
            bins=80,
            cmap="Greys",
            norm=LogNorm(),
        )
        ax.scatter(
            signal_plot[features[fi]].iloc[:2000],
            signal_plot[features[fj]].iloc[:2000],
            s=3,
            color=SIGNAL_COLOR,
            alpha=0.4,
            label="signal",
        )
        ax.set_xlabel(features[fi])
        ax.set_ylabel(features[fj])
        ax.legend(fontsize=8)
    fig.suptitle("Background density on a LOG scale (grey) with signal overlaid")
    fig.savefig("plots/2d_structure.pdf", bbox_inches="tight")
    plt.close(fig)

variation_message = (
    ", including scale-up/down detector variations"
    if args.with_systematics
    else ""
)
print(
    f"Done. Generated background ({n_bkg:,}, yield {LAM_BKG:g}) and "
    f"signal ({n_sig:,}, yield {LAM_SIG:g}){variation_message}. "
    + ("Pseudo-data generation was skipped. " if args.skip_data else "")
    + "Saved to dataframes/"
    + (
        "; diagnostic plots were skipped."
        if args.skip_plots
        else ", plots to plots/."
    )
)
