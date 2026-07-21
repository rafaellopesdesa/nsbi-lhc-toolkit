"""Gaussian-mixture event generation for the ML4HEP-TIF NSBI tutorial.

Unlike the ``nsbi_atlas_workshop`` example -- where every feature is a single
unimodal Gaussian -- here each sample is a *mixture* of several correlated,
differently-oriented Gaussian components.  This gives

  * multi-modal marginals (bumps that a single Gaussian cannot capture), and
  * curved iso-density contours from mixing rotated components,

which make the joint density genuinely non-trivial to model -- a good stress
test for a Normalizing Flow density estimator (planned follow-up), while still
being cheap to sample and to reason about analytically.

We generate nominal ``background`` and ``signal`` samples plus an independent
nominal ``data`` draw for visualisation.  On request, detector-scale up/down
variations are derived in streamed batches.  They keep the latent density and
Gaussian resolution residual fixed while multiplying the response mean by 1.1
or 0.9.

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
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import background_components, signal_components, smearing_parameters

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
parser.add_argument("--systematics-batch-size", type=int, default=100_000)
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


# The mixture definitions (build_cov, BASE_*, background_components,
# signal_components) live in utils.py so the parameter-fitting notebook can
# import the exact same distributions to compute the "truth" density ratios.
def sample_mixture(components, n, rng):
    """Sample ``n`` events from a Gaussian mixture.

    ``components`` is a list of ``(fraction, mean, cov)``.  Fractions need not
    sum to 1 exactly; they are renormalised.  Component assignment is
    multinomial so the empirical fractions match the requested ones.
    """
    fracs = np.array([c[0] for c in components], dtype=float)
    fracs = fracs / fracs.sum()
    counts = rng.multinomial(n, fracs)
    chunks = []
    for (_, mean, cov), c in zip(components, counts):
        if c > 0:
            chunks.append(rng.multivariate_normal(mean, cov, size=c))
    x = np.concatenate(chunks, axis=0)
    rng.shuffle(x)  # avoid block-ordering by component
    return x

def add_reco_smearing(df, rng):
    """Add x1,...,x5 as independently smeared versions of z1,...,z5."""
    scale, resolution = smearing_parameters()

    y = df[features].to_numpy(dtype=float)

    x = rng.normal(
        loc=y * scale[None, :],
        scale=resolution[None, :],
        size=y.shape,
    )

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
    nominal_mean = z * scale[None, :]
    resolution_residual = x_nominal - nominal_mean

    varied = nominal_df.copy()
    varied[reco] = (
        mean_multiplier * nominal_mean + resolution_residual
    )
    return varied


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
    writers = {variation: None for variation in SCALE_VARIATIONS}

    try:
        parquet = pq.ParquetFile(nominal_path)
        for record_batch in parquet.iter_batches(batch_size=batch_size):
            nominal_batch = record_batch.to_pandas()
            for variation, multiplier in SCALE_VARIATIONS.items():
                varied_batch = scale_variation_dataframe(
                    nominal_batch, multiplier
                )
                table = pa.Table.from_pandas(varied_batch, preserve_index=False)
                if writers[variation] is None:
                    writers[variation] = pq.ParquetWriter(
                        temporary_targets[variation], table.schema
                    )
                writers[variation].write_table(table)
    finally:
        for writer in writers.values():
            if writer is not None:
                writer.close()

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

# --- Background ---
background = pd.DataFrame(
    sample_mixture(background_components(), n_bkg, rng), columns=features
)
background["fold"] = rng.integers(0, 2, size=n_bkg)
background["label"] = 0
background["weight"] = LAM_BKG / n_bkg  # total yield = LAM_BKG

# --- Signal ---
signal = pd.DataFrame(
    sample_mixture(signal_components(), n_sig, rng), columns=features
)
signal["fold"] = rng.integers(0, 2, size=n_sig)
signal["label"] = 1
signal["weight"] = LAM_SIG / n_sig  # total yield = LAM_SIG

# --- Pseudo-data: an independent draw from the background mixture ---
data = pd.DataFrame(
    sample_mixture(background_components(), n_bkg, rng), columns=features
)
data["fold"] = rng.integers(0, 2, size=n_bkg)
data["label"] = 0

background = add_reco_smearing(background, rng)
signal = add_reco_smearing(signal, rng)
data = add_reco_smearing(data, rng)

os.makedirs("dataframes", exist_ok=True)
background.to_parquet("dataframes/background.parquet", index=False)
signal.to_parquet("dataframes/signal.parquet", index=False)
data.to_parquet("dataframes/data.parquet", index=False)

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

os.makedirs("plots", exist_ok=True)


def feature_bins(feat):
    lo = min(background[feat].quantile(0.005), signal[feat].quantile(0.005))
    hi = max(background[feat].quantile(0.995), signal[feat].quantile(0.995))
    return np.linspace(lo, hi, 61)


# --- Plot 1: per-feature signal vs background (with pseudo-data points) ---
for feat in features:
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = feature_bins(feat)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    ax.errorbar(
        bin_centers,
        np.histogram(data[feat], bins=bins, density=True)[0],
        yerr=0,
        fmt="k.",
        capsize=3,
        ms=4,
        lw=1.2,
        label="Data",
        zorder=5,
    )
    ax.hist(
        background[feat],
        bins=bins,
        density=True,
        histtype="stepfilled",
        lw=2,
        color="black",
        alpha=0.15,
        label="Background",
    )
    ax.hist(
        signal[feat],
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

# --- Plot 2: 2D views exposing the multi-modal, correlated structure ---
# A LOG colour scale is essential here: on a linear scale the sharp background
# peak saturates the colour map and hides the broad, low-but-nonzero density
# floor (the shared BASE component). That floor is precisely what makes the
# support overlap total and the density ratios bounded, so we must show it.
from matplotlib.colors import LogNorm

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (fi, fj) in zip(axes, [(0, 1), (2, 3)]):
    ax.hist2d(
        background[features[fi]],
        background[features[fj]],
        bins=80,
        cmap="Greys",
        norm=LogNorm(),
    )
    ax.scatter(
        signal[features[fi]][:2000],
        signal[features[fj]][:2000],
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
    "Saved to dataframes/, plots to plots/"
)
