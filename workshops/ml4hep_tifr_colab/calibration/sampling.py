"""Fresh physical toys and the disjoint training samples from notebook 01."""

import json
from pathlib import Path

import numpy as np

from utils_distributions import (
    background_components,
    signal_components,
    smearing_parameters,
)

FEATURES = [f"x{i}" for i in range(1, 6)]


def _components(process):
    return {"signal": signal_components, "background": background_components}[process]()


def _sample_latent(components, n, rng):
    fractions = np.array([component[0] for component in components])
    counts = rng.multinomial(n, fractions / fractions.sum())
    pieces = [
        rng.multivariate_normal(mean, covariance, size=count)
        for (_, mean, covariance), count in zip(components, counts)
    ]
    sample = np.concatenate(pieces)
    rng.shuffle(sample)
    return sample


def sample_process(process, n, rng, alpha=0.0):
    """Draw fresh events with the continuous physical detector scale.

    The response is scale*(1+0.1*alpha), with independent detector resolution.
    Nominal and +/-1 anchors are unchanged. At intermediate nuisance values
    this exact Gaussian-mixture simulator need not coincide with the fitted
    exp-poly interpolation; that interpolation approximation is diagnosed
    separately from learned-density error and intentional epsilon mixing.
    """
    if not -1.0 <= alpha <= 1.0:
        raise ValueError("The detector-scale model is defined on alpha in [-1, 1].")
    z = _sample_latent(_components(process), int(n), rng)
    scale, resolution = smearing_parameters()
    x = z * scale[None, :] * (1.0 + 0.1 * float(alpha))
    x += rng.normal(size=x.shape) * resolution[None, :]
    return x


def sample_experiment(mu, alpha, config, rng):
    """Independent Poisson counts, fresh unbinned events, and uniform auxiliary.

    The auxiliary law is the requested toy ensemble. The analysis likelihood
    still contains the Gaussian constraint exp[-(auxiliary-alpha)^2 / 2].
    """
    n_signal = rng.poisson(mu * config["signal_yield"])
    n_background = rng.poisson(config["background_yield"])
    x = np.concatenate([
        sample_process("signal", n_signal, rng, alpha),
        sample_process("background", n_background, rng, alpha),
    ])
    rng.shuffle(x)
    return {
        "x": x,
        "auxiliary": float(rng.uniform(-2.0, 2.0)),
        "mu": float(mu),
        "alpha": float(alpha),
    }


def process_density(x, process, alpha=0.0):
    """Exact density of the continuously scaled physical Gaussian mixture."""
    from scipy.stats import multivariate_normal

    if not -1.0 <= alpha <= 1.0:
        raise ValueError("The detector-scale model is defined on alpha in [-1, 1].")
    scale, resolution = smearing_parameters()
    response = scale * (1.0 + 0.1 * float(alpha))
    density = np.zeros(len(x))
    components = _components(process)
    total_fraction = sum(component[0] for component in components)
    for fraction, mean, covariance in components:
        reco_covariance = covariance * np.outer(response, response)
        reco_covariance += np.diag(resolution ** 2)
        density += (
            fraction / total_fraction
            * multivariate_normal.pdf(x, mean=mean * response, cov=reco_covariance)
        )
    return density


def read_sample(run, process, partition="ratio_train", anchor="nominal", n=None):
    """Read reco features from one disjoint row block, without loading other rows.

    The nominal/down/up parquets have the same row ordering. An evaluation
    latent event and both of its variations therefore stay out of all training.
    Systematic training uses the first variation_ratio_per_class ratio rows.
    """
    import pyarrow.parquet as pq

    run = Path(run)
    config = json.loads((run / "config.json").read_text())
    sizes = config["samples"]
    flow_size = sizes["flow_train_per_process"]
    ratio_size = sizes["nominal_ratio_per_class"]
    ranges = {
        "flow_train": (0, flow_size),
        "ratio_train": (flow_size, ratio_size),
        "eval": (flow_size + ratio_size, sizes["evaluation_per_process"]),
    }
    first, size = ranges[partition]
    n = size if n is None else int(n)
    if n > size:
        raise ValueError(f"{partition} has only {size:,} events per process.")
    suffix = {"nominal": "", "down": "_scale_down", "up": "_scale_up"}[anchor]
    parquet = pq.ParquetFile(run / "samples" / f"{process}{suffix}.parquet")
    pieces, offset = [], 0
    for batch in parquet.iter_batches(batch_size=100_000, columns=FEATURES):
        stop = offset + batch.num_rows
        if stop > first and offset < first + n:
            start_in_batch = max(0, first - offset)
            stop_in_batch = min(batch.num_rows, first + n - offset)
            values = batch.slice(start_in_batch, stop_in_batch - start_in_batch)
            pieces.append(values.to_pandas().to_numpy(dtype=np.float32))
        offset = stop
        if offset >= first + n:
            break
    return np.concatenate(pieces) if pieces else np.empty((0, len(FEATURES)), dtype=np.float32)
