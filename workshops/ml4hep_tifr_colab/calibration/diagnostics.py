"""Independent interpolation and population checks for notebook 02."""

import numpy as np

from interpolation import (
    derivative_polynomial, evaluate_polynomial, numpy_coefficients,
    normalization_coefficients,
)
from model import fit, with_model_epsilon
from sampling import sample_process


def interpolation_diagnostics(anchors, config):
    """Return two figures and tables; this sample must be independent of Z."""
    import matplotlib.pyplot as plt
    import pandas as pd

    anchors = np.asarray(anchors, dtype=np.float64)
    coefficients, repair = numpy_coefficients(anchors, return_diagnostics=True)
    paper_coefficients = numpy_coefficients(anchors, ensure_positive=False)
    normalizer = normalization_coefficients(config)
    grid = np.linspace(-1., 1., 81)
    rows = []
    for alpha in grid:
        raw = evaluate_polynomial(coefficients, alpha)
        original = evaluate_polynomial(paper_coefficients, alpha)
        z = evaluate_polynomial(normalizer, alpha)
        shaped = raw / z
        for process, label in enumerate(('signal', 'background')):
            r = shaped[:, process]
            rows.append(dict(
                process=label, alpha=alpha, mean=r.mean(),
                standard_error=r.std(ddof=1) / np.sqrt(len(r)),
                raw_integral=z[process], minimum_ratio=r.min(),
                original_nonpositive_fraction=np.mean(original[:, process] <= 0),
                added_mass_fraction=np.mean(raw[:, process] - original[:, process]) / z[process],
                effective_sample_size=r.sum()**2 / np.square(r).sum(),
            ))
    table = pd.DataFrame(rows)
    repair_table = pd.DataFrame([
        dict(process=label, n_events=len(anchors),
             repaired_rows=int(repair['repair_mask'][:, c].sum()),
             repaired_fraction=float(repair['repair_mask'][:, c].mean()),
             maximum_added_mass_fraction=float(table.loc[table.process == label, 'added_mass_fraction'].max()))
        for c, label in enumerate(('signal', 'background'))
    ])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for label in ('signal', 'background'):
        values = table[table.process == label]
        line, = axes[0, 0].plot(values.alpha, values['mean'] - 1, label=label)
        axes[0, 0].fill_between(values.alpha,
                               values['mean'] - 1 - 2 * values.standard_error,
                               values['mean'] - 1 + 2 * values.standard_error,
                               color=line.get_color(), alpha=.15)
        axes[0, 1].plot(values.alpha, values.raw_integral - 1, label=label)
        axes[1, 0].plot(values.alpha, values.minimum_ratio, label=label)
        axes[1, 1].plot(values.alpha, values.added_mass_fraction, label=label)
    axes[0, 0].axhline(0, color='k', ls=':')
    axes[0, 0].set(ylabel='Independent integral minus one', title='Shading: two check-bank standard errors')
    axes[0, 1].axhline(0, color='k', ls=':')
    axes[0, 1].set(ylabel=r'$Z_c(\alpha)-1$', title='Normalization required between anchors')
    axes[1, 0].set(yscale='log', ylabel='Minimum normalized ratio on check bank', title='Positivity across nuisance values')
    axes[1, 1].set(ylabel='Added mass / normalizer', title='Effect of the sparse positivity correction')
    for ax in axes.flat:
        ax.set_xlabel(r'$\alpha$')
        ax.legend()
    fig.tight_layout()

    # Select a visible nominal kink from representative (non-extreme) events.
    # Selection is only for illustration, never normalization or training.
    nominal = anchors[..., 1]
    kink = np.abs(anchors[..., 2] + anchors[..., 0] - 2 * nominal) / nominal
    eligible = (nominal > .05) & (nominal < 20)
    selections = [int(np.argmax(np.where(eligible[:, c], kink[:, c], -np.inf))) for c in range(2)]
    zoom = np.linspace(-.08, .08, 161)
    smooth_fig, smooth_axes = plt.subplots(2, 2, figsize=(11, 7))
    for c, (label, index) in enumerate(zip(('signal', 'background'), selections)):
        event = coefficients[index:index + 1]
        raw = np.array([evaluate_polynomial(event, a)[0, c] for a in zoom])
        slope = np.array([derivative_polynomial(event, a)[0, c] for a in zoom])
        z = np.array([evaluate_polynomial(normalizer, a)[c] for a in zoom])
        dz = np.array([derivative_polynomial(normalizer, a)[c] for a in zoom])
        down, center, up = anchors[index, c]
        linear = (1 - abs(zoom)) * center + np.maximum(zoom, 0) * up + np.maximum(-zoom, 0) * down
        linear_score = np.where(zoom < 0, center - down, up - center) / linear
        linear_score[np.isclose(zoom, 0, atol=1e-15)] = np.nan
        smooth_axes[c, 0].plot(zoom, raw / z / center, label='Normalized exp-poly')
        smooth_axes[c, 0].plot(zoom, linear / center, ls='--', label='Previous linear morph')
        smooth_axes[c, 0].set(ylabel='Ratio / nominal ratio', title=f'{label}: illustrative event')
        smooth_axes[c, 1].plot(zoom, slope / raw - dz / z, label='Normalized exp-poly')
        smooth_axes[c, 1].plot(zoom, linear_score, ls='--', label='Previous linear morph')
        smooth_axes[c, 1].set(ylabel=r'$\partial_\alpha\log R_c(x;\alpha)$', title='Nominal derivative')
    for ax in smooth_axes.flat:
        ax.set_xlabel(r'$\alpha$')
        ax.axvline(0, color='k', ls=':', alpha=.4)
        ax.legend(fontsize=8)
    smooth_fig.tight_layout()
    return fig, smooth_fig, table, repair_table


def physical_population_closure(hybrid, config, *, n_per_process=100_000,
                                n_banks=3, mu_grid=(.5, 1., 1.5), seed=13901):
    """Direct epsilon=0 fits on repeated fresh nominal simulator banks.

    This measures learned-model plus Monte Carlo error, without deliberately
    contaminating the signal. It does not require or train an inference NN.
    """
    import pandas as pd

    good_config = with_model_epsilon(config, 0.)
    rows = []
    for bank in range(n_banks):
        rng = np.random.default_rng(seed + bank)
        anchors = np.concatenate([
            hybrid.anchors(sample_process(process, n_per_process, rng, alpha=0.))
            for process in ('signal', 'background')
        ])
        for mu in mu_grid:
            weights = np.r_[np.full(n_per_process, mu * config['signal_yield'] / n_per_process),
                            np.full(n_per_process, config['background_yield'] / n_per_process)]
            result = fit(anchors, 0., good_config, weights=weights)
            rows.append(dict(bank=bank, mu=float(mu), n_per_process=n_per_process,
                             nu_bias=result['nu'] - mu, alpha_bias=result['alpha'], **result))
        print(f'Epsilon=0 physical closure: bank {bank + 1}/{n_banks}')
    return pd.DataFrame(rows)


def plot_population_closure(table):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for bank, rows in table.groupby('bank'):
        for ax, column in zip(axes, ('nu_bias', 'alpha_bias')):
            ax.plot(rows.mu, rows[column], 'o-', alpha=.7, label=f'Independent bank {bank + 1}')
    for ax, ylabel in zip(axes, (r'$\hat\nu-\mu$', r'$\hat\alpha$')):
        ax.axhline(0, color='k', ls=':')
        ax.set(xlabel=r'Physical $\mu$', ylabel=ylabel)
        ax.legend(fontsize=8)
    fig.suptitle(r'Physical population closure at $\epsilon=0$; bank spread is Monte Carlo variation')
    fig.tight_layout()
    return fig
