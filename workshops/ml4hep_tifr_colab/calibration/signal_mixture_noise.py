"""Known mixture-noise construction for signal NCE; no simulator oracle."""
import numpy as np
from signal_tail_diagnostics import gaussian_log_density, weight_summary
from signal_nce import log_density


def mixture_log_prob(x, reference, mean, std, broad_fraction=.2, width_scale=2., batch_size=8192):
    """log[(1-eps) q(x) + eps g_broad(x)], evaluated on either class."""
    if not 0 < broad_fraction < 1 or width_scale <= 0:
        raise ValueError('Require 0 < broad_fraction < 1 and positive width_scale.')
    result = np.empty(len(x), dtype=np.float64)
    for start in range(0, len(x), batch_size):
        block = x[start:start+batch_size]
        logq = reference.log_prob(block, batch_size=batch_size).astype(np.float64)
        logg = gaussian_log_density(block, np.asarray(mean, dtype=np.float64),
                                   width_scale*np.asarray(std, dtype=np.float64))
        result[start:start+len(block)] = np.logaddexp(np.log1p(-broad_fraction)+logq,
                                                    np.log(broad_fraction)+logg)
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite mixture noise density.')
    return result


def sample_mixture(n, reference, mean, std, seed, broad_fraction=.2, width_scale=2., batch_size=8192):
    """Stratified mixture draw, shuffled before the train/validation split.

    Require exact component counts so density weights match the sampling design.
    """
    if not 0 < broad_fraction < 1 or width_scale <= 0 or n < 1:
        raise ValueError('Invalid mixture configuration.')
    nb = int(round(n*broad_fraction))
    if not np.isclose(nb, n*broad_fraction, rtol=0, atol=1e-8):
        raise ValueError('Choose n so n*broad_fraction is an integer.')
    nq = n-nb
    x = np.empty((n, len(mean)), dtype=np.float32)
    x[:nq] = reference.sample(nq, seed, batch_size=batch_size)
    rng = np.random.default_rng(seed+1)
    x[nq:] = rng.normal(size=(nb, len(mean))) * (width_scale*np.asarray(std)) + mean
    rng.shuffle(x, axis=0)
    return x


def broad_monitor(mean, std, n=100_000, seed=7202812, width_scale=2., every=5):
    """Fixed independent broad bank, monitoring only; never used in gradients."""
    rng = np.random.default_rng(seed)
    x = (rng.normal(size=(n, len(mean))) * (width_scale*np.asarray(std)) + mean).astype('float32')
    logg = gaussian_log_density(x, np.asarray(mean, dtype=np.float64),
                               width_scale*np.asarray(std, dtype=np.float64))
    def evaluate(model, epoch):
        if epoch != 1 and epoch % every:
            return {}
        values = weight_summary(log_density(model, x)-logg)
        print(f"Broad monitor epoch {epoch}: integral={values['mean']:.6g}, "
              f"ESS={values['ess']:.1f}, max share={values['largest_weight_fraction']:.3g}", flush=True)
        return {'broad_'+key: value for key, value in values.items()}
    return evaluate
