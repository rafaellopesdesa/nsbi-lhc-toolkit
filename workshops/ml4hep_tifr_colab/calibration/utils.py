"""Fresh hNDE helpers for the calibration notebooks.

NSBI trains binary classifiers with a logit output. Equal class weights imply
ratio = exp(logit). Six complete process/reference anchors are normalized once
on an independent reference integration sample, then linearly interpolated.
"""
from pathlib import Path
from contextlib import contextmanager
import gc
import hashlib
import json
import os
import numpy as np
import pandas as pd

from model import FEATURES, bad_anchors, intensity
from flow_reference import ReferenceFlow

RATIO_DEFAULTS = dict(hidden_layers=4, neurons=1024, number_of_epochs=50,
                      batch_size=4096, learning_rate=1e-3, scalerType='MinMax',
                      holdout_split=0.25, validation_split=0.20,
                      callback_patience=10, num_workers=0, verbose=1,
                      calibration=False)
RATIO_NAMES = ['signal', 'background', 'signal_down', 'signal_up',
               'background_down', 'background_up']


@contextmanager
def working_directory(path):
    """Keep upstream Lightning logs and temporary checkpoints inside this run."""
    previous = Path.cwd()
    Path(path).mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def train_ratio(numerator, denominator, directory, members=1, seed=13001,
                settings=None, reuse=True):
    """Train an arithmetic ensemble with the existing NSBI training package."""
    import torch
    from nsbi_common_utils.training import density_ratio_trainer

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if reuse and (directory / 'ensemble.json').exists():
        return RatioEnsemble.load(directory)
    x = np.concatenate([numerator, denominator]).astype(np.float32)
    labels = np.concatenate([np.ones(len(numerator)), np.zeros(len(denominator))])
    weights = np.concatenate([np.full(len(numerator), 1 / len(numerator)),
                              np.full(len(denominator), 1 / len(denominator))])
    dataframe = pd.DataFrame(x, columns=FEATURES)
    train_settings = RATIO_DEFAULTS | (settings or {})
    for member in range(members):
        np.random.seed(seed + member)
        torch.manual_seed(seed + member)
        trainer = density_ratio_trainer(
            dataset=dataframe, weights=weights, training_labels=labels,
            features=FEATURES, features_scaling=FEATURES,
            sample_name=[directory.name, 'denominator'], output_name='',
            path_to_figures=str(directory / f'plots_{member}') + '/',
            path_to_models=str(directory) + '/', use_log_loss=True)
        # Common external holdout; member initialization/minibatch RNG differ.
        with working_directory(directory):
            trainer.train(**train_settings, rnd_seed=seed, ensemble_index=member,
                          load_trained_models=reuse)
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    metadata = dict(members=members, output='log_density_ratio', seed=seed,
                    n_numerator=len(numerator), n_denominator=len(denominator),
                    settings=train_settings)
    (directory / 'ensemble.json').write_text(json.dumps(metadata, indent=2))
    return RatioEnsemble.load(directory)


class RatioEnsemble:
    def __init__(self, members):
        self.members = members

    @classmethod
    def load(cls, directory, device=None):
        import joblib
        import onnxruntime as ort
        import torch
        directory = Path(directory)
        metadata = json.loads((directory / 'ensemble.json').read_text())
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        use_cuda = (device is None and torch.cuda.is_available()) or str(device).startswith('cuda')
        providers = ['CPUExecutionProvider']
        if use_cuda and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
        members = []
        for i in range(metadata['members']):
            scaler = joblib.load(directory / f'model_scaler{i}.bin')
            session = ort.InferenceSession(str(directory / f'model{i}.onnx'),
                                           sess_options=options,
                                           providers=providers)
            members.append((scaler, session))
        return cls(members)

    def __call__(self, x, batch_size=8192):
        chunks = []
        for start in range(0, len(x), batch_size):
            frame = pd.DataFrame(x[start:start + batch_size], columns=FEATURES)
            predictions = []
            for scaler, session in self.members:
                inputs = np.asarray(scaler.transform(frame), dtype=np.float32)
                name = session.get_inputs()[0].name
                logits = session.run(None, {name: inputs})[0].reshape(-1)
                predictions.append(np.exp(logits.astype(np.float64)))
            chunks.append(np.mean(predictions, axis=0))
        return np.concatenate(chunks) if chunks else np.empty(0)


class HybridModel:
    def __init__(self, run, flow, ratios, normalization=None, model_id=None):
        self.run, self.flow, self.ratios = Path(run), flow, ratios
        self.normalization = normalization
        self.model_id = model_id
        self._integration_sampler = None

    @classmethod
    def load(cls, run, device='cpu'):
        root = Path(run) / 'hybrid'
        manifest = json.loads((root / 'manifest.json').read_text())
        return cls(run, ReferenceFlow.load(root / 'reference.pt', device),
                   {name: RatioEnsemble.load(root / 'ratios' / name, device=device)
                    for name in RATIO_NAMES},
                   np.load(root / 'anchor_normalization.npy'), manifest['model_id'])

    def raw_anchors(self, x):
        output = np.empty((len(x), 2, 3), dtype=np.float64)
        for c, process in enumerate(['signal', 'background']):
            nominal = self.ratios[process](x)
            output[:, c, 0] = nominal * self.ratios[process + '_down'](x)
            output[:, c, 1] = nominal
            output[:, c, 2] = nominal * self.ratios[process + '_up'](x)
        return output

    def anchors(self, x):
        return self.raw_anchors(x) / self.normalization

    def sample_reference(self, n, seed):
        return self.flow.sample(int(n), int(seed))

    def reference_anchors(self):
        return np.load(self.run / 'hybrid' / 'reference_anchors.npy', mmap_mode='r')

    def make_sampler(self, n, seed):
        """Independent finite proposal bank; reuse across toys only explicitly."""
        x = self.sample_reference(n, seed)
        return ReferenceSampler(x, self.anchors(x), method='independent reference bank')

    def _sampler(self, n, rng, proposal_size):
        if proposal_size == 0:
            if self._integration_sampler is None:
                x = np.load(self.run / 'hybrid' / 'reference_x.npy', mmap_mode='r')
                self._integration_sampler = ReferenceSampler(
                    x, self.reference_anchors(),
                    method='shared integration bank (training approximation)')
            return self._integration_sampler
        size = max(100_000, 20 * n) if proposal_size is None else int(proposal_size)
        result = self.make_sampler(size, int(rng.integers(2**31)))
        result.method = 'fresh importance-resampling proposal'
        return result

    def sample_experiment(self, mu, alpha, config, rng, epsilon=0., proposal_size=None):
        n = rng.poisson(mu * config['signal_yield'] + config['background_yield'])
        return self._sampler(n, rng, proposal_size).sample_experiment(
            mu, alpha, config, rng, epsilon, n=n)

    def sample_anchor_experiment(self, mu, alpha, config, rng, epsilon=0., proposal_size=0):
        result = self.sample_experiment(mu, alpha, config, rng, epsilon, proposal_size)
        result.pop('x')
        return result


def save_hybrid(hybrid, n_reference=5_000_000, seed=13031, batch_size=100_000):
    """Normalize complete anchors; store the independent integration sample."""
    root = hybrid.run / 'hybrid'
    root.mkdir(parents=True, exist_ok=True)
    x = hybrid.sample_reference(n_reference, seed)
    np.save(root / 'reference_x.npy', x)
    # Avoid a second full six-ratio array in memory for a five-million bank.
    anchors = np.lib.format.open_memmap(root / 'reference_anchors.npy', mode='w+',
                                        dtype=np.float64, shape=(n_reference, 2, 3))
    for start in range(0, n_reference, batch_size):
        anchors[start:start + batch_size] = hybrid.raw_anchors(x[start:start + batch_size])
    normalization = np.mean(anchors, axis=0)
    for start in range(0, n_reference, batch_size):
        anchors[start:start + batch_size] /= normalization
    anchors.flush()
    np.save(root / 'anchor_normalization.npy', normalization)
    hybrid.normalization = normalization
    paths = [root / 'reference.pt', root / 'anchor_normalization.npy',
             root / 'reference_anchors.npy', root / 'reference_x.npy']
    paths += sorted((root / 'ratios').glob('*/*.onnx*'))
    paths += sorted((root / 'ratios').glob('*/*.bin'))
    hashes = {str(path.relative_to(root)): file_hash(path) for path in paths}
    model_id = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:16]
    manifest = dict(model_id=model_id, files=hashes, n_reference=n_reference,
                    normalization_seed=seed, anchor_order=['down', 'nominal', 'up'],
                    normalization=normalization.tolist(),
                    reference='balanced nominal S/B spline flow; no preselection',
                    normalization_convention='finite reference integration bank')
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    hybrid.model_id = model_id
    return manifest


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


class ReferenceSampler:
    """Importance resampling approximation to the continuous hNDE distribution.

    Poisson event count is exact. Conditional event density is the weighted
    empirical proposal law. Returned anchors always belong to the GOOD model;
    epsilon changes sampling probabilities, not the returned feature convention.
    """
    def __init__(self, x, anchors, method='reference bank'):
        self.x, self.anchors, self.method = x, anchors, method
        values = np.asarray(anchors).reshape(-1, 6)
        self.sums = values.sum(axis=0)
        self.cdf = np.cumsum(values, axis=0) / self.sums
        self.gram = values.T @ values

    def sample_experiment(self, mu, alpha, config, rng, epsilon=0., n=None):
        if n is None:
            n = rng.poisson(mu * config['signal_yield'] + config['background_yield'])
        # Decompose the linear intensity into six positive anchor coefficients.
        fractions = np.array([max(-alpha, 0.), 1 - abs(alpha), max(alpha, 0.)])
        yields = np.array([mu * config['signal_yield'] * (1 - epsilon),
                           config['background_yield'] + mu * config['signal_yield'] * epsilon])
        coefficients = (yields[:, None] * fractions).reshape(6)
        total = coefficients @ self.sums
        counts = rng.multinomial(n, coefficients * self.sums / total)
        indices = np.concatenate([
            np.searchsorted(self.cdf[:, i], rng.random(count), side='right')
            for i, count in enumerate(counts)])
        rng.shuffle(indices)
        ess = total ** 2 / (coefficients @ self.gram @ coefficients)
        return dict(x=np.asarray(self.x[indices]), anchors=np.asarray(self.anchors[indices]),
                    auxiliary=float(rng.uniform(-2., 2.)), mu=float(mu), alpha=float(alpha),
                    proposal_ess=float(ess), proposal_size=len(self.anchors),
                    sampling_method=self.method)

    def sample_anchor_experiment(self, mu, alpha, config, rng, epsilon=0.):
        result = self.sample_experiment(mu, alpha, config, rng, epsilon)
        result.pop('x')
        return result


def plot_ratio_diagnostics(ratio, numerator, denominator, title, nbins=25):
    """Reliability and reweighting on independent equal-class holdout events."""
    import matplotlib.pyplot as plt
    rn, rd = ratio(numerator), ratio(denominator)
    scores = np.concatenate([rn / (1 + rn), rd / (1 + rd)])
    labels = np.concatenate([np.ones(len(rn)), np.zeros(len(rd))])
    weights = np.concatenate([np.full(len(rn), 1 / len(rn)),
                              np.full(len(rd), 1 / len(rd))])
    edges = np.linspace(0., 1., nbins + 1)
    groups = np.searchsorted(edges[1:-1], scores)
    points = []
    for i in range(nbins):
        selected = groups == i
        if selected.sum() > 20:
            points.append([np.average(scores[selected], weights=weights[selected]),
                           np.average(labels[selected], weights=weights[selected])])
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    if points:
        points = np.asarray(points)
        axes.flat[0].plot(points[:, 0], points[:, 1], 'o-', ms=3)
    axes.flat[0].plot([0, 1], [0, 1], 'k--')
    axes.flat[0].set(xlabel='predicted class probability', ylabel='held-out fraction',
                     title='Equal-class calibration')
    for j, feature in enumerate(FEATURES):
        ax = axes.flat[j + 1]
        limits = np.quantile(np.concatenate([numerator[:, j], denominator[:, j]]), [.001, .999])
        bins = np.linspace(*limits, 45)
        ax.hist(numerator[:, j], bins=bins, density=True, histtype='step', label='target')
        ax.hist(denominator[:, j], bins=bins, weights=rd, density=True,
                histtype='step', label='weighted denominator')
        ax.set(xlabel=feature, ylabel='density')
    axes.flat[1].legend()
    fig.suptitle(title + f'; independent E_den[r]={rd.mean():.5f}')
    fig.tight_layout()
    return fig


def plot_shape_closure(target, proposal, weights=None, title='Shape closure'):
    """Five one-dimensional projections plus a correlation-matrix residual."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for j, feature in enumerate(FEATURES):
        bins = np.linspace(*np.quantile(target[:, j], [.001, .999]), 45)
        axes.flat[j].hist(target[:, j], bins=bins, density=True, histtype='step', label='target')
        axes.flat[j].hist(proposal[:, j], bins=bins, weights=weights, density=True,
                          histtype='step', label='reference' if weights is None else 'weighted reference')
        axes.flat[j].set(xlabel=feature, ylabel='density')
    cov = np.cov(proposal, rowvar=False, aweights=weights)
    corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
    delta = corr - np.corrcoef(target, rowvar=False)
    image = axes.flat[5].imshow(delta, cmap='coolwarm', vmin=-.1, vmax=.1)
    axes.flat[5].set(title='Correlation residual', xticks=range(5), yticks=range(5),
                     xticklabels=FEATURES, yticklabels=FEATURES)
    fig.colorbar(image, ax=axes.flat[5])
    axes.flat[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig
