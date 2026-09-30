"""Fresh hNDE helpers for the calibration notebooks.

NSBI trains binary classifiers with a logit output. Equal class weights imply
ratio = exp(logit). Complete process/reference anchors and the exp-poly shape
interpolation are normalized on one independent reference integration sample.
"""
from pathlib import Path
from contextlib import contextmanager
import gc
import hashlib
import json
import os
import numpy as np
import pandas as pd

from model import FEATURES
from interpolation import numpy_coefficients

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
    def __init__(self, run, flow, ratios, normalization=None, model_id=None,
                 morph_normalization=None, morph_diagnostics=None):
        self.run, self.flow, self.ratios = Path(run), flow, ratios
        self.normalization = normalization
        self.morph_normalization = morph_normalization
        self.morph_diagnostics = morph_diagnostics or {}
        self.model_id = model_id
        self._integration_sampler = None

    @classmethod
    def load(cls, run, device='cpu'):
        from flow_reference import ReferenceFlow
        root = Path(run) / 'hybrid'
        manifest = json.loads((root / 'manifest.json').read_text())
        if not (root / 'morph_normalization.npy').exists():
            raise RuntimeError('This cache predates normalized exp-poly interpolation; '
                               'rerun notebook 02 with the new TAG.')
        return cls(run, ReferenceFlow.load(root / 'reference.pt', device),
                   {name: RatioEnsemble.load(root / 'ratios' / name, device=device)
                    for name in RATIO_NAMES},
                   np.load(root / 'anchor_normalization.npy'), manifest['model_id'],
                   np.load(root / 'morph_normalization.npy'),
                   manifest.get('morph_diagnostics', {}))

    def configure(self, config):
        """Attach this frozen model's normalization to a GOOD-model config."""
        if self.morph_normalization is None:
            raise RuntimeError('Save/normalize the hybrid before configuring its likelihood.')
        return dict(config, morph_normalization=self.morph_normalization.tolist(),
                    interpolation_version='normalized_exp_poly_c2_positive_v1',
                    model_epsilon=0.0)

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
    """Store anchors and the nuisance-dependent process normalization.

    Averaging the polynomial coefficients computes Z_s(alpha) on precisely the
    same integration sample as the anchor normalization. Both likelihoods and
    reference toys use these coefficients; their derivative is inexpensive.
    """
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
    coefficient_sum = None
    repaired = np.zeros(2, dtype=np.int64)
    repair_sum = np.zeros(2, dtype=np.float64)
    repair_max = np.zeros(2, dtype=np.float64)
    for start in range(0, n_reference, batch_size):
        coefficients, diagnostics = numpy_coefficients(
            anchors[start:start + batch_size], return_diagnostics=True)
        if coefficient_sum is None:
            coefficient_sum = np.zeros(coefficients.shape[1:], dtype=np.float64)
        coefficient_sum += coefficients.sum(axis=0)
        amplitudes = diagnostics['repair_amplitude']
        repaired += np.count_nonzero(diagnostics['repair_mask'], axis=0)
        repair_sum += amplitudes.sum(axis=0)
        repair_max = np.maximum(repair_max, amplitudes.max(axis=0))
    morph_normalization = coefficient_sum / n_reference
    np.save(root / 'anchor_normalization.npy', normalization)
    np.save(root / 'morph_normalization.npy', morph_normalization)
    hybrid.normalization = normalization
    hybrid.morph_normalization = morph_normalization
    hybrid.morph_diagnostics = dict(
        process_order=['signal', 'background'],
        repaired_rows=repaired.tolist(), repaired_fraction=(repaired / n_reference).tolist(),
        mean_repair_amplitude=(repair_sum / n_reference).tolist(),
        max_repair_amplitude=repair_max.tolist(),
        repair='Positive degree-eight bubble only on unsafe polynomial rows; '
               'preserves anchors, nominal first derivative, and endpoint C2 matching.')
    hybrid._integration_sampler = None
    paths = [root / 'reference.pt', root / 'anchor_normalization.npy',
             root / 'morph_normalization.npy', root / 'reference_anchors.npy',
             root / 'reference_x.npy']
    paths += sorted((root / 'ratios').glob('*/*.onnx*'))
    paths += sorted((root / 'ratios').glob('*/*.bin'))
    hashes = {str(path.relative_to(root)): file_hash(path) for path in paths}
    interpolation_hash = file_hash(Path(__file__).with_name('interpolation.py'))
    identity = dict(files=hashes, interpolation_source_sha256=interpolation_hash)
    model_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    manifest = dict(model_id=model_id, files=hashes, n_reference=n_reference,
                    interpolation_source_sha256=interpolation_hash,
                    interpolation='normalized_exp_poly',
                    morph_diagnostics=hybrid.morph_diagnostics,
                    morph_normalization=morph_normalization.tolist(),
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
    """Importance resampling approximation to the normalized exp-poly model.

    Counts are Poisson; conditional events follow an explicitly finite weighted
    proposal law. Coefficient prefix sums allow exact sampling from that law in
    O(n_events*log(n_bank)) per toy, without rescanning a five-million-event
    integration bank for every nuisance value. Returned features are always
    GOOD anchors; intentional epsilon mixing changes the sampling law only.
    """
    def __init__(self, x, anchors, method='reference bank', batch_size=100_000):
        if len(x) != len(anchors) or not len(anchors):
            raise ValueError('A reference sampler needs matching, nonempty banks.')
        self.x, self.anchors, self.method = x, anchors, method
        first = numpy_coefficients(np.asarray(anchors[:1]))
        self.n_coefficients = first.shape[-1]
        width = 2 * self.n_coefficients
        self.cumulative_coefficients = np.empty((len(anchors), width), dtype=np.float64)
        self.gram = np.zeros((width, width), dtype=np.float64)
        previous = np.zeros(width, dtype=np.float64)
        self.positivity_repaired_rows = np.zeros(2, dtype=np.int64)
        for start in range(0, len(anchors), batch_size):
            coefficients = numpy_coefficients(np.asarray(anchors[start:start + batch_size]))
            if self.n_coefficients > 7:
                self.positivity_repaired_rows += np.count_nonzero(coefficients[..., 8], axis=0)
            values = coefficients.reshape(-1, width)
            self.gram += values.T @ values
            prefix = np.cumsum(values, axis=0)
            prefix += previous
            self.cumulative_coefficients[start:start + len(values)] = prefix
            previous = prefix[-1].copy()
        self.sums = previous

    def _weights(self, mu, alpha, config, epsilon):
        if not -1.0 <= alpha <= 1.0:
            raise ValueError('Reference interpolation is defined on [-1, 1].')
        if not 0.0 <= epsilon <= 1.0 or mu < 0:
            raise ValueError('Sampling requires mu >= 0 and epsilon in [0, 1].')
        powers = float(alpha) ** np.arange(self.n_coefficients)
        normalization = np.asarray(config['morph_normalization'], dtype=np.float64)
        if normalization.shape != (2, self.n_coefficients):
            raise ValueError('Morph normalization does not match the reference sampler.')
        partitions = normalization @ powers
        if np.any(partitions <= 0) or not np.all(np.isfinite(partitions)):
            raise ValueError('Invalid nuisance-dependent process normalization.')
        yields = np.array([mu * config['signal_yield'] * (1 - epsilon),
                           config['background_yield'] + mu * config['signal_yield'] * epsilon])
        return (yields[:, None] / partitions[:, None] * powers).reshape(-1)

    def sample_experiment(self, mu, alpha, config, rng, epsilon=0., n=None):
        if n is None:
            n = rng.poisson(mu * config['signal_yield'] + config['background_yield'])
        coefficients = self._weights(mu, alpha, config, epsilon)
        total = float(coefficients @ self.sums)
        if not np.isfinite(total) or total <= 0:
            raise ValueError('The reference proposal has invalid total intensity.')
        targets = rng.random(n) * total
        left = np.zeros(n, dtype=np.int64)
        right = np.full(n, len(self.anchors), dtype=np.int64)
        # Vectorized search over an implicit CDF. Polynomial coefficients can
        # have either sign; their evaluated intensity is positive by construction.
        for _ in range(len(self.anchors).bit_length()):
            active = left < right
            if not np.any(active):
                break
            slots = np.flatnonzero(active)
            middle = (left[slots] + right[slots]) // 2
            values = np.einsum('ij,j->i', self.cumulative_coefficients[middle],
                               coefficients, optimize=False)
            go_right = values <= targets[slots]
            left[slots[go_right]] = middle[go_right] + 1
            right[slots[~go_right]] = middle[~go_right]
        indices = np.minimum(left, len(self.anchors) - 1)
        squared_sum = float(coefficients @ self.gram @ coefficients)
        ess = total ** 2 / squared_sum
        expected = mu * config['signal_yield'] + config['background_yield']
        return dict(x=np.asarray(self.x[indices]), anchors=np.asarray(self.anchors[indices]),
                    auxiliary=float(rng.uniform(-2., 2.)), mu=float(mu), alpha=float(alpha),
                    proposal_ess=float(ess), proposal_size=len(self.anchors),
                    proposal_relative_mass=float(total / (len(self.anchors) * expected)),
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
