"""Likelihood-trained population response and unbinned amortized profiling.

No optimized-fit labels are used in either training objective.  Final statistics
are evaluated with the original full-event likelihood at network predictions.
"""
from pathlib import Path
import json
import time

import numpy as np
import torch
from torch import nn

from model import bad_anchors, fit, nll
from sampling import sample_experiment, sample_process


def mlp(widths, activation=nn.Tanh):
    layers = []
    for i, (n_in, n_out) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(nn.Linear(n_in, n_out))
        if i < len(widths) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


def tensor_morph(anchors, alpha):
    """Linear shape interpolation, with alpha already restricted to [-1, 1]."""
    return ((1 - alpha.abs())[..., None] * anchors[..., 1]
            + alpha.clamp(min=0)[..., None] * anchors[..., 2]
            + (-alpha).clamp(min=0)[..., None] * anchors[..., 0])


def tensor_intensity(anchors, nu, alpha, config):
    shaped = tensor_morph(anchors, alpha)
    return (nu * config['signal_yield'] * shaped[..., 0]
            + config['background_yield'] * shaped[..., 1])


def centered_nll(anchors, group, counts, auxiliary, nu, alpha, config):
    """Full-event NLL minus its parameter-independent value at (nu, alpha)=(1,0)."""
    intensity = tensor_intensity(anchors, nu[group], alpha[group], config)
    baseline = (config['signal_yield'] * anchors[:, 0, 1]
                + config['background_yield'] * anchors[:, 1, 1])
    log_terms = torch.log(intensity / baseline)
    event_sum = torch.zeros_like(nu).index_add(0, group, log_terms)
    return ((nu - 1) * config['signal_yield'] - event_sum
            + 0.5 * ((auxiliary - alpha).square() - auxiliary.square()))


class ResponseNetwork(nn.Module):
    """Physical mu -> pseudo-true fitted (nu, alpha), with alpha_gen fixed at 0."""
    def __init__(self, config, width=32):
        super().__init__()
        self.config = config
        self.width = width
        self.net = mlp([1, width, width, 2])

    def forward(self, mu):
        lo, hi = self.config['mu_range']
        out = self.net((2 * (mu.reshape(-1, 1) - lo) / (hi - lo) - 1))
        nu_lo, nu_hi = self.config['nu_bounds']
        return torch.stack([nu_lo + (nu_hi - nu_lo) * out[:, 0].sigmoid(),
                            out[:, 1].tanh()], dim=-1)


def predict_response(network, mu):
    """Evaluate a scalar or array of physical POIs with a trained response NN."""
    shape = np.asarray(mu).shape
    device = next(network.parameters()).device
    with torch.no_grad():
        values = torch.as_tensor(np.asarray(mu).reshape(-1), dtype=torch.float32, device=device)
        result = network(values).cpu().numpy()
    return result.reshape(shape + (2,))


def population_loss(network, mu, signal, background, config):
    """Exact finite-bank population objective (up to parameter-independent terms)."""
    nu, alpha = network(mu).unbind(-1)
    # Broadcasting produces (n_mu, n_events, 2, 3).  No fit labels are needed.
    terms = []
    for anchors in (signal, background):
        values = tensor_intensity(anchors[None], nu[:, None], alpha[:, None], config)
        base = config['signal_yield'] * anchors[:, 0, 1] + config['background_yield'] * anchors[:, 1, 1]
        terms.append(torch.log(values / base[None]).mean(dim=-1))
    return ((nu - 1) * config['signal_yield']
            - mu * config['signal_yield'] * terms[0]
            - config['background_yield'] * terms[1] + 0.5 * alpha.square())


def train_response(signal, background, config, steps=2000, batch_mu=8,
                   learning_rate=0.002, device='cpu', seed=13041):
    """Train on all population-bank events; only the physical mu values are sampled."""
    torch.manual_seed(seed)
    network = ResponseNetwork(config).to(device)
    signal = torch.as_tensor(signal, dtype=torch.float32, device=device)
    background = torch.as_tensor(background, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    mu_lo, mu_hi = config['mu_range']
    monitor_mu = torch.linspace(mu_lo, mu_hi, 25, device=device)
    history, best_loss, best_state = [], np.inf, None
    for step in range(steps):
        mu = mu_lo + (mu_hi - mu_lo) * torch.rand(batch_mu, device=device)
        loss = population_loss(network, mu, signal, background, config).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step == steps - 1:
            with torch.no_grad():
                monitor = population_loss(network, monitor_mu, signal, background, config).mean().item()
            history.append({'step': step, 'loss': loss.item(), 'monitor': monitor})
            if monitor < best_loss:
                best_loss = monitor
                best_state = {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}
    network.load_state_dict(best_state)
    return network, history


class ProfileNetwork(nn.Module):
    """Trainable DeepSet and two optimizers: global (nu,alpha), conditional alpha."""
    def __init__(self, centering_anchors, config, event_width=32, head_width=64):
        super().__init__()
        self.config = config
        self.event_width, self.head_width = event_width, head_width
        self.register_buffer('centering_features', torch.log1p(torch.as_tensor(centering_anchors, dtype=torch.float32).reshape(-1, 6)))
        self.encoder = mlp([6, event_width, event_width])
        self.global_head = mlp([event_width + 2, head_width, head_width, 2])
        self.conditional_head = mlp([event_width + 3, head_width, head_width, 1])

    def context(self, anchors, group, counts, auxiliary):
        features = torch.log1p(anchors.reshape(-1, 6))
        embedded = self.encoder(features)
        pooled = torch.zeros(len(counts), self.event_width, device=anchors.device).index_add(0, group, embedded)
        center = self.encoder(self.centering_features).mean(0)
        # Centered/scaled mean pooling retains count separately.  For N=0 the
        # event summary is exactly zero; this is the definition of the empty set.
        scale = counts.clamp(min=1).sqrt()
        summary = (pooled - counts[:, None] * center) / scale[:, None]
        count_feature = (counts - self.config['background_yield']) / np.sqrt(self.config['background_yield'])
        return torch.cat([summary, count_feature[:, None], auxiliary[:, None]], dim=-1)

    def global_parameters(self, context):
        out = self.global_head(context)
        lo, hi = self.config['nu_bounds']
        return torch.stack([lo + (hi - lo) * out[:, 0].sigmoid(), out[:, 1].tanh()], dim=-1)

    def conditional_alpha(self, context, nu):
        lo, hi = self.config['nu_bounds']
        query = 2 * (nu - lo) / (hi - lo) - 1
        return self.conditional_head(torch.cat([context, query[:, None]], dim=-1))[:, 0].tanh()


def pack_experiments(experiments, device):
    counts = np.asarray([len(t['anchors']) for t in experiments])
    return {
        'anchors': torch.as_tensor(np.concatenate([t['anchors'] for t in experiments]), dtype=torch.float32, device=device),
        'group': torch.as_tensor(np.repeat(np.arange(len(experiments)), counts), dtype=torch.long, device=device),
        'counts': torch.as_tensor(counts, dtype=torch.float32, device=device),
        'auxiliary': torch.as_tensor([t['auxiliary'] for t in experiments], dtype=torch.float32, device=device),
    }


def profile_loss(network, experiments, query, config, device):
    data = pack_experiments(experiments, device)
    context = network.context(**data)
    nu, alpha = network.global_parameters(context).unbind(-1)
    conditional = network.conditional_alpha(context, query)
    global_loss = centered_nll(**data, nu=nu, alpha=alpha, config=config)
    conditional_loss = centered_nll(**data, nu=query, alpha=conditional, config=config)
    return global_loss.mean() + conditional_loss.mean()


def train_profiles(experiments, validation, centering_anchors, config, epochs=60,
                   batch_size=8, learning_rate=0.0005, device='cpu', seed=13042):
    """Jointly train encoder and optimizer heads using full-event likelihoods."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    network = ProfileNetwork(centering_anchors, config).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    lo, hi = config['nu_bounds']
    validation_query = rng.uniform(lo, hi, len(validation))
    history, best_loss, best_state = [], np.inf, None
    for epoch in range(epochs):
        network.train()
        order = rng.permutation(len(experiments))
        losses = []
        for start in range(0, len(order), batch_size):
            batch = [experiments[i] for i in order[start:start + batch_size]]
            query = rng.uniform(lo, hi, len(batch))
            # Half the queries concentrate near the generating POI; no fitted
            # estimates or profile labels enter the training data.
            focused = rng.random(len(batch)) < 0.5
            near = np.asarray([t['mu'] for t in batch]) + rng.normal(0, 0.35, len(batch))
            query[focused] = np.clip(near[focused], lo, hi)
            query = torch.as_tensor(query, dtype=torch.float32, device=device)
            loss = profile_loss(network, batch, query, config, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        network.eval()
        validation_sum = 0.
        with torch.no_grad():
            for start in range(0, len(validation), batch_size):
                batch = validation[start:start + batch_size]
                query = torch.as_tensor(validation_query[start:start + len(batch)], dtype=torch.float32, device=device)
                validation_sum += len(batch) * profile_loss(network, batch, query, config, device).item()
        validation_loss = validation_sum / len(validation)
        history.append({'epoch': epoch, 'train': float(np.mean(losses)), 'validation': validation_loss})
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f'epoch {epoch:3d}: train {np.mean(losses):.5g}, validation {validation_loss:.5g}')
    network.load_state_dict(best_state)
    return network, history


def make_training_experiments(hybrid, config, epsilon, n_per_source, rng):
    """Physical simulator toys + explicitly finite-bank reference training toys."""
    experiments = []
    lo, hi = config['mu_range']
    for source in ('simulator', 'reference'):
        for i in range(n_per_source):
            generating_range = config['mu_range'] if source == 'simulator' else config['nu_bounds']
            mu, alpha = rng.uniform(*generating_range), rng.uniform(-1, 1)
            if source == 'simulator':
                toy = sample_experiment(mu, alpha, config, rng)
                anchors = hybrid.anchors(toy['x'])
            else:
                toy = hybrid.sample_anchor_experiment(mu, alpha, config, rng, epsilon=epsilon, proposal_size=0)
                anchors = toy['anchors']
            experiments.append({'anchors': bad_anchors(anchors, epsilon).astype(np.float32),
                                'auxiliary': toy['auxiliary'], 'mu': mu,
                                'alpha_gen': alpha, 'source': source})
            if (i + 1) % 250 == 0:
                print(f'{source}: {i + 1}/{n_per_source} experiments')
    return experiments


class Inference:
    """Frozen networks with full-likelihood evaluation and explicit boundary checks."""
    def __init__(self, response_network, profile_network, config, epsilon, device='cpu'):
        self.response_network = response_network.to(device).eval()
        self.profile_network = profile_network.to(device).eval()
        self.config, self.epsilon, self.device = config, epsilon, device

    def response(self, mu):
        return predict_response(self.response_network, mu)

    def evaluate(self, anchors, auxiliary, nu):
        """Evaluate a scalar/vector nu. `statistic` is SIGNED; no clipping here.

        Finite candidates at alpha=-1,0,+1 and nu bounds handle the interpolation
        kink and boundaries. They do not perform iterative numerical profiling.
        """
        anchors = np.asarray(anchors, dtype=np.float64)
        query = np.atleast_1d(nu).astype(float)
        data = pack_experiments([{'anchors': anchors, 'auxiliary': auxiliary}], self.device)
        lo, hi = self.config['nu_bounds']
        with torch.no_grad():
            context = self.profile_network.context(**data)
            raw_global = self.profile_network.global_parameters(context)[0].cpu().numpy()
            all_nu = np.r_[query, lo, hi, float(raw_global[0])]
            q_tensor = torch.as_tensor(all_nu, dtype=torch.float32, device=self.device)
            predicted_alpha = self.profile_network.conditional_alpha(context.expand(len(all_nu), -1), q_tensor).cpu().numpy()
        conditional_results = []
        for q, a in zip(all_nu, predicted_alpha):
            candidates = [(nll(anchors, q, candidate, auxiliary, self.config), candidate)
                          for candidate in (float(a), -1., 0., 1.)]
            value, best_alpha = min(candidates)
            conditional_results.append((value, best_alpha))
        global_candidates = [(nll(anchors, float(raw_global[0]), candidate, auxiliary, self.config), float(raw_global[0]), candidate)
                             for candidate in (float(raw_global[1]), -1., 0., 1.)]
        global_candidates.extend([(conditional_results[-3][0], lo, conditional_results[-3][1]),
                                  (conditional_results[-2][0], hi, conditional_results[-2][1]),
                                  (conditional_results[-1][0], float(raw_global[0]), conditional_results[-1][1])])
        global_value, global_nu, global_alpha = min(global_candidates)
        conditional_values = np.asarray([r[0] for r in conditional_results[:-3]])
        statistic = 2 * (conditional_values - global_value)
        result = {'global_nu': float(global_nu), 'global_alpha': float(global_alpha),
                  'nll_global': float(global_value),
                  'conditional_alpha': np.asarray([r[1] for r in conditional_results[:-3]]),
                  'nll_conditional': conditional_values, 'statistic': statistic,
                  'negative_gap': np.minimum(statistic, 0.),
                  'raw_global_nu': float(raw_global[0]), 'raw_global_alpha': float(raw_global[1])}
        if np.ndim(nu) == 0:
            for key in ('conditional_alpha', 'nll_conditional', 'statistic', 'negative_gap'):
                result[key] = float(result[key][0])
        return result


def save_inference(run, inference, metadata=None):
    folder = Path(run) / 'inference'
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = {'config': inference.config, 'epsilon': inference.epsilon,
                  'response_width': inference.response_network.width,
                  'event_width': inference.profile_network.event_width,
                  'head_width': inference.profile_network.head_width,
                  'response_state': {k: v.detach().cpu() for k, v in inference.response_network.state_dict().items()},
                  'profile_state': {k: v.detach().cpu() for k, v in inference.profile_network.state_dict().items()}}
    torch.save(checkpoint, folder / 'networks.pt')
    (folder / 'metadata.json').write_text(json.dumps(metadata or {}, indent=2))


def load_inference(run, device='cpu'):
    run = Path(run)
    metadata = json.loads((run / 'inference/metadata.json').read_text())
    if 'hybrid_model_id' in metadata:
        current_config = json.loads((run / 'config.json').read_text())
        current_epsilon = json.loads((run / 'misspecification.json').read_text())['epsilon']
        current_hybrid = json.loads((run / 'hybrid/manifest.json').read_text())['model_id']
        if (metadata['config'] != current_config or metadata['epsilon'] != current_epsilon
                or metadata['hybrid_model_id'] != current_hybrid):
            raise ValueError('Inference checkpoint belongs to a different model. Rerun notebook 04 in the matching RUN.')
    checkpoint = torch.load(Path(run) / 'inference/networks.pt', map_location='cpu', weights_only=False)
    response = ResponseNetwork(checkpoint['config'], checkpoint['response_width'])
    response.load_state_dict(checkpoint['response_state'])
    # Constructor expects anchors, then replaces its buffer from the checkpoint.
    center = torch.expm1(checkpoint['profile_state']['centering_features']).reshape(-1, 2, 3)
    profile = ProfileNetwork(center, checkpoint['config'], checkpoint['event_width'], checkpoint['head_width'])
    profile.load_state_dict(checkpoint['profile_state'])
    return Inference(response, profile, checkpoint['config'], checkpoint['epsilon'], device)


def response_validation(response_network, signal, background, mu_grid, config):
    """Compare response NN against direct fits on a supplied population bank."""
    anchors = np.concatenate([signal, background]).astype(np.float64)
    rows = []
    for mu in mu_grid:
        weights = np.r_[np.full(len(signal), mu * config['signal_yield'] / len(signal)),
                        np.full(len(background), config['background_yield'] / len(background))]
        exact = fit(anchors, 0., config, weights=weights)
        nu, alpha = predict_response(response_network, mu)
        nn_nll = nll(anchors, nu, alpha, 0., config, weights=weights)
        rows.append({'mu': float(mu), 'nu_nn': float(nu), 'alpha_nn': float(alpha),
                     'nu_exact': exact['nu'], 'alpha_exact': exact['alpha'],
                     'nll_gap': float(nn_nll - exact['nll']), 'fit_success': exact['success']})
    return rows


def validate_toys(inference, hybrid, config, epsilon, mu_grid, n_per_mu, rng, alpha_gen=0.):
    """Actual fresh physical events, independent of all training banks."""
    rows, examples = [], []
    for mu in mu_grid:
        for index in range(n_per_mu):
            toy = sample_experiment(mu, alpha_gen, config, rng)
            anchors = bad_anchors(hybrid.anchors(toy['x']), epsilon)
            query = float(inference.response(mu)[0])
            start = time.perf_counter()
            estimate = inference.evaluate(anchors, toy['auxiliary'], query)
            nn_seconds = time.perf_counter() - start
            start = time.perf_counter()
            global_fit = fit(anchors, toy['auxiliary'], config)
            conditional_fit = fit(anchors, toy['auxiliary'], config, fixed_nu=query)
            exact_seconds = time.perf_counter() - start
            exact_t = 2 * (conditional_fit['nll'] - global_fit['nll'])
            at_own_minimum = inference.evaluate(anchors, toy['auxiliary'], estimate['global_nu'])['statistic']
            row = {'mu': float(mu), 'query_nu': query, 'N': len(anchors), 'auxiliary': toy['auxiliary'],
                   'statistic_nn': estimate['statistic'], 'statistic_exact': exact_t,
                   'global_gap': estimate['nll_global'] - global_fit['nll'],
                   'conditional_gap': estimate['nll_conditional'] - conditional_fit['nll'],
                   'nu_nn': estimate['global_nu'], 'nu_exact': global_fit['nu'],
                   'alpha_nn': estimate['global_alpha'], 'alpha_exact': global_fit['alpha'],
                   'conditional_alpha_nn': estimate['conditional_alpha'], 'conditional_alpha_exact': conditional_fit['alpha'],
                   'statistic_at_own_minimum': at_own_minimum,
                   'nn_seconds': nn_seconds, 'exact_seconds': exact_seconds,
                   'fit_success': bool(global_fit['success'] and conditional_fit['success'])}
            rows.append(row)
            if index == 0:
                examples.append({'anchors': anchors.astype(np.float32), 'auxiliary': toy['auxiliary'], 'mu': float(mu)})
    return rows, examples
