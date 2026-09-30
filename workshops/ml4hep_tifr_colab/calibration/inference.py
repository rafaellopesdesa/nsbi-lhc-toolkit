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

from interpolation import numpy_coefficients, tensor_coefficients, tensor_morph
from model import fit, nll, with_model_epsilon
from sampling import sample_experiment


def mlp(widths, activation=nn.Tanh):
    layers = []
    for i, (n_in, n_out) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(nn.Linear(n_in, n_out))
        if i < len(widths) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


def tensor_intensity(anchors, nu, alpha, config, coefficients=None):
    shaped = tensor_morph(anchors, alpha, config, coefficients=coefficients)
    return (nu * config['signal_yield'] * shaped[..., 0]
            + config['background_yield'] * shaped[..., 1])


def centered_nll(anchors, group, counts, auxiliary, nu, alpha, config,
                 coefficients=None):
    """Full-event NLL minus its parameter-independent value at (nu, alpha)=(1,0)."""
    intensity = tensor_intensity(anchors, nu[group], alpha[group], config, coefficients)
    baseline = tensor_intensity(anchors, torch.ones_like(nu[group]),
                                torch.zeros_like(alpha[group]), config, coefficients)
    log_terms = torch.log(intensity / baseline)
    # The full likelihood can be sharply curved at high event counts. Preserve
    # the precision of the shared normalized morph in the event reduction.
    event_sum = torch.zeros(len(nu), dtype=log_terms.dtype,
                            device=nu.device).index_add(0, group, log_terms)
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


def population_loss(network, mu, signal, background, config,
                    signal_coefficients=None, background_coefficients=None):
    """Exact finite-bank population objective (up to parameter-independent terms)."""
    nu, alpha = network(mu).unbind(-1)
    # Broadcasting produces (n_mu, n_events, 2, 3).  No fit labels are needed.
    terms = []
    for anchors, coefficients in ((signal, signal_coefficients),
                                  (background, background_coefficients)):
        expanded_coefficients = None if coefficients is None else coefficients[None]
        values = tensor_intensity(anchors[None], nu[:, None], alpha[:, None], config,
                                  expanded_coefficients)
        base = tensor_intensity(anchors, torch.ones_like(alpha[:1]),
                                torch.zeros_like(alpha[:1]), config, coefficients)
        terms.append(torch.log(values / base[None]).mean(dim=-1))
    return ((nu - 1) * config['signal_yield']
            - mu * config['signal_yield'] * terms[0]
            - config['background_yield'] * terms[1] + 0.5 * alpha.square())


def train_response(signal, background, config, steps=2000, batch_mu=8,
                   learning_rate=0.002, device='cpu', seed=13041,
                   gradient_clip=5., minimum_lr_fraction=0.05):
    """Train on all population-bank events; only the physical mu values are sampled."""
    torch.manual_seed(seed)
    network = ResponseNetwork(config).to(device)
    # Certify static event polynomials once, not inside every training query.
    signal_coefficients = torch.as_tensor(numpy_coefficients(signal), dtype=torch.float64, device=device)
    background_coefficients = torch.as_tensor(numpy_coefficients(background), dtype=torch.float64, device=device)
    signal = torch.as_tensor(signal, dtype=torch.float64, device=device)
    background = torch.as_tensor(background, dtype=torch.float64, device=device)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(steps, 1), eta_min=learning_rate * minimum_lr_fraction)
    mu_lo, mu_hi = config['mu_range']
    monitor_mu = torch.linspace(mu_lo, mu_hi, 25, device=device)
    history, best_loss, best_state = [], np.inf, None
    for step in range(steps):
        mu = mu_lo + (mu_hi - mu_lo) * torch.rand(batch_mu, device=device)
        boundary_mode = torch.rand(batch_mu, device=device)
        mu[boundary_mode < 0.1] = mu_lo
        mu[(boundary_mode >= 0.1) & (boundary_mode < 0.2)] = mu_hi
        loss = population_loss(network, mu, signal, background, config,
                               signal_coefficients, background_coefficients).mean()
        optimizer.zero_grad()
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(network.parameters(), gradient_clip)
        optimizer.step()
        scheduler.step()
        if step % 50 == 0 or step == steps - 1:
            with torch.no_grad():
                monitor = population_loss(network, monitor_mu, signal, background, config,
                                           signal_coefficients, background_coefficients).mean().item()
            history.append({'step': step, 'loss': loss.item(), 'monitor': monitor,
                            'learning_rate': optimizer.param_groups[0]['lr'],
                            'gradient_norm': float(gradient_norm)})
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

    def context(self, anchors, group, counts, auxiliary, coefficients=None):
        features = torch.log1p(anchors.reshape(-1, 6)).to(dtype=self.centering_features.dtype)
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
    anchors = torch.as_tensor(np.concatenate([t['anchors'] for t in experiments]),
                              dtype=torch.float64, device=device)
    repairs = []
    for toy in experiments:
        if 'repair_amplitude' not in toy:
            _, diagnostics = numpy_coefficients(toy['anchors'], return_diagnostics=True)
            toy['repair_amplitude'] = diagnostics['repair_amplitude']
        repairs.append(toy['repair_amplitude'])
    repair = torch.as_tensor(np.concatenate(repairs), dtype=torch.float64, device=device)
    return {
        'anchors': anchors,
        'coefficients': tensor_coefficients(anchors, repair_amplitude=repair),
        'group': torch.as_tensor(np.repeat(np.arange(len(experiments)), counts), dtype=torch.long, device=device),
        'counts': torch.as_tensor(counts, dtype=torch.float32, device=device),
        'auxiliary': torch.as_tensor([t['auxiliary'] for t in experiments], dtype=torch.float32, device=device),
    }


def profile_loss(network, experiments, query, config, device,
                 own_minimum_weight=0.5, consistency_weight=1.,
                 return_components=False):
    """Optimize both heads and their agreement using achieved likelihoods only.

    The conditional head also sees the detached global POI. A one-sided regret
    penalty improves the worse achieved NLL; it never rewards making the better
    candidate worse. Neither fit labels nor nuisance-parameter labels are used.
    """
    data = pack_experiments(experiments, device)
    context = network.context(**data)
    nu, alpha = network.global_parameters(context).unbind(-1)
    conditional = network.conditional_alpha(context, query)
    global_loss = centered_nll(**data, nu=nu, alpha=alpha, config=config)
    conditional_loss = centered_nll(**data, nu=query, alpha=conditional, config=config)
    own_nu = nu.detach()
    own_alpha = network.conditional_alpha(context, own_nu)
    own_loss = centered_nll(**data, nu=own_nu, alpha=own_alpha, config=config)
    conditional_regret = (own_loss - global_loss.detach()).clamp(min=0).square()
    global_regret = (global_loss - own_loss.detach()).clamp(min=0).square()
    consistency = (conditional_regret + global_regret).mean()
    total = (global_loss.mean() + conditional_loss.mean()
             + own_minimum_weight * own_loss.mean() + consistency_weight * consistency)
    if return_components:
        return total, {'global': global_loss.mean(), 'conditional': conditional_loss.mean(),
                       'own_conditional': own_loss.mean(), 'consistency': consistency,
                       'own_minimum_gap': (2 * (own_loss - global_loss)).mean()}
    return total


def _profile_monitor(network, experiments, queries, config, device, batch_size,
                     own_minimum_weight, consistency_weight):
    totals = {'loss': 0., 'global': 0., 'conditional': 0.,
              'own_conditional': 0., 'consistency': 0., 'own_minimum_gap': 0.}
    with torch.no_grad():
        for start in range(0, len(experiments), batch_size):
            batch = experiments[start:start + batch_size]
            query = torch.as_tensor(queries[start:start + len(batch)],
                                    dtype=torch.float32, device=device)
            loss, components = profile_loss(
                network, batch, query, config, device, own_minimum_weight,
                consistency_weight, return_components=True)
            totals['loss'] += len(batch) * loss.item()
            for name, value in components.items():
                totals[name] += len(batch) * value.item()
    return {name: value / len(experiments) for name, value in totals.items()}


def train_profiles(experiments, validation, centering_anchors, config, epochs=60,
                   batch_size=8, learning_rate=0.0005, device='cpu', seed=13042,
                   gradient_clip=5., minimum_lr_fraction=0.05,
                   own_minimum_weight=0.5, consistency_weight=1.):
    """Joint full-event likelihood training, with matched deterministic monitors."""
    if not experiments or not validation:
        raise ValueError('Training and validation experiments must be nonempty.')
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    network = ProfileNetwork(centering_anchors, config).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=learning_rate * minimum_lr_fraction)
    lo, hi = config['nu_bounds']
    # Both monitors use precisely the same query sequence and sample count.
    # Subsampling across the entire ordered cache keeps both source domains.
    monitor_size = min(len(experiments), len(validation))
    train_indices = np.linspace(0, len(experiments) - 1, monitor_size, dtype=int)
    validation_indices = np.linspace(0, len(validation) - 1, monitor_size, dtype=int)
    training_monitor = [experiments[i] for i in train_indices]
    validation_monitor = [validation[i] for i in validation_indices]
    monitor_query = np.resize(np.linspace(lo, hi, 9), monitor_size)
    history, best_loss, best_state = [], np.inf, None
    for epoch in range(epochs):
        network.train()
        order = rng.permutation(len(experiments))
        losses, gradient_norms = [], []
        for start in range(0, len(order), batch_size):
            batch = [experiments[i] for i in order[start:start + batch_size]]
            query = rng.uniform(lo, hi, len(batch))
            # Explicit endpoints supplement broad and near-generating queries.
            # No fitted estimates or profile labels enter the training data.
            query_mode = rng.random(len(batch))
            focused = query_mode < 0.5
            near = np.asarray([t['mu'] for t in batch]) + rng.normal(0, 0.35, len(batch))
            query[focused] = np.clip(near[focused], lo, hi)
            query[(query_mode >= 0.5) & (query_mode < 0.6)] = lo
            query[(query_mode >= 0.6) & (query_mode < 0.7)] = hi
            query = torch.as_tensor(query, dtype=torch.float32, device=device)
            loss = profile_loss(network, batch, query, config, device,
                                own_minimum_weight, consistency_weight)
            optimizer.zero_grad()
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(network.parameters(), gradient_clip)
            optimizer.step()
            losses.append(loss.item())
            gradient_norms.append(float(gradient_norm))
        network.eval()
        train_monitor = _profile_monitor(
            network, training_monitor, monitor_query, config, device, batch_size,
            own_minimum_weight, consistency_weight)
        heldout_monitor = _profile_monitor(
            network, validation_monitor, monitor_query, config, device, batch_size,
            own_minimum_weight, consistency_weight)
        validation_loss = heldout_monitor['loss']
        record = {'epoch': epoch, 'train': train_monitor['loss'],
                  'validation': validation_loss, 'train_step': float(np.mean(losses)),
                  'learning_rate': optimizer.param_groups[0]['lr'],
                  'gradient_norm': float(np.mean(gradient_norms)),
                  'monitor_experiments': monitor_size}
        for prefix, monitor in [('train', train_monitor), ('validation', heldout_monitor)]:
            record.update({prefix + '_' + key: value for key, value in monitor.items()
                           if key != 'loss'})
        history.append(record)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f'epoch {epoch:3d}: matched train {train_monitor["loss"]:.5g}, '
                  f'validation {validation_loss:.5g}, lr {record["learning_rate"]:.3g}')
        scheduler.step()
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
            generating_mode = rng.random()
            if generating_mode < 0.1:
                mu = float(generating_range[0])
            elif generating_mode < 0.2:
                mu = float(generating_range[1])
            if rng.random() < 0.25:
                alpha = 0.
            if source == 'simulator':
                toy = sample_experiment(mu, alpha, config, rng)
                anchors = hybrid.anchors(toy['x'])
            else:
                toy = hybrid.sample_anchor_experiment(mu, alpha, config, rng, epsilon=epsilon, proposal_size=0)
                anchors = toy['anchors']
            # Certify and retain exactly the same values for the likelihood.
            # Only the encoder casts its features to the network's float32 dtype.
            anchors = np.asarray(anchors, dtype=np.float64)
            if 'repair_amplitude' in toy:
                repair_amplitude = np.asarray(toy['repair_amplitude'], dtype=np.float64)
            else:
                _, diagnostics = numpy_coefficients(anchors, return_diagnostics=True)
                repair_amplitude = diagnostics['repair_amplitude']
            experiments.append({'anchors': anchors, 'repair_amplitude': repair_amplitude,
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
        self.config = with_model_epsilon(config, epsilon)
        self.epsilon, self.device = epsilon, device

    def response(self, mu):
        return predict_response(self.response_network, mu)

    def evaluate(self, anchors, auxiliary, nu, coefficients=None):
        """Evaluate a scalar/vector nu. `statistic` is SIGNED; no clipping here.

        A fixed POI grid, global predictions and nuisance endpoints supplement
        the NN. The global candidate set never depends on the requested query.
        This is a finite search, not iterative numerical profiling.
        """
        anchors = np.asarray(anchors, dtype=np.float64)
        query = np.atleast_1d(nu).astype(float)
        if coefficients is None:
            coefficients, diagnostics = numpy_coefficients(anchors, return_diagnostics=True)
            repair = diagnostics['repair_amplitude']
        else:
            coefficients = np.asarray(coefficients, dtype=np.float64)
            repair = -coefficients[..., 8]
        data = pack_experiments([{'anchors': anchors, 'repair_amplitude': repair,
                                  'auxiliary': auxiliary}], self.device)
        data['coefficients'] = torch.as_tensor(coefficients, dtype=torch.float64, device=self.device)
        lo, hi = self.config['nu_bounds']
        with torch.no_grad():
            context = self.profile_network.context(**data)
            raw_global = self.profile_network.global_parameters(context)[0].cpu().numpy()
            global_queries = np.unique(np.r_[np.linspace(lo, hi, 13), float(raw_global[0])])
            all_nu = np.r_[query, global_queries]
            # Keep the global forward-pass batch fixed as well: its floating-
            # point result should not depend on the number of requested queries.
            predictions = []
            for values in (query, global_queries):
                q_tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
                predictions.append(self.profile_network.conditional_alpha(
                    context.expand(len(values), -1), q_tensor).cpu().numpy())
            predicted_alpha = np.concatenate(predictions)
        conditional_results = []
        def objective(q, a):
            return nll(anchors, q, a, auxiliary, self.config, coefficients=coefficients)
        for q, a in zip(all_nu, predicted_alpha):
            candidates = [(objective(q, candidate), candidate)
                          for candidate in (float(a), -1., 0., 1.)]
            value, best_alpha = min(candidates)
            conditional_results.append((value, best_alpha))
        global_candidates = [(objective(float(raw_global[0]), candidate), float(raw_global[0]), candidate)
                             for candidate in (float(raw_global[1]), -1., 0., 1.)]
        global_candidates.extend((result[0], float(q), result[1])
                                 for q, result in zip(global_queries, conditional_results[len(query):]))
        global_value, global_nu, global_alpha = min(global_candidates)
        global_index = int(np.flatnonzero(global_queries == global_nu)[0])
        at_global = conditional_results[len(query) + global_index]
        head_at_global = objective(global_nu, float(predicted_alpha[len(query) + global_index]))
        # Reuse the achieved global nuisance at every tested POI, so this extra
        # candidate is a continuous likelihood curve rather than an equality-
        # only special case. The denominator is unchanged, and negative values
        # at other queries remain possible. Raw head discrepancies below stay
        # visible and do not claim an improvement in training by themselves.
        for i, q in enumerate(query):
            conditional_results[i] = min(conditional_results[i],
                                          (objective(q, global_alpha), global_alpha))
        conditional_values = np.asarray([r[0] for r in conditional_results[:len(query)]])
        statistic = 2 * (conditional_values - global_value)
        result = {'global_nu': float(global_nu), 'global_alpha': float(global_alpha),
                  'nll_global': float(global_value),
                  'conditional_alpha': np.asarray([r[1] for r in conditional_results[:len(query)]]),
                  'nll_conditional': conditional_values, 'statistic': statistic,
                  'negative_gap': np.minimum(statistic, 0.),
                  'raw_global_nu': float(raw_global[0]), 'raw_global_alpha': float(raw_global[1]),
                  'conditional_head_T_at_global': float(2 * (head_at_global - global_value)),
                  'candidate_T_at_global': float(2 * (at_global[0] - global_value))}
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
        current_config = with_model_epsilon(current_config, current_epsilon)
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
    coefficients = numpy_coefficients(anchors)
    rows = []
    for mu in mu_grid:
        weights = np.r_[np.full(len(signal), mu * config['signal_yield'] / len(signal)),
                        np.full(len(background), config['background_yield'] / len(background))]
        exact = fit(anchors, 0., config, weights=weights, coefficients=coefficients)
        nu, alpha = predict_response(response_network, mu)
        nn_nll = nll(anchors, nu, alpha, 0., config, weights=weights, coefficients=coefficients)
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
            anchors = hybrid.anchors(toy['x'])
            coefficients = numpy_coefficients(anchors)
            query = float(inference.response(mu)[0])
            start = time.perf_counter()
            estimate = inference.evaluate(anchors, toy['auxiliary'], query, coefficients=coefficients)
            nn_seconds = time.perf_counter() - start
            start = time.perf_counter()
            global_fit = fit(anchors, toy['auxiliary'], config, coefficients=coefficients)
            conditional_fit = fit(anchors, toy['auxiliary'], config, fixed_nu=query,
                                  coefficients=coefficients)
            exact_seconds = time.perf_counter() - start
            exact_t = 2 * (conditional_fit['nll'] - global_fit['nll'])
            at_own_minimum = inference.evaluate(anchors, toy['auxiliary'], estimate['global_nu'],
                                                coefficients=coefficients)['statistic']
            row = {'mu': float(mu), 'query_nu': query, 'N': len(anchors), 'auxiliary': toy['auxiliary'],
                   'statistic_nn': estimate['statistic'], 'statistic_exact': exact_t,
                   'global_gap': estimate['nll_global'] - global_fit['nll'],
                   'conditional_gap': estimate['nll_conditional'] - conditional_fit['nll'],
                   'nu_nn': estimate['global_nu'], 'nu_exact': global_fit['nu'],
                   'alpha_nn': estimate['global_alpha'], 'alpha_exact': global_fit['alpha'],
                   'conditional_alpha_nn': estimate['conditional_alpha'], 'conditional_alpha_exact': conditional_fit['alpha'],
                   'statistic_at_own_minimum': at_own_minimum,
                   'conditional_head_T_at_global': estimate['conditional_head_T_at_global'],
                   'candidate_T_at_global': estimate['candidate_T_at_global'],
                   'nn_seconds': nn_seconds, 'exact_seconds': exact_seconds,
                   'fit_success': bool(global_fit['success'] and conditional_fit['success'])}
            rows.append(row)
            if index == 0:
                examples.append({'anchors': np.asarray(anchors, dtype=np.float64),
                                 'auxiliary': toy['auxiliary'], 'mu': float(mu)})
    return rows, examples
