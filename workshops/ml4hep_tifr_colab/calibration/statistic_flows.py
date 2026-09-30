"""Small conditional linear-spline flows, including a possible statistic atom at zero.

A positive statistic is mapped to (0, 1) by sqrt(t)/(sqrt(t)+sqrt(scale)).
A conditional positive density on this interval is a monotone, piecewise-linear
CDF with an analytic inverse. This is a one-dimensional normalizing flow;
its spline shape is unrelated to a quadratic likelihood approximation.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json

import numpy as np
import torch
from torch import nn


class ConditionalCDF(nn.Module):
    """A scalar conditional CDF flow; optional atom plus positive-statistic support."""

    def __init__(self, mu_range=(0., 2.), bins=64, hidden=64,
                 statistic=False, scale=1.):
        super().__init__()
        self.settings = dict(mu_range=list(mu_range), bins=bins, hidden=hidden,
                             statistic=statistic, scale=scale)
        self.mu_range = tuple(mu_range)
        self.bins, self.statistic, self.scale = bins, statistic, scale
        self.network = nn.Sequential(nn.Linear(3, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.SiLU(),
                                     nn.Linear(hidden, bins + int(statistic)))
        # Uniform interval density is the identity flow before training.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        if statistic:
            with torch.no_grad():
                self.network[-1].bias[0] = -3.

    def probabilities(self, mu):
        lo, hi = self.mu_range
        mu = mu.reshape(-1, 1)
        # Endpoint indicators allow a boundary atom without imposing it just inside.
        context = torch.cat([2 * (mu - lo) / (hi - lo) - 1,
                             (mu == lo).to(mu.dtype), (mu == hi).to(mu.dtype)], dim=1)
        return torch.softmax(self.network(context), dim=-1)

    def coordinates(self, value):
        if self.statistic:
            root = torch.sqrt(value)
            return torch.where(torch.isposinf(value), torch.ones_like(value),
                               root / (root + np.sqrt(self.scale)))
        return value

    def log_prob(self, value, mu):
        """Mixed-measure log likelihood: atom probability or continuous density."""
        probability = self.probabilities(mu)
        coordinate = self.coordinates(value)
        # At the right endpoint, use the limiting last-bin density.
        index = torch.minimum((coordinate * self.bins).long(),
                              torch.full_like(value, self.bins - 1, dtype=torch.long))
        rows = torch.arange(len(value), device=value.device)
        if not self.statistic:
            return torch.log(probability[rows, index]) + np.log(self.bins)
        positive = value > 0
        result = torch.log(probability[:, 0])
        # Do not evaluate log(0): zero belongs to a discrete component.
        t = value[positive]
        log_jacobian = (.5 * np.log(self.scale) - np.log(2.)
                        - .5 * torch.log(t)
                        - 2 * torch.log(torch.sqrt(t) + np.sqrt(self.scale)))
        result[positive] = (torch.log(probability[rows[positive], index[positive] + 1])
                            + np.log(self.bins) + log_jacobian)
        return result

    def _inputs(self, mu, value):
        mu, value = np.broadcast_arrays(np.asarray(mu, float), np.asarray(value, float))
        parameter = next(self.parameters())
        inputs = [torch.as_tensor(x.reshape(-1), dtype=parameter.dtype,
                                  device=parameter.device) for x in (mu, value)]
        return mu.shape, inputs

    @torch.no_grad()
    def cdf(self, mu, value, left=False):
        """CDF; left=True returns F(t-), distinct at the statistic's zero atom."""
        shape, (mu, value) = self._inputs(mu, value)
        if torch.any(value < 0) or (not self.statistic and torch.any(value > 1)):
            raise ValueError("The flow input is outside its defined support.")
        probability = self.probabilities(mu)
        if self.statistic:
            atom, masses = probability[:, 0], probability[:, 1:]
        else:
            atom, masses = torch.zeros_like(value), probability
        coordinate = self.coordinates(value)
        location = coordinate * self.bins
        index = torch.minimum(location.long(),
                              torch.full_like(value, self.bins - 1, dtype=torch.long))
        cumulative = torch.cat([torch.zeros_like(masses[:, :1]),
                                torch.cumsum(masses, dim=1)], dim=1)
        rows = torch.arange(len(value), device=value.device)
        result = atom + cumulative[rows, index] + masses[rows, index] * (location - index)
        if self.statistic and left:
            result[value == 0] = 0.
        if self.statistic:
            result[torch.isposinf(value)] = 1.
        if not self.statistic:
            result[value == 1] = 1.
        return result.cpu().numpy().reshape(shape)

    @torch.no_grad()
    def ppf(self, mu, probability):
        """Generalized inverse: probabilities inside the atom return exactly zero."""
        shape, (mu, probability) = self._inputs(mu, probability)
        if torch.any((probability < 0) | (probability > 1)):
            raise ValueError("A quantile probability must lie in [0, 1].")
        masses = self.probabilities(mu)
        if self.statistic:
            atom, masses = masses[:, 0], masses[:, 1:]
        else:
            atom = torch.zeros_like(probability)
        cumulative = torch.cat([torch.zeros_like(masses[:, :1]),
                                torch.cumsum(masses, dim=1)], dim=1)
        target = probability - atom
        index = torch.sum(target[:, None] >= cumulative[:, 1:], dim=1)
        index = torch.minimum(index, torch.full_like(index, self.bins - 1))
        rows = torch.arange(len(probability), device=probability.device)
        coordinate = (index + (target - cumulative[rows, index]) / masses[rows, index]) / self.bins
        coordinate[probability <= atom] = 0.
        coordinate[probability == 1] = 1.
        result = (self.scale * (coordinate / (1 - coordinate)) ** 2
                  if self.statistic else coordinate)
        return result.cpu().numpy().reshape(shape)


def train_flow(mu, values, *, statistic=False, mu_range=(0., 2.),
               bins=64, hidden=64, scale=1., epochs=150, batch_size=1024,
               learning_rate=1e-3, seed=13001, device="cpu", checkpoint=None):
    """Maximum likelihood with a disjoint validation split and best-epoch selection."""
    mu, values = np.asarray(mu, float), np.asarray(values, float)
    if np.any(values < 0) or (not statistic and np.any(values > 1)):
        raise ValueError("Training values lie outside the flow support.")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(mu))
    split = int(.8 * len(order))
    training, validation = order[:split], order[split:]
    model = ConditionalCDF(mu_range, bins, hidden, statistic, scale).to(device).double()
    x = torch.as_tensor(values, dtype=torch.float64, device=device)
    context = torch.as_tensor(mu, dtype=torch.float64, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss, best_state, history = np.inf, None, []
    for epoch in range(epochs):
        model.train()
        epoch_order = rng.permutation(training)
        for start in range(0, len(training), batch_size):
            indices = epoch_order[start:start + batch_size]
            loss = -model.log_prob(x[indices], context[indices]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_loss = float(-model.log_prob(x[training], context[training]).mean())
            validation_loss = float(-model.log_prob(x[validation], context[validation]).mean())
        history.append((train_loss, validation_loss))
        if validation_loss < best_loss:
            best_loss, best_state = validation_loss, deepcopy(model.state_dict())
        if epoch == 0 or (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch + 1:3d}: train {train_loss:.4f}, validation {validation_loss:.4f}")
    model.load_state_dict(best_state)
    if checkpoint is not None:
        Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(settings=model.settings, state_dict=model.state_dict(),
                        history=np.asarray(history).tolist(), seed=seed), checkpoint)
    return model, np.asarray(history)


def load_flow(checkpoint, device="cpu"):
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    flow = ConditionalCDF(**saved["settings"]).to(device).double()
    flow.load_state_dict(saved["state_dict"])
    return flow.eval()


class ComposedCDF:
    """F = G o K o Q; Q can include an atom, K and G are continuous CDF flows."""
    def __init__(self, base, *residuals):
        self.base, self.residuals = base, residuals

    def cdf(self, mu, statistic, left=False):
        value = self.base.cdf(mu, statistic, left=left)
        for flow in self.residuals:
            value = flow.cdf(mu, value)
        return value

    def ppf(self, mu, probability):
        value = probability
        for flow in reversed(self.residuals):
            value = flow.ppf(mu, value)
        return self.base.ppf(mu, value)

    def pit(self, mu, statistic, rng):
        """Randomized PIT: F(t-) + V [F(t)-F(t-)], V independent uniform."""
        lower, upper = self.cdf(mu, statistic, left=True), self.cdf(mu, statistic)
        return lower + rng.uniform(size=np.shape(upper)) * (upper - lower)


def flow_identity(run, config):
    """Identity of the frozen inference, hybrid, and misspecification checkpoints."""
    run = Path(run)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    # Arrays of toy data/plots are deliberately excluded. All saved networks and
    # normalization parameters must remain fixed during the three flow stages.
    paths = [run / "inference" / name for name in
             ("networks.pt", "metadata.json", "run_spec.json")]
    paths += [run / "hybrid" / name for name in
              ("reference.pt", "anchor_normalization.npy", "morph_normalization.npy", "manifest.json")]
    paths += [path for path in (run / "hybrid" / "ratios").rglob("*")
              if path.suffix in (".onnx", ".data", ".bin") or path.name == "ensemble.json"]
    for path in sorted(path for path in paths if path.is_file()):
        digest.update(str(path.relative_to(run)).encode())
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    for name in ("inference.py", "model.py", "interpolation.py", "utils.py", "sampling.py", "statistic_flows.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def generate_statistics(inference, hybrid, config, mu, *, source,
                        seed, epsilon, checkpoint=None, identity="",
                        reference_sampler=None, reference_bank="integration"):
    """Generate independent unbinned experiments at supplied physical mu values.

    Reference generation uses bad-H at (g(mu), a(mu)); simulator generation
    uses the physical model at (mu, 0). Both evaluate at nu=g(mu) and use
    uniform auxiliary observations. The signed NN gap is retained unchanged.
    """
    from sampling import sample_experiment
    mu = np.asarray(mu, float)
    metadata = json.dumps(dict(source=source, seed=seed, epsilon=epsilon,
                               identity=identity, n=len(mu), reference_bank=reference_bank), sort_keys=True)
    if checkpoint is not None and Path(checkpoint).exists():
        saved = np.load(checkpoint)
        if str(saved["metadata"]) != metadata or not np.array_equal(saved["mu"], mu):
            raise ValueError("Cached flow toys use different models/settings. Remove this cache or choose a new run.")
        return {name: saved[name] for name in saved.files if name != "metadata"}
    rng = np.random.default_rng(seed)
    response = np.asarray(inference.response(mu))
    raw, auxiliary, counts, proposal_ess = [], [], [], []
    for index, (truth, (nu, alpha)) in enumerate(zip(mu, response)):
        if source == "reference":
            sampler = hybrid if reference_sampler is None else reference_sampler
            toy = sampler.sample_anchor_experiment(float(nu), float(alpha), config, rng, epsilon=epsilon)
            good_anchors = toy["anchors"]
            proposal_ess.append(toy["proposal_ess"])
        elif source == "simulator":
            toy = sample_experiment(float(truth), 0., config, rng)
            good_anchors = hybrid.anchors(toy["x"])
            proposal_ess.append(np.nan)
        else:
            raise ValueError("source must be 'reference' or 'simulator'")
        # The frozen likelihood mixes normalized process densities after the
        # nuisance morph; the six encoder features remain the GOOD anchors.
        anchors = good_anchors
        result = inference.evaluate(anchors, toy["auxiliary"], float(nu))
        raw.append(float(np.asarray(result["statistic"])))
        auxiliary.append(toy["auxiliary"])
        counts.append(len(anchors))
        if (index + 1) % 500 == 0:
            print(f"{source}: {index + 1:,}/{len(mu):,} experiments")
    result = dict(mu=mu, response=response, raw_statistic=np.asarray(raw),
                  auxiliary=np.asarray(auxiliary), n_events=np.asarray(counts),
                  proposal_ess=np.asarray(proposal_ess))
    if checkpoint is not None:
        Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(checkpoint, metadata=metadata, **result)
    return result


def coverage_summary(mu, statistic, cdf, levels=(.68, .90, .95), bins=8):
    """Conditional-bin coverage and binomial standard errors for conservative cuts."""
    mu, statistic = np.asarray(mu), np.asarray(statistic)
    edges = np.linspace(*cdf.base.mu_range, bins + 1)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        keep = (mu >= low) & (mu <= high if high == edges[-1] else mu < high)
        for level in levels:
            critical = cdf.ppf(mu[keep], level)
            accepted = statistic[keep] <= critical
            fraction = accepted.mean() if len(accepted) else np.nan
            rows.append(dict(mu_low=low, mu_high=high, n=len(accepted), level=level,
                             coverage=fraction,
                             standard_error=np.sqrt(fraction * (1 - fraction) / len(accepted))
                             if len(accepted) else np.nan))
    return rows
