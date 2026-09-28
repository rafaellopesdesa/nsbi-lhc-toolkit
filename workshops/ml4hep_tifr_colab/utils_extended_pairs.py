"""Small numerical and neural helpers for Exercise 13 (extended PAIRS).

The event encoder sees only raw event features. Likelihood ratios are used
solely for population fitting and for numerical teacher labels. The temporary
categorical posterior head is discarded before full-experiment inference.
"""

from copy import deepcopy

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
import torch
from torch import nn
from torch.nn import functional as F


def exact_loglik(q, nu, lam_s):
    """Extended unbinned log likelihood, relative to its value at nu=0.

    q_i = lambda_s p_s(x_i)/(lambda_b p_b(x_i)). Terms independent of
    nu cancel. A vector of candidate nu values is supported.
    """
    q = np.asarray(q, dtype=np.float64)
    nu = np.asarray(nu, dtype=np.float64)
    return -nu * lam_s + np.log1p(nu[..., None] * q).sum(axis=-1)


def exact_fit(q, lam_s):
    """Global concave-likelihood maximum on nu>=0, including the boundary."""
    q = np.asarray(q, dtype=np.float64)

    def score(nu):
        return -lam_s + np.sum(q / (1.0 + nu * q))

    if score(0.0) <= 0.0:
        return 0.0
    upper = 1.0
    while score(upper) > 0.0:
        upper *= 2.0
    return brentq(score, 0.0, upper, xtol=1e-11)


def exact_statistic(q, nu, lam_s, nu_hat=None):
    """Two-sided likelihood-ratio statistic evaluated at candidate nu."""
    if nu_hat is None:
        nu_hat = exact_fit(q, lam_s)
    return np.maximum(
        2.0 * (exact_loglik(q, nu_hat, lam_s) - exact_loglik(q, nu, lam_s)),
        0.0,
    )


def population_loglik(q_sig, q_bkg, mu, nu, lam_s, lam_b,
                       signal_weights=None, background_weights=None):
    """Expected surrogate log likelihood under an empirical simulator bank.

    mu and nu may be equally sized arrays. Each process bank represents its
    normalized post-selection process distribution. Optional weights represent
    weighted reference events; otherwise process events have equal weights.
    """
    mu, nu = np.broadcast_arrays(np.asarray(mu), np.asarray(nu))
    return (
        -nu * lam_s
        + mu * lam_s * np.average(np.log1p(nu[..., None] * q_sig),
                                  axis=-1, weights=signal_weights)
        + lam_b * np.average(np.log1p(nu[..., None] * q_bkg),
                             axis=-1, weights=background_weights)
    )


def exact_response(q_sig, q_bkg, mu, lam_s, lam_b,
                   signal_weights=None, background_weights=None):
    """Scalar population response used only as a validation reference."""
    q_sig = np.asarray(q_sig, dtype=np.float64)
    q_bkg = np.asarray(q_bkg, dtype=np.float64)

    def score(nu):
        return (
            -lam_s
            + mu * lam_s * np.average(q_sig / (1.0 + nu * q_sig),
                                      weights=signal_weights)
            + lam_b * np.average(q_bkg / (1.0 + nu * q_bkg),
                                 weights=background_weights)
        )

    if score(0.0) <= 0.0:
        return 0.0
    upper = 1.0
    while score(upper) > 0.0:
        upper *= 2.0
    return brentq(score, 0.0, upper, xtol=1e-11)


def _mlp(n_in, n_out, width):
    return nn.Sequential(
        nn.Linear(n_in, width), nn.SiLU(),
        nn.Linear(width, width), nn.SiLU(), nn.Linear(width, n_out),
    )


class PairEncoder(nn.Module):
    """An event network; mean pooling is performed by the caller."""

    def __init__(self, n_features, embed_dim=24, width=64,
                 x_mean=None, x_std=None):
        super().__init__()
        self.embed_dim = embed_dim
        mean = np.zeros(n_features) if x_mean is None else x_mean
        std = np.ones(n_features) if x_std is None else x_std
        self.register_buffer("x_mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("x_std", torch.as_tensor(std, dtype=torch.float32))
        self.net = _mlp(n_features, embed_dim, width)

    def forward(self, x):
        return self.net((x - self.x_mean) / self.x_std)


class PairPosterior(nn.Module):
    """Categorical NPE on a discrete mixture-fraction design prior.

    The class index labels the generating fraction, not the event process.
    Domain identifies the simulator or hybrid family only in this temporary
    posterior head. It is never an input to the shared event encoder.
    """

    def __init__(self, encoder, n_classes=32, n_domains=2, width=64):
        super().__init__()
        self.encoder = encoder
        self.n_domains = n_domains
        self.posterior = _mlp(encoder.embed_dim + 1 + n_domains, n_classes, width)

    def forward(self, x, n, domain):
        event_embedding = self.encoder(x)
        mask = torch.arange(x.shape[1], device=x.device)[None, :] < n[:, None]
        pooled = (event_embedding * mask[..., None]).sum(dim=1) / n[:, None]
        context = torch.cat([
            pooled, n[:, None].float(),
            F.one_hot(domain, self.n_domains).float(),
        ], dim=-1)
        return self.posterior(context)


def _tensor_dict(data):
    return {key: torch.as_tensor(value) for key, value in data.items()}


def _supervised_train(model, train, val, loss_fn, epochs, batch_size, lr,
                      patience, seed, device):
    """Train on CPU-held arrays and restore the best validation checkpoint."""
    torch.manual_seed(seed)
    model.to(device)
    train, val = _tensor_dict(train), _tensor_dict(val)
    n_train, n_val = len(next(iter(train.values()))), len(next(iter(val.values())))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)

    def validation_loss():
        model.eval()
        total = 0.0
        with torch.no_grad():
            for start in range(0, n_val, batch_size):
                batch = {key: value[start:start + batch_size].to(device)
                         for key, value in val.items()}
                total += float(loss_fn(model, batch)) * len(next(iter(batch.values())))
        return total / n_val

    best_loss = validation_loss()
    best_state = deepcopy(model.state_dict())
    history = [{"epoch": 0, "train": None, "val": best_loss}]
    bad_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(n_train, generator=generator)
        total = 0.0
        for start in range(0, n_train, batch_size):
            index = order[start:start + batch_size]
            batch = {key: value[index].to(device) for key, value in train.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite supervised training loss.")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(index)
        val_loss = validation_loss()
        history.append({"epoch": epoch, "train": total / n_train, "val": val_loss})
        if val_loss < best_loss:
            best_loss, best_state, bad_epochs = val_loss, deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch {epoch:3d}: train={total / n_train:.5g}, val={val_loss:.5g}")
        if bad_epochs >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return history


def train_pairs(model, train, val, epochs=30, batch_size=512, lr=1e-3,
                patience=6, seed=13, device="cpu"):
    """Data keys: x:(B,2,d), n:(B,), domain:(B,), label:(B,)."""
    keys = ("x", "n", "domain", "label")
    train, val = ({key: data[key] for key in keys} for data in (train, val))

    def loss_fn(net, batch):
        logits = net(batch["x"].float(), batch["n"].long(), batch["domain"].long())
        return F.cross_entropy(logits, batch["label"].long())

    return _supervised_train(model, train, val, loss_fn, epochs, batch_size,
                             lr, patience, seed, device)


def reference_fraction_posterior(q, n, fractions, ratio_scale):
    """Exact fraction-class posterior for reference singletons and pairs.

    q has shape (B,2), n has shape (B,), and fractions has shape (K,).
    ratio_scale*q must equal p_s(x)/p_b(x) for the normalized reference
    distribution that generated these sets. The class prior is uniform.
    Padded events are excluded; no information from their q values is used.
    """
    q = np.asarray(q, dtype=np.float64)
    n = np.asarray(n)
    fractions = np.asarray(fractions, dtype=np.float64)
    ratio = ratio_scale * q
    mixture = (1.0 - fractions) + ratio[..., None] * fractions
    mask = np.arange(q.shape[1])[None, :] < n[:, None]
    mixture = np.where(mask[..., None], mixture, 1.0)
    log_likelihood = np.log(mixture).sum(axis=1)
    return np.exp(log_likelihood - logsumexp(log_likelihood, axis=1, keepdims=True))


@torch.no_grad()
def encode_events(encoder, x, batch_size=65536, device="cpu"):
    """Encode once; downstream pooling uses these individual event vectors."""
    encoder.to(device).eval()
    if len(x) == 0:
        return np.zeros((0, encoder.embed_dim), dtype=np.float32)
    chunks = []
    for start in range(0, len(x), batch_size):
        batch = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32,
                                device=device)
        chunks.append(encoder(batch).cpu().numpy())
    return np.concatenate(chunks, axis=0)


class ResponseNet(nn.Module):
    """A deterministic response, initialized to g(mu)=mu on the physical range."""

    def __init__(self, mu_max=3.0, width=32):
        super().__init__()
        self.mu_max = float(mu_max)
        self.correction = _mlp(1, 1, width)
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, mu):
        return F.relu(mu + self.correction((mu / self.mu_max)[..., None]).squeeze(-1))


def train_response(model, q_sig, q_bkg, lam_s, lam_b, mu_max=3.0,
                   epochs=200, steps_per_epoch=10, mu_batch=64, event_batch=2048,
                   lr=1e-3, patience=20, seed=13, device="cpu",
                   val_q_sig=None, val_q_bkg=None):
    """Stochastic population fitting without numerical g(mu) training labels.

    Validation uses a fixed mu grid and fixed independent event arrays when
    provided. Training evaluates the linear term with the full-bank mean q;
    E[log(1+nu*q)] = nu*E[q] + E[log(1+nu*q)-nu*q]. Only the last expectation
    is estimated with minibatches. This control variate preserves the empirical
    objective while reducing low-purity gradient noise.
    Objective arithmetic is float64; the network itself remains float32.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(device)
    q_sig, q_bkg = np.asarray(q_sig), np.asarray(q_bkg)
    if val_q_sig is None:
        val_q_sig = q_sig[rng.integers(len(q_sig), size=event_batch * 4)]
    if val_q_bkg is None:
        val_q_bkg = q_bkg[rng.integers(len(q_bkg), size=event_batch * 4)]
    validation_mu = torch.linspace(0.0, mu_max, 65, device=device)
    validation_sig = torch.as_tensor(val_q_sig, dtype=torch.float64, device=device)
    validation_bkg = torch.as_tensor(val_q_bkg, dtype=torch.float64, device=device)
    objective_scale = lam_s ** 2 / (lam_b + mu_max * lam_s)
    mean_sig, mean_bkg = float(q_sig.mean()), float(q_bkg.mean())

    def loss_fn(mu, sig, bkg, use_control_variate=False):
        nu = model(mu).double()
        sig_term = torch.log1p(nu[:, None] * sig)
        bkg_term = torch.log1p(nu[:, None] * bkg)
        if use_control_variate:
            sig_term = sig_term - nu[:, None] * sig + nu[:, None] * mean_sig
            bkg_term = bkg_term - nu[:, None] * bkg + nu[:, None] * mean_bkg
        expected = (
            -nu * lam_s
            + mu.double() * lam_s * sig_term.mean(dim=-1)
            + lam_b * bkg_term.mean(dim=-1)
        )
        return -expected.mean() / objective_scale

    def validation_loss():
        model.eval()
        with torch.no_grad():
            return float(loss_fn(validation_mu, validation_sig, validation_bkg))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = validation_loss()
    best_state = deepcopy(model.state_dict())
    history = [{"epoch": 0, "train": None, "val": best_loss}]
    bad_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for _ in range(steps_per_epoch):
            mu = torch.as_tensor(rng.uniform(0, mu_max, mu_batch),
                                 dtype=torch.float32, device=device)
            sig = torch.as_tensor(q_sig[rng.integers(len(q_sig), size=event_batch)],
                                  dtype=torch.float64, device=device)
            bkg = torch.as_tensor(q_bkg[rng.integers(len(q_bkg), size=event_batch)],
                                  dtype=torch.float64, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(mu, sig, bkg, use_control_variate=True)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite population training loss.")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        val_loss = validation_loss()
        history.append({"epoch": epoch, "train": float(np.mean(losses)), "val": val_loss})
        if val_loss < best_loss:
            best_loss, best_state, bad_epochs = val_loss, deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
        if epoch == 1 or epoch % 20 == 0:
            print(f"response epoch {epoch:3d}: val={val_loss:.6g}")
        if bad_epochs >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return history


class FastHeads(nn.Module):
    """One shared embedding trunk with estimator and queried-statistic heads.

    All inputs are experiment embeddings and the tested surrogate coordinate.
    The domain and generating parameter are deliberately absent.
    """

    def __init__(self, input_dim, width=96):
        super().__init__()
        self.register_buffer("z_mean", torch.zeros(input_dim))
        self.register_buffer("z_std", torch.ones(input_dim))
        self.register_buffer("nu_scale", torch.tensor(1.0))
        self.trunk = _mlp(input_dim, width, width)
        self.estimator = nn.Linear(width, 1)
        self.statistic = _mlp(width + 1, 1, width)

    def fit_scaler(self, z, nu_hat):
        z = torch.as_tensor(z, dtype=torch.float32, device=self.z_mean.device)
        nu_hat = torch.as_tensor(nu_hat, dtype=torch.float32, device=self.nu_scale.device)
        self.z_mean.copy_(z.mean(dim=0))
        self.z_std.copy_(z.std(dim=0, unbiased=False).clamp_min(1e-6))
        self.nu_scale.copy_(nu_hat.std(unbiased=False).clamp_min(0.1))

    def forward(self, z, nu):
        h = self.trunk((z - self.z_mean) / self.z_std)
        nu_hat = F.softplus(self.estimator(h).squeeze(-1)) * self.nu_scale
        if nu.ndim == 1:
            context = torch.cat([h, (nu / self.nu_scale)[:, None]], dim=-1)
        else:
            h = h[:, None, :].expand(-1, nu.shape[1], -1)
            context = torch.cat([h, (nu / self.nu_scale)[..., None]], dim=-1)
        root_t = F.softplus(self.statistic(context).squeeze(-1))
        return nu_hat, root_t.square()


class RefinedHeads(FastHeads):
    """Flexible statistic heads with an estimator that can reach zero.

    signed_root learns sign(nu - teacher_nu_hat)*sqrt(T) with a linear
    output; direct learns T with a softplus output. Neither mode imposes a
    quadratic likelihood or ties the minimum to the separate estimator.
    FastHeads is kept unchanged so earlier checkpoints remain loadable.
    """

    def __init__(self, input_dim, width=96, mode="signed_root"):
        super().__init__(input_dim, width)
        if mode not in ("signed_root", "direct"):
            raise ValueError("mode must be 'signed_root' or 'direct'")
        self.mode = mode
        # Start on the active side of ReLU, while allowing fitted boundary zeros.
        nn.init.constant_(self.estimator.bias, 1.0)

    def forward_targets(self, z, nu):
        """Return the estimator and the quantity used as a regression target."""
        h = self.trunk((z - self.z_mean) / self.z_std)
        nu_hat = F.relu(self.estimator(h).squeeze(-1)) * self.nu_scale
        if nu.ndim == 1:
            context = torch.cat([h, (nu / self.nu_scale)[:, None]], dim=-1)
        else:
            h = h[:, None, :].expand(-1, nu.shape[1], -1)
            context = torch.cat([h, (nu / self.nu_scale)[..., None]], dim=-1)
        raw = self.statistic(context).squeeze(-1)
        target = raw if self.mode == "signed_root" else F.softplus(raw)
        return nu_hat, target

    def forward(self, z, nu):
        nu_hat, target = self.forward_targets(z, nu)
        t = target.square() if self.mode == "signed_root" else target
        return nu_hat, t


def train_heads(model, train, val, epochs=80, batch_size=256, lr=1e-3,
                patience=10, seed=13, device="cpu"):
    """Data keys: z:(B,d), nu_hat:(B,), nu:(B,Q), t:(B,Q).

    Train/validation splitting must be by complete experiment, before queries
    are expanded. sqrt(T) regression reduces domination by extreme T labels.
    """
    keys = ("z", "nu_hat", "nu", "t")
    train, val = ({key: data[key] for key in keys} for data in (train, val))
    model.fit_scaler(train["z"], train["nu_hat"])

    def loss_fn(net, batch):
        nu_hat, t = net(batch["z"].float(), batch["nu"].float())
        estimator_loss = F.mse_loss(nu_hat / net.nu_scale,
                                    batch["nu_hat"].float() / net.nu_scale)
        statistic_loss = F.mse_loss(torch.sqrt(t + 1e-8),
                                    torch.sqrt(batch["t"].float() + 1e-8))
        return estimator_loss + statistic_loss

    return _supervised_train(model, train, val, loss_fn, epochs, batch_size,
                             lr, patience, seed, device)


def train_refined_heads(model, train, val, epochs=400, batch_size=256,
                        lr=1e-3, patience=70, lr_patience=20, seed=13,
                        device="cpu", lr_factor=0.3, min_lr=1e-5):
    """Train refined heads on cached experiments and restore the best weights.

    Required keys are z, nu_hat, nu, t; other cache metadata is ignored.
    Splits are by complete experiment, and scalers use training data only.
    The loss adds scaled estimator MSE to signed-root MSE, or to smooth-L1
    loss on direct T (beta=1). The latter reduces the influence of large T
    errors without imposing a particular likelihood shape. Plateau learning
    rate reductions and early stopping use the total validation loss.
    """
    torch.manual_seed(seed)
    keys = ("z", "nu_hat", "nu", "t")
    train, val = ({key: data[key] for key in keys} for data in (train, val))
    model.to(device)
    model.fit_scaler(train["z"], train["nu_hat"])
    train, val = _tensor_dict(train), _tensor_dict(val)
    n_train, n_val = len(train["z"]), len(val["z"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=lr_factor, patience=lr_patience, min_lr=min_lr,
    )
    generator = torch.Generator().manual_seed(seed)

    def losses(batch):
        nu = batch["nu"].float()
        truth_hat, truth_t = batch["nu_hat"].float(), batch["t"].float()
        nu_hat, prediction = model.forward_targets(batch["z"].float(), nu)
        estimator = F.mse_loss(nu_hat / model.nu_scale, truth_hat / model.nu_scale)
        if model.mode == "signed_root":
            center = truth_hat if nu.ndim == 1 else truth_hat[:, None]
            target = torch.sign(nu - center) * torch.sqrt(truth_t)
            statistic = F.mse_loss(prediction, target)
        else:
            statistic = F.smooth_l1_loss(prediction, truth_t)
        return estimator + statistic, estimator, statistic

    def validation_losses():
        model.eval()
        totals = np.zeros(3)
        with torch.no_grad():
            for start in range(0, n_val, batch_size):
                batch = {key: value[start:start + batch_size].to(device)
                         for key, value in val.items()}
                totals += np.array([float(value) for value in losses(batch)]) * len(batch["z"])
        return totals / n_val

    def history_row(epoch, train_loss, val_loss, learning_rate):
        return {
            "epoch": epoch,
            "train": None if train_loss is None else float(train_loss[0]),
            "val": float(val_loss[0]),
            "train_estimator": None if train_loss is None else float(train_loss[1]),
            "train_statistic": None if train_loss is None else float(train_loss[2]),
            "val_estimator": float(val_loss[1]),
            "val_statistic": float(val_loss[2]),
            "lr": float(learning_rate),
        }

    val_loss = validation_losses()
    best_loss, best_state = float(val_loss[0]), deepcopy(model.state_dict())
    history = [history_row(0, None, val_loss, lr)]
    scheduler.step(best_loss)
    bad_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(n_train, generator=generator)
        totals = np.zeros(3)
        learning_rate = optimizer.param_groups[0]["lr"]
        for start in range(0, n_train, batch_size):
            index = order[start:start + batch_size]
            batch = {key: value[index].to(device) for key, value in train.items()}
            optimizer.zero_grad(set_to_none=True)
            batch_losses = losses(batch)
            batch_losses[0].backward()
            optimizer.step()
            totals += np.array([float(value.detach()) for value in batch_losses]) * len(index)
        train_loss = totals / n_train
        val_loss = validation_losses()
        history.append(history_row(epoch, train_loss, val_loss, learning_rate))
        scheduler.step(float(val_loss[0]))
        if val_loss[0] < best_loss:
            best_loss, best_state = float(val_loss[0]), deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if epoch == 1 or epoch % 20 == 0:
            print(f"{model.mode} epoch {epoch:3d}: val={val_loss[0]:.5g}, "
                  f"estimator={val_loss[1]:.5g}, statistic={val_loss[2]:.5g}, "
                  f"lr={learning_rate:.2g}")
        if bad_epochs >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return history


@torch.no_grad()
def predict_heads(model, z, nu, batch_size=1024, device="cpu"):
    model.to(device).eval()
    estimates, statistics = [], []
    for start in range(0, len(z), batch_size):
        z_batch = torch.as_tensor(z[start:start + batch_size],
                                  dtype=torch.float32, device=device)
        nu_batch = torch.as_tensor(nu[start:start + batch_size],
                                   dtype=torch.float32, device=device)
        nu_hat, t = model(z_batch, nu_batch)
        estimates.append(nu_hat.cpu().numpy())
        statistics.append(t.cpu().numpy())
    return np.concatenate(estimates), np.concatenate(statistics)
