"""Signal-only known-noise training; no analytic simulator density is used here."""
import copy
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn


class SignalNCE(nn.Module):
    """Integrable unnormalized density: Gaussian envelope + bounded NN residual.

    log p_tilde = log g + bound*tanh(h/bound) + offset.
    The envelope and standardization are fitted only on optimization events.
    The scalar offset is learned by balanced noise-contrastive estimation.
    """
    def __init__(self, mean, std, width=1024, layers=4, bound=30.):
        super().__init__()
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('std', torch.as_tensor(std, dtype=torch.float32))
        self.bound = float(bound)
        blocks = []
        size = len(mean)
        for _ in range(layers):
            blocks.extend([nn.Linear(size, width), nn.SiLU()])
            size = width
        blocks.append(nn.Linear(size, 1))
        self.net = nn.Sequential(*blocks)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.offset = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        z = (x - self.mean) / self.std
        envelope = -.5 * (z.square() + math.log(2 * math.pi)).sum(-1) - self.std.log().sum()
        residual = self.net(z).squeeze(-1)
        return envelope + self.bound * torch.tanh(residual / self.bound) + self.offset


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def log_density(model, x, batch_size=8192):
    model.eval()
    device = next(model.parameters()).device
    return np.concatenate([model(torch.as_tensor(x[i:i+batch_size], device=device,
                            dtype=torch.float32)).cpu().numpy().astype(np.float64)
                           for i in range(0, len(x), batch_size)])


def signal_envelope(x, seed):
    """Same optimization-only envelope used by training and noise construction."""
    indices = np.random.default_rng(seed).permutation(len(x))[:int(.60 * len(x))]
    mean = x[indices].mean(0, dtype=np.float64).astype('float32')
    std = x[indices].std(0, dtype=np.float64).astype('float32').clip(1e-6)
    return mean, std


def train_signal(x_num, x_den, logq_num, logq_den, directory, settings, provenance,
                 device='cpu', monitor=None):
    """Resume at epoch boundaries; select only by independent validation BCE.

    Each class is split 60% optimization, 15% validation, 25% untouched holdout.
    Equal class counts give logit f(x)-log noise(x). The logq arguments retain
    their original names for compatibility but must contain the actual noise
    log density on BOTH classes (log m for mixture noise).
    Optional monitor(model, epoch) returns diagnostic scalars, never a loss.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if len(x_num) != len(x_den) or len(x_num) < 20:
        raise ValueError('Equal class pools with at least 20 events are required.')
    for x, q in [(x_num, logq_num), (x_den, logq_den)]:
        if len(x) != len(q) or not np.isfinite(x).all() or not np.isfinite(q).all():
            raise ValueError('Invalid training arrays or reference log densities.')
    if settings['batch_size'] < 2 or settings['batch_size'] % 2:
        raise ValueError('batch_size must be positive and even.')
    identity = dict(settings=settings, provenance=provenance, n_per_class=len(x_num),
                    implementation='gaussian_bounded_residual_v1', split=[.60, .15, .25])
    metadata = directory / 'training.json'
    if metadata.exists() and json.loads(metadata.read_text()) != identity:
        raise ValueError('Experiment configuration changed. Choose a new experiment name.')
    metadata.write_text(json.dumps(identity, indent=2))
    seed = settings['seed']
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    permutations = [rng.permutation(len(x_num)) for _ in range(2)]
    nt, nv = int(.60 * len(x_num)), int(.15 * len(x_num))
    train_ids = [p[:nt] for p in permutations]
    val_ids = [p[nt:nt+nv] for p in permutations]
    mean, std = signal_envelope(x_num, seed)
    architecture = dict(mean=mean.tolist(), std=std.tolist(), width=settings['width'],
                        layers=settings['layers'], bound=settings['bound'])
    model = SignalNCE(**architecture).to(device)
    optimizer = torch.optim.NAdam(model.parameters(), lr=settings['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=.5)
    best, start, history = float('inf'), 0, []
    last = directory / 'last.pt'
    if last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start, best, history = state['epoch'], state['best'], state['history']
    half = settings['batch_size'] // 2

    def epoch_loss(indices, training):
        model.train(training)
        total = 0.
        for i in range(0, len(indices[0]), half):
            ids_n, ids_d = [ids[i:i+half] for ids in indices]
            x = np.concatenate([x_num[ids_n], x_den[ids_d]])
            q = np.concatenate([logq_num[ids_n], logq_den[ids_d]])
            x = torch.as_tensor(x, dtype=torch.float32, device=device)
            q = torch.as_tensor(q, dtype=torch.float32, device=device)
            labels = torch.cat([torch.ones(len(ids_n), device=device), torch.zeros(len(ids_d), device=device)])
            with torch.set_grad_enabled(training):
                logits = model(x) - q
                loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite BCE; last completed epoch is preserved.')
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                    optimizer.step()
            total += loss.item() * len(x)
        return total / (2 * len(indices[0]))

    for epoch in range(start, settings['epochs']):
        # Epoch-specific shuffle makes interruption/resumption reproducible.
        shuffle = np.random.default_rng(seed + epoch + 1)
        lr = optimizer.param_groups[0]['lr']
        training_loss = epoch_loss([shuffle.permutation(p) for p in train_ids], True)
        validation_loss = epoch_loss(val_ids, False)
        history.append(dict(epoch=epoch+1, learning_rate=lr, train_bce=training_loss, val_bce=validation_loss))
        if monitor is not None:
            with torch.no_grad():
                diagnostics = monitor(model, epoch+1)
            if set(diagnostics) & set(history[-1]):
                raise ValueError('Monitor keys cannot replace training history fields.')
            history[-1].update(diagnostics)
        if validation_loss < best:
            best = validation_loss
            atomic_save(dict(model=copy.deepcopy(model.state_dict()), architecture=architecture,
                             epoch=epoch+1, val_bce=best, identity=identity), directory / 'best.pt')
        scheduler.step()
        atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                         scheduler=scheduler.state_dict(), epoch=epoch+1, best=best,
                         history=history), last)
        (directory / 'history.json').write_text(json.dumps(history, indent=2))
        print(f'Epoch {epoch+1:03d}: LR={lr:.3g}, train={training_loss:.6f}, val={validation_loss:.6f}', flush=True)
    # Repair the human-readable history if interruption followed the last save.
    (directory / 'history.json').write_text(json.dumps(history, indent=2))
    best_state = torch.load(directory / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(best_state['model'])
    model.eval()
    print(f"Using epoch {best_state['epoch']}, validation BCE={best_state['val_bce']:.6f}")
    return model, history
