"""Five-dimensional spline reference density, without preselection.

Architecture and training defaults follow Exercise 5. All transforms operate on
standardized x; log_prob includes the standardization Jacobian.
"""
from pathlib import Path
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

MODEL_DEFAULTS = dict(n_features=5, n_coupling_layers=10, hidden_features=1024,
                      hidden_layers=4, spline_num_bins=16, spline_tail_bound=5.0)
TRAIN_DEFAULTS = dict(batch_size=2048, n_epochs=70, learning_rate=1e-4,
                      validation_fraction=0.20, patience=5,
                      lr_scheduler_factor=0.2, lr_scheduler_patience=2,
                      min_learning_rate=1e-7)


def build_flow(config):
    from nflows.distributions.normal import StandardNormal
    from nflows.flows.base import Flow
    from nflows.nn.nets import ResidualNet
    from nflows.transforms.base import CompositeTransform
    from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform
    from nflows.transforms.permutations import ReversePermutation
    from nflows.utils.torchutils import create_alternating_binary_mask

    def network(n_in, n_out):
        return ResidualNet(n_in, n_out, hidden_features=config['hidden_features'],
                           num_blocks=config['hidden_layers'], activation=torch.relu,
                           dropout_probability=0., use_batch_norm=False)

    transforms = []
    for i in range(config['n_coupling_layers']):
        transforms.append(PiecewiseRationalQuadraticCouplingTransform(
            mask=create_alternating_binary_mask(config['n_features'], even=i % 2 == 0),
            transform_net_create_fn=network, num_bins=config['spline_num_bins'],
            tails='linear', tail_bound=config['spline_tail_bound'],
            apply_unconditional_transform=False))
        if i + 1 < config['n_coupling_layers']:
            transforms.append(ReversePermutation(config['n_features']))
    return Flow(CompositeTransform(transforms), StandardNormal([config['n_features']]))


class ReferenceFlow:
    def __init__(self, flow, mean, std, model_config):
        self.flow = flow.eval()
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.model_config = model_config

    @property
    def device(self):
        return next(self.flow.parameters()).device

    def save(self, path, history=None):
        torch.save(dict(state_dict=self.flow.state_dict(), mean=self.mean, std=self.std,
                        model_config=self.model_config, history=history), path)

    @classmethod
    def load(cls, path, device='cpu'):
        saved = torch.load(path, map_location=device, weights_only=False)
        flow = build_flow(saved['model_config']).to(device)
        flow.load_state_dict(saved['state_dict'])
        return cls(flow, saved['mean'], saved['std'], saved['model_config'])

    @torch.no_grad()
    def sample(self, n, seed, batch_size=65536):
        # fork_rng makes a local sampling seed without changing NN training RNG.
        devices = [self.device.index or 0] if self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seed))
            chunks = [self.flow.sample(min(batch_size, n - start)).cpu().numpy()
                      for start in range(0, n, batch_size)]
        if n == 0:
            return np.empty((0, len(self.mean)), dtype=np.float32)
        return np.concatenate(chunks) * self.std + self.mean

    @torch.no_grad()
    def log_prob(self, x, batch_size=65536):
        output = []
        for start in range(0, len(x), batch_size):
            z = (np.asarray(x[start:start + batch_size]) - self.mean) / self.std
            t = torch.as_tensor(z, dtype=torch.float32, device=self.device)
            output.append(self.flow.log_prob(t).cpu().numpy() - np.log(self.std).sum())
        return np.concatenate(output)


def train_reference(x, path, model_config=None, training_config=None,
                    device='cpu', seed=13001, reuse=True):
    """Maximum likelihood on a balanced nominal signal/background sample."""
    path = Path(path)
    if reuse and path.exists():
        return ReferenceFlow.load(path, device)
    model_config = MODEL_DEFAULTS | (model_config or {})
    settings = TRAIN_DEFAULTS | (training_config or {})
    torch.manual_seed(seed)
    permutation = np.random.default_rng(seed).permutation(len(x))
    n_val = max(1, round(settings['validation_fraction'] * len(x)))
    x_train, x_val = x[permutation[n_val:]], x[permutation[:n_val]]
    mean, std = x_train.mean(axis=0), x_train.std(axis=0)
    train = DataLoader(TensorDataset(torch.as_tensor((x_train - mean) / std)),
                       batch_size=settings['batch_size'], shuffle=True)
    val = DataLoader(TensorDataset(torch.as_tensor((x_val - mean) / std)),
                     batch_size=settings['batch_size'])
    flow = build_flow(model_config).to(device)
    optimizer = torch.optim.Adam(flow.parameters(), lr=settings['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=settings['lr_scheduler_factor'],
        patience=settings['lr_scheduler_patience'], min_lr=settings['min_learning_rate'])
    best, stale, history = np.inf, 0, []
    for epoch in range(settings['n_epochs']):
        flow.train()
        train_sum = 0.
        for (batch,) in train:
            optimizer.zero_grad()
            loss = -flow.log_prob(batch.to(device)).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(flow.parameters(), 5.)
            optimizer.step()
            train_sum += float(loss.detach()) * len(batch)
        flow.eval()
        with torch.no_grad():
            val_loss = sum(float(-flow.log_prob(batch.to(device)).sum())
                           for (batch,) in val) / len(x_val)
        history.append((train_sum / len(x_train), val_loss))
        scheduler.step(val_loss)
        print(f'flow epoch {epoch + 1:02d}: train={history[-1][0]:.5f}, val={val_loss:.5f}')
        if val_loss < best:
            best, stale = val_loss, 0
            state = copy.deepcopy(flow.state_dict())
        else:
            stale += 1
        if stale >= settings['patience']:
            break
    flow.load_state_dict(state)
    result = ReferenceFlow(flow, mean, std, model_config)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.save(path, history)
    return result
