"""Proposal tuning for the same-sample normalized Exercise 6 scan."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
import torch

from utils_nf import _save_flow, checkpoint_path, flow_log_prob_x, load_flow


def mixture_log_weights(log_g_over_q, epsilon):
    """Log q/g_epsilon, with both q and g conditioned on PRESEL."""
    if epsilon == 1.0:
        return np.zeros_like(log_g_over_q, dtype=np.float64)
    return -np.logaddexp(
        np.log(epsilon), np.log1p(-epsilon) + np.asarray(log_g_over_q)
    )


def epsilon_diagnostics(log_g_over_q, scan_amplitude, q0_amplitude, epsilons):
    """Leading variance proxies on a fixed independent reference sample."""
    scan_a2 = np.asarray(scan_amplitude, dtype=np.float64) ** 2
    q0_a2 = np.asarray(q0_amplitude, dtype=np.float64) ** 2
    rows = []
    for epsilon in epsilons:
        rho = np.exp(mixture_log_weights(log_g_over_q, epsilon))
        scan_j = float(np.mean(rho * scan_a2))
        q0_j = float(np.mean(rho * q0_a2))
        rows.append({
            "epsilon": float(epsilon),
            "scan_objective": scan_j,
            "q0_objective": q0_j,
            "predicted_scan_gain": float(np.mean(scan_a2) / scan_j),
            "predicted_q0_gain": float(np.mean(q0_a2) / q0_j),
            "q_over_g_q99.9": float(np.quantile(rho, 0.999)),
            "q_over_g_max": float(np.max(rho)),
            "hard_bound": 1.0 / epsilon,
        })
    return pd.DataFrame(rows)


def mix_quadratures(reference, proposal, uniforms, epsilon):
    """Couple epsilon choices using common q/g samples and mixture uniforms."""
    use_reference = uniforms < epsilon
    signal = np.where(use_reference, reference["signal"], proposal["signal"])
    background = np.where(
        use_reference, reference["background"], proposal["background"]
    )
    log_ratio = np.where(
        use_reference, reference["log_g_over_q"], proposal["log_g_over_q"]
    )
    return signal, background, mixture_log_weights(log_ratio, epsilon)


def conditional_variance_terms(log_raw_ratio, amplitude_squared, epsilon):
    """Objective and dJ/dlog(g_raw), including its PRESEL normalization."""
    log_raw_ratio = np.asarray(log_raw_ratio, dtype=np.float64)
    # mean(t)=1 estimates the conditional g/q ratio; no raw-ratio clipping.
    t = np.exp(log_raw_ratio - logsumexp(log_raw_ratio) + np.log(len(log_raw_ratio)))
    denominator = epsilon + (1.0 - epsilon) * t
    b = (1.0 - epsilon) * amplitude_squared * t / denominator**2
    coefficient = np.mean(b) * t - b
    return float(np.mean(amplitude_squared / denominator)), coefficient


def finetune_variance(
    flow_pack, train_values, train_log_q, train_amplitude,
    validation_values, validation_log_q, validation_amplitude,
    *, epsilon, model_dir, epochs=20, batch_size=4096,
    learning_rate=1.0e-5, patience=5,
):
    """Two-pass full-sample gradients; select against independent reference data."""
    path = checkpoint_path(
        "asimov_importance", model_dir, flow_pack["model_config"]["flow_type"]
    )
    flow = flow_pack["flow"]
    device = next(flow.parameters()).device
    if path.exists():
        return load_flow(
            "asimov_importance", model_dir=model_dir,
            flow_type=flow_pack["model_config"]["flow_type"], device=device,
            expected_features=flow_pack["features"],
        )
    scaler = flow_pack["scaler"]
    x_scaled = scaler.transform(train_values)
    # A fixed common scale makes Adam's numerical scale well conditioned.
    scale_squared = float(np.mean(np.asarray(train_amplitude, dtype=np.float64)**2))
    train_a2 = np.asarray(train_amplitude, dtype=np.float64)**2 / scale_squared
    validation_a2 = np.asarray(validation_amplitude, dtype=np.float64)**2 / scale_squared
    optimizer = torch.optim.Adam(flow.parameters(), lr=learning_rate)
    flow.eval()  # Deterministic between the two passes; gradients remain enabled.

    def objective(values, log_q, a2):
        log_g = flow_log_prob_x(flow_pack, values, batch_size=batch_size)
        return conditional_variance_terms(
            np.asarray(log_g, dtype=np.float64) - log_q, a2, epsilon
        )

    best_validation, _ = objective(validation_values, validation_log_q, validation_a2)
    _save_flow(path, flow, scaler, flow_pack["features"], flow_pack["model_config"])
    history = [{"epoch": 0, "train_objective": np.nan,
                "validation_objective": best_validation}]
    stale = 0
    for epoch in range(1, epochs + 1):
        # First pass: all global normalizers and derivative coefficients.
        train_j, coefficient = objective(train_values, train_log_q, train_a2)
        optimizer.zero_grad(set_to_none=True)
        # Second pass: one accumulated gradient, one step over the full sample.
        for start in range(0, len(x_scaled), batch_size):
            stop = start + batch_size
            batch = torch.as_tensor(x_scaled[start:stop], device=device)
            log_g = flow.log_prob(batch) + scaler.log_det_x_to_z_standardization
            c = torch.as_tensor(coefficient[start:stop], dtype=log_g.dtype, device=device)
            ((c * log_g).sum() / len(x_scaled)).backward()
        optimizer.step()
        validation_j, _ = objective(validation_values, validation_log_q, validation_a2)
        history.append({"epoch": epoch, "train_objective": train_j,
                        "validation_objective": validation_j})
        print(f"  variance epoch {epoch:03d}: train J={train_j:.6g}, "
              f"validation J={validation_j:.6g}", flush=True)
        if np.isfinite(validation_j) and validation_j < best_validation:
            best_validation = validation_j
            _save_flow(path, flow, scaler, flow_pack["features"], flow_pack["model_config"])
            stale = 0
        else:
            stale += 1
        pd.DataFrame(history).to_csv(Path(model_dir) / "variance_history.csv", index=False)
        if stale >= patience:
            break
    return load_flow(
        "asimov_importance", model_dir=model_dir,
        flow_type=flow_pack["model_config"]["flow_type"], device=device,
        expected_features=flow_pack["features"],
    )
