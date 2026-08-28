"""Configuration contract for Exercise 9c's deliberately hybrid SBIBM study.

The contract is dependency-free on purpose.  Task launchers, the common runner,
and the aggregate notebook all import this file, so a run tag cannot silently
refer to a different flow, ratio-bank, or evaluation campaign.
"""

from __future__ import annotations

import hashlib
import json


ALL_TASKS = (
    "gaussian_linear",
    "gaussian_linear_uniform",
    "gaussian_mixture",
    "two_moons",
    "slcp",
    "slcp_distractors",
    "bernoulli_glm",
    "bernoulli_glm_raw",
    "sir",
    "lotka_volterra",
)

TASK_TITLES = {
    "gaussian_linear": "Gaussian Linear",
    "gaussian_linear_uniform": "Gaussian Linear Uniform",
    "gaussian_mixture": "Gaussian Mixture",
    "two_moons": "Two Moons",
    "slcp": "SLCP",
    "slcp_distractors": "SLCP Distractors",
    "bernoulli_glm": "Bernoulli GLM",
    "bernoulli_glm_raw": "Bernoulli GLM Raw",
    "sir": "SIR",
    "lotka_volterra": "Lotka--Volterra",
}

TASK_STEMS = {
    "gaussian_linear": "GaussianLinear",
    "gaussian_linear_uniform": "GaussianLinearUniform",
    "gaussian_mixture": "GaussianMixture",
    "two_moons": "TwoMoons",
    "slcp": "SLCP",
    "slcp_distractors": "SLCPDistractors",
    "bernoulli_glm": "BernoulliGLM",
    "bernoulli_glm_raw": "BernoulliGLMRaw",
    "sir": "SIR",
    "lotka_volterra": "LotkaVolterra",
}

TASK_NOTEBOOKS = {
    task: f"Exercise_9c_SBIBM_hybrid_{stem}.ipynb"
    for task, stem in TASK_STEMS.items()
}

METHOD_FLOW = "simple_flow"
METHOD_MULTICLASS = "hybrid_multiclass"
METHOD_BINARY = "hybrid_binary"
METHODS = (METHOD_FLOW, METHOD_MULTICLASS, METHOD_BINARY)
METHOD_LABELS = {
    METHOD_FLOW: "Intermediate flow (direct)",
    METHOD_MULTICLASS: "Hybrid: one 3-class ratio",
    METHOD_BINARY: "Hybrid: two binary ratios",
}
METHOD_COLORS = {
    METHOD_FLOW: "#4C78A8",
    METHOD_MULTICLASS: "#F58518",
    METHOD_BINARY: "#54A24B",
}

DEFAULT_SEED = 31082026
MAX_DERIVED_SEED_OFFSET = 2_000_000
MAX_BASE_SEED = 2**32 - 1 - MAX_DERIVED_SEED_OFFSET

INITIAL_LEARNING_RATE = 1.0e-4
MINIMUM_LEARNING_RATE = 1.0e-9
LEARNING_RATE_DROP_FACTOR = 0.1

METRIC_SCHEMA = "paired_reference_and_method_contrasts_v1_9c"
CAMPAIGN_SCHEMA = "intermediate_flow_fresh_large_ratio_banks_multi_vs_binary_v2"
JANA_PAPER_COMMIT = "6cbbc94faf0aa85147986f7f9516d13a52551bd4"


# One deliberately intermediate proposal flow is shared by q_phi and q_eta.
# Alternating masks already exchange transformed and conditioning coordinates.
# The retired fixed reversals must remain disabled; learned LU maps provide
# additional coordinate mixing without recreating the even-dimensional mask
# cancellation that affected the historical Exercise-9 topology.
FLOW_MODEL_CONFIG = {
    "n_coupling_layers": 8,
    "hidden_features": 128,
    "hidden_layers": 2,
    "spline_num_bins": 8,
    "spline_tail_bound": 5.0,
    "dropout_probability": 0.0,
    "use_layer_permutations": False,
    "linear_mixing": "lu",
}

FLOW_TRAINING_POLICY = {
    "learning_rate": INITIAL_LEARNING_RATE,
    "min_learning_rate": MINIMUM_LEARNING_RATE,
    "lr_scheduler": "plateau",
    "lr_scheduler_factor": 0.3,
    "lr_scheduler_patience": 10,
    "validation_fraction": 0.2,
    "patience": 30,
    "print_every": 10,
    "gradient_clip": 5.0,
    "weight_decay": 0.0,
}


# Classifier training is step based.  Bank size controls fresh simulator
# coverage; it does not multiply an epoch count and accidentally dominate the
# compute budget.  PAPER is the intended scientific campaign.  TUTORIAL is a
# useful Colab-sized preview; EXTREME tests the saturation regime discussed in
# the accompanying notebook.
PROFILES = {
    "SMOKE": dict(
        flow_simulations=512,
        ratio_train_simulations=1_024,
        ratio_validation_simulations=512,
        audit_simulations=512,
        flow_members=1,
        flow_epochs=2,
        flow_batch_size=64,
        classifier_members=1,
        classifier_width=96,
        classifier_layers=2,
        classifier_row_batch_budget=128,
        classifier_steps=25,
        classifier_validation_interval=5,
        classifier_patience_validations=6,
        n_proposal=1_500,
        n_posterior=512,
        observations=[1],
        observation_jitters=4,
        predictive_candidates=4,
        predictive_samples=128,
        predictive_reference_calls=128,
        metric_max_samples=256,
    ),
    "TUTORIAL": dict(
        flow_simulations=10_000,
        ratio_train_simulations=100_000,
        ratio_validation_simulations=20_000,
        audit_simulations=20_000,
        flow_members=1,
        flow_epochs=200,
        flow_batch_size=128,
        classifier_members=4,
        classifier_width=1_024,
        classifier_layers=4,
        classifier_row_batch_budget=1_024,
        classifier_steps=1_500,
        classifier_validation_interval=100,
        classifier_patience_validations=8,
        n_proposal=30_000,
        n_posterior=5_000,
        observations=[1, 2, 3],
        observation_jitters=8,
        predictive_candidates=16,
        predictive_samples=1_000,
        predictive_reference_calls=1_000,
        metric_max_samples=2_000,
    ),
    "PAPER": dict(
        flow_simulations=10_000,
        ratio_train_simulations=1_000_000,
        ratio_validation_simulations=100_000,
        audit_simulations=100_000,
        flow_members=1,
        flow_epochs=200,
        flow_batch_size=128,
        classifier_members=10,
        classifier_width=1_024,
        classifier_layers=4,
        classifier_row_batch_budget=1_024,
        classifier_steps=5_000,
        classifier_validation_interval=200,
        classifier_patience_validations=10,
        n_proposal=150_000,
        n_posterior=10_000,
        observations=list(range(1, 11)),
        observation_jitters=32,
        predictive_candidates=32,
        predictive_samples=2_000,
        predictive_reference_calls=2_000,
        metric_max_samples=5_000,
    ),
    "EXTREME": dict(
        flow_simulations=10_000,
        ratio_train_simulations=5_000_000,
        ratio_validation_simulations=200_000,
        audit_simulations=200_000,
        flow_members=1,
        flow_epochs=200,
        flow_batch_size=128,
        classifier_members=10,
        classifier_width=1_024,
        classifier_layers=4,
        classifier_row_batch_budget=1_024,
        classifier_steps=10_000,
        classifier_validation_interval=250,
        classifier_patience_validations=12,
        n_proposal=300_000,
        n_posterior=10_000,
        observations=list(range(1, 11)),
        observation_jitters=32,
        predictive_candidates=64,
        predictive_samples=2_000,
        predictive_reference_calls=2_000,
        metric_max_samples=5_000,
    ),
}


def normalize_profile(profile: str) -> str:
    value = str(profile).upper()
    if value not in PROFILES:
        raise ValueError(f"Profile must be one of {tuple(PROFILES)}; got {profile!r}.")
    return value


def validate_seed(seed: int) -> int:
    value = int(seed)
    if not 0 <= value <= MAX_BASE_SEED:
        raise ValueError(
            f"Seed must lie in [0, {MAX_BASE_SEED}] so all derived seeds remain valid."
        )
    return value


def _signature_payload(profile: str) -> dict:
    profile = normalize_profile(profile)
    return {
        "campaign_schema": CAMPAIGN_SCHEMA,
        "metric_schema": METRIC_SCHEMA,
        "profile": profile,
        "campaign": PROFILES[profile],
        "methods": METHODS,
        "flow_role": "normalized_sampleable_support_complete_proposal_not_precision_model",
        "flow_topology": FLOW_MODEL_CONFIG,
        "flow_training": FLOW_TRAINING_POLICY,
        "flow_objective": "conditional_nll_only",
        "banks": "four_persistent_genuine_simulator_banks_disjoint_by_seed_and_role",
        "ratio_topology": "plain_relu_mlp_4x1024_no_dropout_no_weight_decay_no_layernorm",
        "ratio_objectives": (
            "equal_prior_multiclass_ce_S_P_L_and_equal_prior_binary_ce_S_P_plus_S_L"
        ),
        "ratio_training": "step_based_adam_progressive_lr_best_fresh_validation_ce",
        "ratio_deployment": "arithmetic_mean_memberwise_direct_float64_softmax_quotients",
        "posterior": "shared_qphi_candidates_direct_or_self_normalized_ratio_resampling",
        "predictive": "direct_qeta_or_finite_K_ratio_SIR",
        "diagnostics": "fresh_audit_only_never_checkpoint_selection",
        "jana_paper_commit": JANA_PAPER_COMMIT,
    }


def campaign_signature(profile: str) -> str:
    digest = hashlib.sha256(
        json.dumps(_signature_payload(profile), sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return f"sha256-{digest}"


def campaign_run_tag(profile: str, seed: int) -> str:
    profile = normalize_profile(profile)
    seed = validate_seed(seed)
    cfg = PROFILES[profile]
    return (
        f"v2_{profile.lower()}_seed{seed}_flow{cfg['flow_simulations']}_"
        f"ratio{cfg['ratio_train_simulations']}_ens{cfg['classifier_members']}_"
        f"steps{cfg['classifier_steps']}_cfg{campaign_signature(profile)}_"
        "multi_vs_binary"
    )


def aggregate_run_tag(profile: str, seeds) -> str:
    profile = normalize_profile(profile)
    values = tuple(dict.fromkeys(validate_seed(seed) for seed in seeds))
    if not values:
        raise ValueError("At least one aggregate seed is required.")
    if len(values) == 1:
        return campaign_run_tag(profile, values[0])
    return (
        f"v2_{profile.lower()}_seeds{'-'.join(map(str, values))}_"
        f"cfg{campaign_signature(profile)}_aggregate"
    )


def expected_status(task: str, profile: str, seed: int) -> dict:
    if task not in ALL_TASKS:
        raise ValueError(f"Unknown SBIBM task {task!r}.")
    profile = normalize_profile(profile)
    seed = validate_seed(seed)
    cfg = PROFILES[profile]
    return {
        "task": task,
        "profile": profile,
        "seed": seed,
        "run_tag": campaign_run_tag(profile, seed),
        "campaign_signature": campaign_signature(profile),
        "methods": list(METHODS),
        "observations": list(cfg["observations"]),
    }
