"""Shared configuration contract for the parallel Exercise 9b SBIBM campaign.

This module deliberately has no NumPy, PyTorch, sbibm, or notebook dependency.
The task notebooks, the training runner, and the aggregate collector import the
same functions so that a tag cannot silently refer to different campaigns.
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

TASK_NOTEBOOKS = {
    "gaussian_linear": "Exercise_9b_SBIBM_GaussianLinear.ipynb",
    "gaussian_linear_uniform": "Exercise_9b_SBIBM_GaussianLinearUniform.ipynb",
    "gaussian_mixture": "Exercise_9b_SBIBM_GaussianMixture.ipynb",
    "two_moons": "Exercise_9b_SBIBM_TwoMoons.ipynb",
    "slcp": "Exercise_9b_SBIBM_SLCP.ipynb",
    "slcp_distractors": "Exercise_9b_SBIBM_SLCPDistractors.ipynb",
    "bernoulli_glm": "Exercise_9b_SBIBM_BernoulliGLM.ipynb",
    "bernoulli_glm_raw": "Exercise_9b_SBIBM_BernoulliGLMRaw.ipynb",
    "sir": "Exercise_9b_SBIBM_SIR.ipynb",
    "lotka_volterra": "Exercise_9b_SBIBM_LotkaVolterra.ipynb",
}

TASK_SHORT_LABELS = {
    "gaussian_linear": "G-Lin",
    "gaussian_linear_uniform": "G-Unif",
    "gaussian_mixture": "G-Mix",
    "two_moons": "Moons",
    "slcp": "SLCP",
    "slcp_distractors": "SLCP-D",
    "bernoulli_glm": "B-GLM",
    "bernoulli_glm_raw": "B-Raw",
    "sir": "SIR",
    "lotka_volterra": "Lotka",
}

METHOD_JANA = "jana_direct"
METHOD_HYBRID = "hybrid_ce"
METHODS = (METHOD_JANA, METHOD_HYBRID)
METHOD_LABELS = {
    METHOD_JANA: "Pure JANA (direct flows)",
    METHOD_HYBRID: "Hybrid + CE correction",
}

DEFAULT_SEED = 29082026
MAX_LEGACY_SEED_OFFSET = 100_000 * 9 + 150_000 + 10
MAX_BASE_SEED = 2**32 - 1 - MAX_LEGACY_SEED_OFFSET

INITIAL_LEARNING_RATE = 1.0e-4
MINIMUM_LEARNING_RATE = 1.0e-9
LEARNING_RATE_DROP_FACTOR = 0.1
LEARNING_RATE_STEP_EPOCHS = 40

METRIC_SCHEMA = "paired_long_reference_bandwidth_v4_9a_matched"
CAMPAIGN_SCHEMA = "plain_ce_v14_corrected_alternating_masks_parallel_tasks"
JANA_PAPER_COMMIT = "6cbbc94faf0aa85147986f7f9516d13a52551bd4"


# PAPER and TUTORIAL deliberately have the same training campaign as 9a.
# They differ only in diagnostics and evaluation Monte Carlo.
PROFILES = {
    "SMOKE": dict(
        num_simulations=512,
        n_folds=1,
        flow_members=1,
        classifier_members=1,
        flow_epochs=2,
        class_epochs=2,
        class_width=96,
        class_layers=2,
        flow_batch_size=32,
        classifier_batch_size=32,
        training_patience=2,
        norm_groups=32,
        norm_inner=4,
        bridge_groups=24,
        bridge_inner=4,
        n_proposal=1_500,
        n_posterior=512,
        observations=[1],
        observation_jitters=4,
        predictive_candidates=4,
        predictive_samples=128,
        predictive_reference_calls=128,
    ),
    "TUTORIAL": dict(
        num_simulations=10_000,
        n_folds=1,
        flow_members=4,
        classifier_members=10,
        flow_epochs=250,
        class_epochs=250,
        class_width=1024,
        class_layers=4,
        flow_batch_size=32,
        classifier_batch_size=32,
        training_patience=250,
        norm_groups=256,
        norm_inner=24,
        bridge_groups=256,
        bridge_inner=24,
        n_proposal=30_000,
        n_posterior=5_000,
        observations=[1, 2, 3],
        observation_jitters=8,
        predictive_candidates=8,
        predictive_samples=1_000,
        predictive_reference_calls=1_000,
    ),
    "PAPER": dict(
        num_simulations=10_000,
        n_folds=1,
        flow_members=4,
        classifier_members=10,
        flow_epochs=250,
        class_epochs=250,
        class_width=1024,
        class_layers=4,
        flow_batch_size=32,
        classifier_batch_size=32,
        training_patience=250,
        norm_groups=512,
        norm_inner=64,
        bridge_groups=512,
        bridge_inner=32,
        n_proposal=150_000,
        n_posterior=10_000,
        observations=list(range(1, 11)),
        observation_jitters=32,
        predictive_candidates=16,
        predictive_samples=2_000,
        predictive_reference_calls=2_000,
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
            f"Seed must lie in [0, {MAX_BASE_SEED}] so derived legacy seeds "
            "remain valid."
        )
    return value


def campaign_signature(profile: str) -> str:
    profile = normalize_profile(profile)
    payload = {
        "campaign_schema": CAMPAIGN_SCHEMA,
        "metric_schema": METRIC_SCHEMA,
        "profile": profile,
        "campaign": PROFILES[profile],
        "methods": METHODS,
        "jana_baseline": "same_frozen_qphi_qeta_direct_no_classifier_v2",
        "jana_paper_commit": JANA_PAPER_COMMIT,
        "simulation_budget": "fixed_10000_non_smoke_as_in_9a",
        "flow_topology": (
            "ex9_rqs_10x512x4_bins16_tail5_"
            "alternating_masks_no_reversal_per_member_v2"
        ),
        "flow_training": (
            "four_members_batch32_epochs250_adam_lr1e-4_step40_to1e-9_"
            "full_patience_gradient_clip5_no_weight_decay_v1"
        ),
        "posterior_target": "joint_full_latent_parameter_vector_z_given_full_x_v1",
        "likelihood_target": "joint_full_x_given_full_latent_parameter_vector_z_v1",
        "classifier_topology": "plain_relu_mlp_4x1024_three_logits_no_regularization_v1",
        "classifier_ensemble": "ten_independent_members_arithmetic_probability_ratio_mean_v1",
        "classifier_training": (
            "batch32_row_budget_epochs250_adam_lr1e-4_step40_to1e-9_"
            "full_patience_heldout_ce_v1"
        ),
        "classifier_transform": "fixed_training_column_mean_std_task_scale_only_v2",
        "classifier_objective": "equal_prior_multiclass_ce_only_v1",
        "checkpoint_objective": "heldout_multiclass_ce_only_v1",
        "normalization_role": "post_training_probability_odds_and_heldout_check_v1",
        "bridge_role": "paired_jana_and_hybrid_heldout_evidence_check_v1",
        "data_semantics": "bernoulli_sir_dequant_lv_lognormal_continuous_v1",
        "posterior_predictive": "paired_jana_direct_and_hybrid_softmax_odds_sir_v2",
        "mmd_bandwidth": "shared_reference_only_median_rbf_v1",
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return f"sha256-{digest}"


def campaign_run_tag(profile: str, seed: int) -> str:
    profile = normalize_profile(profile)
    seed = validate_seed(seed)
    campaign = PROFILES[profile]
    signature = campaign_signature(profile)
    return (
        f"v14_{profile.lower()}_seed{seed}_n{campaign['num_simulations']}_"
        f"matched_flowmix{campaign['flow_members']}_"
        f"class{campaign['classifier_members']}_"
        f"fe{campaign['flow_epochs']}_ce{campaign['class_epochs']}_"
        f"cfg{signature}_paired_jana_hybrid_ce"
    )


def aggregate_run_tag(profile: str, seeds) -> str:
    profile = normalize_profile(profile)
    values = tuple(dict.fromkeys(validate_seed(seed) for seed in seeds))
    if not values:
        raise ValueError("At least one aggregate seed is required.")
    if len(values) == 1:
        return campaign_run_tag(profile, values[0])
    return (
        f"v14_{profile.lower()}_seeds{'-'.join(map(str, values))}_"
        f"cfg{campaign_signature(profile)}_aggregate"
    )


def expected_status(task: str, profile: str, seed: int) -> dict:
    if task not in ALL_TASKS:
        raise ValueError(f"Unknown SBIBM task {task!r}.")
    profile = normalize_profile(profile)
    seed = validate_seed(seed)
    campaign = PROFILES[profile]
    return {
        "task": task,
        "profile": profile,
        "seed": seed,
        "run_tag": campaign_run_tag(profile, seed),
        "campaign_schema": CAMPAIGN_SCHEMA,
        "campaign_signature": campaign_signature(profile),
        "metric_schema": METRIC_SCHEMA,
        "jana_paper_commit": JANA_PAPER_COMMIT,
        "methods": list(METHODS),
        "flow_members": campaign["flow_members"],
        "classifier_members": campaign["classifier_members"],
        "flow_epochs": campaign["flow_epochs"],
        "classifier_epochs": campaign["class_epochs"],
        "jana_baseline": "same_frozen_qphi_qeta_direct_no_classifier",
        "hybrid_classifier_objective": "equal_prior_multiclass_CE_only",
        "hybrid_checkpoint_objective": "held_out_multiclass_CE_only",
        "normalization_role": "post_training_hybrid_only",
        "bridge_role": "paired_held_out_check_only",
    }
