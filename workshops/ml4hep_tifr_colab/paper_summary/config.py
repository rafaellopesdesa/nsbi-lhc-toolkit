"""Frozen configuration contract for the paired SLCP paper campaign.

This module contains only configuration, validation, and deterministic naming.
Training and plotting live in the sibling ``utils.py`` and
``utils_plotting.py`` modules so the generated notebooks remain thin.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


CAMPAIGN_SCHEMA = "slcp_jana_hybrid_paired_v2"
METRIC_SCHEMA = "slcp_posterior_likelihood_routes_v2"
IMPLEMENTATION_VERSION = "paper_summary_2026_09_v2"
TASK_NAME = "slcp"

PAPER_BUDGETS = (10_000, 100_000, 1_000_000)
SMOKE_BUDGETS = (256, 512, 1_024)
DEFAULT_ML_SEEDS = (31_082_026, 31_082_027, 31_082_028)
SMOKE_ML_SEEDS = (31_082_026,)
MASTER_SIMULATION_SEED = 31_081_026
JANA_VALIDATION_SIMULATION_SEED = 31_581_026
JANA_PILOT_SIMULATION_SEED = 31_481_026
JANA_SHAPE_SIMULATION_SEED = 31_381_026
AUDIT_SIMULATION_SEED = 32_081_026
SPLIT_SEED = 33_081_026
VALIDATION_FRACTION = 0.10
JANA_VALIDATION_SIMULATIONS = 300
JANA_PILOT_SIMULATIONS = 2
JANA_SHAPE_SIMULATIONS = 2
PAPER_AUDIT_SIMULATIONS = 100_000
SMOKE_AUDIT_SIMULATIONS = 256

JANA_PAPER_REPOSITORY = "https://github.com/bayesflow-org/JANA-Paper.git"
JANA_PAPER_COMMIT = "6cbbc94faf0aa85147986f7f9516d13a52551bd4"
JANA_BAYESFLOW_COMMIT = "153dfefadd347717b7aeb9c4872a4b51ac04e83c"

# Verbatim SLCP choices read from the pinned upstream experiment.  The legacy
# runner may adapt only the simulator input to the fixed bank; these model and
# optimizer choices define the JANA-paper row and are intentionally distinct
# from the no-regularization matched/hybrid models below.
JANA_PAPER_SETTINGS = {
    "amortizer": "AmortizedPosteriorLikelihood",
    "posterior_dimension": 5,
    "posterior_couplings": 6,
    "posterior_coupling_pattern": "interleaved_affine_spline",
    "likelihood_dimension": 8,
    "likelihood_couplings": 4,
    "likelihood_coupling_pattern": "default_affine",
    "learnable_permutations": True,
    "act_norm": True,
    "hidden_layers": 2,
    "hidden_width": 128,
    "activation": "relu",
    "affine_l2_regularization": 5.0e-4,
    "affine_dropout_probability": 0.01,
    "spline_l2_regularization": 5.0e-3,
    "spline_dropout_probability": 0.05,
    "spline_bins": 16,
    "spline_domain": (-5.0, 5.0),
    "epochs": 100,
    "batch_size": 32,
    "validation_simulations": 300,
    "optimizer": "adam",
    "initial_learning_rate": 5.0e-4,
    "learning_rate_schedule": "cosine_decay",
    "clip_norm": 1.0,
    "early_stopping": False,
    "observation_preprocessing": "x_divided_by_30",
    "parameter_preprocessing": "raw_theta_in_minus3_plus3",
}

# The route-specific architecture is selected by held-out NLL.  The one-SE
# rule deliberately prefers the shallowest/narrowest statistically equivalent
# candidate rather than the absolute minimum-NLL architecture.
FLOW_ARCHITECTURE_GRID = tuple(
    {"n_coupling_layers": blocks, "hidden_features": width}
    for blocks in (4, 6, 8)
    for width in (32, 64, 128)
)
FLOW_FIXED_CONFIG = {
    "hidden_layers": 2,
    "spline_num_bins": 8,
    "spline_tail_bound": 5.0,
    "linear_mixing": "lu",
    "use_layer_permutations": False,
    "dropout_probability": 0.0,
}
MATCHED_FLOW_TRAINING = {
    "ensemble_members": 4,
    "all_members_see_complete_training_partition": True,
    "batch_size": 1_024,
    "capacity_screen_epochs": 40,
    "maximum_epochs": 60,
    "optimizer": "adam",
    "initial_learning_rate": 1.0e-4,
    "minimum_learning_rate": 1.0e-9,
    "learning_rate_schedule": "six_equal_log10_plateaus",
    "learning_rate_levels": (
        1.0e-4,
        1.0e-5,
        1.0e-6,
        1.0e-7,
        1.0e-8,
        1.0e-9,
    ),
    "learning_rate_drop_fractions": (1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6),
    "early_stopping": False,
    "checkpoint_selection": "final_epoch",
    "gradient_clip": 5.0,
    "weight_decay": 0.0,
    "selection_metric": "held_out_validation_nll",
    "selection_rule": "smallest_within_one_standard_error_of_best",
    "routes_selected_separately": True,
}

# These settings apply to both multiclass and separate-binary factorisations.
RATIO_TRAINING = {
    "ensemble_members": 10,
    "hidden_width": 1_024,
    "hidden_layers": 4,
    "activation": "relu",
    "batch_size": 1_024,
    "training_steps": 5_000,
    "validation_interval_steps": 200,
    "optimizer": "adam",
    "initial_learning_rate": 1.0e-4,
    "minimum_learning_rate": 1.0e-9,
    "learning_rate_schedule": "six_equal_log10_plateaus",
    "learning_rate_levels": (
        1.0e-4,
        1.0e-5,
        1.0e-6,
        1.0e-7,
        1.0e-8,
        1.0e-9,
    ),
    "learning_rate_drop_fractions": (1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6),
    "objective": "equal_prior_cross_entropy_only",
    "dropout_probability": 0.0,
    "weight_decay": 0.0,
    "layer_normalization": False,
    "calibration_loss": False,
    "bridge_loss": False,
    "normalization_penalty": False,
    "ratio_representation": "log_softmax_difference",
    "ensemble_combination": "logsumexp_arithmetic_mean",
    "explicit_exponentials_in_training": False,
    "genuine_training_rows": "same_complete_training_partition_as_flows",
}

PROPOSAL_CONFIG = {
    "broad_base_scales": (1.25, 1.5),
    "prior_fractions": (0.05, 0.10),
    "selection_budget": 100_000,
    "selection_data": "shared_validation_partition_only",
    "selection_objective": "joint_closure_then_weight_efficiency",
    "selection_pilot_members": 3,
    "selection_pilot_steps": 2_000,
    "selection_pilot_split_fractions": (1 / 3, 1 / 3, 1 / 3),
    "selection_factorizations": ("multiclass", "binary"),
    "selection_aggregation": "worst_route_and_factorization_then_seed_mean",
    "uses_reference_posterior_for_selection": False,
    "freeze_selected_pair_across_budgets": True,
    "posterior": "pure_broad_flow_plus_prior_defense",
    "likelihood": "pure_broad_flow_no_observation_space_prior",
    "nominal_correction_ablation_budget": 100_000,
    "headline_uses_nominal_broad_mixture": False,
}

AUDIT_CONFIG = {
    "normalization_theta": 64,
    "normalization_x_per_theta": 512,
    "audit_joint_samples": 10_000,
}

DETERMINISM_CONFIG = {
    "torch_deterministic_algorithms": True,
    "torch_deterministic_warn_only": False,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
}

METHODS = (
    "jana_paper",
    "jana_paper_corrected_multiclass",
    "jana_paper_corrected_binary",
    "separate_flows",
    "separate_flows_corrected_multiclass",
    "separate_flows_corrected_binary",
)

DIAGNOSTICS = (
    "posterior_C2ST",
    "posterior_MMD",
    "likelihood_posterior_C2ST",
    "likelihood_posterior_MMD",
    "posterior_likelihood_route_C2ST",
    "posterior_likelihood_route_MMD",
    "posterior_joint_C2ST",
    "posterior_joint_MMD",
    "predictive_x_C2ST",
    "predictive_x_MMD",
    "predictive_joint_C2ST",
    "predictive_joint_MMD",
    "posterior_joint_ESS_fraction",
    "likelihood_joint_ESS_fraction",
    "posterior_ESS_fraction",
    "posterior_max_weight",
    "likelihood_posterior_ESS_fraction",
    "likelihood_posterior_max_weight",
    "bayes_cycle_pearson",
    "bayes_cycle_slope",
    "bayes_cycle_residual_rms",
    "likelihood_log_Z_rms",
    "likelihood_log_Z_mean",
    "likelihood_log_Z_max_abs",
    "exact_likelihood_log_error",
    "exact_likelihood_centered_log_error",
)

PROFILE_DEFAULTS = {
    "PAPER": {
        "budgets": PAPER_BUDGETS,
        "ml_seeds": DEFAULT_ML_SEEDS,
        "audit_simulations": PAPER_AUDIT_SIMULATIONS,
        "observations": tuple(range(1, 11)),
        "posterior_samples": 10_000,
        "proposal_candidates": 150_000,
        "metric_max_samples": 5_000,
        "normalization_theta": AUDIT_CONFIG["normalization_theta"],
        "normalization_x_per_theta": AUDIT_CONFIG["normalization_x_per_theta"],
        "audit_joint_samples": AUDIT_CONFIG["audit_joint_samples"],
        "flow_architecture_grid": FLOW_ARCHITECTURE_GRID,
        "flow_epoch_override": None,
        "ratio_member_override": None,
        "ratio_step_override": None,
    },
    "SMOKE": {
        "budgets": SMOKE_BUDGETS,
        "ml_seeds": SMOKE_ML_SEEDS,
        "audit_simulations": SMOKE_AUDIT_SIMULATIONS,
        "observations": (1,),
        "posterior_samples": 128,
        "proposal_candidates": 512,
        "metric_max_samples": 128,
        "normalization_theta": 4,
        "normalization_x_per_theta": 16,
        "audit_joint_samples": 128,
        "flow_architecture_grid": (FLOW_ARCHITECTURE_GRID[0],),
        "flow_epoch_override": 2,
        "ratio_member_override": 1,
        "ratio_step_override": 20,
    },
}


def normalize_profile(profile: str) -> str:
    """Return and validate an upper-case compute profile name."""

    value = str(profile).upper()
    if value not in PROFILE_DEFAULTS:
        raise ValueError(
            f"profile must be one of {tuple(PROFILE_DEFAULTS)}; got {profile!r}"
        )
    return value


def _unique_positive(values: Sequence[int], name: str) -> tuple[int, ...]:
    normalized = tuple(int(value) for value in values)
    if not normalized or any(value <= 0 for value in normalized):
        raise ValueError(f"{name} must contain positive integers")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates")
    return normalized


def campaign_config(
    *,
    profile: str = "PAPER",
    budgets: Sequence[int] | None = None,
    ml_seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Build a validated, JSON-serializable campaign configuration.

    ``budgets`` are always sorted and must be nested prefixes of a single
    master bank.  The runtime owns the concrete row indices and fingerprints.
    """

    profile = normalize_profile(profile)
    defaults = copy.deepcopy(PROFILE_DEFAULTS[profile])
    selected_budgets = _unique_positive(
        defaults["budgets"] if budgets is None else budgets, "budgets"
    )
    if tuple(sorted(selected_budgets)) != selected_budgets:
        raise ValueError("budgets must be strictly increasing")
    selected_seeds = _unique_positive(
        defaults["ml_seeds"] if ml_seeds is None else ml_seeds, "ml_seeds"
    )
    configuration = {
        "campaign_schema": CAMPAIGN_SCHEMA,
        "metric_schema": METRIC_SCHEMA,
        "implementation_version": IMPLEMENTATION_VERSION,
        "task": TASK_NAME,
        "profile": profile,
        "budgets": selected_budgets,
        "master_simulations": max(selected_budgets),
        "master_simulation_seed": MASTER_SIMULATION_SEED,
        "jana_validation_simulations": JANA_VALIDATION_SIMULATIONS,
        "jana_validation_simulation_seed": JANA_VALIDATION_SIMULATION_SEED,
        "jana_pilot_simulations": JANA_PILOT_SIMULATIONS,
        "jana_pilot_simulation_seed": JANA_PILOT_SIMULATION_SEED,
        "jana_shape_simulations": JANA_SHAPE_SIMULATIONS,
        "jana_shape_simulation_seed": JANA_SHAPE_SIMULATION_SEED,
        "audit_simulations": int(defaults["audit_simulations"]),
        "audit_simulation_seed": AUDIT_SIMULATION_SEED,
        "split_seed": SPLIT_SEED,
        "validation_fraction": VALIDATION_FRACTION,
        "bank_rule": "single_master_bank_nested_prefixes_fixed_master_row_split",
        "simulation_accounting": {
            "matched_and_hybrid": "N_total_master_pairs_with_shared_internal_split",
            "jana_paper": "N_training_pairs_plus_2_shape_plus_2_pilot_plus_300_validation_pairs",
            "audit": "separate_evaluation_calls_never_used_for_selection",
            "counting_unit": "unique_simulator_calls_not_network_exposures",
        },
        "ml_seeds": selected_seeds,
        "observations": tuple(defaults["observations"]),
        "posterior_samples": int(defaults["posterior_samples"]),
        "proposal_candidates": int(defaults["proposal_candidates"]),
        "metric_max_samples": int(defaults["metric_max_samples"]),
        "normalization_theta": int(defaults["normalization_theta"]),
        "normalization_x_per_theta": int(defaults["normalization_x_per_theta"]),
        "audit_joint_samples": int(defaults["audit_joint_samples"]),
        "jana": {
            "repository": JANA_PAPER_REPOSITORY,
            "commit": JANA_PAPER_COMMIT,
            "bayesflow_commit": JANA_BAYESFLOW_COMMIT,
            "algorithm": "pinned_paper_source",
            "data_adapter": "fixed_nested_bank_only",
            "legacy_environment": "isolated_subprocess",
            "paper_settings": JANA_PAPER_SETTINGS,
        },
        "flow_architecture_grid": tuple(defaults["flow_architecture_grid"]),
        "flow_fixed_config": FLOW_FIXED_CONFIG,
        "matched_flow_training": {
            **MATCHED_FLOW_TRAINING,
            "maximum_epochs_override": defaults["flow_epoch_override"],
        },
        "ratio_training": {
            **RATIO_TRAINING,
            "ensemble_members_override": defaults["ratio_member_override"],
            "training_steps_override": defaults["ratio_step_override"],
        },
        "proposal": PROPOSAL_CONFIG,
        "methods": METHODS,
        "diagnostics": DIAGNOSTICS,
        "symmetry_augmentation": False,
        "audit_used_for_selection": False,
        "determinism": DETERMINISM_CONFIG,
    }
    validate_campaign_config(configuration)
    return copy.deepcopy(configuration)


def validate_campaign_config(configuration: Mapping[str, Any]) -> None:
    """Fail loudly when a campaign violates the paired-comparison contract."""

    budgets = tuple(int(value) for value in configuration["budgets"])
    if tuple(sorted(set(budgets))) != budgets:
        raise ValueError("Campaign budgets must be unique and increasing")
    if int(configuration["master_simulations"]) != max(budgets):
        raise ValueError("Master bank size must equal the largest requested budget")
    if not 0.0 < float(configuration["validation_fraction"]) < 0.5:
        raise ValueError("Validation fraction must lie strictly between 0 and 0.5")
    bank_seeds = {
        int(configuration["master_simulation_seed"]),
        int(configuration["jana_shape_simulation_seed"]),
        int(configuration["jana_pilot_simulation_seed"]),
        int(configuration["jana_validation_simulation_seed"]),
        int(configuration["audit_simulation_seed"]),
    }
    if len(bank_seeds) != 5:
        raise ValueError(
            "Master, JANA-shape, JANA-pilot, JANA-validation, and audit banks need distinct seeds"
        )
    if int(configuration["jana_shape_simulations"]) != 2:
        raise ValueError("The exact JANA Benchmark protocol requires a 2-row shape pilot")
    if int(configuration["jana_pilot_simulations"]) != 2:
        raise ValueError("The exact JANA Trainer protocol requires a 2-row pilot")
    if int(configuration["jana_validation_simulations"]) != 300:
        raise ValueError("The exact JANA protocol requires 300 validation calls")
    if configuration.get("symmetry_augmentation") is not False:
        raise ValueError("SLCP symmetry is excluded from the primary campaign")
    if configuration.get("audit_used_for_selection") is not False:
        raise ValueError("The audit bank must never select models or proposals")
    if configuration["ratio_training"]["genuine_training_rows"] != (
        "same_complete_training_partition_as_flows"
    ):
        raise ValueError("Flows and ratio estimators must share the training rows")
    if configuration["proposal"]["headline_uses_nominal_broad_mixture"] is not False:
        raise ValueError("The headline proposal uses a pure broadened latent base")
    if any(
        float(value) <= 1.0
        for value in configuration["proposal"]["broad_base_scales"]
    ):
        raise ValueError("Every headline latent broadening scale must exceed one")
    if set(configuration["proposal"]["selection_factorizations"]) != {
        "multiclass",
        "binary",
    }:
        raise ValueError("Proposal selection must treat both ratio factorizations symmetrically")
    pilot_split = tuple(
        float(value)
        for value in configuration["proposal"]["selection_pilot_split_fractions"]
    )
    if (
        len(pilot_split) != 3
        or any(value <= 0.0 for value in pilot_split)
        or abs(sum(pilot_split) - 1.0) > 1.0e-12
    ):
        raise ValueError("Proposal pilot train/validation/closure fractions are invalid")
    if configuration["matched_flow_training"]["checkpoint_selection"] != "final_epoch":
        raise ValueError("Paper flows must deploy the final fixed-schedule epoch")
    if configuration["determinism"]["torch_deterministic_warn_only"] is not False:
        raise ValueError("The paper campaign requires strict deterministic algorithms")


def campaign_signature(configuration: Mapping[str, Any]) -> str:
    """Return a short content hash used by every artifact and checkpoint."""

    validate_campaign_config(configuration)
    serialized = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    return "sha256-" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def run_tag(
    stage: str,
    configuration: Mapping[str, Any],
    *,
    budget: int | None = None,
    ml_seed: int | None = None,
) -> str:
    """Build a readable artifact tag whose suffix fingerprints the contract."""

    parts = [str(stage).strip().lower(), configuration["profile"].lower()]
    if budget is not None:
        if int(budget) not in set(configuration["budgets"]):
            raise ValueError(f"budget {budget} is not part of this campaign")
        parts.append(f"n{int(budget)}")
    if ml_seed is not None:
        if int(ml_seed) not in set(configuration["ml_seeds"]):
            raise ValueError(f"ML seed {ml_seed} is not part of this campaign")
        parts.append(f"seed{int(ml_seed)}")
    parts.append(campaign_signature(configuration))
    return "_".join(parts)


def default_artifact_root(source_directory: str | Path) -> Path:
    """Return the non-Colab default without creating it."""

    return Path(source_directory).resolve() / "artifacts"


__all__ = [
    "AUDIT_SIMULATION_SEED",
    "CAMPAIGN_SCHEMA",
    "DEFAULT_ML_SEEDS",
    "DIAGNOSTICS",
    "DETERMINISM_CONFIG",
    "FLOW_ARCHITECTURE_GRID",
    "FLOW_FIXED_CONFIG",
    "IMPLEMENTATION_VERSION",
    "JANA_PAPER_COMMIT",
    "JANA_PAPER_SETTINGS",
    "JANA_PILOT_SIMULATIONS",
    "JANA_PILOT_SIMULATION_SEED",
    "JANA_SHAPE_SIMULATIONS",
    "JANA_SHAPE_SIMULATION_SEED",
    "JANA_VALIDATION_SIMULATIONS",
    "JANA_VALIDATION_SIMULATION_SEED",
    "MASTER_SIMULATION_SEED",
    "MATCHED_FLOW_TRAINING",
    "METHODS",
    "METRIC_SCHEMA",
    "PAPER_BUDGETS",
    "PROFILE_DEFAULTS",
    "PROPOSAL_CONFIG",
    "RATIO_TRAINING",
    "SMOKE_BUDGETS",
    "SMOKE_ML_SEEDS",
    "TASK_NAME",
    "campaign_config",
    "campaign_signature",
    "default_artifact_root",
    "normalize_profile",
    "run_tag",
    "validate_campaign_config",
]
