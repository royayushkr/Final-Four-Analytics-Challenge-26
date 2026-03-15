#!/usr/bin/env python3
"""Experiment registry for v5 seed-ranking runs."""

from __future__ import annotations

from dataclasses import dataclass


PRIMARY_MODEL_SWEEP = ("ridge", "xgb", "lightgbm", "catboost")
V4_BASELINE_MODELS = ("ridge", "xgb", "extra_trees")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    feature_blocks: tuple[str, ...]
    model_families: tuple[str, ...]
    include_bid_type_feature: bool = True
    include_team_feature: bool = True
    use_bid_type_selection: bool = True
    status: str = "active"
    skip_reason: str | None = None
    notes: str = ""


EXPERIMENTS: dict[str, ExperimentConfig] = {
    "current_v4": ExperimentConfig(
        name="current_v4",
        feature_blocks=("core_parsed", "season_relative", "conference_relative"),
        model_families=V4_BASELINE_MODELS,
        notes="Reproduced v4-style feature set under the unified v5 evaluator.",
    ),
    "v5_base_core": ExperimentConfig(
        name="v5_base_core",
        feature_blocks=("core_parsed",),
        model_families=PRIMARY_MODEL_SWEEP,
    ),
    "v5_base_core_missing": ExperimentConfig(
        name="v5_base_core_missing",
        feature_blocks=("core_parsed", "missing_indicators"),
        model_families=PRIMARY_MODEL_SWEEP,
    ),
    "v5_base_season_relative": ExperimentConfig(
        name="v5_base_season_relative",
        feature_blocks=("core_parsed", "missing_indicators", "season_relative"),
        model_families=PRIMARY_MODEL_SWEEP,
    ),
    "v5_base_conference_relative": ExperimentConfig(
        name="v5_base_conference_relative",
        feature_blocks=(
            "core_parsed",
            "missing_indicators",
            "season_relative",
            "conference_relative",
        ),
        model_families=PRIMARY_MODEL_SWEEP,
    ),
    "v5_base_interactions_ratios": ExperimentConfig(
        name="v5_base_interactions_ratios",
        feature_blocks=(
            "core_parsed",
            "missing_indicators",
            "season_relative",
            "conference_relative",
            "robust_numeric",
            "interaction_refinements",
        ),
        model_families=PRIMARY_MODEL_SWEEP,
    ),
    "v5_base_target_safe_encodings": ExperimentConfig(
        name="v5_base_target_safe_encodings",
        feature_blocks=(
            "core_parsed",
            "missing_indicators",
            "season_relative",
            "conference_relative",
            "robust_numeric",
            "interaction_refinements",
            "team_conference_encoding",
        ),
        model_families=PRIMARY_MODEL_SWEEP,
    ),
    "v5_base_full": ExperimentConfig(
        name="v5_base_full",
        feature_blocks=(
            "core_parsed",
            "missing_indicators",
            "season_relative",
            "conference_relative",
            "robust_numeric",
            "interaction_refinements",
            "team_conference_encoding",
        ),
        model_families=PRIMARY_MODEL_SWEEP,
        notes="Full non-market v5 feature stack.",
    ),
    "v5_base": ExperimentConfig(
        name="v5_base",
        feature_blocks=(
            "core_parsed",
            "missing_indicators",
            "season_relative",
            "conference_relative",
            "robust_numeric",
            "interaction_refinements",
            "team_conference_encoding",
        ),
        model_families=PRIMARY_MODEL_SWEEP,
        notes="Alias of the full non-market base stack.",
    ),
    "v5_no_market": ExperimentConfig(
        name="v5_no_market",
        feature_blocks=(
            "core_parsed",
            "missing_indicators",
            "season_relative",
            "conference_relative",
            "robust_numeric",
            "interaction_refinements",
            "team_conference_encoding",
        ),
        model_families=PRIMARY_MODEL_SWEEP,
        notes="Control variant with market toggles explicitly absent.",
    ),
    "v5_goto": ExperimentConfig(
        name="v5_goto",
        feature_blocks=(),
        model_families=(),
        status="skipped",
        skip_reason="SKIPPED_NO_MARKET_INPUTS",
        notes="goto_conversion is not relevant because the dataset has no odds-like inputs.",
    ),
    "v5_shin": ExperimentConfig(
        name="v5_shin",
        feature_blocks=(),
        model_families=(),
        status="skipped",
        skip_reason="SKIPPED_NO_MARKET_INPUTS",
        notes="efficient_shin_conversion is not relevant because the dataset has no odds-like inputs.",
    ),
    "v5_market_all": ExperimentConfig(
        name="v5_market_all",
        feature_blocks=(),
        model_families=(),
        status="skipped",
        skip_reason="SKIPPED_NO_MARKET_INPUTS",
        notes="All market-conversion methods skipped due to lack of bookmaker or implied-probability fields.",
    ),
    "v5_mult_baseline": ExperimentConfig(
        name="v5_mult_baseline",
        feature_blocks=(),
        model_families=(),
        status="skipped",
        skip_reason="SKIPPED_NO_MARKET_INPUTS",
        notes="No valid market conversion baseline exists for this repository.",
    ),
}


ABLATION_ORDER = [
    "current_v4",
    "v5_base_core",
    "v5_base_core_missing",
    "v5_base_season_relative",
    "v5_base_conference_relative",
    "v5_base_interactions_ratios",
    "v5_base_target_safe_encodings",
    "v5_base_full",
    "v5_base",
    "v5_no_market",
    "v5_goto",
    "v5_shin",
    "v5_market_all",
    "v5_mult_baseline",
]
