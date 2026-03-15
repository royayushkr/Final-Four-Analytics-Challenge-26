#!/usr/bin/env python3
"""Private 2026 NCAA seed submission pipeline.

This script implements a private-task-specific v7 flow:
- trains only on the official training file,
- evaluates candidates with expanding-window season backtests,
- checks weekly 2026 stability without refitting,
- writes exactly one final submission aligned to the 2026-03-15 snapshot.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from private_features import FeatureMatrix, make_feature_matrix
from private_validation import (
    compute_weekly_stability_metrics,
    make_expanding_window_splits,
    score_full_holdout_season,
    select_final_candidate,
    write_model_comparison,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "NCAA_Seed_Training_Set2.0.csv"
DEFAULT_HISTORICAL_TEST_PATH = PROJECT_ROOT / "data" / "raw" / "NCAA_Seed_Test_Set2.0.csv"
DEFAULT_FINAL_TEST_PATH = PROJECT_ROOT / "data" / "raw" / "NCAA_Seed_Test_Set_2026_20260315.csv"
DEFAULT_WEEKLY_GLOB = PROJECT_ROOT / "data" / "raw"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "final_20260315"
DEFAULT_SUBMISSION_PATH = PROJECT_ROOT / "submissions" / "final" / "submission_2026_20260315.csv"
RIDGE_ALPHA_GRID = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])


@dataclass
class CandidateSpec:
    name: str
    model_weights: dict[str, float]
    include_conference_context: bool
    use_band_model: bool = False
    band_weight: float = 0.0
    use_pu_selector: bool = False
    pu_weight: float = 0.0
    notes: str = ""


@dataclass
class FittedModel:
    name: str
    model: Any
    preprocessor: ColumnTransformer | None
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    medians: dict[str, float] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--historical-test-path", type=Path, default=DEFAULT_HISTORICAL_TEST_PATH)
    parser.add_argument("--final-test-path", type=Path, default=DEFAULT_FINAL_TEST_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--submission-path", type=Path, default=DEFAULT_SUBMISSION_PATH)
    return parser.parse_args()


def load_datasets(
    train_path: Path,
    historical_test_path: Path,
    final_test_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    train_df = pd.read_csv(train_path)
    historical_test_df = pd.read_csv(historical_test_path)
    final_test_df = pd.read_csv(final_test_path)

    weekly_files = sorted(DEFAULT_WEEKLY_GLOB.glob("NCAA_Seed_Test_Set_2026_*.csv"))
    weekly_snapshots: dict[str, pd.DataFrame] = {}
    for path in weekly_files:
        stamp = path.stem.split("_")[-1]
        weekly_snapshots[stamp] = pd.read_csv(path)
    return train_df, historical_test_df, final_test_df, weekly_snapshots


def build_holdout_universe(
    official_valid_df: pd.DataFrame,
    historical_test_df: pd.DataFrame,
    season: str,
) -> pd.DataFrame:
    official = official_valid_df.copy()
    official["AuditSelected"] = official["Overall Seed"].notna().astype(int)
    official["RowSource"] = "official_train"

    audit_extra = historical_test_df[historical_test_df["Season"] == season].copy()
    audit_extra["Overall Seed"] = np.nan
    audit_extra["AuditSelected"] = audit_extra["Bid Type"].notna().astype(int)
    audit_extra["RowSource"] = "historical_audit"

    combined = pd.concat([official, audit_extra], ignore_index=True, sort=False)
    return combined


def _build_linear_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                numeric_columns,
            ),
            (
                "cat",
                Pipeline([
                    ("imp", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                    ("oh", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_columns,
            ),
        ]
    )


def _build_tree_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), numeric_columns),
            (
                "cat",
                Pipeline([
                    ("imp", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                    ("oh", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_columns,
            ),
        ]
    )


def _prepare_catboost_frames(
    train_x: pd.DataFrame,
    eval_x: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[int], dict[str, float]]:
    train_cb = train_x.copy()
    eval_cb = eval_x.copy()
    medians: dict[str, float] = {}
    for col in numeric_columns:
        median = float(pd.to_numeric(train_cb[col], errors="coerce").median())
        medians[col] = median
        train_cb[col] = pd.to_numeric(train_cb[col], errors="coerce").fillna(median)
        eval_cb[col] = pd.to_numeric(eval_cb[col], errors="coerce").fillna(median)
    for col in categorical_columns:
        train_cb[col] = train_cb[col].fillna("__MISSING__").astype(str)
        eval_cb[col] = eval_cb[col].fillna("__MISSING__").astype(str)
    cat_idx = [train_cb.columns.get_loc(col) for col in categorical_columns]
    return train_cb, eval_cb, cat_idx, medians


def fit_seed_strength_models(
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
    model_weights: dict[str, float],
) -> dict[str, FittedModel]:
    seeded = train_frame[train_frame["Overall Seed"].notna()].copy().reset_index(drop=True)
    x_train = seeded[feature_columns].copy()
    y_train = 69.0 - seeded["Overall Seed"].to_numpy(dtype=float)

    fitted: dict[str, FittedModel] = {}

    if "ridge" in model_weights:
        pre = _build_linear_preprocessor(numeric_columns, categorical_columns)
        x_t = pre.fit_transform(x_train)
        model = RidgeCV(alphas=RIDGE_ALPHA_GRID)
        model.fit(x_t, y_train)
        fitted["ridge"] = FittedModel(
            name="ridge",
            model=model,
            preprocessor=pre,
            feature_columns=feature_columns,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
        )

    if "lgbm" in model_weights:
        pre = _build_tree_preprocessor(numeric_columns, categorical_columns)
        x_t = pre.fit_transform(x_train)
        model = LGBMRegressor(
            objective="regression",
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=10,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            random_state=42,
            verbosity=-1,
            force_col_wise=True,
        )
        model.fit(x_t, y_train)
        fitted["lgbm"] = FittedModel(
            name="lgbm",
            model=model,
            preprocessor=pre,
            feature_columns=feature_columns,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
        )

    if "catboost" in model_weights:
        train_cb, _, cat_idx, medians = _prepare_catboost_frames(
            x_train,
            x_train,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
        )
        model = CatBoostRegressor(
            loss_function="RMSE",
            iterations=500,
            depth=5,
            learning_rate=0.03,
            random_seed=42,
            verbose=False,
        )
        model.fit(train_cb, y_train, cat_features=cat_idx)
        fitted["catboost"] = FittedModel(
            name="catboost",
            model=model,
            preprocessor=None,
            feature_columns=feature_columns,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            medians=medians,
        )

    return fitted


def _predict_single_model(fitted: FittedModel, eval_x: pd.DataFrame) -> np.ndarray:
    x_eval = eval_x[fitted.feature_columns].copy()
    if fitted.name == "catboost":
        for col in fitted.numeric_columns:
            fill_value = 0.0 if fitted.medians is None else fitted.medians[col]
            x_eval[col] = pd.to_numeric(x_eval[col], errors="coerce").fillna(fill_value)
        for col in fitted.categorical_columns:
            x_eval[col] = x_eval[col].fillna("__MISSING__").astype(str)
        return np.asarray(fitted.model.predict(x_eval), dtype=float)

    assert fitted.preprocessor is not None
    x_t = fitted.preprocessor.transform(x_eval)
    return np.asarray(fitted.model.predict(x_t), dtype=float)


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    std = float(np.nanstd(values))
    if not np.isfinite(std) or std == 0.0:
        return np.zeros_like(values)
    return (values - float(np.nanmean(values))) / std


def fit_optional_band_model(
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
    enabled: bool,
) -> FittedModel | None:
    if not enabled:
        return None

    seeded = train_frame[train_frame["Overall Seed"].notna()].copy().reset_index(drop=True)
    if seeded.empty:
        return None
    x_train = seeded[feature_columns].copy()
    y_seed = seeded["Overall Seed"].astype(int)
    band = pd.cut(
        y_seed,
        bins=[0, 16, 32, 48, 68],
        labels=[0, 1, 2, 3],
        include_lowest=True,
    ).astype(int)
    train_cb, _, cat_idx, medians = _prepare_catboost_frames(
        x_train,
        x_train,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )
    model = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=300,
        depth=4,
        learning_rate=0.03,
        random_seed=42,
        verbose=False,
    )
    model.fit(train_cb, band, cat_features=cat_idx)
    return FittedModel(
        name="band_catboost",
        model=model,
        preprocessor=None,
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        medians=medians,
    )


def fit_optional_pu_selector(
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
    enabled: bool,
) -> FittedModel | None:
    if not enabled:
        return None

    x_train = train_frame[feature_columns].copy()
    y_train = train_frame["Overall Seed"].notna().astype(int).to_numpy(dtype=int)
    pre = _build_linear_preprocessor(numeric_columns, categorical_columns)
    x_t = pre.fit_transform(x_train)
    model = LogisticRegression(max_iter=5000, C=0.5, class_weight="balanced", random_state=42)
    model.fit(x_t, y_train)
    return FittedModel(
        name="pu_logistic",
        model=model,
        preprocessor=pre,
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )


def generate_final_predictions(
    eval_frame: pd.DataFrame,
    fitted_models: dict[str, FittedModel],
    candidate: CandidateSpec,
    band_model: FittedModel | None,
    pu_model: FittedModel | None,
) -> pd.DataFrame:
    scored = eval_frame.copy().reset_index(drop=True)
    component_scores: dict[str, np.ndarray] = {}
    base_score = np.zeros(len(scored), dtype=float)

    total_weight = float(sum(candidate.model_weights.values()))
    for model_name, weight in candidate.model_weights.items():
        preds = _predict_single_model(fitted_models[model_name], scored)
        component_scores[f"{model_name}_score"] = preds
        base_score += (weight / total_weight) * preds

    scored["BaseScore"] = base_score
    final_score = base_score.copy()

    if band_model is not None:
        band_cb, _, _, _ = _prepare_catboost_frames(
            scored[band_model.feature_columns],
            scored[band_model.feature_columns],
            numeric_columns=band_model.numeric_columns,
            categorical_columns=band_model.categorical_columns,
        )
        proba = band_model.model.predict_proba(band_cb)
        band_strength = proba @ np.array([4.0, 3.0, 2.0, 1.0], dtype=float)
        scored["BandScore"] = band_strength
        final_score = final_score + candidate.band_weight * _zscore(band_strength)
    else:
        scored["BandScore"] = 0.0

    if pu_model is not None:
        pu_x = scored[pu_model.feature_columns].copy()
        assert pu_model.preprocessor is not None
        pu_prob = pu_model.model.predict_proba(pu_model.preprocessor.transform(pu_x))[:, 1]
        scored["PUScore"] = pu_prob
        rank_order = pd.Series(-final_score).rank(method="first").to_numpy(dtype=float)
        bubble_mask = (rank_order >= 55.0) & (rank_order <= 80.0)
        final_score = final_score + candidate.pu_weight * _zscore(pu_prob) * bubble_mask.astype(float)
    else:
        scored["PUScore"] = 0.0

    for col, values in component_scores.items():
        scored[col] = values

    scored["FinalScore"] = final_score
    sort_cols = ["FinalScore"]
    ascending = [False]
    if "NET Rank" in scored.columns:
        sort_cols.append("NET Rank")
        ascending.append(True)
    if "RecordID" in scored.columns:
        sort_cols.append("RecordID")
        ascending.append(True)
    scored = scored.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    scored["RankOrder"] = np.arange(1, len(scored) + 1)
    scored["PredictedSeed"] = 0
    selected_n = min(68, len(scored))
    scored.loc[: selected_n - 1, "PredictedSeed"] = np.arange(1, selected_n + 1)
    scored = scored.sort_values("RankOrder").reset_index(drop=True)
    return scored


def _extract_model_feature_importance(fitted: FittedModel) -> pd.DataFrame:
    if fitted.name == "catboost":
        values = np.asarray(fitted.model.get_feature_importance(), dtype=float)
        frame = pd.DataFrame({"model": fitted.name, "feature": fitted.feature_columns, "importance": values})
        return frame.sort_values("importance", ascending=False).reset_index(drop=True)

    assert fitted.preprocessor is not None
    feature_names = fitted.preprocessor.get_feature_names_out()
    if hasattr(fitted.model, "coef_"):
        values = np.abs(np.asarray(fitted.model.coef_, dtype=float))
    else:
        values = np.asarray(fitted.model.feature_importances_, dtype=float)
    frame = pd.DataFrame({"model": fitted.name, "feature": feature_names, "importance": values})
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def _summarize_candidate_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate, grp in fold_metrics.groupby("candidate"):
        weight_sum = grp["fold_weight"].sum()
        rows.append(
            {
                "candidate": candidate,
                "WeightedCompositeLoss": float(np.average(grp["CompositeLoss"], weights=grp["fold_weight"])),
                "WeightedConfirmedSeedRMSE": float(np.average(grp["ConfirmedSeedRMSE"], weights=grp["fold_weight"])),
                "WeightedFieldRecall": float(np.average(grp["FieldRecall@68"], weights=grp["fold_weight"])),
                "WeightedFieldPrecision": float(np.average(grp["FieldPrecision@68"], weights=grp["fold_weight"])),
                "WeightedFieldF1": float(np.average(grp["FieldF1@68"], weights=grp["fold_weight"])),
                "WeightedConfirmedTop68Recall": float(np.average(grp["ConfirmedTop68Recall"], weights=grp["fold_weight"])),
                "WeightedSeedOrderSpearman": float(np.average(grp["SeedOrderSpearman"].fillna(0.0), weights=grp["fold_weight"])),
                "RecentFoldRecallMin": float(grp.sort_values("valid_season").tail(2)["FieldRecall@68"].min()),
                "FoldCount": int(len(grp)),
                "WeightSum": float(weight_sum),
            }
        )
    return pd.DataFrame(rows).sort_values("WeightedCompositeLoss").reset_index(drop=True)


def run_backtests(
    train_df: pd.DataFrame,
    historical_test_df: pd.DataFrame,
    candidates: list[CandidateSpec],
) -> tuple[pd.DataFrame, dict[str, list[pd.DataFrame]]]:
    folds = make_expanding_window_splits(train_df)
    fold_rows: list[dict[str, Any]] = []
    ranked_outputs: dict[str, list[pd.DataFrame]] = {candidate.name: [] for candidate in candidates}

    for split in folds:
        train_hist = train_df[train_df["Season"].isin(split.train_seasons)].copy().reset_index(drop=True)
        valid_official = train_df[train_df["Season"] == split.valid_season].copy().reset_index(drop=True)
        holdout_full = build_holdout_universe(valid_official, historical_test_df, split.valid_season)

        for candidate in candidates:
            feature_bundle = make_feature_matrix(
                train_hist,
                holdout_full,
                include_conference_context=candidate.include_conference_context,
            )
            fitted_models = fit_seed_strength_models(
                train_frame=feature_bundle.train_frame,
                feature_columns=feature_bundle.feature_columns,
                numeric_columns=feature_bundle.numeric_columns,
                categorical_columns=feature_bundle.categorical_columns,
                model_weights=candidate.model_weights,
            )
            band_model = fit_optional_band_model(
                train_frame=feature_bundle.train_frame,
                feature_columns=feature_bundle.feature_columns,
                numeric_columns=feature_bundle.numeric_columns,
                categorical_columns=feature_bundle.categorical_columns,
                enabled=candidate.use_band_model,
            )
            pu_model = fit_optional_pu_selector(
                train_frame=feature_bundle.train_frame,
                feature_columns=feature_bundle.feature_columns,
                numeric_columns=feature_bundle.numeric_columns,
                categorical_columns=feature_bundle.categorical_columns,
                enabled=candidate.use_pu_selector,
            )
            scored = generate_final_predictions(
                eval_frame=feature_bundle.eval_frame,
                fitted_models=fitted_models,
                candidate=candidate,
                band_model=band_model,
                pu_model=pu_model,
            )
            metrics, ranked = score_full_holdout_season(scored, score_col="FinalScore", top_k=68)
            ranked["candidate"] = candidate.name
            ranked["valid_season"] = split.valid_season
            ranked_outputs[candidate.name].append(ranked)
            fold_rows.append(
                {
                    "candidate": candidate.name,
                    "fold_name": split.fold_name,
                    "valid_season": split.valid_season,
                    "fold_weight": split.weight,
                    **metrics,
                }
            )
    return pd.DataFrame(fold_rows), ranked_outputs


def run_2026_stability_checks(
    train_df: pd.DataFrame,
    weekly_snapshots: dict[str, pd.DataFrame],
    candidates: list[CandidateSpec],
) -> tuple[pd.DataFrame, dict[str, dict[str, pd.DataFrame]]]:
    detail: dict[str, dict[str, pd.DataFrame]] = {}
    rows: list[pd.DataFrame] = []
    for candidate in candidates:
        predictions_by_week: dict[str, pd.DataFrame] = {}
        for stamp, weekly_df in weekly_snapshots.items():
            feature_bundle = make_feature_matrix(
                train_df,
                weekly_df,
                include_conference_context=candidate.include_conference_context,
            )
            fitted_models = fit_seed_strength_models(
                train_frame=feature_bundle.train_frame,
                feature_columns=feature_bundle.feature_columns,
                numeric_columns=feature_bundle.numeric_columns,
                categorical_columns=feature_bundle.categorical_columns,
                model_weights=candidate.model_weights,
            )
            band_model = fit_optional_band_model(
                train_frame=feature_bundle.train_frame,
                feature_columns=feature_bundle.feature_columns,
                numeric_columns=feature_bundle.numeric_columns,
                categorical_columns=feature_bundle.categorical_columns,
                enabled=candidate.use_band_model,
            )
            pu_model = fit_optional_pu_selector(
                train_frame=feature_bundle.train_frame,
                feature_columns=feature_bundle.feature_columns,
                numeric_columns=feature_bundle.numeric_columns,
                categorical_columns=feature_bundle.categorical_columns,
                enabled=candidate.use_pu_selector,
            )
            scored = generate_final_predictions(
                eval_frame=feature_bundle.eval_frame,
                fitted_models=fitted_models,
                candidate=candidate,
                band_model=band_model,
                pu_model=pu_model,
            )
            scored = weekly_df[["RecordID", "Team", "Conference", "NET Rank"]].merge(
                scored[["RecordID", "FinalScore", "RankOrder", "PredictedSeed"]],
                on="RecordID",
                how="left",
            )
            predictions_by_week[stamp] = scored
        detail[candidate.name] = predictions_by_week
        summary = compute_weekly_stability_metrics(predictions_by_week)
        summary["candidate"] = candidate.name
        rows.append(summary)
    return pd.concat(rows, ignore_index=True), detail


def _fit_full_candidate(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    candidate: CandidateSpec,
) -> tuple[pd.DataFrame, dict[str, FittedModel], FittedModel | None, FittedModel | None, FeatureMatrix]:
    feature_bundle = make_feature_matrix(
        train_df,
        eval_df,
        include_conference_context=candidate.include_conference_context,
    )
    fitted_models = fit_seed_strength_models(
        train_frame=feature_bundle.train_frame,
        feature_columns=feature_bundle.feature_columns,
        numeric_columns=feature_bundle.numeric_columns,
        categorical_columns=feature_bundle.categorical_columns,
        model_weights=candidate.model_weights,
    )
    band_model = fit_optional_band_model(
        train_frame=feature_bundle.train_frame,
        feature_columns=feature_bundle.feature_columns,
        numeric_columns=feature_bundle.numeric_columns,
        categorical_columns=feature_bundle.categorical_columns,
        enabled=candidate.use_band_model,
    )
    pu_model = fit_optional_pu_selector(
        train_frame=feature_bundle.train_frame,
        feature_columns=feature_bundle.feature_columns,
        numeric_columns=feature_bundle.numeric_columns,
        categorical_columns=feature_bundle.categorical_columns,
        enabled=candidate.use_pu_selector,
    )
    scored = generate_final_predictions(
        eval_frame=feature_bundle.eval_frame,
        fitted_models=fitted_models,
        candidate=candidate,
        band_model=band_model,
        pu_model=pu_model,
    )
    return scored, fitted_models, band_model, pu_model, feature_bundle


def write_final_submission(
    final_scored: pd.DataFrame,
    final_test_df: pd.DataFrame,
    submission_path: Path,
) -> pd.DataFrame:
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    final_submission = final_test_df[["RecordID"]].merge(
        final_scored[["RecordID", "PredictedSeed"]],
        on="RecordID",
        how="left",
    )
    final_submission = final_submission.rename(columns={"PredictedSeed": "Overall Seed"})
    final_submission["Overall Seed"] = final_submission["Overall Seed"].fillna(0).astype(int)

    non_zero = final_submission.loc[final_submission["Overall Seed"] > 0, "Overall Seed"].tolist()
    if len(final_submission) != len(final_test_df):
        raise ValueError("Final submission row count mismatch")
    if final_submission["Overall Seed"].isna().any():
        raise ValueError("Final submission has missing predictions")
    if sum(seed > 0 for seed in non_zero) != 68:
        raise ValueError("Final submission does not contain exactly 68 seeded teams")
    if sorted(non_zero) != list(range(1, 69)):
        raise ValueError("Final submission non-zero seeds are not exactly 1..68")

    final_submission.to_csv(submission_path, index=False)
    return final_submission


def candidate_grid() -> list[CandidateSpec]:
    return [
        CandidateSpec(
            name="ridge_only_context",
            model_weights={"ridge": 1.0},
            include_conference_context=True,
            notes="Stable linear baseline with full committee feature block.",
        ),
        CandidateSpec(
            name="catboost_only_context",
            model_weights={"catboost": 1.0},
            include_conference_context=True,
            notes="Single nonlinear baseline with native categorical handling.",
        ),
        CandidateSpec(
            name="ensemble_base",
            model_weights={"ridge": 0.6, "catboost": 0.4},
            include_conference_context=False,
            notes="Default seed-strength ensemble without conference-context expansion.",
        ),
        CandidateSpec(
            name="ensemble_context",
            model_weights={"ridge": 0.6, "catboost": 0.4},
            include_conference_context=True,
            notes="Default private-final ensemble with conference context.",
        ),
        CandidateSpec(
            name="ensemble_context_band",
            model_weights={"ridge": 0.6, "catboost": 0.4},
            include_conference_context=True,
            use_band_model=True,
            band_weight=0.10,
            notes="Default ensemble plus low-weight seed-band regularizer.",
        ),
        CandidateSpec(
            name="ensemble_context_lgbm",
            model_weights={"ridge": 0.5, "catboost": 0.35, "lgbm": 0.15},
            include_conference_context=True,
            notes="Ensemble with low-weight LightGBM diversity component.",
        ),
        CandidateSpec(
            name="ensemble_context_pu",
            model_weights={"ridge": 0.6, "catboost": 0.4},
            include_conference_context=True,
            use_pu_selector=True,
            pu_weight=0.10,
            notes="Default ensemble plus low-impact PU bubble reranker.",
        ),
    ]


def main() -> None:
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
        category=UserWarning,
    )
    args = parse_args()
    artifact_dir = args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    train_df, historical_test_df, final_test_df, weekly_snapshots = load_datasets(
        train_path=args.train_path,
        historical_test_path=args.historical_test_path,
        final_test_path=args.final_test_path,
    )

    candidates = candidate_grid()
    fold_metrics, _ranked_outputs = run_backtests(train_df, historical_test_df, candidates)
    fold_metrics.to_csv(artifact_dir / "rolling_origin_metrics.csv", index=False)

    candidate_summary = _summarize_candidate_metrics(fold_metrics)
    weekly_summary, _weekly_detail = run_2026_stability_checks(train_df, weekly_snapshots, candidates)
    weekly_summary.to_csv(artifact_dir / "weekly_stability_metrics.csv", index=False)

    chosen = select_final_candidate(candidate_summary, weekly_summary)
    chosen_name = str(chosen["candidate"])
    chosen_spec = next(candidate for candidate in candidates if candidate.name == chosen_name)

    final_scored, fitted_models, band_model, pu_model, feature_bundle = _fit_full_candidate(
        train_df=train_df,
        eval_df=final_test_df,
        candidate=chosen_spec,
    )

    final_submission = write_final_submission(
        final_scored=final_scored,
        final_test_df=final_test_df,
        submission_path=args.submission_path,
    )

    feature_importance = pd.concat(
        [_extract_model_feature_importance(model) for model in fitted_models.values()],
        ignore_index=True,
    )
    feature_importance.to_csv(artifact_dir / "feature_importance.csv", index=False)

    final_audit = final_test_df[["RecordID", "Season", "Team", "Conference", "NET Rank"]].merge(
        final_scored[
            [
                "RecordID",
                "BaseScore",
                "BandScore",
                "PUScore",
                "FinalScore",
                "RankOrder",
                "PredictedSeed",
            ]
            + [col for col in final_scored.columns if col.endswith("_score")]
        ],
        on="RecordID",
        how="left",
    )
    final_audit.to_csv(artifact_dir / "final_selection_audit.csv", index=False)

    comparison_payload = {
        "locked_assumptions": {
            "final_row_set": str(args.final_test_path.name),
            "official_train_only_for_fitting": True,
            "historical_test_used_for_audit_only": True,
            "weekly_2026_snapshots_used_for_stability_only": True,
            "bid_type_used_in_final_model": False,
            "team_identity_used_in_final_model": False,
        },
        "candidate_specs": [asdict(candidate) for candidate in candidates],
        "candidate_summary": candidate_summary.to_dict(orient="records"),
        "chosen_candidate": chosen,
        "chosen_candidate_spec": asdict(chosen_spec),
        "submission_path": str(args.submission_path),
    }
    write_model_comparison(artifact_dir / "model_comparison.json", comparison_payload)

    print(f"Chosen candidate: {chosen_name}")
    print(f"Wrote rolling metrics: {artifact_dir / 'rolling_origin_metrics.csv'}")
    print(f"Wrote weekly metrics: {artifact_dir / 'weekly_stability_metrics.csv'}")
    print(f"Wrote feature importance: {artifact_dir / 'feature_importance.csv'}")
    print(f"Wrote final audit: {artifact_dir / 'final_selection_audit.csv'}")
    print(f"Wrote final submission: {args.submission_path}")
    print(f"Final seeded teams: {(final_submission['Overall Seed'] > 0).sum()}")


if __name__ == "__main__":
    main()
