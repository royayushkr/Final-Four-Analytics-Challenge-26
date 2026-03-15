#!/usr/bin/env python3
"""Run a single v5 seed-ranking experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from v5_config import EXPERIMENTS, ExperimentConfig
from v5_features import FeatureArtifacts, build_feature_frame, fit_feature_artifacts

try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover - optional dependency
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover - optional dependency
    CatBoostRegressor = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "NCAA_Seed_Training_Set2.0.csv"
DEFAULT_TEST_PATH = PROJECT_ROOT / "data" / "raw" / "NCAA_Seed_Test_Set2.0.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "submissions" / "v5_experiments"
RIDGE_ALPHA_GRID = [8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 40.0]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def season_rank_rmse(
    season_series: pd.Series,
    y_true: np.ndarray,
    score: np.ndarray,
) -> tuple[float, np.ndarray]:
    pred = np.zeros(len(y_true), dtype=float)
    for _, idx in season_series.groupby(season_series).indices.items():
        season_idx = np.array(idx, dtype=int)
        actual = y_true[season_idx]
        assigned = np.empty(len(season_idx), dtype=float)
        assigned[np.argsort(-score[season_idx])] = np.sort(actual)
        pred[season_idx] = assigned
    return rmse(y_true, pred), pred


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    cat_cols = [c for c in x.columns if x[c].dtype == "object"]
    num_cols = [c for c in x.columns if c not in cat_cols]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )


def get_model(model_family: str) -> object:
    if model_family == "ridge":
        return RidgeCV(alphas=RIDGE_ALPHA_GRID)
    if model_family == "xgb":
        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=700,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            random_state=42,
            n_jobs=4,
        )
    if model_family == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=900,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    if model_family == "lightgbm":
        if LGBMRegressor is None:
            raise ImportError("lightgbm is not installed")
        return LGBMRegressor(
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
    if model_family == "catboost":
        if CatBoostRegressor is None:
            raise ImportError("catboost is not installed")
        return CatBoostRegressor(
            loss_function="RMSE",
            iterations=600,
            depth=5,
            learning_rate=0.03,
            random_seed=42,
            verbose=False,
        )
    raise ValueError(f"Unknown model family: {model_family}")


def model_is_available(model_family: str) -> bool:
    try:
        get_model(model_family)
        return True
    except ImportError:
        return False


def fit_model(model: object, x_train_t: Any, y_train: np.ndarray) -> object:
    if model.__class__.__name__.startswith("CatBoost"):
        matrix = x_train_t.toarray() if hasattr(x_train_t, "toarray") else x_train_t
        model.fit(matrix, y_train)
        return model
    if model.__class__.__name__.startswith("LGBM"):
        matrix = x_train_t.toarray() if hasattr(x_train_t, "toarray") else x_train_t
        model.fit(matrix, y_train)
        return model
    model.fit(x_train_t, y_train)
    return model


def predict_model(model: object, x_data_t: Any) -> np.ndarray:
    if model.__class__.__name__.startswith("CatBoost"):
        matrix = x_data_t.toarray() if hasattr(x_data_t, "toarray") else x_data_t
        return np.asarray(model.predict(matrix), dtype=float)
    if model.__class__.__name__.startswith("LGBM"):
        matrix = x_data_t.toarray() if hasattr(x_data_t, "toarray") else x_data_t
        return np.asarray(model.predict(matrix), dtype=float)
    return np.asarray(model.predict(x_data_t), dtype=float)


def extract_feature_importance(
    model: object,
    preprocessor: ColumnTransformer,
) -> pd.DataFrame:
    feature_names = preprocessor.get_feature_names_out()
    if hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_, dtype=float))
    elif hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "get_feature_importance"):
        values = np.asarray(model.get_feature_importance(), dtype=float)
    else:
        values = np.zeros(len(feature_names), dtype=float)
    imp = pd.DataFrame({"feature": feature_names, "importance": values})
    return imp.sort_values("importance", ascending=False).reset_index(drop=True)


def validate_bid_type_selection(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tournament_size: int,
) -> pd.Series:
    selected_flag = test_df["Bid Type"].notna()
    available_by_season = (
        train_df[train_df["Overall Seed"].notna()]
        .groupby("Season")["Overall Seed"]
        .apply(lambda s: tournament_size - int(s.notna().sum()))
        .to_dict()
    )
    for season, grp in test_df.groupby("Season"):
        expected = int(available_by_season[season])
        actual = int(selected_flag.loc[grp.index].sum())
        if expected != actual:
            raise ValueError(
                f"Bid Type selection mismatch for {season}: expected {expected}, got {actual}"
            )
    return selected_flag


def assign_test_seeds(
    train_df: pd.DataFrame,
    test_selected_df: pd.DataFrame,
    score: np.ndarray,
    tournament_size: int,
) -> np.ndarray:
    final_seed = np.zeros(len(test_selected_df), dtype=float)
    seeded_train_by_season = (
        train_df[train_df["Overall Seed"].notna()]
        .groupby("Season")["Overall Seed"]
        .apply(lambda s: set(s.astype(int)))
        .to_dict()
    )
    for season, idx in test_selected_df.groupby("Season").indices.items():
        season_idx = np.array(idx, dtype=int)
        known = seeded_train_by_season.get(season, set())
        available = sorted([s for s in range(1, tournament_size + 1) if s not in known])
        assigned = np.empty(len(season_idx), dtype=float)
        assigned[np.argsort(-score[season_idx])] = np.array(available, dtype=float)
        final_seed[season_idx] = assigned
    return final_seed


def make_cv_splits(train_df: pd.DataFrame, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    strata = (
        train_df["Season"].astype(str)
        + "_"
        + train_df["Overall Seed"].notna().astype(int).astype(str)
    )
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return list(splitter.split(train_df, strata))


def run_experiment(
    experiment_name: str,
    model_family: str,
    output_root: Path,
    train_path: Path,
    test_path: Path,
    tournament_size: int = 68,
) -> dict[str, Any]:
    config = EXPERIMENTS[experiment_name]
    output_dir = output_root / experiment_name / model_family
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.status == "skipped":
        summary = {
            "experiment_name": experiment_name,
            "model_family": model_family,
            "status": config.status,
            "skip_reason": config.skip_reason,
            "feature_blocks": list(config.feature_blocks),
            "notes": config.notes,
        }
        (output_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    if not model_is_available(model_family):
        summary = {
            "experiment_name": experiment_name,
            "model_family": model_family,
            "status": "skipped",
            "skip_reason": f"DEPENDENCY_NOT_INSTALLED_{model_family.upper()}",
            "feature_blocks": list(config.feature_blocks),
            "notes": config.notes,
        }
        (output_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if not config.use_bid_type_selection:
        raise NotImplementedError("Non-BidType selection is not implemented in v5 runner")

    test_selected_flag = validate_bid_type_selection(train_df, test_df, tournament_size)
    train_selected_flag = train_df["Overall Seed"].notna()

    train_selected = train_df.loc[train_selected_flag].copy()
    test_selected = test_df.loc[test_selected_flag].copy()

    oof_score = pd.Series(np.nan, index=train_df.index, dtype=float)
    oof_seed_pred = pd.Series(0.0, index=train_df.index, dtype=float)
    fold_rows: list[dict[str, Any]] = []
    fold_assignments = np.full(len(train_df), -1, dtype=int)

    for fold, (train_idx, valid_idx) in enumerate(make_cv_splits(train_df), start=1):
        train_full = train_df.iloc[train_idx].copy()
        valid_full = train_df.iloc[valid_idx].copy()
        train_sel = train_full[train_full["Overall Seed"].notna()].copy()
        valid_sel = valid_full[valid_full["Overall Seed"].notna()].copy()

        artifacts: FeatureArtifacts = fit_feature_artifacts(train_sel, config.feature_blocks)
        x_train = build_feature_frame(
            train_sel,
            feature_blocks=config.feature_blocks,
            include_team_feature=config.include_team_feature,
            include_bid_type_feature=config.include_bid_type_feature,
            artifacts=artifacts,
        )
        x_valid = build_feature_frame(
            valid_sel,
            feature_blocks=config.feature_blocks,
            include_team_feature=config.include_team_feature,
            include_bid_type_feature=config.include_bid_type_feature,
            artifacts=artifacts,
        )
        y_train = train_sel["Overall Seed"].to_numpy(dtype=float)
        y_valid = valid_sel["Overall Seed"].to_numpy(dtype=float)

        pre = build_preprocessor(x_train)
        x_train_t = pre.fit_transform(x_train)
        x_valid_t = pre.transform(x_valid)

        model = fit_model(get_model(model_family), x_train_t, y_train)
        raw_seed = predict_model(model, x_valid_t)
        valid_score = -raw_seed
        primary_rmse, assigned_seed = season_rank_rmse(valid_sel["Season"], y_valid, valid_score)

        valid_pred_full = np.zeros(len(valid_full), dtype=float)
        valid_pred_full[valid_full["Overall Seed"].notna().to_numpy()] = assigned_seed
        secondary_rmse = rmse(
            valid_full["Overall Seed"].fillna(0.0).to_numpy(dtype=float),
            valid_pred_full,
        )

        selected_valid_idx = valid_sel.index
        oof_score.loc[selected_valid_idx] = valid_score
        oof_seed_pred.loc[selected_valid_idx] = assigned_seed
        fold_assignments[valid_idx] = fold

        fold_rows.append(
            {
                "fold": fold,
                "selected_rows": int(len(valid_sel)),
                "total_rows": int(len(valid_full)),
                "season_rank_rmse": primary_rmse,
                "full_rmse_zero": secondary_rmse,
            }
        )

    train_oof = train_df[["RecordID", "Season", "Team", "Conference", "Bid Type", "Overall Seed"]].copy()
    train_oof["Fold"] = fold_assignments
    train_oof["IsSelected"] = train_df["Overall Seed"].notna().astype(int)
    train_oof["Score"] = oof_score
    train_oof["PredAssignedSeed"] = oof_seed_pred
    train_oof["PredFullTask"] = oof_seed_pred

    artifacts = fit_feature_artifacts(train_selected, config.feature_blocks)
    x_train_full = build_feature_frame(
        train_selected,
        feature_blocks=config.feature_blocks,
        include_team_feature=config.include_team_feature,
        include_bid_type_feature=config.include_bid_type_feature,
        artifacts=artifacts,
    )
    x_test_selected = build_feature_frame(
        test_selected,
        feature_blocks=config.feature_blocks,
        include_team_feature=config.include_team_feature,
        include_bid_type_feature=config.include_bid_type_feature,
        artifacts=artifacts,
    )
    y_train_full = train_selected["Overall Seed"].to_numpy(dtype=float)

    pre = build_preprocessor(x_train_full)
    x_train_full_t = pre.fit_transform(x_train_full)
    x_test_selected_t = pre.transform(x_test_selected)
    final_model = fit_model(get_model(model_family), x_train_full_t, y_train_full)
    test_score = -predict_model(final_model, x_test_selected_t)
    assigned_test_seed = assign_test_seeds(train_df, test_selected, test_score, tournament_size)

    full_test_pred = np.zeros(len(test_df), dtype=float)
    full_test_pred[test_selected_flag.to_numpy()] = assigned_test_seed

    test_predictions = test_df[["RecordID", "Season", "Team", "Conference", "Bid Type"]].copy()
    test_predictions["IsSelected"] = test_selected_flag.astype(int)
    test_predictions["Score"] = np.nan
    test_predictions.loc[test_selected.index, "Score"] = test_score
    test_predictions["PredAssignedSeed"] = full_test_pred

    feature_importance = extract_feature_importance(final_model, pre)
    fold_metrics = pd.DataFrame(fold_rows)
    summary = {
        "experiment_name": experiment_name,
        "model_family": model_family,
        "status": "completed",
        "feature_blocks": list(config.feature_blocks),
        "include_bid_type_feature": config.include_bid_type_feature,
        "include_team_feature": config.include_team_feature,
        "use_bid_type_selection": config.use_bid_type_selection,
        "notes": config.notes,
        "season_rank_rmse_mean": float(fold_metrics["season_rank_rmse"].mean()),
        "season_rank_rmse_std": float(fold_metrics["season_rank_rmse"].std(ddof=0)),
        "full_rmse_zero_mean": float(fold_metrics["full_rmse_zero"].mean()),
        "full_rmse_zero_std": float(fold_metrics["full_rmse_zero"].std(ddof=0)),
        "selected_count_test": int(test_selected_flag.sum()),
        "non_zero_count_test": int((full_test_pred > 0).sum()),
    }

    train_oof.to_csv(output_dir / "oof_predictions.csv", index=False)
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame({"RecordID": test_df["RecordID"], "Overall Seed": full_test_pred}).to_csv(
        output_dir / "submission.csv", index=False
    )
    (output_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2))

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument("--model-family", required=True)
    parser.add_argument("--train-path", default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--test-path", default=str(DEFAULT_TEST_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--tournament-size", type=int, default=68)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_experiment(
        experiment_name=args.experiment,
        model_family=args.model_family,
        output_root=Path(args.output_root),
        train_path=Path(args.train_path),
        test_path=Path(args.test_path),
        tournament_size=args.tournament_size,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
