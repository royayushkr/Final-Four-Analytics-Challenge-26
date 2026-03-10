#!/usr/bin/env python3
"""Leakage-safe training pipeline for NCAA seed prediction."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor

MONTH_TO_INT = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}

RECORD_COLS = [
    "WL",
    "Conf.Record",
    "Non-ConferenceRecord",
    "RoadWL",
    "Quadrant1",
    "Quadrant2",
    "Quadrant3",
    "Quadrant4",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "NCAA_Seed_Training_Set2.0.csv"
DEFAULT_TEST_PATH = PROJECT_ROOT / "data" / "raw" / "NCAA_Seed_Test_Set2.0.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "submissions" / "generated"


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def parse_wins_losses(value: object) -> tuple[float, float]:
    """Parse win-loss strings including spreadsheet-converted forms."""
    if pd.isna(value):
        return np.nan, np.nan

    text = str(value).strip()

    standard = re.match(r"^(\d+)-(\d+)$", text)
    if standard:
        return float(standard.group(1)), float(standard.group(2))

    converted_num_mon = re.match(r"^(\d+)-([A-Za-z]{3})$", text)
    if converted_num_mon:
        wins = float(converted_num_mon.group(1))
        losses = float(MONTH_TO_INT.get(converted_num_mon.group(2), np.nan))
        return wins, losses

    converted_mon_zero = re.match(r"^([A-Za-z]{3})-00$", text)
    if converted_mon_zero:
        wins = float(MONTH_TO_INT.get(converted_mon_zero.group(1), np.nan))
        return wins, 0.0

    return np.nan, np.nan


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["SeasonStart"] = out["Season"].str.split("-").str[0].astype(float)

    for col in RECORD_COLS:
        parsed = out[col].apply(parse_wins_losses)
        out[f"{col}_W"] = parsed.str[0]
        out[f"{col}_L"] = parsed.str[1]
        out[f"{col}_G"] = out[f"{col}_W"] + out[f"{col}_L"]
        out[f"{col}_PCT"] = np.where(
            out[f"{col}_G"] > 0, out[f"{col}_W"] / out[f"{col}_G"], np.nan
        )

    out["NET_Improvement"] = out["PrevNET"] - out["NET Rank"]
    out["OppNetDiff"] = out["AvgOppNETRank"] - out["AvgOppNET"]
    out["SOS_Diff"] = out["NETNonConfSOS"] - out["NETSOS"]
    return out


def to_features(
    df: pd.DataFrame, include_bid_type: bool = False, include_team: bool = True
) -> pd.DataFrame:
    engineered = feature_engineering(df)
    drop_cols = ["RecordID", "Overall Seed", *RECORD_COLS]
    if not include_bid_type:
        drop_cols.append("Bid Type")
    if not include_team:
        drop_cols.append("Team")
    return engineered.drop(columns=drop_cols, errors="ignore")


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = [c for c in x.columns if x[c].dtype == "object"]
    numeric_cols = [c for c in x.columns if c not in categorical_cols]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("impute", SimpleImputer(strategy="median"))]), numeric_cols),
            (
                "cat",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(strategy="constant", fill_value="__MISSING__"),
                        ),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )


def build_models(preprocessor: ColumnTransformer) -> tuple[Pipeline, Pipeline]:
    clf = Pipeline(
        steps=[
            ("pre", clone(preprocessor)),
            (
                "model",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_estimators=500,
                    max_depth=4,
                    learning_rate=0.03,
                    subsample=0.9,
                    colsample_bytree=0.85,
                    random_state=42,
                ),
            ),
        ]
    )

    reg = Pipeline(
        steps=[
            ("pre", clone(preprocessor)),
            (
                "model",
                XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=700,
                    max_depth=3,
                    learning_rate=0.02,
                    subsample=0.95,
                    colsample_bytree=0.9,
                    reg_alpha=0.0,
                    reg_lambda=1.5,
                    random_state=42,
                ),
            ),
        ]
    )

    return clf, reg


def apply_strategy(
    selected_probs: np.ndarray,
    seed_preds: np.ndarray,
    strategy: str,
    threshold: float,
) -> np.ndarray:
    seed_preds = np.clip(seed_preds, 1.0, 68.0)

    if strategy == "hard":
        final_preds = np.where(selected_probs >= threshold, seed_preds, 0.0)
    elif strategy == "soft":
        final_preds = selected_probs * seed_preds
    elif strategy == "soft-threshold":
        final_preds = np.where(selected_probs >= threshold, selected_probs * seed_preds, 0.0)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return np.clip(final_preds, 0.0, 68.0)


def choose_best_strategy(
    y_all_zero: np.ndarray,
    selected_probs: np.ndarray,
    seed_preds: np.ndarray,
) -> tuple[str, float, float]:
    candidates: list[tuple[str, float]] = [("soft", 0.5)]
    for threshold in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65):
        candidates.append(("hard", threshold))
        candidates.append(("soft-threshold", threshold))

    best_score = float("inf")
    best_strategy = "soft"
    best_threshold = 0.5
    for strategy, threshold in candidates:
        preds = apply_strategy(selected_probs, seed_preds, strategy, threshold)
        score = rmse(y_all_zero, preds)
        if score < best_score:
            best_score = score
            best_strategy = strategy
            best_threshold = threshold
    return best_strategy, best_threshold, best_score


def cross_validate(
    train_df: pd.DataFrame,
    folds: int,
    include_bid_type: bool,
    include_team: bool,
    strategy: str,
    threshold: float,
) -> dict[str, Any]:
    x_all = to_features(
        train_df, include_bid_type=include_bid_type, include_team=include_team
    )
    y_seed = train_df["Overall Seed"]
    y_selected = y_seed.notna().astype(int)
    y_all_zero = y_seed.fillna(0.0).to_numpy()

    preprocessor = build_preprocessor(x_all)
    clf, reg = build_models(preprocessor)

    group_kfold = GroupKFold(n_splits=folds)
    groups = train_df["Season"]

    selected_probs = np.zeros(len(train_df))
    seed_preds = np.zeros(len(train_df))

    for train_idx, valid_idx in group_kfold.split(x_all, y_selected, groups):
        x_train = x_all.iloc[train_idx]
        x_valid = x_all.iloc[valid_idx]

        y_train_selected = y_selected.iloc[train_idx]
        y_train_seed = y_seed.iloc[train_idx]
        seed_mask = y_train_seed.notna().to_numpy()

        clf.fit(x_train, y_train_selected)
        selected_probs[valid_idx] = clf.predict_proba(x_valid)[:, 1]

        reg.fit(x_train.iloc[seed_mask], y_train_seed.iloc[seed_mask])
        seed_preds[valid_idx] = reg.predict(x_valid)

    if strategy == "auto":
        best_strategy, best_threshold, final_rmse = choose_best_strategy(
            y_all_zero=y_all_zero, selected_probs=selected_probs, seed_preds=seed_preds
        )
    else:
        best_strategy = strategy
        best_threshold = threshold
        final_preds = apply_strategy(selected_probs, seed_preds, best_strategy, best_threshold)
        final_rmse = rmse(y_all_zero, final_preds)

    final_preds = apply_strategy(selected_probs, seed_preds, best_strategy, best_threshold)
    seed_preds_clipped = np.clip(seed_preds, 1.0, 68.0)

    selected_mask = y_seed.notna().to_numpy()
    cls_acc = float(((selected_probs >= 0.5).astype(int) == y_selected.to_numpy()).mean())
    selected_seed_rmse = rmse(
        y_seed[selected_mask].to_numpy(), seed_preds_clipped[selected_mask]
    )

    metrics = {
        "selection_accuracy": cls_acc,
        "seed_rmse_selected": selected_seed_rmse,
        "final_rmse_all": final_rmse,
        "selected_count_oof_at_0_5": int((selected_probs >= 0.5).sum()),
        "prediction_strategy": best_strategy,
        "selection_threshold": best_threshold,
        "include_bid_type": include_bid_type,
        "include_team": include_team,
    }

    return metrics


def fit_and_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    strategy: str,
    threshold: float,
    include_bid_type: bool = False,
    include_team: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_train = to_features(
        train_df, include_bid_type=include_bid_type, include_team=include_team
    )
    x_test = to_features(test_df, include_bid_type=include_bid_type, include_team=include_team)

    y_seed = train_df["Overall Seed"]
    y_selected = y_seed.notna().astype(int)
    selected_mask = y_seed.notna().to_numpy()

    preprocessor = build_preprocessor(x_train)
    clf, reg = build_models(preprocessor)

    clf.fit(x_train, y_selected)
    reg.fit(x_train.iloc[selected_mask], y_seed.iloc[selected_mask])

    selected_probs_test = clf.predict_proba(x_test)[:, 1]
    seed_preds_test = np.clip(reg.predict(x_test), 1.0, 68.0)
    final_preds_test = apply_strategy(
        selected_probs_test, seed_preds_test, strategy=strategy, threshold=threshold
    )

    return selected_probs_test, seed_preds_test, final_preds_test


def write_submission(record_ids: pd.Series, preds: np.ndarray, path: Path) -> None:
    submission = pd.DataFrame({"RecordID": record_ids, "Overall Seed": preds})
    submission.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-path",
        default=str(DEFAULT_TRAIN_PATH),
        help="Training CSV path",
    )
    parser.add_argument(
        "--test-path",
        default=str(DEFAULT_TEST_PATH),
        help="Test CSV path",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where submission files are written",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of GroupKFold splits for local validation",
    )
    parser.add_argument(
        "--prediction-strategy",
        choices=["auto", "hard", "soft", "soft-threshold"],
        default="auto",
        help="How to combine selection probability and seed prediction.",
    )
    parser.add_argument(
        "--selection-threshold",
        type=float,
        default=0.5,
        help="Threshold used for hard/soft-threshold prediction strategies.",
    )
    parser.add_argument(
        "--include-bid-type",
        action="store_true",
        help="Include Bid Type as a model feature (off by default to avoid leakage risk).",
    )
    parser.add_argument(
        "--exclude-team",
        action="store_true",
        help="Drop Team feature from the model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)

    if train_df["Overall Seed"].notna().sum() == 0:
        raise ValueError("No non-null seed labels found in training data.")

    include_team = not args.exclude_team
    metrics = cross_validate(
        train_df=train_df,
        folds=args.cv_folds,
        include_bid_type=args.include_bid_type,
        include_team=include_team,
        strategy=args.prediction_strategy,
        threshold=args.selection_threshold,
    )

    chosen_strategy = str(metrics["prediction_strategy"])
    chosen_threshold = float(metrics["selection_threshold"])

    probs, seed_only_preds, final_preds = fit_and_predict(
        train_df=train_df,
        test_df=test_df,
        strategy=chosen_strategy,
        threshold=chosen_threshold,
        include_bid_type=args.include_bid_type,
        include_team=include_team,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_only_path = output_dir / "submission_seed_only_no_leak.csv"
    final_path = output_dir / "submission_no_leak_v2.csv"
    diagnostics_path = output_dir / "test_selection_probabilities_no_leak.csv"
    metrics_path = output_dir / "cv_metrics_no_leak.json"

    write_submission(test_df["RecordID"], seed_only_preds, seed_only_path)
    write_submission(test_df["RecordID"], final_preds, final_path)
    pd.DataFrame(
        {"RecordID": test_df["RecordID"], "SelectedProbability": probs}
    ).to_csv(diagnostics_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print("CV Metrics (GroupKFold by Season):")
    for key, value in metrics.items():
        print(f"- {key}: {value}")

    print(f"Wrote: {seed_only_path}")
    print(f"Wrote: {final_path}")
    print(f"Wrote: {diagnostics_path}")
    print(f"Wrote: {metrics_path}")


if __name__ == "__main__":
    main()
