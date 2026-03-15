#!/usr/bin/env python3
"""Final historical evaluation and 2026 seed prediction pipeline.

This extends the private v7 path by using a fully labeled historical universe:
- official training rows + historical same-season test rows,
- externally collected true overall seeds for all historical selected teams,
- a predicted AQ/AL/NONE bid-class model,
- quota-aware selection with a final 31 AQ / 37 AL composition for 2026.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from build_historical_true_seeds import TEAM_CANONICAL
from private_features import make_feature_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "NCAA_Seed_Training_Set2.0.csv"
HIST_TEST_PATH = PROJECT_ROOT / "data" / "raw" / "NCAA_Seed_Test_Set2.0.csv"
TRUE_SEED_PATH = PROJECT_ROOT / "data" / "external" / "historical_true_seeds_2021_2025.csv"
FINAL_TEST_PATH = PROJECT_ROOT / "data" / "raw" / "NCAA_Seed_Test_Set_2026_20260315.csv"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "final_20260315_eval"
SUBMISSION_PATH = PROJECT_ROOT / "submissions" / "final" / "submission_2026_20260315.csv"
RIDGE_ALPHA_GRID = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
BID_CLASSES = ["NONE", "AQ", "AL"]
SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
FINAL_AQ_TARGET = 31
FINAL_AL_TARGET = 37
SEED_LOCK_COUNT = 24


@dataclass
class CandidateSpec:
    name: str
    bid_model: str
    seed_model: str
    use_cfa: bool = False
    use_seed_ensemble: bool = False
    use_catboost_seed: bool = False
    weight_seed: float = 0.7
    weight_include: float = 0.3
    weight_cfa: float = 0.0
    quota_boost: float = 0.25
    notes: str = ""
    status: str = "active"
    skip_reason: str = ""


@dataclass
class ModelBundle:
    name: str
    model: Any
    preprocessor: ColumnTransformer | None
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    medians: dict[str, float] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--submission-path", type=Path, default=SUBMISSION_PATH)
    parser.add_argument("--skip-seed-build", action="store_true")
    return parser.parse_args()


def canonicalize_team(team: str) -> str:
    team = TEAM_CANONICAL.get(team, team)
    team = team.replace("Saint Mary's", "Saint Mary's (CA)")
    team = team.replace("Mount St. Mary's", "Mount St. Mary's")
    return team


def team_key(team: str) -> str:
    team = canonicalize_team(team)
    team = team.lower().replace("&", "and")
    team = team.replace("saint ", "st ")
    team = team.replace("st.", "st")
    team = team.replace("uc santa barbara", "ucsantabarbara")
    team = team.replace("southern california", "southerncalifornia")
    team = team.replace("col. of charleston", "collegeofcharleston")
    team = team.replace("college of charleston", "collegeofcharleston")
    team = team.replace("mount st mary's", "mountstmarys")
    team = team.replace("saint mary's (ca)", "saintmarysca")
    team = team.replace("fdu", "fairleighdickinson")
    team = team.replace("(fl)", "fl").replace("(oh)", "oh").replace("(ny)", "ny").replace("(ca)", "ca")
    team = re.sub(r"[^a-z0-9]+", "", team)
    return team


def build_true_seed_file_if_missing(skip_seed_build: bool) -> None:
    if TRUE_SEED_PATH.exists() or skip_seed_build:
        return
    import subprocess
    subprocess.run(["python3", str(PROJECT_ROOT / "code" / "build_historical_true_seeds.py")], check=True)


def load_labeled_historical_universe(skip_seed_build: bool) -> pd.DataFrame:
    build_true_seed_file_if_missing(skip_seed_build)
    train = pd.read_csv(TRAIN_PATH)
    hist_test = pd.read_csv(HIST_TEST_PATH)
    hist_test["Overall Seed"] = np.nan
    combined = pd.concat([train, hist_test], ignore_index=True, sort=False)

    true_seed = pd.read_csv(TRUE_SEED_PATH)
    combined["team_key"] = combined["Team"].map(team_key)
    true_seed["team_key"] = true_seed["TeamCanonical"].map(team_key)

    merged = combined.merge(
        true_seed[["Season", "team_key", "TrueSeed", "IsSelected"]],
        on=["Season", "team_key"],
        how="left",
    )
    merged["TrueSeed"] = merged["TrueSeed"].fillna(0).astype(int)
    merged["IsSelected"] = merged["IsSelected"].fillna(0).astype(int)

    # Fill the historical selected teams' seed labels into Overall Seed for easier reuse.
    merged.loc[merged["TrueSeed"] > 0, "Overall Seed"] = merged.loc[merged["TrueSeed"] > 0, "TrueSeed"]

    merged["BidClass"] = np.where(merged["IsSelected"] == 0, "NONE", merged["Bid Type"].fillna("UNKNOWN"))
    if (merged.loc[merged["IsSelected"] == 1, "BidClass"] == "UNKNOWN").any():
        missing = merged.loc[(merged["IsSelected"] == 1) & (merged["BidClass"] == "UNKNOWN"), ["Season", "Team"]]
        raise ValueError(f"Selected historical rows missing Bid Type labels:\n{missing.to_string(index=False)}")

    per_season = merged.groupby("Season")["TrueSeed"].apply(lambda s: int((s > 0).sum())).to_dict()
    expected = {season: 68 for season in SEASONS}
    if per_season != expected:
        raise ValueError(f"Historical selected counts mismatch: {per_season}")
    return merged.drop(columns=["team_key"])


def load_weekly_2026_snapshots() -> dict[str, pd.DataFrame]:
    snapshots: dict[str, pd.DataFrame] = {}
    for path in sorted(RAW_DIR.glob("NCAA_Seed_Test_Set_2026_*.csv")):
        stamp = path.stem.split("_")[-1]
        snapshots[stamp] = pd.read_csv(path)
    return snapshots


def build_linear_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
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


def build_tree_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
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


def prepare_catboost_frames(
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


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    std = float(np.nanstd(values))
    if not np.isfinite(std) or std == 0.0:
        return np.zeros_like(values)
    return (values - float(np.nanmean(values))) / std


def softmax_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    arr = arr - np.max(arr, axis=1, keepdims=True)
    exp = np.exp(arr)
    return exp / exp.sum(axis=1, keepdims=True)


def build_component_scores(df: pd.DataFrame) -> dict[str, np.ndarray]:
    net_cols = [col for col in ["NET Rank_INV", "PrevNET_INV", "NET Rank_INV_season_pct", "PrevNET_INV_season_pct"] if col in df.columns]
    resume_cols = [col for col in ["quality_win_score", "resume_balance", "top_resume_index", "q12_wins", "q12_win_rate"] if col in df.columns]
    sos_cols = [col for col in ["NETSOS_INV", "NETNonConfSOS_INV", "AvgOppNETRank_INV", "road_quality", "nonconf_quality"] if col in df.columns]
    quadrant_cols = [col for col in ["Quadrant1_W", "Quadrant2_W", "q12_wins", "q34_losses_neg", "bad_loss_penalty_neg"] if col in df.columns]

    def combine(cols: list[str]) -> np.ndarray:
        if not cols:
            return np.zeros(len(df), dtype=float)
        mat = np.column_stack([pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy(dtype=float) for c in cols])
        return zscore(mat.mean(axis=1))

    return {
        "net": combine(net_cols),
        "resume": combine(resume_cols),
        "sos": combine(sos_cols),
        "quadrant": combine(quadrant_cols),
    }


def reciprocal_rank_fusion(score_map: dict[str, np.ndarray], weights: dict[str, float], k: int = 60) -> np.ndarray:
    fused = np.zeros(len(next(iter(score_map.values()))), dtype=float)
    for name, scores in score_map.items():
        weight = weights.get(name, 0.0)
        if weight == 0.0:
            continue
        rank = pd.Series(-scores).rank(method="first", ascending=False).to_numpy(dtype=float)
        fused += weight / (k + rank)
    return fused


def bucket_accuracy(y_true_seed: np.ndarray, y_pred_seed: np.ndarray, bucket_size: int = 4) -> float:
    mask = y_true_seed > 0
    if not mask.any():
        return float("nan")
    true_bucket = ((y_true_seed[mask] - 1) // bucket_size).astype(int)
    pred_bucket = ((np.maximum(y_pred_seed[mask], 1) - 1) // bucket_size).astype(int)
    return float((true_bucket == pred_bucket).mean())


def top16_bucket_accuracy(y_true_seed: np.ndarray, y_pred_seed: np.ndarray) -> float:
    mask = (y_true_seed > 0) & (y_true_seed <= 16)
    if not mask.any():
        return float("nan")
    true_bucket = ((y_true_seed[mask] - 1) // 4).astype(int)
    pred_bucket = ((np.clip(y_pred_seed[mask], 1, 16) - 1) // 4).astype(int)
    return float((true_bucket == pred_bucket).mean())


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_true.astype(float) - y_prob.astype(float)) ** 2))


def fit_bid_classifier(
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
    family: str,
) -> ModelBundle:
    x_train = train_frame[feature_columns].copy()
    y_train = pd.Categorical(train_frame["BidClass"], categories=BID_CLASSES).codes

    if family == "lgbm":
        pre = build_tree_preprocessor(numeric_columns, categorical_columns)
        x_t = pre.fit_transform(x_train)
        model = LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=400,
            learning_rate=0.04,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            random_state=42,
            verbosity=-1,
            force_col_wise=True,
        )
        model.fit(x_t, y_train)
        return ModelBundle(family, model, pre, feature_columns, numeric_columns, categorical_columns)

    if family == "logit":
        pre = build_linear_preprocessor(numeric_columns, categorical_columns)
        x_t = pre.fit_transform(x_train)
        model = LogisticRegression(
            max_iter=5000,
            C=0.75,
            class_weight="balanced",
            random_state=42,
        )
        model.fit(x_t, y_train)
        return ModelBundle(family, model, pre, feature_columns, numeric_columns, categorical_columns)

    raise ValueError(f"Unknown bid model family: {family}")


def predict_bid_classifier(bundle: ModelBundle, eval_frame: pd.DataFrame) -> pd.DataFrame:
    x_eval = eval_frame[bundle.feature_columns].copy()
    assert bundle.preprocessor is not None
    x_t = bundle.preprocessor.transform(x_eval)
    proba = np.asarray(bundle.model.predict_proba(x_t), dtype=float)
    if proba.shape[1] != 3:
        proba = softmax_rows(proba)
    return pd.DataFrame(
        {
            "P_NONE": proba[:, 0],
            "P_AQ": proba[:, 1],
            "P_AL": proba[:, 2],
        },
        index=eval_frame.index,
    )


def fit_seed_model(
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
    family: str,
) -> ModelBundle:
    selected = train_frame[train_frame["TrueSeed"] > 0].copy().reset_index(drop=True)
    x_train = selected[feature_columns].copy()
    y_train = 69.0 - selected["TrueSeed"].to_numpy(dtype=float)

    if family == "ridge":
        pre = build_linear_preprocessor(numeric_columns, categorical_columns)
        x_t = pre.fit_transform(x_train)
        model = RidgeCV(alphas=RIDGE_ALPHA_GRID)
        model.fit(x_t, y_train)
        return ModelBundle(family, model, pre, feature_columns, numeric_columns, categorical_columns)

    if family == "lgbm":
        pre = build_tree_preprocessor(numeric_columns, categorical_columns)
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
        return ModelBundle(family, model, pre, feature_columns, numeric_columns, categorical_columns)

    if family == "catboost":
        train_cb, _, cat_idx, medians = prepare_catboost_frames(x_train, x_train, numeric_columns, categorical_columns)
        model = CatBoostRegressor(
            loss_function="RMSE",
            iterations=500,
            depth=5,
            learning_rate=0.03,
            random_seed=42,
            verbose=False,
        )
        model.fit(train_cb, y_train, cat_features=cat_idx)
        return ModelBundle(family, model, None, feature_columns, numeric_columns, categorical_columns, medians)

    raise ValueError(f"Unknown seed model family: {family}")


def predict_seed_model(bundle: ModelBundle, eval_frame: pd.DataFrame) -> np.ndarray:
    x_eval = eval_frame[bundle.feature_columns].copy()
    if bundle.name == "catboost":
        medians = bundle.medians or {}
        for col in bundle.numeric_columns:
            x_eval[col] = pd.to_numeric(x_eval[col], errors="coerce").fillna(medians.get(col, 0.0))
        for col in bundle.categorical_columns:
            x_eval[col] = x_eval[col].fillna("__MISSING__").astype(str)
        return np.asarray(bundle.model.predict(x_eval), dtype=float)
    assert bundle.preprocessor is not None
    x_t = bundle.preprocessor.transform(x_eval)
    return np.asarray(bundle.model.predict(x_t), dtype=float)


def combine_scores(seed_score: np.ndarray, include_prob: np.ndarray, cfa_score: np.ndarray | None, candidate: CandidateSpec) -> np.ndarray:
    combined = candidate.weight_seed * zscore(seed_score) + candidate.weight_include * zscore(include_prob)
    if candidate.use_cfa and cfa_score is not None:
        combined += candidate.weight_cfa * zscore(cfa_score)
    return combined


def assign_seeds_with_bid_quota(
    df: pd.DataFrame,
    final_score: np.ndarray,
    p_aq: np.ndarray,
    p_al: np.ndarray,
    aq_target: int,
    al_target: int,
    quota_boost: float,
) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out["FinalScore"] = final_score
    out["P_AQ"] = p_aq
    out["P_AL"] = p_al
    out["IncludePriority"] = zscore(final_score)
    out["AQPriority"] = zscore(final_score) + quota_boost * zscore(p_aq)
    out["ALPriority"] = zscore(final_score) + quota_boost * zscore(p_al)
    out["BidMarginAQ"] = zscore(p_aq) - zscore(p_al)

    field_size = aq_target + al_target
    ordered_idx = (
        out.sort_values(["IncludePriority", "NET Rank", "RecordID"], ascending=[False, True, True])
        .index
    )
    # Protect the top of the board from quota distortion; use AQ/AL structure
    # primarily to shape the back half of the field where auto-bid dynamics live.
    seed_lock = min(SEED_LOCK_COUNT, field_size, aq_target, al_target)
    locked_idx = ordered_idx[:seed_lock]

    out["PredBidType"] = "NONE"
    locked = out.loc[locked_idx].copy()
    locked["PredBidType"] = np.where(locked["P_AQ"] >= locked["P_AL"], "AQ", "AL")
    aq_locked = int((locked["PredBidType"] == "AQ").sum())
    al_locked = int((locked["PredBidType"] == "AL").sum())

    remaining_aq = max(aq_target - aq_locked, 0)
    remaining_al = max(al_target - al_locked, 0)

    remaining_pool = out.drop(index=locked_idx)
    aq_pick = (
        remaining_pool.sort_values(
            ["AQPriority", "FinalScore", "NET Rank", "RecordID"],
            ascending=[False, False, True, True],
        )
        .head(remaining_aq)
        .index
    )
    remaining_pool = remaining_pool.drop(index=aq_pick)
    al_pick = (
        remaining_pool.sort_values(
            ["ALPriority", "FinalScore", "NET Rank", "RecordID"],
            ascending=[False, False, True, True],
        )
        .head(remaining_al)
        .index
    )
    selected_idx = list(locked_idx) + list(aq_pick) + list(al_pick)

    out.loc[locked.index, "PredBidType"] = locked["PredBidType"].to_numpy()
    out.loc[aq_pick, "PredBidType"] = "AQ"
    out.loc[al_pick, "PredBidType"] = "AL"
    out["PredictedSeed"] = 0

    seeded = (
        out.loc[selected_idx]
        .sort_values(["FinalScore", "NET Rank", "RecordID"], ascending=[False, True, True])
        .copy()
    )
    seeded["PredictedSeed"] = np.arange(1, len(seeded) + 1)
    out.loc[seeded.index, "PredictedSeed"] = seeded["PredictedSeed"].to_numpy(dtype=int)
    out["RankOrder"] = out["PredictedSeed"].replace(0, 9999)
    return out


def compute_metrics(eval_df: pd.DataFrame, scored: pd.DataFrame, fit_seconds: float) -> dict[str, float]:
    y_true = eval_df["TrueSeed"].to_numpy(dtype=float)
    y_pred = scored["PredictedSeed"].to_numpy(dtype=float)
    mask_selected = y_true > 0
    pred_selected = y_pred > 0
    tp = int((mask_selected & pred_selected).sum())
    fp = int((~mask_selected & pred_selected).sum())
    fn = int((mask_selected & ~pred_selected).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else (2.0 * precision * recall) / (precision + recall)

    spearman = float(pd.Series(y_true[mask_selected]).corr(pd.Series(y_pred[mask_selected]), method="spearman")) if mask_selected.sum() > 1 else float("nan")
    selected_rmse = rmse(y_true[mask_selected], y_pred[mask_selected]) if mask_selected.any() else float("nan")
    brier = brier_score(mask_selected.astype(int), scored["P_AQ"].to_numpy(dtype=float) + scored["P_AL"].to_numpy(dtype=float))

    return {
        "FullRMSE": rmse(y_true, y_pred),
        "FullMAE": float(np.mean(np.abs(y_true - y_pred))),
        "SelectedRMSE": selected_rmse,
        "InclusionPrecision": float(precision),
        "InclusionRecall": float(recall),
        "InclusionF1": float(f1),
        "Bucket4Accuracy": bucket_accuracy(y_true, y_pred, bucket_size=4),
        "Top16BucketAccuracy": top16_bucket_accuracy(y_true, y_pred),
        "SelectedSpearman": spearman,
        "BrierSelected": brier,
        "TrainSeconds": float(fit_seconds),
    }


def candidate_specs() -> list[CandidateSpec]:
    specs = [
        CandidateSpec(
            name="exp1_lgbm_bid_ridge_seed",
            bid_model="lgbm",
            seed_model="ridge",
            notes="Highest-value baseline from the plan.",
        ),
        CandidateSpec(
            name="exp2_lgbm_bid_lgbm_seed",
            bid_model="lgbm",
            seed_model="lgbm",
            notes="Nonlinear seed stage with the same bid-class model.",
        ),
        CandidateSpec(
            name="exp3_logit_bid_ridge_seed",
            bid_model="logit",
            seed_model="ridge",
            notes="Simpler selector to test bubble stability.",
        ),
        CandidateSpec(
            name="exp4_lgbm_bid_ridge_seed_cfa",
            bid_model="lgbm",
            seed_model="ridge",
            use_cfa=True,
            weight_seed=0.60,
            weight_include=0.20,
            weight_cfa=0.20,
            notes="Primary CFA / weighted RRF experiment.",
        ),
        CandidateSpec(
            name="exp5_lgbm_bid_seed_ensemble",
            bid_model="lgbm",
            seed_model="ridge",
            use_seed_ensemble=True,
            notes="Simple seed ensemble: ridge + lgbm.",
        ),
        CandidateSpec(
            name="exp6_lgbm_bid_catboost_seed",
            bid_model="lgbm",
            seed_model="catboost",
            use_catboost_seed=True,
            notes="Optional CatBoost seed stage.",
        ),
        CandidateSpec(
            name="exp7_tabnet_feasibility",
            bid_model="skip",
            seed_model="skip",
            status="skipped",
            skip_reason="pytorch_tabnet is not installed and deep tabular models are not required for the final path.",
            notes="Recorded feasibility result only.",
        ),
    ]
    return specs


def fit_candidate_scores(train_subset: pd.DataFrame, eval_subset: pd.DataFrame, candidate: CandidateSpec) -> tuple[pd.DataFrame, dict[str, Any]]:
    feature_bundle = make_feature_matrix(train_subset, eval_subset, include_conference_context=True)
    train_frame = feature_bundle.train_frame.copy()
    eval_frame = feature_bundle.eval_frame.copy()

    start = time.perf_counter()
    bid_model = fit_bid_classifier(
        train_frame=train_frame,
        feature_columns=feature_bundle.feature_columns,
        numeric_columns=feature_bundle.numeric_columns,
        categorical_columns=feature_bundle.categorical_columns,
        family=candidate.bid_model,
    )
    bid_pred = predict_bid_classifier(bid_model, eval_frame)

    seed_model_primary = fit_seed_model(
        train_frame=train_frame,
        feature_columns=feature_bundle.feature_columns,
        numeric_columns=feature_bundle.numeric_columns,
        categorical_columns=feature_bundle.categorical_columns,
        family=candidate.seed_model,
    )
    seed_score = predict_seed_model(seed_model_primary, eval_frame)

    model_details: dict[str, Any] = {
        "bid_model": candidate.bid_model,
        "seed_model": candidate.seed_model,
        "feature_count": len(feature_bundle.feature_columns),
        "numeric_count": len(feature_bundle.numeric_columns),
        "categorical_count": len(feature_bundle.categorical_columns),
    }

    if candidate.use_seed_ensemble:
        seed_model_secondary = fit_seed_model(
            train_frame=train_frame,
            feature_columns=feature_bundle.feature_columns,
            numeric_columns=feature_bundle.numeric_columns,
            categorical_columns=feature_bundle.categorical_columns,
            family="lgbm",
        )
        seed_score_secondary = predict_seed_model(seed_model_secondary, eval_frame)
        seed_score = 0.65 * zscore(seed_score) + 0.35 * zscore(seed_score_secondary)
        model_details["seed_model_secondary"] = "lgbm"

    cfa_score = None
    if candidate.use_cfa:
        component_scores = build_component_scores(eval_frame)
        ml_seed_score = zscore(seed_score)
        score_map = {
            "net": component_scores["net"],
            "resume": component_scores["resume"],
            "sos": component_scores["sos"],
            "quadrant": component_scores["quadrant"],
            "ml_seed": ml_seed_score,
        }
        cfa_score = reciprocal_rank_fusion(
            score_map,
            weights={"net": 0.20, "resume": 0.25, "sos": 0.15, "quadrant": 0.15, "ml_seed": 0.25},
        )

    include_prob = bid_pred["P_AQ"].to_numpy(dtype=float) + bid_pred["P_AL"].to_numpy(dtype=float)
    final_score = combine_scores(seed_score, include_prob, cfa_score, candidate)
    fit_seconds = time.perf_counter() - start

    scored = eval_subset[["RecordID", "Season", "Team", "Conference", "NET Rank", "TrueSeed", "BidClass"]].copy().reset_index(drop=True)
    scored = assign_seeds_with_bid_quota(
        df=scored,
        final_score=final_score,
        p_aq=bid_pred["P_AQ"].to_numpy(dtype=float),
        p_al=bid_pred["P_AL"].to_numpy(dtype=float),
        aq_target=int((eval_subset["BidClass"] == "AQ").sum()),
        al_target=int((eval_subset["BidClass"] == "AL").sum()),
        quota_boost=candidate.quota_boost,
    )
    return scored, {**model_details, "fit_seconds": fit_seconds, "feature_bundle": feature_bundle, "bid_model_bundle": bid_model, "seed_model_bundle": seed_model_primary}


def rolling_splits() -> list[tuple[list[str], str, float]]:
    return [
        (["2020-21", "2021-22"], "2022-23", 0.2),
        (["2020-21", "2021-22", "2022-23"], "2023-24", 0.3),
        (["2020-21", "2021-22", "2022-23", "2023-24"], "2024-25", 0.5),
    ]


def loocv_splits() -> list[tuple[list[str], str]]:
    return [([season for season in SEASONS if season != holdout], holdout) for holdout in SEASONS]


def summarize_weighted(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate, grp in df.groupby("candidate"):
        rows.append(
            {
                "candidate": candidate,
                "WeightedFullRMSE": float(np.average(grp["FullRMSE"], weights=grp["weight"])),
                "WeightedFullMAE": float(np.average(grp["FullMAE"], weights=grp["weight"])),
                "WeightedSelectedRMSE": float(np.average(grp["SelectedRMSE"], weights=grp["weight"])),
                "WeightedInclusionF1": float(np.average(grp["InclusionF1"], weights=grp["weight"])),
                "WeightedSelectedSpearman": float(np.average(grp["SelectedSpearman"].fillna(0.0), weights=grp["weight"])),
                "WeightedBucket4Accuracy": float(np.average(grp["Bucket4Accuracy"], weights=grp["weight"])),
                "WeightedTop16BucketAccuracy": float(np.average(grp["Top16BucketAccuracy"], weights=grp["weight"])),
                "RMSEStd": float(grp["FullRMSE"].std(ddof=0)),
                "RecentFullRMSEMax": float(grp.sort_values("valid_season").tail(2)["FullRMSE"].max()),
                "RecentInclusionF1Min": float(grp.sort_values("valid_season").tail(2)["InclusionF1"].min()),
                "MeanTrainSeconds": float(grp["TrainSeconds"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["WeightedFullRMSE", "WeightedFullMAE", "RMSEStd"]).reset_index(drop=True)


def run_primary_validation(historical_df: pd.DataFrame, candidates: list[CandidateSpec]) -> tuple[pd.DataFrame, dict[str, list[pd.DataFrame]]]:
    metric_rows: list[dict[str, Any]] = []
    scored_by_candidate: dict[str, list[pd.DataFrame]] = {candidate.name: [] for candidate in candidates if candidate.status == "active"}

    for train_seasons, valid_season, weight in rolling_splits():
        train_subset = historical_df[historical_df["Season"].isin(train_seasons)].copy().reset_index(drop=True)
        eval_subset = historical_df[historical_df["Season"] == valid_season].copy().reset_index(drop=True)
        for candidate in candidates:
            if candidate.status != "active":
                metric_rows.append({
                    "candidate": candidate.name,
                    "valid_season": valid_season,
                    "weight": weight,
                    "status": candidate.status,
                    "skip_reason": candidate.skip_reason,
                })
                continue
            scored, details = fit_candidate_scores(train_subset, eval_subset, candidate)
            metrics = compute_metrics(eval_subset, scored, fit_seconds=details["fit_seconds"])
            scored["candidate"] = candidate.name
            scored["valid_season"] = valid_season
            scored_by_candidate[candidate.name].append(scored)
            metric_rows.append({
                "candidate": candidate.name,
                "valid_season": valid_season,
                "weight": weight,
                "status": "active",
                **metrics,
            })
    return pd.DataFrame(metric_rows), scored_by_candidate


def run_secondary_loocv(historical_df: pd.DataFrame, candidates: list[CandidateSpec]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    shortlist = [candidate for candidate in candidates if candidate.status == "active"]
    for train_seasons, valid_season in loocv_splits():
        train_subset = historical_df[historical_df["Season"].isin(train_seasons)].copy().reset_index(drop=True)
        eval_subset = historical_df[historical_df["Season"] == valid_season].copy().reset_index(drop=True)
        for candidate in shortlist:
            scored, details = fit_candidate_scores(train_subset, eval_subset, candidate)
            metrics = compute_metrics(eval_subset, scored, fit_seconds=details["fit_seconds"])
            rows.append({"candidate": candidate.name, "valid_season": valid_season, **metrics})
    return pd.DataFrame(rows)


def choose_final_candidate(summary: pd.DataFrame, stability_summary: pd.DataFrame | None) -> dict[str, Any]:
    merged = summary.copy()
    if stability_summary is not None and not stability_summary.empty:
        merged = merged.merge(stability_summary, on="candidate", how="left")
    else:
        merged["Top68Jaccard"] = np.nan
        merged["Top16Overlap"] = np.nan
        merged["Top80MeanSeedMovement"] = np.nan

    best_rmse = float(merged["WeightedFullRMSE"].min())
    near = merged[merged["WeightedFullRMSE"] <= best_rmse * 1.01].copy()
    near = near.sort_values(
        [
            "WeightedFullMAE",
            "WeightedInclusionF1",
            "WeightedSelectedSpearman",
            "RMSEStd",
            "Top68Jaccard",
            "Top16Overlap",
            "Top80MeanSeedMovement",
            "WeightedFullRMSE",
        ],
        ascending=[True, False, False, True, False, False, True, True],
    )
    chosen = near.iloc[0].to_dict()
    chosen["selection_reason"] = "Lowest weighted rolling-origin RMSE with defined tie-break order"
    return chosen


def compute_weekly_stability(
    historical_df: pd.DataFrame,
    final_test_df: pd.DataFrame,
    weekly_snapshots: dict[str, pd.DataFrame],
    candidates: list[CandidateSpec],
) -> tuple[pd.DataFrame, dict[str, dict[str, pd.DataFrame]]]:
    detail: dict[str, dict[str, pd.DataFrame]] = {}
    rows: list[dict[str, Any]] = []
    top_names = [candidate.name for candidate in candidates]
    for candidate in candidates:
        if candidate.status != "active":
            continue
        train_subset = historical_df.copy().reset_index(drop=True)
        week_preds: dict[str, pd.DataFrame] = {}
        for stamp, weekly_df in weekly_snapshots.items():
            weekly_eval = weekly_df.copy()
            weekly_eval["TrueSeed"] = 0
            weekly_eval["BidClass"] = "NONE"
            scored, _details = fit_candidate_scores(train_subset, weekly_eval, candidate)
            # Final 2026 must respect 31 AQ / 37 AL.
            scored = assign_seeds_with_bid_quota(
                df=scored[["RecordID", "Season", "Team", "Conference", "NET Rank", "TrueSeed", "BidClass"]],
                final_score=scored["FinalScore"].to_numpy(dtype=float),
                p_aq=scored["P_AQ"].to_numpy(dtype=float),
                p_al=scored["P_AL"].to_numpy(dtype=float),
                aq_target=FINAL_AQ_TARGET,
                al_target=FINAL_AL_TARGET,
                quota_boost=candidate.quota_boost,
            )
            week_preds[stamp] = scored
        detail[candidate.name] = week_preds
        stamps = sorted(week_preds)
        for i in range(1, len(stamps)):
            prev_df = week_preds[stamps[i - 1]]
            curr_df = week_preds[stamps[i]]
            prev_top68 = set(prev_df.loc[prev_df["PredictedSeed"] > 0, "RecordID"])
            curr_top68 = set(curr_df.loc[curr_df["PredictedSeed"] > 0, "RecordID"])
            prev_top16 = set(prev_df.loc[(prev_df["PredictedSeed"] > 0) & (prev_df["PredictedSeed"] <= 16), "RecordID"])
            curr_top16 = set(curr_df.loc[(curr_df["PredictedSeed"] > 0) & (curr_df["PredictedSeed"] <= 16), "RecordID"])
            prev80 = set(prev_df.sort_values("FinalScore", ascending=False).head(80)["RecordID"])
            curr80 = set(curr_df.sort_values("FinalScore", ascending=False).head(80)["RecordID"])
            union80 = sorted(prev80 | curr80)
            prev_map = prev_df.set_index("RecordID")["PredictedSeed"]
            curr_map = curr_df.set_index("RecordID")["PredictedSeed"]
            movement = [abs(int(prev_map.get(rid, 0)) - int(curr_map.get(rid, 0))) for rid in union80]
            rows.append(
                {
                    "candidate": candidate.name,
                    "prev_week": stamps[i - 1],
                    "curr_week": stamps[i],
                    "Top68Jaccard": float(len(prev_top68 & curr_top68) / max(len(prev_top68 | curr_top68), 1)),
                    "Top16Overlap": float(len(prev_top16 & curr_top16) / max(len(prev_top16 | curr_top16), 1)),
                    "Top80MeanSeedMovement": float(np.mean(movement) if movement else 0.0),
                }
            )
    return pd.DataFrame(rows), detail


def extract_feature_importance(bundle: ModelBundle) -> pd.DataFrame:
    if bundle.name == "catboost":
        values = np.asarray(bundle.model.get_feature_importance(), dtype=float)
        return pd.DataFrame({"model": bundle.name, "feature": bundle.feature_columns, "importance": values}).sort_values("importance", ascending=False)
    assert bundle.preprocessor is not None
    names = bundle.preprocessor.get_feature_names_out()
    if hasattr(bundle.model, "coef_"):
        values = np.abs(np.asarray(bundle.model.coef_, dtype=float))
        if values.ndim == 2:
            values = values.mean(axis=0)
    else:
        values = np.asarray(bundle.model.feature_importances_, dtype=float)
    return pd.DataFrame({"model": bundle.name, "feature": names, "importance": values}).sort_values("importance", ascending=False)


def final_train_and_predict(historical_df: pd.DataFrame, final_test_df: pd.DataFrame, candidate: CandidateSpec) -> tuple[pd.DataFrame, dict[str, Any]]:
    eval_df = final_test_df.copy()
    eval_df["TrueSeed"] = 0
    eval_df["BidClass"] = "NONE"
    scored, details = fit_candidate_scores(historical_df, eval_df, candidate)
    scored = assign_seeds_with_bid_quota(
        df=scored[["RecordID", "Season", "Team", "Conference", "NET Rank", "TrueSeed", "BidClass"]],
        final_score=scored["FinalScore"].to_numpy(dtype=float),
        p_aq=scored["P_AQ"].to_numpy(dtype=float),
        p_al=scored["P_AL"].to_numpy(dtype=float),
        aq_target=FINAL_AQ_TARGET,
        al_target=FINAL_AL_TARGET,
        quota_boost=candidate.quota_boost,
    )
    return scored, details


def write_submission(final_test_df: pd.DataFrame, scored: pd.DataFrame, submission_path: Path) -> pd.DataFrame:
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    out = final_test_df[["RecordID"]].merge(scored[["RecordID", "PredictedSeed"]], on="RecordID", how="left")
    out = out.rename(columns={"PredictedSeed": "Overall Seed"})
    out["Overall Seed"] = out["Overall Seed"].fillna(0).astype(int)
    if len(out) != len(final_test_df):
        raise ValueError("Submission row count mismatch")
    seeds = out.loc[out["Overall Seed"] > 0, "Overall Seed"].tolist()
    if len(seeds) != 68 or sorted(seeds) != list(range(1, 69)):
        raise ValueError("Final seed distribution is invalid")
    out.to_csv(submission_path, index=False)
    return out


def main() -> None:
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    warnings.filterwarnings("ignore", message=".*feature names.*")
    args = parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    historical_df = load_labeled_historical_universe(skip_seed_build=args.skip_seed_build)
    final_test_df = pd.read_csv(FINAL_TEST_PATH)
    weekly_snapshots = load_weekly_2026_snapshots()

    candidates = candidate_specs()
    primary_metrics, _ = run_primary_validation(historical_df, candidates)
    primary_metrics.to_csv(args.artifact_dir / "rolling_origin_metrics_v8.csv", index=False)

    summary = summarize_weighted(primary_metrics[primary_metrics["status"] == "active"].copy())
    summary.to_csv(args.artifact_dir / "candidate_summary_v8.csv", index=False)

    shortlist_names = summary[summary["WeightedFullRMSE"] <= summary["WeightedFullRMSE"].min() * 1.01]["candidate"].tolist()
    shortlist_candidates = [candidate for candidate in candidates if candidate.name in shortlist_names]
    loocv = run_secondary_loocv(historical_df, shortlist_candidates)
    loocv.to_csv(args.artifact_dir / "leave_one_season_out_metrics_v8.csv", index=False)

    weekly_summary, _ = compute_weekly_stability(historical_df, final_test_df, weekly_snapshots, shortlist_candidates)
    weekly_summary.to_csv(args.artifact_dir / "weekly_stability_metrics_v8.csv", index=False)
    weekly_agg = weekly_summary.groupby("candidate")[["Top68Jaccard", "Top16Overlap", "Top80MeanSeedMovement"]].mean().reset_index()

    chosen = choose_final_candidate(summary, weekly_agg)
    chosen_name = str(chosen["candidate"])
    chosen_candidate = next(candidate for candidate in candidates if candidate.name == chosen_name)

    final_scored, final_details = final_train_and_predict(historical_df, final_test_df, chosen_candidate)
    submission = write_submission(final_test_df, final_scored, args.submission_path)

    feature_importance = pd.concat(
        [
            extract_feature_importance(final_details["bid_model_bundle"]),
            extract_feature_importance(final_details["seed_model_bundle"]),
        ],
        ignore_index=True,
    )
    feature_importance.to_csv(args.artifact_dir / "feature_importance_v8.csv", index=False)

    final_audit = final_test_df[["RecordID", "Season", "Team", "Conference", "NET Rank"]].merge(
        final_scored[["RecordID", "PredictedSeed", "PredBidType", "FinalScore", "P_AQ", "P_AL", "AQPriority", "ALPriority"]],
        on="RecordID",
        how="left",
    )
    final_audit.to_csv(args.artifact_dir / "final_selection_audit_v8.csv", index=False)

    model_comparison = {
        "candidate_specs": [asdict(candidate) for candidate in candidates],
        "chosen_candidate": chosen,
        "final_aq_target": FINAL_AQ_TARGET,
        "final_al_target": FINAL_AL_TARGET,
        "submission_path": str(args.submission_path),
    }
    (args.artifact_dir / "model_comparison_v8.json").write_text(json.dumps(model_comparison, indent=2))

    print(f"Chosen candidate: {chosen_name}")
    print(f"Wrote rolling metrics: {args.artifact_dir / 'rolling_origin_metrics_v8.csv'}")
    print(f"Wrote candidate summary: {args.artifact_dir / 'candidate_summary_v8.csv'}")
    print(f"Wrote LOO metrics: {args.artifact_dir / 'leave_one_season_out_metrics_v8.csv'}")
    print(f"Wrote weekly stability: {args.artifact_dir / 'weekly_stability_metrics_v8.csv'}")
    print(f"Wrote feature importance: {args.artifact_dir / 'feature_importance_v8.csv'}")
    print(f"Wrote final audit: {args.artifact_dir / 'final_selection_audit_v8.csv'}")
    print(f"Wrote final submission: {args.submission_path}")
    print(f"Final AQ count: {(final_audit['PredBidType'] == 'AQ').sum()}")
    print(f"Final AL count: {(final_audit['PredBidType'] == 'AL').sum()}")
    print(f"Final selected teams: {(submission['Overall Seed'] > 0).sum()}")


if __name__ == "__main__":
    main()
