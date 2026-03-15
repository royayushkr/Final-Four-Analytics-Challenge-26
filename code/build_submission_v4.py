#!/usr/bin/env python3
"""Leakage-safe v4 selected-team seed ranking pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

import build_submission_v3 as v3

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "NCAA_Seed_Training_Set2.0.csv"
DEFAULT_TEST_PATH = PROJECT_ROOT / "data" / "raw" / "NCAA_Seed_Test_Set2.0.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "submissions" / "v4_selected_seed"
RIDGE_ALPHA_GRID = [8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 40.0]

SEASON_CONTEXT_COLS = [
    "NET Rank",
    "PrevNET",
    "AvgOppNETRank",
    "AvgOppNET",
    "NETSOS",
    "NETNonConfSOS",
    "WL_PCT",
    "Conf.Record_PCT",
    "Non-ConferenceRecord_PCT",
    "RoadWL_PCT",
    "Q_NET_SCORE",
]

CONFERENCE_CONTEXT_COLS = [
    "NET Rank",
    "PrevNET",
    "AvgOppNETRank",
    "AvgOppNET",
]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    out = np.divide(
        num.to_numpy(dtype=float),
        den.to_numpy(dtype=float),
        out=np.full(len(num), np.nan, dtype=float),
        where=den.to_numpy(dtype=float) != 0,
    )
    return pd.Series(out, index=num.index, dtype=float)


def add_selected_seed_features(df: pd.DataFrame) -> pd.DataFrame:
    out = v3.feature_engineering(df.copy())
    extra: dict[str, pd.Series | np.ndarray] = {
        "GOOD_WIN_RATE": safe_ratio(out["Quadrant1_W"] + out["Quadrant2_W"], out["WL_G"]),
        "BAD_LOSS_RATE": safe_ratio(out["Quadrant3_L"] + out["Quadrant4_L"], out["WL_G"]),
        "Q1_GAME_SHARE": safe_ratio(out["Quadrant1_G"], out["WL_G"]),
        "Q2_GAME_SHARE": safe_ratio(out["Quadrant2_G"], out["WL_G"]),
        "Q1_WIN_RATE": safe_ratio(out["Quadrant1_W"], out["Quadrant1_G"]),
        "Q2_WIN_RATE": safe_ratio(out["Quadrant2_W"], out["Quadrant2_G"]),
        "Q_GOOD_WIN_SHARE": safe_ratio(
            out["Quadrant1_W"] + out["Quadrant2_W"],
            out["Quadrant1_G"] + out["Quadrant2_G"],
        ),
        "Q_BAD_LOSS_SHARE": safe_ratio(
            out["Quadrant3_L"] + out["Quadrant4_L"],
            out["Quadrant3_G"] + out["Quadrant4_G"],
        ),
        "Q_NET_SCORE_PER_GAME": safe_ratio(out["Q_NET_SCORE"], out["WL_G"]),
        "ROAD_SOS_INTERACTION": out["RoadWL_PCT"] * out["NETSOS_INV"],
        "NONCONF_SOS_INTERACTION": out["Non-ConferenceRecord_PCT"] * out["NETNonConfSOS_INV"],
        "TOP_RESUME_INDEX": (
            2.5 * out["Quadrant1_W"]
            + 1.5 * out["Quadrant2_W"]
            - 1.5 * out["Quadrant3_L"]
            - 2.5 * out["Quadrant4_L"]
        ),
    }
    extra["SEED_STRENGTH_HEUR"] = (
        -0.45 * out["NET Rank"].fillna(out["NET Rank"].median())
        - 0.20 * out["PrevNET"].fillna(out["PrevNET"].median())
        - 0.10 * out["AvgOppNET"].fillna(out["AvgOppNET"].median())
        + 35.0 * out["WL_PCT"].fillna(out["WL_PCT"].median())
        + 25.0 * pd.Series(extra["GOOD_WIN_RATE"], index=out.index).fillna(
            pd.Series(extra["GOOD_WIN_RATE"], index=out.index).median()
        )
        - 25.0 * pd.Series(extra["BAD_LOSS_RATE"], index=out.index).fillna(
            pd.Series(extra["BAD_LOSS_RATE"], index=out.index).median()
        )
    )

    context: dict[str, pd.Series | np.ndarray] = {}
    for col in SEASON_CONTEXT_COLS:
        if col not in out.columns:
            continue
        grp = out.groupby("Season")[col]
        context[f"{col}_SEASON_PCT"] = grp.rank(method="average", pct=True)
        mean = grp.transform("mean")
        std = grp.transform(lambda s: s.std(ddof=0))
        context[f"{col}_SEASON_Z"] = np.where(std > 0, (out[col] - mean) / std, 0.0)

    for col in CONFERENCE_CONTEXT_COLS:
        if col not in out.columns:
            continue
        context[f"{col}_CONF_PCT"] = out.groupby(["Season", "Conference"])[col].rank(
            method="average", pct=True
        )

    return pd.concat([out, pd.DataFrame(extra, index=out.index), pd.DataFrame(context, index=out.index)], axis=1)


def to_seed_features(
    df: pd.DataFrame,
    include_team_feature: bool,
    include_bid_type_feature: bool,
) -> pd.DataFrame:
    out = add_selected_seed_features(df)
    drop_cols = ["RecordID", "Overall Seed", *v3.RECORD_COLS]
    if not include_team_feature:
        drop_cols.append("Team")
    if not include_bid_type_feature:
        drop_cols.append("Bid Type")
    return out.drop(columns=drop_cols, errors="ignore")


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


def model_specs(ridge_alpha: float) -> dict[str, object]:
    return {
        "ridge": Ridge(alpha=ridge_alpha),
        "xgb": XGBRegressor(
            objective="reg:squarederror",
            n_estimators=900,
            max_depth=4,
            learning_rate=0.02,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.5,
            random_state=42,
            n_jobs=4,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=800,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }


def season_rank_rmse(
    seed_df: pd.DataFrame,
    y_true: np.ndarray,
    score: np.ndarray,
) -> float:
    pred = np.zeros(len(seed_df), dtype=float)
    for season, idx in seed_df.groupby("Season").groups.items():
        season_idx = np.array(list(idx))
        actual = y_true[season_idx]
        season_pred = np.empty(len(season_idx), dtype=float)
        season_pred[np.argsort(-score[season_idx])] = np.sort(actual)
        pred[season_idx] = season_pred
    return rmse(y_true, pred)


def per_season_rank_score(seasons: pd.Series, values: np.ndarray) -> np.ndarray:
    ranked = np.zeros(len(values), dtype=float)
    for season, idx in seasons.groupby(seasons).groups.items():
        season_idx = np.array(list(idx))
        ranked[season_idx] = (
            pd.Series(values[season_idx]).rank(method="average", pct=True).to_numpy(dtype=float)
        )
    return ranked


def choose_best_ridge_alpha(
    seed_df: pd.DataFrame,
    include_team_feature: bool,
    include_bid_type_feature: bool,
) -> tuple[float, dict[float, float]]:
    x_all = to_seed_features(
        seed_df,
        include_team_feature=include_team_feature,
        include_bid_type_feature=include_bid_type_feature,
    )
    y = seed_df["Overall Seed"].to_numpy(dtype=float)
    groups = seed_df["Season"]
    fold_index = list(GroupKFold(n_splits=5).split(x_all, y, groups))
    scores: dict[float, float] = {}

    for alpha in RIDGE_ALPHA_GRID:
        oof = np.zeros(len(seed_df), dtype=float)
        for train_idx, valid_idx in fold_index:
            x_train = x_all.iloc[train_idx]
            x_valid = x_all.iloc[valid_idx]
            y_train = y[train_idx]

            pre = build_preprocessor(x_train)
            x_train_t = pre.fit_transform(x_train)
            x_valid_t = pre.transform(x_valid)

            model = Ridge(alpha=alpha)
            model.fit(x_train_t, y_train)
            oof[valid_idx] = -model.predict(x_valid_t)

        scores[alpha] = season_rank_rmse(seed_df, y, oof)

    best_alpha = min(scores, key=scores.get)
    return best_alpha, scores


def oof_model_scores(
    seed_df: pd.DataFrame,
    ridge_alpha: float,
    include_team_feature: bool,
    include_bid_type_feature: bool,
) -> dict[str, np.ndarray]:
    x_all = to_seed_features(
        seed_df,
        include_team_feature=include_team_feature,
        include_bid_type_feature=include_bid_type_feature,
    )
    y = seed_df["Overall Seed"].to_numpy(dtype=float)
    groups = seed_df["Season"]
    scores = {name: np.zeros(len(seed_df), dtype=float) for name in model_specs(ridge_alpha)}

    for train_idx, valid_idx in GroupKFold(n_splits=5).split(x_all, y, groups):
        x_train = x_all.iloc[train_idx]
        x_valid = x_all.iloc[valid_idx]
        y_train = y[train_idx]

        pre = build_preprocessor(x_train)
        x_train_t = pre.fit_transform(x_train)
        x_valid_t = pre.transform(x_valid)

        for name, model in model_specs(ridge_alpha).items():
            model.fit(x_train_t, y_train)
            scores[name][valid_idx] = -model.predict(x_valid_t)

    return scores


def choose_blend_weights(
    seed_df: pd.DataFrame,
    y_true: np.ndarray,
    raw_scores: dict[str, np.ndarray],
) -> tuple[dict[str, float], float, dict[str, float]]:
    ranked_scores = {
        name: per_season_rank_score(seed_df["Season"], values)
        for name, values in raw_scores.items()
    }
    model_metrics = {
        name: season_rank_rmse(seed_df, y_true, values)
        for name, values in ranked_scores.items()
    }

    best_score = float("inf")
    best_weights = {"ridge": 1.0, "xgb": 0.0, "extra_trees": 0.0}
    grid = [i / 20 for i in range(21)]
    for ridge_w in grid:
        for xgb_w in grid:
            et_w = 1.0 - ridge_w - xgb_w
            if et_w < 0:
                continue
            blend = (
                ridge_w * ranked_scores["ridge"]
                + xgb_w * ranked_scores["xgb"]
                + et_w * ranked_scores["extra_trees"]
            )
            score = season_rank_rmse(seed_df, y_true, blend)
            if score < best_score:
                best_score = score
                best_weights = {
                    "ridge": ridge_w,
                    "xgb": xgb_w,
                    "extra_trees": et_w,
                }

    return best_weights, best_score, model_metrics


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
        if actual != expected:
            raise ValueError(
                f"Bid Type selection count mismatch for {season}: expected {expected}, got {actual}"
            )

    return selected_flag


def fit_predict_scores(
    train_seeded_df: pd.DataFrame,
    test_selected_df: pd.DataFrame,
    ridge_alpha: float,
    include_team_feature: bool,
    include_bid_type_feature: bool,
) -> dict[str, np.ndarray]:
    x_train = to_seed_features(
        train_seeded_df,
        include_team_feature=include_team_feature,
        include_bid_type_feature=include_bid_type_feature,
    )
    x_test = to_seed_features(
        test_selected_df,
        include_team_feature=include_team_feature,
        include_bid_type_feature=include_bid_type_feature,
    )
    y_train = train_seeded_df["Overall Seed"].to_numpy(dtype=float)

    pre = build_preprocessor(x_train)
    x_train_t = pre.fit_transform(x_train)
    x_test_t = pre.transform(x_test)

    out: dict[str, np.ndarray] = {}
    for name, model in model_specs(ridge_alpha).items():
        model.fit(x_train_t, y_train)
        out[name] = -model.predict(x_test_t)
    return out


def blend_rank_scores(
    seasons: pd.Series,
    raw_scores: dict[str, np.ndarray],
    weights: dict[str, float],
) -> np.ndarray:
    ranked_scores = {
        name: per_season_rank_score(seasons, values)
        for name, values in raw_scores.items()
    }
    blend = np.zeros(len(seasons), dtype=float)
    for name, weight in weights.items():
        blend += weight * ranked_scores[name]
    return blend


def assign_available_seeds(
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

    for season, idx in test_selected_df.groupby("Season").groups.items():
        season_idx = np.array(list(idx))
        used = seeded_train_by_season.get(season, set())
        available = sorted([s for s in range(1, tournament_size + 1) if s not in used])
        if len(available) != len(season_idx):
            raise ValueError(
                f"Available seed count mismatch for {season}: expected {len(available)}, got {len(season_idx)}"
            )
        assigned = np.empty(len(season_idx), dtype=float)
        assigned[np.argsort(-score[season_idx])] = np.array(available, dtype=float)
        final_seed[season_idx] = assigned

    return final_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--test-path", default=str(DEFAULT_TEST_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--tournament-size", type=int, default=68)
    parser.add_argument(
        "--exclude-team-feature",
        action="store_true",
        help="Drop Team identity from the seed model feature set.",
    )
    parser.add_argument(
        "--disable-bid-type-feature",
        action="store_true",
        help="Drop Bid Type from the seed model feature set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)

    selected_flag = validate_bid_type_selection(
        train_df=train_df,
        test_df=test_df,
        tournament_size=args.tournament_size,
    )
    train_seeded_df = train_df[train_df["Overall Seed"].notna()].copy().reset_index(drop=True)
    test_selected_df = test_df.loc[selected_flag].copy().reset_index(drop=True)

    include_team_feature = not args.exclude_team_feature
    include_bid_type_feature = not args.disable_bid_type_feature

    ridge_alpha, alpha_scores = choose_best_ridge_alpha(
        seed_df=train_seeded_df,
        include_team_feature=include_team_feature,
        include_bid_type_feature=include_bid_type_feature,
    )

    raw_oof_scores = oof_model_scores(
        seed_df=train_seeded_df,
        ridge_alpha=ridge_alpha,
        include_team_feature=include_team_feature,
        include_bid_type_feature=include_bid_type_feature,
    )
    y_seed = train_seeded_df["Overall Seed"].to_numpy(dtype=float)
    blend_weights, best_oof_rmse, model_oof_rmse = choose_blend_weights(
        seed_df=train_seeded_df,
        y_true=y_seed,
        raw_scores=raw_oof_scores,
    )

    raw_test_scores = fit_predict_scores(
        train_seeded_df=train_seeded_df,
        test_selected_df=test_selected_df,
        ridge_alpha=ridge_alpha,
        include_team_feature=include_team_feature,
        include_bid_type_feature=include_bid_type_feature,
    )
    blended_test_score = blend_rank_scores(
        seasons=test_selected_df["Season"],
        raw_scores=raw_test_scores,
        weights=blend_weights,
    )
    selected_final_seed = assign_available_seeds(
        train_df=train_df,
        test_selected_df=test_selected_df,
        score=blended_test_score,
        tournament_size=args.tournament_size,
    )

    final_seed = np.zeros(len(test_df), dtype=float)
    final_seed[selected_flag.to_numpy()] = selected_final_seed

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    submission_path = output_dir / "submission_v4_selected_seed.csv"
    submission_alias_path = output_dir / "submission_v4.csv"
    diagnostics_path = output_dir / "diagnostics_v4_selected_seed.csv"
    metrics_path = output_dir / "oof_metrics_v4_selected_seed.json"

    submission_df = pd.DataFrame({"RecordID": test_df["RecordID"], "Overall Seed": final_seed})
    submission_df.to_csv(submission_path, index=False)
    submission_df.to_csv(submission_alias_path, index=False)

    pd.DataFrame(
        {
            "RecordID": test_selected_df["RecordID"],
            "Season": test_selected_df["Season"],
            "Team": test_selected_df["Team"],
            "Bid Type": test_selected_df["Bid Type"],
            "RidgeScore": raw_test_scores["ridge"],
            "XgbScore": raw_test_scores["xgb"],
            "ExtraTreesScore": raw_test_scores["extra_trees"],
            "BlendScore": blended_test_score,
            "AssignedSeed": selected_final_seed,
        }
    ).to_csv(diagnostics_path, index=False)

    metrics = {
        "oof_rank_rmse_ridge_alpha_grid": alpha_scores,
        "best_ridge_alpha": ridge_alpha,
        "oof_rank_rmse_by_model": model_oof_rmse,
        "oof_rank_rmse_blend": best_oof_rmse,
        "blend_weights": blend_weights,
        "selected_count_test": int(selected_flag.sum()),
        "non_zero_count_test": int((final_seed > 0).sum()),
        "include_team_feature": include_team_feature,
        "include_bid_type_feature": include_bid_type_feature,
        "selection_source": "Bid Type non-null rows",
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"Wrote: {submission_path}")
    print(f"Wrote: {submission_alias_path}")
    print(f"Wrote: {diagnostics_path}")
    print(f"Wrote: {metrics_path}")
    print("Summary metrics:")
    print(f"- best_ridge_alpha: {ridge_alpha}")
    print(f"- oof_rank_rmse_blend: {best_oof_rmse}")
    print(f"- blend_weights: {blend_weights}")
    print(f"- selected_count_test: {int(selected_flag.sum())}")


if __name__ == "__main__":
    main()
