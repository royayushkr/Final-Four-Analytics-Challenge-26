#!/usr/bin/env python3
"""No-external-data v3 pipeline with stronger feature engineering and postprocessing."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "NCAA_Seed_Training_Set2.0.csv"
DEFAULT_TEST_PATH = PROJECT_ROOT / "data" / "raw" / "NCAA_Seed_Test_Set2.0.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "submissions" / "v3_no_external"

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


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def parse_wins_losses(value: object) -> tuple[float, float]:
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
        out[f"{col}_MARGIN"] = out[f"{col}_W"] - out[f"{col}_L"]

    # Core rank transforms
    for rank_col in ["NET Rank", "PrevNET", "AvgOppNETRank", "AvgOppNET", "NETSOS", "NETNonConfSOS"]:
        if rank_col in out.columns:
            out[f"{rank_col}_INV"] = 1.0 / (1.0 + out[rank_col])
            out[f"{rank_col}_LOG"] = np.log1p(out[rank_col])

    out["NET_Improvement"] = out["PrevNET"] - out["NET Rank"]
    out["OppNetDiff"] = out["AvgOppNETRank"] - out["AvgOppNET"]
    out["SOS_Diff"] = out["NETNonConfSOS"] - out["NETSOS"]

    # Aggregate quality signals
    out["Q_WEIGHTED_WINS"] = (
        4 * out["Quadrant1_W"] + 3 * out["Quadrant2_W"] + 2 * out["Quadrant3_W"] + out["Quadrant4_W"]
    )
    out["Q_WEIGHTED_LOSSES"] = (
        out["Quadrant1_L"] + 2 * out["Quadrant2_L"] + 3 * out["Quadrant3_L"] + 4 * out["Quadrant4_L"]
    )
    out["Q_NET_SCORE"] = out["Q_WEIGHTED_WINS"] - out["Q_WEIGHTED_LOSSES"]

    out["WL_TOTAL_GAMES"] = out["WL_G"]
    out["CONF_GAME_SHARE"] = np.where(
        out["WL_G"] > 0, out["Conf.Record_G"] / out["WL_G"], np.nan
    )
    out["NONCONF_GAME_SHARE"] = np.where(
        out["WL_G"] > 0, out["Non-ConferenceRecord_G"] / out["WL_G"], np.nan
    )
    out["ROAD_GAME_SHARE"] = np.where(out["WL_G"] > 0, out["RoadWL_G"] / out["WL_G"], np.nan)
    out["ROAD_PERF_DELTA"] = out["RoadWL_PCT"] - out["WL_PCT"]
    out["CONF_PERF_DELTA"] = out["Conf.Record_PCT"] - out["WL_PCT"]
    out["NONCONF_PERF_DELTA"] = out["Non-ConferenceRecord_PCT"] - out["WL_PCT"]
    return out


def to_features(
    df: pd.DataFrame, include_bid_type: bool, include_team: bool
) -> pd.DataFrame:
    engineered = feature_engineering(df)
    drop_cols = ["RecordID", "Overall Seed", *RECORD_COLS]
    if not include_bid_type:
        drop_cols.append("Bid Type")
    if not include_team:
        drop_cols.append("Team")
    return engineered.drop(columns=drop_cols, errors="ignore")


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


def build_selection_model(pre: ColumnTransformer) -> Pipeline:
    return Pipeline(
        steps=[
            ("pre", pre),
            (
                "model",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_estimators=700,
                    max_depth=5,
                    learning_rate=0.03,
                    subsample=0.9,
                    colsample_bytree=0.85,
                    random_state=42,
                ),
            ),
        ]
    )


def build_seed_models(pre: ColumnTransformer) -> tuple[Pipeline, Pipeline]:
    m1 = Pipeline(
        steps=[
            ("pre", pre),
            (
                "model",
                XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=900,
                    max_depth=4,
                    learning_rate=0.02,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.5,
                    random_state=42,
                ),
            ),
        ]
    )
    m2 = Pipeline(
        steps=[
            ("pre", pre),
            (
                "model",
                XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=450,
                    max_depth=2,
                    learning_rate=0.04,
                    subsample=1.0,
                    colsample_bytree=0.8,
                    reg_lambda=2.0,
                    random_state=42,
                ),
            ),
        ]
    )
    return m1, m2


def season_selection_targets(train_df: pd.DataFrame, tournament_size: int) -> dict[str, int]:
    seeded_per_season = (
        train_df.assign(is_seeded=train_df["Overall Seed"].notna().astype(int))
        .groupby("Season")["is_seeded"]
        .sum()
        .to_dict()
    )
    return {season: max(0, tournament_size - int(count)) for season, count in seeded_per_season.items()}


def choose_selected_flags(
    test_df: pd.DataFrame,
    selected_prob: np.ndarray,
    train_df: pd.DataFrame,
    tournament_size: int,
    use_bid_type_hint: bool,
) -> np.ndarray:
    out = np.zeros(len(test_df), dtype=bool)
    targets = season_selection_targets(train_df, tournament_size)
    bid_type_known = test_df["Bid Type"].notna().to_numpy()

    for season, idx in test_df.groupby("Season").groups.items():
        season_idx = np.array(list(idx))
        target_k = min(targets.get(season, 0), len(season_idx))

        if use_bid_type_hint:
            base = season_idx[bid_type_known[season_idx]]
        else:
            base = np.array([], dtype=int)

        base = np.array(base, dtype=int)
        if len(base) >= target_k:
            # If hints exceed target, keep highest-probability subset.
            keep_order = np.argsort(-selected_prob[base])[:target_k]
            out[base[keep_order]] = True
            continue

        out[base] = True
        remaining_slots = target_k - len(base)
        remaining_pool = np.array([i for i in season_idx if i not in set(base)], dtype=int)
        if remaining_slots > 0 and len(remaining_pool) > 0:
            pick = np.argsort(-selected_prob[remaining_pool])[:remaining_slots]
            out[remaining_pool[pick]] = True

    return out


def fit_local_isotonic_models(train_seeded_df: pd.DataFrame) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for season, grp in train_seeded_df.groupby("Season"):
        x = grp["NET Rank"].to_numpy(dtype=float)
        y = grp["Overall Seed"].to_numpy(dtype=float)
        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() < 6:
            continue
        model = IsotonicRegression(increasing=True, out_of_bounds="clip")
        model.fit(x[valid], y[valid])
        models[season] = model
    return models


def assign_seeds_by_order(order_score: np.ndarray, available_seeds: list[int]) -> np.ndarray:
    n = len(order_score)
    assigned = np.zeros(n, dtype=float)

    if n == 0:
        return assigned

    sorted_team_idx = np.argsort(order_score)
    seeds = np.array(sorted(available_seeds), dtype=float)

    if len(seeds) == n:
        assigned[sorted_team_idx] = seeds
        return assigned

    if len(seeds) > n:
        # Choose subset of available seeds closest to evenly spaced quantiles.
        quantile_idx = np.linspace(0, len(seeds) - 1, n).round().astype(int)
        seeds = seeds[quantile_idx]
        assigned[sorted_team_idx] = np.sort(seeds)
        return assigned

    # If there are fewer seeds than selected rows (unexpected), use all and fill rest.
    assigned.fill(np.nan)
    assigned[sorted_team_idx[: len(seeds)]] = seeds
    fallback = np.linspace(1, 68, n)
    assigned = np.where(np.isnan(assigned), fallback, assigned)
    return assigned


def oof_metrics(
    train_df: pd.DataFrame,
    include_bid_type_feature: bool,
    include_team_feature: bool,
) -> dict[str, float]:
    x_all = to_features(
        train_df,
        include_bid_type=include_bid_type_feature,
        include_team=include_team_feature,
    )
    y_seed = train_df["Overall Seed"]
    y_selected = y_seed.notna().astype(int).to_numpy()
    y_zero = y_seed.fillna(0.0).to_numpy()

    pre = build_preprocessor(x_all)
    gkf = GroupKFold(n_splits=5)

    prob = np.zeros(len(train_df))
    seed_pred = np.zeros(len(train_df))

    for tr_idx, va_idx in gkf.split(x_all, y_selected, groups=train_df["Season"]):
        x_tr, x_va = x_all.iloc[tr_idx], x_all.iloc[va_idx]
        y_sel_tr = y_selected[tr_idx]
        y_seed_tr = y_seed.iloc[tr_idx]
        seed_mask = y_seed_tr.notna().to_numpy()

        clf = build_selection_model(pre)
        clf.fit(x_tr, y_sel_tr)
        prob[va_idx] = clf.predict_proba(x_va)[:, 1]

        reg1, reg2 = build_seed_models(pre)
        reg1.fit(x_tr.iloc[seed_mask], y_seed_tr.iloc[seed_mask])
        reg2.fit(x_tr.iloc[seed_mask], y_seed_tr.iloc[seed_mask])
        seed_pred[va_idx] = 0.55 * reg1.predict(x_va) + 0.45 * reg2.predict(x_va)

    # Fixed threshold for simple OOF metric
    final = np.where(prob >= 0.5, np.clip(seed_pred, 1, 68), 0.0)

    return {
        "oof_selection_accuracy_0_5": float(((prob >= 0.5).astype(int) == y_selected).mean()),
        "oof_seed_rmse_selected": rmse(
            y_seed[y_seed.notna()].to_numpy(), np.clip(seed_pred[y_seed.notna().to_numpy()], 1, 68)
        ),
        "oof_rmse_with_zero_threshold_0_5": rmse(y_zero, final),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--test-path", default=str(DEFAULT_TEST_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--tournament-size", type=int, default=68)
    parser.add_argument(
        "--disable-bid-type-hint",
        action="store_true",
        help="Disable non-null Bid Type hint for selection.",
    )
    parser.add_argument(
        "--disable-bid-type-feature",
        action="store_true",
        help="Disable Bid Type as model feature.",
    )
    parser.add_argument(
        "--exclude-team-feature",
        action="store_true",
        help="Exclude Team identity feature.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)

    include_team_feature = not args.exclude_team_feature

    use_bid_type_hint = not args.disable_bid_type_hint
    include_bid_type_feature = not args.disable_bid_type_feature

    # Selection model
    x_train_sel = to_features(
        train_df,
        include_bid_type=include_bid_type_feature,
        include_team=include_team_feature,
    )
    x_test_sel = to_features(
        test_df,
        include_bid_type=include_bid_type_feature,
        include_team=include_team_feature,
    )
    y_selected = train_df["Overall Seed"].notna().astype(int).to_numpy()

    sel_pre = build_preprocessor(x_train_sel)
    sel_model = build_selection_model(sel_pre)
    sel_model.fit(x_train_sel, y_selected)
    selected_prob = sel_model.predict_proba(x_test_sel)[:, 1]

    selected_flags = choose_selected_flags(
        test_df=test_df,
        selected_prob=selected_prob,
        train_df=train_df,
        tournament_size=args.tournament_size,
        use_bid_type_hint=use_bid_type_hint,
    )

    # Seed scoring model (trained on seeded training rows only)
    train_seeded = train_df[train_df["Overall Seed"].notna()].copy()
    x_train_seed = to_features(
        train_seeded,
        include_bid_type=True,  # AQ/AL is available for seeded train rows
        include_team=include_team_feature,
    )
    x_test_seed = to_features(test_df, include_bid_type=True, include_team=include_team_feature)
    y_seed = train_seeded["Overall Seed"].to_numpy()

    seed_pre = build_preprocessor(x_train_seed)
    seed_m1, seed_m2 = build_seed_models(seed_pre)
    seed_m1.fit(x_train_seed, y_seed)
    seed_m2.fit(x_train_seed, y_seed)
    seed_pred_raw = np.clip(0.55 * seed_m1.predict(x_test_seed) + 0.45 * seed_m2.predict(x_test_seed), 1, 68)

    # Season-local isotonic correction on NET Rank
    local_models = fit_local_isotonic_models(train_seeded)
    local_iso_pred = np.copy(seed_pred_raw)
    for season, idx in test_df.groupby("Season").groups.items():
        model = local_models.get(season)
        if model is None:
            continue
        season_idx = np.array(list(idx))
        net_rank = test_df.loc[season_idx, "NET Rank"].to_numpy(dtype=float)
        fill = float(np.nanmedian(train_seeded.loc[train_seeded["Season"] == season, "NET Rank"]))
        net_rank = np.where(np.isnan(net_rank), fill, net_rank)
        local_iso_pred[season_idx] = model.predict(net_rank)

    # Combined ordering score for constrained assignment.
    net_rank_for_order = test_df["NET Rank"].to_numpy(dtype=float)
    net_rank_fill = float(np.nanmedian(train_df["NET Rank"]))
    net_rank_for_order = np.where(np.isnan(net_rank_for_order), net_rank_fill, net_rank_for_order)
    order_score = 0.55 * seed_pred_raw + 0.30 * local_iso_pred + 0.15 * net_rank_for_order

    # Constrained assignment by season.
    final_seed = np.zeros(len(test_df), dtype=float)
    seeded_train_by_season = (
        train_df[train_df["Overall Seed"].notna()].groupby("Season")["Overall Seed"].apply(lambda s: set(s.astype(int)))
    )

    for season, idx in test_df.groupby("Season").groups.items():
        season_idx = np.array(list(idx))
        season_selected = season_idx[selected_flags[season_idx]]
        if len(season_selected) == 0:
            continue

        known = seeded_train_by_season.get(season, set())
        available = sorted([s for s in range(1, args.tournament_size + 1) if s not in known])
        assigned = assign_seeds_by_order(order_score[season_selected], available)
        final_seed[season_selected] = np.clip(assigned, 1, 68)

    # Final non-selected teams stay at zero.
    final_seed[~selected_flags] = 0.0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    submission_path = output_dir / "submission_v3_no_external.csv"
    diagnostics_path = output_dir / "diagnostics_v3_no_external.csv"
    metrics_path = output_dir / "oof_metrics_v3_no_external.json"

    pd.DataFrame({"RecordID": test_df["RecordID"], "Overall Seed": final_seed}).to_csv(
        submission_path, index=False
    )
    pd.DataFrame(
        {
            "RecordID": test_df["RecordID"],
            "Season": test_df["Season"],
            "Bid Type": test_df["Bid Type"],
            "SelectedProb": selected_prob,
            "SelectedFlag": selected_flags.astype(int),
            "SeedPredRaw": seed_pred_raw,
            "SeedPredLocalIso": local_iso_pred,
            "SeedAssignedFinal": final_seed,
        }
    ).to_csv(diagnostics_path, index=False)

    metrics = oof_metrics(
        train_df=train_df,
        include_bid_type_feature=include_bid_type_feature,
        include_team_feature=include_team_feature,
    )
    metrics.update(
        {
            "use_bid_type_hint": bool(use_bid_type_hint),
            "include_bid_type_feature": bool(include_bid_type_feature),
            "include_team_feature": bool(include_team_feature),
            "selected_count_test": int(selected_flags.sum()),
            "non_zero_count_test": int((final_seed > 0).sum()),
        }
    )
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"Wrote: {submission_path}")
    print(f"Wrote: {diagnostics_path}")
    print(f"Wrote: {metrics_path}")
    print("Summary metrics:")
    for key, val in metrics.items():
        print(f"- {key}: {val}")


if __name__ == "__main__":
    main()
