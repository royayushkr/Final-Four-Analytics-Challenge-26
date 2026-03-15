#!/usr/bin/env python3
"""Private-task feature engineering for the 2026 NCAA seed pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

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

RANK_STYLE_COLS = [
    "NET Rank",
    "PrevNET",
    "AvgOppNETRank",
    "AvgOppNET",
    "NETSOS",
    "NETNonConfSOS",
]

CONFERENCE_NORMALIZATION = {
    "The American": "American",
}

DROP_COLS = [
    "RecordID",
    "Overall Seed",
    "Bid Type",
    "Team",
    "Season",
    *RECORD_COLS,
]

SEASON_RELATIVE_COLS = [
    "NET Rank_INV",
    "PrevNET_INV",
    "AvgOppNETRank_INV",
    "AvgOppNET_INV",
    "NETSOS_INV",
    "NETNonConfSOS_INV",
    "WL_PCT",
    "Conf.Record_PCT",
    "Non-ConferenceRecord_PCT",
    "RoadWL_PCT",
    "q12_wins",
    "q12_win_rate",
    "q34_losses_neg",
    "quality_win_score",
    "bad_loss_penalty_neg",
    "resume_balance",
    "resume_efficiency",
    "road_quality",
    "nonconf_quality",
    "quality_minus_bad",
    "top_resume_index",
]

CONFERENCE_CONTEXT_COLS = [
    "NET Rank_INV",
    "WL_PCT",
    "RoadWL_PCT",
    "quality_win_score",
    "resume_balance",
    "top_resume_index",
]


@dataclass
class FeatureMatrix:
    train_frame: pd.DataFrame
    eval_frame: pd.DataFrame
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    clip_bounds: dict[str, tuple[float, float]]


def parse_wins_losses_strict(value: object) -> tuple[float, float]:
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


def normalize_conference_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Conference" in out.columns:
        out["Conference"] = out["Conference"].replace(CONFERENCE_NORMALIZATION)
    return out


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    out = np.full(len(num), np.nan, dtype=float)
    valid = den.to_numpy(dtype=float) > 0
    out[valid] = num.to_numpy(dtype=float)[valid] / den.to_numpy(dtype=float)[valid]
    return pd.Series(out, index=num.index, dtype=float)


def _safe_inverse(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return 1.0 / (1.0 + values)


def _add_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in RANK_STYLE_COLS:
        if col in out.columns:
            out[f"{col}_MISSING"] = out[col].isna().astype(float)
    return out


def build_resume_features(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_conference_names(df)
    out = out.copy()
    out["SeasonStart"] = out["Season"].astype(str).str.split("-").str[0].astype(float)

    for col in RECORD_COLS:
        parsed = out[col].apply(parse_wins_losses_strict)
        out[f"{col}_W"] = parsed.str[0].astype(float)
        out[f"{col}_L"] = parsed.str[1].astype(float)
        out[f"{col}_G"] = out[f"{col}_W"] + out[f"{col}_L"]
        out[f"{col}_PCT"] = _safe_ratio(out[f"{col}_W"], out[f"{col}_G"])
        out[f"{col}_LOSS_PCT"] = _safe_ratio(out[f"{col}_L"], out[f"{col}_G"])
        out[f"{col}_MARGIN"] = out[f"{col}_W"] - out[f"{col}_L"]

    for col in RANK_STYLE_COLS:
        if col in out.columns:
            out[f"{col}_INV"] = _safe_inverse(out[col])
            out[f"{col}_LOG"] = np.log1p(pd.to_numeric(out[col], errors="coerce"))

    out["net_delta"] = out["PrevNET"] - out["NET Rank"]
    out["net_move_abs"] = out["net_delta"].abs()
    out["net_improved_flag"] = (out["net_delta"] > 0).astype(float)
    out["net_worsened_flag"] = (out["net_delta"] < 0).astype(float)

    out["q12_wins"] = out["Quadrant1_W"] + out["Quadrant2_W"]
    out["q12_losses"] = out["Quadrant1_L"] + out["Quadrant2_L"]
    out["q12_games"] = out["Quadrant1_G"] + out["Quadrant2_G"]
    out["q12_win_rate"] = _safe_ratio(out["q12_wins"], out["q12_games"])

    out["q34_wins"] = out["Quadrant3_W"] + out["Quadrant4_W"]
    out["q34_losses"] = out["Quadrant3_L"] + out["Quadrant4_L"]
    out["q34_games"] = out["Quadrant3_G"] + out["Quadrant4_G"]
    out["q34_loss_rate"] = _safe_ratio(out["q34_losses"], out["q34_games"])

    out["overall_wins"] = out["WL_W"]
    out["overall_losses"] = out["WL_L"]
    out["overall_games"] = out["WL_G"]
    out["overall_win_rate"] = out["WL_PCT"]
    out["conference_win_rate"] = out["Conf.Record_PCT"]
    out["nonconference_win_rate"] = out["Non-ConferenceRecord_PCT"]
    out["road_wins"] = out["RoadWL_W"]
    out["road_losses"] = out["RoadWL_L"]
    out["road_win_rate"] = out["RoadWL_PCT"]

    out["q1_quality_score"] = 4.5 * out["Quadrant1_W"] - 0.5 * out["Quadrant1_L"]
    out["q2_quality_score"] = 3.0 * out["Quadrant2_W"] - 1.0 * out["Quadrant2_L"]
    out["q3_risk_score"] = 1.0 * out["Quadrant3_W"] - 2.5 * out["Quadrant3_L"]
    out["q4_risk_score"] = 0.5 * out["Quadrant4_W"] - 4.0 * out["Quadrant4_L"]
    out["quality_win_score"] = 4.0 * out["Quadrant1_W"] + 2.5 * out["Quadrant2_W"] + 1.0 * out["Quadrant3_W"]
    out["bad_loss_penalty"] = 2.5 * out["Quadrant3_L"] + 4.0 * out["Quadrant4_L"]
    out["resume_balance"] = out["quality_win_score"] - out["bad_loss_penalty"]
    out["resume_efficiency"] = _safe_ratio(out["resume_balance"], out["overall_games"])
    out["quality_minus_bad"] = out["q12_wins"] - out["q34_losses"]
    out["q12_share"] = _safe_ratio(out["q12_games"], out["overall_games"])
    out["q34_share"] = _safe_ratio(out["q34_games"], out["overall_games"])
    out["q3_bad_loss_flag"] = (out["Quadrant3_L"] > 0).astype(float)
    out["q4_bad_loss_flag"] = (out["Quadrant4_L"] > 0).astype(float)
    out["multiple_bad_loss_flag"] = (out["q34_losses"] >= 2).astype(float)

    out["sos_gap"] = out["NETNonConfSOS"] - out["NETSOS"]
    out["sos_ratio"] = _safe_ratio(out["NETNonConfSOS"], out["NETSOS"])
    out["road_quality"] = out["road_win_rate"] * out["NETSOS_INV"]
    out["nonconf_quality"] = out["nonconference_win_rate"] * out["NETNonConfSOS_INV"]
    out["conference_nonconference_gap"] = out["nonconference_win_rate"] - out["conference_win_rate"]
    out["road_vs_overall_gap"] = out["road_win_rate"] - out["overall_win_rate"]
    out["conf_vs_overall_gap"] = out["conference_win_rate"] - out["overall_win_rate"]

    out["top_resume_index"] = (
        0.35 * out["NET Rank_INV"]
        + 0.20 * out["AvgOppNETRank_INV"]
        + 0.15 * out["NETSOS_INV"]
        + 0.20 * out["q12_win_rate"].fillna(0.0)
        + 0.10 * out["road_win_rate"].fillna(0.0)
    )

    for threshold in [10, 16, 25, 45, 50, 75]:
        out[f"net_top_{threshold}_flag"] = (out["NET Rank"] <= threshold).astype(float)
        out[f"net_top_{threshold}_distance"] = threshold - out["NET Rank"]

    out["NET_x_q12_wins"] = out["NET Rank_INV"] * out["q12_wins"]
    out["NET_x_bad_losses"] = out["NET Rank_INV"] * out["q34_losses"]
    out["road_quality_x_q12"] = out["road_quality"] * out["q12_wins"]
    out["net_delta_x_current"] = out["net_delta"] * out["NET Rank_INV"]
    out["resume_x_sos"] = out["resume_balance"] * out["NETSOS_INV"]
    out["q34_losses_neg"] = -out["q34_losses"]
    out["bad_loss_penalty_neg"] = -out["bad_loss_penalty"]

    out = _add_missing_indicators(out)
    return out


def build_season_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Season" not in out.columns:
        return out

    for col in [c for c in SEASON_RELATIVE_COLS if c in out.columns]:
        season_group = out.groupby("Season")[col]
        out[f"{col}_season_pct"] = season_group.rank(method="average", pct=True)
        mean = season_group.transform("mean")
        std = season_group.transform("std").replace(0.0, np.nan)
        out[f"{col}_season_z"] = ((out[col] - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def build_conference_context_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Conference" not in out.columns or "Season" not in out.columns:
        return out

    conf_group = out.groupby(["Season", "Conference"], dropna=False)
    out["conference_team_count"] = conf_group["RecordID"].transform("count") if "RecordID" in out.columns else conf_group["SeasonStart"].transform("count")

    for col in [c for c in CONFERENCE_CONTEXT_COLS if c in out.columns]:
        conf_mean = conf_group[col].transform("mean")
        conf_std = conf_group[col].transform("std").replace(0.0, np.nan)
        out[f"{col}_conf_mean"] = conf_mean
        out[f"{col}_minus_conf_mean"] = out[col] - conf_mean
        out[f"{col}_conf_z"] = ((out[col] - conf_mean) / conf_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if {"NET Rank_INV_conf_mean", "quality_win_score_conf_mean", "resume_balance_conf_mean"}.issubset(out.columns):
        out["conference_strength_proxy"] = (
            0.5 * out["NET Rank_INV_conf_mean"]
            + 0.25 * out["quality_win_score_conf_mean"]
            + 0.25 * out["resume_balance_conf_mean"]
        )
    if {"NET Rank_INV_minus_conf_mean", "WL_PCT_minus_conf_mean"}.issubset(out.columns):
        out["conference_dominance_proxy"] = (
            0.6 * out["NET Rank_INV_minus_conf_mean"]
            + 0.4 * out["WL_PCT_minus_conf_mean"]
        )
    return out


def apply_train_fold_clipping(
    train_df: pd.DataFrame,
    other_dfs: Iterable[pd.DataFrame],
    columns: Iterable[str] | None = None,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> tuple[pd.DataFrame, list[pd.DataFrame], dict[str, tuple[float, float]]]:
    train_out = train_df.copy()
    other_out = [df.copy() for df in other_dfs]

    if columns is None:
        columns = [
            col
            for col in train_out.columns
            if pd.api.types.is_numeric_dtype(train_out[col]) and col != "Overall Seed"
        ]
    else:
        columns = list(columns)

    bounds: dict[str, tuple[float, float]] = {}
    for col in columns:
        series = pd.to_numeric(train_out[col], errors="coerce")
        valid = series.dropna()
        if valid.empty:
            continue
        low = float(valid.quantile(lower_q))
        high = float(valid.quantile(upper_q))
        bounds[col] = (low, high)
        train_out[col] = series.clip(low, high)
        for df in other_out:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").clip(low, high)
    return train_out, other_out, bounds


def make_feature_matrix(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    include_conference_context: bool = True,
) -> FeatureMatrix:
    train_feat = build_resume_features(train_df)
    eval_feat = build_resume_features(eval_df)

    train_feat = build_season_relative_features(train_feat)
    eval_feat = build_season_relative_features(eval_feat)

    if include_conference_context:
        train_feat = build_conference_context_features(train_feat)
        eval_feat = build_conference_context_features(eval_feat)

    train_feat, [eval_feat], clip_bounds = apply_train_fold_clipping(train_feat, [eval_feat])

    feature_columns = [c for c in train_feat.columns if c not in DROP_COLS]
    feature_columns = [c for c in feature_columns if c in eval_feat.columns]
    feature_columns = sorted(feature_columns)

    categorical_columns = [c for c in feature_columns if train_feat[c].dtype == "object"]
    numeric_columns = [c for c in feature_columns if c not in categorical_columns]

    return FeatureMatrix(
        train_frame=train_feat,
        eval_frame=eval_feat,
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        clip_bounds=clip_bounds,
    )
