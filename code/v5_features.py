#!/usr/bin/env python3
"""Feature builders for v5 NCAA seed experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

import build_submission_v3 as v3

BASE_NUMERIC_MISSING_COLS = [
    "NET Rank",
    "PrevNET",
    "AvgOppNETRank",
    "AvgOppNET",
    "NETSOS",
    "NETNonConfSOS",
]

SEASON_RELATIVE_COLS = [
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
    "GOOD_WIN_RATE",
    "BAD_LOSS_RATE",
    "TOP_RESUME_INDEX",
]

CONFERENCE_RELATIVE_COLS = [
    "NET Rank",
    "PrevNET",
    "AvgOppNETRank",
    "AvgOppNET",
    "NETSOS",
]

ROBUST_NUMERIC_COLS = [
    "NET Rank",
    "PrevNET",
    "AvgOppNETRank",
    "AvgOppNET",
    "NETSOS",
    "NETNonConfSOS",
    "NET_Improvement",
    "OppNetDiff",
    "SOS_Diff",
    "Q_NET_SCORE",
    "Q_NET_SCORE_PER_GAME",
    "TOP_RESUME_INDEX",
    "SEED_STRENGTH_HEUR",
]


@dataclass(frozen=True)
class FeatureArtifacts:
    clip_bounds: dict[str, tuple[float, float]]
    target_encoding: dict[str, dict[str, Any]]


def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    out = np.divide(
        num.to_numpy(dtype=float),
        den.to_numpy(dtype=float),
        out=np.full(len(num), np.nan, dtype=float),
        where=den.to_numpy(dtype=float) != 0,
    )
    return pd.Series(out, index=num.index, dtype=float)


def add_core_parsed_features(df: pd.DataFrame) -> pd.DataFrame:
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
        "NET_WL_INTERACTION": out["NET Rank"].fillna(out["NET Rank"].median()) * out["WL_PCT"],
        "PREVNET_WL_INTERACTION": out["PrevNET"].fillna(out["PrevNET"].median()) * out["WL_PCT"],
        "Q_NET_WL_INTERACTION": out["Q_NET_SCORE"] * out["WL_PCT"],
    }
    good_win = pd.Series(extra["GOOD_WIN_RATE"], index=out.index)
    bad_loss = pd.Series(extra["BAD_LOSS_RATE"], index=out.index)
    extra["SEED_STRENGTH_HEUR"] = (
        -0.45 * out["NET Rank"].fillna(out["NET Rank"].median())
        - 0.20 * out["PrevNET"].fillna(out["PrevNET"].median())
        - 0.10 * out["AvgOppNET"].fillna(out["AvgOppNET"].median())
        + 35.0 * out["WL_PCT"].fillna(out["WL_PCT"].median())
        + 25.0 * good_win.fillna(good_win.median())
        - 25.0 * bad_loss.fillna(bad_loss.median())
    )
    return pd.concat([out, pd.DataFrame(extra, index=out.index)], axis=1)


def add_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    extra = {
        f"{col}_IS_MISSING": df[col].isna().astype(int)
        for col in BASE_NUMERIC_MISSING_COLS
        if col in df.columns
    }
    if not extra:
        return df
    return pd.concat([df, pd.DataFrame(extra, index=df.index)], axis=1)


def add_season_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    extra: dict[str, pd.Series | np.ndarray] = {}
    for col in SEASON_RELATIVE_COLS:
        if col not in df.columns:
            continue
        grp = df.groupby("Season")[col]
        extra[f"{col}_SEASON_PCT"] = grp.rank(method="average", pct=True)
        mean = grp.transform("mean")
        std = grp.transform(lambda s: s.std(ddof=0))
        extra[f"{col}_SEASON_Z"] = np.where(std > 0, (df[col] - mean) / std, 0.0)
    if not extra:
        return df
    return pd.concat([df, pd.DataFrame(extra, index=df.index)], axis=1)


def add_conference_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    extra: dict[str, pd.Series] = {}
    for col in CONFERENCE_RELATIVE_COLS:
        if col not in df.columns:
            continue
        extra[f"{col}_CONF_PCT"] = df.groupby(["Season", "Conference"])[col].rank(
            method="average", pct=True
        )
    if not extra:
        return df
    return pd.concat([df, pd.DataFrame(extra, index=df.index)], axis=1)


def fit_clip_bounds(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for col in ROBUST_NUMERIC_COLS:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        non_null = values.dropna()
        if non_null.empty:
            continue
        lo = float(non_null.quantile(0.02))
        hi = float(non_null.quantile(0.98))
        bounds[col] = (lo, hi)
    return bounds


def add_robust_numeric_features(
    df: pd.DataFrame, clip_bounds: dict[str, tuple[float, float]]
) -> pd.DataFrame:
    extra: dict[str, pd.Series | np.ndarray] = {}
    for col, (lo, hi) in clip_bounds.items():
        if col not in df.columns:
            continue
        raw = pd.to_numeric(df[col], errors="coerce")
        clipped = raw.clip(lower=lo, upper=hi)
        extra[f"{col}_CLIP"] = clipped
        extra[f"{col}_EXTREME_FLAG"] = ((raw < lo) | (raw > hi)).fillna(False).astype(int)
    if not extra:
        return df
    return pd.concat([df, pd.DataFrame(extra, index=df.index)], axis=1)


def add_interaction_refinements(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    extra = {
        "GOOD_BAD_BALANCE": out["GOOD_WIN_RATE"] - out["BAD_LOSS_RATE"],
        "Q1_Q2_WIN_PRESSURE": out["Q1_WIN_RATE"] * out["Q1_GAME_SHARE"]
        + out["Q2_WIN_RATE"] * out["Q2_GAME_SHARE"],
        "NETSOS_GOODWIN_INTERACTION": out["GOOD_WIN_RATE"] * out["NETSOS_INV"],
        "NONCONF_QUALITY_INDEX": out["Non-ConferenceRecord_PCT"] * out["NETNonConfSOS_INV"],
        "ROAD_QUALITY_INDEX": out["RoadWL_PCT"] * out["NETSOS_INV"],
        "RESUME_EFFICIENCY": safe_ratio(out["TOP_RESUME_INDEX"], out["WL_G"]),
        "SEED_STRENGTH_x_QNET": out["SEED_STRENGTH_HEUR"] * out["Q_NET_SCORE_PER_GAME"],
    }
    return pd.concat([out, pd.DataFrame(extra, index=out.index)], axis=1)


def fit_target_encoding(
    train_df: pd.DataFrame,
    columns: tuple[str, ...] = ("Team", "Conference"),
    alpha: float = 10.0,
) -> dict[str, dict[str, Any]]:
    state: dict[str, dict[str, Any]] = {}
    global_mean = float(train_df["Overall Seed"].mean())
    for col in columns:
        stats = train_df.groupby(col)["Overall Seed"].agg(["mean", "count"])
        smooth = (stats["mean"] * stats["count"] + global_mean * alpha) / (
            stats["count"] + alpha
        )
        state[col] = {
            "mapping": smooth.to_dict(),
            "global_mean": global_mean,
        }
    return state


def add_target_encoding_features(
    df: pd.DataFrame,
    state: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    extra: dict[str, pd.Series] = {}
    for col, spec in state.items():
        if col not in df.columns:
            continue
        extra[f"{col}_SEED_TE"] = df[col].map(spec["mapping"]).fillna(spec["global_mean"])
    if not extra:
        return df
    return pd.concat([df, pd.DataFrame(extra, index=df.index)], axis=1)


def fit_feature_artifacts(
    train_selected_df: pd.DataFrame,
    feature_blocks: tuple[str, ...],
) -> FeatureArtifacts:
    base = add_core_parsed_features(train_selected_df)
    clip_bounds = fit_clip_bounds(base) if "robust_numeric" in feature_blocks else {}
    target_encoding = (
        fit_target_encoding(train_selected_df)
        if "team_conference_encoding" in feature_blocks
        else {}
    )
    return FeatureArtifacts(clip_bounds=clip_bounds, target_encoding=target_encoding)


def build_feature_frame(
    df: pd.DataFrame,
    feature_blocks: tuple[str, ...],
    include_team_feature: bool,
    include_bid_type_feature: bool,
    artifacts: FeatureArtifacts | None = None,
) -> pd.DataFrame:
    out = add_core_parsed_features(df)
    if "missing_indicators" in feature_blocks:
        out = add_missing_indicators(out)
    if "season_relative" in feature_blocks:
        out = add_season_relative_features(out)
    if "conference_relative" in feature_blocks:
        out = add_conference_relative_features(out)
    if "robust_numeric" in feature_blocks:
        bounds = artifacts.clip_bounds if artifacts is not None else fit_clip_bounds(out)
        out = add_robust_numeric_features(out, bounds)
    if "interaction_refinements" in feature_blocks:
        out = add_interaction_refinements(out)
    if "team_conference_encoding" in feature_blocks and artifacts is not None:
        out = add_target_encoding_features(out, artifacts.target_encoding)

    drop_cols = ["RecordID", "Overall Seed", *v3.RECORD_COLS]
    if not include_team_feature:
        drop_cols.append("Team")
    if not include_bid_type_feature:
        drop_cols.append("Bid Type")
    return out.drop(columns=drop_cols, errors="ignore")
