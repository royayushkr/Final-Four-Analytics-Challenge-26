#!/usr/bin/env python3
"""Validation utilities for the private 2026 NCAA seed pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ExpandingWindowSplit:
    fold_name: str
    train_seasons: list[str]
    valid_season: str
    weight: float


def make_expanding_window_splits(train_df: pd.DataFrame) -> list[ExpandingWindowSplit]:
    seasons = sorted(train_df["Season"].astype(str).unique())
    if len(seasons) < 5:
        raise ValueError(f"Expected at least 5 seasons, found {len(seasons)}")
    target_weights = [0.2, 0.3, 0.5]
    valid_seasons = seasons[2:]
    return [
        ExpandingWindowSplit(
            fold_name=f"{'+'.join(seasons[:i])}->{seasons[i]}",
            train_seasons=seasons[:i],
            valid_season=seasons[i],
            weight=target_weights[i - 2],
        )
        for i in range(2, len(seasons))
    ]


def assign_unique_overall_seeds(df: pd.DataFrame, score_col: str, top_k: int = 68) -> pd.DataFrame:
    out = df.copy()
    out["PredictedSeed"] = 0
    sort_cols = [score_col]
    ascending = [False]
    if "NET Rank" in out.columns:
        sort_cols.append("NET Rank")
        ascending.append(True)
    if "RecordID" in out.columns:
        sort_cols.append("RecordID")
        ascending.append(True)

    ranked = out.sort_values(sort_cols, ascending=ascending).reset_index(drop=False)
    selected = ranked.head(top_k).copy()
    selected["PredictedSeed"] = np.arange(1, len(selected) + 1)
    out.loc[selected["index"], "PredictedSeed"] = selected["PredictedSeed"].to_numpy(dtype=int)
    return out


def compute_field_metrics_at_68(
    y_true_selected: np.ndarray,
    predicted_seed: np.ndarray,
    top_k: int = 68,
) -> dict[str, float]:
    pred_selected = (predicted_seed > 0).astype(int)
    true_selected = y_true_selected.astype(int)

    tp = int(((pred_selected == 1) & (true_selected == 1)).sum())
    fp = int(((pred_selected == 1) & (true_selected == 0)).sum())
    fn = int(((pred_selected == 0) & (true_selected == 1)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else (2.0 * precision * recall) / (precision + recall)
    return {
        "FieldPrecision@68": float(precision),
        "FieldRecall@68": float(recall),
        "FieldF1@68": float(f1),
        "TrueSelectedCount": int(true_selected.sum()),
        "PredSelectedCount": int(pred_selected.sum()),
        "TP": tp,
        "FP": fp,
        "FN": fn,
    }


def compute_confirmed_seed_rmse(
    y_true_seed: pd.Series,
    predicted_seed: pd.Series,
) -> float:
    y_true = y_true_seed.fillna(0.0).to_numpy(dtype=float)
    y_pred = predicted_seed.fillna(0.0).to_numpy(dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _compute_spearman(y_true_seed: pd.Series, predicted_seed: pd.Series) -> float:
    frame = pd.DataFrame({"y_true": y_true_seed, "y_pred": predicted_seed}).dropna()
    if len(frame) < 2:
        return float("nan")
    return float(frame["y_true"].corr(frame["y_pred"], method="spearman"))


def score_full_holdout_season(
    holdout_df: pd.DataFrame,
    score_col: str,
    top_k: int = 68,
) -> tuple[dict[str, float], pd.DataFrame]:
    ranked = assign_unique_overall_seeds(holdout_df, score_col=score_col, top_k=top_k)
    field_metrics = compute_field_metrics_at_68(
        y_true_selected=ranked["AuditSelected"].to_numpy(dtype=int),
        predicted_seed=ranked["PredictedSeed"].to_numpy(dtype=int),
        top_k=top_k,
    )

    confirmed_mask = ranked["Overall Seed"].notna()
    confirmed_seed_rmse = compute_confirmed_seed_rmse(
        y_true_seed=ranked.loc[confirmed_mask, "Overall Seed"],
        predicted_seed=ranked.loc[confirmed_mask, "PredictedSeed"],
    )
    confirmed_top68_recall = float(
        (ranked.loc[confirmed_mask, "PredictedSeed"] > 0).mean() if int(confirmed_mask.sum()) > 0 else np.nan
    )
    spearman = _compute_spearman(
        y_true_seed=ranked.loc[confirmed_mask, "Overall Seed"],
        predicted_seed=ranked.loc[confirmed_mask, "PredictedSeed"],
    )
    composite_loss = confirmed_seed_rmse + 8.0 * (1.0 - field_metrics["FieldRecall@68"]) + 4.0 * (1.0 - field_metrics["FieldPrecision@68"])

    metrics = {
        **field_metrics,
        "ConfirmedSeedRMSE": confirmed_seed_rmse,
        "ConfirmedTop68Recall": confirmed_top68_recall,
        "SeedOrderSpearman": spearman,
        "CompositeLoss": float(composite_loss),
    }
    return metrics, ranked


def compute_weekly_stability_metrics(predictions_by_week: dict[str, pd.DataFrame]) -> pd.DataFrame:
    weeks = sorted(predictions_by_week)
    rows: list[dict[str, Any]] = []
    for i in range(1, len(weeks)):
        prev_week = weeks[i - 1]
        curr_week = weeks[i]
        prev_df = predictions_by_week[prev_week].copy()
        curr_df = predictions_by_week[curr_week].copy()
        prev_top68 = set(prev_df.loc[prev_df["PredictedSeed"] > 0, "RecordID"])
        curr_top68 = set(curr_df.loc[curr_df["PredictedSeed"] > 0, "RecordID"])
        union_top68 = prev_top68 | curr_top68
        inter_top68 = prev_top68 & curr_top68
        jaccard = len(inter_top68) / max(len(union_top68), 1)

        prev_top16 = set(prev_df.loc[(prev_df["PredictedSeed"] > 0) & (prev_df["PredictedSeed"] <= 16), "RecordID"])
        curr_top16 = set(curr_df.loc[(curr_df["PredictedSeed"] > 0) & (curr_df["PredictedSeed"] <= 16), "RecordID"])
        top16_overlap = len(prev_top16 & curr_top16) / max(len(prev_top16 | curr_top16), 1)

        prev80 = set(prev_df.nsmallest(80, "RankOrder")["RecordID"])
        curr80 = set(curr_df.nsmallest(80, "RankOrder")["RecordID"])
        union80 = sorted(prev80 | curr80)
        prev_map = prev_df.set_index("RecordID")["PredictedSeed"]
        curr_map = curr_df.set_index("RecordID")["PredictedSeed"]
        seed_movement = []
        for record_id in union80:
            prev_seed = int(prev_map.get(record_id, 0))
            curr_seed = int(curr_map.get(record_id, 0))
            seed_movement.append(abs(prev_seed - curr_seed))

        rows.append(
            {
                "prev_week": prev_week,
                "curr_week": curr_week,
                "Top68Jaccard": float(jaccard),
                "Top16Overlap": float(top16_overlap),
                "Top80MeanSeedMovement": float(np.mean(seed_movement) if seed_movement else 0.0),
            }
        )

    if "20260308" in predictions_by_week and "20260315" in predictions_by_week:
        prev_df = predictions_by_week["20260308"]
        curr_df = predictions_by_week["20260315"]
        prev_top68 = set(prev_df.loc[prev_df["PredictedSeed"] > 0, "RecordID"])
        curr_top68 = set(curr_df.loc[curr_df["PredictedSeed"] > 0, "RecordID"])
        jaccard = len(prev_top68 & curr_top68) / max(len(prev_top68 | curr_top68), 1)
        rows.append(
            {
                "prev_week": "20260308",
                "curr_week": "20260315",
                "Top68Jaccard": float(jaccard),
                "Top16Overlap": float(
                    len(
                        set(prev_df.loc[(prev_df["PredictedSeed"] > 0) & (prev_df["PredictedSeed"] <= 16), "RecordID"])
                        & set(curr_df.loc[(curr_df["PredictedSeed"] > 0) & (curr_df["PredictedSeed"] <= 16), "RecordID"])
                    )
                    / max(
                        len(
                            set(prev_df.loc[(prev_df["PredictedSeed"] > 0) & (prev_df["PredictedSeed"] <= 16), "RecordID"])
                            | set(curr_df.loc[(curr_df["PredictedSeed"] > 0) & (curr_df["PredictedSeed"] <= 16), "RecordID"])
                        ),
                        1,
                    )
                ),
                "Top80MeanSeedMovement": float(
                    np.mean(
                        [
                            abs(
                                int(prev_df.set_index("RecordID")["PredictedSeed"].get(rid, 0))
                                - int(curr_df.set_index("RecordID")["PredictedSeed"].get(rid, 0))
                            )
                            for rid in sorted(
                                set(prev_df.nsmallest(80, "RankOrder")["RecordID"])
                                | set(curr_df.nsmallest(80, "RankOrder")["RecordID"])
                            )
                        ]
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def select_final_candidate(
    candidate_summary: pd.DataFrame,
    weekly_summary: pd.DataFrame,
) -> dict[str, Any]:
    summary = candidate_summary.copy()
    stability = (
        weekly_summary.groupby("candidate")[["Top68Jaccard", "Top16Overlap", "Top80MeanSeedMovement"]]
        .mean()
        .reset_index()
    )
    merged = summary.merge(stability, on="candidate", how="left")
    recent_ok = merged[["RecentFoldRecallMin"]].fillna(0.0)["RecentFoldRecallMin"] >= 0.90
    filtered = merged[recent_ok].copy()
    if filtered.empty:
        filtered = merged.copy()

    best_loss = float(filtered["WeightedCompositeLoss"].min())
    near_best = filtered[filtered["WeightedCompositeLoss"] <= best_loss * 1.02].copy()
    near_best = near_best.sort_values(
        ["WeightedCompositeLoss", "Top68Jaccard", "Top16Overlap", "Top80MeanSeedMovement"],
        ascending=[True, False, False, True],
    )
    chosen = near_best.iloc[0].to_dict()
    chosen["selection_reason"] = (
        "Best weighted composite loss with recency recall guard and weekly stability tiebreak"
    )
    return chosen


def write_model_comparison(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2))
