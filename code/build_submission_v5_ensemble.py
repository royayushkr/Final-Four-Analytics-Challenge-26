#!/usr/bin/env python3
"""Build a v5 ensemble from completed base experiment outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "NCAA_Seed_Training_Set2.0.csv"
DEFAULT_TEST_PATH = PROJECT_ROOT / "data" / "raw" / "NCAA_Seed_Test_Set2.0.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "submissions" / "v5_experiments"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "submissions" / "v5_ensemble"


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))


def per_season_rank(seasons: pd.Series, values: np.ndarray) -> np.ndarray:
    ranked = np.zeros(len(values), dtype=float)
    for _, idx in seasons.groupby(seasons).indices.items():
        season_idx = np.array(idx, dtype=int)
        ranked[season_idx] = (
            pd.Series(values[season_idx]).rank(method="average", pct=True).to_numpy(dtype=float)
        )
    return ranked


def season_assign(seasons: pd.Series, y_true: np.ndarray, score: np.ndarray) -> np.ndarray:
    pred = np.zeros(len(y_true), dtype=float)
    for _, idx in seasons.groupby(seasons).indices.items():
        season_idx = np.array(idx, dtype=int)
        actual = y_true[season_idx]
        assigned = np.empty(len(season_idx), dtype=float)
        assigned[np.argsort(-score[season_idx])] = np.sort(actual)
        pred[season_idx] = assigned
    return pred


def choose_top_runs(summary_csv: Path, top_k: int = 3) -> pd.DataFrame:
    summary = pd.read_csv(summary_csv)
    usable = summary[
        (summary["status"] == "completed")
        & (summary["experiment_name"] != "v5_ensemble")
        & (~summary["experiment_name"].isin(["v5_base", "v5_no_market"]))
    ].copy()
    usable = usable.sort_values(
        ["season_rank_rmse_mean", "season_rank_rmse_std", "full_rmse_zero_mean"],
        ascending=[True, True, True],
    )
    usable = usable.groupby("model_family", as_index=False).head(1)
    usable = usable.sort_values(
        ["season_rank_rmse_mean", "season_rank_rmse_std", "full_rmse_zero_mean"],
        ascending=[True, True, True],
    )
    return usable.head(top_k)


def load_oof_and_test(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    oof = pd.read_csv(run_dir / "oof_predictions.csv")
    test = pd.read_csv(run_dir / "test_predictions.csv")
    return oof, test


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", default=str(DEFAULT_OUTPUT_ROOT / "experiment_summary_table.csv"))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--train-path", default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--test-path", default=str(DEFAULT_TEST_PATH))
    parser.add_argument("--tournament-size", type=int, default=68)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_csv = Path(args.summary_csv)
    output_root = Path(args.output_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    top_runs = choose_top_runs(summary_csv, top_k=args.top_k)
    selected_runs: list[tuple[str, str]] = list(
        zip(top_runs["experiment_name"], top_runs["model_family"], strict=True)
    )

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)

    oof_merged = None
    test_merged = None
    for experiment_name, model_family in selected_runs:
        run_dir = output_root / experiment_name / model_family
        oof, test = load_oof_and_test(run_dir)
        label = f"{experiment_name}__{model_family}"
        oof = oof[["RecordID", "Season", "Overall Seed", "IsSelected", "Score"]].rename(
            columns={"Score": f"Score__{label}"}
        )
        test = test[["RecordID", "Season", "IsSelected", "Score"]].rename(
            columns={"Score": f"Score__{label}"}
        )
        oof_merged = oof if oof_merged is None else oof_merged.merge(oof, on=["RecordID", "Season", "Overall Seed", "IsSelected"], how="left")
        test_merged = test if test_merged is None else test_merged.merge(test, on=["RecordID", "Season", "IsSelected"], how="left")

    selected_train = oof_merged[oof_merged["IsSelected"] == 1].copy().reset_index(drop=True)
    selected_test = test_merged[test_merged["IsSelected"] == 1].copy().reset_index(drop=True)
    score_cols = [c for c in selected_train.columns if c.startswith("Score__")]

    train_ranked = {
        col: per_season_rank(selected_train["Season"], selected_train[col].to_numpy(dtype=float))
        for col in score_cols
    }
    test_ranked = {
        col: per_season_rank(selected_test["Season"], selected_test[col].to_numpy(dtype=float))
        for col in score_cols
    }

    y_true = selected_train["Overall Seed"].to_numpy(dtype=float)
    best_score = float("inf")
    best_weights: dict[str, float] = {}
    grid = [i / 20 for i in range(21)]

    if len(score_cols) != 3:
        raise ValueError("v5 ensemble expects exactly 3 base runs")

    for w0 in grid:
        for w1 in grid:
            w2 = 1.0 - w0 - w1
            if w2 < 0:
                continue
            weights = {score_cols[0]: w0, score_cols[1]: w1, score_cols[2]: w2}
            blend = sum(weights[col] * train_ranked[col] for col in score_cols)
            pred = season_assign(selected_train["Season"], y_true, blend)
            value = rmse(y_true, pred)
            if value < best_score:
                best_score = value
                best_weights = weights

    train_blend = sum(best_weights[col] * train_ranked[col] for col in score_cols)
    test_blend = sum(best_weights[col] * test_ranked[col] for col in score_cols)
    train_assigned = season_assign(selected_train["Season"], y_true, train_blend)

    test_selected_df = test_df.loc[test_df["Bid Type"].notna()].copy().reset_index(drop=True)
    test_assigned = assign_test_seeds(
        train_df=train_df,
        test_selected_df=test_selected_df,
        score=test_blend,
        tournament_size=args.tournament_size,
    )

    full_test_pred = np.zeros(len(test_df), dtype=float)
    full_test_pred[test_df["Bid Type"].notna().to_numpy()] = test_assigned

    pd.DataFrame(
        {
            "RecordID": selected_train["RecordID"],
            "Season": selected_train["Season"],
            "Overall Seed": y_true,
            "BlendScore": train_blend,
            "PredAssignedSeed": train_assigned,
        }
    ).to_csv(output_dir / "oof_predictions.csv", index=False)

    pd.DataFrame(
        {
            "RecordID": test_selected_df["RecordID"],
            "Season": test_selected_df["Season"],
            "BlendScore": test_blend,
            "PredAssignedSeed": test_assigned,
        }
    ).to_csv(output_dir / "test_predictions.csv", index=False)

    submission = pd.DataFrame({"RecordID": test_df["RecordID"], "Overall Seed": full_test_pred})
    submission.to_csv(output_dir / "submission.csv", index=False)

    summary = {
        "experiment_name": "v5_ensemble",
        "model_family": "rank_blend",
        "status": "completed",
        "base_runs": [f"{exp}/{model}" for exp, model in selected_runs],
        "season_rank_rmse_mean": best_score,
        "blend_weights": best_weights,
        "selected_count_test": int((full_test_pred > 0).sum()),
    }
    (output_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2))
    if summary_csv.exists():
        summary_df = pd.read_csv(summary_csv)
        summary_df = summary_df[
            ~(
                (summary_df["experiment_name"] == "v5_ensemble")
                & (summary_df["model_family"] == "rank_blend")
            )
        ].copy()
        row = {
            "experiment_name": "v5_ensemble",
            "model_family": "rank_blend",
            "status": "completed",
            "feature_blocks": "meta_scores",
            "notes": f"Rank blend of {', '.join(summary['base_runs'])}",
            "season_rank_rmse_mean": best_score,
            "season_rank_rmse_std": 0.0,
            "full_rmse_zero_mean": np.nan,
            "full_rmse_zero_std": np.nan,
            "selected_count_test": int((full_test_pred > 0).sum()),
            "non_zero_count_test": int((full_test_pred > 0).sum()),
            "base_runs": "|".join(summary["base_runs"]),
            "blend_weights": json.dumps(best_weights, sort_keys=True),
        }
        summary_df = pd.concat([summary_df, pd.DataFrame([row])], ignore_index=True)
        summary_df.to_csv(summary_csv, index=False)
        summary_df.to_json(summary_csv.with_suffix(".json"), orient="records", indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
