#!/usr/bin/env python3
"""Generate leakage-aware Kaggle-test predictions with the final v8 model.

This script does not score the Kaggle test rows in-sample. Instead, it performs
leave-one-season-out scoring on the full historical season universe, then
extracts predictions only for the historical Kaggle test rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from build_submission_v8_final import (
    HIST_TEST_PATH,
    SEASONS,
    SUBMISSION_PATH,
    CandidateSpec,
    candidate_specs,
    compute_metrics,
    fit_candidate_scores,
    load_labeled_historical_universe,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "kaggle_backtest_v8"
OUTPUT_PATH = PROJECT_ROOT / "submissions" / "kaggle_backtest" / "submission_kaggle_v8_exp3_loocv.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--skip-seed-build", action="store_true")
    parser.add_argument("--candidate", default="exp3_logit_bid_ridge_seed")
    return parser.parse_args()


def pick_candidate(name: str) -> CandidateSpec:
    for candidate in candidate_specs():
        if candidate.name == name:
            if candidate.status != "active":
                raise ValueError(f"Candidate {name} is not active")
            return candidate
    raise ValueError(f"Unknown candidate: {name}")


def main() -> None:
    args = parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    historical_df = load_labeled_historical_universe(skip_seed_build=args.skip_seed_build)
    kaggle_test = pd.read_csv(HIST_TEST_PATH)
    candidate = pick_candidate(args.candidate)

    scored_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, float | str]] = []

    for holdout_season in SEASONS:
        train_subset = historical_df[historical_df["Season"] != holdout_season].copy().reset_index(drop=True)
        eval_subset = historical_df[historical_df["Season"] == holdout_season].copy().reset_index(drop=True)
        scored, details = fit_candidate_scores(train_subset, eval_subset, candidate)

        season_metrics = compute_metrics(eval_subset, scored, fit_seconds=details["fit_seconds"])
        metric_rows.append({"season": holdout_season, **season_metrics})

        kaggle_ids = set(kaggle_test.loc[kaggle_test["Season"] == holdout_season, "RecordID"])
        season_scored = scored[scored["RecordID"].isin(kaggle_ids)].copy()
        scored_rows.append(season_scored)

    scored_kaggle = pd.concat(scored_rows, ignore_index=True)
    submission = kaggle_test[["RecordID"]].merge(
        scored_kaggle[["RecordID", "PredictedSeed"]],
        on="RecordID",
        how="left",
    )
    submission = submission.rename(columns={"PredictedSeed": "Overall Seed"})
    submission["Overall Seed"] = submission["Overall Seed"].fillna(0).astype(int)
    submission.to_csv(args.output_path, index=False)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(args.artifact_dir / "kaggle_loyo_full_universe_metrics.csv", index=False)

    kaggle_eval = historical_df[historical_df["RecordID"].isin(kaggle_test["RecordID"])].copy()
    kaggle_eval = kaggle_eval[["RecordID", "TrueSeed"]].merge(
        submission,
        on="RecordID",
        how="left",
    )
    kaggle_eval["abs_err"] = (kaggle_eval["TrueSeed"] - kaggle_eval["Overall Seed"]).abs()
    kaggle_eval.to_csv(args.artifact_dir / "kaggle_test_predictions_with_truth.csv", index=False)

    print(f"Wrote Kaggle submission: {args.output_path}")
    print(f"Wrote season metrics: {args.artifact_dir / 'kaggle_loyo_full_universe_metrics.csv'}")
    print(f"Wrote scored Kaggle rows: {args.artifact_dir / 'kaggle_test_predictions_with_truth.csv'}")


if __name__ == "__main__":
    main()
