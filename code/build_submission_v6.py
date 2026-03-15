#!/usr/bin/env python3
"""Build conservative v6 hedge submissions anchored on the best public baseline."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "NCAA_Seed_Training_Set2.0.csv"
TEST_PATH = PROJECT_ROOT / "data" / "raw" / "NCAA_Seed_Test_Set2.0.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "submissions" / "v6"

V2_SUBMISSION = PROJECT_ROOT / "submissions" / "no_external_best" / "submission_no_leak_v2.csv"
V3_DIAGNOSTICS = PROJECT_ROOT / "submissions" / "v3_no_external" / "diagnostics_v3_no_external.csv"
V3_ENSEMBLE = PROJECT_ROOT / "submissions" / "v3_ensemble" / "submission_v3_ensemble.csv"

V5_TEST_FILES = {
    "v5_cur_ridge": PROJECT_ROOT
    / "submissions"
    / "v5_experiments"
    / "current_v4"
    / "ridge"
    / "test_predictions.csv",
    "v5_lgbm": PROJECT_ROOT
    / "submissions"
    / "v5_experiments"
    / "v5_base_core"
    / "lightgbm"
    / "test_predictions.csv",
    "v5_ridge": PROJECT_ROOT
    / "submissions"
    / "v5_experiments"
    / "v5_base_full"
    / "ridge"
    / "test_predictions.csv",
}

CANDIDATES = {
    "v6_primary": {
        "weights": {
            "base_rank": 0.65,
            "v5_ridge_rank": 0.175,
            "v3_order_rank": 0.175,
        },
        "description": (
            "Primary hedge. Changes only the Davidson/Notre Dame ordering from v3_ensemble, "
            "a swap supported by v3 final assignment, v4, and the v5 single-model variants."
        ),
    },
    "v6_conservative_low_seed": {
        "weights": {
            "base_rank": 0.75,
            "v5_cur_ridge_rank": 0.25,
        },
        "description": (
            "Most conservative alternative. Only touches a low-seed 61/62 swap while "
            "adding a small amount of stable current_v4 ridge signal."
        ),
    },
    "v6_aggressive_ridge": {
        "weights": {
            "base_rank": 0.75,
            "v5_ridge_rank": 0.25,
        },
        "description": (
            "Higher-variance hedge driven by the strongest v5 ridge rank signal. "
            "Makes four medium-impact swaps versus v3_ensemble."
        ),
    },
    "v6_lgbm_low_seed": {
        "weights": {
            "base_rank": 0.75,
            "v5_lgbm_rank": 0.25,
        },
        "description": (
            "Alternative low-impact hedge using the v5 LightGBM rank signal. "
            "Changes only one 50/51 swap versus v3_ensemble."
        ),
    },
}


def run_cmd(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def ensure_inputs() -> None:
    if not V2_SUBMISSION.exists():
        run_cmd(
            [
                sys.executable,
                str(PROJECT_ROOT / "code" / "build_submission_v2.py"),
                "--include-bid-type",
                "--exclude-team",
                "--output-dir",
                str(PROJECT_ROOT / "submissions" / "no_external_best"),
            ]
        )

    if not V3_DIAGNOSTICS.exists():
        run_cmd([sys.executable, str(PROJECT_ROOT / "code" / "build_submission_v3.py")])

    if not V3_ENSEMBLE.exists():
        run_cmd([sys.executable, str(PROJECT_ROOT / "code" / "build_submission_v3_ensemble.py")])

    if not V5_TEST_FILES["v5_cur_ridge"].exists():
        run_cmd(
            [
                sys.executable,
                str(PROJECT_ROOT / "code" / "build_submission_v5.py"),
                "--experiment",
                "current_v4",
                "--model-family",
                "ridge",
            ]
        )

    if not V5_TEST_FILES["v5_lgbm"].exists():
        run_cmd(
            [
                sys.executable,
                str(PROJECT_ROOT / "code" / "build_submission_v5.py"),
                "--experiment",
                "v5_base_core",
                "--model-family",
                "lightgbm",
            ]
        )

    if not V5_TEST_FILES["v5_ridge"].exists():
        run_cmd(
            [
                sys.executable,
                str(PROJECT_ROOT / "code" / "build_submission_v5.py"),
                "--experiment",
                "v5_base_full",
                "--model-family",
                "ridge",
            ]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def assign_available_seeds(
    train_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    score: np.ndarray,
    tournament_size: int = 68,
) -> np.ndarray:
    assigned_all = np.zeros(len(selected_df), dtype=float)
    seeded_train_by_season = (
        train_df[train_df["Overall Seed"].notna()]
        .groupby("Season")["Overall Seed"]
        .apply(lambda s: set(s.astype(int)))
        .to_dict()
    )

    for season, idx in selected_df.groupby("Season").indices.items():
        season_idx = np.array(idx, dtype=int)
        known = seeded_train_by_season.get(season, set())
        available = sorted([s for s in range(1, tournament_size + 1) if s not in known])
        if len(available) != len(season_idx):
            raise ValueError(
                f"Unexpected available seed count for {season}: {len(available)} vs {len(season_idx)}"
            )
        assigned = np.empty(len(season_idx), dtype=float)
        assigned[np.argsort(-score[season_idx])] = np.array(available, dtype=float)
        assigned_all[season_idx] = assigned

    return assigned_all


def build_selected_frame(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    v2_pred = pd.read_csv(V2_SUBMISSION)
    v3_diag = pd.read_csv(V3_DIAGNOSTICS)
    v5_cur = pd.read_csv(V5_TEST_FILES["v5_cur_ridge"])
    v5_lgbm = pd.read_csv(V5_TEST_FILES["v5_lgbm"])
    v5_ridge = pd.read_csv(V5_TEST_FILES["v5_ridge"])

    selected = (
        test_df[["RecordID", "Season", "Team", "Bid Type", "NET Rank"]]
        .merge(v2_pred, on="RecordID", how="left")
        .merge(v3_diag[["RecordID", "SeedPredRaw", "SeedPredLocalIso"]], on="RecordID", how="left")
        .merge(v5_cur[["RecordID", "Score"]].rename(columns={"Score": "v5_cur_ridge"}), on="RecordID", how="left")
        .merge(v5_lgbm[["RecordID", "Score"]].rename(columns={"Score": "v5_lgbm"}), on="RecordID", how="left")
        .merge(v5_ridge[["RecordID", "Score"]].rename(columns={"Score": "v5_ridge"}), on="RecordID", how="left")
    )
    selected = selected[selected["Bid Type"].notna()].copy().reset_index(drop=True)
    required_cols = [
        "Overall Seed",
        "SeedPredRaw",
        "SeedPredLocalIso",
        "v5_cur_ridge",
        "v5_lgbm",
        "v5_ridge",
    ]
    missing_cols = [col for col in required_cols if selected[col].isna().any()]
    if missing_cols:
        raise ValueError(f"Missing required inputs for v6 build: {missing_cols}")

    net_fill = float(np.nanmedian(train_df["NET Rank"]))
    net_rank = selected["NET Rank"].fillna(net_fill).to_numpy(dtype=float)

    selected["base_score"] = 0.60 * selected["Overall Seed"] + 0.40 * selected["SeedPredRaw"]
    selected["v3_order_score"] = (
        0.55 * selected["SeedPredRaw"] + 0.30 * selected["SeedPredLocalIso"] + 0.15 * net_rank
    )

    # Lower predicted seed is stronger, so use descending rank after negating the score.
    selected["base_rank"] = selected.groupby("Season")["base_score"].rank(
        method="average", pct=True, ascending=False
    )
    selected["v3_order_rank"] = selected.groupby("Season")["v3_order_score"].rank(
        method="average", pct=True, ascending=False
    )
    selected["v5_cur_ridge_rank"] = selected.groupby("Season")["v5_cur_ridge"].rank(
        method="average", pct=True, ascending=True
    )
    selected["v5_lgbm_rank"] = selected.groupby("Season")["v5_lgbm"].rank(
        method="average", pct=True, ascending=True
    )
    selected["v5_ridge_rank"] = selected.groupby("Season")["v5_ridge"].rank(
        method="average", pct=True, ascending=True
    )

    return selected


def build_candidate_submission(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    weights: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score = np.zeros(len(selected_df), dtype=float)
    for feature_name, weight in weights.items():
        score += weight * selected_df[feature_name].to_numpy(dtype=float)

    assigned_selected = assign_available_seeds(train_df, selected_df, score)
    full_pred = np.zeros(len(test_df), dtype=float)
    full_pred[test_df["Bid Type"].notna().to_numpy()] = assigned_selected

    submission = pd.DataFrame({"RecordID": test_df["RecordID"], "Overall Seed": full_pred})
    selected_diag = selected_df[
        [
            "RecordID",
            "Season",
            "Team",
            "base_score",
            "v3_order_score",
            "v5_cur_ridge",
            "v5_lgbm",
            "v5_ridge",
            "base_rank",
            "v3_order_rank",
            "v5_cur_ridge_rank",
            "v5_lgbm_rank",
            "v5_ridge_rank",
        ]
    ].copy()
    selected_diag["BlendScore"] = score
    selected_diag["AssignedSeed"] = assigned_selected
    return submission, selected_diag


def main() -> None:
    args = parse_args()
    ensure_inputs()

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    baseline = pd.read_csv(V3_ENSEMBLE).rename(columns={"Overall Seed": "BaselineSeed"})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_df = build_selected_frame(train_df, test_df)

    summary_rows: list[dict[str, object]] = []
    candidate_diag = selected_df[["RecordID", "Season", "Team"]].copy()

    primary_path: Path | None = None
    for candidate_name, spec in CANDIDATES.items():
        submission, diag = build_candidate_submission(
            train_df=train_df,
            test_df=test_df,
            selected_df=selected_df,
            weights=spec["weights"],
        )

        candidate_path = output_dir / f"{candidate_name}.csv"
        submission.to_csv(candidate_path, index=False)
        diag.to_csv(output_dir / f"{candidate_name}_diagnostics.csv", index=False)

        merged = submission.merge(baseline, on="RecordID", how="left")
        changed = merged[merged["Overall Seed"] != merged["BaselineSeed"]].copy()
        changed_records = changed["RecordID"].tolist()

        summary_rows.append(
            {
                "candidate_name": candidate_name,
                "description": spec["description"],
                "weights": json.dumps(spec["weights"], sort_keys=True),
                "changed_rows_vs_v3_ensemble": int(len(changed)),
                "changed_records": "|".join(changed_records),
            }
        )

        candidate_diag[candidate_name] = diag["AssignedSeed"].to_numpy(dtype=float)

        if candidate_name == "v6_primary":
            primary_path = candidate_path

    if primary_path is None:
        raise ValueError("Missing v6_primary candidate")

    shutil.copyfile(primary_path, output_dir / "submission_v6.csv")
    candidate_diag.to_csv(output_dir / "v6_candidate_comparison.csv", index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "v6_candidate_summary.csv", index=False)
    (output_dir / "v6_candidate_summary.json").write_text(
        json.dumps(summary_rows, indent=2)
    )

    print(f"Wrote primary submission: {output_dir / 'submission_v6.csv'}")
    print(f"Wrote candidate summary: {output_dir / 'v6_candidate_summary.csv'}")


if __name__ == "__main__":
    main()
