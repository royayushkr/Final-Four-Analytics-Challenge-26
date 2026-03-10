#!/usr/bin/env python3
"""Blend v2 and v3 local predictions and assign constrained seeds by season."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "NCAA_Seed_Training_Set2.0.csv"
TEST_PATH = PROJECT_ROOT / "data" / "raw" / "NCAA_Seed_Test_Set2.0.csv"
V2_DIR = PROJECT_ROOT / "submissions" / "no_external_best"
V3_DIR = PROJECT_ROOT / "submissions" / "v3_no_external"
DEFAULT_OUT_DIR = PROJECT_ROOT / "submissions" / "v3_ensemble"


def ensure_inputs() -> None:
    v2_file = V2_DIR / "submission_no_leak_v2.csv"
    v3_diag = V3_DIR / "diagnostics_v3_no_external.csv"

    if not v2_file.exists():
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "code" / "build_submission_v2.py"),
                "--include-bid-type",
                "--exclude-team",
                "--output-dir",
                str(V2_DIR),
            ],
            check=True,
        )

    if not v3_diag.exists():
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "code" / "build_submission_v3.py")],
            check=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight-v2", type=float, default=0.6)
    parser.add_argument("--weight-v3", type=float, default=0.4)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total = args.weight_v2 + args.weight_v3
    if total <= 0:
        raise ValueError("weight-v2 + weight-v3 must be > 0")
    w2 = args.weight_v2 / total
    w3 = args.weight_v3 / total

    ensure_inputs()

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    pred_v2 = pd.read_csv(V2_DIR / "submission_no_leak_v2.csv")
    diag_v3 = pd.read_csv(V3_DIR / "diagnostics_v3_no_external.csv")

    merged = (
        test[["RecordID", "Season", "Bid Type"]]
        .merge(pred_v2, on="RecordID", how="left")
        .merge(diag_v3[["RecordID", "SeedPredRaw"]], on="RecordID", how="left")
    )
    if merged["Overall Seed"].isna().any() or merged["SeedPredRaw"].isna().any():
        raise ValueError("Missing inputs while merging v2 and v3 predictions")

    score = w2 * merged["Overall Seed"].to_numpy(dtype=float) + w3 * merged["SeedPredRaw"].to_numpy(dtype=float)
    selected = merged["Bid Type"].notna().to_numpy()
    final = np.zeros(len(merged), dtype=float)

    for season, idx in merged.groupby("Season").groups.items():
        season_idx = np.array(list(idx))
        season_selected = season_idx[selected[season_idx]]
        if len(season_selected) == 0:
            continue

        known = set(
            train[(train["Season"] == season) & train["Overall Seed"].notna()]["Overall Seed"].astype(int)
        )
        available = sorted([s for s in range(1, 69) if s not in known])

        order = np.argsort(score[season_selected])
        assigned = np.zeros(len(season_selected), dtype=float)
        if len(available) == len(season_selected):
            assigned[order] = np.array(available, dtype=float)
        else:
            # Defensive fallback for unexpected shape mismatch.
            avail = np.array(available, dtype=float)
            q = np.linspace(0, len(avail) - 1, len(season_selected)).round().astype(int)
            assigned[order] = np.sort(avail[q])
        final[season_selected] = assigned

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "submission_v3_ensemble.csv"
    pd.DataFrame({"RecordID": merged["RecordID"], "Overall Seed": final}).to_csv(out_path, index=False)

    print(f"Wrote: {out_path}")
    print(f"Weights -> v2: {w2:.3f}, v3: {w3:.3f}")
    print(f"Non-zero predictions: {(final > 0).sum()}")


if __name__ == "__main__":
    main()
