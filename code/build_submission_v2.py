#!/usr/bin/env python3
"""Leakage-safe v2 runner.

This script is a thin wrapper around build_submission.py with strict defaults:
- no external data
- Bid Type excluded
- strategy selected via train-only CV
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "NCAA_Seed_Training_Set2.0.csv"
DEFAULT_TEST_PATH = PROJECT_ROOT / "data" / "raw" / "NCAA_Seed_Test_Set2.0.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "submissions" / "generated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--test-path", default=str(DEFAULT_TEST_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--include-bid-type",
        action="store_true",
        help="Use Bid Type as a feature (provided in dataset, but target-adjacent).",
    )
    parser.add_argument(
        "--exclude-team",
        action="store_true",
        help="Drop Team feature from model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = Path(__file__).with_name("build_submission.py")

    cmd = [
        sys.executable,
        str(runner),
        "--train-path",
        args.train_path,
        "--test-path",
        args.test_path,
        "--output-dir",
        args.output_dir,
        "--cv-folds",
        str(args.cv_folds),
        "--prediction-strategy",
        "auto",
    ]
    if args.include_bid_type:
        cmd.append("--include-bid-type")
    if args.exclude_team:
        cmd.append("--exclude-team")

    print("Running leakage-safe v2 pipeline:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
