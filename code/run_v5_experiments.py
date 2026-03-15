#!/usr/bin/env python3
"""Batch runner for v5 experiments and summary aggregation."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from build_submission_v5 import run_experiment
from v5_config import ABLATION_ORDER, EXPERIMENTS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "submissions" / "v5_experiments"
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "NCAA_Seed_Training_Set2.0.csv"
DEFAULT_TEST_PATH = PROJECT_ROOT / "data" / "raw" / "NCAA_Seed_Test_Set2.0.csv"

ALIASES = {
    "v5_base": "v5_base_full",
    "v5_no_market": "v5_base_full",
}


def clone_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--train-path", default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--test-path", default=str(DEFAULT_TEST_PATH))
    parser.add_argument("--tournament-size", type=int, default=68)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []

    for experiment_name in ABLATION_ORDER:
        config = EXPERIMENTS[experiment_name]

        if config.status == "skipped":
            skip_dir = output_root / experiment_name
            skip_dir.mkdir(parents=True, exist_ok=True)
            skip_summary = {
                "experiment_name": experiment_name,
                "model_family": "n/a",
                "status": config.status,
                "skip_reason": config.skip_reason,
                "feature_blocks": list(config.feature_blocks),
                "notes": config.notes,
            }
            (skip_dir / "experiment_summary.json").write_text(json.dumps(skip_summary, indent=2))
            summary_rows.append(
                {
                    "experiment_name": experiment_name,
                    "model_family": "n/a",
                    "status": config.status,
                    "skip_reason": config.skip_reason,
                    "feature_blocks": "|".join(config.feature_blocks),
                    "notes": config.notes,
                }
            )
            continue

        if experiment_name in ALIASES:
            source_experiment = ALIASES[experiment_name]
            for model_family in EXPERIMENTS[source_experiment].model_families:
                src = output_root / source_experiment / model_family
                dst = output_root / experiment_name / model_family
                clone_tree(src, dst)
                summary = json.loads((dst / "experiment_summary.json").read_text())
                summary["experiment_name"] = experiment_name
                summary["notes"] = f"Alias of {source_experiment}. {summary.get('notes', '')}".strip()
                (dst / "experiment_summary.json").write_text(json.dumps(summary, indent=2))
                summary_rows.append(
                    {
                        "experiment_name": experiment_name,
                        "model_family": model_family,
                        **{
                            k: v
                            for k, v in summary.items()
                            if k not in {"experiment_name", "model_family"}
                        },
                        "feature_blocks": "|".join(summary["feature_blocks"]),
                    }
                )
            continue

        for model_family in config.model_families:
            summary = run_experiment(
                experiment_name=experiment_name,
                model_family=model_family,
                output_root=output_root,
                train_path=Path(args.train_path),
                test_path=Path(args.test_path),
                tournament_size=args.tournament_size,
            )
            row = {
                "experiment_name": experiment_name,
                "model_family": model_family,
                **{k: v for k, v in summary.items() if k not in {"experiment_name", "model_family"}},
            }
            if "feature_blocks" in row and isinstance(row["feature_blocks"], list):
                row["feature_blocks"] = "|".join(row["feature_blocks"])
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_root / "experiment_summary_table.csv", index=False)
    summary_df.to_json(output_root / "experiment_summary_table.json", orient="records", indent=2)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
