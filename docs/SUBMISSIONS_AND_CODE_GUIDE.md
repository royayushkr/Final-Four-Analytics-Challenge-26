# Submissions and Code Guide

This guide explains:

1. What each main `submissions/` folder means.
2. How each script in `code/` works.
3. Which artifacts are judge-safe for no-external-data submissions.

## Quick Recommendation

Recommended clean file (no external web data):  
`submissions/v3_ensemble/submission_v3_ensemble.csv`

## Submission Folders

### `submissions/generated/`
- Purpose: historical output bucket from earlier pipeline runs.
- Contents: mixed v1/v2 outputs and diagnostics generated during development.
- Data policy: mixed history.
- Judge note: do not use as primary judge handoff because it includes artifacts from different experiment phases.

### `submissions/with_bid_type/`
- Purpose: v2 local-model outputs where `Bid Type` is used as a model feature.
- Contents: `submission_no_leak_v2.csv`, seed-only variant, probability diagnostics, CV metrics.
- Data policy: raw competition files only.
- Behavior: stronger selection performance because `Bid Type` is highly informative.

### `submissions/no_external_best/`
- Purpose: best-performing v2 run among no-external-data experiments.
- Contents: v2 submission and diagnostics.
- Data policy: raw competition files only.
- Behavior: two-stage v2 model tuned with `Bid Type` enabled and `Team` optionally excluded to improve generalization.

### `submissions/v3_no_external/`
- Purpose: primary v3 single-pipeline outputs.
- Contents: `submission_v3_no_external.csv`, diagnostics, OOF metrics.
- Data policy: raw competition files only.
- Behavior: stronger feature engineering plus season-constrained assignment.

### `submissions/v3_ensemble/`
- Purpose: blended final candidate combining v2 and v3 prediction signals.
- Contents: `submission_v3_ensemble.csv`.
- Data policy: raw competition files only.
- Behavior: blends local outputs, then applies season-level constrained seed assignment.
- Why this is preferred: it captures stable signal from v2 and richer ranking signal from v3.

### `submissions/legacy/`
- Purpose: archive of prior invalid or non-compliant artifacts.
- Contents: `submission_v2_leaked.csv`.
- Data policy: includes external/internet-derived mapping history.
- Judge note: keep for audit only; do not submit.

## Script-by-Script Explanation

### `code/build_submission.py`
- Role: base v2 training and inference pipeline.
- Core steps:
1. Load train/test CSVs.
2. Parse malformed win-loss fields (`8-Sep` style conversions).
3. Engineer derived features (`*_PCT`, NET deltas, SOS deltas).
4. Build preprocessing stack:
   numeric median imputation + categorical one-hot encoding.
5. Train two-stage models:
   selection classifier + seed regressor.
6. Run GroupKFold validation by `Season` and report metrics.
7. Auto-select output strategy (`hard`, `soft`, `soft-threshold`) from OOF results.
8. Write submission and diagnostics.
- External data usage: none.

### `code/build_submission_v2.py`
- Role: wrapper runner around `build_submission.py`.
- Core steps:
1. Sets default project paths.
2. Passes selected CLI flags to `build_submission.py`.
3. Supports easy toggles:
   `--include-bid-type`, `--exclude-team`.
- External data usage: none.

### `code/build_submission_v3.py`
- Role: stronger no-external-data pipeline with richer modeling and post-processing.
- Core steps:
1. Feature engineering expansion:
   rank inverse/log transforms, quadrant weighted scores, margin and share features.
2. Selection modeling:
   XGB classifier predicts tournament inclusion probability.
3. Seed modeling:
   two XGB regressors ensemble for seed score.
4. Local season correction:
   isotonic calibration by season using `NET Rank` vs seed relation.
5. Selection quotas:
   season target selected count derived from train (`68 - seeded_in_train_season`).
6. Constrained assignment:
   selected teams are mapped to season-available seeds (missing from train for that season).
7. Write submission, diagnostics, and OOF metrics.
- External data usage: none.

### `code/build_submission_v3_ensemble.py`
- Role: meta-blend of local v2 and local v3 outputs.
- Core steps:
1. Ensures required local input files exist.
2. Loads:
   `submissions/no_external_best/submission_no_leak_v2.csv`
   and `submissions/v3_no_external/diagnostics_v3_no_external.csv`.
3. Blends seed scores with configurable weights (default `0.6 * v2 + 0.4 * v3_raw`).
4. Applies season-constrained assignment to legal available seeds.
5. Writes final ensemble submission.
- External data usage: none.

## `Bid Type` and `Hint` Clarification

### `Bid Type` as feature
- Means the model directly uses `Bid Type` column during training/inference.
- This usually improves score but uses target-adjacent signal from provided files.

### `Bid Type` as hint
- Means post-processing uses non-null `Bid Type` in test as a prior for selected teams.
- This affects selection flags and assignment logic, not direct regression target fitting.

### Strict no-`Bid Type` mode
- Disable both feature and hint.
- Most conservative and usually least competitive in leaderboard score.

## Suggested Judge-Safe Submission Order

1. `submissions/v3_ensemble/submission_v3_ensemble.csv`
2. `submissions/v3_no_external/submission_v3_no_external.csv`
3. `submissions/no_external_best/submission_no_leak_v2.csv`

## Repro Commands

```bash
python3 code/build_submission_v2.py
python3 code/build_submission_v3.py
python3 code/build_submission_v3_ensemble.py
```
