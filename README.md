# Final-Four-Analytics-Challenge-26

This is the single source of documentation for this repository.
It combines project structure, model design, submission catalog, leakage policy, and experiment history so contributors can work without reading source code first.

## 1) Competition Objective

The repository targets the NCAA Final Four Analytics Challenge seed prediction task.

- Prediction target: `Overall Seed`
- Evaluation metric: RMSE between predicted and observed seeds
- Submission format: `RecordID,Overall Seed`
- Tournament framing in this project: non-selected teams are predicted as `0`, selected teams are assigned seeds in `[1, 68]`

## 2) Repository Structure

```text
final-four-analytics-challenge-26/
├── code/
│   ├── build_submission.py
│   ├── build_submission_v2.py
│   ├── build_submission_v3.py
│   ├── build_submission_v3_ensemble.py
│   └── build_submission_v3_ensemble.ipynb
├── data/
│   └── raw/
│       ├── NCAA_Seed_Training_Set2.0.csv
│       ├── NCAA_Seed_Test_Set2.0.csv
│       ├── submission_template2.0.csv
│       └── FFAC Data Dictionary.xlsx
├── kaggle/
│   └── final_four_end_to_end_kaggle_notebook.ipynb
├── submissions/
│   ├── generated/
│   ├── legacy/
│   ├── no_external_best/
│   ├── v3_bid_hint_no_bid_feature/
│   ├── v3_bidtype_hint/
│   ├── v3_bidtype_hint_with_team/
│   ├── v3_ensemble/
│   ├── v3_no_external/
│   ├── v3_strict/
│   ├── v3_strict_no_bid/
│   └── with_bid_type/
└── README.md
```

## 3) Data Inputs and Scope

Official files used by all clean pipelines:

- `data/raw/NCAA_Seed_Training_Set2.0.csv`
- `data/raw/NCAA_Seed_Test_Set2.0.csv`
- `data/raw/submission_template2.0.csv` (format reference)
- `data/raw/FFAC Data Dictionary.xlsx` (schema support)

No script in `code/` requires internet or external downloads.

## 4) Leakage and Compliance Policy

This project contains historical outputs from multiple development phases. Some artifacts are archive-only.

### Clean-policy definition used here

An artifact is considered competition-guideline compliant in this repository when:

- it is generated only from `data/raw/*`,
- it does not merge external web/internet mappings,
- it does not use future-season labels from test rows.

### Folder compliance classification

- `submissions/legacy`: contains archived externally influenced output. Keep for history only.
- `submissions/generated`: mixed-history folder from early experimentation. Contains both clean and non-clean historical artifacts.
- all other `submissions/*` folders: generated from local project files only.

## 5) End-to-End Modeling Workflow (High Level)

All model versions follow a shared two-part logic:

1. Predict tournament selection tendency (explicitly or implicitly).
2. Predict a seed score and convert it to legal season-consistent seed assignments.

Shared technical patterns:

- Parse and repair malformed W-L strings such as `8-Sep`, `Apr-00`.
- Engineer W/L/PCT-style features from record columns.
- Use season-aware validation (GroupKFold grouped by `Season`).
- Clip seed predictions to the valid range and write Kaggle-ready CSVs.

## 6) Code Walkthrough (No Code Reading Required)

### `code/build_submission.py` (Core v2 pipeline)

Purpose:

- Base leakage-safe trainer/inference pipeline for v2 family runs.

How it works:

1. Loads train and test CSV files.
2. Parses record fields (`WL`, `Conf.Record`, `Quadrant*`, etc.) into wins/losses/games/win%.
3. Builds additional features:
   `NET_Improvement`, `OppNetDiff`, `SOS_Diff`, plus parsed record derivatives.
4. Preprocesses with:
   numeric median imputation and categorical one-hot encoding.
5. Trains:
   `XGBClassifier` for selected-vs-non-selected and `XGBRegressor` for seed values.
6. Runs GroupKFold CV by season and evaluates candidate blending strategies:
   `hard`, `soft`, `soft-threshold`.
7. Chooses best strategy if `--prediction-strategy auto`.
8. Fits on full train and writes outputs.

Outputs:

- `submission_seed_only_no_leak.csv`
- `submission_no_leak_v2.csv`
- `test_selection_probabilities_no_leak.csv`
- `cv_metrics_no_leak.json`

Main toggles:

- `--include-bid-type`
- `--exclude-team`
- `--prediction-strategy`
- `--selection-threshold`

### `code/build_submission_v2.py` (v2 wrapper runner)

Purpose:

- Reproducible launcher around `build_submission.py`.

How it works:

1. Resolves default project paths.
2. Forces `--prediction-strategy auto`.
3. Passes user toggles (`--include-bid-type`, `--exclude-team`) through.

Why it exists:

- Easy repeatable runs into named output folders without manually rebuilding long CLI commands.

### `code/build_submission_v3.py` (Stronger no-external-data pipeline)

Purpose:

- Higher-capacity v3 model with richer feature engineering and constrained post-processing.

How it works:

1. Extends v2 feature set with:
   rank inverse/log transforms, margin features, weighted quadrant scores, game-share and performance-delta features.
2. Trains selection model:
   `XGBClassifier` for inclusion probability.
3. Trains seed score ensemble:
   two `XGBRegressor` models blended as `0.55 * reg1 + 0.45 * reg2`.
4. Learns season-local isotonic models from seeded training rows:
   maps `NET Rank -> Seed` shape per season.
5. Creates combined season ordering score from:
   raw seed prediction, local isotonic prediction, and NET rank.
6. Determines selected flags per season using target counts with optional `Bid Type` hint.
7. Assigns legal seeds from the season-specific available seed set
   (seeds missing in train for that season).
8. Writes submission + diagnostics + OOF metrics.

Outputs:

- `submission_v3_no_external.csv`
- `diagnostics_v3_no_external.csv`
- `oof_metrics_v3_no_external.json`

Main toggles:

- `--disable-bid-type-hint`
- `--disable-bid-type-feature`
- `--exclude-team-feature`
- `--tournament-size`

### `code/build_submission_v3_ensemble.py` (v2+v3 blend)

Purpose:

- Blend strongest local v2 and v3 signals into one final candidate.

How it works:

1. Ensures required local inputs exist:
   - v2 file: `submissions/no_external_best/submission_no_leak_v2.csv`
   - v3 diagnostics: `submissions/v3_no_external/diagnostics_v3_no_external.csv`
2. Auto-runs missing prerequisites by calling local scripts only:
   `build_submission_v2.py` and `build_submission_v3.py`.
3. Blends score:
   `score = w2 * v2_overall_seed + w3 * v3_seed_pred_raw`.
4. Uses test `Bid Type` non-null rows as selected teams.
5. Reassigns season-consistent available seeds.
6. Writes final ensemble submission.

Outputs:

- `submissions/v3_ensemble/submission_v3_ensemble.csv`

External-data status:

- No internet usage.
- No dependency on legacy external artifacts.
- Uses only local files and locally generated clean predictions.

## 7) `Bid Type`, `Hint`, and Model Behavior

These terms are intentionally separated in this project:

### `Bid Type` as feature

- The column `Bid Type` is passed into the ML feature matrix.
- Affects fitted model parameters directly.
- Usually improves selection accuracy because this field is highly informative.

### `Bid Type` as hint

- Used only in post-processing selection logic.
- Example in v3:
  if `Bid Type` is non-null, those rows get priority in selected-team allocation for the season.
- Does not directly train regression coefficients on that step.

### Related model dimensions

- `Team` feature included/excluded:
  affects identity memorization vs generalization.
- strict mode:
  disables both `Bid Type` feature and `Bid Type` hint.
- ensemble mode:
  combines already-generated v2 and v3 signals, then performs constrained assignment.

## 8) Submission Folder Catalog (Comprehensive)

All submission folders and their intent are documented below.

| Folder | External Data Used | Model Family | Bid Type Feature | Bid Type Hint | Team Feature | Typical Files | Recommended Usage |
|---|---|---|---|---|---|---|---|
| `submissions/legacy` | Yes (historical archive) | older pre-clean artifact | mixed/unknown | mixed/unknown | mixed/unknown | `submission_v2_leaked.csv` | Archive only. Do not use for clean competition submission. |
| `submissions/generated` | Mixed history | early v2 and transitional outputs | mixed | mixed | mixed | `submission_finalized.csv`, `submission_no_leak_v2.csv`, diagnostics, metrics | Development traceability only, not primary handoff folder. |
| `submissions/with_bid_type` | No | v2 | Yes | implicit through model behavior | Yes | v2 submission, seed-only file, probabilities, CV metrics | Clean v2 run with team + bid type enabled. |
| `submissions/no_external_best` | No | v2 | Yes | implicit through model behavior | No | v2 submission, seed-only file, probabilities, CV metrics | Best v2 clean baseline. |
| `submissions/v3_no_external` | No | v3 | Yes | Yes | Yes | v3 submission, diagnostics, OOF metrics | Primary single-model v3 output. |
| `submissions/v3_ensemble` | No | v3 ensemble (v2+v3) | Yes via source models | Yes via source models | mixed from source models | `submission_v3_ensemble.csv` | Current preferred clean final candidate. |
| `submissions/v3_bidtype_hint` | No | v3 variant | Yes | Yes | No | v3 submission, diagnostics, OOF metrics | Variant to isolate effect of removing team feature. |
| `submissions/v3_bidtype_hint_with_team` | No | v3 variant | Yes | Yes | Yes | v3 submission, diagnostics, OOF metrics | Variant equivalent to default v3 configuration in current artifacts. |
| `submissions/v3_bid_hint_no_bid_feature` | No | v3 variant | No | Yes | No | v3 submission, diagnostics, OOF metrics | Variant to test hint-only behavior without bid feature. |
| `submissions/v3_strict` | No | v3 strict variant | No | No | Yes | v3 submission, diagnostics, OOF metrics | Conservative strict-mode benchmark. |
| `submissions/v3_strict_no_bid` | No | v3 strict variant | No | No | Yes | v3 submission, diagnostics, OOF metrics | Same strict setting as above, kept as separate run folder. |

## 9) Local Metrics Snapshot from Saved Artifacts

These values are read from JSON files already present in `submissions/*`.
They are useful for internal comparison and do not guarantee public leaderboard ranking.

### v2-family metrics (`cv_metrics_no_leak.json`)

| Folder | Selection Accuracy | Seed RMSE (Selected) | Final RMSE (All Rows) | Strategy |
|---|---|---|---|---|
| `generated` | 0.9387 | 5.7124 | 12.2746 | `soft @ 0.5` |
| `with_bid_type` | 1.0000 | 5.7239 | 2.4555 | `hard @ 0.3` |
| `no_external_best` | 1.0000 | 5.4881 | 2.3544 | `hard @ 0.3` |

### v3-family metrics (`oof_metrics_v3_no_external.json`)

| Folder | OOF Selection Acc (@0.5) | OOF Seed RMSE (Selected) | OOF RMSE (Zero-threshold @0.5) | Bid Type Feature | Bid Type Hint |
|---|---|---|---|---|---|
| `v3_no_external` | 1.0000 | 5.6230 | 2.4122 | Yes | Yes |
| `v3_bidtype_hint_with_team` | 1.0000 | 5.6230 | 2.4122 | Yes | Yes |
| `v3_bidtype_hint` | 1.0000 | 5.5662 | 2.3879 | Yes | Yes |
| `v3_bid_hint_no_bid_feature` | 0.9342 | 5.5657 | 14.2479 | No | Yes |
| `v3_strict` | 0.9372 | 5.6667 | 13.9081 | No | No |
| `v3_strict_no_bid` | 0.9372 | 5.6667 | 13.9081 | No | No |

## 10) Version Journey and Why Each Version Exists

This section documents how the repository evolved.

### Phase A: Early baseline and mixed artifacts

- Initial outputs landed in `submissions/generated`.
- That folder became a mixed bucket with both clean and historical/transitional files.
- One separate archive artifact with external influence was preserved in `submissions/legacy`.

### Phase B: Leakage-safe v2 foundation

- Built `build_submission.py` and `build_submission_v2.py`.
- Enforced train/test-only workflow from local files.
- Added season-grouped CV and automatic blending strategy selection.
- Produced stable clean runs in:
  `with_bid_type` and `no_external_best`.

### Phase C: v3 feature and post-processing upgrade

- Introduced richer feature engineering in `build_submission_v3.py`.
- Added two-regressor seed ensemble.
- Added season-local isotonic adjustment.
- Added season-constrained seed assignment to preserve bracket structure.
- Produced `v3_no_external` and controlled ablation variants.

### Phase D: v3 variant experiments

- Tested whether performance comes from:
  `Bid Type` as feature, `Bid Type` as hint, `Team` feature, or strict removal.
- Captured ablations in:
  `v3_bidtype_hint`, `v3_bidtype_hint_with_team`, `v3_bid_hint_no_bid_feature`, `v3_strict`, `v3_strict_no_bid`.
- Result pattern:
  removing both bid feature and hint substantially hurts selection quality.

### Phase E: final ensemble candidate

- Implemented `build_submission_v3_ensemble.py`.
- Blended clean v2 and v3 signals.
- Re-applied season-constrained assignment.
- Saved final candidate in:
  `submissions/v3_ensemble/submission_v3_ensemble.csv`.

## 11) Reproducible Runbook

### Environment

Install Python dependencies used by scripts:

```bash
python3 -m pip install pandas numpy scikit-learn xgboost
```

### Run v2 default

```bash
python3 code/build_submission_v2.py
```

### Run v2 with explicit Bid Type and no Team feature

```bash
python3 code/build_submission_v2.py \
  --include-bid-type \
  --exclude-team \
  --output-dir submissions/no_external_best
```

### Run v3 default

```bash
python3 code/build_submission_v3.py
```

### Run v3 strict (no Bid Type feature and no hint)

```bash
python3 code/build_submission_v3.py \
  --disable-bid-type-feature \
  --disable-bid-type-hint \
  --output-dir submissions/v3_strict_no_bid
```

### Run ensemble final

```bash
python3 code/build_submission_v3_ensemble.py
```

## 12) Recommended Competition Submission Paths

Primary clean candidate:

- `submissions/v3_ensemble/submission_v3_ensemble.csv`

Single-model fallback:

- `submissions/v3_no_external/submission_v3_no_external.csv`

v2 fallback:

- `submissions/no_external_best/submission_no_leak_v2.csv`

Strict conservative option (no Bid Type feature/hint):

- `submissions/v3_strict_no_bid/submission_v3_no_external.csv`

## 13) FAQ

### Is `build_submission_v3_ensemble.py` using internet data?

No. It only reads local files and, if missing, runs local scripts that also read only `data/raw/*`.

### Why keep `legacy` if it is not submission-safe?

For audit trail and reproducibility history.

### Why does strict mode usually score worse?

Because it removes strong provided signals (`Bid Type`) that are predictive of selection.
