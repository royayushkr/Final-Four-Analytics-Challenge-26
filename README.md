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
├── requirements.txt
├── code/
│   ├── build_submission.py
│   ├── build_submission_v2.py
│   ├── build_submission_v3.py
│   ├── build_submission_v3_ensemble.py
│   ├── build_submission_v4.py
│   ├── build_submission_v5.py
│   ├── build_submission_v5_ensemble.py
│   ├── build_submission_v6.py
│   ├── build_submission_v3_ensemble.ipynb
│   ├── run_v5_experiments.py
│   ├── v5_config.py
│   └── v5_features.py
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
│   ├── v4_selected_seed/
│   ├── v6/
│   ├── with_bid_type/
│   ├── v5_ensemble/
│   └── v5_experiments/
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
| `submissions/v3_ensemble` | No | v3 ensemble (v2+v3) | Yes via source models | Yes via source models | mixed from source models | `submission_v3_ensemble.csv` | Historical clean ensemble candidate from the pre-v4 pipeline stage. |
| `submissions/v3_bidtype_hint` | No | v3 variant | Yes | Yes | No | v3 submission, diagnostics, OOF metrics | Variant to isolate effect of removing team feature. |
| `submissions/v3_bidtype_hint_with_team` | No | v3 variant | Yes | Yes | Yes | v3 submission, diagnostics, OOF metrics | Variant equivalent to default v3 configuration in current artifacts. |
| `submissions/v3_bid_hint_no_bid_feature` | No | v3 variant | No | Yes | No | v3 submission, diagnostics, OOF metrics | Variant to test hint-only behavior without bid feature. |
| `submissions/v3_strict` | No | v3 strict variant | No | No | Yes | v3 submission, diagnostics, OOF metrics | Conservative strict-mode benchmark. |
| `submissions/v3_strict_no_bid` | No | v3 strict variant | No | No | Yes | v3 submission, diagnostics, OOF metrics | Same strict setting as above, kept as separate run folder. |
| `submissions/v4_selected_seed` | No | v4 selected-team ranker | Yes | Yes through selection policy | Yes | `submission_v4.csv`, diagnostics, OOF metrics | Best current competition-facing candidate in this repository. |
| `submissions/v6` | No | v6 hedge candidates anchored on `v3_ensemble` | inherited from source models | Yes through selection policy | inherited from source models | `submission_v6.csv`, candidate CSVs, candidate summary | Current best hedge folder for leaderboard testing after the v5 audit. |
| `submissions/v5_experiments` | No | v5 ablation framework outputs | configurable per experiment | Yes through selection policy | configurable per experiment | summary table, per-run OOF/test predictions, feature importances, submissions | Research and audit folder for v5, not a single direct submit target. |
| `submissions/v5_ensemble` | No | v5 rank-blend ensemble | inherited from base runs | Yes through selection policy | inherited from base runs | `submission.csv`, OOF/test predictions, experiment summary | Implemented for completeness, but not recommended based on offline results. |

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

### Phase A: early baseline and mixed artifacts

- Initial outputs landed in `submissions/generated`.
- That folder became a mixed bucket with both clean and historical/transitional files.
- One separate archive artifact with external influence was preserved in `submissions/legacy`.

### Phase B: leakage-safe v2 foundation

- Built `build_submission.py` and `build_submission_v2.py`.
- Enforced train/test-only workflow from local files.
- Added season-grouped CV and automatic blending strategy selection.
- Produced stable clean runs in `with_bid_type` and `no_external_best`.

### Phase C: v3 feature and post-processing upgrade

- Introduced richer feature engineering in `build_submission_v3.py`.
- Added two-regressor seed ensemble.
- Added season-local isotonic adjustment.
- Added season-constrained seed assignment to preserve bracket structure.
- Produced `v3_no_external` and controlled ablation variants.

### Phase D: v3 variant experiments

- Tested whether performance comes from `Bid Type` as feature, `Bid Type` as hint, `Team` feature, or strict removal.
- Captured ablations in `v3_bidtype_hint`, `v3_bidtype_hint_with_team`, `v3_bid_hint_no_bid_feature`, `v3_strict`, and `v3_strict_no_bid`.
- Result pattern: removing both bid feature and hint substantially hurts selection quality.

### Phase E: v3 ensemble candidate

- Implemented `build_submission_v3_ensemble.py`.
- Blended clean v2 and v3 signals.
- Re-applied season-constrained assignment.
- Saved the ensemble candidate in `submissions/v3_ensemble/submission_v3_ensemble.csv`.

### Phase F: v4 selected-team ranker

- Added `build_submission_v4.py` to treat the task more explicitly as seeded-team ranking followed by legal seed assignment.
- Kept selection tied to provided official fields rather than external sources.
- Saved candidate outputs in `submissions/v4_selected_seed/`.
- This branch produced the strongest known public leaderboard result in this repository lineage before v5 work began.

### Phase G: v5 non-market upgrade and audit

- Added a modular experiment framework:
  `v5_config.py`, `v5_features.py`, `build_submission_v5.py`, `run_v5_experiments.py`, and `build_submission_v5_ensemble.py`.
- Standardized the primary offline evaluator around seeded-team `season_rank_rmse`.
- Added reproducible ablation outputs, fold metrics, OOF predictions, test predictions, and feature importance exports.
- Explicitly audited `goto_conversion` and `efficient_shin_conversion` and kept them out of the main pipeline because the dataset has no bookmaker-odds or implied-probability inputs.

### Phase H: v6 leaderboard hedge

- Added `build_submission_v6.py`.
- Kept `v3_ensemble` as the anchor because it remains the best verified public-leaderboard submission in this repository.
- Built conservative hedge candidates that only change a handful of seeded-team assignments where multiple newer models disagree with the original ensemble.
- Saved candidate files and change summaries in `submissions/v6/`.

## 11) V5 Upgrade and Results

### What changed from v4 to v5

v5 is primarily a framework and feature-engineering upgrade rather than a single hand-tuned script.

Key additions:

- experiment registry in `code/v5_config.py`
- reusable feature builders in `code/v5_features.py`
- unified single-run trainer in `code/build_submission_v5.py`
- batch ablation runner in `code/run_v5_experiments.py`
- rank-blend ensemble builder in `code/build_submission_v5_ensemble.py`
- dependency pinning in `requirements.txt`

New feature blocks implemented:

- `core_parsed`
- `missing_indicators`
- `season_relative`
- `conference_relative`
- `robust_numeric`
- `interaction_refinements`
- `team_conference_encoding`

New model families evaluated:

- `Ridge`
- `XGBRegressor`
- `LGBMRegressor`
- `CatBoostRegressor`
- `ExtraTreesRegressor` for the reproduced `current_v4` comparison

### Validation design

Primary offline metric:

- `season_rank_rmse`
- For each season, rank predicted seeded-team scores and assign the observed season seed set before computing RMSE.

Secondary sanity metric:

- `full_rmse_zero`
- Non-selected rows are set to zero to mimic the full Kaggle submission shape.

Stability metric:

- fold-wise standard deviation of `season_rank_rmse`

Selection policy used in v5:

- test selected rows are the rows with non-null `Bid Type`
- this matched the expected held-out tournament slot count exactly: `91` test rows

### Market-method relevance verdict

`goto_conversion` and `efficient_shin_conversion` were reviewed and intentionally excluded from the main pipeline.

Reason:

- they convert bookmaker odds or prices into implied probabilities
- this dataset has no odds, prices, overround, or market books
- our model outputs are rank/seed scores, not mutually exclusive market prices

Result:

- `v5_goto`: `SKIPPED_NO_MARKET_INPUTS`
- `v5_shin`: `SKIPPED_NO_MARKET_INPUTS`
- `v5_market_all`: `SKIPPED_NO_MARKET_INPUTS`
- `v5_mult_baseline`: `SKIPPED_NO_MARKET_INPUTS`

### V5 experiment summary

The full batch table is written to `submissions/v5_experiments/experiment_summary_table.csv`.

Best completed runs by the primary offline metric:

| Rank | Experiment | Model | Season Rank RMSE | Fold Std | Full RMSE Zero | Notes |
|---|---|---|---:|---:|---:|---|
| 1 | `v5_base_core` | `lightgbm` | 4.1075 | 1.5633 | 1.7608 | Best mean, but unstable across folds |
| 2 | `v5_base_core_missing` | `lightgbm` | 4.1075 | 1.5633 | 1.7608 | Same as above; missing flags added no gain |
| 3 | `current_v4` | `ridge` | 4.2235 | 0.8812 | 1.8126 | Best stable baseline reproduced under v5 metric |
| 4 | `v5_base_conference_relative` | `ridge` | 4.2235 | 0.8812 | 1.8126 | Effectively matched reproduced v4 |
| 5 | `v5_base_season_relative` | `ridge` | 4.2461 | 0.8909 | 1.8223 | Season-relative block helped more than raw core ridge |
| 6 | `v5_base_target_safe_encodings` | `ridge` | 4.2623 | 0.4674 | 1.8283 | Most stable ridge variant, but not best mean |
| 7 | `v5_base_full` | `ridge` | 4.2623 | 0.4674 | 1.8283 | Full non-market stack matched target-safe ridge |
| 8 | `v5_base_target_safe_encodings` | `catboost` | 4.4724 | 0.9272 | 1.9188 | Best CatBoost variant |

Important negative result:

- the richer full-stack feature set hurt `xgb` and `lightgbm` materially
- `v5_base_full/xgb` rose to `7.0914`
- `v5_base_full/lightgbm` rose to `6.8472`
- this indicates the encoding-heavy full feature block is overfitting for tree boosters on the seeded-row sample size

### Ensemble result

The saved v5 ensemble lives in `submissions/v5_ensemble/`.

Chosen base runs:

- `v5_base_core/lightgbm`
- `current_v4/ridge`
- `v5_base_target_safe_encodings/catboost`

Saved blend weights:

- `0.20` on `v5_base_core/lightgbm`
- `0.65` on `current_v4/ridge`
- `0.15` on `v5_base_target_safe_encodings/catboost`

Observed result:

- `v5_ensemble` season-rank RMSE = `4.9251`
- worse than the best single-model ridge baseline

Interpretation:

- the ensemble was implemented and logged, but it is not the recommended v5 output
- the base models are too correlated or too noisy for this rank-blend to help

### Feature relevance snapshot

`v5_base_core/lightgbm` top signals:

- `SEED_STRENGTH_HEUR`
- `CONF_GAME_SHARE`
- `Q1_GAME_SHARE`
- `CONF_PERF_DELTA`
- `WL_PCT`
- `Q_GOOD_WIN_SHARE`

`current_v4/ridge` top signals:

- `NET Rank_LOG`
- `PrevNET_LOG`
- `PrevNET_SEASON_PCT`
- `NET Rank_SEASON_PCT`
- conference indicators
- `Bid Type` categories

`v5_base_target_safe_encodings/ridge` top signals:

- `PrevNET_CONF_PCT`
- `NET Rank_CONF_PCT`
- `PrevNET_LOG`
- `NET Rank_LOG`
- `Team_SEED_TE`
- `Conference_SEED_TE`

### Diagnosis after implementation

Why v4 remains hard to beat:

- the seeded training sample is only `249` rows, so high-capacity tree models overfit quickly
- `Bid Type` already resolves much of the selection problem, leaving v5 to win only on fine-grained seed ordering
- rank-assignment post-processing matters as much as raw regression loss here
- stable linear structure is outperforming more complex models on the current feature blocks

What v5 accomplished anyway:

- standardized evaluation
- modularized feature engineering
- added fold-safe target encoding
- made market-method irrelevance explicit and auditable
- created a reproducible ablation framework for v6

## 12) V6 Hedge Strategy

### Why v6 exists

v5 improved the research framework, but it did not beat the known public-leaderboard result from `v3_ensemble`.

That changed the design goal for v6:

- do not replace `v3_ensemble` wholesale
- keep the proven baseline ordering as the anchor
- make only a few defensible swaps where other local models agree against the baseline

### v6 methodology

Inputs used:

- v2 final prediction from `submissions/no_external_best/submission_no_leak_v2.csv`
- v3 raw and local-isotonic diagnostics from `submissions/v3_no_external/diagnostics_v3_no_external.csv`
- v5 single-model score outputs:
  - `current_v4/ridge`
  - `v5_base_core/lightgbm`
  - `v5_base_full/ridge`

Anchor score:

- `base_score = 0.60 * v2_overall_seed + 0.40 * v3_seed_pred_raw`
- this matches the structure that produced `v3_ensemble`

Additional hedge signals:

- `v3_order_rank`
- `v5_cur_ridge_rank`
- `v5_lgbm_rank`
- `v5_ridge_rank`

### Saved v6 candidates

All candidates are written under `submissions/v6/`.

Primary file:

- `submissions/v6/submission_v6.csv`

Candidate table:

| Candidate | Weights | Rows Changed vs `v3_ensemble` | Intended Use |
|---|---|---:|---|
| `v6_primary` | `0.65 base + 0.175 v5_ridge_rank + 0.175 v3_order_rank` | 2 | Main submit hedge |
| `v6_conservative_low_seed` | `0.75 base + 0.25 v5_cur_ridge_rank` | 2 | Very low-risk alternate |
| `v6_aggressive_ridge` | `0.75 base + 0.25 v5_ridge_rank` | 4 | Higher-variance alternate |
| `v6_lgbm_low_seed` | `0.75 base + 0.25 v5_lgbm_rank` | 2 | Low-impact LightGBM alternate |

Actual `v6_primary` changes versus `v3_ensemble`:

- `2021-22-Davidson`: `47 -> 43`
- `2021-22-NotreDame`: `43 -> 47`

Why those two rows were chosen for the primary file:

- they are supported by `v3` final constrained assignment
- they are supported by `v4`
- they are supported by the v5 single-model variants
- the hedge changes only two rows, so it preserves most of the best known baseline

## 13) Reproducible Runbook

### Environment

Install all dependencies used by the current repository:

```bash
python3 -m pip install -r requirements.txt
```

### Run v2 default

```bash
python3 code/build_submission_v2.py
```

### Run v3 default

```bash
python3 code/build_submission_v3.py
```

### Run v3 ensemble

```bash
python3 code/build_submission_v3_ensemble.py
```

### Run v4

```bash
python3 code/build_submission_v4.py
```

### Run one v5 experiment

Example: full non-market ridge run

```bash
python3 code/build_submission_v5.py \
  --experiment v5_base_full \
  --model-family ridge
```

### Run the full v5 ablation matrix

```bash
python3 code/run_v5_experiments.py
```

Main outputs:

- `submissions/v5_experiments/experiment_summary_table.csv`
- per-run `fold_metrics.csv`
- per-run `oof_predictions.csv`
- per-run `test_predictions.csv`
- per-run `feature_importance.csv`
- per-run `submission.csv`

### Run the v5 ensemble

```bash
python3 code/build_submission_v5_ensemble.py
```

Main outputs:

- `submissions/v5_ensemble/oof_predictions.csv`
- `submissions/v5_ensemble/test_predictions.csv`
- `submissions/v5_ensemble/submission.csv`
- `submissions/v5_ensemble/experiment_summary.json`

### Run v6 hedge candidates

```bash
python3 code/build_submission_v6.py
```

Main outputs:

- `submissions/v6/submission_v6.csv`
- `submissions/v6/v6_primary.csv`
- `submissions/v6/v6_conservative_low_seed.csv`
- `submissions/v6/v6_aggressive_ridge.csv`
- `submissions/v6/v6_lgbm_low_seed.csv`
- `submissions/v6/v6_candidate_summary.csv`

## 14) Recommended Competition Submission Paths

### Current best competition-facing choice

- `submissions/v3_ensemble/submission_v3_ensemble.csv`

Reason:

- this is still the best verified public-leaderboard file in the repository
- v6 is a hedge layer designed to test a few targeted changes, not a proven replacement yet

### Best new hedge candidate

- `submissions/v6/submission_v6.csv`

Reason:

- only changes two rows from `v3_ensemble`
- those rows are supported by multiple local models
- this is the cleanest serious attempt to improve on the current best public score without discarding the winning baseline structure

### Best v5 research artifacts

Best mean offline metric:

- `submissions/v5_experiments/v5_base_core/lightgbm/submission.csv`

Most stable v5 ridge variant:

- `submissions/v5_experiments/v5_base_full/ridge/submission.csv`

Implemented but not recommended:

- `submissions/v5_ensemble/submission.csv`

Historical clean fallback candidates:

- `submissions/v6/v6_conservative_low_seed.csv`
- `submissions/v6/v6_aggressive_ridge.csv`
- `submissions/v6/v6_lgbm_low_seed.csv`
- `submissions/v3_ensemble/submission_v3_ensemble.csv`
- `submissions/v3_no_external/submission_v3_no_external.csv`
- `submissions/no_external_best/submission_no_leak_v2.csv`

## 15) Next Steps for V7

Priority directions:

- compute season-relative features against the full season pool, then train only on seeded rows
- replace simple rank blending with a fold-safe meta-model trained on OOF base predictions
- tune post-processing directly around season-consistent seed assignment, not only regression loss
- prune or regularize target encodings for tree models, since the full-stack tree variants clearly overfit
- test monotonic or pairwise ranking objectives rather than plain RMSE regressors

## 16) FAQ

### Is `build_submission_v3_ensemble.py` using internet data?

No. It only reads local files and, if missing, runs local scripts that also read only `data/raw/*`.

### Is v5 using `goto_conversion` or any market conversion package?

No. Those methods were reviewed and intentionally skipped because the repository has no odds-like inputs.

### Why keep `legacy` if it is not submission-safe?

For audit trail and reproducibility history.

### Why do the complex v5 tree models get worse?

Because the seeded-label sample is small, and the richer feature stack plus encodings overfit more easily than the linear ridge baseline.

## 17) Private 2026 Final Pipeline

This repository now also contains a private-task-specific final pipeline for the `2025-26` season snapshot dated `2026-03-15`.

Primary entry point:

- `code/build_submission_v7_private.py`

Supporting modules:

- `code/private_features.py`
- `code/private_validation.py`

### What changed from Kaggle-era versions

The private v7 flow intentionally drops the old competition assumptions:

- does not use `Bid Type` in modeling or post-processing
- does not use the historical Kaggle test file for fitting
- does not assume the test file already identifies selected teams
- does not assign only season-residual seeds from prior years
- writes one final `365`-row submission for the `2026-03-15` snapshot

### Final modeling approach

The private final task is treated as a constrained ranking problem:

1. Train only on official train rows with known `Overall Seed`.
2. Predict a continuous seed-strength score from committee-style resume features.
3. Score all `365` teams in the final `2026-03-15` snapshot.
4. Select the top `68` teams by model score.
5. Assign exact unique overall seeds `1..68`.
6. Assign `0` to all remaining teams.

### New feature design

The private feature stack includes:

- strict parsing of all record-style fields into wins, losses, games, percentages, and margins
- committee-style resume features such as `q12_wins`, `q34_losses`, `quality_win_score`, `bad_loss_penalty`, and `resume_balance`
- NET movement features derived from `PrevNET` and current `NET Rank`
- SOS context and interaction features
- season-relative percentile and z-score features
- conference-context features after normalizing `American` / `The American`
- fold-safe clipping and missing-value indicators

Raw `Team` identity and `Bid Type` are excluded from the final private model.

### Validation framework

The private final pipeline uses:

- expanding-window folds:
  - `2020-21..2021-22 -> 2022-23`
  - `2020-21..2022-23 -> 2023-24`
  - `2020-21..2023-24 -> 2024-25`
- weighted fold emphasis on recency: `0.2 / 0.3 / 0.5`
- 2026 weekly stability checks across all provided `2026` snapshots

Candidate models tested in v7:

- `ridge_only_context`
- `catboost_only_context`
- `ensemble_base`
- `ensemble_context`
- `ensemble_context_band`
- `ensemble_context_lgbm`
- `ensemble_context_pu`

### Chosen final candidate

The selected final candidate from the private v7 run is:

- `ridge_only_context`

Why it won:

- best weighted composite loss on the expanding-window backtests
- strongest recent-fold recall among the top candidates
- weekly 2026 stability that was competitive with the more complex variants
- lower implementation risk than the ensemble variants

### Final private artifacts

Diagnostics:

- `artifacts/final_20260315/rolling_origin_metrics.csv`
- `artifacts/final_20260315/weekly_stability_metrics.csv`
- `artifacts/final_20260315/model_comparison.json`
- `artifacts/final_20260315/feature_importance.csv`
- `artifacts/final_20260315/final_selection_audit.csv`

Final submission:

- `submissions/final/submission_2026_20260315.csv`

The final submission file satisfies these checks:

- exactly `365` rows
- columns exactly `RecordID,Overall Seed`
- exactly `68` non-zero predictions
- non-zero seeds are exactly the integers `1..68`
- remaining `297` teams are assigned `0`
