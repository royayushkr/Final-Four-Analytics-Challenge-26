# Submission Audit and Method Differences

This document explains each submission folder, whether it is external-data free, and how `Bid Type` / `Hint` settings change model behavior.

## Data Governance Summary

- **External data used**: `submissions/legacy` only.
- **Mixed history (not fully clean)**: `submissions/generated` (contains historical files from multiple phases, including one externally influenced artifact).
- **Raw competition files only**: all other `submissions/*` folders listed below.

---

## What `Bid Type` and `Hint` Mean

### 1) `Bid Type` as a **feature**
- The model directly uses the provided column `Bid Type` (`AQ` / `AL` / null) as an input variable.
- This can improve performance because `Bid Type` is strongly related to tournament inclusion.
- This is still **local-data only** when it comes from the provided train/test CSVs.

### 2) `Bid Type` as a **hint**
- Post-processing uses non-null `Bid Type` in test rows to guide which teams are marked as selected.
- It does not directly change regression targets; it guides selection flags and season-level assignment.
- This is also **local-data only** (from official provided files), but it is a stronger prior.

### 3) No `Bid Type` usage (strict mode)
- Neither model features nor selection logic use `Bid Type`.
- Purely performance-stat based.
- Most conservative approach, typically lower leaderboard score.

---

## Model Families

### `v2` family
- Two-stage modeling:
  - classifier predicts selected vs non-selected
  - regressor predicts seed
- Simpler feature space and lighter post-processing.

### `v3` family
- Stronger feature engineering (rank transforms, weighted quadrant signals, deltas).
- Ensemble seed scoring (multiple regressors).
- Season-constrained seed assignment to keep predictions structurally consistent.

### `v3_ensemble`
- Blends predictions from best local `v2` and local `v3`.
- Uses season-constrained assignment after blending.
- Designed as strongest **no-external-data** candidate.

---

## Folder-by-Folder Explanation

| Folder | External Data? | Core Method | `Bid Type` Feature | `Bid Type` Hint | Notes |
|---|---|---|---|---|---|
| `submissions/legacy` | **Yes** | Older externally seeded mapping artifact | N/A | N/A | Keep for archive only. Do not submit for clean judging. |
| `submissions/generated` | Mixed | Historical mixed outputs | Mixed | Mixed | Contains both clean and earlier artifacts; not ideal for judge handoff. |
| `submissions/with_bid_type` | No | `v2` local model | Yes | Implicit via model | Raw files only. |
| `submissions/no_external_best` | No | Best `v2` local run | Yes | Implicit via model | Strong local baseline. |
| `submissions/v3_no_external` | No | `v3` default | Yes | Yes | Main v3 local output. |
| `submissions/v3_ensemble` | No | Blend of local `v2` + local `v3` | Yes (via sources) | Yes (via sources) | Recommended clean submission candidate. |
| `submissions/v3_bidtype_hint` | No | `v3` variant | Yes | Yes | Variant run. |
| `submissions/v3_bidtype_hint_with_team` | No | `v3` variant | Yes | Yes | Variant run with team feature. |
| `submissions/v3_bid_hint_no_bid_feature` | No | `v3` variant | No | Yes | Uses hint only, not feature. |
| `submissions/v3_strict` | No | `v3` strict variant | No | No | Conservative strict mode. |
| `submissions/v3_strict_no_bid` | No | `v3` strict variant | No | No | Conservative strict mode variant. |

---

## Judge-Friendly Recommendation

If judges require **strictly no external data**, submit from:

- `submissions/v3_ensemble/submission_v3_ensemble.csv`

If judges require **strict no `Bid Type` usage**, submit from:

- `submissions/v3_strict_no_bid/submission_v3_no_external.csv`

