# Final-Four-Analytics-Challenge-26

## Project Outline

```
final-four-analytics-challenge-26/
├── code/
│   ├── build_submission.py
│   ├── build_submission_v2.py
│   ├── build_submission_v3.py
│   └── build_submission_v3_ensemble.py
├── data/
│   └── raw/
│       ├── NCAA_Seed_Training_Set2.0.csv
│       ├── NCAA_Seed_Test_Set2.0.csv
│       ├── submission_template2.0.csv
│       └── FFAC Data Dictionary.xlsx
├── kaggle/
│   ├── final_four_end_to_end_kaggle_notebook.ipynb
│   ├── SUBMISSION_AUDIT.md
│   └── submission_folders_audit.txt
├── submissions/
│   ├── generated/
│   ├── with_bid_type/
│   ├── no_external_best/
│   ├── v3_no_external/
│   ├── v3_ensemble/
│   └── legacy/
└── docs/
```

## Usage

Strict no-leak default run:

```bash
python3 code/build_submission_v2.py
```

Run with `Bid Type` feature enabled (still local-data only):

```bash
python3 code/build_submission_v2.py --include-bid-type --exclude-team --output-dir submissions/with_bid_type
```

Improved v3 no-external pipeline:

```bash
python3 code/build_submission_v3.py
```

Improved ensemble of local v2 + local v3 models:

```bash
python3 code/build_submission_v3_ensemble.py
```

Stricter v3 mode without any `Bid Type` usage:

```bash
python3 code/build_submission_v3.py --disable-bid-type-hint --disable-bid-type-feature --output-dir submissions/v3_strict_no_bid
```

## Notes

- `submissions/legacy/` keeps historical or invalid/leaky files for reference only.
- Recommended clean no-external candidate:
  `submissions/v3_ensemble/submission_v3_ensemble.csv`

## Documentation

- Detailed submission-folder and code explanations:
  [`docs/SUBMISSIONS_AND_CODE_GUIDE.md`](docs/SUBMISSIONS_AND_CODE_GUIDE.md)
- Judge-facing submission audit:
  [`kaggle/SUBMISSION_AUDIT.md`](kaggle/SUBMISSION_AUDIT.md)
