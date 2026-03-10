# Final Four Analytics Challenge 2026

## Project Outline

```
final-four-analytics-challenge-26/
├── code/
│   ├── build_submission.py
│   └── build_submission_v2.py
├── data/
│   └── raw/
│       ├── NCAA_Seed_Training_Set2.0.csv
│       ├── NCAA_Seed_Test_Set2.0.csv
│       ├── submission_template2.0.csv
│       └── FFAC Data Dictionary.xlsx
├── submissions/
│   ├── generated/
│   ├── with_bid_type/
│   ├── no_external_best/
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

Improved v3 no-external pipeline (recommended):

```bash
python3 code/build_submission_v3.py
```

Improved ensemble of v2 + v3 local models (best candidate):

```bash
python3 code/build_submission_v3_ensemble.py
```

Stricter v3 mode without any `Bid Type` usage:

```bash
python3 code/build_submission_v3.py --disable-bid-type-hint --disable-bid-type-feature --output-dir submissions/v3_strict_no_bid
```

## Notes

- `submissions/legacy/` keeps historical or invalid/leaky files for reference only.
- Current recommended file for strict no-leak submission is:
  `submissions/generated/submission_no_leak_v2.csv`
- Current recommended improved non-external file is:
  `submissions/v3_ensemble/submission_v3_ensemble.csv`
