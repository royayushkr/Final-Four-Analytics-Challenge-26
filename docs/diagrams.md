# Final V8 Presentation Diagrams

This file isolates the final production visuals from the full project README so they can be reused in slides, reports, and video walkthroughs.

## 1) End-to-End Final Pipeline

![End-to-End Final Pipeline](assets/final_pipeline_overview.svg)

```mermaid
flowchart TD
    A["Historical Team Metrics 2020-21 to 2024-25"] --> B["Attach Historical True Seeds and Historical AQ/AL Labels"]
    B --> C["Feature Engineering"]
    C --> D["Bid-Class Model: LogisticRegression"]
    C --> E["Seed-Strength Model: RidgeCV"]
    D --> F["Predicted AQ, AL, NONE Probabilities"]
    E --> G["Seed Strength Score"]
    F --> H["Combined Final Score"]
    G --> H
    H --> I["Conservative AQ/AL Post-Processing"]
    I --> J["Select Final 68 Teams"]
    J --> K["Assign Unique Seeds 1 to 68"]
    K --> L["Write Final Submission CSV"]
```

## 2) Data Boundaries and Leakage Control

![Data Boundaries and Leakage Control](assets/data_boundaries.svg)

```mermaid
flowchart LR
    subgraph HistoricalData["Historical supervised universe"]
        T1["Training_Set2.0"]
        T2["Historical Test_Set2.0"]
        T3["historical_true_seeds_2021_2025.csv"]
    end

    subgraph Diagnostics["Diagnostics only"]
        W1["2026-02-06 to 2026-03-08 weekly snapshots"]
    end

    subgraph FinalInference["Final inference only"]
        F1["2026-03-15 snapshot"]
    end

    T1 --> M["Build labeled historical universe"]
    T2 --> M
    T3 --> M
    M --> V["Rolling-origin validation and model selection"]
    V --> R["Refit chosen model on all historical seasons"]
    R --> P["Predict 2026-03-15 only"]
    W1 --> S["Stability checks"]
    S --> V
    P --> Q["Quota-aware field selection and seed assignment"]
    Q --> O["submission_2026_20260315.csv"]
```

## 3) AQ/AL Post-Processing Logic

![AQ/AL Post-Processing Logic](assets/aq_al_post_processing.svg)

```mermaid
flowchart TD
    A["Score all 365 teams"] --> B["Rank by final combined score"]
    B --> C["Lock top 24 teams by overall score"]
    C --> D["Assign provisional AQ/AL labels from bid-class probabilities"]
    D --> E["Apply AQ/AL quota mainly to the remaining field"]
    E --> F["Reach exactly 31 AQ and 37 AL"]
    F --> G["Sort final 68 by final score"]
    G --> H["Assign seeds 1 to 68"]
    H --> I["Assign 0 to remaining 297 teams"]
```

## 4) Suggested Presentation Order

1. Start with the end-to-end pipeline to establish the modeling structure.
2. Use the data-boundary diagram to explain leakage controls and why weekly snapshots are diagnostics only.
3. Finish with the AQ/AL diagram to explain how the final 68-team field is converted into realistic seeds.
