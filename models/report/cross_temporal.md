# Cross-Temporal Robustness Check

Production margin-Elo evaluated on each held-out year. The 2025 row is the
canonical number from `models/precision_improved.json`; the other rows are
computed by `python -m src.models.cross_temporal_check` on the same production
Elo signal from `DB/features/match_features.csv`.

| Year | n | logloss | AUC | Brier | accuracy | source |
|---:|---:|---:|---:|---:|---:|---|
| 2022 | 118 | 0.6313 | 0.6694 | 0.2217 | 0.6356 | cross_temporal_check |
| 2023 | 162 | 0.6629 | 0.6246 | 0.2345 | 0.5802 | cross_temporal_check |
| 2024 | 222 | 0.5765 | 0.7561 | 0.1967 | 0.6937 | cross_temporal_check |
| 2025 | 314 | 0.5677 | 0.7624 | 0.1930 | 0.7038 | precision_improved.json |
