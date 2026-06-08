# `data/` — Dataset Layout

This directory holds all dataset files used by FraudGuard. Contents are **gitignored** (see project `.gitignore`); only `.gitkeep` files and this README are tracked. Everything else must be reproducible from scripts in `src/fraudguard/data/` and `scripts/`.

## Subfolders

| Path | Purpose | Populated by |
|---|---|---|
| `raw/` | Original PaySim CSV from Kaggle, untouched. Treated as immutable input. | Phase 2.2 — kaggle downloader (`scripts/download_paysim.py`) |
| `processed/` | Cleaned, feature-engineered parquet splits (`train.parquet`, `val.parquet`, `test.parquet`) produced by the build pipeline. | Phase 2.5–2.6 — `python -m fraudguard.data.build` |
| `processed/baselines/` | Persisted baseline models (logistic regression, random forest, XGBoost) as joblib pickles. | Phase 3.8 |
| `external/` | Anything fetched at runtime (e.g., reference lookups). Empty for the PaySim-only build. | Reserved |

## Why parquet, not CSV

- Columnar; ~5–10× smaller on disk than CSV for typed numeric data
- Preserves dtypes (no `read_csv` dtype guessing on each load)
- Fast partial reads (`columns=[...]`) for EDA

## Reproducing from a clean clone

```bash
# 1. Fetch PaySim (requires kaggle CLI auth — see Phase 2.2)
python scripts/download_paysim.py

# 2. Build splits
python -m fraudguard.data.build
```

After step 2, `processed/{train,val,test}.parquet` should exist. The build script is deterministic given `config.random_seed`.

## What does **not** belong here

- Source code (lives in `src/`)
- Notebooks (live in `notebooks/`)
- Trained GNN checkpoints (live in `models/checkpoints/`, also gitignored)
