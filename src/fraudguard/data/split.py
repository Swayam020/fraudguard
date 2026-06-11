"""Split feature-engineered PaySim into train/val/test parquet files.

Reads:
    data/processed/paysim_200k_features.parquet

Writes:
    data/processed/train.parquet  (70%, earliest in time)
    data/processed/val.parquet    (15%, middle)
    data/processed/test.parquet   (15%, latest in time)

Strategy:
    1. Sort by step (already sorted from subsample.py, but we re-sort defensively).
    2. Find row cutoffs at 70% and 85% of total length.
    3. Slice the DataFrame -- contiguous temporal blocks, no shuffling.
    4. Report fraud counts and step ranges per split for inspection.

Why temporal split (NOT random):
    Fraud detection is a forecasting problem. A random split lets the test set
    contain transactions from time periods seen during training -- temporal
    leakage that inflates metrics. Sorting by step and cutting by row position
    guarantees train < val < test in time order, matching how the model would
    run in production.

Why not stratified:
    Stratification (forcing equal fraud rates across splits) would require
    shuffling within the temporal blocks, which reintroduces leakage. We
    instead report the natural fraud rate in each split and accept that
    production reality has uneven fraud distributions over time.

Known limitation:
    Total subsample fraud count is small (~147). At 15% test, that is ~22
    fraud rows -- workable for PR-AUC but limits test-set statistical power.
    If Phase 6 metric variance is high we will revisit subsample size.

Usage:
    python -m fraudguard.data.split
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
INPUT_PARQUET = REPO_ROOT / "data" / "processed" / "paysim_200k_features.parquet"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# Test ratio is implicitly 1 - TRAIN - VAL = 0.15


def log(msg: str) -> None:
    print(f"[split] {msg}", flush=True)


def describe_split(name: str, df: pd.DataFrame) -> None:
    fraud_count = int(df["isFraud"].sum())
    fraud_pct = 100.0 * fraud_count / len(df)
    step_min = int(df["step"].min())
    step_max = int(df["step"].max())
    log(
        f"{name:5s}: {len(df):>7,} rows | "
        f"steps {step_min:>3d}..{step_max:<3d} | "
        f"fraud {fraud_count:>4d} ({fraud_pct:.4f}%)"
    )


def main() -> int:
    if not INPUT_PARQUET.exists():
        log(f"missing {INPUT_PARQUET}. Run features.py first.")
        return 1

    log(f"reading {INPUT_PARQUET.name}...")
    df = pd.read_parquet(INPUT_PARQUET)
    log(f"loaded {len(df):,} rows")

    # Defensive re-sort: features.py preserves order from subsample.py (which
    # already sorted by step), but explicit beats implicit when correctness matters.
    df = df.sort_values("step", kind="mergesort").reset_index(drop=True)

    n_total = len(df)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    # Test gets the remainder so totals always sum to n_total exactly.

    train = df.iloc[:n_train]
    val = df.iloc[n_train : n_train + n_val]
    test = df.iloc[n_train + n_val :]

    log(f"splits ({TRAIN_RATIO:.0%} / {VAL_RATIO:.0%} / {1 - TRAIN_RATIO - VAL_RATIO:.0%}):")
    describe_split("train", train)
    describe_split("val", val)
    describe_split("test", test)

    # Sanity: splits should be disjoint and cover everything.
    assert len(train) + len(val) + len(test) == n_total, "split row counts dont sum"
    assert train["step"].max() <= val["step"].min(), "train/val temporal overlap"
    assert val["step"].max() <= test["step"].min(), "val/test temporal overlap"
    log("temporal-ordering assertions passed.")

    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        out = PROCESSED_DIR / f"{name}.parquet"
        split_df.to_parquet(out, engine="pyarrow", compression="snappy", index=False)
        size_mb = out.stat().st_size / (1024**2)
        log(f"wrote {out.name} ({size_mb:.2f} MB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
