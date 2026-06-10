"""Build a deterministic 200K-row temporal subsample of PaySim.

Reads:
    data/raw/paysim.csv (full 6.36M-row Kaggle CSV; ~470 MB)

Writes:
    data/processed/paysim_200k.parquet (~30-40 MB)

Strategy:
    1. Read the full CSV.
    2. Sort by the `step` column (PaySim's integer hour-offset from t=0).
    3. Take the first SUBSAMPLE_SIZE rows in time order.
    4. Persist as parquet with snappy compression.

Why first-200K-by-time (not random):
    Fraud is forecast-shaped: train on past, predict future. Random sampling
    spreads rows across the full 30-day timeline, which makes any subsequent
    train/test split leak future information into the past. A contiguous
    temporal slice gives Phase 2.6's split a clean time boundary.

Reproducibility:
    No randomness involved -- temporal sort + head(N) is fully deterministic.
    Running this script twice produces a byte-identical parquet file.

Usage:
    python -m fraudguard.data.subsample
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_CSV = REPO_ROOT / "data" / "raw" / "paysim.csv"
OUT_PARQUET = REPO_ROOT / "data" / "processed" / "paysim_200k.parquet"

SUBSAMPLE_SIZE = 200_000


def log(msg: str) -> None:
    print(f"[subsample] {msg}", flush=True)


def build_subsample() -> Path:
    """Read full PaySim CSV, take first 200K rows by `step`, save as parquet."""
    if not RAW_CSV.exists():
        log(f"missing {RAW_CSV}. Run scripts/download_paysim.py first.")
        sys.exit(1)

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    log(f"reading {RAW_CSV.name} ({RAW_CSV.stat().st_size / (1024 ** 2):.1f} MB)...")
    df = pd.read_csv(RAW_CSV)
    log(f"loaded {len(df):,} rows, {len(df.columns)} columns")

    log("sorting by `step` (temporal order)...")
    # mergesort is stable: rows with the same `step` keep their original
    # order, which matters for reproducibility across pandas versions.
    df = df.sort_values("step", kind="mergesort").reset_index(drop=True)

    log(f"taking first {SUBSAMPLE_SIZE:,} rows in time order...")
    sub = df.head(SUBSAMPLE_SIZE)
    step_min, step_max = int(sub["step"].min()), int(sub["step"].max())
    fraud_count = int(sub["isFraud"].sum())
    fraud_pct = 100.0 * fraud_count / len(sub)
    log(f"step range: {step_min}..{step_max}  |  fraud rows: {fraud_count:,} ({fraud_pct:.3f}%)")

    log(f"writing parquet -> {OUT_PARQUET}")
    sub.to_parquet(OUT_PARQUET, engine="pyarrow", compression="snappy", index=False)
    size_mb = OUT_PARQUET.stat().st_size / (1024**2)
    log(f"done. {OUT_PARQUET.name} ({size_mb:.1f} MB)")
    return OUT_PARQUET


if __name__ == "__main__":
    build_subsample()
