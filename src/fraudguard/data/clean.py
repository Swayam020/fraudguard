"""Clean PaySim features: fix balance-column data quality issues, drop outliers.

Reads:  data/processed/paysim_200k_features.parquet
Writes: data/processed/paysim_200k_features_clean.parquet

Steps:
    1. Diagnose balance_mismatch fraud-conditional rate (before).
    2. Mask balance_mismatch=0 where both dest balances are 0 (missing-data signal).
    3. Drop |amount_z| > 5 outliers.
    4. Diagnose balance_mismatch fraud-conditional rate (after).

Usage: python -m fraudguard.data.clean
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
INPUT_PARQUET = REPO_ROOT / "data" / "processed" / "paysim_200k_features.parquet"
RAW_PARQUET = REPO_ROOT / "data" / "processed" / "paysim_200k.parquet"
OUT_PARQUET = REPO_ROOT / "data" / "processed" / "paysim_200k_features_clean.parquet"

Z_THRESHOLD = 5.0


def log(msg: str) -> None:
    print(f"[clean] {msg}", flush=True)


def diagnose(df: pd.DataFrame, label: str) -> None:
    fraud_mask = df["isFraud"] == 1
    legit_mask = df["isFraud"] == 0
    fraud_rate = df.loc[fraud_mask, "balance_mismatch"].mean()
    legit_rate = df.loc[legit_mask, "balance_mismatch"].mean()
    log(f"{label}: balance_mismatch rate -- fraud={fraud_rate:.4f}, legit={legit_rate:.4f}")


def main() -> int:
    if not INPUT_PARQUET.exists() or not RAW_PARQUET.exists():
        log("missing input parquets. Run subsample.py and features.py first.")
        return 1

    log(f"reading {INPUT_PARQUET.name}...")
    df = pd.read_parquet(INPUT_PARQUET)
    raw = pd.read_parquet(RAW_PARQUET)
    log(f"loaded {len(df):,} rows")

    diagnose(df, "before")

    # Mask balance_mismatch where both destination balances are 0
    # (PaySim's missing-data signal for non-tracked merchant accounts).
    dest_both_zero = (raw["oldbalanceDest"] == 0) & (raw["newbalanceDest"] == 0)
    masked = dest_both_zero.sum()
    df.loc[dest_both_zero, "balance_mismatch"] = 0
    log(f"masked {masked:,} rows where both dest balances are 0")

    # Drop |amount_z| > Z_THRESHOLD outliers.
    before_n = len(df)
    df = df.loc[df["amount_z"].abs() <= Z_THRESHOLD].reset_index(drop=True)
    dropped = before_n - len(df)
    log(f"dropped {dropped:,} outlier rows (|amount_z| > {Z_THRESHOLD})")

    diagnose(df, "after ")

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, engine="pyarrow", compression="snappy", index=False)
    size_mb = OUT_PARQUET.stat().st_size / (1024**2)
    log(f"done. {OUT_PARQUET.name} ({size_mb:.2f} MB, {len(df):,} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
