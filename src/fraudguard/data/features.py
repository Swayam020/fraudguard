"""Feature engineering for PaySim — produces the 9-feature input array.

Reads:
    data/processed/paysim_200k.parquet (output of subsample.py)

Writes:
    data/processed/paysim_200k_features.parquet

The 9 features (matching the paper's spec):
    1. amount_log         : log1p(amount); compresses heavy tail, handles zero safely
    2. amount_z           : z-score of amount_log; mean=0, std=1
    3. step_hour          : step % 24 -- hour of day
    4. step_day           : step // 24 -- day index
    5. is_transfer        : type == 'TRANSFER' (binary)
    6. is_cashout         : type == 'CASH_OUT' (binary)
    7. orig_balance_delta : oldbalanceOrg - newbalanceOrig (sender side change)
    8. dest_balance_delta : newbalanceDest - oldbalanceDest (receiver side change)
    9. balance_mismatch   : 1 if orig_balance_delta != amount, else 0
                            (PaySim fraud signature pattern)

Kept alongside for downstream use:
    - isFraud (target label)
    - step    (for temporal train/val/test split in Milestone 2.6)

Usage:
    python -m fraudguard.data.features
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
INPUT_PARQUET = REPO_ROOT / "data" / "processed" / "paysim_200k.parquet"
OUT_PARQUET = REPO_ROOT / "data" / "processed" / "paysim_200k_features.parquet"

EPSILON = 1e-9


def log(msg: str) -> None:
    print(f"[features] {msg}", flush=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the 9-feature transformation; return a new DataFrame."""
    out = pd.DataFrame(index=df.index)

    out["amount_log"] = np.log1p(df["amount"])

    mean = out["amount_log"].mean()
    std = out["amount_log"].std()
    out["amount_z"] = (out["amount_log"] - mean) / (std + EPSILON)

    out["step_hour"] = (df["step"] % 24).astype("int8")
    out["step_day"] = (df["step"] // 24).astype("int8")

    out["is_transfer"] = (df["type"] == "TRANSFER").astype("int8")
    out["is_cashout"] = (df["type"] == "CASH_OUT").astype("int8")

    out["orig_balance_delta"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    out["dest_balance_delta"] = df["newbalanceDest"] - df["oldbalanceDest"]

    out["balance_mismatch"] = ((out["orig_balance_delta"] - df["amount"]).abs() > 0.01).astype(
        "int8"
    )

    out["isFraud"] = df["isFraud"].astype("int8")
    out["step"] = df["step"].astype("int16")
    out["nameOrig"] = df["nameOrig"]
    out["nameDest"] = df["nameDest"]

    return out


def main() -> int:
    if not INPUT_PARQUET.exists():
        log(f"missing {INPUT_PARQUET}. Run subsample.py first.")
        return 1

    log(f"reading {INPUT_PARQUET.name}...")
    df = pd.read_parquet(INPUT_PARQUET)
    log(f"loaded {len(df):,} rows, {len(df.columns)} columns")

    log("computing 9 features...")
    features = build_features(df)
    log(f"output has {len(features):,} rows, {len(features.columns)} columns")
    log(f"columns: {list(features.columns)}")

    log("feature summary:")
    print(features.describe().round(4))

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUT_PARQUET, engine="pyarrow", compression="snappy", index=False)
    size_mb = OUT_PARQUET.stat().st_size / (1024**2)
    log(f"done. {OUT_PARQUET.name} ({size_mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
