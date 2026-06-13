"""Random forest baseline for PaySim fraud detection.

Tree ensembles handle non-linear feature interactions that logistic
regression misses (e.g. amount x type x balance_mismatch). Also gives
feature importances for free.

Usage: python -m fraudguard.models.baseline_rf
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from fraudguard.data.constants import FEATURE_COLUMNS
from fraudguard.models.metrics import evaluate

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_PARQUET = REPO_ROOT / "data" / "processed" / "train.parquet"
VAL_PARQUET = REPO_ROOT / "data" / "processed" / "val.parquet"

RANDOM_SEED = 42


def log(msg: str) -> None:
    print(f"[rf] {msg}", flush=True)


def main() -> int:
    if not TRAIN_PARQUET.exists() or not VAL_PARQUET.exists():
        log("missing splits. Run the data pipeline first.")
        return 1

    train = pd.read_parquet(TRAIN_PARQUET)
    val = pd.read_parquet(VAL_PARQUET)
    log(f"train {len(train):,} rows | val {len(val):,} rows")

    X_train = train[list(FEATURE_COLUMNS)]
    y_train = train["isFraud"]
    X_val = val[list(FEATURE_COLUMNS)]
    y_val = val["isFraud"]

    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )

    log("training (200 trees, all cores)...")
    clf.fit(X_train, y_train)

    val_proba = clf.predict_proba(X_val)[:, 1]
    m = evaluate(y_val.to_numpy(), val_proba)
    log(m.summary())

    log("feature importances:")
    importances = sorted(
        zip(FEATURE_COLUMNS, clf.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    for name, imp in importances:
        log(f"  {name:20s} {imp:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
