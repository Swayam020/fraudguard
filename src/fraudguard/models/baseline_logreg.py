"""Logistic regression baseline for PaySim fraud detection.

Reads:  data/processed/{train,val}.parquet
Writes: nothing yet (model persistence lands in Milestone 3.8)

Trains a class-weighted logistic regression on the 9 engineered features
and reports validation metrics. Provides the floor the GNN must beat.

Usage: python -m fraudguard.models.baseline_logreg
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraudguard.data.constants import FEATURE_COLUMNS

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_PARQUET = REPO_ROOT / "data" / "processed" / "train.parquet"
VAL_PARQUET = REPO_ROOT / "data" / "processed" / "val.parquet"

RANDOM_SEED = 42


def log(msg: str) -> None:
    print(f"[logreg] {msg}", flush=True)


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

    # class_weight='balanced' reweights the loss by inverse class frequency --
    # essential at ~1100:1 imbalance, otherwise the model predicts all-legit.
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    log("training...")
    pipeline.fit(X_train, y_train)

    val_proba = pipeline.predict_proba(X_val)[:, 1]
    val_pred = pipeline.predict(X_val)

    roc_auc = roc_auc_score(y_val, val_proba)
    pr_auc = average_precision_score(y_val, val_proba)
    log(f"val ROC-AUC: {roc_auc:.4f}")
    log(f"val PR-AUC : {pr_auc:.4f}")
    log("val classification report (threshold=0.5):")
    print(classification_report(y_val, val_pred, digits=4, zero_division=0))

    return 0


if __name__ == "__main__":
    sys.exit(main())
