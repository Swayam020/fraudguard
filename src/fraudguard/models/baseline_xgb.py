"""XGBoost baseline for PaySim fraud detection.

Gradient-boosted trees are the standard strong tabular baseline. We use
scale_pos_weight (ratio of negatives to positives) to handle imbalance,
the XGBoost-native equivalent of class_weight='balanced'.

Usage: python -m fraudguard.models.baseline_xgb
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

from fraudguard.data.constants import FEATURE_COLUMNS
from fraudguard.models.metrics import evaluate

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_PARQUET = REPO_ROOT / "data" / "processed" / "train.parquet"
VAL_PARQUET = REPO_ROOT / "data" / "processed" / "val.parquet"

RANDOM_SEED = 42


def log(msg: str) -> None:
    print(f"[xgb] {msg}", flush=True)


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

    # scale_pos_weight = (# negatives) / (# positives): tells XGBoost to
    # weight the rare positive class up by that ratio in the loss gradient.
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)
    log(f"scale_pos_weight = {scale_pos_weight:.1f} ({n_neg} neg / {n_pos} pos)")

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )

    log("training (300 rounds, hist)...")
    clf.fit(X_train, y_train)

    val_proba = clf.predict_proba(X_val)[:, 1]
    m = evaluate(y_val.to_numpy(), val_proba)
    log(m.summary())

    log("feature importances (gain):")
    booster = clf.get_booster()
    scores = booster.get_score(importance_type="gain")
    # XGBoost names features f0, f1, ... map back to our names.
    named = {}
    for i, col in enumerate(FEATURE_COLUMNS):
        named[col] = scores.get(f"f{i}", 0.0)
    for name, imp in sorted(named.items(), key=lambda x: x[1], reverse=True):
        log(f"  {name:20s} {imp:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
