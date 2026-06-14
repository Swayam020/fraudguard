"""Threshold tuning for the best baseline (RandomForest).

The default 0.5 decision threshold is arbitrary for imbalanced data. We
sweep thresholds on validation, select the one maximizing F1, and report
the precision/recall trade-off curve. Writes results/threshold_tuning.md.

Usage: python -m fraudguard.models.tune_threshold
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from fraudguard.data.constants import FEATURE_COLUMNS
from fraudguard.models.metrics import evaluate

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_PARQUET = REPO_ROOT / "data" / "processed" / "train.parquet"
VAL_PARQUET = REPO_ROOT / "data" / "processed" / "val.parquet"
RESULTS_DIR = REPO_ROOT / "results"
OUT_MD = RESULTS_DIR / "threshold_tuning.md"

RANDOM_SEED = 42
THRESHOLDS = np.arange(0.05, 1.0, 0.05)


def log(msg: str) -> None:
    print(f"[tune] {msg}", flush=True)


def main() -> int:
    if not TRAIN_PARQUET.exists() or not VAL_PARQUET.exists():
        log("missing splits. Run the data pipeline first.")
        return 1

    train = pd.read_parquet(TRAIN_PARQUET)
    val = pd.read_parquet(VAL_PARQUET)

    X_train = train[list(FEATURE_COLUMNS)]
    y_train = train["isFraud"]
    X_val = val[list(FEATURE_COLUMNS)]
    y_val = val["isFraud"].to_numpy()

    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    log("training RandomForest...")
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_val)[:, 1]

    log("sweeping thresholds...")
    rows = []
    best_f1 = -1.0
    best_thr = 0.5
    for thr in THRESHOLDS:
        m = evaluate(y_val, proba, threshold=float(thr))
        rows.append((float(thr), m.precision, m.recall, m.f1, m.tp, m.fp, m.fn))
        if m.f1 > best_f1:
            best_f1 = m.f1
            best_thr = float(thr)

    log(f"best F1={best_f1:.4f} at threshold={best_thr:.2f}")
    best_m = evaluate(y_val, proba, threshold=best_thr)
    log(f"at best threshold: {best_m.summary()}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Threshold Tuning (RandomForest)",
        "",
        "Validation-set precision/recall/F1 across decision thresholds.",
        f"PR-AUC is threshold-independent: {best_m.pr_auc:.4f}.",
        "",
        "| Threshold | Precision | Recall | F1 | TP | FP | FN |",
        "|---|---|---|---|---|---|---|",
    ]
    for thr, p, r, f1, tp, fp, fn in rows:
        marker = " **<- best F1**" if abs(thr - best_thr) < 1e-9 else ""
        lines.append(f"| {thr:.2f} | {p:.4f} | {r:.4f} | {f1:.4f} | {tp} | {fp} | {fn} |{marker}")
    lines.append("")
    lines.append(f"**Selected threshold:** {best_thr:.2f} (max F1 = {best_f1:.4f})")
    lines.append("")
    OUT_MD.write_text("\n".join(lines))
    log(f"wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
