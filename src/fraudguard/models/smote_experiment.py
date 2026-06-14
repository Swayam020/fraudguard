"""SMOTE oversampling vs class-weighting for the RandomForest baseline.

Tests whether synthetic minority oversampling (SMOTE, applied train-only to
avoid leakage) beats class_weight='balanced' on validation PR-AUC. With only
~120 fraud rows in train, SMOTE interpolates from very few real neighbors, so
it may not help -- reporting that honestly is a valid result.

Both arms use the same RF config for an apples-to-apples comparison; the only
difference is the imbalance strategy. Winner chosen by PR-AUC.

Reads:  data/processed/train.parquet, data/processed/val.parquet
Writes: results/smote_experiment.md
Usage:  python -m fraudguard.models.smote_experiment
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

from fraudguard.data.constants import FEATURE_COLUMNS
from fraudguard.models.metrics import FraudMetrics, evaluate

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_PARQUET = REPO_ROOT / "data" / "processed" / "train.parquet"
VAL_PARQUET = REPO_ROOT / "data" / "processed" / "val.parquet"
RESULTS_MD = REPO_ROOT / "results" / "smote_experiment.md"

RANDOM_SEED = 42
N_ESTIMATORS = 200


def log(msg: str) -> None:
    print(f"[smote] {msg}", flush=True)


def _rf(class_weight):
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )


def _row(name: str, m: FraudMetrics) -> str:
    return (
        f"| {name} | {m.pr_auc:.4f} | {m.roc_auc:.4f} | "
        f"{m.precision:.4f} | {m.recall:.4f} | {m.f1:.4f} | "
        f"{m.tp} | {m.fp} | {m.fn} |"
    )


def _write_results(m_a, m_b, n_pos, k, winner):
    lines = [
        "# SMOTE vs Class-Weighting (RandomForest)",
        "",
        f"Validation PR-AUC comparison. Train fraud rows: {n_pos}. "
        f"SMOTE k_neighbors={k}. Both arms: RF n_estimators=200, seed=42.",
        "Metrics at threshold 0.50 (PR-AUC / ROC-AUC are threshold-independent).",
        "",
        "| Strategy | PR-AUC | ROC-AUC | Precision | Recall | F1 | TP | FP | FN |",
        "|----------|--------|---------|-----------|--------|----|----|----|----|",
        _row("class_weight=balanced", m_a),
        _row("SMOTE + plain RF", m_b),
        "",
        f"**Winner by PR-AUC: {winner}.**",
        "",
        "Note: with so few real fraud rows, SMOTE interpolates synthetic "
        "minorities from a thin neighborhood; class-weighting reweights the loss "
        "without inventing samples. This table is the evidence for the "
        "imbalance-strategy choice carried into the GNN phase.",
    ]
    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_MD, "w") as f:
        f.write("\n".join(lines) + "\n")


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

    n_pos = int(y_train.sum())
    log(f"train {len(train):,} rows ({n_pos} fraud) | val {len(val):,} rows")

    log("arm A: RF class_weight=balanced (no resampling)")
    clf_a = _rf("balanced")
    clf_a.fit(X_train, y_train)
    proba_a = clf_a.predict_proba(X_val)[:, 1]
    m_a = evaluate(y_val, proba_a)
    log(m_a.summary())

    k = min(5, n_pos - 1)
    log(f"arm B: SMOTE(k_neighbors={k}) on train, then plain RF")
    sm = SMOTE(random_state=RANDOM_SEED, k_neighbors=k)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    log(f"resampled train: {len(X_res):,} rows ({int(y_res.sum())} fraud)")
    clf_b = _rf(None)
    clf_b.fit(X_res, y_res)
    proba_b = clf_b.predict_proba(X_val)[:, 1]
    m_b = evaluate(y_val, proba_b)
    log(m_b.summary())

    winner = "class_weight" if m_a.pr_auc >= m_b.pr_auc else "SMOTE"
    log(f"winner by PR-AUC: {winner}")

    _write_results(m_a, m_b, n_pos, k, winner)
    log(f"wrote {RESULTS_MD.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
