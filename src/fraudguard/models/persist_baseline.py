"""Persist the best baseline model (RandomForest) for downstream serving.

The Phase-3 winner is RandomForest with class_weight=balanced (val PR-AUC
0.4162, the number the GNN must beat). This module trains it deterministically,
saves it via joblib, and writes a metadata sidecar so the artifact is
version-guarded and reproducible. The load/predict helpers let the API and
decision layers (Phases 7-8) score new transactions.

Reads:  data/processed/train.parquet, data/processed/val.parquet
Writes: data/processed/baselines/rf_baseline.joblib
        data/processed/baselines/rf_baseline.meta.json
Usage:  python -m fraudguard.models.persist_baseline
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier

from fraudguard.data.constants import FEATURE_COLUMNS
from fraudguard.models.metrics import evaluate

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_PARQUET = REPO_ROOT / "data" / "processed" / "train.parquet"
VAL_PARQUET = REPO_ROOT / "data" / "processed" / "val.parquet"
BASELINE_DIR = REPO_ROOT / "data" / "processed" / "baselines"
MODEL_PATH = BASELINE_DIR / "rf_baseline.joblib"
META_PATH = BASELINE_DIR / "rf_baseline.meta.json"

RANDOM_SEED = 42
N_ESTIMATORS = 200


def log(msg: str) -> None:
    print(f"[persist] {msg}", flush=True)


def train_best_baseline(train: pd.DataFrame) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    clf.fit(train[list(FEATURE_COLUMNS)], train["isFraud"])
    return clf


def score(model: RandomForestClassifier, df: pd.DataFrame):
    """Fraud probabilities for df, using the canonical feature order."""
    return model.predict_proba(df[list(FEATURE_COLUMNS)])[:, 1]


def load_baseline():
    """Load persisted model + metadata. Warns on sklearn version drift."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"no model at {MODEL_PATH}; run persist_baseline first")
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    saved = meta.get("sklearn_version")
    if saved and saved != sklearn.__version__:
        log(f"WARNING: saved with sklearn {saved}, loaded under {sklearn.__version__}")
    return model, meta


def main() -> int:
    if not TRAIN_PARQUET.exists() or not VAL_PARQUET.exists():
        log("missing splits. Run the data pipeline first.")
        return 1

    train = pd.read_parquet(TRAIN_PARQUET)
    val = pd.read_parquet(VAL_PARQUET)
    log(f"train {len(train):,} rows | val {len(val):,} rows")

    log("training winner: RF class_weight=balanced, 200 trees, seed 42")
    model = train_best_baseline(train)

    y_val = val["isFraud"].to_numpy()
    m = evaluate(y_val, score(model, val))
    log(f"val {m.summary()}")
    log("(val PR-AUC should match the 0.4162 leaderboard number)")

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    meta = {
        "model": "RandomForestClassifier",
        "strategy": "class_weight=balanced",
        "n_estimators": N_ESTIMATORS,
        "random_seed": RANDOM_SEED,
        "feature_columns": list(FEATURE_COLUMNS),
        "val_pr_auc": m.pr_auc,
        "val_roc_auc": m.roc_auc,
        "sklearn_version": sklearn.__version__,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    META_PATH.write_text(json.dumps(meta, indent=2) + "\n")
    size_kb = MODEL_PATH.stat().st_size / 1024
    log(f"saved {MODEL_PATH.relative_to(REPO_ROOT)} ({size_kb:.0f} KB)")
    log(f"saved {META_PATH.relative_to(REPO_ROOT)}")

    log("round-trip: reload + re-score val")
    reloaded, _ = load_baseline()
    m2 = evaluate(y_val, score(reloaded, val))
    if abs(m.pr_auc - m2.pr_auc) > 1e-9:
        log(f"FAIL: reloaded PR-AUC {m2.pr_auc:.6f} != {m.pr_auc:.6f}")
        return 1
    log(f"round-trip OK: PR-AUC {m2.pr_auc:.4f} reproduced exactly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
