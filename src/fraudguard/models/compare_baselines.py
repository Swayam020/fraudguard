"""Run all baseline models and produce a comparison table.

Trains logreg, random forest, and xgboost on the same splits, evaluates
each on validation, and writes a markdown comparison table to
results/baseline_comparison.md.

Usage: python -m fraudguard.models.compare_baselines
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from fraudguard.data.constants import FEATURE_COLUMNS
from fraudguard.models.metrics import FraudMetrics, evaluate

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_PARQUET = REPO_ROOT / "data" / "processed" / "train.parquet"
VAL_PARQUET = REPO_ROOT / "data" / "processed" / "val.parquet"
RESULTS_DIR = REPO_ROOT / "results"
OUT_MD = RESULTS_DIR / "baseline_comparison.md"

RANDOM_SEED = 42


def log(msg: str) -> None:
    print(f"[compare] {msg}", flush=True)


def make_models(scale_pos_weight: float) -> dict:
    return {
        "LogisticRegression": Pipeline(
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
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        ),
    }


def write_table(results: dict[str, FraudMetrics]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Baseline Model Comparison",
        "",
        "Validation-set metrics on the PaySim 200K temporal split.",
        "Primary metric: PR-AUC (average precision). Threshold = 0.5 for P/R/F1.",
        "",
        "| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | TP | FP | FN |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in results.items():
        lines.append(
            f"| {name} | {m.roc_auc:.4f} | {m.pr_auc:.4f} | "
            f"{m.precision:.4f} | {m.recall:.4f} | {m.f1:.4f} | "
            f"{m.tp} | {m.fp} | {m.fn} |"
        )
    lines.append("")
    best = max(results.items(), key=lambda kv: kv[1].pr_auc)
    lines.append(f"**Best by PR-AUC:** {best[0]} ({best[1].pr_auc:.4f})")
    lines.append("")
    OUT_MD.write_text("\n".join(lines))
    log(f"wrote {OUT_MD}")


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

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    results: dict[str, FraudMetrics] = {}
    for name, model in make_models(scale_pos_weight).items():
        log(f"training {name}...")
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)[:, 1]
        m = evaluate(y_val, proba)
        results[name] = m
        log(f"  {name}: {m.summary()}")

    write_table(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
