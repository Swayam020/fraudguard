"""Test-set evaluation: HeteroGAT vs RandomForest, head-to-head (Milestone 4.5)."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier

from fraudguard.data.constants import FEATURE_COLUMNS
from fraudguard.models.hetero_gat import HeteroGAT
from fraudguard.models.metrics import evaluate

REPO_ROOT = Path(__file__).resolve().parents[3]
PROC = REPO_ROOT / "data" / "processed"
GRAPH_PT = PROC / "graph" / "hetero_data.pt"
CKPT = PROC / "graph" / "hetero_gat_best.pt"
RESULTS_DIR = REPO_ROOT / "results"
OUT_MD = RESULTS_DIR / "test_evaluation.md"
SEED = 42
THRESHOLDS = np.arange(0.05, 1.0, 0.05)


def best_threshold(y_val, p_val):
    best_f1, best_thr = -1.0, 0.5
    for thr in THRESHOLDS:
        m = evaluate(y_val, p_val, threshold=float(thr))
        if m.f1 > best_f1:
            best_f1, best_thr = m.f1, float(thr)
    return best_thr


def gnn_probs():
    data = torch.load(GRAPH_PT, weights_only=False)
    model = HeteroGAT()
    blob = torch.load(CKPT, weights_only=False)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    with torch.no_grad():
        prob = F.softmax(model(data), dim=1)[:, 1].numpy()
    y = data["transaction"].y.numpy()
    vm = data["transaction"].val_mask.numpy()
    tm = data["transaction"].test_mask.numpy()
    return y[vm], prob[vm], y[tm], prob[tm]


def rf_probs():
    cols = list(FEATURE_COLUMNS)
    train = pd.read_parquet(PROC / "train.parquet")
    val = pd.read_parquet(PROC / "val.parquet")
    test = pd.read_parquet(PROC / "test.parquet")
    clf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=SEED
    )
    clf.fit(train[cols], train["isFraud"])
    p_val = clf.predict_proba(val[cols])[:, 1]
    p_test = clf.predict_proba(test[cols])[:, 1]
    return (
        val["isFraud"].to_numpy(),
        p_val,
        test["isFraud"].to_numpy(),
        p_test,
    )


def run_model(name, y_val, p_val, y_test, p_test):
    thr = best_threshold(y_val, p_val)
    m = evaluate(y_test, p_test, threshold=thr)
    print(f"\n{name} (val-tuned thr={thr:.2f})")
    print("  " + m.summary())
    return name, thr, m


def main():
    print(f"GNN test fraud nodes: checking against {CKPT.name}")
    results = [
        run_model("HeteroGAT", *gnn_probs()),
        run_model("RandomForest", *rf_probs()),
    ]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Test-Set Evaluation (Milestone 4.5)",
        "",
        "Held-out test split. Threshold tuned on validation (max F1), applied "
        "to test. PR-AUC is threshold-independent and is the primary metric.",
        "",
        "| Model | PR-AUC | ROC-AUC | Thr | P | R | F1 | TP | FP | FN |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, thr, m in results:
        lines.append(
            f"| {name} | {m.pr_auc:.4f} | {m.roc_auc:.4f} | {thr:.2f} | "
            f"{m.precision:.4f} | {m.recall:.4f} | {m.f1:.4f} | "
            f"{m.tp} | {m.fp} | {m.fn} |"
        )
    gnn_pr = results[0][2].pr_auc
    rf_pr = results[1][2].pr_auc
    delta = gnn_pr - rf_pr
    lines += [
        "",
        f"**HeteroGAT vs RandomForest (test PR-AUC):** "
        f"{gnn_pr:.4f} vs {rf_pr:.4f} "
        f"({'+' if delta >= 0 else ''}{delta:.4f})",
        "",
    ]
    OUT_MD.write_text("\n".join(lines))
    print(f"\nwrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
