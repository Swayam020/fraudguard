"""Evaluation metrics for imbalanced fraud classification.

Centralizes metric computation so every model (baselines + GNN) reports
identically. Accuracy is deliberately de-emphasized: at ~1100:1 imbalance
an all-legit predictor scores 99.9% accuracy.

Primary metric: PR-AUC (average precision). Secondary: ROC-AUC, F1,
precision, recall at a given threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class FraudMetrics:
    """Container for one evaluation pass."""

    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    threshold: float
    tn: int
    fp: int
    fn: int
    tp: int

    def summary(self) -> str:
        return (
            f"ROC-AUC={self.roc_auc:.4f} PR-AUC={self.pr_auc:.4f} | "
            f"@thr={self.threshold:.3f}: P={self.precision:.4f} "
            f"R={self.recall:.4f} F1={self.f1:.4f} | "
            f"TN={self.tn} FP={self.fp} FN={self.fn} TP={self.tp}"
        )


def evaluate(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> FraudMetrics:
    """Compute the full metric set from true labels and predicted probabilities."""
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return FraudMetrics(
        roc_auc=float(roc_auc_score(y_true, y_proba)),
        pr_auc=float(average_precision_score(y_true, y_proba)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        threshold=threshold,
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )
