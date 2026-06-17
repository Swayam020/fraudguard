"""Unit tests for models.metrics.evaluate and the persisted baseline.

The evaluate() tests use a hand-built array with known answers, so the
assertions are exact and run anywhere (no data, no model, CI-safe). The
persisted-model tests are skipped unless rf_baseline.joblib exists locally
(it is gitignored, so it is absent in CI and on fresh clones).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudguard.data.constants import FEATURE_COLUMNS
from fraudguard.models.metrics import evaluate
from fraudguard.models.persist_baseline import MODEL_PATH, load_baseline, score

TOL = 1e-9

# Hand-built example: 4 negatives, 2 positives, perfectly rank-separated.
Y_TRUE = np.array([0, 0, 0, 0, 1, 1])
Y_PROBA = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.9])


class TestEvaluateConfusion:
    def test_confusion_counts_at_half(self) -> None:
        m = evaluate(Y_TRUE, Y_PROBA, threshold=0.5)
        assert (m.tn, m.fp, m.fn, m.tp) == (3, 1, 0, 2)

    def test_precision_recall_f1_at_half(self) -> None:
        m = evaluate(Y_TRUE, Y_PROBA, threshold=0.5)
        assert abs(m.precision - 2 / 3) < TOL
        assert abs(m.recall - 1.0) < TOL
        assert abs(m.f1 - 0.8) < TOL

    def test_threshold_shifts_predictions(self) -> None:
        m = evaluate(Y_TRUE, Y_PROBA, threshold=0.65)
        assert (m.tn, m.fp, m.fn, m.tp) == (4, 0, 0, 2)
        assert abs(m.precision - 1.0) < TOL
        assert abs(m.recall - 1.0) < TOL

    def test_threshold_is_recorded(self) -> None:
        assert abs(evaluate(Y_TRUE, Y_PROBA, threshold=0.42).threshold - 0.42) < TOL

    def test_no_positive_predictions_is_zero_not_crash(self) -> None:
        m = evaluate(Y_TRUE, Y_PROBA, threshold=0.999)
        assert (m.tn, m.fp, m.fn, m.tp) == (4, 0, 2, 0)
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0


class TestEvaluateRanking:
    def test_perfect_separation_aucs_are_one(self) -> None:
        m = evaluate(Y_TRUE, Y_PROBA)
        assert abs(m.roc_auc - 1.0) < TOL
        assert abs(m.pr_auc - 1.0) < TOL

    def test_aucs_are_threshold_independent(self) -> None:
        a = evaluate(Y_TRUE, Y_PROBA, threshold=0.1)
        b = evaluate(Y_TRUE, Y_PROBA, threshold=0.9)
        assert a.roc_auc == b.roc_auc
        assert a.pr_auc == b.pr_auc

    def test_metrics_within_unit_interval(self) -> None:
        rng = np.random.default_rng(42)
        y = rng.integers(0, 2, size=500)
        p = rng.random(500)
        m = evaluate(y, p)
        for v in (m.roc_auc, m.pr_auc, m.precision, m.recall, m.f1):
            assert 0.0 <= v <= 1.0


class TestFraudMetricsType:
    def test_counts_are_ints(self) -> None:
        m = evaluate(Y_TRUE, Y_PROBA)
        assert all(isinstance(v, int) for v in (m.tn, m.fp, m.fn, m.tp))

    def test_scores_are_floats(self) -> None:
        m = evaluate(Y_TRUE, Y_PROBA)
        assert all(isinstance(v, float) for v in (m.roc_auc, m.pr_auc, m.f1))

    def test_summary_is_str_with_key_tokens(self) -> None:
        s = evaluate(Y_TRUE, Y_PROBA).summary()
        assert isinstance(s, str)
        assert "PR-AUC" in s and "F1" in s


@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="rf_baseline.joblib absent (gitignored); run persist_baseline first",
)
class TestPersistedBaseline:
    def test_load_returns_model_and_meta(self) -> None:
        model, meta = load_baseline()
        assert hasattr(model, "predict_proba")
        assert meta["feature_columns"] == list(FEATURE_COLUMNS)

    def test_meta_records_provenance(self) -> None:
        _, meta = load_baseline()
        assert meta["strategy"] == "class_weight=balanced"
        assert meta["random_seed"] == 42
        assert abs(meta["val_pr_auc"] - 0.4162) < 1e-3

    def test_score_is_deterministic_and_bounded(self) -> None:
        model, _ = load_baseline()
        df = pd.DataFrame(0.0, index=range(5), columns=list(FEATURE_COLUMNS))
        p1 = score(model, df)
        p2 = score(model, df)
        assert np.array_equal(p1, p2)
        assert ((p1 >= 0.0) & (p1 <= 1.0)).all()
