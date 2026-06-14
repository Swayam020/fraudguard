# SMOTE vs Class-Weighting (RandomForest)

Validation PR-AUC comparison. Train fraud rows: 123. SMOTE k_neighbors=5. Both arms: RF n_estimators=200, seed=42.
Metrics at threshold 0.50 (PR-AUC / ROC-AUC are threshold-independent).

| Strategy | PR-AUC | ROC-AUC | Precision | Recall | F1 | TP | FP | FN |
|----------|--------|---------|-----------|--------|----|----|----|----|
| class_weight=balanced | 0.4162 | 0.8068 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 13 |
| SMOTE + plain RF | 0.3106 | 0.9190 | 0.5000 | 0.1538 | 0.2353 | 2 | 2 | 11 |

**Winner by PR-AUC: class_weight.**

Note: with so few real fraud rows, SMOTE interpolates synthetic minorities from a thin neighborhood; class-weighting reweights the loss without inventing samples. This table is the evidence for the imbalance-strategy choice carried into the GNN phase.
