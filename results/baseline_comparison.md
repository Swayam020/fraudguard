# Baseline Model Comparison

Validation-set metrics on the PaySim 200K temporal split.
Primary metric: PR-AUC (average precision). Threshold = 0.5 for P/R/F1.

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|
| LogisticRegression | 0.9731 | 0.3351 | 0.0099 | 0.9231 | 0.0196 | 12 | 1202 | 1 |
| RandomForest | 0.8068 | 0.4162 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 13 |
| XGBoost | 0.9246 | 0.2735 | 0.3333 | 0.4615 | 0.3871 | 6 | 12 | 7 |

**Best by PR-AUC:** RandomForest (0.4162)
