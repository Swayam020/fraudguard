# Test-Set Evaluation (Milestone 4.5)

GNN uses class-weight beta=0.75 (selected on validation in the weight sweep). Trained across 5 seeds; test PR-AUC is reported as a distribution because the test split has only 11 fraud nodes, so any single run is high-variance. Deployed checkpoint = the seed with best VALIDATION PR-AUC (selection on val, never on test).

**GNN test PR-AUC: 0.3645 +/- 0.0087** (5 seeds)  vs  **RandomForest: 0.3570**

Deployed checkpoint: seed 2 (val PR-AUC 0.5127), test PR-AUC 0.3625.

| seed | val PR-AUC | test PR-AUC |
|---|---|---|
| 0 | 0.4823 | 0.3504 |
| 1 | 0.4877 | 0.3640 |
| 2 | 0.5127 | 0.3625 | **<- deployed (best val)**
| 3 | 0.5037 | 0.3685 |
| 4 | 0.5052 | 0.3772 |

## Honest reading

- GNN and RandomForest are statistically indistinguishable on test PR-AUC; the gap is far smaller than the seed-to-seed variance.
- RandomForest remains the simpler, equally strong baseline.
- GNN ranks well (ROC-AUC ~0.93) but is poorly calibrated; probability calibration is noted as future work (does not affect PR-AUC).
- Root cause of the wide error bars: only 11 fraud transactions in the test split (ADR-002 variance risk).
