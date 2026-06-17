# ADR-003: Baseline Models, Imbalance Strategy, and Primary Metric

## Status
Accepted

## Context
Before building the heterogeneous GAT (Phases 4-6), FraudGuard needs defensible
baselines: a GNN result is only meaningful relative to what simpler models
already achieve on the same temporal split. The PaySim 200K split is severely
imbalanced (train 123 fraud / 140K rows = 0.088%; val 13 / 30K; test 11 / 30K),
which shapes every choice below.

Decisions needed:
1. Which baseline models to establish before the GNN.
2. How to handle class imbalance: resampling (SMOTE) vs cost-sensitive learning
   (class weighting).
3. Which metric is primary -- the number the GNN must beat -- and how to treat
   the decision threshold.

## Decision
**Three baselines spanning model families.** LogisticRegression (linear floor),
RandomForest (nonlinear bagged trees), and XGBoost (gradient-boosted trees). All
share the same features, split, seed (42), and evaluation code so differences
are attributable to the model, not the harness. RandomForest wins on the primary
metric (val PR-AUC 0.4162) and becomes the bar the GNN must beat. LogReg 0.3351,
XGBoost 0.2735.

**Class weighting over SMOTE.** Milestone 3.7 tested SMOTE oversampling
(train-only, k_neighbors=5) against `class_weight="balanced"` on the same RF.
Class weighting won decisively on PR-AUC (0.4162 vs 0.3106). SMOTE raised
*ROC-AUC* (0.9190 vs 0.8068) but *lowered* PR-AUC -- with only 123 real fraud
rows, synthetic interpolation manufactures minority points from too thin a
neighborhood, adding noise rather than signal. Class weighting reweights the
loss without inventing data, so it is the strategy carried into the GNN.

**PR-AUC (average precision) as the primary metric.** ROC-AUC is secondary,
accuracy is rejected outright (an all-legit predictor scores 99.96%). The
decision threshold is treated as a *separate* concern from model selection:
models are ranked by threshold-independent PR-AUC, then the operating threshold
is tuned afterward (Milestone 3.6: RF best F1 0.6667 at threshold 0.10).

## Consequences
Positive:
- A clear, reproducible target (PR-AUC 0.4162) for the GNN, established across
  three model families rather than a single arbitrary baseline.
- Metric choice is grounded in this project's own evidence (see below), not
  convention -- directly defensible in review/viva.
- Imbalance handled without fabricating data, keeping the train distribution
  honest.

Negative / risks:
- PR-AUC on 13 val frauds has high variance; the 0.4162 bar carries an error
  bar. Mitigation is the same as ADR-002: enlarge the subsample and re-split if
  Phase 6 comparisons are too noisy to call.
- RandomForest scores P=R=F1=0 at the default 0.5 threshold despite the best
  PR-AUC -- a strong ranker with a badly-placed default cutoff. This is expected
  on imbalanced data and is precisely why threshold tuning is a separate step;
  it must be remembered whenever the persisted model is served (Phase 7+).

## Alternatives Considered
- **Skip baselines, go straight to the GNN**: rejected -- a headline GNN number
  with nothing to compare against proves nothing.
- **SMOTE for imbalance**: rejected on evidence (3.7: lower PR-AUC, ROC-AUC
  optimism). Revisit only if a much larger fraud count makes interpolation
  trustworthy.
- **Accuracy or ROC-AUC as primary metric**: rejected. Accuracy is meaningless
  at 0.04% prevalence. ROC-AUC misleads here -- LogReg posts the highest ROC-AUC
  (0.9731) yet needs 1202 false positives to catch 12 frauds (precision 0.0099),
  while its PR-AUC (0.3351) correctly ranks it below RandomForest. PR-AUC tracks
  the precision/recall tradeoff that actually matters for fraud triage.

## Related ADRs
- ADR-002: dataset choice, sampling, temporal split (defines the split these
  baselines are trained and evaluated on).
