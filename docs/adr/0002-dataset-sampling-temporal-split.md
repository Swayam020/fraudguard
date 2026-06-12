# ADR-002: Dataset Choice, Sampling Strategy, and Temporal Split

## Status

Accepted

## Context

FraudGuard requires a transaction dataset with labeled fraud for supervised
training. Real financial data is unavailable (privacy/licensing). The paper
this rebuild reproduces used PaySim, a synthetic mobile-money simulator
dataset published on Kaggle (ealaxi/paysim1), subsampled to 200K rows.

Decisions needed:
1. Which dataset, and how to guarantee we have the same bytes the pipeline
   was built against.
2. How to subsample 6.36M rows down to a tractable 200K on 8GB-RAM hardware.
3. How to split train/val/test without leaking future information.

## Decision

**Dataset: PaySim, pinned by SHA-256.** The downloader script
(`scripts/download_paysim.py`) verifies the upstream CSV against a pinned
hash and fails loudly on mismatch. This guards against silent upstream
re-uploads changing our inputs.

**Subsample: first 200K rows in temporal order (steps 1-13).** Sorted by
`step` with a stable mergesort, then head(200_000). Fully deterministic --
no random seed needed. Verified by hashing the output parquet across runs.

**Split: temporal 70/15/15 by row position after step-sort.** Train gets
the earliest 140K rows (steps 1-11), val the next 30K (steps 11-12), test
the final 30K (steps 12-13). Assertions enforce
`train.step.max() <= val.step.min() <= test.step.max()`.

**No stratification.** Forcing equal fraud rates across splits would require
shuffling within temporal blocks, reintroducing leakage. We report natural
fraud rates per split instead (train 0.088%, val 0.043%, test 0.037%).

## Consequences

Positive:
- Zero temporal leakage: the model never trains on data from the future of
  its evaluation window, matching production reality.
- Full reproducibility: every pipeline stage is deterministic and
  hash-verified.
- Honest evaluation: uneven fraud rates across splits reflect real-world
  drift.

Negative / risks:
- Small fraud counts in val (13) and test (11) limit statistical power of
  PR-AUC estimates. Run-to-run metric variance will be visible.
  Mitigation: if Phase 6 variance is unacceptable, enlarge the subsample
  (e.g. 500K rows) and re-derive splits -- the pipeline supports this with
  a single constant change.
- First-200K-by-time covers only ~13 hours of the 30-day simulation; later
  temporal patterns (e.g. day-of-week effects) are absent. Accepted because
  it matches the paper rebuild target.

## Alternatives Considered

- **Random 200K sample + random split**: rejected -- temporal leakage
  inflates metrics and misrepresents production performance.
- **Stratified temporal split**: rejected -- stratification within temporal
  blocks requires shuffling, defeating the purpose.
- **Full 6.36M dataset**: rejected for hardware constraints (8GB RAM) and
  paper parity (paper used 200K).
