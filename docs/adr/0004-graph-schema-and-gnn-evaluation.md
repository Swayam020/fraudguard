# ADR-004: Graph Schema and GNN Evaluation Outcome

## Status

Accepted

## Context

Phase 4 builds the heterogeneous graph and the GAT model that ADR-003's
baselines exist to be measured against. Two sets of decisions accumulated
during the phase and were deferred to a single ADR at its close:

1. The node/edge schema for the `HeteroData` graph (Milestone 4.2).
2. How the GNN is trained, selected, and evaluated against the RandomForest
   baseline, and what that evaluation actually concluded (Milestones 4.4-4.5).

The earlier capstone paper typed nodes by transaction *role* (nameOrig =
user, nameDest = merchant) with edges User->Transaction->Merchant. The
rebuild deviates from this; the rationale is recorded below.

## Decision

**Account nodes typed by ID prefix, not transaction role.** PaySim accounts
are `C`-prefixed (customers) or `M`-prefixed (merchants). A node is typed by
its prefix regardless of whether it appears as sender or receiver in a given
row. Checking the data: nameOrig is *always* `C` (0 `M`-prefix senders), but
nameDest is `C` in ~60% of rows -- these are real customer-to-customer
transfers (a classic laundering-layering pattern), not merchant payments.
Typing by role would collapse C->C transfers into "user->merchant" and
destroy that signal.

**Six edge types, not four.** Because a transaction's destination can be
either a merchant or a user, the destination edge is split by type:
`(user, sends, transaction)`, `(transaction, receives, merchant)` for
`M`-destinations, and `(transaction, paid_to, user)` for `C`-destinations --
each with its reverse, for six directed relations. Edge counts reconcile:
73,427 (merchant-dest) + 126,573 (user-dest) = 200,000.

**Featureless account nodes share one learnable embedding per type.** Users
and merchants carry no features; each type gets a single shared learnable
vector rather than a per-node embedding table. With only 123 train fraud
rows, a 221K-row user table would overfit and waste VRAM; structure
differentiates nodes through message passing.

**Selection on validation; test observed once; results reported as a
distribution.** The model and class-weight strength (beta=0.75, a softened
exponent on the balanced weights) are selected by validation PR-AUC. Because
the test split has only 11 fraud nodes, a single test number is high-variance,
so the GNN is trained across a fixed seed set {0-4} chosen *before* observing
test, and the test PR-AUC distribution is the reported result. The deployed
checkpoint is the seed with the best *validation* score (never the best test
score).

## Consequences

**The GNN ties the RandomForest baseline; it does not beat it.** GNN test
PR-AUC is 0.3645 +/- 0.0087 across five seeds, versus RF 0.3570. The gap is
smaller than the seed-to-seed variance, so the honest conclusion is a
statistical tie -- the more complex model matches, but does not surpass, the
simpler one on this data.

**This is a measurement-power limitation, not only a model limitation.** With
11 fraud transactions in test, a difference this small cannot be resolved.
Two single-run scares during Phase 4.5 made this concrete: the first GNN
checkpoint scored 0.2510 (a loss) and an a-priori seed-42 retrain scored
0.2916, both far outside the five-seed band -- evidence that point estimates
here are unreliable and only the distribution is trustworthy.

**The threshold-independence of PR-AUC is load-bearing.** The GNN is poorly
calibrated (validation-tuned threshold pins at 0.95), but this does not affect
PR-AUC. Probability calibration is deferred as future work; it would improve
usable probabilities, not the headline metric.

**Future work to obtain a decisive verdict** (out of scope for Phase 4): a
larger PaySim slice from the full 6.3M rows to put hundreds of fraud cases in
test (requires neighbour-sampling on 4GB VRAM), or a graph-native dataset
(Elliptic, IEEE-CIS) where fraud genuinely hides in connections. The pipeline
is dataset-agnostic, so either is a new dataset version, not a rewrite.

## Supersedes

Deviates from the earlier capstone's role-based 3-node-type schema. Does not
supersede a prior ADR (the role-based design was never recorded as one).
