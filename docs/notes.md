# Learning Notes

> Personal running notes — gotchas, "aha" moments, things to remember.
> Informal. Not for viva or recruiters — for me.

## Phase 0 — Setup

- .git is a directory, not a file. Directories have 'd' in ls -la permissions.
- git init is local only. Files only get to GitHub after git push.
- .gitignore prevents files from being added to tracking — it does NOT remove files already committed. If you commit a secret, the secret is in history forever — must rotate it.
- git add (stage) and git commit (record) are separate so you can pick which changes go in each commit.
- git commit "message" without -m makes git look for a FILE named "message". Always use -m.
- Venv must be activated (source venv/bin/activate) before pip install, otherwise install fails or pollutes system Python.
- Never commit venv/ — too large (GBs), platform-specific (Linux binaries don't work on Mac), and has hardcoded paths.
- Empty folders aren't tracked by git. Use .gitkeep placeholder files.
- Broad gitignore rules like data/raw/ block EVERYTHING inside — including .gitkeep. Use data/raw/* + !data/raw/.gitkeep to keep the folder visible.
- from fraudguard.models import X doesn't work just because folders have __init__.py. Python needs src/ to be on sys.path — solved later with pip install -e . and pyproject.toml.
- ADRs are immutable. To change a past decision, write a NEW ADR that supersedes the old one, and update the old one's Status to "Superseded by ADR-XXX".
- Triple-backtick code blocks inside markdown files cause issues when pasted into nano. Use cat > file << 'EOF' heredoc method instead.

## Phase 0 — Interview Answers

- FastAPI vs Flask interview answer = async-native + Pydantic validation. Concrete I/O wait in this project = MongoDB vector query + GNN forward pass.
- ADR-001 is the only ADR allowed to cover multiple decisions (the "stack snapshot"). Every later ADR covers ONE decision.

## Phase 2 — Data Foundations (completed)

- kaggle CLI: legacy API key at ~/.kaggle/kaggle.json, chmod 600 required. Never cat secrets to screen.
- SHA-256 pinning: paysim.csv = 16910f90...eea6b. Script fails loudly on upstream change.
- Heredocs corrupt large files in terminal; use python with open().write() instead.
- pip install -e . requires pyproject.toml AND __init__.py in every package dir (namespace packages have __file__=None).
- mergesort (stable) for reproducible sorts; quicksort is not stable.
- log1p over log: handles amount=0.
- Temporal split, never random: fraud is forecast-shaped. Assertions: train.step.max() <= val.step.min().
- No stratification within temporal split — would require shuffling = leakage. Report natural rates instead.
- Pipeline: subsample -> features -> clean -> split. Each stage hash-verified deterministic.
- Test fraud counts are small (train 123 / val 13 / test 11) — PR-AUC variance risk, flagged in ADR-002.
- black reformat-then-abort on first commit is normal: re-add, re-commit.
## Phase 3 — Baseline Models (completed)
- Leaderboard (val PR-AUC): RF 0.4162 (best) > LogReg 0.3351 > XGB 0.2735.
- SMOTE did NOT help (PR-AUC 0.3106 vs 0.4162) — only 123 fraud train rows, too thin for synthetic interpolation. SMOTE raised ROC-AUC (0.9190 vs 0.8068) while hurting PR-AUC — textbook case for why PR-AUC is the primary metric here, not ROC-AUC.
- RF at default threshold 0.5 gives P=R=F1=0 despite best PR-AUC — great ranker, wrong cutoff. Tuned threshold to 0.10 -> F1 0.6667 (P 0.7273, R 0.6154).
- LogReg has highest ROC-AUC (0.9731) but needs 1202 false positives to catch 12 frauds — cautionary example for why ROC-AUC alone is misleading on rare-event problems.
- rf_baseline.joblib gitignored (5.1MB > 500KB pre-commit cap). Regenerate via python -m fraudguard.models.persist_baseline after fresh clone.
- Workflow: forgetting source venv/bin/activate -> ModuleNotFoundError or system python picked up (check with which python). black + ruff both can reformat-then-abort the first commit attempt — re-run the same add+commit, passes 2nd try.
- CI test count: 25 passed / 3 skipped (joblib absent in fresh clone) is correct, not a regression; locally 28 passed.

## Phase 4.1 — Node Indexing (completed)
- nameOrig/nameDest were dropped silently by clean.py's positional alignment + reset_index(drop=True) after dropping outliers — fixed at features.py source, not downstream, to avoid re-breaking on every pipeline re-run.
- 294,797 unique accounts from 200k txns (400k name-slots) is expected for PaySim: merchant (M-prefix) accounts are mostly one-time destinations, low reuse.
- account_id_map: account_name -> contiguous int. txn_id_map: global txn_id with (split, row_in_split) to trace back to train/val/test rows.
- Stray __init.py (missing 2nd underscore) sat untouched in graph/ since Phase 0 — git tracked it; cleaned up before adding real module.

## 2026-06-20 — Milestone 4.2 (HeteroData) + 4.3 (HeteroGAT)

### 4.2 — graph construction
- Resolved the open edge-type question by checking the data, not assuming: `nameOrig` is **always** C-prefix (0 M-prefix senders across all splits); `nameDest` is C-prefix ~60% of rows (86343/19851/20379 train/val/test). Those C-dest rows are real C→C transfers (CASH_OUT/TRANSFER), not merchant payments.
- Decision: **6 edge types**, not the 4 from the handoff plan. Routing a C-dest to a merchant node would be wrong (it's a user node per the prefix-typing from 4.1). Forward edges: user-sends-transaction, transaction-receives-merchant (M-dest, 73427 edges), transaction-paid_to-user (C-dest, 126573 edges); each with its reverse. 73427 + 126573 = 200000 ✓.
- `build_graph.py` writes `hetero_data.pt` (22M, gitignored — regenerate via `python -m fraudguard.graph.build_graph`). transaction nodes carry `.x` (9 feats), `.y` (isFraud), and train/val/test bool masks. user/merchant are featureless (num_nodes only).
- Verified: fraud count 147 = 123+13+11 across splits ✓; `data.validate()` passes; all edge indices in bounds.
- `FEATURE_COLS` is duplicated in build_graph.py (not a shared constant). Minor tech-debt — could centralise later if a 3rd consumer appears.

### 4.3 — model
- `HeteroGAT`: HeteroConv-wrapped GATConv per edge type, 2 layers, hidden 64, 4 heads, dropout 0.3. Kept the paper's hyperparams but adapted for hetero. `add_self_loops=False` is mandatory for inter-type edges.
- Featureless user/merchant nodes each get **one shared learnable embedding vector** (expanded across all nodes), NOT a per-node embedding table. With only 147 fraud rows, a 221k-row user table would overfit and waste VRAM. Structure differentiates nodes after message passing.
- Tiny: 52,354 params. Forward gives (200000, 2) logits.
- Gotcha: the final-layer convs that produce user/merchant outputs get **no gradient** — we only read `x_dict["transaction"]` after the last layer, so those outputs are dead-ends. This is correct for single-target-type hetero GNNs, not a bug. First version of the backward test naively asserted grads on *all* params and failed; fixed it to assert grads on core params (txn_lin, head, both embeddings) + at least one conv.
- Training loop deliberately deferred to 4.4 (its own decisions: loss, class weight, LR, full-graph vs mini-batch on 4GB).

## 2026-06-21 — Milestone 4.5 (test-set evaluation)

The honest headline: **the GNN does NOT beat RandomForest on the held-out test set. They tie.** GNN test PR-AUC 0.3645 ± 0.0087 (5 seeds) vs RF 0.3570.

### How we got there (the messy, real version)
- First single-checkpoint eval (the 4.4 model) gave GNN test PR-AUC **0.2510 vs RF 0.3570** — the GNN *lost*. The 4.4 validation win (0.48 vs 0.42) did NOT carry to test.
- Diagnosis: model was early-stopped/selected on a val set with only **13 fraud nodes** → overfit to those 13 (the ADR-002 variance risk, made concrete). Plus poor calibration: val-tuned GNN threshold landed at 0.95 (overconfident, probs squashed toward 1).
- Fix attempt 1 — multi-seed + gentler class weighting. Added `weight_beta` exponent to `class_weights` (beta=1.0 = original aggressive weighting; lower = softer) and a `seed` param. `sweep_gat.py` ran beta ∈ {0.5, 0.75, 1.0} × 5 seeds, selecting beta by **mean val PR-AUC**. Winner: beta=0.75 (val 0.498), test 0.3645 ± 0.0087.
- **Scare:** retrained the "official" checkpoint on seed 42 (picked a priori to avoid cherry-picking) → test **0.2916**, below everything in the sweep. Lesson: the sweep's ±0.0087 was a small-sample illusion; true variance is bigger. Pooling seed 42 in, GNN ≈ 0.35 ± 0.03 — a tie with RF, not a win.
- **Resolution (`finalize_gat.py`):** committed to a fixed seed set {0–4} *before* looking at test, report the test **distribution** as the headline (not a single number), and deploy the checkpoint with the best **validation** score (seed 2, val 0.5127 → test 0.3625). Selection on val, never on test. Seed 42 was an exploratory probe, not part of the reported protocol, so it's not in the final table.

### What this means
- GNN reliably *matches and barely edges* a much simpler RandomForest. Not a blowout. Defensible, honest result.
- The gap (0.0075) is far smaller than seed-to-seed variance → statistically a tie.
- Root cause of the wide error bars: **11 fraud transactions in the test split.** Can't measure a small model difference on 11 positives.

### Future work (deliberately out of scope here)
- **Bigger fraud sample.** Full PaySim is 6.3M rows (~8k fraud); a larger slice would put hundreds of fraud cases in test and give a *trustworthy* verdict. Needs neighbour-sampling for the graph on 4GB VRAM. This is the single highest-impact next step.
- **A graph-native dataset** (e.g. Elliptic Bitcoin, or IEEE-CIS) where fraud actually hides in the connections — would play to the GNN's strengths. The whole pipeline is dataset-agnostic, so this is a v2, not a rewrite.
- **Probability calibration** (Platt/isotonic) to fix the threshold-0.95 overconfidence. Note: this does NOT affect PR-AUC (threshold-independent) — it's about usable probabilities, not the headline metric.

### Methodology principle established this milestone
Choose fixes on validation; look at test once. Don't re-roll seeds (or swap datasets) until you "win" — that's the same self-deception, just bigger. A measured tie is a real result.
