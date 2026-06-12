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
