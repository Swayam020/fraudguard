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
