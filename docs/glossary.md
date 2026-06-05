# FraudGuard Glossary

> Running list of every technical term introduced during the project.
> Used as a viva study guide.

## Git & Version Control

- **Repository (repo):** A folder being tracked by git.
- **`.git` directory:** Hidden folder where git stores history and bookkeeping.
- **Commit:** A saved snapshot of files at a point in time.
- **Branch:** A parallel timeline of commits.
- **`main`:** The default primary branch (industry standard name).
- **SHA hash:** The unique ID of a commit (e.g., `c9051f8`).
- **Conventional commits:** Commit message format `type: description` (e.g., `chore:`, `feat:`, `fix:`, `docs:`, `refactor:`, `test:`).
- **Staging area (index):** Holding zone for changes you want in the next commit.
- **Working directory:** Files on disk you're editing.
- **HEAD:** Git's pointer to the current commit.
- **`.gitignore`:** Tells git which files to never track.
- **Gitignore negation (`!`):** Un-ignore a specific file inside an ignored folder.
- **`.gitkeep`:** Convention — empty placeholder file so git tracks an otherwise-empty folder.

## Python Packaging

- **Virtual environment (venv):** Project-local isolated Python installation.
- **`requirements.txt`:** Plain-text list of packages a project needs.
- **`pip install -r`:** Install all packages listed in a file.
- **Python package:** A folder containing `__init__.py`, making it importable.
- **`__init__.py`:** Marker file telling Python "this folder is a package."
- **`src/` layout:** Industry-standard project structure where source code lives in a `src/` directory.
- **`sys.path`:** List of directories Python searches when importing packages.
- **Editable install (`pip install -e .`):** Install your own project as a package, with live source links.

## Documentation & Open Source

- **README:** Front-page document of a repo, rendered by GitHub.
- **Markdown (`.md`):** Lightweight plain-text formatting language that renders to HTML.
- **Badges:** Visual shields (shields.io) showing tech versions, license, build status.
- **MIT License:** Permissive open-source license — anyone can use the code, must keep your copyright notice.
- **ADR (Architecture Decision Record):** Short document capturing one technical decision and its rationale.

## Project-Specific Terms

_(To be populated as we build.)_
