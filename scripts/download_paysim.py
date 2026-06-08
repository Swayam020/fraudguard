"""Download the PaySim dataset from Kaggle into data/raw/.

Usage:
    python scripts/download_paysim.py

Requires:
    - kaggle CLI configured with ~/.kaggle/kaggle.json (chmod 600)
    - Kaggle account with phone verification (Kaggle requirement for API downloads)

Behavior:
    1. Downloads ealaxi/paysim1 ZIP into data/raw/.
    2. Unzips PS_20174392719_1491204439457_log.csv (~470 MB) into data/raw/paysim.csv.
    3. Computes SHA-256 of the extracted CSV.
    4. If EXPECTED_SHA256 is set, fails loudly on mismatch.
    5. Removes the ZIP to save disk space.

Exit codes:
    0 = success
    1 = failure (auth, download, unzip, or hash mismatch)
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

# Repo root: this file lives at <repo>/scripts/download_paysim.py
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"

KAGGLE_SLUG = "ealaxi/paysim1"
ZIP_NAME = "paysim1.zip"
KAGGLE_CSV_NAME = "PS_20174392719_1491204439457_log.csv"
LOCAL_CSV_NAME = "paysim.csv"

# Pinned hash of the full upstream CSV. Empty string disables the check on
# first run; populate after first successful download (see README).
EXPECTED_SHA256 = "16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b"

CHUNK_SIZE = 1024 * 1024  # 1 MiB, for streaming hash computation


def log(msg: str) -> None:
    print(f"[download_paysim] {msg}", flush=True)


def sha256_of_file(path: Path) -> str:
    """Compute SHA-256 by streaming the file in chunks. Avoids loading 470 MB into RAM."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest()


def run_kaggle_download(dest_dir: Path) -> None:
    """Invoke the kaggle CLI to download the dataset ZIP into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", KAGGLE_SLUG, "-p", str(dest_dir)]
    log(f"running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        log(
            "kaggle CLI failed. Common causes: bad ~/.kaggle/kaggle.json, "
            "missing chmod 600, or Kaggle phone-verification required."
        )
        sys.exit(1)


def unzip_csv(zip_path: Path, dest_dir: Path) -> Path:
    """Extract the PaySim CSV from the ZIP and rename to a stable filename."""
    log(f"extracting {KAGGLE_CSV_NAME} from {zip_path.name}")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if KAGGLE_CSV_NAME not in names:
            log(f"expected {KAGGLE_CSV_NAME} inside zip; found {names}")
            sys.exit(1)
        zf.extract(KAGGLE_CSV_NAME, path=dest_dir)
    extracted = dest_dir / KAGGLE_CSV_NAME
    target = dest_dir / LOCAL_CSV_NAME
    if target.exists():
        target.unlink()
    extracted.rename(target)
    return target


def main() -> int:
    log(f"target directory: {RAW_DIR}")

    target_csv = RAW_DIR / LOCAL_CSV_NAME
    if target_csv.exists():
        log(
            f"{target_csv.name} already exists; skipping download. "
            "Delete it manually to force a refresh."
        )
    else:
        run_kaggle_download(RAW_DIR)
        zip_path = RAW_DIR / ZIP_NAME
        if not zip_path.exists():
            log(f"expected {zip_path} after download; not found")
            return 1
        unzip_csv(zip_path, RAW_DIR)
        zip_path.unlink()
        log(f"removed {ZIP_NAME}")

    log("computing SHA-256 (this takes ~5 seconds on a 470 MB file)...")
    digest = sha256_of_file(target_csv)
    log(f"sha256 = {digest}")

    if EXPECTED_SHA256:
        if digest != EXPECTED_SHA256:
            log(f"HASH MISMATCH. expected {EXPECTED_SHA256}, got {digest}")
            log("upstream dataset may have changed. investigate before proceeding.")
            return 1
        log("hash matches EXPECTED_SHA256 — integrity check passed.")
    else:
        log(
            "EXPECTED_SHA256 is empty — first run. "
            "Paste the digest above into EXPECTED_SHA256 and commit."
        )

    size_mb = target_csv.stat().st_size / (1024 * 1024)
    log(f"done. {target_csv} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
