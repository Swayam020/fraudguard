"""Project-wide configuration as a typed, immutable dataclass.

All tunable settings (paths, hyperparameters, API options) live here so
the rest of the codebase imports `Config()` instead of hardcoding values.
"""

from dataclasses import dataclass, field
from pathlib import Path


# Project root = three levels up from this file:
#   config.py -> utils -> fraudguard -> src -> <ROOT>
# Path(__file__) is the path to this file; .resolve() makes it absolute;
# .parents[3] walks up 3 directories.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Config:
    """Immutable container for all FraudGuard settings.

    `frozen=True` means once you create a Config, you cannot reassign
    its fields. Prevents accidental mutation across the codebase.
    """

    # --- Paths ---
    # `field(default_factory=...)` is needed for mutable defaults like Path.
    # Without it, every Config() would share the same Path object (bug).
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    data_raw_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "raw")
    data_processed_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "processed"
    )

    # --- Dataset ---
    dataset_name: str = "paysim"
    sample_size: int = 200_000  # rows to sample from PaySim (matches paper)
    random_seed: int = 42  # fixed seed = reproducible results

    # --- Model hyperparameters (placeholders, tuned in Phase 6) ---
    embedding_dim: int = 64
    num_attention_heads: int = 4
    learning_rate: float = 1e-3
    batch_size: int = 256
    num_epochs: int = 20

    # --- API ---
    api_host: str = "127.0.0.1"
    api_port: int = 8000


# Module-level singleton: most code can just do `from ...config import config`
# instead of constructing a new Config() each time.
config = Config()
