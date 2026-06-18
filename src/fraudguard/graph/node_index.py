"""Node indexing for the heterogeneous transaction graph.

Assigns contiguous integer IDs to unique account names (nameOrig + nameDest)
and unique transaction rows. These IDs are the foundation for PyG HeteroData
edge_index tensors built in Milestone 4.2.

Reads:
    data/processed/train.parquet, val.parquet, test.parquet
Writes:
    data/processed/graph/account_id_map.parquet  (account_name -> account_id)
    data/processed/graph/txn_id_map.parquet       (split, row_in_split -> txn_id)

Usage:
    python -m fraudguard.graph.node_index
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
GRAPH_DIR = PROCESSED_DIR / "graph"

SPLITS = ["train", "val", "test"]


def log(msg: str) -> None:
    print(f"[node_index] {msg}", flush=True)


def build_account_index(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Assign a contiguous integer ID to every unique account name
    seen across nameOrig and nameDest, across all splits.
    """
    all_names = pd.concat(
        [df["nameOrig"] for df in dfs.values()] + [df["nameDest"] for df in dfs.values()],
        ignore_index=True,
    )
    unique_names = pd.Index(all_names.unique()).sort_values()
    node_type = ["user" if name.startswith("C") else "merchant" for name in unique_names]
    account_map = pd.DataFrame(
        {
            "account_name": unique_names,
            "account_id": range(len(unique_names)),
            "node_type": node_type,
        }
    )
    return account_map


def build_txn_index(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Assign a contiguous integer ID to every transaction row, tagged
    with which split it came from and its row position within that split.
    """
    frames = []
    next_id = 0
    for split in SPLITS:
        n = len(dfs[split])
        frames.append(
            pd.DataFrame(
                {
                    "txn_id": range(next_id, next_id + n),
                    "split": split,
                    "row_in_split": range(n),
                }
            )
        )
        next_id += n
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    dfs = {}
    for split in SPLITS:
        path = PROCESSED_DIR / f"{split}.parquet"
        if not path.exists():
            log(f"missing {path}. Run subsample/features/clean/split first.")
            return 1
        dfs[split] = pd.read_parquet(path, columns=["nameOrig", "nameDest"])

    log("loaded splits: " + ", ".join(f"{s}={len(dfs[s]):,}" for s in SPLITS))

    account_map = build_account_index(dfs)
    log(f"unique accounts: {len(account_map):,}")

    txn_map = build_txn_index(dfs)
    log(f"total transactions: {len(txn_map):,}")

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    account_out = GRAPH_DIR / "account_id_map.parquet"
    txn_out = GRAPH_DIR / "txn_id_map.parquet"

    account_map.to_parquet(account_out, engine="pyarrow", compression="snappy", index=False)
    txn_map.to_parquet(txn_out, engine="pyarrow", compression="snappy", index=False)

    log(f"wrote {account_out.name} ({account_out.stat().st_size / 1024:.1f} KB)")
    log(f"wrote {txn_out.name} ({txn_out.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
