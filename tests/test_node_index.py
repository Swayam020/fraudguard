"""Unit tests for node indexing: account/transaction ID uniqueness and coverage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_DIR = REPO_ROOT / "data" / "processed" / "graph"
ACCOUNT_MAP_PARQUET = GRAPH_DIR / "account_id_map.parquet"
TXN_MAP_PARQUET = GRAPH_DIR / "txn_id_map.parquet"

PROCESSED = REPO_ROOT / "data" / "processed"
TRAIN_PARQUET = PROCESSED / "train.parquet"
VAL_PARQUET = PROCESSED / "val.parquet"
TEST_PARQUET = PROCESSED / "test.parquet"

pytestmark = pytest.mark.skipif(
    not ACCOUNT_MAP_PARQUET.exists(),
    reason="node index outputs missing; run python -m fraudguard.graph.node_index first",
)


@pytest.fixture(scope="module")
def account_map() -> pd.DataFrame:
    return pd.read_parquet(ACCOUNT_MAP_PARQUET)


@pytest.fixture(scope="module")
def txn_map() -> pd.DataFrame:
    return pd.read_parquet(TXN_MAP_PARQUET)


@pytest.fixture(scope="module")
def splits() -> dict[str, pd.DataFrame]:
    return {
        "train": pd.read_parquet(TRAIN_PARQUET, columns=["nameOrig", "nameDest"]),
        "val": pd.read_parquet(VAL_PARQUET, columns=["nameOrig", "nameDest"]),
        "test": pd.read_parquet(TEST_PARQUET, columns=["nameOrig", "nameDest"]),
    }


class TestAccountIndex:
    def test_account_ids_unique(self, account_map: pd.DataFrame) -> None:
        assert account_map["account_id"].is_unique

    def test_account_ids_contiguous(self, account_map: pd.DataFrame) -> None:
        ids = account_map["account_id"].sort_values().to_numpy()
        assert ids[0] == 0
        assert ids[-1] == len(ids) - 1

    def test_account_names_unique(self, account_map: pd.DataFrame) -> None:
        assert not account_map["account_name"].duplicated().any()

    def test_covers_all_split_accounts(
        self, account_map: pd.DataFrame, splits: dict[str, pd.DataFrame]
    ) -> None:
        known = set(account_map["account_name"])
        for name, df in splits.items():
            missing_orig = set(df["nameOrig"]) - known
            missing_dest = set(df["nameDest"]) - known
            assert not missing_orig, f"{name}: {len(missing_orig)} nameOrig not indexed"
            assert not missing_dest, f"{name}: {len(missing_dest)} nameDest not indexed"


class TestTxnIndex:
    def test_txn_ids_unique(self, txn_map: pd.DataFrame) -> None:
        assert txn_map["txn_id"].is_unique

    def test_txn_ids_contiguous(self, txn_map: pd.DataFrame) -> None:
        ids = txn_map["txn_id"].sort_values().to_numpy()
        assert ids[0] == 0
        assert ids[-1] == len(ids) - 1

    def test_row_count_matches_splits(
        self, txn_map: pd.DataFrame, splits: dict[str, pd.DataFrame]
    ) -> None:
        expected = sum(len(df) for df in splits.values())
        assert len(txn_map) == expected

    def test_row_in_split_matches_split_length(
        self, txn_map: pd.DataFrame, splits: dict[str, pd.DataFrame]
    ) -> None:
        for name, df in splits.items():
            split_rows = txn_map.loc[txn_map["split"] == name, "row_in_split"]
            assert split_rows.max() == len(df) - 1
            assert split_rows.min() == 0
