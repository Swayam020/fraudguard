"""Unit tests for the data pipeline: split reproducibility and feature ranges."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = REPO_ROOT / "data" / "processed"

FEATURES_PARQUET = PROCESSED / "paysim_200k_features.parquet"
TRAIN_PARQUET = PROCESSED / "train.parquet"
VAL_PARQUET = PROCESSED / "val.parquet"
TEST_PARQUET = PROCESSED / "test.parquet"

pytestmark = pytest.mark.skipif(
    not FEATURES_PARQUET.exists(),
    reason="pipeline outputs missing; run subsample/features/split first",
)


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    return pd.read_parquet(FEATURES_PARQUET)


@pytest.fixture(scope="module")
def splits() -> dict[str, pd.DataFrame]:
    return {
        "train": pd.read_parquet(TRAIN_PARQUET),
        "val": pd.read_parquet(VAL_PARQUET),
        "test": pd.read_parquet(TEST_PARQUET),
    }


class TestFeatureRanges:
    def test_row_count(self, features: pd.DataFrame) -> None:
        assert len(features) == 200_000

    def test_amount_log_nonnegative(self, features: pd.DataFrame) -> None:
        assert (features["amount_log"] >= 0).all()

    def test_amount_z_standardized(self, features: pd.DataFrame) -> None:
        assert abs(features["amount_z"].mean()) < 1e-6
        assert abs(features["amount_z"].std() - 1.0) < 1e-3

    @pytest.mark.parametrize("col", ["is_transfer", "is_cashout", "balance_mismatch", "isFraud"])
    def test_binary_columns(self, features: pd.DataFrame, col: str) -> None:
        assert set(features[col].unique()).issubset({0, 1})

    def test_step_hour_range(self, features: pd.DataFrame) -> None:
        assert features["step_hour"].between(0, 23).all()

    def test_no_missing_values(self, features: pd.DataFrame) -> None:
        assert features.isna().sum().sum() == 0


class TestSplits:
    def test_sizes_sum_to_total(self, splits: dict[str, pd.DataFrame]) -> None:
        total = sum(len(df) for df in splits.values())
        assert total == 200_000

    def test_ratios(self, splits: dict[str, pd.DataFrame]) -> None:
        assert len(splits["train"]) == 140_000
        assert len(splits["val"]) == 30_000
        assert len(splits["test"]) == 30_000

    def test_temporal_ordering(self, splits: dict[str, pd.DataFrame]) -> None:
        assert splits["train"]["step"].max() <= splits["val"]["step"].min()
        assert splits["val"]["step"].max() <= splits["test"]["step"].min()

    def test_each_split_has_fraud(self, splits: dict[str, pd.DataFrame]) -> None:
        for name, df in splits.items():
            assert df["isFraud"].sum() > 0, f"{name} has zero fraud rows"

    def test_columns_consistent(self, splits: dict[str, pd.DataFrame]) -> None:
        cols = [tuple(df.columns) for df in splits.values()]
        assert cols[0] == cols[1] == cols[2]
